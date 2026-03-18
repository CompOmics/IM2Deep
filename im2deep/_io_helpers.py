"""
Utility functions for IM2Deep package.

This module provides utility functions for converting between different
mobility measurements and configuration settings for multi-conformer models.

Functions:
    im2ccs: Convert ion mobility to collisional cross section
    ccs2im: Convert collisional cross section to ion mobility

Constants:
    multi_config: Configuration dictionary for multi-conformer model
    MULTI_BACKBONE_PATH: Path to the multi-conformer model backbone
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import psm_utils.io
from psm_utils.psm import PSM
from psm_utils.psm_list import PSMList
from rich.console import Console
from rich.logging import RichHandler

from im2deep.exceptions import IM2DeepError, PSMMetadataError
from im2deep.utils import ccs2im, im2ccs

console = Console()

LOGGER = logging.getLogger(__name__)

ConvertibleToFloat = str | float | int | np.floating[Any] | np.ndarray[Any, Any]


def _normalize_ccs_metadata_value(value: ConvertibleToFloat) -> str:
    """Validate and normalize a CCS metadata value to the canonical string form."""
    try:
        return str(float(value))
    except (TypeError, ValueError) as exc:
        raise PSMMetadataError(
            f"Invalid CCS metadata value {value!r}; expected a numeric string or number."
        ) from exc


def parse_input(
    input_file: str | Path | PSMList | pd.DataFrame, filetype: str | None = None
) -> PSMList:
    """
    Parse input file or PSMList into a PSMList object.

    Parameters
    ----------
    file_path : str, Path, or PSMList
        Path to the input file or a PSMList object.

    Returns
    -------
    PSMList
        Parsed PSMList object.
    """
    if isinstance(input_file, PSMList):
        LOGGER.debug(f"Parsed {len(input_file)} PSMs from provided PSMList.")
        return input_file

    if isinstance(input_file, pd.DataFrame):
        LOGGER.debug(f"Parsing PSMs from provided DataFrame with {len(input_file)} rows.")
        list_of_precursors = []

        # Check if it's legacy format (has seq/modifications/charge) or standard format (has peptidoform)
        has_peptidoform = "peptidoform" in input_file.columns
        has_legacy_cols = all(
            col in input_file.columns for col in ["seq", "modifications", "charge"]
        )

        for idx, row in input_file.iterrows():
            try:
                if has_peptidoform:
                    # Standard format with peptidoform column
                    precursor = PSM(peptidoform=row["peptidoform"], spectrum_id=idx)
                elif has_legacy_cols:
                    # Legacy format - convert to peptidoform
                    peptidoform = psm_utils.io.peptide_record.peprec_to_proforma(
                        peptide=row["seq"],
                        modifications=row["modifications"],
                        charge=int(row["charge"]),
                    )
                    precursor = PSM(peptidoform=peptidoform, spectrum_id=idx)
                else:
                    LOGGER.warning("Row %d missing required columns. Skipping.", idx)
                    continue

                if "CCS" in row:
                    if precursor.metadata is None:
                        precursor.metadata = {}
                    precursor.metadata["CCS"] = _normalize_ccs_metadata_value(row["CCS"])  # type: ignore
                list_of_precursors.append(precursor)
            except Exception as e:
                LOGGER.warning("Error parsing row %d: %s. Skipping.", idx, e)
                continue

        if not list_of_precursors:
            raise IM2DeepError("No valid PSMs could be parsed from the DataFrame.")

        psm_list = PSMList(psm_list=list_of_precursors)
        LOGGER.debug(f"Parsed {len(psm_list)} PSMs from DataFrame.")
        return psm_list

    if not isinstance(input_file, (str, Path, PSMList)):
        raise TypeError("input_file must be a str, Path, or PSMList.")

    LOGGER.info("Reading PSMs from file: %s", input_file)

    # First, check if it's a legacy format by inspecting the header
    is_legacy_format = False
    try:
        # Read first line to check column names
        with open(input_file) as f:
            first_line = f.readline().strip()

        # Check if it has legacy format columns
        if "seq" in first_line.lower() and "modifications" in first_line.lower():
            # Additional check: legacy format should NOT have standard PSM format columns
            if not any(
                col in first_line.lower() for col in ["peptidoform", "protein", "spectrum_id"]
            ):
                is_legacy_format = True
                LOGGER.debug("Detected legacy internal format based on header.")
    except Exception as e:
        LOGGER.debug(f"Could not pre-check file format: {e}")

    # Parse based on detected format
    if is_legacy_format:
        psm_list = _parse_legacy_format(input_file)
    else:
        # Try to parse with psm_utils
        try:
            psm_list = psm_utils.io.read_file(input_file, filetype=filetype or "infer")
            LOGGER.debug("Successfully read file using psm_utils.")
        except Exception as e:
            # If psm_utils fails, try legacy format as fallback
            LOGGER.warning(f"Failed to read PSM file using psm_utils: {e}")
            LOGGER.info("Attempting to read as legacy internal format.")
            psm_list = _parse_legacy_format(input_file)

    LOGGER.debug(f"Parsed {len(psm_list)} PSMs from file.")
    return psm_list


def _parse_legacy_format(input_file: str | Path) -> PSMList:
    """
    Parse legacy internal format delimited file.

    Expected columns: seq, modifications, charge, and optionally CCS.
    Supports CSV, TSV, and other delimited formats.

    Parameters
    ----------
    input_file : str or Path
        Path to the legacy format file.

    Returns
    -------
    PSMList
        Parsed PSMList object.

    Raises
    ------
    IM2DeepError
        If required columns are missing or parsing fails.
    """
    try:
        # Use sep=None with engine='python' to auto-detect delimiter
        df = pd.read_csv(input_file, sep=None, engine="python")
        df = df.fillna("")  # Replace NaN with empty strings
    except Exception as e:
        raise IM2DeepError(f"Failed to read file as delimited text: {e}") from e

    required_cols_legacy = ["seq", "modifications", "charge"]
    missing_cols = set(required_cols_legacy) - set(df.columns)

    # Handle peprec format (uses 'peptide' instead of 'seq')
    if "seq" not in df.columns and "peptide" in df.columns:
        df.rename(columns={"peptide": "seq"}, inplace=True)
        missing_cols = set(required_cols_legacy) - set(df.columns)

    if missing_cols:
        raise IM2DeepError(
            f"Legacy format file is missing required columns: {missing_cols}. "
            f"Expected columns: seq (or peptide), modifications, charge"
        )

    has_ccs = "CCS" in df.columns

    list_of_precursors = []
    for idx, row in df.iterrows():
        metadata = {}
        try:
            peptidoform = psm_utils.io.peptide_record.peprec_to_proforma(
                peptide=row["seq"],
                modifications=row["modifications"],
                charge=int(row["charge"]),
            )
            if has_ccs:
                metadata = {"CCS": _normalize_ccs_metadata_value(row["CCS"])}

            LOGGER.debug(f"Parsed PSM: {peptidoform} with metadata: {metadata}")
            precursor = PSM(peptidoform=peptidoform, metadata=metadata, spectrum_id=idx)
            list_of_precursors.append(precursor)
        except Exception as e:
            LOGGER.warning("Error parsing row %d: %s. Skipping.", idx, e)
            continue

    if not list_of_precursors:
        raise IM2DeepError("No valid PSMs could be parsed from the legacy format file.")

    psm_list = PSMList(psm_list=list_of_precursors)
    LOGGER.info(f"Successfully read {len(psm_list)} PSMs as legacy internal format.")
    return psm_list


def validate_psm_list(psm_list: PSMList, needs_target: bool = False) -> PSMList:
    """
    Validate that the PSM list contains necessary fields. And homogenizes the data.
    Also removes entries with charge state higher than 6.

    Parameters
    ----------
    psm_list : PSMList
        The PSM list to validate.
    needs_target : bool, optional
        Whether target IM or CCS values are required. Default is False.

    Returns
    -------
    PSMList
        The validated and filtered PSM list.
    """
    # Check if it's a PSMList
    if not isinstance(psm_list, PSMList):
        raise IM2DeepError(
            f"Expected PSMList, got {type(psm_list).__name__}. "
            "Please provide a valid PSMList object."
        )

    # Filter missing and high charge states (IM2Deep predictions are not reliable for charges >6)
    original_size = len(psm_list)
    charges = np.array([psm.peptidoform.precursor_charge for psm in psm_list])
    psm_list_filtered = psm_list[(charges != None) & (charges <= 6)]  # noqa: E711

    # TODO: Is deepcopy really necessary or can it be avoided?
    psm_list_filtered = deepcopy(psm_list_filtered)

    if len(psm_list_filtered) < original_size:
        filtered_count = original_size - len(psm_list_filtered)
        LOGGER.warning(
            f"Filtered out {filtered_count} PSMs with charge states missing or >6 for shift"
            "calculation.Predictions are not reliable for charge states >6."
        )

    if len(psm_list_filtered) == 0:
        raise IM2DeepError("No PSMs present in provided PSMLists.")

    all_has_targets = True
    if needs_target:
        # Check if PSMs have either ion_mobility or CCS
        all_has_targets = all(
            psm.ion_mobility is not None or psm.metadata.get("CCS") is not None
            for psm in psm_list_filtered
        )

        # TODO: Could be vectorized over all ion mobility values
        # Normalize targets while preserving the PSM metadata str -> str contract.
        for psm in psm_list_filtered:
            if psm.metadata is None:
                psm.metadata = {}

            if psm.ion_mobility is not None:
                if "CCS" not in psm.metadata:
                    psm.metadata["CCS"] = _normalize_ccs_metadata_value(
                        im2ccs(
                            psm.ion_mobility,
                            psm.peptidoform.theoretical_mz,
                            psm.peptidoform.precursor_charge,
                        )
                    )
            elif psm.metadata.get("CCS") is not None:
                psm.metadata["CCS"] = _normalize_ccs_metadata_value(psm.metadata["CCS"])

    if needs_target and not all_has_targets:
        raise IM2DeepError("PSMList must contain 'ion_mobility' or 'CCS' metadata for all PSMs.")

    return psm_list_filtered


class DefaultCommandGroup(click.Group):
    """Custom Click Group that invokes a default command if no subcommand is specified."""

    def __init__(self, *args, **kwargs):
        self.default_command = kwargs.pop("default_command", None)
        super().__init__(*args, **kwargs)

    def resolve_command(self, ctx, args):
        try:
            # Try to resolve the command normally
            return super().resolve_command(ctx, args)
        except click.UsageError:
            # If it fails and we have a default command, use that
            if self.default_command and args:
                # Get the default command
                cmd_name = self.default_command
                cmd = self.commands.get(cmd_name)
                if cmd:
                    return cmd_name, cmd, args
            # Re-raise the error if no default or command not found
            raise


def setup_logging(passed_level: str) -> None:
    """
    Configure logging with Rich formatting.

    Parameters
    ----------
    passed_level : str
        Logging level name (debug, info, warning, error, critical)

    Raises
    ------
    ValueError
        If invalid logging level provided
    """
    log_mapping = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    if passed_level.lower() not in log_mapping:
        raise ValueError(
            f"Invalid log level: {passed_level}. Should be one of {list(log_mapping.keys())}"
        )

    # Get the root logger and set its level
    root_logger = logging.getLogger()
    root_logger.setLevel(log_mapping[passed_level.lower()])

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add Rich handler
    rich_handler = RichHandler(
        rich_tracebacks=True, console=console, show_level=True, show_path=True
    )
    rich_handler.setLevel(log_mapping[passed_level.lower()])
    root_logger.addHandler(rich_handler)

    # Also set the level for all existing loggers (including im2deep modules)
    for logger_name in logging.Logger.manager.loggerDict:
        if logger_name.startswith("im2deep"):
            logger = logging.getLogger(logger_name)
            logger.setLevel(log_mapping[passed_level.lower()])


def infer_output_name(
    input_filename: str,
    output_name: str | None = None,
) -> Path:
    """Infer output filename from input filename if output_filename was not defined."""
    if output_name:
        return Path(output_name)
    else:
        input__filename = Path(input_filename)
        return input__filename.with_name(
            input__filename.stem + "_IM2Deep-predictions"
        ).with_suffix("")


def write_output(
    output_name: Path, predictions: np.ndarray, psm_list: PSMList, ion_mobility: bool = False
) -> None:
    """
    Write the predictions to a CSV file.

    Parameters
    ----------
    output_name : Path
        The output file path.
    predictions : np.ndarray
        The predicted CCS values.
    psm_list : PSMList
        The original PSMList.
    ion_mobility : bool, optional
        Whether to include ion mobility in the output. Default is False.
    """
    output_data = []
    for idx, psm in enumerate(psm_list):
        entry = {
            "index": psm.spectrum_id,
            "peptidoform": str(psm.peptidoform),
            "predicted_CCS": predictions[idx],
        }
        if ion_mobility:
            im_value = ccs2im(
                predictions[idx],
                psm.peptidoform.theoretical_mz,  # type: ignore -  already checked charge present
                psm.peptidoform.precursor_charge,  # type: ignore -  already checked charge present
            )
            entry["predicted_ion_mobility"] = im_value
        output_data.append(entry)

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_name, index=False)
    LOGGER.info(f"Predictions written to {output_name}")
