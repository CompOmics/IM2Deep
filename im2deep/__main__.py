"""
Command line interface for IM2Deep.

This module provides a comprehensive command-line interface for the IM2Deep
CCS prediction package. It handles input file parsing, model configuration,
calibration setup, and output generation.

The CLI supports:
- Multiple input file formats (CSV with seq/modifications or PSM formats)
- Optional calibration using reference datasets
- Single-conformer and multi-conformer predictions
- Ion mobility output conversion
- Ensemble or single model prediction
- Comprehensive logging and error reporting

Usage:
    Basic prediction:
        im2deep input_peptides.csv

    With calibration (recommended):
        im2deep input_peptides.csv -c calibration_data.csv

    Multi-conformer prediction:
        im2deep input_peptides.csv -c calibration_data.csv -e

    Ion mobility output:
        im2deep input_peptides.csv -c calibration_data.csv -i

Dependencies:
    - click: Command-line interface framework
    - psm_utils: Peptide and PSM data handling
    - rich: Enhanced logging and progress display
    - pandas: Data manipulation

Authors:
    - Robbe Devreese
    - Robbin Bouwmeester
    - Ralf Gabriels
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from psm_utils.io import read_file
from psm_utils.io.exceptions import PSMUtilsIOException
from psm_utils.io.peptide_record import peprec_to_proforma
from psm_utils.psm import PSM
from psm_utils.psm_list import PSMList
from rich.logging import RichHandler

REFERENCE_DATASET_PATH = Path(__file__).parent / "reference_data" / "reference_ccs.zip"

LOGGER = logging.getLogger(__name__)


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
            f"Invalid log level: {passed_level}. " f"Should be one of {list(log_mapping.keys())}"
        )

    logging.basicConfig(
        level=log_mapping[passed_level.lower()],
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )


def check_optional_dependencies() -> None:
    """
    Check if optional dependencies for multi-conformer prediction are available.

    Raises
    ------
    SystemExit
        If required dependencies are missing
    """
    try:
        import torch
        import im2deeptrainer

        LOGGER.debug("Optional dependencies for multi-conformer prediction found")
    except ImportError:
        LOGGER.error(
            "Multi-conformer prediction requires optional dependencies.\n"
            "Please install IM2Deep with optional dependencies:\n"
            "pip install 'im2deep[er]'"
        )
        sys.exit(1)


def _validate_file_format(file_path: str, file_type: str = "input") -> bool:
    """
    Validate file format and accessibility.

    Parameters
    ----------
    file_path : str
        Path to file to validate
    file_type : str
        Type of file for error messages

    Returns
    -------
    bool
        True if file is valid

    Raises
    ------
    click.ClickException
        If file validation fails
    """
    path = Path(file_path)

    if not path.exists():
        raise click.ClickException(f"{file_type.capitalize()} file not found: {file_path}")

    if not path.is_file():
        raise click.ClickException(f"{file_type.capitalize()} path is not a file: {file_path}")

    if path.suffix.lower() not in [".csv", ".txt", ".tsv"]:
        LOGGER.warning(f"Unexpected file extension for {file_type} file: {path.suffix}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line:
                raise click.ClickException(f"{file_type.capitalize()} file appears to be empty")
    except Exception as e:
        raise click.ClickException(f"Error reading {file_type} file: {e}")

    return True


def _parse_csv_input(file_path: str, file_type: str = "prediction") -> PSMList:
    """
    Parse CSV input file into PSMList.

    Parameters
    ----------
    file_path : str
        Path to CSV file
    file_type : str
        Type of file for error messages

    Returns
    -------
    PSMList
        Parsed PSM data

    Raises
    ------
    click.ClickException
        If parsing fails
    """
    try:
        df = pd.read_csv(file_path)
        df = df.fillna("")

        required_cols = ["seq", "modifications", "charge"]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise click.ClickException(
                f"Missing required columns in {file_type} file: {missing_cols}\n"
                f"Required columns: {required_cols}"
            )

        if file_type == "calibration" and "CCS" not in df.columns:
            raise click.ClickException("Calibration file must contain 'CCS' column")

        list_of_psms = []
        for idx, row in df.iterrows():
            try:
                peptidoform = peprec_to_proforma(row["seq"], row["modifications"], row["charge"])
                metadata = {}
                if file_type == "calibration" and "CCS" in row:
                    metadata["CCS"] = float(row["CCS"])

                psm = PSM(peptidoform=peptidoform, metadata=metadata, spectrum_id=idx)
                list_of_psms.append(psm)
            except Exception as e:
                LOGGER.warning(f"Skipping row {idx} due to parsing error: {e}")
                continue

        if not list_of_psms:
            raise click.ClickException(f"No valid peptides found in {file_type} file")

        LOGGER.info(f"Parsed {len(list_of_psms)} peptides from {file_type} file")
        return PSMList(psm_list=list_of_psms)

    except pd.errors.EmptyDataError:
        raise click.ClickException(f"{file_type.capitalize()} file is empty")
    except pd.errors.ParserError as e:
        raise click.ClickException(f"Error parsing {file_type} file: {e}")
    except Exception as e:
        raise click.ClickException(f"Unexpected error reading {file_type} file: {e}")


# Command line interface with comprehensive options
@click.command()
@click.argument("psm-file", type=click.Path(exists=True, dir_okay=False), metavar="INPUT_FILE")
@click.option(
    "-c",
    "--calibration-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to calibration file with known CCS values. Highly recommended for accurate predictions.",
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(dir_okay=False),
    default=None,
    help="Output file path. If not specified, creates file next to input with '_IM2Deep-predictions.csv' suffix.",
)
@click.option(
    "-m",
    "--model-name",
    type=click.Choice(["tims"], case_sensitive=False),
    default="tims",
    help="Neural network model to use for prediction.",
)
@click.option(
    "-e",
    "--multi",
    is_flag=True,
    default=False,
    help="Enable multi-conformer prediction. Requires optional dependencies: pip install 'im2deep[er]'",
)
@click.option(
    "-l",
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
    default="info",
    help="Set logging verbosity level.",
)
@click.option(
    "-n",
    "--n-jobs",
    type=click.IntRange(min=1),
    default=None,
    help="Number of parallel jobs for model inference. Default uses all available CPU cores.",
)
@click.option(
    "--calibrate-per-charge",
    type=click.BOOL,
    default=True,
    help="Apply calibration per charge state for improved accuracy. Disable for global calibration.",
)
@click.option(
    "--use-charge-state",
    type=click.IntRange(min=1, max=6),
    default=2,
    help="Charge state for global calibration when --calibrate-per-charge is disabled.",
)
@click.option(
    "--use-single-model",
    type=click.BOOL,
    default=True,
    help="Use single model (faster) vs ensemble of models (potentially slightly more accurate).",
)
@click.option(
    "-i",
    "--ion-mobility",
    is_flag=True,
    default=False,
    help="Output ion mobility (1/K0) instead of CCS values.",
)
def main(
    psm_file: str,
    calibration_file: Optional[str] = None,
    output_file: Optional[str] = None,
    model_name: str = "tims",
    multi: bool = False,
    log_level: str = "info",
    n_jobs: Optional[int] = None,
    use_single_model: bool = True,
    calibrate_per_charge: bool = True,
    use_charge_state: int = 2,
    ion_mobility: bool = False,
) -> None:
    """
    IM2Deep: Predict CCS values for peptides using deep learning.

    IM2Deep predicts Collisional Cross Section (CCS) values for peptides,
    including those with post-translational modifications. The tool supports
    both single-conformer and multi-conformer predictions with optional
    calibration using reference datasets.

    INPUT_FILE should be a CSV file with columns:
    \b
    - seq: Peptide sequence (required)
    - modifications: Modifications in format "position|name" (required, can be empty)
    - charge: Charge state (required)

    For calibration files, an additional 'CCS' column with observed values is required.

    Examples:
    \b
        # Basic prediction
        im2deep peptides.csv

        # With calibration (recommended)
        im2deep peptides.csv -c calibration.csv

        # Multi-conformer prediction
        im2deep peptides.csv -c calibration.csv -e

        # Ion mobility output
        im2deep peptides.csv -c calibration.csv -i

        # Ensemble prediction with custom output
        im2deep peptides.csv -c calibration.csv -o results.csv --use-single-model False
    """
    try:
        # Setup logging first
        setup_logging(log_level)

        LOGGER.info("IM2Deep command-line interface started")
        LOGGER.debug(
            f"Input arguments: psm_file={psm_file}, calibration_file={calibration_file}, "
            f"multi={multi}, ion_mobility={ion_mobility}"
        )

        # Import main functionality (after logging setup)
        from im2deep._exceptions import IM2DeepError
        from im2deep.im2deep import predict_ccs

        # Check optional dependencies if multi-conformer requested
        if multi:
            check_optional_dependencies()

        # Validate input files
        _validate_file_format(psm_file, "input")
        if calibration_file:
            _validate_file_format(calibration_file, "calibration")

        # Parse input files
        LOGGER.info("Parsing input files...")

        # Try to determine file format
        with open(psm_file, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()

        # Check if it's the expected CSV format
        if "modifications" in first_line and "seq" in first_line:
            psm_list_pred = _parse_csv_input(psm_file, "prediction")
            df_pred = pd.read_csv(psm_file).fillna("")
        else:
            # Try psm_utils for other formats
            try:
                psm_list_pred = read_file(psm_file)
                df_pred = None
                LOGGER.info(f"Loaded {len(psm_list_pred)} PSMs using psm_utils")
            except PSMUtilsIOException as e:
                raise click.ClickException(
                    f"Could not parse input file. Expected CSV with columns 'seq', 'modifications', 'charge' "
                    f"or a format supported by psm_utils. Error: {e}"
                )

        # Parse calibration file
        psm_list_cal = None
        df_cal = None
        if calibration_file:
            with open(calibration_file, "r", encoding="utf-8") as f:
                cal_first_line = f.readline().strip()

            if (
                "modifications" in cal_first_line
                and "seq" in cal_first_line
                and "CCS" in cal_first_line
            ):
                psm_list_cal = _parse_csv_input(calibration_file, "calibration")
                df_cal = pd.read_csv(calibration_file).fillna("")
            else:
                raise click.ClickException(
                    "Calibration file must be CSV with columns: 'seq', 'modifications', 'charge', 'CCS'"
                )
        else:
            LOGGER.warning(
                "No calibration file provided. Predictions will be uncalibrated. "
                "Calibration is HIGHLY recommended for accurate results."
            )

        # Set up output file
        if not output_file:
            input_path = Path(psm_file)
            output_file = input_path.parent / f"{input_path.stem}_IM2Deep-predictions.csv"

        LOGGER.info(f"Output will be written to: {output_file}")

        # Run prediction
        LOGGER.info("Starting CCS prediction...")
        predict_ccs(
            psm_list_pred,
            psm_list_cal,
            output_file=output_file,
            model_name=model_name,
            multi=multi,
            calibrate_per_charge=calibrate_per_charge,
            use_charge_state=use_charge_state,
            n_jobs=n_jobs,
            use_single_model=use_single_model,
            ion_mobility=ion_mobility,
            pred_df=df_pred,
            cal_df=df_cal,
            write_output=True,
        )

        LOGGER.info("IM2Deep completed successfully!")

    except IM2DeepError as e:
        LOGGER.error(f"IM2Deep error: {e}")
        sys.exit(1)
    except click.ClickException:
        # Re-raise click exceptions to preserve formatting
        raise
    except Exception as e:
        LOGGER.error(f"Unexpected error: {e}")
        if log_level.lower() == "debug":
            LOGGER.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
