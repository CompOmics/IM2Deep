"""
# TODO: update docstring
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
from rich.console import Console

from psm_utils.io import read_file
from psm_utils.io.exceptions import PSMUtilsIOException
from psm_utils.io.peptide_record import peprec_to_proforma
from psm_utils.psm import PSM
from psm_utils.psm_list import PSMList
from rich.logging import RichHandler

from im2deep import __version__, core
from im2deep.utils import parse_input, build_credits, check_optional_dependencies

console = Console()
LOGGER = logging.getLogger(__name__)


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


# TODO: move to utils
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


# Command line interface
@click.group(cls=DefaultCommandGroup, default_command="predict", invoke_without_command=True)
@click.pass_context
@click.option(
    "--logging-level",
    "-l",
    type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
    default="info",
    help="Set logging verbosity level.",
)
@click.version_option(version=__version__)
def cli(ctx, logging_level):
    """IM2Deep: Predict CCS values for peptides using deep learning.

    Run prediction with: im2deep INPUT_FILE [OPTIONS]

    With calibration: im2deep INPUT_FILE -c CALIBRATION_FILE

    Use subcommands for additional functionality:
        im2deep train ...
    """
    setup_logging(logging_level)

    # Store parameters in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["logging_level"] = logging_level

    console.print(build_credits())


# Implement psm_utils reading for calibration and prediction PSMLists
@cli.command()
@click.argument(
    "precursors", type=click.Path(exists=True, dir_okay=False), metavar="INPUT_FILE", required=True
)
@click.option(
    "-c",
    "--calibration-precursors",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to file with precursors with known CCS values. If provided, calibration is performed.",
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
    help="Neural network model to use for prediction. Currently only 'tims' is supported.",
)
@click.option(
    "-e",
    "--multi",
    is_flag=True,
    default=False,
    help="Enable multi-conformer prediction. Requires optional dependencies: pip install 'im2deep[er]'",
)
@click.option(
    "-n",
    "--processes",
    type=int,
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
@click.option(
    "-l",
    "--logging-level",
    type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
    default="info",
    help="Set logging verbosity level.",
)
def predict(*args, **kwargs):
    """Predict CCS values for peptides (default command).

    If no calibration file is provided with -c, performs prediction only.
    With -c, performs calibration and prediction for improved accuracy.
    """
    # Setup logging first
    setup_logging(kwargs.get("logging_level", "info"))

    LOGGER.info("Starting IM2Deep CCS prediction...")
    LOGGER.debug(
        f"Input arguments: precursors={kwargs.get('precursors')}, "
        f"calibration_precursors={kwargs.get('calibration_precursors')}, multi={kwargs.get('multi')}, "
        f"ion_mobility={kwargs.get('ion_mobility')}"
    )

    # Check optional dependencies if multi-conformer requested
    if kwargs.get("multi"):
        check_optional_dependencies()

    # Parse input files
    LOGGER.info("Parsing input files...")
    psm_list = parse_input(kwargs.get("precursors"))

    # Run prediction
    LOGGER.info("Running CCS prediction...")
    if kwargs.get("calibration_precursors"):
        LOGGER.info("Calibration file provided, performing calibration and prediction...")
        psm_list_cal = parse_input(kwargs.get("calibration_precursors"))
        predictions = core.predict_and_calibrate(psm_list, psm_list_cal, *args, **kwargs)
    else:
        LOGGER.info(
            "No calibration file provided (calibration is HIGHLY recommended), performing prediction only..."
        )
        predictions = core.predict(*args, **kwargs)

    # Output results
    LOGGER.info("IM2Deep CCS prediction completed successfully!")
    output_name = kwargs.pop("output_file")
    output_name = _infer_output_name(kwargs["precursors"], output_name).with_suffix(".csv")
    LOGGER.info(f"Writing output file to {output_name}...")
    write_output(output_name, predictions, psm_list, kwargs.get("ion_mobility", False))


# TODO: implement train command
# @cli.command()
# @click.argument("training_data", type=click.Path(exists=True, dir_okay=False))
# @click.option(
#     "-o",
#     "--output-model",
#     type=click.Path(dir_okay=False),
#     required=True,
#     help="Path to save the trained model.",
# )
# @click.option(
#     "--epochs",
#     type=int,
#     default=100,
#     help="Number of training epochs.",
# )
# @click.option(
#     "-l",
#     "--logging-level",
#     type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
#     default="info",
#     help="Set logging verbosity level.",
# )
# def train(training_data, output_model, epochs, logging_level):
#     """Train a new IM2Deep model.

#     Example: im2deep train training_data.csv -o my_model.ckpt
#     """
#     setup_logging(logging_level)
#     LOGGER.info("Starting IM2Deep training...")

#     # Parse training data
#     psm_list_train = _parse_csv_input(training_data, "training")

#     # Call training function
#     core.train(
#         psm_list=psm_list_train,
#         model_save_path=output_model,
#         training_kwargs={"epochs": epochs},
#     )

#     LOGGER.info(f"Training completed. Model saved to {output_model}")


def main():
    # try:
    cli(obj={})
    # except Exception as e:
    #     LOGGER.error(f"Unexpected error in IM2Deep CLI: {e}")
    #     sys.exit(1)


# def main(
#     psm_file: str,
#     calibration_file: Optional[str] = None,
#     output_file: Optional[str] = None,
#     model_name: str = "tims",
#     multi: bool = False,
#     log_level: str = "info",
#     n_jobs: Optional[int] = None,
#     use_single_model: bool = True,
#     calibrate_per_charge: bool = True,
#     use_charge_state: int = 2,
#     ion_mobility: bool = False,
# ) -> None:
#     """
#     IM2Deep: Predict CCS values for peptides using deep learning.

#     IM2Deep predicts Collisional Cross Section (CCS) values for peptides,
#     including those with post-translational modifications. The tool supports
#     both single-conformer and multi-conformer predictions with optional
#     calibration using reference datasets.

#     INPUT_FILE should be a CSV file with columns:
#     \b
#     - seq: Peptide sequence (required)
#     - modifications: Modifications in format "position|name" (required, can be empty)
#     - charge: Charge state (required)

#     For calibration files, an additional 'CCS' column with observed values is required.

#     Examples:
#     \b
#         # Basic prediction
#         im2deep peptides.csv

#         # With calibration (recommended)
#         im2deep peptides.csv -c calibration.csv

#         # Multi-conformer prediction
#         im2deep peptides.csv -c calibration.csv -e

#         # Ion mobility output
#         im2deep peptides.csv -c calibration.csv -i

#         # Ensemble prediction with custom output
#         im2deep peptides.csv -c calibration.csv -o results.csv --use-single-model False
#     """
#     try:
#         # Setup logging first
#         setup_logging(log_level)

#         LOGGER.info("IM2Deep command-line interface started")
#         LOGGER.debug(
#             f"Input arguments: psm_file={psm_file}, calibration_file={calibration_file}, "
#             f"multi={multi}, ion_mobility={ion_mobility}"
#         )

#         # Import main functionality (after logging setup)
#         from im2deep._exceptions import IM2DeepError
#         from im2deep.im2deep import predict_ccs

#         # Check optional dependencies if multi-conformer requested
#         if multi:
#             check_optional_dependencies()

#         # Validate input files
#         _validate_file_format(psm_file, "input")
#         if calibration_file:
#             _validate_file_format(calibration_file, "calibration")

#         # Parse input files
#         LOGGER.info("Parsing input files...")

#         # Try to determine file format
#         with open(psm_file, "r", encoding="utf-8") as f:
#             first_line = f.readline().strip()

#         # Check if it's the expected CSV format
#         if "modifications" in first_line and "seq" in first_line:
#             psm_list_pred = _parse_csv_input(psm_file, "prediction")
#             df_pred = pd.read_csv(psm_file).fillna("")
#         else:
#             # Try psm_utils for other formats
#             try:
#                 psm_list_pred = read_file(psm_file)
#                 df_pred = None
#                 LOGGER.info(f"Loaded {len(psm_list_pred)} PSMs using psm_utils")
#             except PSMUtilsIOException as e:
#                 raise click.ClickException(
#                     f"Could not parse input file. Expected CSV with columns 'seq', 'modifications', 'charge' "
#                     f"or a format supported by psm_utils. Error: {e}"
#                 )

#         # Parse calibration file
#         psm_list_cal = None
#         df_cal = None
#         if calibration_file:
#             with open(calibration_file, "r", encoding="utf-8") as f:
#                 cal_first_line = f.readline().strip()

#             if (
#                 "modifications" in cal_first_line
#                 and "seq" in cal_first_line
#                 and "CCS" in cal_first_line
#             ):
#                 psm_list_cal = _parse_csv_input(calibration_file, "calibration")
#                 df_cal = pd.read_csv(calibration_file).fillna("")
#             else:
#                 raise click.ClickException(
#                     "Calibration file must be CSV with columns: 'seq', 'modifications', 'charge', 'CCS'"
#                 )
#         else:
#             LOGGER.warning(
#                 "No calibration file provided. Predictions will be uncalibrated. "
#                 "Calibration is HIGHLY recommended for accurate results."
#             )

#         # Set up output file
#         if not output_file:
#             input_path = Path(psm_file)
#             output_file = input_path.parent / f"{input_path.stem}_IM2Deep-predictions.csv"

#         LOGGER.info(f"Output will be written to: {output_file}")

#         # Run prediction
#         LOGGER.info("Starting CCS prediction...")
#         predict_ccs(
#             psm_list_pred,
#             psm_list_cal,
#             output_file=output_file,
#             model_name=model_name,
#             multi=multi,
#             calibrate_per_charge=calibrate_per_charge,
#             use_charge_state=use_charge_state,
#             n_jobs=n_jobs,
#             use_single_model=use_single_model,
#             ion_mobility=ion_mobility,
#             pred_df=df_pred,
#             cal_df=df_cal,
#             write_output=True,
#         )

#         LOGGER.info("IM2Deep completed successfully!")

#     except IM2DeepError as e:
#         LOGGER.error(f"IM2Deep error: {e}")
#         sys.exit(1)
#     except click.ClickException:
#         # Re-raise click exceptions to preserve formatting
#         raise
#     except Exception as e:
#         LOGGER.error(f"Unexpected error: {e}")
#         if log_level.lower() == "debug":
#             LOGGER.exception("Full traceback:")
#         sys.exit(1)


if __name__ == "__main__":
    main()
