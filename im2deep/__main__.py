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

"""

from __future__ import annotations

import cProfile
import logging
from pathlib import Path

import click
from rich.console import Console
from rich.text import Text

from im2deep import __version__, core
from im2deep._io_helpers import (
    DefaultCommandGroup,
    infer_output_name,
    parse_input,
    setup_logging,
    write_output,
)

console = Console()
LOGGER = logging.getLogger(__name__)


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
@click.option(
    "--profile",
    is_flag=True,
    default=False,
    help="Enable profiling with cProfile. Results saved to 'im2deep_profile.prof'.",
)
@click.option(
    "--profile-name",
    type=click.Path(dir_okay=False),
    default="im2deep_profile.prof",
    help="Output file name for cProfile results when --profile is enabled.",
)
@click.version_option(version=__version__)
def cli(ctx, logging_level, profile, profile_name):
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
    ctx.obj["profile"] = profile
    ctx.obj["profile_name"] = profile_name

    console.print(_build_credits())


# TODO:  Check that parameters match predict function in core
# Implement psm_utils reading for calibration and prediction PSMLists
@cli.command()
@click.pass_context
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
    help="Enable multi-conformer prediction.",
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
def predict(ctx, *args, **kwargs):
    """
    Predict CCS values for peptides (default command).

    If no calibration file is provided with -c, performs prediction only.
    With -c, performs calibration and prediction for improved accuracy.
    """
    # Check if profiling is enabled from parent context
    profile_enabled = ctx.obj.get("profile", False)
    profiler = None

    if profile_enabled:
        # Run with profiling
        profiler = cProfile.Profile()
        profiler.enable()

    try:
        _run_predict(*args, **kwargs)
    finally:
        if profiler is not None:
            profiler.disable()

            # Get the IM2Deep root directory (two levels up from this file)
            root_dir = Path(__file__).parent.parent
            profiles_dir = root_dir / "profiles"
            profiles_dir.mkdir(exist_ok=True)

            profile_output = profiles_dir / ctx.obj.get("profile_name", "im2deep_profile.prof")
            profiler.dump_stats(profile_output)
            LOGGER.info(f"Profiling data saved to {profile_output}")
            LOGGER.info(f"View with: snakeviz {profile_output}")


def _run_predict(*args, **kwargs):
    """Internal function that performs the actual prediction work."""
    # Setup logging first
    setup_logging(kwargs.get("logging_level", "info"))

    LOGGER.info("Starting IM2Deep CCS prediction...")
    LOGGER.debug(
        f"Input arguments: precursors={kwargs.get('precursors')}, "
        f"calibration_precursors={kwargs.get('calibration_precursors')}, multi={kwargs.get('multi')}, "
        f"ion_mobility={kwargs.get('ion_mobility')}"
    )

    # Parse input files
    LOGGER.info("Parsing input files...")
    psm_list = parse_input(Path(kwargs["precursors"]))

    # Run prediction
    LOGGER.info("Running CCS prediction...")
    if kwargs.get("calibration_precursors"):
        LOGGER.info("Calibration file provided, performing calibration and prediction...")
        psm_list_cal = parse_input(Path(kwargs["calibration_precursors"]))
        predictions = core.predict_and_calibrate(psm_list, psm_list_cal, *args, **kwargs)
    else:
        LOGGER.info(
            "No calibration file provided (calibration is HIGHLY recommended), performing prediction only..."
        )
        predictions = core.predict(*args, **kwargs)

    # Output results
    LOGGER.info("IM2Deep CCS prediction completed successfully!")
    output_name = kwargs.pop("output_file")
    output_name = infer_output_name(kwargs["precursors"], output_name).with_suffix(".csv")
    LOGGER.info(f"Writing output file to {output_name}...")
    write_output(output_name, predictions, psm_list, kwargs.get("ion_mobility", False))
    LOGGER.info("Output file written successfully.")
    LOGGER.info("IM2Deep finished.")


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


def _build_credits():
    """Build credits"""
    text = Text()
    text.append("\n")
    text.append("IM2Deep\n", style="bold link https://github.com/compomics/im2deep")
    text.append("Developed at CompOmics, VIB / Ghent University, Belgium.\n")
    text.append("Please cite: ")
    text.append(
        "Devreese et al. Anal. Chem. (2025)",
        style="link https://pubs.acs.org/doi/10.1021/acs.analchem.5c01142",
    )
    text.append("\n")
    text.stylize("cyan")
    return text


def main():
    cli(obj={})


if __name__ == "__main__":
    main()
    _build_credits()
