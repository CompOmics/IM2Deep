"""
Main CCS prediction module for IM2Deep.

This module provides the core functionality for predicting Collisional Cross Section (CCS)
values for peptides using deep learning models. It supports both single-conformer and
multi-conformer predictions with optional calibration.

The module handles:
- Loading and running neural network models for CCS prediction
- Calibrating predictions using reference datasets
- Converting between CCS and ion mobility
- Outputting results in various formats

Functions:
    predict_ccs: Main function for CCS prediction with optional calibration

Dependencies:
    - deeplc: For neural network model infrastructure
    - psm_utils: For peptide data handling
    - pandas/numpy: For data manipulation

Example:
    Basic CCS prediction:
    >>> from im2deep.im2deep import predict_ccs
    >>> predictions = predict_ccs(psm_list, calibration_data)

    Multi-conformer prediction:
    >>> predictions = predict_ccs(psm_list, calibration_data, multi=True)
"""

from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path
from typing import cast

import pandas as pd
from deeplc import DeepLC
from psm_utils.psm_list import PSMList

from im2deep._exceptions import IM2DeepError
from im2deep.calibrate import linear_calibration
from im2deep.utils import ccs2im

LOGGER = logging.getLogger(__name__)
REFERENCE_DATASET_PATH = Path(__file__).parent / "reference_data" / "reference_ccs.zip"


def _validate_inputs(psm_list_pred: PSMList, output_file: str | PathLike | None = None) -> None:
    """
    Validate input parameters for prediction.

    Parameters
    ----------
    psm_list_pred
        PSM list for prediction
    output_file
        Output file path

    Raises
    ------
    IM2DeepError
        If validation fails
    """
    if not isinstance(psm_list_pred, PSMList):
        raise IM2DeepError("psm_list_pred must be a PSMList instance")

    if len(psm_list_pred) == 0:
        raise IM2DeepError("PSM list for prediction is empty")

    if output_file and not isinstance(output_file, (str, PathLike)):
        raise IM2DeepError("output_file must be a string or PathLike object")


def _get_model_paths(model_name: str, use_single_model: bool) -> list[Path]:
    """
    Get model file paths based on model name and configuration.

    Parameters
    ----------
    model_name
        Name of the model ('tims')
    use_single_model
        Whether to use single model or ensemble

    Returns
    -------
    list[Path]
        List of model file paths

    Raises
    ------
    IM2DeepError
        If model files not found
    """
    if model_name == "tims":
        path_model = Path(__file__).parent / "models" / "TIMS"
    else:
        raise IM2DeepError(f"Unsupported model name: {model_name}")

    if not path_model.exists():
        raise IM2DeepError(f"Model directory not found: {path_model}")

    path_model_list = list(path_model.glob("*.keras"))

    if not path_model_list:
        raise IM2DeepError(f"No model files found in {path_model}")

    if use_single_model:
        # Use the third model by default (index 2) for consistency
        if len(path_model_list) > 2:
            selected_model = path_model_list[2]
            LOGGER.debug(f"Using single model: {selected_model}")
            return [selected_model]
        else:
            LOGGER.warning("Less than 3 models available, using first model")
            return [path_model_list[0]]
    else:
        LOGGER.debug(f"Using ensemble of {len(path_model_list)} models")
        return path_model_list


def _write_output_file(
    output_file: str | PathLike,
    psm_list_pred_df: pd.DataFrame,
    pred_df: pd.DataFrame | None = None,
    ion_mobility: bool = False,
    multi: bool = False,
) -> None:
    """
    Write predictions to output file.

    Parameters
    ----------
    output_file
        Path to output file
    psm_list_pred_df
        DataFrame with predictions
    pred_df
        Multi-conformer predictions
    ion_mobility
        Whether to output ion mobility instead of CCS
    multi
        Whether multi-conformer predictions are included
    """
    if multi and pred_df is None:
        raise IM2DeepError("Multi-conformer predictions requested but pred_df is None")
    else:
        pred_df = cast(pd.DataFrame, pred_df)
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            # TODO: Consider using dictwriter or Pandas to_csv
            if ion_mobility:
                if multi:
                    f.write(
                        "modified_seq,charge,predicted IM single,predicted IM multi 1,predicted IM multi 2\n"
                    )
                    for peptidoform, charge, IM_single, IM_multi_1, IM_multi_2 in zip(
                        psm_list_pred_df["peptidoform"],
                        psm_list_pred_df["charge"],
                        psm_list_pred_df["predicted_im"],
                        psm_list_pred_df["predicted_im_multi_1"],
                        psm_list_pred_df["predicted_im_multi_2"],
                        strict=True,
                    ):
                        f.write(f"{peptidoform},{charge},{IM_single},{IM_multi_1},{IM_multi_2}\n")
                else:
                    f.write("modified_seq,charge,predicted IM\n")
                    for peptidoform, charge, IM in zip(
                        psm_list_pred_df["peptidoform"],
                        psm_list_pred_df["charge"],
                        psm_list_pred_df["predicted_im"],
                        strict=True,
                    ):
                        f.write(f"{peptidoform},{charge},{IM}\n")
            else:
                if multi:
                    f.write(
                        "modified_seq,charge,predicted CCS single,predicted CCS multi 1,predicted CCS multi 2\n"
                    )
                    for peptidoform, charge, CCS_single, CCS_multi_1, CCS_multi_2 in zip(
                        psm_list_pred_df["peptidoform"],
                        psm_list_pred_df["charge"],
                        psm_list_pred_df["predicted_ccs"],
                        pred_df["predicted_ccs_multi_1"],
                        pred_df["predicted_ccs_multi_2"],
                        strict=True,
                    ):
                        f.write(
                            f"{peptidoform},{charge},{CCS_single},{CCS_multi_1},{CCS_multi_2}\n"
                        )
                else:
                    f.write("modified_seq,charge,predicted CCS\n")
                    for peptidoform, charge, CCS in zip(
                        psm_list_pred_df["peptidoform"],
                        psm_list_pred_df["charge"],
                        psm_list_pred_df["predicted_ccs"],
                        strict=True,
                    ):
                        f.write(f"{peptidoform},{charge},{CCS}\n")

        LOGGER.info(f"Results written to: {output_file}")

    except OSError as e:
        raise IM2DeepError(f"Failed to write output file {output_file}: {e}") from e


def predict_ccs(
    psm_list_pred: PSMList,
    psm_list_cal: PSMList | pd.DataFrame | None = None,
    file_reference: PathLike | None = None,
    output_file: PathLike | None = None,
    model_name: str = "tims",
    multi: bool = False,
    calibrate_per_charge: bool = True,
    use_charge_state: int = 2,
    use_single_model: bool = True,
    n_jobs: int | None = None,
    write_output: bool = False,
    ion_mobility: bool = False,
    pred_df: pd.DataFrame | None = None,
    cal_df: pd.DataFrame | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Predict CCS values for peptides using IM2Deep models.

    This is the main function for CCS prediction. It can perform single-conformer
    or multi-conformer predictions with optional calibration using reference datasets.

    Parameters
    ----------
    psm_list_pred
        PSM list containing peptides for CCS prediction. Each PSM should contain
        a valid peptidoform with sequence and modifications.
    psm_list_cal
        PSM list or DataFrame for calibration with observed CCS values.
        If PSMList: CCS values should be in metadata with key "CCS".
        If DataFrame: should have "ccs_observed" column.
        Required for calibration. Default is None (no calibration).
    file_reference
        Path to reference dataset file for calibration. Default uses built-in
        reference dataset.
    output_file
        Path to write output predictions. If None, no file is written.
    model_name
        Name of the model to use. Currently only "tims" is supported.
    multi
        Whether to include multi-conformer predictions. Requires optional
        dependencies (torch, im2deeptrainer).
    calibrate_per_charge
        Whether to perform calibration per charge state. If False, uses
        global calibration with specified charge state.
    use_charge_state
        Charge state to use for global calibration when calibrate_per_charge=False.
        Should be in range [2,4] for best results.
    use_single_model
        Whether to use a single model (faster) or ensemble of models (potentially
        more accurate). Single model recommended for most applications.
    n_jobs
        Number of parallel jobs for model prediction. If None, uses all available CPUs.
    write_output
        Whether to write predictions to output file.
    ion_mobility
        Whether to output ion mobility (1/K0) instead of CCS values.
    pred_df
        Pre-computed prediction DataFrame (used internally).
    cal_df
        Pre-computed calibration DataFrame (used internally).

    Returns
    -------
    pd.Series or pd.DataFrame
        If ion_mobility=True: Series with predicted ion mobility values
        If ion_mobility=False: Series with predicted CCS values
        For multi-conformer predictions, additional columns are included.

    Raises
    ------
    IM2DeepError
        If prediction fails due to invalid inputs, missing models, or other errors.

    Notes
    -----
    The prediction workflow:
    1. Validate inputs and load appropriate models
    2. Generate CCS predictions using neural networks
    3. Apply calibration if calibration data provided
    4. Optionally run multi-conformer predictions
    5. Convert to ion mobility if requested
    6. Write output file if requested

    Calibration is highly recommended for accurate predictions and requires a set of peptides with
    known CCS values that overlap with the reference dataset.

    Examples
    --------
    Basic CCS prediction without calibration:
    >>> predictions = predict_ccs(psm_list)

    CCS prediction with calibration:
    >>> predictions = predict_ccs(psm_list, psm_list_calibration)

    Multi-conformer prediction with ion mobility output:
    >>> predictions = predict_ccs(
    ...     psm_list,
    ...     psm_list_calibration,
    ...     multi=True,
    ...     ion_mobility=True
    ... )

    Ensemble prediction with file output:
    >>> predictions = predict_ccs(
    ...     psm_list,
    ...     psm_list_calibration,
    ...     use_single_model=False,
    ...     output_file="predictions.csv",
    ...     write_output=True
    ... )
    """
    LOGGER.info("IM2Deep started.")

    # Validate inputs
    _validate_inputs(psm_list_pred, output_file)

    # Load reference dataset
    if file_reference is None:
        file_reference = REFERENCE_DATASET_PATH

    try:
        reference_dataset = pd.read_csv(file_reference)
        LOGGER.debug(f"Loaded reference dataset with {len(reference_dataset)} entries")
    except Exception as e:
        raise IM2DeepError(f"Failed to load reference dataset from {file_reference}: {e}") from e

    if reference_dataset.empty:
        raise IM2DeepError("Reference dataset is empty")

    # Get model paths
    try:
        path_model_list = _get_model_paths(model_name, use_single_model)
    except Exception as e:
        raise IM2DeepError(f"Failed to load models: {e}") from e

    # Initialize DeepLC for CCS prediction
    try:
        dlc = DeepLC(path_model=path_model_list, n_jobs=n_jobs, predict_ccs=True)
        LOGGER.info("Predicting CCS values...")
        preds = dlc.make_preds(psm_list=psm_list_pred, calibrate=False)
        LOGGER.info(f"CCS values predicted for {len(preds)} peptides.")
    except Exception as e:
        raise IM2DeepError(f"CCS prediction failed: {e}") from e

    if len(preds) == 0:
        raise IM2DeepError("No predictions generated")

    # Convert PSM list to DataFrame and add predictions
    try:
        psm_list_pred_df = psm_list_pred.to_dataframe()
        psm_list_pred_df["predicted_ccs"] = preds
        psm_list_pred_df["charge"] = psm_list_pred_df["peptidoform"].apply(
            lambda x: x.precursor_charge
        )
    except Exception as e:
        raise IM2DeepError(f"Failed to process predictions: {e}") from e

    # Apply calibration if calibration data provided
    pred_df = None
    if psm_list_cal is not None:
        try:
            LOGGER.info("Applying calibration...")

            # Handle both PSMList and DataFrame input
            if isinstance(psm_list_cal, pd.DataFrame):
                # Input is already a DataFrame with ccs_observed column
                psm_list_cal_df = psm_list_cal.copy()
                if "ccs_observed" not in psm_list_cal_df.columns:
                    raise IM2DeepError(
                        "DataFrame calibration data must contain 'ccs_observed' column"
                    )
            else:
                # Input is PSMList, extract CCS from metadata
                ccs_values = []
                for psm in psm_list_cal:
                    if psm.metadata and "CCS" in psm.metadata:
                        ccs_values.append(float(psm.metadata["CCS"]))
                    else:
                        ccs_values.append(None)

                # Convert to DataFrame and add CCS values
                psm_list_cal_df = psm_list_cal.to_dataframe()
                psm_list_cal_df["ccs_observed"] = ccs_values

            # Filter out entries without CCS values
            psm_list_cal_df = psm_list_cal_df[psm_list_cal_df["ccs_observed"].notnull()]

            if psm_list_cal_df.empty:
                LOGGER.warning("No valid calibration data found (missing CCS values)")
            else:
                psm_list_pred_df = linear_calibration(
                    psm_list_pred_df,
                    calibration_dataset=psm_list_cal_df,
                    reference_dataset=reference_dataset,
                    per_charge=calibrate_per_charge,
                    use_charge_state=use_charge_state,
                )
                LOGGER.info("Calibration applied successfully.")

        except Exception as e:
            LOGGER.error(f"Calibration failed: {e}")
            # Continue without calibration rather than failing completely
            LOGGER.warning("Continuing without calibration")

    # Multi-conformer prediction
    if multi:
        try:
            from im2deep.predict_multi import predict_multi

            LOGGER.info("Predicting multiconformer CCS values...")
            pred_df = predict_multi(
                psm_list_pred,
                cal_df,
                calibrate_per_charge,
                use_charge_state,
            )
            LOGGER.info("Multiconformational predictions completed.")
        except ImportError as e:
            raise IM2DeepError(
                "Multi-conformer prediction requires optional dependencies. "
                "Please install with: pip install 'im2deep[er]'"
            ) from e
        except Exception as e:
            raise IM2DeepError(f"Multi-conformer prediction failed: {e}") from e

    # Convert to ion mobility if requested
    if ion_mobility:
        try:
            mz_array = psm_list_pred_df["peptidoform"].apply(lambda x: x.theoretical_mz).to_numpy()
            charge_array = psm_list_pred_df["charge"].to_numpy()

            psm_list_pred_df["predicted_im"] = ccs2im(
                psm_list_pred_df["predicted_ccs"].to_numpy(),
                mz_array,
                charge_array,
            )

            if multi and pred_df is not None:
                psm_list_pred_df["predicted_im_multi_1"] = ccs2im(
                    pred_df["predicted_ccs_multi_1"].to_numpy(),
                    mz_array,
                    charge_array,
                )
                psm_list_pred_df["predicted_im_multi_2"] = ccs2im(
                    pred_df["predicted_ccs_multi_2"].to_numpy(),
                    mz_array,
                    charge_array,
                )

        except Exception as e:
            raise IM2DeepError(f"Ion mobility conversion failed: {e}") from e

    # Write output file if requested
    if write_output and output_file:
        try:
            _write_output_file(output_file, psm_list_pred_df, pred_df, ion_mobility, multi)
        except Exception as e:
            LOGGER.error(f"Failed to write output: {e}")
            # Don't fail the entire prediction because of output issues

    LOGGER.info("IM2Deep finished!")

    # Return appropriate predictions
    if ion_mobility:
        return psm_list_pred_df["predicted_im"]
    else:
        return psm_list_pred_df["predicted_ccs"]
