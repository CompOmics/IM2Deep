"""
Multi-conformer CCS prediction module for IM2Deep.

This module provides functionality for predicting CCS values for peptides that
can exist in multiple conformational states. It uses specialized neural network
models trained to predict multiple CCS values per peptide.

The multi-conformer prediction pipeline:
1. Extract molecular features from peptide sequences
2. Run multi-output neural network models
3. Apply calibration using multi-conformer reference data
4. Return multiple CCS predictions per peptide

Functions:
    get_ccs_shift_multi: Calculate CCS shift for multi-conformer predictions
    get_ccs_shift_per_charge_multi: Calculate per-charge shifts for multi predictions
    calculate_ccs_shift_multi: Main shift calculation with validation
    linear_calibration_multi: Apply calibration to multi-conformer predictions
    predict_multi: Main function for multi-conformer CCS prediction

Dependencies:
    - torch: For neural network inference
    - im2deeptrainer: For handling specialized im2deep models
    - pandas/numpy: For data manipulation

Note:
    This module requires optional dependencies that can be installed with:
    pip install 'im2deep[er]'
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

try:
    import torch
    from im2deeptrainer.extract_data import _get_matrices  # TODO: Should be public function?
    from im2deeptrainer.model import IM2DeepMultiTransfer
    from im2deeptrainer.utils import FlexibleLossSorted

    TORCH_AVAILABLE = True
except ImportError:
    # Optional dependencies not available
    torch = None
    IM2DeepMultiTransfer = None
    _get_matrices = None
    FlexibleLossSorted = None
    TORCH_AVAILABLE = False

from im2deep._exceptions import CalibrationError, IM2DeepError
from im2deep.utils import multi_config

LOGGER = logging.getLogger(__name__)
MULTI_CKPT_PATH: Path = Path(__file__).parent / "models" / "TIMS_multi" / "multi_output.ckpt"
REFERENCE_DATASET_PATH: Path = Path(__file__).parent / "reference_data" / "multi_reference_ccs.gz"


def _validate_multi_inputs(df_cal: pd.DataFrame, reference_dataset: pd.DataFrame) -> None:
    """
    Validate inputs for multi-conformer calibration.

    Parameters
    ----------
    df_cal
        Calibration dataset
    reference_dataset
        Reference dataset

    Raises
    ------
    CalibrationError
        If validation fails
    """
    required_cal_cols = ["seq", "modifications", "charge", "CCS"]
    required_ref_cols = ["seq", "modifications", "charge", "ccs_observed"]

    if df_cal.empty:
        raise CalibrationError("Calibration dataset is empty")
    if reference_dataset.empty:
        raise CalibrationError("Reference dataset is empty")

    missing_cal = set(required_cal_cols) - set(df_cal.columns)
    if missing_cal:
        raise CalibrationError(f"Missing columns in calibration data: {missing_cal}")

    missing_ref = set(required_ref_cols) - set(reference_dataset.columns)
    if missing_ref:
        raise CalibrationError(f"Missing columns in reference data: {missing_ref}")


def get_ccs_shift_multi(
    df_cal: pd.DataFrame, reference_dataset: pd.DataFrame, use_charge_state: int = 2
) -> float:
    """
    Calculate CCS shift factor for multi-conformer predictions.

    This function calculates a shift factor specifically for multi-conformer
    predictions by comparing calibration data with reference data for a
    specific charge state.

    Parameters
    ----------
    df_cal
        Calibration peptides with observed CCS values. Must contain columns:
        'seq', 'modifications', 'charge', 'ccs_observed'
    reference_dataset
        Reference dataset with known CCS values. Must contain columns:
        'seq', 'modifications', 'charge', 'CCS'
    use_charge_state
        Charge state to use for CCS shift calculation. Recommended range [2,4].

    Returns
    -------
    float
        CCS shift factor for multi-conformer predictions. Positive values
        indicate calibration CCS is higher than reference on average.

    Raises
    ------
    CalibrationError
        If inputs are invalid or no overlapping data found

    Notes
    -----
    Multi-conformer shift calculation differs from single-conformer by:
    - Using sequence + modifications for matching instead of peptidoform
    - Typically having fewer overlapping peptides due to stricter matching
    - Requiring specific reference data trained for multi-conformer models

    Examples
    --------
    >>> shift = get_ccs_shift_multi(cal_df, ref_df, use_charge_state=2)
    >>> print(f"Multi-conformer shift: {shift:.2f} Ų")
    """
    _validate_multi_inputs(df_cal, reference_dataset)

    if not use_charge_state <= 6:
        raise CalibrationError(f"Invalid charge state {use_charge_state}")

    LOGGER.debug(
        f"Using charge state {use_charge_state} for calibration of multi-conformer predictions."
    )

    # Filter by charge state
    reference_tmp = reference_dataset[reference_dataset["charge"] == use_charge_state]
    df_tmp = df_cal[df_cal["charge"] == use_charge_state]

    if reference_tmp.empty or df_tmp.empty:
        LOGGER.warning(
            f"No data found for charge state {use_charge_state} in multi-conformer calibration"
        )
        return 0.0

    # Merge on sequence and modifications for multi-conformer matching
    both = pd.merge(
        left=reference_tmp,
        right=df_tmp,
        on=["seq", "modifications"],
        how="inner",
        suffixes=("_ref", "_data"),
    )

    LOGGER.debug(f"Head of overlapping peptides:\n{both.head()}")

    LOGGER.debug(
        ""
        f"Calculating CCS shift based on {both.shape[0]} overlapping peptide-charge pairs "
        f"between PSMs and reference dataset."
    )

    if both.empty:
        LOGGER.warning("No overlapping peptides found for multi-conformer calibration")
        return 0.0

    if both.shape[0] < 10:
        LOGGER.warning(
            f"Only {both.shape[0]} overlapping peptides found for multi-conformer calibration. "
            "Results may be unreliable."
        )

    # Calculate mean shift
    shift = np.mean(both["ccs_observed"] - both["CCS"])

    if abs(shift) > 50:
        LOGGER.warning(f"Large multi-conformer CCS shift detected ({shift:.2f} Å^2)")

    return float(shift)


def get_ccs_shift_per_charge_multi(
    df_cal: pd.DataFrame, reference_dataset: pd.DataFrame
) -> dict[int, float]:
    """
    Calculate CCS shift factors per charge state for multi-conformer predictions.

    This function calculates charge-specific shift factors for multi-conformer
    predictions, allowing for more accurate calibration across different
    charge states.

    Parameters
    ----------
    df_cal
        Calibration peptides with observed CCS values. Must contain columns:
        'seq', 'modifications', 'charge', 'ccs_observed'
    reference_dataset
        Reference dataset with known CCS values. Must contain columns:
        'seq', 'modifications', 'charge', 'CCS'

    Returns
    -------
    Dict[int, float]
        Dictionary mapping charge states to their shift factors.

    Raises
    ------
    CalibrationError
        If inputs are invalid or no overlapping data found

    Notes
    -----
    Multi-conformer per-charge calibration:
    - Matches peptides exactly on sequence, modifications, and charge
    - Typically yields fewer matches than single-conformer calibration
    - Provides charge-specific corrections for systematic biases

    Examples
    --------
    >>> shifts = get_ccs_shift_per_charge_multi(cal_df, ref_df)
    >>> print("Multi-conformer shifts:", shifts)
    {2: 4.1, 3: 2.8, 4: 1.5}
    """
    _validate_multi_inputs(df_cal, reference_dataset)

    # Merge datasets for exact matching
    both = pd.merge(
        left=reference_dataset,
        right=df_cal,
        on=["seq", "modifications", "charge"],
        how="inner",
        suffixes=("_ref", "_data"),
    )

    if both.empty:
        raise CalibrationError(
            "No overlapping peptides found for multi-conformer per-charge calibration"
        )

    LOGGER.debug(
        f"Found {both.shape[0]} overlapping peptides for multi-conformer per-charge calibration"
    )

    # Check data distribution
    charge_counts = both.groupby("charge").size()
    LOGGER.debug(f"Multi-conformer peptides per charge: {charge_counts.to_dict()}")

    # Warn about insufficient data
    low_data_charges = charge_counts[charge_counts < 5].index.tolist()
    if low_data_charges:
        LOGGER.warning(
            f"Charge states with <5 peptides in multi-conformer calibration: {low_data_charges}"
        )

    # Calculate shifts per charge
    shift_dict = (
        both.groupby("charge").apply(lambda x: np.mean(x["ccs_observed"] - x["CCS"])).to_dict()
    )

    # Convert to native Python types
    shift_dict = {int(k): float(v) for k, v in shift_dict.items()}

    return shift_dict


def calculate_ccs_shift_multi(
    df_cal: pd.DataFrame,
    reference_dataset: pd.DataFrame,
    per_charge: bool = True,
    use_charge_state: int | None = None,
) -> float | dict[int, float]:
    """
    Calculate CCS shift factors for multi-conformer predictions with validation.

    This is the main interface for calculating shift factors for multi-conformer
    predictions. It provides input validation, charge filtering, and supports
    both global and per-charge calibration modes.

    Parameters
    ----------
    df_cal
        Calibration peptides with observed CCS values.
    reference_dataset
        Reference dataset with known CCS values.
    per_charge
        Whether to calculate shift factors per charge state.
    use_charge_state
        Charge state for global calibration when per_charge=False.
        Default is 2 if not specified.

    Returns
    -------
    float | dict[int, float]
        If per_charge=True: Dictionary of shift factors per charge
        If per_charge=False: Single global shift factor

    Raises
    ------
    CalibrationError
        If validation fails or invalid parameters

    Notes
    -----
    Multi-conformer models are typically trained for charges 2-4, so higher
    charges are filtered out automatically. The function logs filtering actions
    for transparency.

    Examples
    --------
    >>> # Per-charge calibration (recommended)
    >>> shifts = calculate_ccs_shift_multi(cal_df, ref_df, per_charge=True)
    >>>
    >>> # Global calibration
    >>> shift = calculate_ccs_shift_multi(cal_df, ref_df, per_charge=False, use_charge_state=2)
    """
    _validate_multi_inputs(df_cal, reference_dataset)

    if use_charge_state is not None and not use_charge_state <= 6:
        raise CalibrationError(f"Invalid charge state {use_charge_state}")

    # Filter charge states (multi-conformer models typically work best for 2-4)
    original_size = len(df_cal)
    df_cal = df_cal[(df_cal["charge"] < 5)].copy()

    if len(df_cal) < original_size:
        filtered_count = original_size - len(df_cal)
        LOGGER.info(
            f"Filtered {filtered_count} peptides outside charge range 2-4 "
            "for multi-conformer calibration"
        )

    if df_cal.empty:
        raise CalibrationError(
            "No valid calibration data for multi-conformer prediction after filtering"
        )

    if not per_charge:
        if use_charge_state is None:
            use_charge_state = 2
            LOGGER.debug("Using charge 2 for global multi-conformer calibration")

        shift_factor = get_ccs_shift_multi(df_cal, reference_dataset, use_charge_state)
        LOGGER.debug(f"Multi-conformer global shift factor: {shift_factor:.3f}")
        return shift_factor
    else:
        shift_factor_dict = get_ccs_shift_per_charge_multi(df_cal, reference_dataset)
        LOGGER.debug(f"Multi-conformer shift factors: {shift_factor_dict}")
        return shift_factor_dict


def linear_calibration_multi(
    df_pred: pd.DataFrame,
    df_cal: pd.DataFrame,
    reference_dataset: pd.DataFrame,
    per_charge: bool = True,
    use_charge_state: int | None = None,
) -> pd.DataFrame:
    """
    Calibrate multi-conformer CCS predictions using linear calibration.

    This function applies linear calibration specifically designed for
    multi-conformer CCS predictions. It calculates and applies shift factors
    to both conformer predictions.

    Parameters
    ----------
    df_pred
        DataFrame with multi-conformer CCS predictions. Must contain columns:
        'predicted_ccs_multi_1', 'predicted_ccs_multi_2', 'peptidoform'
    df_cal
        Calibration dataset with observed CCS values.
    reference_dataset
        Reference dataset for multi-conformer calibration.
    per_charge
        Whether to apply calibration per charge state.
    use_charge_state
        Charge state for global calibration when per_charge=False.

    Returns
    -------
    pd.DataFrame
        DataFrame with calibrated multi-conformer predictions.

    Raises
    ------
    CalibrationError
        If calibration fails

    Notes
    -----
    Multi-conformer calibration:
    - Applies the same shift to both conformer predictions
    - Uses specialized reference data for multi-conformer models
    - Supports both global and per-charge calibration strategies

    The calibration preserves the relative differences between conformers
    while correcting systematic biases.

    Examples
    --------
    >>> calibrated_df = linear_calibration_multi(
    ...     pred_df, cal_df, ref_df, per_charge=True
    ... )
    """
    LOGGER.info("Calibrating multi-conformer predictions using linear calibration...")

    if df_pred.empty:
        raise CalibrationError("Predictions dataframe is empty")

    required_cols = ["predicted_ccs_multi_1", "predicted_ccs_multi_2", "peptidoform"]
    missing_cols = set(required_cols) - set(df_pred.columns)
    if missing_cols:
        raise CalibrationError(f"Missing columns in predictions: {missing_cols}")

    # Create working copy
    df_pred = df_pred.copy()

    try:
        if per_charge:
            LOGGER.info("Generating general shift factor for multi-conformer predictions...")
            general_shift = calculate_ccs_shift_multi(
                df_cal, reference_dataset, per_charge=False, use_charge_state=use_charge_state or 2
            )
            general_shift = cast(float, general_shift)  # per_charge=False returns float

            LOGGER.info("Getting shift factors per charge state for multi-conformer...")
            df_pred["charge"] = df_pred["peptidoform"].apply(lambda x: x.precursor_charge)
            shift_factor_dict = calculate_ccs_shift_multi(
                df_cal, reference_dataset, per_charge=True
            )
            # per_charge=True returns dict[int, float]
            shift_factor_dict = cast(dict[int, float], shift_factor_dict)

            # Apply charge-specific shifts with fallback
            df_pred["shift_multi"] = df_pred["charge"].map(shift_factor_dict).fillna(general_shift)
            df_pred["predicted_ccs_multi_1"] = (
                df_pred["predicted_ccs_multi_1"] + df_pred["shift_multi"]
            )
            df_pred["predicted_ccs_multi_2"] = (
                df_pred["predicted_ccs_multi_2"] + df_pred["shift_multi"]
            )

        else:
            shift_factor = calculate_ccs_shift_multi(
                df_cal, reference_dataset, per_charge=False, use_charge_state=use_charge_state or 2
            )
            shift_factor = cast(float, shift_factor)  # per_charge=False returns float
            df_pred["predicted_ccs_multi_1"] = df_pred["predicted_ccs_multi_1"] + shift_factor
            df_pred["predicted_ccs_multi_2"] = df_pred["predicted_ccs_multi_2"] + shift_factor
            df_pred["shift_multi"] = shift_factor

        LOGGER.info("Multi-conformer predictions calibrated successfully.")
        return df_pred

    except Exception as e:
        raise CalibrationError(f"Multi-conformer calibration failed: {e}") from e


def predict_multi(
    df_pred_psm_list,
    df_cal: pd.DataFrame | None,
    calibrate_per_charge: bool,
    use_charge_state: int,
) -> pd.DataFrame:
    """
    Generate multi-conformer CCS predictions for peptides.

    This is the main function for multi-conformer CCS prediction. It loads
    the specialized multi-output neural network model and generates predictions
    for multiple conformational states of each peptide.

    Parameters
    ----------
    df_pred_psm_list
        PSM list containing peptides for prediction.
    df_cal
        Calibration dataset. If provided, predictions will be calibrated.
    calibrate_per_charge
        Whether to perform per-charge calibration.
    use_charge_state
        Charge state for global calibration.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'predicted_ccs_multi_1' and 'predicted_ccs_multi_2'
        containing CCS predictions for two conformational states.

    Raises
    ------
    IM2DeepError
        If multi-conformer prediction fails

    Notes
    -----
    Multi-conformer prediction workflow:
    1. Extract molecular features using im2deeptrainer
    2. Load pre-trained multi-output model
    3. Generate predictions for two conformational states
    4. Apply calibration if calibration data provided
    5. Return predictions as DataFrame

    The model predicts two CCS values per peptide, representing the most
    probable conformational states based on the training data.

    Examples
    --------
    >>> multi_preds = predict_multi(psm_list, cal_df, True, 2)
    >>> print(multi_preds.columns)
    ['predicted_ccs_multi_1', 'predicted_ccs_multi_2']
    """
    # Check if optional dependencies are available
    if not TORCH_AVAILABLE:
        raise IM2DeepError(
            "Multi-conformer prediction requires optional dependencies. "
            "Please install with: pip install 'im2deep[er]'"
        )

    try:
        # Initialize model components
        criterion = FlexibleLossSorted()

        # Check if model file exists
        if not MULTI_CKPT_PATH.exists():
            raise IM2DeepError(f"Multi-conformer model not found: {MULTI_CKPT_PATH}")

        model = IM2DeepMultiTransfer.load_from_checkpoint(
            MULTI_CKPT_PATH, config=multi_config, criterion=criterion
        )

        LOGGER.debug("Multi-conformer model loaded successfully")

        # Extract molecular features
        LOGGER.debug("Extracting molecular features for multi-conformer prediction...")
        matrices = _get_matrices(df_pred_psm_list, inference=True)

        # Convert to tensors
        tensors = {}
        for key in matrices:
            tensors[key] = torch.tensor(matrices[key]).type(torch.FloatTensor)

        # Create data loader
        dataset = torch.utils.data.TensorDataset(*[tensors[key] for key in tensors])
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=multi_config["batch_size"], shuffle=False
        )

        # Generate predictions
        model.eval()
        with torch.no_grad():
            preds = []
            for batch in dataloader:
                prediction = model.predict_step(batch, inference=True)
                preds.append(prediction)
            predictions = torch.cat(preds).numpy()

        LOGGER.debug(f"Generated multi-conformer predictions for {len(predictions)} peptides")

        # Convert PSM list to DataFrame and add predictions
        df_pred = df_pred_psm_list.to_dataframe()

        if len(predictions) != len(df_pred):
            raise IM2DeepError(f"Prediction count mismatch: {len(predictions)} vs {len(df_pred)}")

        df_pred["predicted_ccs_multi_1"] = predictions[:, 0]
        df_pred["predicted_ccs_multi_2"] = predictions[:, 1]

        # Apply calibration if calibration data provided
        if df_cal is not None:
            try:
                LOGGER.debug("Loading multi-conformer reference dataset...")
                reference_dataset = pd.read_csv(
                    REFERENCE_DATASET_PATH, compression="gzip", keep_default_na=False
                )

                df_pred = linear_calibration_multi(
                    df_pred,
                    df_cal,
                    reference_dataset=reference_dataset,
                    per_charge=calibrate_per_charge,
                    use_charge_state=use_charge_state,
                )
            except Exception as e:
                LOGGER.warning(f"Multi-conformer calibration failed: {e}")
                LOGGER.warning("Returning uncalibrated multi-conformer predictions")

        return df_pred[["predicted_ccs_multi_1", "predicted_ccs_multi_2"]]

    except Exception as e:
        raise IM2DeepError(f"Multi-conformer prediction failed: {e}") from e
