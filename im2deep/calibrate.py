"""
Calibration functions for CCS predictions in IM2Deep.

This module provides functions for calibrating CCS predictions using reference datasets. Calibration is performed by calculating
shift factors based on overlapping peptides between calibration and reference data.

The calibration process involves:
1. Finding overlapping peptide-charge pairs between calibration and reference datasets
2. Calculating mean CCS differences (shift factors)
3. Applying shifts to predictions either globally or per charge state

Functions:
    get_ccs_shift: Calculate global CCS shift factor for a specific charge state
    get_ccs_shift_per_charge: Calculate CCS shift factors per charge state
    calculate_ccs_shift: Wrapper function for shift calculation with validation
    linear_calibration: Apply linear calibration to CCS predictions

Example:
    >>> calibrated_df = linear_calibration(
    ...     predictions_df,
    ...     calibration_df,
    ...     reference_df,
    ...     per_charge=True
    ... )
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
import pandas as pd

from im2deep._exceptions import CalibrationError

LOGGER = logging.getLogger(__name__)


def _validate_calibration_inputs(
    cal_df: pd.DataFrame,
    reference_dataset: pd.DataFrame,
    required_cal_columns: list | None = None,
    required_ref_columns: list | None = None,
) -> None:
    """
    Validate input dataframes for calibration functions.

    Parameters
    ----------
    cal_df
        Calibration dataset
    reference_dataset
        Reference dataset
    required_cal_columns
        Required columns for calibration dataset
    required_ref_columns
        Required columns for reference dataset

    Raises
    ------
    CalibrationError
        If validation fails

    """
    if cal_df.empty:
        raise CalibrationError("Calibration dataset is empty")
    if reference_dataset.empty:
        raise CalibrationError("Reference dataset is empty")

    if required_cal_columns:
        missing_cols = set(required_cal_columns) - set(cal_df.columns)
        if missing_cols:
            raise CalibrationError(f"Missing columns in calibration data: {missing_cols}")

    if required_ref_columns:
        missing_cols = set(required_ref_columns) - set(reference_dataset.columns)
        if missing_cols:
            raise CalibrationError(f"Missing columns in reference data: {missing_cols}")


def get_ccs_shift(
    cal_df: pd.DataFrame, reference_dataset: pd.DataFrame, use_charge_state: int = 2
) -> float:
    """
    Calculate CCS shift factor for a specific charge state.

    This function calculates a constant offset based on identical precursors
    between calibration and reference datasets for a specific charge state.
    The shift represents how much the calibration CCS values differ from
    reference CCS values on average.

    Parameters
    ----------
    cal_df
        PSMs with CCS values. Must contain columns: 'sequence', 'charge', 'ccs_observed'
    reference_dataset
        Reference dataset with CCS values. Must contain columns: 'peptidoform', 'charge', 'CCS'
    use_charge_state
        Charge state to use for CCS shift calculation. Should be in range [2,4].

    Returns
    -------
    float
        CCS shift factor. Positive values indicate calibration CCS is higher
        than reference CCS on average.

    Raises
    ------
    CalibrationError
        If charge state is invalid or no overlapping peptides found

    Notes
    -----
    The function:
    1. Filters both datasets to the specified charge state
    2. Merges on sequence and charge to find overlapping peptides
    3. Calculates mean difference: mean(ccs_observed - CCS_reference)

    Examples
    --------
    >>> shift = get_ccs_shift(calibration_df, reference_df, use_charge_state=2)
    >>> print(f"CCS shift factor: {shift:.2f} Ų")

    """
    # Validate inputs
    _validate_calibration_inputs(
        cal_df,
        reference_dataset,
        required_cal_columns=["sequence", "charge", "ccs_observed"],
        required_ref_columns=["peptidoform", "charge", "CCS"],
    )

    if not 1 <= use_charge_state <= 6:
        raise CalibrationError(
            f"Invalid charge state {use_charge_state}. Should be between 1 and 6."
        )

    LOGGER.debug(f"Using charge state {use_charge_state} for CCS shift calculation.")

    # Filter data by charge state
    reference_tmp = reference_dataset[reference_dataset["charge"] == use_charge_state]
    df_tmp = cal_df[cal_df["charge"] == use_charge_state]

    if reference_tmp.empty:
        LOGGER.warning(f"No reference data found for charge state {use_charge_state}")
        return 0.0

    if df_tmp.empty:
        LOGGER.warning(f"No calibration data found for charge state {use_charge_state}")
        return 0.0

    # Merge datasets to find overlapping peptides
    both = pd.merge(
        left=reference_tmp,
        right=df_tmp,
        right_on=["sequence", "charge"],
        left_on=["peptidoform", "charge"],
        how="inner",
        suffixes=("_ref", "_data"),
    )

    LOGGER.debug(
        f"Calculating CCS shift based on {both.shape[0]} overlapping peptide-charge pairs "
        f"between PSMs and reference dataset"
    )

    if both.empty:
        LOGGER.warning("No overlapping peptides found between calibration and reference data")
        return 0.0

    if both.shape[0] < 10:
        LOGGER.warning(
            f"Only {both.shape[0]} overlapping peptides found. "
            "Consider using more calibration data for reliable results."
        )

    # Calculate shift: how much calibration CCS is larger than reference CCS
    shift = np.mean(both["ccs_observed"] - both["CCS"])

    if abs(shift) > 100:  # Sanity check for unreasonably large shifts
        LOGGER.warning(
            f"Large CCS shift detected ({shift:.2f} Å^2). "
            "Please verify calibration and reference data quality."
        )

    return float(shift)


def get_ccs_shift_per_charge(
    cal_df: pd.DataFrame, reference_dataset: pd.DataFrame
) -> dict[int, float]:
    """
    Calculate CCS shift factors per charge state.

    This function calculates individual shift factors for each charge state
    present in both calibration and reference datasets. This allows for
    charge-specific calibration which often improves accuracy.

    Parameters
    ----------
    cal_df
        PSMs with CCS values. Must contain columns: 'sequence', 'charge', 'ccs_observed'
    reference_dataset
        Reference dataset with CCS values. Must contain columns: 'peptidoform', 'charge', 'CCS'

    Returns
    -------
    Dict[int, float]
        Dictionary mapping charge states to their shift factors.
        Keys are charge states (int), values are shift factors (float).

    Raises
    ------
    CalibrationError
        If required columns are missing or no overlapping data found

    Notes
    -----
    The function:
    1. Merges calibration and reference data on sequence and charge
    2. Groups by charge state
    3. Calculates mean difference for each charge state

    Charge states with insufficient data (< 5 overlapping peptides) will be
    logged as warnings but still included in results.

    Examples
    --------
    >>> shifts = get_ccs_shift_per_charge(calibration_df, reference_df)
    >>> print(shifts)
    {2: 5.2, 3: 3.8, 4: 2.1}

    """
    # Validate inputs
    _validate_calibration_inputs(
        cal_df,
        reference_dataset,
        required_cal_columns=["sequence", "charge", "ccs_observed"],
        required_ref_columns=["peptidoform", "charge", "CCS"],
    )

    # Merge datasets to find overlapping peptides
    both = pd.merge(
        left=reference_dataset,
        right=cal_df,
        right_on=["sequence", "charge"],
        left_on=["peptidoform", "charge"],
        how="inner",
        suffixes=("_ref", "_data"),
    )

    if both.empty:
        raise CalibrationError(
            "No overlapping peptides found between calibration and reference data"
        )

    LOGGER.debug(f"Found {both.shape[0]} total overlapping peptide-charge pairs")

    # Check data distribution across charge states
    charge_counts = both.groupby("charge").size()
    LOGGER.debug(f"Peptides per charge state: {charge_counts.to_dict()}")

    # Warn about charge states with low data
    low_data_charges = charge_counts[charge_counts < 5].index.tolist()
    if low_data_charges:
        LOGGER.warning(
            f"Charge states with <5 peptides: {low_data_charges}. "
            "Consider using global calibration for these charges."
        )

    # Calculate shift per charge state
    shift_dict = (
        both.groupby("charge").apply(lambda x: np.mean(x["ccs_observed"] - x["CCS"])).to_dict()
    )

    # Convert numpy types to native Python types for JSON serialization
    shift_dict = {int(k): float(v) for k, v in shift_dict.items()}

    # Check for unreasonably large shifts
    large_shifts = {k: v for k, v in shift_dict.items() if abs(v) > 100}
    if large_shifts:
        LOGGER.warning(f"Large CCS shifts detected: {large_shifts}. Please verify data quality.")

    return shift_dict


def calculate_ccs_shift(
    cal_df: pd.DataFrame,
    reference_dataset: pd.DataFrame,
    per_charge: bool = True,
    use_charge_state: int | None = None,
) -> float | dict[int, float]:
    """
    Calculate CCS shift factors with validation and filtering.

    This is the main interface for calculating CCS shift factors. It provides
    input validation, charge filtering, and can return either global or
    per-charge shift factors.

    Parameters
    ----------
    cal_df
        PSMs with CCS values. Must contain columns: 'sequence', 'charge', 'ccs_observed'
    reference_dataset
        Reference dataset with CCS values. Must contain columns: 'peptidoform', 'charge', 'CCS'
    per_charge
        Whether to calculate shift factors per charge state. If False, calculates
        a single global shift factor using the specified charge state.
    use_charge_state
        Charge state to use for global shift calculation when per_charge=False.
        Should be in range [2,4]. Default is 2 if not specified.

    Returns
    -------
    Union[float, Dict[int, float]]
        If per_charge=True: Dictionary mapping charge states to shift factors
        If per_charge=False: Single shift factor (float)

    Raises
    ------
    CalibrationError
        If validation fails or invalid parameters provided

    Notes
    -----
    The function automatically filters out charges >6 as IM2Deep predictions
    are not reliable for very high charge states. A warning is logged if
    any peptides are filtered out.

    Examples
    --------
    >>> # Per-charge calibration
    >>> shifts = calculate_ccs_shift(cal_df, ref_df, per_charge=True)
    >>>
    >>> # Global calibration using charge 2
    >>> shift = calculate_ccs_shift(cal_df, ref_df, per_charge=False, use_charge_state=2)

    """
    # Validate inputs
    _validate_calibration_inputs(cal_df, reference_dataset)

    if use_charge_state is not None and not 1 <= use_charge_state <= 6:
        raise CalibrationError(
            f"Invalid charge state {use_charge_state}. Should be between 1 and 6."
        )

    # Filter high charge states (IM2Deep predictions are unreliable >6)
    original_size = len(cal_df)
    cal_df = cal_df[cal_df["charge"] < 7].copy()

    if len(cal_df) < original_size:
        filtered_count = original_size - len(cal_df)
        LOGGER.info(
            f"Filtered out {filtered_count} peptides with charge >6 "
            "(predictions not reliable for z>6)"
        )

    if cal_df.empty:
        raise CalibrationError("No valid calibration data remaining after filtering")

    if not per_charge:
        # Global calibration using specified charge state
        if use_charge_state is None:
            use_charge_state = 2
            LOGGER.debug("No charge state specified for global calibration, using charge 2")

        shift_factor = get_ccs_shift(cal_df, reference_dataset, use_charge_state)
        LOGGER.debug(f"Global CCS shift factor: {shift_factor:.3f}")
        return shift_factor
    else:
        # Per-charge calibration
        shift_factor_dict = get_ccs_shift_per_charge(cal_df, reference_dataset)
        LOGGER.debug(f"CCS shift factors per charge: {shift_factor_dict}")
        return shift_factor_dict


def linear_calibration(
    preds_df: pd.DataFrame,
    calibration_dataset: pd.DataFrame,
    reference_dataset: pd.DataFrame,
    per_charge: bool = True,
    use_charge_state: int | None = None,
) -> pd.DataFrame:
    """
    Calibrate CCS predictions using linear calibration.

    This function performs linear calibration of CCS predictions by applying
    shift factors calculated from overlapping peptides between calibration
    and reference datasets. Calibration can be applied globally or per charge state.

    Parameters
    ----------
    preds_df
        PSMs with CCS predictions. Must contain 'predicted_ccs' column.
        Will be modified to include 'charge' and 'shift' columns.
    calibration_dataset
        Calibration dataset with observed CCS values. Must contain columns:
        'peptidoform', 'ccs_observed'
    reference_dataset
        Reference dataset with CCS values. Must contain columns:
        'peptidoform', 'CCS'
    per_charge
        Whether to calculate and apply shift factors per charge state.
        If True, uses charge-specific calibration with fallback to global shift.
        If False, applies single global shift factor.
    use_charge_state
        Charge state to use for global shift calculation when per_charge=False.
        Default is 2 if not specified.

    Returns
    -------
    pd.DataFrame
        Calibrated PSMs with updated 'predicted_ccs' values and added 'shift' column.

    Raises
    ------
    CalibrationError
        If calibration fails due to data issues or missing columns

    Notes
    -----
    The calibration process:
    1. Extracts sequence and charge information from peptidoforms
    2. Calculates shift factors from calibration vs reference data
    3. Applies shifts to predictions
    4. For per-charge calibration: uses charge-specific shifts with global fallback

    Per-charge calibration is recommended as it typically provides better accuracy
    by accounting for charge-dependent systematic biases.

    Examples
    --------
    >>> # Per-charge calibration (recommended)
    >>> calibrated_df = linear_calibration(
    ...     predictions_df,
    ...     calibration_df,
    ...     reference_df,
    ...     per_charge=True
    ... )
    >>>
    >>> # Global calibration using charge 2
    >>> calibrated_df = linear_calibration(
    ...     predictions_df,
    ...     calibration_df,
    ...     reference_df,
    ...     per_charge=False,
    ...     use_charge_state=2
    ... )
    """

    LOGGER.info("Calibrating CCS values using linear calibration...")

    # Validate input dataframes
    if preds_df.empty:
        raise CalibrationError("Predictions dataframe is empty")
    if "predicted_ccs" not in preds_df.columns:
        raise CalibrationError("Predictions dataframe missing 'predicted_ccs' column")

    # Create working copy to avoid modifying original
    preds_df = preds_df.copy()
    calibration_dataset = calibration_dataset.copy()
    reference_dataset = reference_dataset.copy()

    try:
        # Extract sequence and charge from calibration peptidoforms
        LOGGER.debug("Extracting sequence and charge from calibration peptidoforms...")
        calibration_dataset["sequence"] = calibration_dataset["peptidoform"].apply(
            lambda x: x.proforma.split("\\")[0] if hasattr(x, "proforma") else str(x).split("/")[0]
        )
        calibration_dataset["charge"] = calibration_dataset["peptidoform"].apply(
            lambda x: (
                x.precursor_charge if hasattr(x, "precursor_charge") else int(str(x).split("/")[1])
            )
        )

        # Extract charge from reference peptidoforms
        LOGGER.debug("Extracting charge from reference peptidoforms...")
        reference_dataset["charge"] = reference_dataset["peptidoform"].apply(
            lambda x: int(x.split("/")[1]) if isinstance(x, str) else x.precursor_charge
        )

    except (AttributeError, ValueError, IndexError) as e:
        raise CalibrationError(f"Error parsing peptidoform data: {e}") from e

    if per_charge:
        LOGGER.info("Calculating general shift factor for fallback...")
        try:
            general_shift = calculate_ccs_shift(
                calibration_dataset,
                reference_dataset,
                per_charge=False,
                use_charge_state=use_charge_state or 2,
            )
            # per_charge=False returns float
            general_shift = cast(float, general_shift)
        except CalibrationError as e:
            LOGGER.warning(
                f"Could not calculate general shift factor: {e}. Using 0.0 as fallback."
            )
            general_shift = 0.0

        LOGGER.info("Calculating shift factors per charge state...")
        shift_factor_dict = calculate_ccs_shift(
            calibration_dataset, reference_dataset, per_charge=True
        )
        # per_charge=True returns dict[int, float]
        shift_factor_dict = cast(dict[int, float], shift_factor_dict)

        # Add charge information to predictions if not present
        if "charge" not in preds_df.columns:
            preds_df["charge"] = preds_df["peptidoform"].apply(
                lambda x: x.precursor_charge if hasattr(x, "precursor_charge") else 2
            )

        # Apply charge-specific shifts with fallback to general shift
        preds_df["shift"] = preds_df["charge"].map(shift_factor_dict).fillna(general_shift)
        preds_df["predicted_ccs"] = preds_df["predicted_ccs"] + preds_df["shift"]

        # Log calibration statistics
        used_charges = set(shift_factor_dict.keys())
        fallback_charges = set(preds_df[preds_df["shift"] == general_shift]["charge"].unique())
        if fallback_charges:
            LOGGER.info(f"Used charge-specific calibration for charges: {sorted(used_charges)}")
            LOGGER.info(f"Used fallback calibration for charges: {sorted(fallback_charges)}")

    else:
        # Global calibration
        shift_factor = calculate_ccs_shift(
            calibration_dataset,
            reference_dataset,
            per_charge=False,
            use_charge_state=use_charge_state or 2,
        )
        # per_charge=False returns floats
        shift_factor = cast(float, shift_factor)
        preds_df["predicted_ccs"] += shift_factor
        preds_df["shift"] = shift_factor
        LOGGER.info(f"Applied global shift factor: {shift_factor:.3f}")

    LOGGER.info("CCS values calibrated successfully.")
    return preds_df
