"""
CCS calibration utilities.

This module provides calibration strategies to map predicted CCS values to the aligned target scale.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import cast

import numpy as np
import pandas as pd
from psm_utils import Peptidoform, PSMList

from im2deep.constants import DEFAULT_MULTI_REFERENCE_DATASET_PATH, DEFAULT_REFERENCE_DATASET_PATH
from im2deep.exceptions import CalibrationError

LOGGER = logging.getLogger(__name__)


class Calibration(ABC):
    """Abstract base class for CCS calibration methods."""

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Check if the calibration has been fitted."""
        ...

    @abstractmethod
    def fit(
        self,
        psm_df_target: pd.DataFrame,
        psm_df_source: pd.DataFrame,
    ) -> None:
        """Fit the calibration using target and source CCS values."""
        ...

    @abstractmethod
    def transform(
        self,
        psm_df: pd.DataFrame,
    ) -> np.ndarray:
        """Transform source CCS into the calibrated target space."""
        ...


class LinearCCSCalibration(Calibration):
    """
    Linear calibration for CCS predictions.

    This class implements a simple linear calibration method for CCS predictions by
    applying shift factors calculated from overlapping peptides between calibration
    and reference datasets. Shift factor calculation can  be performed globally or per
    charge state.

    Parameters
    ----------
    per_charge : bool, optional
        Whether to calculate shift factors per charge state. Default is True.
    use_charge_state : int or None, optional
        Charge state to use for global shift calculation when per_charge is False.
        Default is 2 if not specified.
    """

    def __init__(self, per_charge: bool = True, use_charge_state: int | None = None) -> None:
        super().__init__()
        self.per_charge = per_charge
        self.use_charge_state = use_charge_state
        self.fitted = False
        self.charge_shifts: dict[int, float] = {}
        self.general_shift: float | None = None
        self.used_charges: set[int] = set()
        self.reference_psm_list: PSMList | None = None

    @property
    def is_fitted(self) -> bool:
        return self.fitted

    def fit(
        self,
        psm_df_target: pd.DataFrame,
        psm_df_source: pd.DataFrame | None = None,
        multi: bool = False,
    ) -> None:
        """Fit the calibration using target and source CCS values."""
        if psm_df_source is None:
            LOGGER.debug("No reference PSMList provided, loading default reference dataset.")
            psm_df_source = get_default_reference(multi=multi)

        LOGGER.debug("Calculating calibration parameters...")

        if self.per_charge:
            # For per-charge calibration, calculate shifts for all charges
            LOGGER.debug("Calculating shift factors per charge state...")
            try:
                self.charge_shifts: dict[int, float] = self.calculate_ccs_shift_per_charge(
                    psm_df_target,
                    psm_df_source,
                )
                LOGGER.debug(f"Calculated charge-specific shifts: {self.charge_shifts}")
            except CalibrationError as e:
                LOGGER.warning(
                    f"Could not calculate charge-specific shift factors: {e}. Using 0.0 as fallback."
                )
                self.charge_shifts = dict.fromkeys(range(1, 7), 0.0)

            # Set general shift as the mean of calculated charge shifts or charge 2 if available
            if 2 in self.charge_shifts and self.charge_shifts[2] != 0.0:
                self.general_shift = self.charge_shifts[2]
            else:
                # Use mean of non-zero charge shifts
                available_shifts = [
                    shift
                    for shift in self.charge_shifts.values()
                    if shift is not None and shift != 0.0
                ]
                if available_shifts:
                    self.general_shift = float(np.mean(available_shifts))
                else:
                    self.general_shift = 0.0

            # Fill in missing charge states with general shift
            no_shift_calculated = []
            for charge in range(1, 7):
                if (
                    charge not in self.charge_shifts
                    or self.charge_shifts[charge] is None
                    or self.charge_shifts[charge] == 0.0
                ):
                    no_shift_calculated.append(charge)
                    self.charge_shifts[charge] = float(self.general_shift)
            LOGGER.debug(
                f"No shift factor calculated for charge states: {no_shift_calculated}. "
                f"Using general shift: {self.general_shift:.3f}."
            )
        else:
            # For global calibration, calculate a single shift
            try:
                self.general_shift = self.calculate_ccs_shift_global(
                    psm_df_target,
                    psm_df_source,
                )
            except CalibrationError as e:
                LOGGER.warning(
                    f"Could not calculate general shift factor: {e}. Using 0.0 as fallback."
                )
                self.general_shift = 0.0
            self.charge_shifts = dict.fromkeys(range(1, 7), self.general_shift)

        self.used_charges = set(self.charge_shifts.keys())
        self.fitted = True
        LOGGER.debug(f"CCS shift factors per charge: {self.charge_shifts}")

    def transform(
        self,
        psm_df: pd.DataFrame,
    ) -> np.ndarray:
        """Transform source CCS into the calibrated target space."""
        if not self.is_fitted:
            raise CalibrationError("Calibration has not been fitted yet.")

        LOGGER.debug("Applying calibration to source CCS values...")

        if "peptidoform" not in psm_df.columns:
            raise CalibrationError("Input DataFrame must contain 'peptidoform' column.")

        if "predicted_CCS_uncalibrated" not in psm_df.columns and "metadata" in psm_df.columns:
            psm_df["predicted_CCS_uncalibrated"] = psm_df["metadata"].apply(
                lambda x: (
                    x["predicted_CCS_uncalibrated"]
                    if "predicted_CCS_uncalibrated" in x
                    else np.nan
                )
            )

        # Extract charge from peptidoform column efficiently
        psm_df["charge"] = psm_df["peptidoform"].apply(
            lambda x: int(str(x).split("/")[-1]) if isinstance(x, str) else int(x.precursor_charge)
        )

        if self.per_charge:
            # Per-charge calibration using vectorized map operation
            psm_df["shift"] = psm_df["charge"].map(self.charge_shifts).fillna(0.0)
        else:
            # Global calibration - use same shift for all
            psm_df["shift"] = self.general_shift

        # Apply shift, handling both scalar and array CCS values (for multiconformer predictions)
        def apply_shift(ccs_value, shift_value):
            if isinstance(ccs_value, (list, np.ndarray)):
                # Multiconformer: apply shift to each conformer
                return np.array(ccs_value, dtype=np.float32) + shift_value
            else:
                # Single value
                return float(ccs_value + shift_value)

        psm_df["calibrated_CCS"] = psm_df.apply(
            lambda row: apply_shift(row["predicted_CCS_uncalibrated"], row["shift"]), axis=1
        )

        # Return as numpy object array to preserve multiconformer arrays
        predicted_ccs_calibrated = np.empty(len(psm_df), dtype=object)
        predicted_ccs_calibrated[:] = psm_df["calibrated_CCS"].tolist()

        return predicted_ccs_calibrated

    def calculate_ccs_shift_global(
        self,
        target_df: pd.DataFrame,
        source_df: pd.DataFrame,
    ) -> float:
        """
        Calculate a single global CCS shift factor.

        Parameters
        ----------
        target_df
            DataFrame containing peptidoforms and observed CCS values from the target PSMList.
        source_df
            DataFrame containing peptidoforms and predicted CCS values from the source PSMList.

        Returns
        -------
        dict[int, float] | float
            Shift factors per charge state if per_charge is True, otherwise a single shift factor.

        Raises
        ------
        CalibrationError
            If no overlapping peptides are found for shift calculation.

        Notes
        -----
        The function automatically filters out charges >6 as IM2Deep predictions are not reliable for higher charge states.
        A warning is logged if any peptides are filtered out.
        """
        if self.use_charge_state is not None and not 1 <= self.use_charge_state <= 6:
            raise CalibrationError(
                f"Invalid charge state {self.use_charge_state} for global shift calculation."
            )

        if self.use_charge_state is None:
            self.use_charge_state = 2  # Default charge state
            LOGGER.debug(
                "No charge state specified for global calibration. Using default charge state 2 "
                "for global shift calculation."
            )

        shift_factor = self._compute_ccs_shift(
            target_df,
            source_df,
            self.use_charge_state,
        )
        LOGGER.debug(f"Global CCS shift factor: {shift_factor:.3f}")
        return shift_factor

    def calculate_ccs_shift_per_charge(
        self,
        target_df: pd.DataFrame,
        source_df: pd.DataFrame,
    ) -> dict[int, float]:
        """
        Calculate CCS shift factors per charge state.

        Parameters
        ----------
        target_df
            DataFrame containing peptidoforms and observed CCS values from the target PSMList.
        source_df
            DataFrame containing peptidoforms and predicted CCS values from the source PSMList.

        Returns
        -------
        dict[int, float]
            Shift factors per charge state.

        Raises
        ------
        CalibrationError
            If no overlapping peptides are found for any charge state.
        """
        if self.use_charge_state is not None and not 1 <= self.use_charge_state <= 6:
            raise CalibrationError(
                f"Invalid charge state {self.use_charge_state} for global shift calculation."
            )

        return self._compute_ccs_shift_per_charge(
            target_df,
            source_df,
        )

    @staticmethod
    def _compute_ccs_shift(
        target_df: pd.DataFrame,
        source_df: pd.DataFrame,
        charge_state: int,
    ) -> float:
        """Compute CCS shift for a specific charge state using DataFrame operations."""
        # Prepare DataFrames with proper columns
        target_work = target_df.copy()
        source_work = source_df.copy()

        # Extract peptide keys and charges
        def get_peptide_key(pf):
            if isinstance(pf, Peptidoform):
                # For Peptidoform objects, convert proforma to string and strip charge suffix
                return str(pf.proforma).rsplit("/", 1)[0]
            else:
                # For strings in format "PEPTIDE/charge", split off charge
                return str(pf).rsplit("/", 1)[0]

        def get_charge(pf):
            if isinstance(pf, Peptidoform):
                if pf.precursor_charge is None:
                    raise CalibrationError(
                        f"Peptidoform {pf} is missing precursor charge information."
                    )
                return int(pf.precursor_charge)
            else:
                return int(str(pf).split("/")[-1])

        target_work["peptide_key"] = target_work["peptidoform"].apply(get_peptide_key)
        target_work["charge"] = target_work["peptidoform"].apply(get_charge)

        # Extract CCS from metadata if it's not a direct column
        if "CCS" not in target_work.columns and "metadata" in target_work.columns:
            target_work["CCS"] = target_work["metadata"].apply(
                lambda x: (
                    float(x.get("CCS"))  # type: ignore[union-attr]
                    if isinstance(x, dict) and x.get("CCS") is not None
                    else np.nan
                )
            )

        source_work["peptide_key"] = source_work["peptidoform"].apply(get_peptide_key)
        source_work["charge"] = source_work["peptidoform"].apply(get_charge)

        # Filter by charge state
        target_filtered = target_work[target_work["charge"] == charge_state].copy()
        source_filtered = source_work[source_work["charge"] == charge_state].copy()

        # Merge on peptide key to find overlapping peptides
        merged = pd.merge(
            target_filtered[["peptide_key", "CCS"]],
            source_filtered[["peptide_key", "CCS"]],
            on="peptide_key",
            suffixes=("_target", "_source"),
        )

        LOGGER.debug(
            f"Number of overlapping peptides for charge state {charge_state}: {len(merged)}"
        )

        num_overlapping = len(merged)

        LOGGER.debug(
            f"Calculating CCS shift based on {num_overlapping} overlapping peptides for charge state {charge_state}."
        )

        if num_overlapping == 0:
            LOGGER.warning(f"No overlapping peptides found for charge state {charge_state}.")
            return 0.0

        if num_overlapping < 10:
            LOGGER.warning(
                f"Only {num_overlapping} overlapping peptides found for charge state {charge_state}. "
                "Shift calculation may be unreliable."
            )

        # Calculate shift as mean difference
        shift = (merged["CCS_target"] - merged["CCS_source"]).mean()

        if abs(shift) > 100.0:
            LOGGER.warning(
                f"Unusually large CCS shift ({shift:.2f}) detected for charge state {charge_state}."
                " Please verify the calibration datasets."
            )

        return float(shift)

    @staticmethod
    def _compute_ccs_shift_per_charge(
        target_df: pd.DataFrame,
        source_df: pd.DataFrame,
    ) -> dict[int, float]:
        """
        Calculate CCS shift factors per charge state using DataFrame groupby.

        Parameters
        ----------
        target_df
            DataFrame with peptidoforms and observed CCS values from the target PSMList.
        source_df
            DataFrame with peptidoforms and predicted CCS values from the source PSMList.

        Returns
        -------
        dict[int, float]
            Shift factors per charge state.

        Raises
        ------
        CalibrationError
            If no overlapping peptides are found for any charge state.
        """
        # Prepare DataFrames with proper columns
        target_work = target_df.copy()
        source_work = source_df.copy()

        # Extract peptide keys and charges
        def get_peptide_key(pf):
            if isinstance(pf, Peptidoform):
                # For Peptidoform objects, use proforma property which excludes charge
                return str(pf.proforma).rsplit("/", 1)[0]
            else:
                # For strings in format "PEPTIDE/charge", split off charge
                return str(pf).rsplit("/", 1)[0]

        def get_charge(pf):
            if isinstance(pf, Peptidoform):
                if pf.precursor_charge is None:
                    raise CalibrationError(
                        f"Peptidoform {pf} is missing precursor charge information."
                    )
                return int(pf.precursor_charge)
            else:
                return int(str(pf).split("/")[-1])

        target_work["peptide_key"] = target_work["peptidoform"].apply(get_peptide_key)
        target_work["charge"] = target_work["peptidoform"].apply(get_charge)

        if "CCS" not in target_work.columns and "metadata" in target_work.columns:
            target_work["CCS"] = target_work["metadata"].apply(
                lambda x: float(x["CCS"]) if isinstance(x, dict) and "CCS" in x else np.nan
            )
        if "CCS" not in source_work.columns and "metadata" in source_work.columns:
            source_work["CCS"] = source_work["metadata"].apply(
                lambda x: float(x["CCS"]) if isinstance(x, dict) and "CCS" in x else np.nan
            )

        source_work["peptide_key"] = source_work["peptidoform"].apply(get_peptide_key)
        source_work["charge"] = source_work["peptidoform"].apply(get_charge)

        # Merge on peptide key and charge to find overlapping peptides
        merged = pd.merge(
            target_work[["peptide_key", "charge", "CCS"]],
            source_work[["peptide_key", "charge", "CCS"]],
            on=["peptide_key", "charge"],
            suffixes=("_target", "_source"),
        )

        if len(merged) == 0:
            raise CalibrationError("No overlapping peptides found for shift calculation.")

        # Calculate shift per charge using groupby
        merged["shift"] = merged["CCS_target"] - merged["CCS_source"]
        shift_by_charge = merged.groupby("charge")["shift"].mean()
        shift_factors: dict[int, float] = {
            int(cast(int, charge)): float(shift) for charge, shift in shift_by_charge.items()
        }

        # Log information for each charge state
        charge_counts = merged.groupby("charge").size()
        for charge, count in charge_counts.items():
            charge_int = int(cast(int, charge))
            count_int = int(cast(int, count))
            if count_int < 10:
                LOGGER.warning(
                    f"Only {count_int} overlapping peptides found for charge state {charge_int}. "
                    "Shift calculation may be unreliable."
                )
            if abs(shift_factors[charge_int]) > 100.0:
                LOGGER.warning(
                    f"Unusually large CCS shift ({shift_factors[charge_int]:.2f}) detected for charge state {charge_int}."
                    " Please verify the calibration datasets."
                )

        if len(shift_factors) == 0:
            raise CalibrationError("No CCS shift factors could be calculated.")

        return shift_factors


class SplineCCSCalibration(Calibration):
    """
    Piecewise spline calibration for CCS predictions.

    Fits a low-dimensional piecewise linear (or cubic) spline to capture
    non-linear CCS biases that a single global shift cannot correct (e.g.,
    charge- and mass-dependent drift between MALDI and LC-MS/MS).

    Parameters
    ----------
    n_knots : int
        Number of internal knots for the spline. Default is 3 (giving a
        4-segment piecewise fit).
    degree : int
        Spline degree. 1 = piecewise linear, 3 = cubic. Default is 1.
    """

    def __init__(self, n_knots: int = 3, degree: int = 1) -> None:
        super().__init__()
        self.n_knots = n_knots
        self.degree = degree
        self.fitted = False
        self._spline = None

    @property
    def is_fitted(self) -> bool:
        return self.fitted

    def fit(
        self,
        psm_df_target: pd.DataFrame,
        psm_df_source: pd.DataFrame | None = None,
        multi: bool = False,
    ) -> None:
        """
        Fit a spline mapping predicted (source) CCS to observed (target) CCS.

        Parameters
        ----------
        psm_df_target
            DataFrame with 'peptidoform' and 'CCS' columns (observed values).
        psm_df_source
            DataFrame with 'peptidoform' and 'CCS' columns (predicted values).
            If None, loads the default reference dataset.
        """
        from scipy.interpolate import UnivariateSpline

        if psm_df_source is None:
            psm_df_source = get_default_reference(multi=multi)

        # Find overlapping peptides
        def get_peptide_key(pf):
            if isinstance(pf, Peptidoform):
                return str(pf.proforma).rsplit("/", 1)[0]
            return str(pf).rsplit("/", 1)[0]

        target_work = psm_df_target.copy()
        source_work = psm_df_source.copy()

        target_work["peptide_key"] = target_work["peptidoform"].apply(get_peptide_key)
        source_work["peptide_key"] = source_work["peptidoform"].apply(get_peptide_key)

        merged = pd.merge(
            target_work[["peptide_key", "CCS"]],
            source_work[["peptide_key", "CCS"]],
            on="peptide_key",
            suffixes=("_target", "_source"),
        )

        if len(merged) < self.n_knots + self.degree + 1:
            LOGGER.warning(
                f"Only {len(merged)} overlapping peptides found, too few for "
                f"spline with {self.n_knots} knots. Falling back to linear shift."
            )
            # Fall back to simple linear shift
            if len(merged) > 0:
                shift = (merged["CCS_target"] - merged["CCS_source"]).mean()
            else:
                shift = 0.0
            self._spline = None
            self._fallback_shift = shift
            self.fitted = True
            return

        # Deduplicate by peptide_key (many-to-many merges create duplicates)
        # and average CCS values per unique peptide for a clean spline fit
        merged = merged.groupby("peptide_key", as_index=False).agg(
            CCS_source=("CCS_source", "mean"),
            CCS_target=("CCS_target", "mean"),
        )

        # Sort by source CCS (required for spline fitting)
        merged = merged.sort_values("CCS_source").reset_index(drop=True)

        source_vals = merged["CCS_source"].values.astype(np.float64)
        target_vals = merged["CCS_target"].values.astype(np.float64)

        # Fit spline: source → target
        # Use scipy's UnivariateSpline with a generous smoothing factor
        # to avoid overfitting. Fall back to interp1d if it fails.
        try:
            self._spline = UnivariateSpline(
                source_vals, target_vals,
                k=min(self.degree, 3),
                s=len(merged) * np.var(target_vals - source_vals),
            )
            # Verify it actually works
            test_result = self._spline(source_vals[:3])
            if np.any(np.isnan(test_result)):
                raise ValueError("Spline produced NaN on training data")
        except (ValueError, Exception) as e:
            LOGGER.warning(f"UnivariateSpline failed ({e}), using linear interpolation.")
            from scipy.interpolate import interp1d
            self._spline = interp1d(
                source_vals, target_vals,
                kind="linear", fill_value="extrapolate",
            )

        self._fallback_shift = 0.0
        self.fitted = True

        residuals = merged["CCS_target"] - self._spline(source_vals)
        LOGGER.info(
            f"Spline calibration fitted on {len(merged)} peptides. "
            f"Residual MAE: {np.abs(residuals).mean():.2f} Å²"
        )

    def transform(self, psm_df: pd.DataFrame) -> np.ndarray:
        """Transform predicted CCS values using the fitted spline."""
        if not self.is_fitted:
            raise CalibrationError("Calibration has not been fitted yet.")

        if "predicted_CCS_uncalibrated" not in psm_df.columns and "metadata" in psm_df.columns:
            psm_df["predicted_CCS_uncalibrated"] = psm_df["metadata"].apply(
                lambda x: (
                    x["predicted_CCS_uncalibrated"]
                    if "predicted_CCS_uncalibrated" in x
                    else np.nan
                )
            )

        pred_ccs = psm_df["predicted_CCS_uncalibrated"].values

        if self._spline is not None:
            calibrated = self._spline(pred_ccs)
        else:
            calibrated = pred_ccs + self._fallback_shift

        return calibrated.astype(np.float64)


def get_default_reference(multi: bool = False) -> pd.DataFrame:
    """
    Get the default reference DataFrame for calibration.

    Parameters
    ----------
    multi
        Whether to use the multi-charge reference dataset.

    Returns
    -------
    pd.DataFrame
        Default reference DataFrame with 'peptidoform' and 'CCS' columns.
    """
    reference_data_path = (
        DEFAULT_MULTI_REFERENCE_DATASET_PATH if multi else DEFAULT_REFERENCE_DATASET_PATH
    )
    LOGGER.info(f"Loading default reference dataset from {reference_data_path}")
    # dataset is in .gz format, so we need to extract it
    reference_df = pd.read_csv(reference_data_path, compression="gzip", keep_default_na=False)
    return reference_df
