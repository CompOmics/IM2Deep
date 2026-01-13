"""
CCS calibration utilities.

This module provides calibration strategies to map predicted CCS values to the aligned target scale.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import cast

import pandas as pd
import numpy as np
from psm_utils import PSMList, Peptidoform

from im2deep._exceptions import CalibrationError
from im2deep.utils import parse_input
from im2deep.constants import DEFAULT_REFERENCE_DATASET_PATH, DEFAULT_MULTI_REFERENCE_DATASET_PATH

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
        target: PSMList,
        source: PSMList,
    ) -> None:
        """Fit the calibration using target and source CCS values."""
        ...

    @abstractmethod
    def transform(
        self,
        source: PSMList,
    ) -> PSMList:
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
                self.charge_shifts = self.calculate_ccs_shift(
                    psm_df_target,
                    psm_df_source,
                )
                LOGGER.debug(f"Calculated charge-specific shifts: {self.charge_shifts}")
            except CalibrationError as e:
                LOGGER.warning(
                    f"Could not calculate charge-specific shift factors: {e}. Using 0.0 as fallback."
                )
                self.charge_shifts = {charge: 0.0 for charge in range(1, 7)}

            # Also calculate a general shift for reference (using charge 2 as default)
            try:
                self.general_shift = self._compute_ccs_shift(
                    psm_df_target,
                    psm_df_source,
                    2,
                )
            except Exception:
                # If charge 2 fails, try to get any available charge
                available_charges = [
                    c
                    for c in self.charge_shifts.keys()
                    if self.charge_shifts[c] is not None and self.charge_shifts[c] != 0.0
                ]
                if available_charges:
                    self.general_shift = self.charge_shifts[available_charges[0]]
                else:
                    self.general_shift = 0.0

            # Fill in missing charge states with general shift
            for charge in range(1, 7):
                if (
                    charge not in self.charge_shifts
                    or self.charge_shifts[charge] is None
                    or self.charge_shifts[charge] == 0.0
                ):
                    LOGGER.debug(
                        f"No shift factor calculated for charge state {charge}. "
                        f"Using general shift: {self.general_shift:.3f}."
                    )
                    self.charge_shifts[charge] = float(self.general_shift)
        else:
            # For global calibration, calculate a single shift
            try:
                self.general_shift = self.calculate_ccs_shift(
                    psm_df_target,
                    psm_df_source,
                )
            except CalibrationError as e:
                LOGGER.warning(
                    f"Could not calculate general shift factor: {e}. Using 0.0 as fallback."
                )
                self.general_shift = 0.0
            self.charge_shifts = {charge: self.general_shift for charge in range(1, 7)}

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

        psm_df["predicted_CCS_uncalibrated"] = psm_df["metadata"].apply(
            lambda x: (
                x["predicted_CCS_uncalibrated"] if "predicted_CCS_uncalibrated" in x else np.nan
            )
        )

        # Extract charge from peptidoform column efficiently
        psm_df["charge"] = psm_df["peptidoform"].apply(
            lambda x: int(str(x).split("/")[-1]) if isinstance(x, str) else x.precursor_charge
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
            lambda row: apply_shift(row["predicted_CCS_uncalibrated"], row["shift"]), 
            axis=1
        )

        # Return as numpy object array to preserve multiconformer arrays
        predicted_ccs_calibrated = np.empty(len(psm_df), dtype=object)
        predicted_ccs_calibrated[:] = psm_df["calibrated_CCS"].tolist()

        return predicted_ccs_calibrated

    def calculate_ccs_shift(
        self,
        target_df: pd.DataFrame,
        source_df: pd.DataFrame,
    ) -> dict[int, float] | float:
        """
        Calculate CCS shift factors between target and source PSMLists.

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

        if not self.per_charge:
            # Global calibration using specified charge state
            if self.use_charge_state is None:
                self.use_charge_state = 2  # Default charge state
                LOGGER.debug(
                    "No charge state specified for global calibration. Using default charge state 2 for global shift calculation."
                )

            shift_factor = self._compute_ccs_shift(
                target_df,
                source_df,
                self.use_charge_state,
            )
            LOGGER.debug(f"Global CCS shift factor: {shift_factor:.3f}")
            return shift_factor
        else:
            # Per-charge calibration
            shift_factor_dict = self._compute_ccs_shift_per_charge(
                target_df,
                source_df,
            )

            return shift_factor_dict

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
                # For Peptidoform objects, use proforma property which excludes charge
                return pf.proforma
            else:
                # For strings in format "PEPTIDE/charge", split off charge
                return str(pf).rsplit("/", 1)[0]

        def get_charge(pf):
            if isinstance(pf, Peptidoform):
                return pf.precursor_charge
            else:
                return int(str(pf).split("/")[-1])

        target_work["peptide_key"] = target_work["peptidoform"].apply(get_peptide_key)
        target_work["charge"] = target_work["peptidoform"].apply(get_charge)

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
                return pf.precursor_charge
            else:
                return int(str(pf).split("/")[-1])

        target_work["peptide_key"] = target_work["peptidoform"].apply(get_peptide_key)
        target_work["charge"] = target_work["peptidoform"].apply(get_charge)
        target_work["CCS"] = target_work["metadata"].apply(
            lambda x: x["CCS"] if "CCS" in x else np.nan
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
        shift_factors = merged.groupby("charge")["shift"].mean().to_dict()

        # Log information for each charge state
        charge_counts = merged.groupby("charge").size()
        for charge, count in charge_counts.items():
            LOGGER.debug(
                f"Calculated shift for charge {charge} based on {count} overlapping peptides: "
                f"{shift_factors[charge]:.3f}"
            )
            if count < 10:
                LOGGER.warning(
                    f"Only {count} overlapping peptides found for charge state {charge}. "
                    "Shift calculation may be unreliable."
                )
            if abs(shift_factors[charge]) > 100.0:
                LOGGER.warning(
                    f"Unusually large CCS shift ({shift_factors[charge]:.2f}) detected for charge state {charge}."
                    " Please verify the calibration datasets."
                )

        if len(shift_factors) == 0:
            raise CalibrationError("No CCS shift factors could be calculated.")

        return shift_factors


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
