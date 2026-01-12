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
        peptidoforms_target: np.ndarray,
        observed_ccs_target: np.ndarray,
        peptidoforms_source: np.ndarray | None = None,
        observed_ccs_source: np.ndarray | None = None,
        multi: bool = False,
    ) -> None:
        """Fit the calibration using target and source CCS values."""
        if peptidoforms_source is None and observed_ccs_source is None:
            if self.reference_psm_list is None:
                LOGGER.debug("No reference PSMList provided, loading default reference dataset.")
                peptidoforms_source, observed_ccs_source = self.get_default_reference(multi=multi)
            else:
                peptidoforms_source = np.array(
                    [psm.peptidoform for psm in self.reference_psm_list]
                )
                observed_ccs_source = np.array(
                    [psm.metadata["CCS"] for psm in self.reference_psm_list],
                    dtype=np.float32,
                )
        # if only one of peptidoforms_target or observed_ccs_target is None, raise error
        elif peptidoforms_source is None or observed_ccs_source is None:
            raise CalibrationError(
                "Both peptidoforms_source and observed_ccs_source must be provided together."
            )
        LOGGER.debug("Calculating calibration parameters...")

        if self.per_charge:
            # For per-charge calibration, calculate shifts for all charges
            LOGGER.debug("Calculating shift factors per charge state...")
            try:
                self.charge_shifts = self.calculate_ccs_shift(
                    peptidoforms_target,
                    observed_ccs_target,
                    peptidoforms_source,
                    observed_ccs_source,
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
                    peptidoforms_target,
                    observed_ccs_target,
                    peptidoforms_source,
                    observed_ccs_source,
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
                    peptidoforms_target,
                    observed_ccs_target,
                    peptidoforms_source,
                    observed_ccs_source,
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
        peptidoforms: np.ndarray,
        predicted_ccs: np.ndarray,
    ) -> np.ndarray:
        """Transform source CCS into the calibrated target space."""
        if not self.is_fitted:
            raise CalibrationError("Calibration has not been fitted yet.")

        # Vectorized approach for speed
        # Extract charges for all peptidoforms at once
        charges = []
        for pf in peptidoforms:
            if isinstance(pf, Peptidoform):
                charges.append(pf.precursor_charge)
            else:
                # Parse from string representation
                pf_str = str(pf)
                charges.append(int(pf_str.split("/")[-1]))
        charges = np.array(charges, dtype=np.int32)

        # Vectorized shift application
        shifts = np.array(
            [self.charge_shifts.get(c, self.general_shift) for c in charges], dtype=np.float32
        )
        
        # Handle both single-output and multi-output predictions
        if predicted_ccs.ndim == 2:
            # Multi-output: reshape shifts to broadcast correctly
            shifts = shifts.reshape(-1, 1)
        
        predicted_ccs_calibrated = predicted_ccs + shifts

        return predicted_ccs_calibrated

    def calculate_ccs_shift(
        self,
        target_peptidoforms: np.ndarray,
        target_ccs: np.ndarray,
        source_peptidoforms: np.ndarray,
        source_ccs: np.ndarray,
    ) -> dict[int, float] | float:
        """
        Calculate CCS shift factors between target and source PSMLists.

        Parameters
        ----------
        target_peptidoforms
            Peptidoforms from the target PSMList.
        target_ccs
            Observed CCS values from the target PSMList.
        source_peptidoforms
            Peptidoforms from the source PSMList.
        source_ccs
            Predicted CCS values from the source PSMList.

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
                target_peptidoforms,
                target_ccs,
                source_peptidoforms,
                source_ccs,
                self.use_charge_state,
            )
            LOGGER.debug(f"Global CCS shift factor: {shift_factor:.3f}")
            return shift_factor
        else:
            # Per-charge calibration
            shift_factor_dict = self._compute_ccs_shift_per_charge(
                target_peptidoforms,
                target_ccs,
                source_peptidoforms,
                source_ccs,
            )

            return shift_factor_dict

    def get_default_reference(self, multi: bool = False) -> PSMList:
        """
        Get the default reference PSMList for calibration.

        Parameters
        ----------
        multi
            Whether to use the multi-charge reference dataset.

        Returns
        -------
        PSMList
            Default reference PSMList.
        """
        reference_data_path = (
            DEFAULT_MULTI_REFERENCE_DATASET_PATH if multi else DEFAULT_REFERENCE_DATASET_PATH
        )
        LOGGER.info(f"Loading default reference dataset from {reference_data_path}")
        # dataset is in .gz format, so we need to extract it
        reference_dataset = pd.read_csv(
            reference_data_path, compression="gzip", keep_default_na=False
        )
        reference_peptidoforms = reference_dataset["peptidoform"].tolist()
        reference_ccs = reference_dataset["CCS"].astype(np.float32).to_numpy()
        return reference_peptidoforms, reference_ccs

    @staticmethod
    def _compute_ccs_shift(
        target_peptidoforms,
        target_ccs,
        source_peptidoforms,
        source_ccs,
        charge_state: int,
    ) -> float:
        """Compute CCS shift for a specific charge state."""
        # Extract charges vectorized
        source_charges = []
        source_keys = []
        for pf in source_peptidoforms:
            if isinstance(pf, Peptidoform):
                source_charges.append(pf.precursor_charge)
                source_keys.append(pf.proforma)
            else:
                pf_str = str(pf)
                source_charges.append(int(pf_str.split("/")[-1]))
                source_keys.append(pf_str)
        
        source_charges = np.array(source_charges, dtype=np.int32)
        source_keys = np.array(source_keys, dtype=object)

        target_charges = []
        target_keys = []
        for pf in target_peptidoforms:
            if isinstance(pf, Peptidoform):
                target_charges.append(pf.precursor_charge)
                target_keys.append(pf.proforma)
            else:
                pf_str = str(pf)
                target_charges.append(int(pf_str.split("/")[-1]))
                target_keys.append(pf_str)
        
        target_charges = np.array(target_charges, dtype=np.int32)
        target_keys = np.array(target_keys, dtype=object)

        # Filter by charge state using boolean indexing (much faster)
        source_mask = source_charges == charge_state
        target_mask = target_charges == charge_state

        source_keys_filtered = source_keys[source_mask]
        source_ccs_filtered = source_ccs[source_mask].astype(np.float64)
        target_keys_filtered = target_keys[target_mask]
        target_ccs_filtered = target_ccs[target_mask].astype(np.float64)

        # Find overlapping peptides using set operations
        source_set = set(source_keys_filtered)
        target_set = set(target_keys_filtered)
        overlapping_peptides = source_set.intersection(target_set)

        LOGGER.debug(
            f"Calculating CCS shift based on {len(overlapping_peptides)} overlapping peptides for charge state {charge_state}."
        )

        if len(overlapping_peptides) == 0:
            LOGGER.warning(f"No overlapping peptides found for charge state {charge_state}.")
            return 0.0

        if len(overlapping_peptides) < 10:
            LOGGER.warning(
                f"Only {len(overlapping_peptides)} overlapping peptides found for charge state {charge_state}. "
                "Shift calculation may be unreliable."
            )

        # Build lookup dictionaries for overlapping peptides only
        source_dict = {
            key: ccs
            for key, ccs in zip(source_keys_filtered, source_ccs_filtered)
            if key in overlapping_peptides
        }
        target_dict = {
            key: ccs
            for key, ccs in zip(target_keys_filtered, target_ccs_filtered)
            if key in overlapping_peptides
        }

        # Extract CCS values in the same order
        source_ccs_array = np.array(
            [source_dict[pep] for pep in overlapping_peptides], dtype=np.float64
        )
        target_ccs_array = np.array(
            [target_dict[pep] for pep in overlapping_peptides], dtype=np.float64
        )

        shift = np.mean(target_ccs_array - source_ccs_array)

        if abs(shift) > 100.0:
            LOGGER.warning(
                f"Unusually large CCS shift ({shift:.2f}) detected for charge state {charge_state}."
                " Please verify the calibration datasets."
            )
        return float(shift)

    @staticmethod
    def _compute_ccs_shift_per_charge(
        target_peptidoforms, target_ccs, source_peptidoforms, source_ccs
    ) -> dict[int, float]:
        """
        Calculate CCS shift factors per charge state.

        Parameters
        ----------
        target_peptidoforms
            Peptidoforms from the target PSMList.
        target_ccs
            Observed CCS values from the target PSMList.
        source_peptidoforms
            Peptidoforms from the source PSMList.
        source_ccs
            Predicted CCS values from the source PSMList.

        Returns
        -------
        dict[int, float]
            Shift factors per charge state.

        Raises
        ------
        CalibrationError
            If no overlapping peptides are found for any charge state.
        """
        source_charges = []
        for pf in source_peptidoforms:
            if isinstance(pf, Peptidoform):
                source_charges.append(int(pf.precursor_charge))
            else:
                pf_str = str(pf)
                source_charges.append(int(pf_str.split("/")[-1]))
        
        source_charges = np.array(source_charges, dtype=np.int32)
        shift_factors = {}
        charges_in_source = set(source_charges)

        for charge in charges_in_source:
            shift = LinearCCSCalibration._compute_ccs_shift(
                target_peptidoforms, target_ccs, source_peptidoforms, source_ccs, charge
            )
            if shift == np.nan:
                LOGGER.warning(f"No valid CCS shift calculated for charge state {charge}.")
            shift_factors[charge] = shift

        if len(shift_factors) == 0:
            raise CalibrationError("No CCS shift factors could be calculated.")

        return shift_factors
