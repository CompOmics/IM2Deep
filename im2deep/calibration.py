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
        peptidoforms_source: np.ndarray,
        observed_ccs_source: np.ndarray,
        peptidoforms_target: np.ndarray | None = None,
        observed_ccs_target: np.ndarray | None = None,
        multi: bool = False,
    ) -> None:
        """Fit the calibration using target and source CCS values."""
        if peptidoforms_target is None and observed_ccs_target is None:
            if self.reference_psm_list is None:
                LOGGER.debug("No reference PSMList provided, loading default reference dataset.")
                peptidoforms_target, observed_ccs_target = self.get_default_reference(multi=multi)
            else:
                peptidoforms_target = np.array(
                    [psm.peptidoform for psm in self.reference_psm_list]
                )
                observed_ccs_target = np.array(
                    [psm.metadata["CCS"] for psm in self.reference_psm_list],
                    dtype=np.float32,
                )
        # if only one of peptidoforms_target or observed_ccs_target is None, raise error
        elif peptidoforms_target is None or observed_ccs_target is None:
            raise CalibrationError(
                "Both peptidoforms_target and observed_ccs_target must be provided together."
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
                    c for c in self.charge_shifts.keys() if self.charge_shifts[c] is not None
                ]
                if available_charges:
                    self.general_shift = self.charge_shifts[available_charges[0]]
                else:
                    self.general_shift = 0.0
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

    def transform(
        self,
        peptidoforms: np.ndarray,
        predicted_ccs: np.ndarray,
    ) -> np.ndarray:
        """Transform source CCS into the calibrated target space."""
        if not self.is_fitted:
            raise CalibrationError("Calibration has not been fitted yet.")

        predicted_ccs_calibrated = []
        for pf, ccs in zip(peptidoforms, predicted_ccs):
            if isinstance(pf, Peptidoform):
                charge = pf.precursor_charge
            else:
                charge = str(pf).split("/")[-1]
            charge = int(charge)
            if self.per_charge and charge in self.charge_shifts:
                shift = self.charge_shifts[charge]
            else:
                LOGGER.warning(
                    f"Charge state {charge} not found in calibration shifts; "
                    f"using general shift factor: {self.general_shift}."
                )
                shift = cast(float, self.general_shift)
            predicted_ccs_calibrated.append(ccs + shift)
        return np.array(predicted_ccs_calibrated)

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
            # For any missing charge states, use 0.0 as fallback
            for charge in range(1, 7):
                if charge not in shift_factor_dict or shift_factor_dict[charge] is None:
                    # Try to use the shift from charge 2 as a reasonable fallback
                    fallback_shift = shift_factor_dict.get(2, 0.0)
                    if fallback_shift is None:
                        fallback_shift = 0.0
                    LOGGER.debug(
                        f"No shift factor calculated for charge state {charge}. "
                        f"Using fallback shift: {fallback_shift:.3f}."
                    )
                    shift_factor_dict[charge] = float(fallback_shift)
            LOGGER.debug(f"CCS shift factors per charge: {shift_factor_dict}")
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
        # Build dictionaries only for peptides with the specified charge state
        source_dict = {}
        for i in range(len(source_peptidoforms)):
            pf = source_peptidoforms[i]
            # Extract charge from peptidoform (format: SEQUENCE/CHARGE)
            charge = int(pf.precursor_charge)
            if charge == charge_state:
                source_dict[pf.proforma] = float(source_ccs[i])

        target_dict = {}
        for i in range(len(target_peptidoforms)):
            pf = target_peptidoforms[i]
            # Extract charge from peptidoform (format: SEQUENCE/CHARGE)
            if isinstance(pf, Peptidoform):
                charge = int(pf.precursor_charge)
            else:
                charge = int(str(pf).split("/")[-1])
            if charge == charge_state:
                if isinstance(pf, Peptidoform):
                    target_dict[pf.proforma] = float(target_ccs[i])
                else:
                    target_dict[str(pf)] = float(target_ccs[i])

        overlapping_peptides = set(source_dict.keys()).intersection(set(target_dict.keys()))

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

        source_ccs_list = []
        target_ccs_list = []
        for peptide in overlapping_peptides:
            source_ccs_list.append(source_dict[peptide])
            target_ccs_list.append(target_dict[peptide])

        source_ccs_array = np.array(source_ccs_list, dtype=np.float64)
        target_ccs_array = np.array(target_ccs_list, dtype=np.float64)

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
        source_charges = np.array(
            [int(peptidoform.precursor_charge) for peptidoform in source_peptidoforms]
        )
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
