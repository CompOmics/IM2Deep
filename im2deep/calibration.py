"""
CCS calibration utilities.

This module provides calibration strategies to map predicted CCS values to the aligned target scale.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import cast

import numpy as np
from deeplc.calibration import Calibration
from psm_utils import PSMList

from im2deep._exceptions import CalibrationError

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


class IdentityCalibration(Calibration):
    """No calibration; returns inputs unchanged."""

    @property
    def is_fitted(self) -> bool:
        return True

    def fit(self, target: np.ndarray, source: np.ndarray) -> None:  # noqa: ARG002
        return None

    def transform(self, source: PSMList) -> PSMList:
        return source


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

    @property
    def is_fitted(self) -> bool:
        return self.fitted

    def fit(
        self,
        target: PSMList,
        source: PSMList,
    ) -> None:
        """Fit the calibration using target and source CCS values."""

        LOGGER.debug("Calculating calibration parameters...")
        try:
            self.general_shift = self.calculate_ccs_shift(
                target, source, per_charge=False, use_charge_state=self.use_charge_state or 2
            )
        except CalibrationError as e:
            LOGGER.warning(
                f"Could not calculate general shift factor: {e}. Using 0.0 as fallback."
            )
            self.general_shift = 0.0

        if self.per_charge:
            LOGGER.debug("Calculating shift factors per charge state.")

            LOGGER.debug("Calculating shift factors per charge state...")
            self.charge_shifts = self.calculate_ccs_shift(
                target, source, per_charge=self.per_charge
            )
            LOGGER.debug(f"Calculated charge-specific shifts: {self.charge_shifts}")
        self.used_charges = set(self.charge_shifts.keys())
        self.fitted = True

    def transform(
        self,
        source: PSMList,
    ) -> PSMList:
        """Transform source CCS into the calibrated target space."""
        if not self.is_fitted:
            raise CalibrationError("Calibration has not been fitted yet.")

        calibrated_source = source.copy()
        for idx, psm in enumerate(calibrated_source):
            charge = psm.peptidoform.precursor_charge
            ccs = psm.metadata["CCS"]
            if self.per_charge and charge in self.charge_shifts:
                shift = self.charge_shifts[charge]
            else:
                LOGGER.warning(
                    f"Charge state {charge} not found in calibration shifts; "
                    f"using general shift factor: {self.general_shift}."
                )
                shift = cast(float, self.general_shift)
            calibrated_source[idx].metadata["CCS"] = ccs + shift
        return calibrated_source

    @staticmethod
    def calculate_ccs_shift(
        target: PSMList,
        source: PSMList,
        per_charge: bool = True,
        use_charge_state: int | None = None,
    ) -> dict[int, float] | float:
        """Calculate CCS shift factors between target and source PSMLists."""
        # TODO: Implement actual shift calculation logic
        raise NotImplementedError("CCS shift calculation not implemented yet.")
