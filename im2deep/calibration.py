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
from sklearn.linear_model import LinearRegression

from im2deep.constants import DEFAULT_MULTI_REFERENCE_DATASET_PATH, DEFAULT_REFERENCE_DATASET_PATH
from im2deep.exceptions import CalibrationError

LOGGER = logging.getLogger(__name__)

#: A charge with fewer overlapping peptides than this falls back to a single
#: model fit across every charge's overlap together, mirroring
#: `collate_dataset.py`'s `MIN_OVERLAP_PER_CHARGE` (same value, kept as a
#: separate constant since this module cannot import a top-level project
#: script as a dependency).
MIN_OVERLAP_PER_CHARGE = 10


def _peptide_key(pf) -> str:
    """Stripped-of-charge ProForma string, for merging target/source on identity."""
    if isinstance(pf, Peptidoform):
        return str(pf.proforma).rsplit("/", 1)[0]
    return str(pf).rsplit("/", 1)[0]


def _charge_of(pf) -> int:
    if isinstance(pf, Peptidoform):
        if pf.precursor_charge is None:
            raise CalibrationError(f"Peptidoform {pf} is missing precursor charge information.")
        return int(pf.precursor_charge)
    return int(str(pf).split("/")[-1])


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
        # Calibration-confidence bookkeeping, populated by fit(): how many
        # overlapping anchors were found for a charge, whether that charge's
        # shift fell back to the general/all-charge shift, and the std of
        # (target - source) residuals around the fitted shift. Read back via
        # stats_for_charge() - not used internally, exposed for rescoring
        # features that want to judge how much to trust a given PSM's
        # calibrated CCS.
        self.charge_n_anchors: dict[int, int] = {}
        self.charge_used_fallback: dict[int, bool] = {}
        self.charge_residual_std: dict[int, float] = {}

    @property
    def is_fitted(self) -> bool:
        return self.fitted

    def stats_for_charge(self, charge: int) -> dict:
        """Calibration-confidence bookkeeping for one charge state, post-fit."""
        charge = int(charge)
        return {
            "slope": 1.0,
            "intercept": self.charge_shifts.get(charge, self.general_shift or 0.0),
            "n_anchors": self.charge_n_anchors.get(charge, 0),
            "used_fallback": self.charge_used_fallback.get(charge, True),
            "residual_std": self.charge_residual_std.get(charge, float("nan")),
        }

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
                self.charge_n_anchors = dict.fromkeys(range(1, 7), 0)
                self.charge_residual_std = dict.fromkeys(range(1, 7), float("nan"))

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
                self.charge_n_anchors.setdefault(charge, 0)
                self.charge_residual_std.setdefault(charge, float("nan"))
            for charge in range(1, 7):
                self.charge_used_fallback[charge] = charge in no_shift_calculated
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
            # Global (non-per-charge) mode doesn't track anchors per charge -
            # not exercised by this project's pipeline (always per_charge=True),
            # but keep the bookkeeping dicts populated so stats_for_charge()
            # never KeyErrors if this path is ever used.
            self.charge_n_anchors = dict.fromkeys(range(1, 7), 0)
            self.charge_used_fallback = dict.fromkeys(range(1, 7), True)
            self.charge_residual_std = dict.fromkeys(range(1, 7), float("nan"))

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

    def _compute_ccs_shift_per_charge(
        self,
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

        # Calibration-confidence bookkeeping (idea 1): anchor count and
        # residual spread per charge, computed here since this is the one
        # place `merged`'s per-charge residuals already exist - discarded by
        # the rest of this method beyond the small-overlap warning below.
        residual_std_by_charge = merged.groupby("charge")["shift"].std()
        for charge, count in merged.groupby("charge").size().items():
            self.charge_n_anchors[int(cast(int, charge))] = int(cast(int, count))
        for charge, std in residual_std_by_charge.items():
            self.charge_residual_std[int(cast(int, charge))] = float(std)

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


class RegressionCCSCalibration(Calibration):
    """
    Per-charge slope + intercept calibration for CCS predictions.

    ``LinearCCSCalibration`` fits an additive-only shift, which is the right
    calibration for a model whose training labels sit on the reference's own
    scale, and it is what the bundled model is validated with. It can be the
    wrong one for a model whose training
    labels were harmonised with a regression-based collation method (e.g.
    ``collate_dataset.py``'s ``linreg``, per-charge slope + intercept): fitting
    only a shift against a reference on the model's own scale still discards
    the multiplicative component that method's own harmonisation applied. This
    class fits ``target = slope * source + intercept`` per charge instead,
    mirroring ``collate_dataset.py``'s ``RegressionCalibrator`` (see
    ROADMAP.md step 6b, "Only `shift` exists on both sides", for why the two
    need to differ rather than sharing one calibrator).

    A charge with fewer than ``MIN_OVERLAP_PER_CHARGE`` overlapping peptides
    falls back to a single model fit across every charge's overlap together,
    the same fallback ``RegressionCalibrator`` uses and for the same reason: a
    regression already carries a slope, so a transform fitted across all
    charges still scales with CCS rather than assuming a fixed offset.

    Parameters
    ----------
    per_charge : bool, optional
        Whether to fit a separate slope/intercept per charge state, falling
        back to one general model where a charge's own overlap is too thin.
        Default is True.
    """

    def __init__(self, per_charge: bool = True) -> None:
        super().__init__()
        self.per_charge = per_charge
        self.fitted = False
        self.charge_models: dict[int, LinearRegression] = {}
        self.general_model: LinearRegression | None = None
        # Calibration-confidence bookkeeping, populated by fit() - see
        # LinearCCSCalibration's identical attributes for the rationale.
        self.charge_n_anchors: dict[int, int] = {}
        self.charge_used_fallback: dict[int, bool] = {}
        self.charge_residual_std: dict[int, float] = {}

    @property
    def is_fitted(self) -> bool:
        return self.fitted

    def stats_for_charge(self, charge: int) -> dict:
        """Calibration-confidence bookkeeping for one charge state, post-fit."""
        charge = int(charge)
        model = self.charge_models.get(charge, self.general_model)
        slope = float(model.coef_[0]) if model is not None else 1.0
        intercept = float(model.intercept_) if model is not None else 0.0
        return {
            "slope": slope,
            "intercept": intercept,
            "n_anchors": self.charge_n_anchors.get(charge, 0),
            "used_fallback": self.charge_used_fallback.get(charge, True),
            "residual_std": self.charge_residual_std.get(charge, float("nan")),
        }

    def fit(
        self,
        psm_df_target: pd.DataFrame,
        psm_df_source: pd.DataFrame | None = None,
        multi: bool = False,
    ) -> None:
        """
        Fit ``target = slope * source + intercept`` per charge.

        ``psm_df_target`` is this run's own confidently-identified,
        observed CCS (real scale); ``psm_df_source`` is the reference dataset,
        which for this calibrator must be on the same scale as the model's raw
        predictions (e.g. a reference built from the model's own training
        pool), so the fitted mapping can be applied directly to
        ``predicted_CCS_uncalibrated`` in `transform`.
        """
        if psm_df_source is None:
            LOGGER.debug("No reference PSMList provided, loading default reference dataset.")
            psm_df_source = get_default_reference(multi=multi)

        merged = self._merge_on_peptide_and_charge(psm_df_target, psm_df_source)

        # Calibration-confidence bookkeeping (idea 1): every charge starts as
        # "no anchors seen at all", overwritten below for charges that
        # actually appear in `merged`.
        self.charge_n_anchors = dict.fromkeys(range(1, 7), 0)
        self.charge_used_fallback = dict.fromkeys(range(1, 7), True)
        self.charge_residual_std = dict.fromkeys(range(1, 7), float("nan"))
        thin_charge_groups: dict[int, pd.DataFrame] = {}

        if self.per_charge:
            for charge, group in merged.groupby("charge"):
                charge_int = int(cast(int, charge))
                self.charge_n_anchors[charge_int] = len(group)
                if len(group) < MIN_OVERLAP_PER_CHARGE:
                    LOGGER.debug(
                        f"Only {len(group)} overlapping peptides for charge {charge}; "
                        "deferring to the general fallback model."
                    )
                    thin_charge_groups[charge_int] = group
                    continue
                model = LinearRegression()
                model.fit(group[["CCS_source"]].to_numpy(), group["CCS_target"].to_numpy())
                self.charge_models[charge_int] = model
                self.charge_used_fallback[charge_int] = False
                residuals = group["CCS_target"].to_numpy() - model.predict(
                    group[["CCS_source"]].to_numpy()
                )
                self.charge_residual_std[charge_int] = float(np.std(residuals))

        if len(merged) >= MIN_OVERLAP_PER_CHARGE:
            self.general_model = LinearRegression()
            self.general_model.fit(
                merged[["CCS_source"]].to_numpy(), merged["CCS_target"].to_numpy()
            )
        elif not self.charge_models:
            # No overlap at all is expected, not just possible, on a genuinely
            # unseen dataset (e.g. a rare-organism downstream check against a
            # reference sampled from an unrelated training pool) - exactly the
            # case this calibrator most needs to keep working for. Fall back
            # to an identity transform (slope=1, intercept=0) with a WARNING,
            # mirroring LinearCCSCalibration's own "fall back to 0.0 shift
            # rather than crash" behaviour under the same starvation.
            LOGGER.warning(
                f"Only {len(merged)} overlapping peptides in total; falling back to an "
                "identity transform (no calibration applied) rather than fitting a "
                f"general regression calibration (needs >= {MIN_OVERLAP_PER_CHARGE})."
            )
            self.general_model = LinearRegression()
            self.general_model.coef_ = np.array([1.0])
            self.general_model.intercept_ = 0.0

        # Now that general_model exists, compute residual std for charges
        # that fell back to it (thin per-charge overlap, still non-zero).
        if self.general_model is not None:
            for charge_int, group in thin_charge_groups.items():
                residuals = group["CCS_target"].to_numpy() - self.general_model.predict(
                    group[["CCS_source"]].to_numpy()
                )
                self.charge_residual_std[charge_int] = float(np.std(residuals))

        self.fitted = True

    def transform(self, psm_df: pd.DataFrame) -> np.ndarray:
        """Apply the fitted per-charge (or general fallback) regression."""
        if not self.is_fitted:
            raise CalibrationError("Calibration has not been fitted yet.")

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

        charges = psm_df["peptidoform"].apply(_charge_of).to_numpy()
        source = psm_df["predicted_CCS_uncalibrated"].to_numpy(dtype=float).reshape(-1, 1)
        out = np.empty(len(psm_df), dtype=object)

        for charge in np.unique(charges):
            mask = charges == charge
            model = self.charge_models.get(int(charge), self.general_model)
            if model is None:
                # Only reached if fit() saw less overlap than
                # MIN_OVERLAP_PER_CHARGE in total, which fit() itself refuses
                # (raises CalibrationError) - a guard, not a real code path.
                out[mask] = source[mask, 0]
            else:
                out[mask] = model.predict(source[mask]).astype(np.float32)

        return out

    @staticmethod
    def _merge_on_peptide_and_charge(
        target_df: pd.DataFrame, source_df: pd.DataFrame
    ) -> pd.DataFrame:
        target_work = target_df.copy()
        source_work = source_df.copy()

        target_work["peptide_key"] = target_work["peptidoform"].apply(_peptide_key)
        target_work["charge"] = target_work["peptidoform"].apply(_charge_of)
        if "CCS" not in target_work.columns and "metadata" in target_work.columns:
            target_work["CCS"] = target_work["metadata"].apply(
                lambda x: float(x["CCS"]) if isinstance(x, dict) and "CCS" in x else np.nan
            )

        source_work["peptide_key"] = source_work["peptidoform"].apply(_peptide_key)
        source_work["charge"] = source_work["peptidoform"].apply(_charge_of)
        if "CCS" not in source_work.columns and "metadata" in source_work.columns:
            source_work["CCS"] = source_work["metadata"].apply(
                lambda x: float(x["CCS"]) if isinstance(x, dict) and "CCS" in x else np.nan
            )

        merged = pd.merge(
            target_work[["peptide_key", "charge", "CCS"]],
            source_work[["peptide_key", "charge", "CCS"]],
            on=["peptide_key", "charge"],
            suffixes=("_target", "_source"),
        )
        LOGGER.debug(f"Number of overlapping peptides for regression calibration: {len(merged)}")
        return merged


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
