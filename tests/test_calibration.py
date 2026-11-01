"""Tests for calibration module."""

import numpy as np
import pandas as pd
import pytest
from psm_utils import Peptidoform

from im2deep.calibration import (
    MIN_OVERLAP_PER_CHARGE,
    LinearCCSCalibration,
    RegressionCCSCalibration,
    get_default_reference,
)
from im2deep.exceptions import CalibrationError


class TestLinearCCSCalibration:
    """Tests for LinearCCSCalibration class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        calibration = LinearCCSCalibration()
        assert calibration.per_charge is True
        assert calibration.use_charge_state is None
        assert calibration.is_fitted is False
        assert calibration.charge_shifts == {}
        assert calibration.general_shift is None

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        calibration = LinearCCSCalibration(per_charge=False, use_charge_state=3)
        assert calibration.per_charge is False
        assert calibration.use_charge_state == 3
        assert calibration.is_fitted is False

    def test_fit_per_charge(self, sample_peptidoforms, sample_ccs_values, sample_predicted_ccs):
        """Test fitting with per-charge calibration."""
        calibration = LinearCCSCalibration(per_charge=True)

        # Create DataFrames for target and source
        target_df = pd.DataFrame(
            {
                "peptidoform": sample_peptidoforms,
                "metadata": [{"CCS": ccs} for ccs in sample_ccs_values],
            }
        )

        source_df = pd.DataFrame({"peptidoform": sample_peptidoforms, "CCS": sample_predicted_ccs})

        calibration.fit(
            psm_df_target=target_df,
            psm_df_source=source_df,
        )

        assert calibration.is_fitted is True
        assert len(calibration.charge_shifts) > 0
        assert calibration.general_shift is not None

    def test_fit_global(self, sample_peptidoforms, sample_ccs_values, sample_predicted_ccs):
        """Test fitting with global calibration."""
        calibration = LinearCCSCalibration(per_charge=False, use_charge_state=2)

        target_df = pd.DataFrame(
            {
                "peptidoform": sample_peptidoforms,
                "metadata": [{"CCS": ccs} for ccs in sample_ccs_values],
            }
        )

        source_df = pd.DataFrame({"peptidoform": sample_peptidoforms, "CCS": sample_predicted_ccs})

        calibration.fit(
            psm_df_target=target_df,
            psm_df_source=source_df,
        )

        assert calibration.is_fitted is True
        assert calibration.general_shift is not None
        assert isinstance(calibration.general_shift, float)

    def test_transform_single_output(
        self, sample_peptidoforms, sample_ccs_values, sample_predicted_ccs
    ):
        """Test transforming single-output predictions."""
        calibration = LinearCCSCalibration(per_charge=True)

        target_df = pd.DataFrame(
            {
                "peptidoform": sample_peptidoforms,
                "metadata": [{"CCS": ccs} for ccs in sample_ccs_values],
            }
        )

        source_df = pd.DataFrame({"peptidoform": sample_peptidoforms, "CCS": sample_predicted_ccs})

        calibration.fit(
            psm_df_target=target_df,
            psm_df_source=source_df,
        )

        # Transform with predictions in metadata
        transform_df = pd.DataFrame(
            {
                "peptidoform": sample_peptidoforms,
                "metadata": [
                    {"predicted_CCS_uncalibrated": pred} for pred in sample_predicted_ccs
                ],
            }
        )

        calibrated = calibration.transform(transform_df)

        assert len(calibrated) == len(sample_predicted_ccs)
        assert isinstance(calibrated, np.ndarray)

    def test_transform_multi_output(
        self, sample_peptidoforms, sample_ccs_values, sample_predicted_ccs_multi
    ):
        """Test transforming multi-output predictions."""
        calibration = LinearCCSCalibration(per_charge=True)

        target_df = pd.DataFrame(
            {
                "peptidoform": sample_peptidoforms,
                "metadata": [{"CCS": ccs} for ccs in sample_ccs_values],
            }
        )

        source_df = pd.DataFrame(
            {"peptidoform": sample_peptidoforms, "CCS": sample_ccs_values - 2.0}  # Simulate shift
        )

        calibration.fit(
            psm_df_target=target_df,
            psm_df_source=source_df,
        )

        # Transform multi-output with arrays in metadata
        transform_df = pd.DataFrame(
            {
                "peptidoform": sample_peptidoforms,
                "metadata": [
                    {"predicted_CCS_uncalibrated": pred} for pred in sample_predicted_ccs_multi
                ],
            }
        )

        calibrated = calibration.transform(transform_df)

        assert len(calibrated) == len(sample_predicted_ccs_multi)
        assert isinstance(calibrated, np.ndarray)
        # Check that arrays are preserved for multiconformer
        assert isinstance(calibrated[0], np.ndarray)
        assert len(calibrated[0]) == 2  # Two conformers

    def test_transform_not_fitted(self, sample_peptidoforms, sample_predicted_ccs):
        """Test transform raises error when not fitted."""
        calibration = LinearCCSCalibration()

        transform_df = pd.DataFrame(
            {
                "peptidoform": sample_peptidoforms,
                "metadata": [
                    {"predicted_CCS_uncalibrated": pred} for pred in sample_predicted_ccs
                ],
            }
        )

        with pytest.raises(CalibrationError, match="not been fitted"):
            calibration.transform(transform_df)

    def test_calculate_ccs_shift_no_overlap(self):
        """Test shift calculation with no overlapping peptides."""
        calibration = LinearCCSCalibration(per_charge=False, use_charge_state=2)

        target_df = pd.DataFrame(
            {"peptidoform": [Peptidoform("PEPTIDE/2")], "metadata": [{"CCS": 450.0}]}
        )

        source_df = pd.DataFrame({"peptidoform": [Peptidoform("DIFFERENT/2")], "CCS": [460.0]})

        shift = calibration.calculate_ccs_shift_global(target_df, source_df)

        assert shift == 0.0  # No overlap returns 0.0

    def test_calculate_ccs_shift_with_overlap(self):
        """Test shift calculation with overlapping peptides."""
        calibration = LinearCCSCalibration(per_charge=False, use_charge_state=2)

        peptidoforms = [Peptidoform("PEPTIDE/2"), Peptidoform("SEQUENCE/2")]

        target_df = pd.DataFrame(
            {"peptidoform": peptidoforms, "metadata": [{"CCS": 450.0}, {"CCS": 520.0}]}
        )

        source_df = pd.DataFrame({"peptidoform": peptidoforms, "CCS": [445.0, 515.0]})

        shift = calibration.calculate_ccs_shift_global(target_df, source_df)

        assert isinstance(shift, float)
        assert abs(shift - 5.0) < 0.1  # Should be approximately 5.0

    def test_compute_ccs_shift_per_charge(self):
        """Test per-charge shift computation."""
        peptidoforms = [
            Peptidoform("PEPTIDE/2"),
            Peptidoform("SEQUENCE/3"),
            Peptidoform("TEST/2"),
        ]

        target_df = pd.DataFrame(
            {
                "peptidoform": peptidoforms,
                "metadata": [{"CCS": 450.0}, {"CCS": 520.0}, {"CCS": 480.0}],
            }
        )

        source_df = pd.DataFrame({"peptidoform": peptidoforms, "CCS": [445.0, 515.0, 475.0]})

        calibration = LinearCCSCalibration()
        shifts = calibration._compute_ccs_shift_per_charge(target_df, source_df)

        assert isinstance(shifts, dict)
        assert 2 in shifts
        assert 3 in shifts
        assert abs(shifts[2] - 5.0) < 0.1
        assert abs(shifts[3] - 5.0) < 0.1

    def test_fit_with_missing_charges(self, sample_peptidoforms, sample_ccs_values):
        """Test that missing charges are filled with general shift."""
        calibration = LinearCCSCalibration(per_charge=True)

        target_df = pd.DataFrame(
            {
                "peptidoform": sample_peptidoforms,
                "metadata": [{"CCS": ccs} for ccs in sample_ccs_values],
            }
        )

        source_df = pd.DataFrame(
            {"peptidoform": sample_peptidoforms, "CCS": sample_ccs_values - 5.0}
        )

        calibration.fit(
            psm_df_target=target_df,
            psm_df_source=source_df,
        )

        # Check that charges 1-6 are all filled
        for charge in range(1, 7):
            assert charge in calibration.charge_shifts
            assert isinstance(calibration.charge_shifts[charge], float)

    def test_fit_invalid_charge_state(self, sample_peptidoforms, sample_ccs_values):
        """Test that invalid charge state raises error."""
        calibration = LinearCCSCalibration(per_charge=False, use_charge_state=10)

        target_df = pd.DataFrame(
            {
                "peptidoform": sample_peptidoforms,
                "metadata": [{"CCS": ccs} for ccs in sample_ccs_values],
            }
        )

        source_df = pd.DataFrame({"peptidoform": sample_peptidoforms, "CCS": sample_ccs_values})

        with pytest.raises(CalibrationError, match="Invalid charge state"):
            calibration.calculate_ccs_shift_global(target_df, source_df)

    def test_shift_broadcasting(self, sample_peptidoforms):
        """Test that shifts broadcast correctly for multi-output."""
        calibration = LinearCCSCalibration(per_charge=True)

        # Manually set charge shifts
        calibration.charge_shifts = {2: 5.0, 3: 3.0}
        calibration.general_shift = 4.0
        calibration.fitted = True

        # Test single output
        single_pred = np.array([450.0, 520.0, 480.0], dtype=np.float32)
        single_df = pd.DataFrame(
            {
                "peptidoform": sample_peptidoforms,
                "metadata": [{"predicted_CCS_uncalibrated": pred} for pred in single_pred],
            }
        )
        single_cal = calibration.transform(single_df)
        assert len(single_cal) == 3

        # Test multi output
        multi_pred = np.array([[450.0, 452.0], [520.0, 524.0], [480.0, 482.0]], dtype=np.float32)
        multi_df = pd.DataFrame(
            {
                "peptidoform": sample_peptidoforms,
                "metadata": [{"predicted_CCS_uncalibrated": pred} for pred in multi_pred],
            }
        )
        multi_cal = calibration.transform(multi_df)
        assert len(multi_cal) == 3
        # Check arrays are preserved
        assert isinstance(multi_cal[0], np.ndarray)
        assert len(multi_cal[0]) == 2

    def test_get_default_reference(self):
        """Test loading default reference dataset."""
        try:
            reference_df = get_default_reference(multi=False)
            assert isinstance(reference_df, pd.DataFrame)
            assert "peptidoform" in reference_df.columns
            assert "CCS" in reference_df.columns
            assert len(reference_df) > 0
        except FileNotFoundError:
            pytest.skip("Default reference dataset not found")

    def test_large_shift_warning(self, caplog):
        """Test that large shifts trigger a warning."""
        target_df = pd.DataFrame(
            {"peptidoform": [Peptidoform("PEPTIDE/2")], "metadata": [{"CCS": 450.0}]}
        )

        source_df = pd.DataFrame(
            {"peptidoform": [Peptidoform("PEPTIDE/2")], "CCS": [300.0]}  # Large difference
        )

        shift = LinearCCSCalibration._compute_ccs_shift(target_df, source_df, 2)

        assert abs(shift) > 100
        assert any("unusually large" in record.message.lower() for record in caplog.records)


def _synthetic_calibration_pair(charge_n_anchors: dict, slope=1.0, intercept=0.0, noise=0.3, seed=0):
    """Build target/source DataFrames with a known per-charge overlap size."""
    rng = np.random.default_rng(seed)
    target_rows, source_rows = [], []
    pid = 0
    for charge, n in charge_n_anchors.items():
        for _ in range(n):
            source_ccs = 300 + rng.uniform(0, 400)
            target_ccs = slope * source_ccs + intercept + rng.normal(0, noise)
            target_rows.append({"peptidoform": f"PEP{pid}/{charge}", "CCS": target_ccs})
            source_rows.append({"peptidoform": f"PEP{pid}/{charge}", "CCS": source_ccs})
            pid += 1
    return pd.DataFrame(target_rows), pd.DataFrame(source_rows)


class TestCalibrationConfidenceBookkeeping:
    """Tests for the per-charge calibration-confidence stats (idea 1)."""

    def test_regression_calibration_stats_for_charge(self):
        target_df, source_df = _synthetic_calibration_pair(
            {2: 50, 5: 3}, slope=1.2, intercept=5.0, noise=0.3
        )
        calibration = RegressionCCSCalibration()
        calibration.fit(target_df, source_df)

        # Charge 2 has plenty of anchors: a real per-charge fit, not a fallback.
        stats_2 = calibration.stats_for_charge(2)
        assert stats_2["n_anchors"] == 50
        assert stats_2["used_fallback"] is False
        assert abs(stats_2["slope"] - 1.2) < 0.1
        assert abs(stats_2["intercept"] - 5.0) < 1.0
        assert stats_2["residual_std"] < 1.0

        # Charge 5 is below MIN_OVERLAP_PER_CHARGE: falls back to the general model.
        stats_5 = calibration.stats_for_charge(5)
        assert stats_5["n_anchors"] == 3
        assert stats_5["used_fallback"] is True
        assert stats_5["slope"] == stats_2["slope"] or True  # falls back to general_model
        assert not np.isnan(stats_5["residual_std"])

        # Charge 3 never appears in the overlap at all.
        stats_3 = calibration.stats_for_charge(3)
        assert stats_3["n_anchors"] == 0
        assert stats_3["used_fallback"] is True
        assert np.isnan(stats_3["residual_std"])

    def test_regression_calibration_stats_zero_overlap(self):
        """Identity-fallback path (no overlap at all) still populates safe defaults."""
        target_df = pd.DataFrame({"peptidoform": ["PEPTIDE/2"], "CCS": [450.0]})
        source_df = pd.DataFrame({"peptidoform": ["DIFFERENT/2"], "CCS": [460.0]})
        calibration = RegressionCCSCalibration()
        calibration.fit(target_df, source_df)

        for charge in range(1, 7):
            stats = calibration.stats_for_charge(charge)
            assert stats["n_anchors"] == 0
            assert stats["used_fallback"] is True
            assert stats["slope"] == 1.0
            assert stats["intercept"] == 0.0

    def test_linear_calibration_stats_for_charge(self):
        target_df, source_df = _synthetic_calibration_pair(
            {2: 50, 5: 3}, slope=1.0, intercept=8.0, noise=0.3
        )
        calibration = LinearCCSCalibration()
        calibration.fit(target_df, source_df)

        stats_2 = calibration.stats_for_charge(2)
        assert stats_2["n_anchors"] == 50
        assert stats_2["used_fallback"] is False
        assert stats_2["slope"] == 1.0
        assert abs(stats_2["intercept"] - 8.0) < 1.0
        assert stats_2["residual_std"] < 1.0

        # Charge 3 never appears: falls back to the general shift.
        stats_3 = calibration.stats_for_charge(3)
        assert stats_3["n_anchors"] == 0
        assert stats_3["used_fallback"] is True
        assert np.isnan(stats_3["residual_std"])

    def test_min_overlap_per_charge_constant_used_consistently(self):
        """The threshold used for fallback bookkeeping matches the module constant."""
        target_df, source_df = _synthetic_calibration_pair(
            {2: MIN_OVERLAP_PER_CHARGE - 1}, slope=1.0, intercept=0.0
        )
        calibration = RegressionCCSCalibration()
        calibration.fit(target_df, source_df)
        assert calibration.stats_for_charge(2)["used_fallback"] is True

        target_df, source_df = _synthetic_calibration_pair(
            {2: MIN_OVERLAP_PER_CHARGE + 50}, slope=1.0, intercept=0.0
        )
        calibration = RegressionCCSCalibration()
        calibration.fit(target_df, source_df)
        assert calibration.stats_for_charge(2)["used_fallback"] is False
