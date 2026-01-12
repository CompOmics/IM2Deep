"""Tests for calibration module."""

import pytest
import numpy as np
from psm_utils import Peptidoform

from im2deep.calibration import LinearCCSCalibration
from im2deep._exceptions import CalibrationError


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

        # Create reference data
        ref_peptidoforms = sample_peptidoforms.copy()
        ref_ccs = sample_ccs_values.copy()

        calibration.fit(
            peptidoforms_target=sample_peptidoforms,
            observed_ccs_target=sample_ccs_values,
            peptidoforms_source=ref_peptidoforms,
            observed_ccs_source=sample_predicted_ccs,
        )

        assert calibration.is_fitted is True
        assert len(calibration.charge_shifts) > 0
        assert calibration.general_shift is not None

    def test_fit_global(self, sample_peptidoforms, sample_ccs_values, sample_predicted_ccs):
        """Test fitting with global calibration."""
        calibration = LinearCCSCalibration(per_charge=False, use_charge_state=2)

        ref_peptidoforms = sample_peptidoforms.copy()
        ref_ccs = sample_predicted_ccs.copy()

        calibration.fit(
            peptidoforms_target=sample_peptidoforms,
            observed_ccs_target=sample_ccs_values,
            peptidoforms_source=ref_peptidoforms,
            observed_ccs_source=ref_ccs,
        )

        assert calibration.is_fitted is True
        assert calibration.general_shift is not None
        assert isinstance(calibration.general_shift, float)

    def test_transform_single_output(
        self, sample_peptidoforms, sample_ccs_values, sample_predicted_ccs
    ):
        """Test transforming single-output predictions."""
        calibration = LinearCCSCalibration(per_charge=True)

        calibration.fit(
            peptidoforms_target=sample_peptidoforms,
            observed_ccs_target=sample_ccs_values,
            peptidoforms_source=sample_peptidoforms.copy(),
            observed_ccs_source=sample_predicted_ccs,
        )

        calibrated = calibration.transform(
            peptidoforms=sample_peptidoforms,
            predicted_ccs=sample_predicted_ccs,
        )

        assert calibrated.shape == sample_predicted_ccs.shape
        assert isinstance(calibrated, np.ndarray)
        assert calibrated.dtype == np.float32

    def test_transform_multi_output(
        self, sample_peptidoforms, sample_ccs_values, sample_predicted_ccs_multi
    ):
        """Test transforming multi-output predictions."""
        calibration = LinearCCSCalibration(per_charge=True)

        # Fit with single output
        calibration.fit(
            peptidoforms_target=sample_peptidoforms,
            observed_ccs_target=sample_ccs_values,
            peptidoforms_source=sample_peptidoforms.copy(),
            observed_ccs_source=sample_ccs_values - 2.0,  # Simulate shift
        )

        # Transform multi-output
        calibrated = calibration.transform(
            peptidoforms=sample_peptidoforms,
            predicted_ccs=sample_predicted_ccs_multi,
        )

        assert calibrated.shape == sample_predicted_ccs_multi.shape
        assert calibrated.ndim == 2
        assert calibrated.shape[1] == 2  # Two conformers
        assert isinstance(calibrated, np.ndarray)

    def test_transform_not_fitted(self, sample_peptidoforms, sample_predicted_ccs):
        """Test transform raises error when not fitted."""
        calibration = LinearCCSCalibration()

        with pytest.raises(CalibrationError, match="not been fitted"):
            calibration.transform(
                peptidoforms=sample_peptidoforms,
                predicted_ccs=sample_predicted_ccs,
            )

    def test_calculate_ccs_shift_no_overlap(self):
        """Test shift calculation with no overlapping peptides."""
        calibration = LinearCCSCalibration(per_charge=False, use_charge_state=2)

        target_peptidoforms = [Peptidoform("PEPTIDE/2")]
        target_ccs = np.array([450.0])
        source_peptidoforms = [Peptidoform("DIFFERENT/2")]
        source_ccs = np.array([460.0])

        shift = calibration.calculate_ccs_shift(
            target_peptidoforms, target_ccs, source_peptidoforms, source_ccs
        )

        assert shift == 0.0  # No overlap returns 0.0

    def test_calculate_ccs_shift_with_overlap(self):
        """Test shift calculation with overlapping peptides."""
        calibration = LinearCCSCalibration(per_charge=False, use_charge_state=2)

        peptidoforms = [Peptidoform("PEPTIDE/2"), Peptidoform("SEQUENCE/2")]
        target_ccs = np.array([450.0, 520.0])
        source_ccs = np.array([445.0, 515.0])

        shift = calibration.calculate_ccs_shift(
            peptidoforms, target_ccs, list(peptidoforms), source_ccs
        )

        assert isinstance(shift, float)
        assert abs(shift - 5.0) < 0.1  # Should be approximately 5.0

    def test_compute_ccs_shift_per_charge(self):
        """Test per-charge shift computation."""
        peptidoforms = [
            Peptidoform("PEPTIDE/2"),
            Peptidoform("SEQUENCE/3"),
            Peptidoform("TEST/2"),
        ]
        target_ccs = np.array([450.0, 520.0, 480.0])
        source_ccs = np.array([445.0, 515.0, 475.0])

        shifts = LinearCCSCalibration._compute_ccs_shift_per_charge(
            peptidoforms, target_ccs, list(peptidoforms), source_ccs
        )

        assert isinstance(shifts, dict)
        assert 2 in shifts
        assert 3 in shifts
        assert abs(shifts[2] - 5.0) < 0.1
        assert abs(shifts[3] - 5.0) < 0.1

    def test_fit_with_missing_charges(self, sample_peptidoforms, sample_ccs_values):
        """Test that missing charges are filled with general shift."""
        calibration = LinearCCSCalibration(per_charge=True)

        calibration.fit(
            peptidoforms_target=sample_peptidoforms,
            observed_ccs_target=sample_ccs_values,
            peptidoforms_source=sample_peptidoforms.copy(),
            observed_ccs_source=sample_ccs_values - 5.0,
        )

        # Check that charges 1-6 are all filled
        for charge in range(1, 7):
            assert charge in calibration.charge_shifts
            assert isinstance(calibration.charge_shifts[charge], float)

    def test_fit_invalid_charge_state(self, sample_peptidoforms, sample_ccs_values):
        """Test that invalid charge state raises error."""
        calibration = LinearCCSCalibration(per_charge=False, use_charge_state=10)

        with pytest.raises(CalibrationError, match="Invalid charge state"):
            calibration.calculate_ccs_shift(
                sample_peptidoforms,
                sample_ccs_values,
                sample_peptidoforms.copy(),
                sample_ccs_values,
            )

    def test_shift_broadcasting(self, sample_peptidoforms):
        """Test that shifts broadcast correctly for multi-output."""
        calibration = LinearCCSCalibration(per_charge=True)

        # Manually set charge shifts
        calibration.charge_shifts = {2: 5.0, 3: 3.0}
        calibration.general_shift = 4.0
        calibration.fitted = True

        # Test single output
        single_pred = np.array([450.0, 520.0, 480.0], dtype=np.float32)
        single_cal = calibration.transform(sample_peptidoforms, single_pred)
        assert single_cal.shape == (3,)

        # Test multi output
        multi_pred = np.array([[450.0, 452.0], [520.0, 524.0], [480.0, 482.0]], dtype=np.float32)
        multi_cal = calibration.transform(sample_peptidoforms, multi_pred)
        assert multi_cal.shape == (3, 2)

    def test_get_default_reference(self):
        """Test loading default reference dataset."""
        calibration = LinearCCSCalibration()

        try:
            peptidoforms, ccs_values = calibration.get_default_reference(multi=False)
            assert len(peptidoforms) > 0
            assert len(ccs_values) > 0
            assert len(peptidoforms) == len(ccs_values)
        except FileNotFoundError:
            pytest.skip("Default reference dataset not found")

    def test_large_shift_warning(self, caplog):
        """Test that large shifts trigger a warning."""
        calibration = LinearCCSCalibration(per_charge=False, use_charge_state=2)

        peptidoforms = [Peptidoform("PEPTIDE/2")]
        target_ccs = np.array([450.0])
        source_ccs = np.array([300.0])  # Large difference

        shift = LinearCCSCalibration._compute_ccs_shift(
            peptidoforms, target_ccs, list(peptidoforms), source_ccs, 2
        )

        assert abs(shift) > 100
        assert any("unusually large" in record.message.lower() for record in caplog.records)
