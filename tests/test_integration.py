"""Integration tests for IM2Deep package."""

import pytest
import numpy as np
from pathlib import Path
from psm_utils import PSM, PSMList, Peptidoform

from im2deep import core
from im2deep.calibration import LinearCCSCalibration
from im2deep._exceptions import IM2DeepError


class TestEndToEndWorkflow:
    """Integration tests for end-to-end workflows."""

    @pytest.mark.integration
    @pytest.mark.skipif(True, reason="Requires trained models")
    def test_predict_workflow(self, sample_psm_list):
        """Test complete prediction workflow."""
        # This would require actual trained models
        predictions = core.predict(sample_psm_list)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(sample_psm_list)
        assert np.all(predictions > 0)  # CCS values should be positive

    @pytest.mark.integration
    @pytest.mark.skipif(True, reason="Requires trained models")
    def test_predict_and_calibrate_workflow(self, sample_psm_list, sample_psm_list_with_ccs):
        """Test complete prediction and calibration workflow."""
        predictions = core.predict_and_calibrate(
            psm_list=sample_psm_list,
            psm_list_cal=sample_psm_list_with_ccs,
        )

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(sample_psm_list)
        assert np.all(predictions > 0)

    def test_calibration_workflow(
        self,
        sample_peptidoforms,
        sample_ccs_values,
        sample_predicted_ccs,
    ):
        """Test complete calibration workflow without prediction."""
        calibration = LinearCCSCalibration(per_charge=True)

        # Fit calibration
        calibration.fit(
            peptidoforms_target=sample_peptidoforms,
            observed_ccs_target=sample_ccs_values,
            peptidoforms_source=sample_peptidoforms.copy(),
            observed_ccs_source=sample_predicted_ccs,
        )

        assert calibration.is_fitted

        # Transform predictions
        calibrated = calibration.transform(
            peptidoforms=sample_peptidoforms,
            predicted_ccs=sample_predicted_ccs,
        )

        assert len(calibrated) == len(sample_predicted_ccs)
        assert np.all(calibrated > 0)

        # Calibrated values should be closer to targets
        original_error = np.mean(np.abs(sample_predicted_ccs - sample_ccs_values))
        calibrated_error = np.mean(np.abs(calibrated - sample_ccs_values))
        assert calibrated_error <= original_error

    def test_multi_output_calibration_workflow(
        self, sample_peptidoforms, sample_ccs_values, sample_predicted_ccs_multi
    ):
        """Test calibration workflow with multi-output predictions."""
        calibration = LinearCCSCalibration(per_charge=True)

        # Fit with single output targets
        calibration.fit(
            peptidoforms_target=sample_peptidoforms,
            observed_ccs_target=sample_ccs_values,
            peptidoforms_source=sample_peptidoforms.copy(),
            observed_ccs_source=sample_ccs_values - 2.0,
        )

        # Transform multi-output predictions
        calibrated = calibration.transform(
            peptidoforms=sample_peptidoforms,
            predicted_ccs=sample_predicted_ccs_multi,
        )

        assert calibrated.shape == sample_predicted_ccs_multi.shape
        assert calibrated.ndim == 2
        assert np.all(calibrated > 0)

    @pytest.mark.integration
    def test_file_parsing_to_prediction(self, tmp_path):
        """Test complete workflow from file parsing to prediction."""
        from im2deep.utils import parse_input

        # Create test file
        input_file = tmp_path / "input.csv"
        with open(input_file, "w") as f:
            f.write("seq,modifications,charge\n")
            f.write("PEPTIDE,,2\n")
            f.write("SEQUENCE,,3\n")

        # Parse input
        psm_list = parse_input(input_file)

        assert isinstance(psm_list, PSMList)
        assert len(psm_list) == 2

    def test_charge_state_coverage(self):
        """Test that calibration covers all relevant charge states."""
        # Create peptides with various charge states
        peptidoforms = [
            Peptidoform("PEPTIDE/1"),
            Peptidoform("PEPTIDE/2"),
            Peptidoform("PEPTIDE/3"),
            Peptidoform("PEPTIDE/4"),
            Peptidoform("PEPTIDE/5"),
        ]
        ccs_target = np.array([300.0, 400.0, 500.0, 600.0, 700.0])
        ccs_source = ccs_target - 5.0

        calibration = LinearCCSCalibration(per_charge=True)
        calibration.fit(
            peptidoforms_target=peptidoforms,
            observed_ccs_target=ccs_target,
            peptidoforms_source=peptidoforms.copy(),
            observed_ccs_source=ccs_source,
        )

        # All charges 1-6 should be covered (including extrapolation)
        assert all(c in calibration.charge_shifts for c in range(1, 7))

    def test_error_propagation(self, sample_psm_list):
        """Test that errors propagate correctly through the workflow."""
        # Invalid PSMList should raise IM2DeepError
        with pytest.raises(IM2DeepError):
            core.predict(None)

        # Empty PSMList should raise error
        with pytest.raises(IM2DeepError):
            core.predict(PSMList(psm_list=[]))


class TestDataConsistency:
    """Tests for data consistency across the pipeline."""

    def test_psm_list_preservation(self, sample_psm_list):
        """Test that PSMList properties are preserved through processing."""
        from im2deep.utils import validate_psm_list

        validated = validate_psm_list(sample_psm_list)

        # Check that all PSMs are preserved
        assert len(validated) == len(sample_psm_list)

        # Check that peptidoforms are preserved
        for orig, val in zip(sample_psm_list, validated):
            assert orig.peptidoform == val.peptidoform

    def test_ccs_value_consistency(self, sample_psm_list_with_ccs):
        """Test that CCS values remain consistent."""
        from im2deep.utils import validate_psm_list

        validated = validate_psm_list(sample_psm_list_with_ccs, needs_target=True)

        for orig, val in zip(sample_psm_list_with_ccs, validated):
            assert orig.metadata["CCS"] == val.metadata["CCS"]

    def test_array_shape_consistency(self, sample_peptidoforms, sample_predicted_ccs):
        """Test that array shapes remain consistent."""
        calibration = LinearCCSCalibration()

        # Set up a simple calibration
        calibration.charge_shifts = {2: 5.0, 3: 3.0}
        calibration.general_shift = 4.0
        calibration.fitted = True

        result = calibration.transform(sample_peptidoforms, sample_predicted_ccs)

        assert result.shape == sample_predicted_ccs.shape
        assert result.dtype == sample_predicted_ccs.dtype


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_peptide(self):
        """Test prediction with single peptide."""
        psm = PSM(
            peptidoform=Peptidoform("PEPTIDE/2"),
            spectrum_id="test_001",
            run="test_run",
            is_decoy=False,
        )
        psm_list = PSMList(psm_list=[psm])

        from im2deep.utils import validate_psm_list

        validated = validate_psm_list(psm_list)

        assert len(validated) == 1

    def test_high_charge_state(self):
        """Test handling of high charge states."""
        # Create PSMs with both valid and invalid charge states
        psm_valid = PSM(
            peptidoform=Peptidoform("PEPTIDE/3"),
            spectrum_id="test_001",
            run="test_run",
            is_decoy=False,
        )
        psm_high_charge = PSM(
            peptidoform=Peptidoform("PEPTIDE/10"),
            spectrum_id="test_002",
            run="test_run",
            is_decoy=False,
        )
        psm_list = PSMList(psm_list=[psm_valid, psm_high_charge])

        # Should filter out high charges (>6) but keep valid ones
        from im2deep.utils import validate_psm_list

        validated = validate_psm_list(psm_list)
        # After filtering, only the valid charge state should remain
        assert len(validated) == 1
        assert validated[0].peptidoform.precursor_charge == 3

    def test_modified_peptides(self):
        """Test handling of modified peptides."""
        psm = PSM(
            peptidoform=Peptidoform("PEP[+15.99]TIDE/2"),
            spectrum_id="test_001",
            run="test_run",
            is_decoy=False,
        )
        psm_list = PSMList(psm_list=[psm])

        from im2deep.utils import validate_psm_list

        validated = validate_psm_list(psm_list)
        assert len(validated) == 1

    def test_very_long_peptide(self):
        """Test handling of very long peptides."""
        long_seq = "A" * 100
        psm = PSM(
            peptidoform=Peptidoform(f"{long_seq}/2"),
            spectrum_id="test_001",
            run="test_run",
            is_decoy=False,
        )
        psm_list = PSMList(psm_list=[psm])

        from im2deep.utils import validate_psm_list

        validated = validate_psm_list(psm_list)
        assert len(validated) == 1

    def test_very_short_peptide(self):
        """Test handling of very short peptides."""
        psm = PSM(
            peptidoform=Peptidoform("AA/2"),
            spectrum_id="test_001",
            run="test_run",
            is_decoy=False,
        )
        psm_list = PSMList(psm_list=[psm])

        from im2deep.utils import validate_psm_list

        validated = validate_psm_list(psm_list)
        assert len(validated) == 1
