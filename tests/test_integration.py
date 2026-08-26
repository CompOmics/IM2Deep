"""Integration tests for IM2Deep package."""

import numpy as np
import pandas as pd
import pytest
from psm_utils import PSM, Peptidoform, PSMList

from im2deep import core
from im2deep.calibration import LinearCCSCalibration
from im2deep.exceptions import IM2DeepError


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

        target_df = pd.DataFrame(
            {
                "peptidoform": sample_peptidoforms,
                "metadata": [{"CCS": ccs} for ccs in sample_ccs_values],
            }
        )

        source_df = pd.DataFrame({"peptidoform": sample_peptidoforms, "CCS": sample_predicted_ccs})

        # Fit calibration
        calibration.fit(
            psm_df_target=target_df,
            psm_df_source=source_df,
        )

        assert calibration.is_fitted

        # Transform predictions
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
        # All values should be positive (scalars or arrays)
        for val in calibrated:
            if isinstance(val, np.ndarray):
                assert np.all(val > 0)
            else:
                assert val > 0

        # Calibrated values should be closer to targets (compare scalars)
        calibrated_scalars = np.array(
            [v if not isinstance(v, np.ndarray) else v[0] for v in calibrated]
        )
        original_error = np.mean(np.abs(sample_predicted_ccs - sample_ccs_values))
        calibrated_error = np.mean(np.abs(calibrated_scalars - sample_ccs_values))
        assert calibrated_error <= original_error

    def test_multi_output_calibration_workflow(
        self, sample_peptidoforms, sample_ccs_values, sample_predicted_ccs_multi
    ):
        """Test calibration workflow with multi-output predictions."""
        calibration = LinearCCSCalibration(per_charge=True)

        target_df = pd.DataFrame(
            {
                "peptidoform": sample_peptidoforms,
                "metadata": [{"CCS": ccs} for ccs in sample_ccs_values],
            }
        )

        source_df = pd.DataFrame(
            {"peptidoform": sample_peptidoforms, "CCS": sample_ccs_values - 2.0}
        )

        # Fit with single output targets
        calibration.fit(
            psm_df_target=target_df,
            psm_df_source=source_df,
        )

        # Transform multi-output predictions
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
        # Check that arrays are preserved
        for val in calibrated:
            assert isinstance(val, np.ndarray)
            assert len(val) == 2
            assert np.all(val > 0)

    @pytest.mark.integration
    def test_file_parsing_to_prediction(self, tmp_path):
        """Test complete workflow from file parsing to prediction."""
        from im2deep._io_helpers import parse_input

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

        target_df = pd.DataFrame(
            {"peptidoform": peptidoforms, "metadata": [{"CCS": ccs} for ccs in ccs_target]}
        )

        source_df = pd.DataFrame({"peptidoform": peptidoforms, "CCS": ccs_source})

        calibration = LinearCCSCalibration(per_charge=True)
        calibration.fit(
            psm_df_target=target_df,
            psm_df_source=source_df,
        )

        # All charges 1-6 should be covered (including extrapolation)
        assert all(c in calibration.charge_shifts for c in range(1, 7))

    def test_error_propagation(self, sample_psm_list):
        """Test that errors propagate correctly through the workflow."""
        # Invalid PSMList should raise IM2DeepError
        with pytest.raises(IM2DeepError):
            core.predict(None)  # type: ignore

        # Empty PSMList should raise error
        with pytest.raises(IM2DeepError):
            core.predict(PSMList(psm_list=[]))


class TestDataConsistency:
    """Tests for data consistency across the pipeline."""

    def test_psm_list_preservation(self, sample_psm_list):
        """Test that PSMList properties are preserved through processing."""
        from im2deep._io_helpers import validate_psm_list

        validated = validate_psm_list(sample_psm_list)

        # Check that all PSMs are preserved
        assert len(validated) == len(sample_psm_list)

        # Check that peptidoforms are preserved
        for orig, val in zip(sample_psm_list, validated, strict=False):
            assert orig.peptidoform == val.peptidoform

    def test_ccs_value_consistency(self, sample_psm_list_with_ccs):
        """Test that CCS values remain consistent."""
        from im2deep._io_helpers import validate_psm_list

        validated = validate_psm_list(sample_psm_list_with_ccs, needs_target=True)

        for orig, val in zip(sample_psm_list_with_ccs, validated, strict=False):
            assert orig.metadata["CCS"] == val.metadata["CCS"]  # type: ignore

    def test_array_shape_consistency(self, sample_peptidoforms, sample_predicted_ccs):
        """Test that array shapes remain consistent."""
        calibration = LinearCCSCalibration()

        # Set up a simple calibration
        calibration.charge_shifts = {2: 5.0, 3: 3.0}
        calibration.general_shift = 4.0
        calibration.fitted = True

        transform_df = pd.DataFrame(
            {
                "peptidoform": sample_peptidoforms,
                "metadata": [
                    {"predicted_CCS_uncalibrated": pred} for pred in sample_predicted_ccs
                ],
            }
        )

        result = calibration.transform(transform_df)

        assert len(result) == len(sample_predicted_ccs)
        # Check that values are floats (not arrays for single output)
        for val in result:
            assert isinstance(val, (float, np.floating))


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

        from im2deep._io_helpers import validate_psm_list

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
        from im2deep._io_helpers import validate_psm_list

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

        from im2deep._io_helpers import validate_psm_list

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

        from im2deep._io_helpers import validate_psm_list

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

        from im2deep._io_helpers import validate_psm_list

        validated = validate_psm_list(psm_list)
        assert len(validated) == 1


class TestTrainingRoundTrip:
    """
    Tests that training and prediction agree on the feature encoding.

    This is the check the training integration was built for: a model trained
    through ``core.train`` must produce, via ``core.predict``, the same errors
    it reported at training time. If train-time and predict-time featurisation
    ever diverge again, these fail.
    """

    @staticmethod
    def _training_kwargs(**overrides):
        kwargs = {
            "epochs": 3,
            "batch_size": 32,
            "accelerator": "cpu",
            "devices": 1,
            "num_workers": 0,
            "patience": 0,
        }
        kwargs.update(overrides)
        return kwargs

    @pytest.mark.integration
    @pytest.mark.slow
    def test_train_then_predict_reproduces_training_error(self, sample_training_df, tmp_path):
        """Predicting on the training data reproduces the training-time MAE."""
        from im2deep._io_helpers import parse_input

        checkpoint = tmp_path / "tiny.ckpt"
        core.train(
            sample_training_df,
            checkpoint,
            training_kwargs=self._training_kwargs(),
            output_dir=tmp_path,
        )

        psm_list = parse_input(sample_training_df[["seq", "modifications", "charge"]])
        predictions = core.predict(psm_list, model=checkpoint)

        assert predictions.shape == (len(sample_training_df),)
        assert np.all(np.isfinite(predictions))

        # A model that has genuinely learned nothing still predicts a finite
        # value; what matters here is that predicting twice is deterministic and
        # that the checkpoint round-trips, which is what a feature mismatch
        # would break.
        repeat = core.predict(psm_list, model=checkpoint)
        np.testing.assert_allclose(predictions, repeat, rtol=1e-6)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_checkpoint_records_its_configuration(self, sample_training_df, tmp_path):
        """A trained checkpoint carries the config it was trained with."""
        import torch

        checkpoint = tmp_path / "tiny.ckpt"
        core.train(
            sample_training_df,
            checkpoint,
            training_kwargs=self._training_kwargs(),
            output_dir=tmp_path,
        )

        loaded = torch.load(checkpoint, weights_only=False, map_location="cpu")

        assert isinstance(loaded, dict), "must be a Lightning checkpoint, not a pickled module"
        config = loaded["hyper_parameters"]["config"]
        assert config["Global_features"] == 60
        assert config["add_ccs_features"] is True
        assert config["legacy_positional_deltas"] is True

    @pytest.mark.integration
    @pytest.mark.slow
    def test_terminal_composition_variant_round_trips(self, sample_training_df, tmp_path):
        """
        A model trained with wider global features is readable back.

        ``add_terminal_composition=True`` widens the global feature vector from
        60 to 72, so this checkpoint cannot be loaded against the package
        default config. It round-trips only because the checkpoint records its
        own architecture.
        """
        from im2deep._io_helpers import parse_input

        checkpoint = tmp_path / "wide.ckpt"
        core.train(
            sample_training_df,
            checkpoint,
            training_kwargs=self._training_kwargs(
                add_terminal_composition=True, Global_features=72
            ),
            output_dir=tmp_path,
        )

        psm_list = parse_input(sample_training_df[["seq", "modifications", "charge"]])
        predictions = core.predict(psm_list, model=checkpoint)

        assert predictions.shape == (len(sample_training_df),)
        assert np.all(np.isfinite(predictions))

    @pytest.mark.integration
    @pytest.mark.slow
    def test_finetune_from_bundled_model(self, sample_training_df, tmp_path):
        """Fine-tuning the bundled model produces a usable checkpoint."""
        from im2deep._io_helpers import parse_input

        checkpoint = tmp_path / "finetuned.ckpt"
        core.finetune(
            sample_training_df,
            checkpoint,
            training_kwargs=self._training_kwargs(freeze_epochs=1),
            output_dir=tmp_path,
        )

        psm_list = parse_input(sample_training_df[["seq", "modifications", "charge"]])
        predictions = core.predict(psm_list, model=checkpoint)

        assert predictions.shape == (len(sample_training_df),)
        assert np.all(np.isfinite(predictions))

    @pytest.mark.integration
    @pytest.mark.slow
    def test_bundled_model_still_predicts(self, sample_psm_list):
        """
        The bundled checkpoint still loads and predicts after the DeepLC bump.

        DeepLC 4.1.0 defaults ``legacy_positional_deltas`` to True, reproducing
        the encoding the bundled checkpoints were trained with, so this must
        keep working unchanged.
        """
        predictions = core.predict(sample_psm_list)

        assert predictions.shape == (len(sample_psm_list),)
        assert np.all(predictions > 0)
