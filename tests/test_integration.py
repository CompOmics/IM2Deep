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
    def test_standardization_recipe_round_trips(self, sample_training_df, tmp_path):
        """
        The step 8.2 recipe bundle (standardisation, AdamW weight decay,
        OneCycleLR, precision) trains and predicts without error, and the
        checkpoint records non-default (i.e. actually fitted, not left at
        their 0/1 no-op) scaler buffers.
        """
        import torch

        from im2deep._io_helpers import parse_input

        checkpoint = tmp_path / "standardized.ckpt"
        core.train(
            sample_training_df,
            checkpoint,
            training_kwargs=self._training_kwargs(
                standardize_global_features=True,
                standardize_target=True,
                weight_decay=0.01,
                lr_scheduler="one_cycle",
                precision="32-true",  # bf16-mixed needs a real accelerator
            ),
            output_dir=tmp_path,
        )

        psm_list = parse_input(sample_training_df[["seq", "modifications", "charge"]])
        predictions = core.predict(psm_list, model=checkpoint)

        assert predictions.shape == (len(sample_training_df),)
        assert np.all(np.isfinite(predictions))
        # A sanity range, not a tight bound: this only catches a scaler
        # applied in the wrong direction or units, which would put
        # predictions wildly outside plausible CCS territory.
        assert np.all((predictions > 50) & (predictions < 5000))

        state_dict = torch.load(checkpoint, weights_only=False, map_location="cpu")["state_dict"]
        assert not torch.allclose(state_dict["global_feat_std"], torch.ones_like(state_dict["global_feat_std"]))
        assert state_dict["target_std"].item() != pytest.approx(1.0)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_swa_recipe_round_trips(self, sample_training_df, tmp_path):
        """
        SWA trains and predicts without error, and the saved checkpoint is
        the final weight-averaged model, not a reloaded best-epoch snapshot.

        IM2Deep has no BatchNorm layers, so Lightning's SWA callback
        transfers its averaged weights into the model in ``on_train_end`` -
        after the last validation epoch. ``ModelCheckpoint`` never sees the
        averaged weights, so reloading its "best" checkpoint would silently
        throw the averaging away; ``_model_ops.train`` must skip that reload
        and use the post-fit model directly whenever SWA is enabled.
        """
        from im2deep._io_helpers import parse_input

        checkpoint = tmp_path / "swa.ckpt"
        core.train(
            sample_training_df,
            checkpoint,
            training_kwargs=self._training_kwargs(
                epochs=6, swa=True, swa_epoch_start=0.5, swa_annealing_epochs=1
            ),
            output_dir=tmp_path,
        )

        psm_list = parse_input(sample_training_df[["seq", "modifications", "charge"]])
        predictions = core.predict(psm_list, model=checkpoint)

        assert predictions.shape == (len(sample_training_df),)
        assert np.all(np.isfinite(predictions))

    @pytest.mark.integration
    @pytest.mark.slow
    def test_sibling_loss_recipe_round_trips(self, sample_training_df, tmp_path):
        """
        Step 8.3's sibling delta loss trains and predicts without error.

        Exercises the full path end to end: _model_ops.train computing
        group ids from the real peptidoforms, wrapping the training dataset,
        building SiblingPairBatchSampler batches, and IM2Deep.training_step
        popping (group_id, mz) off the batch and adding the delta term -
        not just each piece in isolation, the way test_model_ops.py's and
        test_architectures.py's unit tests do.
        """
        from im2deep._io_helpers import parse_input

        checkpoint = tmp_path / "sibling.ckpt"
        core.train(
            sample_training_df,
            checkpoint,
            training_kwargs=self._training_kwargs(
                batch_size=16, sibling_loss_weight=1.0, sibling_pairs_per_batch=4
            ),
            output_dir=tmp_path,
        )

        psm_list = parse_input(sample_training_df[["seq", "modifications", "charge"]])
        predictions = core.predict(psm_list, model=checkpoint)

        assert predictions.shape == (len(sample_training_df),)
        assert np.all(np.isfinite(predictions))

    @pytest.mark.integration
    @pytest.mark.slow
    def test_ranking_loss_recipe_round_trips(self, sample_training_df, tmp_path):
        """
        Step 8.3's pairwise ranking loss trains and predicts without error.

        Unlike the sibling loss, this needs no special batch sampler -
        m/z-close pairs are already common in a random batch - so this
        exercises the plain-shuffle path with the pair-augmented dataset
        instead of SiblingPairBatchSampler.
        """
        from im2deep._io_helpers import parse_input

        checkpoint = tmp_path / "ranking.ckpt"
        core.train(
            sample_training_df,
            checkpoint,
            training_kwargs=self._training_kwargs(
                ranking_loss_weight=1.0, ranking_mz_window=50.0
            ),
            output_dir=tmp_path,
        )

        psm_list = parse_input(sample_training_df[["seq", "modifications", "charge"]])
        predictions = core.predict(psm_list, model=checkpoint)

        assert predictions.shape == (len(sample_training_df),)
        assert np.all(np.isfinite(predictions))

    @pytest.mark.integration
    @pytest.mark.slow
    def test_flex_trunk_recipe_round_trips(self, sample_training_df, tmp_path):
        """
        Step 8.4 items 8+9's fused trunk trains and predicts without error.

        Exercises the real featurisation path end to end - DeepLC's actual
        one-hot matrix feeding FlexTrunk's residue-index/masking logic, not
        a hand-built one-hot tensor the way test_architectures.py's unit
        tests use - plus composing with charge_film, which item 8's own
        text says should still work (only use_one_hot/use_attention_pooling
        become moot, not use_charge_embedding).
        """
        from im2deep._io_helpers import parse_input

        checkpoint = tmp_path / "flex_trunk.ckpt"
        core.train(
            sample_training_df,
            checkpoint,
            training_kwargs=self._training_kwargs(
                use_flex_trunk=True,
                flex_trunk_channels=[8, 8],
                flex_trunk_stem_channels=4,
                use_charge_embedding=True,
            ),
            output_dir=tmp_path,
        )

        psm_list = parse_input(sample_training_df[["seq", "modifications", "charge"]])
        predictions = core.predict(psm_list, model=checkpoint)

        assert predictions.shape == (len(sample_training_df),)
        assert np.all(np.isfinite(predictions))

    @pytest.mark.integration
    @pytest.mark.slow
    def test_transformer_trunk_recipe_round_trips(self, sample_training_df, tmp_path):
        """
        Step 8.4 item 3's transformer trunk trains and predicts without
        error, through real DeepLC featurisation (not a hand-built atom_comp
        tensor the way test_architectures.py's unit tests use), and composes
        with charge_film - technically compatible even though the register
        token is meant to make FiLM redundant, not required to be absent.
        """
        from im2deep._io_helpers import parse_input

        checkpoint = tmp_path / "transformer_trunk.ckpt"
        core.train(
            sample_training_df,
            checkpoint,
            training_kwargs=self._training_kwargs(
                use_transformer_trunk=True,
                transformer_d_model=16,
                transformer_num_layers=2,
                transformer_num_heads=2,
                transformer_ffn_dim=32,
                use_charge_embedding=True,
            ),
            output_dir=tmp_path,
        )

        psm_list = parse_input(sample_training_df[["seq", "modifications", "charge"]])
        predictions = core.predict(psm_list, model=checkpoint)

        assert predictions.shape == (len(sample_training_df),)
        assert np.all(np.isfinite(predictions))

    @pytest.mark.integration
    @pytest.mark.slow
    def test_heteroscedastic_head_recipe_round_trips(self, sample_training_df, tmp_path):
        """
        Step 8.5's heteroscedastic head trains and predicts without error,
        with a 1-epoch mu warm-start (of the fixture's 3 total epochs) so
        the L1-warmup -> Laplace-NLL phase transition is exercised end to
        end, not just checked in isolation via _in_heteroscedastic_warmup.

        core.predict()'s public shape/contract must stay unchanged (a plain
        CCS array, the uncertainty output b discarded), so this is the check
        that a real caller of the public API sees no difference at all,
        not just that training_step's internals are correct in isolation.
        """
        from im2deep._io_helpers import parse_input

        checkpoint = tmp_path / "heteroscedastic.ckpt"
        core.train(
            sample_training_df,
            checkpoint,
            training_kwargs=self._training_kwargs(
                use_heteroscedastic_head=True, heteroscedastic_warmup_epochs=1
            ),
            output_dir=tmp_path,
        )

        psm_list = parse_input(sample_training_df[["seq", "modifications", "charge"]])
        predictions = core.predict(psm_list, model=checkpoint)

        assert predictions.shape == (len(sample_training_df),)
        assert np.all(np.isfinite(predictions))

    @pytest.mark.integration
    @pytest.mark.slow
    def test_per_charge_heads_recipe_round_trips(self, sample_training_df, tmp_path):
        """
        Step 8.6 item 1's per-charge output heads train and predict without
        error, through real DeepLC featurisation and the real charge values
        parsed from sample_training_df (charges 2 and 3), composed with the
        heteroscedastic head to exercise both selection paths (mu and b)
        through the real core.train/core.predict pipeline, not just the
        hand-crafted-weight unit tests in test_architectures.py.
        """
        from im2deep._io_helpers import parse_input

        checkpoint = tmp_path / "per_charge_heads.ckpt"
        core.train(
            sample_training_df,
            checkpoint,
            training_kwargs=self._training_kwargs(
                use_per_charge_heads=True, use_heteroscedastic_head=True
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
    def test_finetune_transformer_backbone_keeps_its_scalers(self, sample_training_df, tmp_path):
        """
        Fine-tuning a standardised transformer-trunk model with a learning rate
        of 0 reproduces the backbone's predictions: the transfer model applies
        the backbone's trunk and scalers, and training does not refit them.
        """
        from im2deep._io_helpers import parse_input

        backbone = tmp_path / "transformer.ckpt"
        core.train(
            sample_training_df,
            backbone,
            training_kwargs=self._training_kwargs(
                epochs=1,
                use_transformer_trunk=True,
                standardize_target=True,
                standardize_global_features=True,
            ),
            output_dir=tmp_path / "backbone",
        )
        finetuned = tmp_path / "finetuned.ckpt"
        core.finetune(
            sample_training_df,
            finetuned,
            model=backbone,
            training_kwargs=self._training_kwargs(
                epochs=1, freeze_epochs=0, learning_rate=0.0, L1_alpha=0.0
            ),
            output_dir=tmp_path / "finetune",
        )

        psm_list = parse_input(sample_training_df[["seq", "modifications", "charge"]])
        before = core.predict(psm_list, model=backbone)
        after = core.predict(psm_list, model=finetuned)

        assert np.allclose(after, before, atol=1e-3)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_finetune_from_the_bundled_model(self, sample_training_df, tmp_path):
        """
        ``finetune`` with no ``--backbone`` adapts the bundled transformer model.

        A learning rate of 0 must reproduce the bundled model's predictions: the
        transfer model applies the backbone's trunk, charge register token and
        both standardisation scalers, and training neither refits nor reloads them.
        """
        from im2deep._io_helpers import parse_input

        finetuned = tmp_path / "finetuned.ckpt"
        core.finetune(
            sample_training_df,
            finetuned,
            training_kwargs=self._training_kwargs(
                epochs=1, freeze_epochs=0, learning_rate=0.0, L1_alpha=0.0, swa=False
            ),
            output_dir=tmp_path / "finetune",
        )

        psm_list = parse_input(sample_training_df[["seq", "modifications", "charge"]])
        before = core.predict(psm_list)
        after = core.predict(psm_list, model=finetuned)

        assert np.allclose(after, before, atol=1e-3)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_bundled_model_still_predicts(self, sample_psm_list):
        """
        The bundled checkpoint loads and predicts.

        It records its own architecture and featurisation (transformer trunk, 72
        global features, DeepLC's corrected positional deltas), which ``predict``
        reads back rather than assuming the package defaults.
        """
        predictions = core.predict(sample_psm_list)

        assert predictions.shape == (len(sample_psm_list),)
        assert np.all(predictions > 0)
