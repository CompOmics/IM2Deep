"""Tests for model operations module."""

from collections import Counter
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from im2deep import _model_ops
from im2deep._data import build_training_dataset
from im2deep.constants import DEFAULT_TRAINING_CONFIG


class TestLoadModel:
    """Tests for load_model function."""

    def test_load_model_from_path(self, temp_model_path):
        """Test loading model from file path."""
        # Create a mock model
        model = torch.nn.Linear(10, 1)
        torch.save(model, temp_model_path)

        loaded_model = _model_ops.load_model(temp_model_path)

        assert isinstance(loaded_model, torch.nn.Module)
        # load_model does not force eval mode; caller controls train/eval state.
        assert loaded_model.training is model.training

    def test_load_model_from_module(self):
        """Test loading model from existing module."""
        model = torch.nn.Linear(10, 1)
        loaded_model = _model_ops.load_model(model)

        assert loaded_model is model
        assert isinstance(loaded_model, torch.nn.Module)

    def test_load_model_none(self):
        """Test loading model with None raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            _model_ops.load_model(None)

    def test_load_model_invalid_type(self):
        """Test loading model with invalid type raises TypeError."""
        with pytest.raises(TypeError):
            _model_ops.load_model(12345)  # type: ignore

    def test_load_model_dict_checkpoint(self, temp_model_path):
        """Test loading model from dict checkpoint."""
        model = torch.nn.Linear(10, 1)
        checkpoint = {"model": model, "epoch": 10}
        torch.save(checkpoint, temp_model_path)

        loaded_model = _model_ops.load_model(temp_model_path)

        assert isinstance(loaded_model, torch.nn.Module)

    def test_load_model_state_dict_only(self, temp_model_path):
        """Test loading model with state_dict only raises NotImplementedError."""
        model = torch.nn.Linear(10, 1)
        checkpoint = {"state_dict": model.state_dict()}
        torch.save(checkpoint, temp_model_path)

        with pytest.raises(NotImplementedError, match="state_dict"):
            _model_ops.load_model(temp_model_path)

    def test_load_model_device_cpu(self, temp_model_path):
        """Test loading model on CPU."""
        model = torch.nn.Linear(10, 1)
        torch.save(model, temp_model_path)

        loaded_model = _model_ops.load_model(temp_model_path, device="cpu")

        # Check that model is on CPU
        assert next(loaded_model.parameters()).device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_load_model_device_cuda(self, temp_model_path):
        """Test loading model on CUDA."""
        model = torch.nn.Linear(10, 1)
        torch.save(model, temp_model_path)

        loaded_model = _model_ops.load_model(temp_model_path, device="cuda")

        assert next(loaded_model.parameters()).device.type == "cuda"


class TestPredict:
    """Tests for predict function."""

    @patch("im2deep._model_ops.read_checkpoint_architecture")
    @patch("im2deep._model_ops.read_checkpoint_config")
    @patch("im2deep._model_ops._get_loss_function")
    def test_predict_single_output(self, mock_loss, mock_config, mock_arch, sample_psm_list):
        """Test prediction with single-output model."""
        # Create mock model
        mock_model_instance = MagicMock()
        mock_model_instance.eval.return_value = None
        mock_model_instance.return_value = torch.tensor([[450.0], [520.0], [480.0]])

        mock_arch.return_value.load_from_checkpoint.return_value = mock_model_instance
        mock_config.return_value = {}
        mock_loss.return_value = torch.nn.L1Loss()

        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 3
        mock_dataset.__getitem__.return_value = (
            [torch.randn(10), torch.randn(5)],
            torch.tensor([0.0]),
        )

        with patch("im2deep._model_ops._predict_loop") as mock_predict_loop:
            mock_predict_loop.return_value = torch.tensor([450.0, 520.0, 480.0])

            predictions = _model_ops.predict(
                model="fake_model.ckpt",
                data=mock_dataset,
                multi=False,
            )

            assert isinstance(predictions, torch.Tensor)
            assert len(predictions) == 3

    @patch("im2deep._model_ops.read_checkpoint_architecture")
    @patch("im2deep._model_ops.read_checkpoint_config")
    @patch("im2deep._model_ops._get_loss_function")
    def test_predict_multi_output(self, mock_loss, mock_config, mock_arch):
        """Test prediction with multi-output model."""
        mock_model_instance = MagicMock()
        mock_model_instance.eval.return_value = None

        mock_arch.return_value.load_from_checkpoint.return_value = mock_model_instance
        mock_config.return_value = {}
        mock_loss.return_value = MagicMock()

        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 3

        with patch("im2deep._model_ops._predict_loop") as mock_predict_loop:
            mock_predict_loop.return_value = torch.tensor(
                [[450.0, 452.0], [520.0, 524.0], [480.0, 482.0]]
            )

            predictions = _model_ops.predict(
                model="fake_model.ckpt",
                data=mock_dataset,
                multi=True,
            )

            assert isinstance(predictions, torch.Tensor)
            assert predictions.shape == (3, 2)

    def test_predict_no_data(self):
        """Test prediction without data raises ValueError."""
        with pytest.raises(ValueError, match="Data must be provided"):
            _model_ops.predict(model="fake_model.ckpt", data=None)


class TestPredictLoop:
    """Tests for _predict_loop function."""

    def test_predict_loop_single_output(self):
        """Test prediction loop with single-output model."""
        # Create mock model
        model = MagicMock()
        model.eval.return_value = None
        model.return_value = torch.tensor([[450.0], [520.0]])

        # Create mock data loader
        mock_data = [
            ([torch.randn(2, 10), torch.randn(2, 5)], torch.zeros(2)),
        ]

        with patch("im2deep._model_ops.track", return_value=mock_data):
            predictions = _model_ops._predict_loop(
                model=model,
                data_loader=mock_data,  # type: ignore[unknown-arg]
                device="cpu",
            )

            assert isinstance(predictions, torch.Tensor)

    def test_predict_loop_multi_output(self):
        """Test prediction loop with multi-output model (e.g. IM2DeepMulti)."""
        # Create mock model that returns tuple
        model = MagicMock()
        model.eval.return_value = None
        # A MagicMock auto-vivifies any unset attribute (truthy by default),
        # unlike a real IM2DeepMulti instance, which simply has no
        # use_heteroscedastic_head attribute at all - set it explicitly so
        # this mock matches that real behaviour (getattr(..., False)).
        model.use_heteroscedastic_head = False
        model.return_value = (
            torch.tensor([450.0, 520.0]),
            torch.tensor([452.0, 524.0]),
        )

        mock_data = [
            ([torch.randn(2, 10), torch.randn(2, 5)], torch.zeros(2)),
        ]

        with patch("im2deep._model_ops.track", return_value=mock_data):
            predictions = _model_ops._predict_loop(
                model=model,
                data_loader=mock_data,  # type: ignore[unknown-arg]
                device="cpu",
            )

            # Both conformer predictions are stacked side by side, not one discarded.
            assert predictions.shape == (2, 2)
            assert torch.equal(predictions[:, 0], torch.tensor([450.0, 520.0]))
            assert torch.equal(predictions[:, 1], torch.tensor([452.0, 524.0]))

    def test_predict_loop_heteroscedastic_head_discards_b(self):
        """Step 8.5: (mu, b) is a different kind of tuple than IM2DeepMulti's two
        conformer predictions - only mu should reach the returned predictions."""
        model = MagicMock()
        model.eval.return_value = None
        model.use_heteroscedastic_head = True
        model.return_value = (
            torch.tensor([450.0, 520.0]),  # mu
            torch.tensor([12.0, 8.0]),  # b
        )

        mock_data = [
            ([torch.randn(2, 10), torch.randn(2, 5)], torch.zeros(2)),
        ]

        with patch("im2deep._model_ops.track", return_value=mock_data):
            predictions = _model_ops._predict_loop(
                model=model,
                data_loader=mock_data,  # type: ignore[unknown-arg]
                device="cpu",
            )

            assert predictions.shape == (2,)
            assert torch.equal(predictions, torch.tensor([450.0, 520.0]))

    def test_predict_loop_no_grad(self):
        """Test that prediction loop uses no_grad context."""
        model = torch.nn.Linear(10, 1)
        model.eval()

        # Create data in the format expected by _predict_loop
        # Each batch should be ([features], targets) where features is a list
        mock_data = [
            ([torch.randn(2, 10)], torch.randn(2, 1)),
            ([torch.randn(2, 10)], torch.randn(2, 1)),
        ]

        # Mock track to return our mock data
        with patch("im2deep._model_ops.track", return_value=mock_data):
            predictions = _model_ops._predict_loop(
                model=model,
                data_loader=mock_data,  # type: ignore[unknown-arg]
                device="cpu",
            )

            assert not predictions.requires_grad


class TestGetArchitecture:
    """Tests for _get_architecture function."""

    @patch("im2deep._model_ops.IM2Deep")
    def test_get_architecture_single(self, mock_im2deep):
        """Test getting single-output architecture."""
        arch = _model_ops._get_architecture(multi=False)
        # Should import IM2Deep
        assert arch is mock_im2deep

    @patch("im2deep._model_ops.IM2DeepMultiTransfer")
    def test_get_architecture_multi(self, mock_multi):
        """Test getting multi-output architecture."""
        arch = _model_ops._get_architecture(multi=True)
        # Should import IM2DeepMultiTransfer
        assert arch is mock_multi


class TestGetModelConfig:
    """Tests for _get_model_config function."""

    def test_get_model_config_single(self):
        """Test getting single-output model config."""
        config = _model_ops._get_model_config(multi=False)
        assert isinstance(config, dict)

    def test_get_model_config_multi(self):
        """Test getting multi-output model config."""
        config = _model_ops._get_model_config(multi=True)
        assert isinstance(config, dict)


class TestGetLossFunction:
    """Tests for _get_loss_function function."""

    def test_get_loss_function_single(self):
        """Test getting single-output loss function."""
        loss = _model_ops._get_loss_function(multi=False)
        assert isinstance(loss, torch.nn.modules.loss._Loss)  # type: ignore[unresolved-attr]

    @patch("im2deep._model_ops.FlexibleLossSorted")
    def test_get_loss_function_multi(self, mock_loss):
        """Test getting multi-output loss function."""
        mock_instance = MagicMock()
        mock_loss.return_value = mock_instance
        loss = _model_ops._get_loss_function(multi=True)
        assert loss is mock_instance


class TestCheckpointInspection:
    """Tests for reading a checkpoint's own configuration and architecture."""

    def test_bundled_checkpoint_falls_back_to_default_config(self):
        """The bundled checkpoints record no config, so defaults apply."""
        from im2deep.constants import DEFAULT_CONFIG, DEFAULT_MODEL

        config = _model_ops.read_checkpoint_config(DEFAULT_MODEL)

        assert config == DEFAULT_CONFIG

    def test_bundled_checkpoint_is_not_a_transfer_model(self):
        """The bundled single-conformer checkpoint loads as plain IM2Deep."""
        from im2deep._architectures.im2deep_single import IM2Deep
        from im2deep.constants import DEFAULT_MODEL

        assert _model_ops.read_checkpoint_architecture(DEFAULT_MODEL) is IM2Deep

    def test_recorded_config_is_preferred(self, tmp_path):
        """A checkpoint that records a config is read back with it."""
        checkpoint_path = tmp_path / "recorded.ckpt"
        torch.save(
            {"state_dict": {}, "hyper_parameters": {"config": {"Global_features": 72}}},
            checkpoint_path,
        )

        config = _model_ops.read_checkpoint_config(checkpoint_path)

        assert config["Global_features"] == 72

    def test_transfer_checkpoint_detected_from_state_dict(self, tmp_path):
        """A state_dict with backbone.* keys means a transfer model."""
        from im2deep._architectures.im2deep_single import IM2DeepTransfer

        checkpoint_path = tmp_path / "transfer.ckpt"
        torch.save(
            {"state_dict": {"backbone.ConvGlobal.0.linear.weight": torch.zeros(1)}},
            checkpoint_path,
        )

        assert _model_ops.read_checkpoint_architecture(checkpoint_path) is IM2DeepTransfer

    def test_non_checkpoint_falls_back_to_default(self):
        """A model object rather than a path returns the package default."""
        from im2deep.constants import DEFAULT_CONFIG

        assert _model_ops.read_checkpoint_config(None) == DEFAULT_CONFIG

    def test_missing_file_falls_back_to_default(self, tmp_path):
        """An unreadable path does not raise here; loading reports it later."""
        from im2deep.constants import DEFAULT_CONFIG

        config = _model_ops.read_checkpoint_config(tmp_path / "missing.ckpt")

        assert config == DEFAULT_CONFIG


class TestWandbLogger:
    """Tests for Weights & Biases logger setup."""

    def test_returns_none_when_disabled(self):
        """No logger is built when wandb is not enabled."""
        config = {"wandb": {"enabled": False}, "model_name": "m"}

        assert _model_ops._setup_logger(config, MagicMock()) is None

    def test_returns_none_when_absent(self):
        """A config with no wandb block does not log."""
        assert _model_ops._setup_logger({"model_name": "m"}, MagicMock()) is None

    def test_run_is_named_after_the_model(self):
        """Runs are named so a set of variants is distinguishable."""
        config = {
            "wandb": {"enabled": True, "project_name": "proj"},
            "model_name": "shift-variant",
            "epochs": 3,
        }

        with patch("lightning.pytorch.loggers.WandbLogger") as mock_logger:
            _model_ops._setup_logger(config, MagicMock())

        assert mock_logger.call_args.kwargs["name"] == "shift-variant"
        assert mock_logger.call_args.kwargs["project"] == "proj"

    def test_explicit_run_name_wins(self):
        """An explicit run name overrides the model name."""
        config = {
            "wandb": {"enabled": True, "name": "explicit"},
            "model_name": "shift-variant",
        }

        with patch("lightning.pytorch.loggers.WandbLogger") as mock_logger:
            _model_ops._setup_logger(config, MagicMock())

        assert mock_logger.call_args.kwargs["name"] == "explicit"

    def test_config_is_logged_without_the_wandb_block(self):
        """The training config is recorded with the run, so runs compare."""
        config = {
            "wandb": {"enabled": True, "project_name": "proj"},
            "model_name": "m",
            "epochs": 3,
            "Global_features": 72,
        }

        with patch("lightning.pytorch.loggers.WandbLogger") as mock_logger:
            _model_ops._setup_logger(config, MagicMock())

        logged = mock_logger.return_value.log_hyperparams.call_args.args[0]
        assert logged["epochs"] == 3
        assert logged["Global_features"] == 72
        assert "wandb" not in logged


class TestFitScalers:
    """Tests for the standardisation fitting helper."""

    def test_disabled_by_default(self, sample_training_df):
        """Neither scaler is fit when both config flags are off."""
        dataset = build_training_dataset(sample_training_df, dict(DEFAULT_TRAINING_CONFIG))

        result = _model_ops._fit_scalers(dataset, dict(DEFAULT_TRAINING_CONFIG))

        assert result == (None, None, None, None)

    def test_target_only(self, sample_training_df):
        """Target mean/std are fit from the raw CCS distribution; global stays off."""
        dataset = build_training_dataset(sample_training_df, dict(DEFAULT_TRAINING_CONFIG))
        config = {**DEFAULT_TRAINING_CONFIG, "standardize_target": True}

        global_mean, global_std, target_mean, target_std = _model_ops._fit_scalers(dataset, config)

        expected = torch.as_tensor(dataset.target_retention_times, dtype=torch.float32)
        assert global_mean is None and global_std is None
        assert target_mean == pytest.approx(expected.mean().item(), rel=1e-4)
        assert target_std == pytest.approx(expected.std().item(), rel=1e-4)

    def test_global_features_only(self, sample_training_df):
        """Global-feature mean/std are fit per feature; target stays off."""
        dataset = build_training_dataset(sample_training_df, dict(DEFAULT_TRAINING_CONFIG))
        config = {**DEFAULT_TRAINING_CONFIG, "standardize_global_features": True, "num_workers": 0}

        global_mean, global_std, target_mean, target_std = _model_ops._fit_scalers(dataset, config)

        assert target_mean is None and target_std is None
        n_features = DEFAULT_TRAINING_CONFIG["Global_features"]
        assert global_mean.shape == (n_features,)
        assert global_std.shape == (n_features,)
        assert (global_std > 0).all()  # constant columns are guarded to 1.0, never left at 0

    def test_degenerate_target_std_guarded(self):
        """All-equal targets would otherwise fit a std of 0 and divide by zero later."""

        class _FakeDataset:
            target_retention_times = [500.0, 500.0, 500.0]

        _, _, target_mean, target_std = _model_ops._fit_scalers(
            _FakeDataset(), {"standardize_target": True}
        )

        assert target_mean == pytest.approx(500.0)
        assert target_std == 1.0


class TestChargeBalancedWeights:
    """Tests for the charge-balanced WeightedRandomSampler helper (step 8.3)."""

    def test_weights_proportional_to_inverse_sqrt_charge_count(self, sample_training_df):
        """sample_training_df repeats each of 60 sequences at charge 2 and charge 3: 60/60 split."""
        dataset = build_training_dataset(sample_training_df, dict(DEFAULT_TRAINING_CONFIG))

        weights = _model_ops._charge_balanced_weights(dataset)

        assert weights.shape == (len(dataset),)
        # Equal charge counts (60 each) -> equal weights across the board.
        assert torch.allclose(weights, weights[0] * torch.ones_like(weights))

    def test_rare_charge_gets_a_larger_weight(self):
        """A charge seen once should get sqrt(n_common) times the weight of a common one."""

        class _FakeDataset:
            peptidoforms = ["ACDEFGHIK/2"] * 8 + ["ACDEFGHIK/5"] * 2 + ["ACDEFGHIK/6"] * 1

        weights = _model_ops._charge_balanced_weights(_FakeDataset())

        w_z2, w_z5, w_z6 = weights[0].item(), weights[8].item(), weights[10].item()
        assert w_z6 > w_z5 > w_z2
        assert w_z6 == pytest.approx(1.0)  # 1/sqrt(1)
        assert w_z5 == pytest.approx(1 / (2**0.5))
        assert w_z2 == pytest.approx(1 / (8**0.5))

    def test_unwraps_a_subset(self, sample_training_df):
        """A Subset (e.g. from grouped_split) is read correctly, not treated as the full dataset."""
        from torch.utils.data import Subset

        dataset = build_training_dataset(sample_training_df, dict(DEFAULT_TRAINING_CONFIG))
        subset = Subset(dataset, [0, 1, 2])

        weights = _model_ops._charge_balanced_weights(subset)

        assert weights.shape == (3,)


class TestComputePairIds:
    """Step 8.3: per-row sibling-group id and theoretical m/z for the pairwise losses."""

    def test_shapes_and_dtypes(self, sample_training_df):
        dataset = build_training_dataset(sample_training_df, dict(DEFAULT_TRAINING_CONFIG))
        peptidoforms = dataset.peptidoforms

        group_ids, mz = _model_ops._compute_pair_ids(peptidoforms)

        assert group_ids.shape == (len(peptidoforms),)
        assert mz.shape == (len(peptidoforms),)
        assert group_ids.dtype == np.int64
        assert mz.dtype == np.float32

    def test_every_row_has_exactly_one_sibling(self, sample_training_df):
        """sample_training_df repeats each of 60 sequences at charge 2 and 3."""
        dataset = build_training_dataset(sample_training_df, dict(DEFAULT_TRAINING_CONFIG))

        group_ids, _ = _model_ops._compute_pair_ids(dataset.peptidoforms)

        counts = Counter(group_ids.tolist())
        assert len(counts) == 60
        assert set(counts.values()) == {2}

    def test_charge_does_not_change_the_group_id(self, sample_training_df):
        """Two charge states of the same sequence are still one sibling group."""
        dataset = build_training_dataset(sample_training_df, dict(DEFAULT_TRAINING_CONFIG))

        group_ids, _ = _model_ops._compute_pair_ids(dataset.peptidoforms)

        # dataset.peptidoforms interleaves (seq, z=2) then (seq, z=3) per the
        # fixture's construction order.
        assert group_ids[0] == group_ids[1]

    def test_higher_charge_gives_a_smaller_mz_for_the_same_sequence(self, sample_training_df):
        dataset = build_training_dataset(sample_training_df, dict(DEFAULT_TRAINING_CONFIG))

        _, mz = _model_ops._compute_pair_ids(dataset.peptidoforms)

        assert mz[1] < mz[0]  # z=3 (index 1) vs z=2 (index 0) of the same sequence

    def test_matches_psm_utils_directly(self, sample_training_df):
        from psm_utils.peptidoform import Peptidoform

        dataset = build_training_dataset(sample_training_df, dict(DEFAULT_TRAINING_CONFIG))

        _, mz = _model_ops._compute_pair_ids(dataset.peptidoforms)

        expected = Peptidoform(dataset.peptidoforms[0]).theoretical_mz
        assert mz[0] == pytest.approx(expected, rel=1e-5)


class TestPairAugmentedDataset:
    """Step 8.3: wraps a training dataset with per-row group_id/mz, unpacked in training_step."""

    def test_appends_group_id_and_mz_after_the_existing_tuple(self, sample_training_df):
        dataset = build_training_dataset(sample_training_df, dict(DEFAULT_TRAINING_CONFIG))
        group_ids, mz = _model_ops._compute_pair_ids(dataset.peptidoforms)

        wrapped = _model_ops._PairAugmentedDataset(dataset, group_ids, mz)
        base_item = dataset[0]
        wrapped_item = wrapped[0]

        assert len(wrapped_item) == len(base_item) + 2
        for original, returned in zip(base_item, wrapped_item[:-2], strict=True):
            # The target (last element of base_item) is a plain numpy
            # float32 from DeepLC's own DeepLCDataset, not a tensor.
            if isinstance(original, torch.Tensor):
                assert torch.equal(original, returned)
            else:
                assert original == pytest.approx(returned)
        assert wrapped_item[-2].item() == group_ids[0]
        assert wrapped_item[-1].item() == pytest.approx(mz[0])

    def test_length_matches_the_base_dataset(self, sample_training_df):
        dataset = build_training_dataset(sample_training_df, dict(DEFAULT_TRAINING_CONFIG))
        group_ids, mz = _model_ops._compute_pair_ids(dataset.peptidoforms)

        wrapped = _model_ops._PairAugmentedDataset(dataset, group_ids, mz)

        assert len(wrapped) == len(dataset)


class TestSiblingPairBatchSampler:
    """Step 8.3: deliberately places sibling pairs inside otherwise random batches."""

    def test_raises_when_no_sibling_groups_exist(self):
        group_ids = np.arange(100)  # every row its own group

        with pytest.raises(_model_ops.IM2DeepError, match="no sibling groups"):
            _model_ops.SiblingPairBatchSampler(
                group_ids=group_ids, batch_size=16, pairs_per_batch=4, num_batches=5
            )

    def test_yields_the_configured_number_of_batches(self, sample_training_df):
        dataset = build_training_dataset(sample_training_df, dict(DEFAULT_TRAINING_CONFIG))
        group_ids, _ = _model_ops._compute_pair_ids(dataset.peptidoforms)
        sampler = _model_ops.SiblingPairBatchSampler(
            group_ids=group_ids, batch_size=16, pairs_per_batch=4, num_batches=3
        )

        batches = list(sampler)

        assert len(batches) == 3
        assert len(sampler) == 3
        assert all(len(batch) == 16 for batch in batches)

    def test_every_batch_contains_at_least_the_deliberately_placed_pairs(self, sample_training_df):
        """Every group in this fixture has exactly one sibling, so every drawn pair is real."""
        dataset = build_training_dataset(sample_training_df, dict(DEFAULT_TRAINING_CONFIG))
        group_ids, _ = _model_ops._compute_pair_ids(dataset.peptidoforms)
        sampler = _model_ops.SiblingPairBatchSampler(
            group_ids=group_ids, batch_size=16, pairs_per_batch=4, num_batches=10
        )

        for batch in sampler:
            groups_in_batch = Counter(int(group_ids[i]) for i in batch)
            # Each of the 4 deliberately-placed pairs contributes 2 indices
            # to some group that ends up with count>=2 in the batch - true
            # even if two draws happen to land on the same group (that group
            # then has count 4, not 2, but still counts all 4 slots here).
            # Counting matched *slots* rather than matched *groups* avoids
            # undercounting that coincidence, which plain groups>=2 counting
            # does not.
            matched_slots = sum(count for count in groups_in_batch.values() if count >= 2)
            assert matched_slots >= 2 * 4

    def test_indices_stay_within_range(self, sample_training_df):
        dataset = build_training_dataset(sample_training_df, dict(DEFAULT_TRAINING_CONFIG))
        group_ids, _ = _model_ops._compute_pair_ids(dataset.peptidoforms)
        sampler = _model_ops.SiblingPairBatchSampler(
            group_ids=group_ids, batch_size=16, pairs_per_batch=4, num_batches=5
        )

        for batch in sampler:
            assert all(0 <= i < len(dataset) for i in batch)
