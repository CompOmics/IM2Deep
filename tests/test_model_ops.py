"""Tests for model operations module."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from im2deep import _model_ops


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
        """Test prediction loop with multi-output model."""
        # Create mock model that returns tuple
        model = MagicMock()
        model.eval.return_value = None
        model.return_value = (
            torch.tensor([[450.0], [520.0]]),
            torch.tensor([[452.0], [524.0]]),
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

            assert isinstance(predictions, torch.Tensor)
            # Should stack both outputs

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

        logged = mock_logger.return_value.experiment.config.update.call_args.args[0]
        assert logged["epochs"] == 3
        assert logged["Global_features"] == 72
        assert "wandb" not in logged
