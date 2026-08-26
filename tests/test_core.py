"""Tests for core module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from im2deep import core
from im2deep.exceptions import IM2DeepError


class TestPredict:
    """Tests for predict function."""

    @patch("im2deep.core._model_ops.predict")
    @patch("im2deep.core.DeepLCDataset")
    def test_predict_basic(self, mock_dataset, mock_predict, sample_psm_list):
        """Test basic prediction."""
        mock_dataset.from_psm_list.return_value = MagicMock()
        mock_predict.return_value = torch.tensor([450.0, 520.0, 480.0])

        predictions = core.predict(sample_psm_list)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 3
        mock_dataset.from_psm_list.assert_called_once()
        mock_predict.assert_called_once()

    @patch("im2deep.core._model_ops.predict")
    @patch("im2deep.core.DeepLCDataset")
    def test_predict_with_model(self, mock_dataset, mock_predict, sample_psm_list):
        """Test prediction with custom model."""
        mock_dataset.from_psm_list.return_value = MagicMock()
        mock_predict.return_value = torch.tensor([450.0, 520.0, 480.0])

        custom_model = "custom_model.ckpt"
        predictions = core.predict(sample_psm_list, model=custom_model)

        assert isinstance(predictions, np.ndarray)
        # Check that custom model was passed
        call_kwargs = mock_predict.call_args[1]
        assert "model" in call_kwargs

    @patch("im2deep.core._model_ops.predict")
    @patch("im2deep.core.DeepLCDataset")
    def test_predict_multi(self, mock_dataset, mock_predict, sample_psm_list):
        """Test prediction with multi-output model."""
        mock_dataset.from_psm_list.return_value = MagicMock()
        mock_predict.return_value = torch.tensor([[450.0, 452.0], [520.0, 524.0], [480.0, 482.0]])

        predictions = core.predict(sample_psm_list, multi=True)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (3, 2)

    @patch("im2deep.core._model_ops.predict")
    @patch("im2deep.core.DeepLCDataset")
    def test_predict_with_kwargs(self, mock_dataset, mock_predict, sample_psm_list):
        """Test prediction with additional kwargs."""
        mock_dataset.from_psm_list.return_value = MagicMock()
        mock_predict.return_value = torch.tensor([450.0, 520.0, 480.0])

        predictions = core.predict(
            sample_psm_list,
            predict_kwargs={"batch_size": 256, "device": "cpu"},
        )

        assert isinstance(predictions, np.ndarray)
        call_kwargs = mock_predict.call_args[1]
        assert call_kwargs["batch_size"] == 256
        assert call_kwargs["device"] == "cpu"

    def test_predict_invalid_psm_list(self):
        """Test prediction with invalid PSMList."""
        with pytest.raises(IM2DeepError):
            core.predict([1, 2, 3])  # type: ignore[invalid-arg]


class TestPredictAndCalibrate:
    """Tests for predict_and_calibrate function."""

    @patch("im2deep.core.predict")
    @patch("im2deep.core.LinearCCSCalibration")
    def test_predict_and_calibrate_basic(
        self,
        mock_calibration_class,
        mock_predict,
        sample_psm_list,
        sample_psm_list_with_ccs,
    ):
        """Test basic predict and calibrate."""
        # Mock predict to return predictions
        mock_predict.return_value = np.array([448.0, 516.0, 478.0])

        # Mock calibration
        mock_calibration = MagicMock()
        mock_calibration.is_fitted = False
        mock_calibration.transform.return_value = np.array([450.0, 520.0, 480.0])
        mock_calibration_class.return_value = mock_calibration

        predictions = core.predict_and_calibrate(
            psm_list=sample_psm_list,
            psm_list_cal=sample_psm_list_with_ccs,
        )

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 3
        mock_calibration.fit.assert_called_once()
        mock_calibration.transform.assert_called_once()

    @patch("im2deep.core.predict")
    @patch("im2deep.core.LinearCCSCalibration")
    def test_predict_and_calibrate_with_reference(
        self,
        mock_calibration_class,
        mock_predict,
        sample_psm_list,
        sample_psm_list_with_ccs,
        sample_reference_psm_list,
    ):
        """Test predict and calibrate with reference PSMList."""
        mock_predict.return_value = np.array([448.0, 516.0, 478.0])

        mock_calibration = MagicMock()
        mock_calibration.is_fitted = False
        mock_calibration.transform.return_value = np.array([450.0, 520.0, 480.0])
        mock_calibration_class.return_value = mock_calibration

        predictions = core.predict_and_calibrate(
            psm_list=sample_psm_list,
            psm_list_cal=sample_psm_list_with_ccs,
            psm_list_reference=sample_reference_psm_list,
        )

        assert isinstance(predictions, np.ndarray)

    @patch("im2deep.core.predict")
    def test_predict_and_calibrate_custom_calibration(
        self,
        mock_predict,
        sample_psm_list,
        sample_psm_list_with_ccs,
    ):
        """Test predict and calibrate with custom calibration."""
        from im2deep.calibration import Calibration

        # Create a real mock class that inherits from Calibration
        class MockCalibration(Calibration):
            def __init__(self):
                self._is_fitted = False
                self.fit_called = False
                self.transform_called = False

            @property
            def is_fitted(self):
                return self._is_fitted

            def fit(self, *args, **kwargs):
                self._is_fitted = True
                self.fit_called = True

            def transform(self, *args, **kwargs):
                self.transform_called = True
                return np.array([450.0, 520.0, 480.0])

        mock_predict.return_value = np.array([448.0, 516.0, 478.0])

        custom_calibration = MockCalibration()

        predictions = core.predict_and_calibrate(
            psm_list=sample_psm_list,
            psm_list_cal=sample_psm_list_with_ccs,
            calibration=custom_calibration,  # type: ignore[invalid-arg]
        )

        assert isinstance(predictions, np.ndarray)
        assert custom_calibration.fit_called
        assert custom_calibration.transform_called

    @patch("im2deep.core.predict")
    def test_predict_and_calibrate_multi_output(
        self, mock_predict, sample_psm_list, sample_psm_list_with_ccs
    ):
        """Test predict and calibrate with multi-output predictions."""
        mock_predict.return_value = np.array([[448.0, 452.0], [516.0, 524.0], [478.0, 482.0]])

        with patch("im2deep.core.LinearCCSCalibration") as mock_cal_class:
            mock_calibration = MagicMock()
            mock_calibration.is_fitted = False
            mock_calibration.transform.return_value = np.array(
                [[450.0, 454.0], [520.0, 528.0], [480.0, 484.0]]
            )
            mock_cal_class.return_value = mock_calibration

            predictions = core.predict_and_calibrate(
                psm_list=sample_psm_list,
                psm_list_cal=sample_psm_list_with_ccs,
                multi=True,
            )

            assert isinstance(predictions, np.ndarray)
            assert predictions.shape == (3, 2)

    def test_predict_and_calibrate_invalid_cal_psm(self, sample_psm_list):
        """Test that calibration PSMList must have CCS values."""
        with pytest.raises(IM2DeepError):
            core.predict_and_calibrate(
                psm_list=sample_psm_list,
                psm_list_cal=sample_psm_list,  # Missing CCS values
            )


class TestResolveConfig:
    """Tests for training configuration merging."""

    def test_defaults_when_nothing_given(self):
        """With no config, the training defaults are used as-is."""
        config = core._resolve_config(None, None)
        assert config["epochs"] == 100
        assert config["monitor"] == "Validation MAE"
        assert config["Global_features"] == 60

    def test_training_kwargs_override_defaults(self):
        """Explicit keyword overrides win over the defaults."""
        config = core._resolve_config(None, {"epochs": 7})
        assert config["epochs"] == 7

    def test_training_kwargs_override_config(self):
        """An explicit argument beats a configuration file."""
        config = core._resolve_config({"epochs": 5}, {"epochs": 7})
        assert config["epochs"] == 7

    def test_config_overrides_base(self):
        """A configuration file beats a caller's task-specific defaults."""
        config = core._resolve_config({"epochs": 5}, None, base_overrides={"epochs": 50})
        assert config["epochs"] == 5

    def test_base_overrides_beat_defaults(self):
        """Task-specific defaults still beat the package defaults."""
        config = core._resolve_config(None, None, base_overrides={"epochs": 50})
        assert config["epochs"] == 50

    def test_unwraps_model_params_block(self):
        """im2deeptrainer-style configs with a model_params block still work."""
        config = core._resolve_config(
            {"epochs": 5, "model_params": {"batch_size": 32, "learning_rate": 0.01}}, None
        )
        assert config["batch_size"] == 32
        assert config["learning_rate"] == 0.01
        assert config["epochs"] == 5
        assert "model_params" not in config

    def test_reads_json_file(self, tmp_path):
        """A path to a JSON configuration file is read."""
        config_path = tmp_path / "config.json"
        config_path.write_text('{"epochs": 3, "model_params": {"batch_size": 8}}')

        config = core._resolve_config(config_path, None)

        assert config["epochs"] == 3
        assert config["batch_size"] == 8

    def test_rejects_non_dict(self):
        """A config that is neither a dict nor a path is a type error."""
        with pytest.raises(TypeError, match="config must be"):
            core._resolve_config(["not", "a", "dict"], None)


class TestTrain:
    """Tests for the train function."""

    @patch("im2deep.core._model_ops.train")
    @patch("im2deep.core._data.build_training_dataset")
    @patch("im2deep.core._data.grouped_split")
    def test_train_basic(self, mock_split, mock_build, mock_train, sample_training_df, tmp_path):
        """Training builds a dataset, splits it and saves a checkpoint."""
        mock_build.return_value = MagicMock()
        mock_split.return_value = (MagicMock(), MagicMock())
        trainer, model = MagicMock(), MagicMock()
        mock_train.return_value = (trainer, model, None)
        save_path = tmp_path / "model.ckpt"

        result = core.train(sample_training_df, save_path, training_kwargs={"epochs": 2})

        assert result is model
        mock_build.assert_called_once()
        mock_split.assert_called_once()
        # No best-checkpoint path was produced, so the trainer writes it.
        trainer.save_checkpoint.assert_called_once_with(save_path)

    @patch("im2deep.core._model_ops.train")
    @patch("im2deep.core._data.build_training_dataset")
    @patch("im2deep.core._data.grouped_split")
    def test_train_copies_best_checkpoint(
        self, mock_split, mock_build, mock_train, sample_training_df, tmp_path
    ):
        """The best checkpoint is copied, preserving the Lightning format."""
        best_path = tmp_path / "best.ckpt"
        best_path.write_bytes(b"checkpoint-contents")
        mock_build.return_value = MagicMock()
        mock_split.return_value = (MagicMock(), MagicMock())
        mock_train.return_value = (MagicMock(), MagicMock(), str(best_path))
        save_path = tmp_path / "out" / "model.ckpt"

        core.train(sample_training_df, save_path)

        assert save_path.read_bytes() == b"checkpoint-contents"

    @patch("im2deep.core._model_ops.train")
    @patch("im2deep.core._data.build_training_dataset")
    @patch("im2deep.core._data.grouped_split")
    def test_training_kwargs_reach_the_trainer(
        self, mock_split, mock_build, mock_train, sample_training_df, tmp_path
    ):
        """Overrides passed by the caller arrive in the training config."""
        mock_build.return_value = MagicMock()
        mock_split.return_value = (MagicMock(), MagicMock())
        mock_train.return_value = (MagicMock(), MagicMock(), None)

        core.train(sample_training_df, tmp_path / "model.ckpt", training_kwargs={"epochs": 3})

        assert mock_train.call_args.kwargs["config"]["epochs"] == 3

    @patch("im2deep.core._model_ops.train")
    @patch("im2deep.core._data.build_training_dataset")
    @patch("im2deep.core._data.grouped_split")
    def test_explicit_validation_set_skips_split(
        self, mock_split, mock_build, mock_train, sample_training_df, tmp_path
    ):
        """An explicit validation set means no split is taken."""
        mock_build.return_value = MagicMock()
        mock_train.return_value = (MagicMock(), MagicMock(), None)

        core.train(
            sample_training_df,
            tmp_path / "model.ckpt",
            validation_psm_list=sample_training_df,
        )

        mock_split.assert_not_called()
        assert mock_build.call_count == 2

    def test_missing_target_raises(self, sample_training_df, tmp_path):
        """Training data with no CCS column is rejected before any training."""
        with pytest.raises(IM2DeepError, match="target CCS column"):
            core.train(sample_training_df.drop(columns=["ccs"]), tmp_path / "model.ckpt")


class TestFinetune:
    """Tests for the finetune function."""

    @patch("im2deep.core.train")
    def test_finetune_sets_backbone(self, mock_train, sample_training_df, tmp_path):
        """Fine-tuning points the transfer model at the bundled backbone."""
        core.finetune(sample_training_df, tmp_path / "model.ckpt")

        config = mock_train.call_args.kwargs["config"]
        assert config["backbone_SD_path"].endswith("IM2DeepUni.ckpt")
        assert config["freeze_epochs"] == 5

    @patch("im2deep.core.train")
    def test_finetune_uses_given_backbone(self, mock_train, sample_training_df, tmp_path):
        """An explicit backbone is used instead of the default."""
        backbone = tmp_path / "backbone.ckpt"

        core.finetune(sample_training_df, tmp_path / "model.ckpt", model=backbone)

        assert mock_train.call_args.kwargs["config"]["backbone_SD_path"] == str(backbone)

    @patch("im2deep.core.train")
    def test_user_kwargs_beat_finetune_defaults(self, mock_train, sample_training_df, tmp_path):
        """A caller's epochs win over the fine-tuning default of 50."""
        core.finetune(sample_training_df, tmp_path / "model.ckpt", training_kwargs={"epochs": 9})

        assert mock_train.call_args.kwargs["config"]["epochs"] == 9

    @patch("im2deep.core.train")
    def test_backbone_architecture_is_carried_across(
        self, mock_train, sample_training_df, tmp_path
    ):
        """The transfer model inherits the backbone's architecture width."""
        core.finetune(sample_training_df, tmp_path / "model.ckpt")

        config = mock_train.call_args.kwargs["config"]
        assert config["Global_units"] == 16
        assert config["Concat_units"] == 128


class TestWandbConfig:
    """Tests for Weights & Biases configuration merging."""

    def test_disabled_by_default(self):
        """Training does not log to wandb unless asked to."""
        assert core._resolve_config(None, None)["wandb"]["enabled"] is False

    def test_enabling_keeps_project_from_config_file(self):
        """--wandb must not discard a project name set in a config file."""
        config = core._resolve_config(
            {"wandb": {"project_name": "my-project"}}, {"wandb": {"enabled": True}}
        )

        assert config["wandb"]["enabled"] is True
        assert config["wandb"]["project_name"] == "my-project"

    def test_run_name_overrides_config_file(self):
        """An explicit run name still wins over the file."""
        config = core._resolve_config(
            {"wandb": {"project_name": "p", "name": "from-file"}},
            {"wandb": {"name": "from-cli"}},
        )

        assert config["wandb"]["name"] == "from-cli"
        assert config["wandb"]["project_name"] == "p"

    def test_non_wandb_keys_still_replace(self):
        """The nested merge is specific to wandb, not general."""
        config = core._resolve_config({"epochs": 5}, {"epochs": 7})

        assert config["epochs"] == 7
