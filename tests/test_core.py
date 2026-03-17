"""Tests for core module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from im2deep import core
from im2deep._exceptions import IM2DeepError


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
            core.predict([1, 2, 3])


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
            calibration=custom_calibration,
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
