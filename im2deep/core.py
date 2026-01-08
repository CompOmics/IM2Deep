"""IM2Deep core functionality."""

from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path

import numpy as np
from psm_utils.psm_list import PSMList
import torch
from deeplc._data import DeepLCDataset
from deeplc.calibration import Calibration

from im2deep.utils import _validate_psm_list

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_NAME = "IM2DeepUni.ckpt"
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "TIMS" / DEFAULT_MODEL_NAME
DEFAULT_MULTI_MODEL_NAME = "IM2DeepMulti.ckpt"
DEFAULT_MULTI_MODEL = (
    Path(__file__).resolve().parent / "models" / "TIMS" / DEFAULT_MULTI_MODEL_NAME
)


def predict(
    psm_list: PSMList,
    model: torch.nn.Module | PathLike | str | None = None,
    multi=False,
    predict_kwargs: dict | None = None,
) -> np.ndarray:
    """
    Predict CCS values for a list of PSMs using a trained model.

    Parameters
    ----------
    psm_list
        List of PSMs to predict CCS values for.
    model
        Trained model or path to model file. If None, the default IM2Deep model is used.
    predict_kwargs
        Additional keyword arguments to pass to the prediction function.

    Returns
    -------
    np.ndarray
        CCS predictions.

    """
    LOGGER.info("Predicting CCS values using IM2Deep.")
    _validate_psm_list(psm_list)
    return _model_ops.predict(
        model=model or DEFAULT_MODEL if not multi else DEFAULT_MULTI_MODEL,
        data=DeepLCDataset.from_psm_list(psm_list, add_ccs_features=True),
        **(predict_kwargs or {}),
        # TODO: check if "backbone" argument is needed
    ).numpy()


def calibrate_and_predict(
    psm_list: PSMList,
    psm_list_cal: PSMList,
    psm_list_reference: PSMList,
    model: torch.nn.Module | PathLike | str | None = None,
    calibration: Calibration | None = None,
    multi: bool = False,
    predict_kwargs: dict | None = None,
) -> np.ndarray:
    """
    Calibrate and predict CCS values for a list of PSMs using a reference PSM list.

    Parameters
    ----------
    psm_list
        List of PSMs to predict CCS values for.
    psm_list_reference
        Reference list of PSMs for calibration.
    model
        Trained model or path to model file. If None, the default IM2Deep model is used.
    calibration
        Calibration object to use for calibration. If None, #TODO: name is applied.
    predict_kwargs
        Additional keyword arguments to pass to the prediction function.

    Returns
    -------
    np.ndarray
        Calibrated CCS predictions.

    """
    # Predict initial CCS values
    LOGGER.info("Predicting uncalibrated CCS values...")
    _validate_psm_list(psm_list)
    _validate_psm_list(psm_list_cal)
    _validate_psm_list(psm_list_reference)
    predicted_ccs = predict(
        psm_list,
        model=model,
        multi=multi,
        predict_kwargs=predict_kwargs,
    )

    # Perform calibration
    if calibration is None:
        LOGGER.info("No calibration provided, using #TODO: name by default.")
        calibration = #TODO: name()
    elif not isinstance(calibration, Calibration):
        raise TypeError(
            f"Calibration must be an instance of Calibration, got {type(calibration)} instead."
        )
    
    if not calibration.is_fitted:
        LOGGER.info("Fitting calibration...")
        # Below is probably slower than np.array(psm_list_reference["ion_mobility"], 
        # dtype=np.float32), but it makes more sense to work with CCS? 
        target_ccs_cal = np.array(
            [float(psm.metadata["CCS"]) for psm in psm_list_reference],
            dtype=np.float32,
        ) 
        source_ccs_cal = np.array(
            [float(psm.metadata["CCS"]) for psm in psm_list_cal],
            dtype=np.float32,
        )
        calibration.fit(source_ccs_cal, target_ccs_cal)
    else:
        LOGGER.info("Calibration is already fitted, skipping fitting step.")
    
    # Apply calibration to predictions
    calibrated_ccs = calibration.transform(predicted_ccs)

    return calibrated_ccs

def train(
    psm_list,
    model_save_path,
    training_kwargs=None,
):
    """
    Train a new IM2Deep model using the provided PSM list.

    Parameters
    ----------
    psm_list
        List of PSMs to use for training.
    model_save_path
        Path to save the trained model.
    training_kwargs
        Additional keyword arguments to pass to the training function.

    Returns
    -------
    None

    """
    raise NotImplementedError(
        "Training functionality is not yet implemented for IM2Deep. Use the IM2DeepTrainer package instead."
    )

# TODO: finetune and finetune_and_predict functions?