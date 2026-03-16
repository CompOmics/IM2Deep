"""IM2Deep core functionality."""

from __future__ import annotations

import logging
from os import PathLike

import numpy as np
import torch
from deeplc.data import DeepLCDataset
from psm_utils.psm_list import PSMList

from im2deep import _model_ops
from im2deep.calibration import Calibration, LinearCCSCalibration
from im2deep.constants import DEFAULT_MODEL, DEFAULT_MULTI_MODEL
from im2deep.utils import validate_psm_list

LOGGER = logging.getLogger(__name__)


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
    psm_list = validate_psm_list(psm_list)
    return _model_ops.predict(
        model=model or DEFAULT_MODEL if not multi else DEFAULT_MULTI_MODEL,
        data=DeepLCDataset.from_psm_list(psm_list, add_ccs_features=True),
        multi=multi,
        **(predict_kwargs or {}),
        # TODO: check if "backbone" argument is needed for multi
    ).numpy()


def predict_and_calibrate(
    psm_list: PSMList,
    psm_list_cal: PSMList,
    psm_list_reference: PSMList | None = None,
    model: torch.nn.Module | PathLike | str | None = None,
    calibration: Calibration | None = None,
    multi: bool = False,
    predict_kwargs: dict | None = None,
    **kwargs,
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
        Calibration object to use for calibration. If None, LinearCCSCalibration is applied.
    predict_kwargs
        Additional keyword arguments to pass to the prediction function.

    Returns
    -------
    np.ndarray
        Calibrated CCS predictions.

    """
    # Predict initial CCS values
    LOGGER.info("Predicting uncalibrated CCS values...")
    psm_list = validate_psm_list(psm_list)
    psm_list_cal = validate_psm_list(psm_list_cal, needs_target=True)

    predicted_ccs = predict(
        psm_list=psm_list,
        model=model,
        multi=multi,
        predict_kwargs=predict_kwargs,
    )

    # Assign the predicted CCS to the PSM metadata
    for idx, psm in enumerate(psm_list):
        psm.metadata["predicted_CCS_uncalibrated"] = predicted_ccs[idx]

    psm_df = psm_list.to_dataframe()
    psm_df_cal = psm_list_cal.to_dataframe()
    if psm_list_reference is not None:
        psm_list_reference = validate_psm_list(psm_list_reference, needs_target=True)
        psm_df_reference = psm_list_reference.to_dataframe()
    else:
        psm_df_reference = None

    # Perform calibration
    if calibration is None:
        LOGGER.info("No calibration provided, using LinearCCSCalibration by default.")
        calibration = LinearCCSCalibration(
            per_charge=kwargs.get("calibrate_per_charge", True),
            use_charge_state=(
                kwargs.get("use_charge_state", 2)
                if not kwargs.get("calibrate_per_charge", True)
                else None
            ),
        )
    elif not isinstance(calibration, Calibration):
        raise TypeError(
            f"Calibration must be an instance of Calibration, got {type(calibration)} instead."
        )

    if not calibration.is_fitted:
        LOGGER.info("Fitting calibration...")
        if any(psm_list_cal["is_decoy"]):
            LOGGER.warning(
                "Calibration PSM list contains decoy PSMs. "
                "These will be ignored during calibration fitting."
            )
        calibration.fit(
            psm_df_cal,
            psm_df_reference,
            multi=multi,
        )
    else:
        LOGGER.info("Calibration is already fitted, skipping fitting step.")

    # Apply calibration to predictions
    predicted_ccs_calibrated = calibration.transform(psm_df)

    # Return as-is (already numpy array, may be object array for multiconformer)
    return predicted_ccs_calibrated


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
