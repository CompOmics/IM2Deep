"""IM2Deep core functionality."""

from __future__ import annotations

import json
import logging
import shutil
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from deeplc.data import DeepLCDataset
from psm_utils.psm_list import PSMList

from im2deep import _data, _model_ops
from im2deep._data import _featurisation_kwargs
from im2deep._io_helpers import validate_psm_list
from im2deep.calibration import Calibration, LinearCCSCalibration
from im2deep.constants import (
    DEFAULT_MODEL,
    DEFAULT_MULTI_MODEL,
    DEFAULT_TRAINING_CONFIG,
)

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
    model = model or (DEFAULT_MODEL if not multi else DEFAULT_MULTI_MODEL)

    # Featurise the way this particular checkpoint was trained. Checkpoints
    # written by train() record their flags; the bundled ones predate that and
    # fall back to the package defaults, which is what they were trained with.
    model_config = _model_ops.read_checkpoint_config(model, multi=multi)
    featurisation = _featurisation_kwargs(model_config)

    return _model_ops.predict(
        model=model,
        data=DeepLCDataset.from_psm_list(psm_list, **featurisation),
        multi=multi,
        **(predict_kwargs or {}),
        # TODO: check if "backbone" argument is needed for multi
    ).numpy()


def predict_and_calibrate(
    psm_list: PSMList,
    psm_list_cal: PSMList,
    psm_list_reference: PSMList | None = None,
    model: torch.nn.Module | PathLike | str | None = None,
    calibration: LinearCCSCalibration | None = None,
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
        if psm.metadata is None:
            psm.metadata = {}
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
            psm_df_reference,  # None is fine; fit() loads the default reference if needed
            multi=multi,
        )
    else:
        LOGGER.info("Calibration is already fitted, skipping fitting step.")

    # Apply calibration to predictions
    predicted_ccs_calibrated = calibration.transform(psm_df)

    # Return as-is (already numpy array, may be object array for multiconformer)
    return predicted_ccs_calibrated


def _resolve_config(
    config: dict | PathLike | str | None,
    training_kwargs: dict | None,
    base_overrides: dict | None = None,
) -> dict:
    """
    Merge a user configuration over the training defaults.

    Accepts a dict or a path to a JSON file. A ``model_params`` block is
    unwrapped and merged, so configuration files written for the standalone
    ``im2deeptrainer`` package remain usable.

    Precedence, weakest first:
    :data:`~im2deep.constants.DEFAULT_TRAINING_CONFIG`, ``base_overrides``,
    ``config``, ``training_kwargs``. So an explicit argument always wins over a
    configuration file, and a configuration file always wins over a caller's
    task-specific defaults.

    Parameters
    ----------
    config
        Configuration dict, or path to a JSON configuration file.
    training_kwargs
        Keyword overrides applied on top.
    base_overrides
        Task-specific defaults applied under the user's own settings, used by
        :func:`finetune` to carry the backbone's architecture across.

    Returns
    -------
    dict
        Merged configuration.

    """
    resolved = dict(DEFAULT_TRAINING_CONFIG)

    if base_overrides:
        resolved.update(base_overrides)

    if config is not None:
        if isinstance(config, (str, Path, PathLike)):
            LOGGER.info(f"Reading training configuration from {config}")
            with open(config) as config_file:
                config = json.load(config_file)
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dict or a path, got {type(config).__name__}.")
        # im2deeptrainer nested the model hyperparameters under "model_params";
        # accept both that shape and a flat dict.
        merged = {**config, **config.get("model_params", {})}
        merged.pop("model_params", None)
        _update_config(resolved, merged)

    if training_kwargs:
        _update_config(resolved, training_kwargs)

    return resolved


def _update_config(resolved: dict, overrides: dict) -> None:
    """
    Apply overrides in place, merging the nested ``wandb`` block rather than
    replacing it.

    ``wandb`` is the only nested key in the training config. A plain
    ``dict.update`` would mean that passing ``--wandb`` on the command line
    silently discarded a project name set in a configuration file.
    """
    for key, value in overrides.items():
        if key == "wandb" and isinstance(value, dict) and isinstance(resolved.get(key), dict):
            resolved[key] = {**resolved[key], **value}
        else:
            resolved[key] = value


def train(
    psm_list: PSMList | pd.DataFrame | PathLike | str,
    model_save_path: PathLike | str,
    training_kwargs: dict | None = None,
    validation_psm_list: PSMList | pd.DataFrame | PathLike | str | None = None,
    validation_split: float = 0.1,
    config: dict | PathLike | str | None = None,
    output_dir: PathLike | str | None = None,
) -> torch.nn.Module:
    """
    Train a new IM2Deep model.

    Features are built with the same DeepLC featuriser :func:`predict` uses, so
    a model trained here can be used for prediction directly. The resulting
    checkpoint records its own configuration, including the featurisation flags
    it was trained with.

    Parameters
    ----------
    psm_list
        Training data: a :class:`~psm_utils.psm_list.PSMList` carrying a target
        CCS on every PSM, a :class:`pandas.DataFrame`, or a path to a delimited
        file with a ``CCS``/``ccs`` column plus either ``peptidoform`` or
        ``seq``/``modifications``/``charge``.
    model_save_path
        Path to save the trained model checkpoint to.
    training_kwargs
        Configuration overrides, applied on top of ``config``. For example
        ``{"epochs": 50, "batch_size": 1024}``.
    validation_psm_list
        Explicit validation set. If None, a validation set is split off
        ``psm_list``, grouped by stripped sequence.
    validation_split
        Fraction held out for validation when ``validation_psm_list`` is None.
    config
        Configuration dict or path to a JSON configuration file, merged over
        :data:`im2deep.constants.DEFAULT_TRAINING_CONFIG`.
    output_dir
        Directory for Lightning logs and intermediate checkpoints. Defaults to
        the parent directory of ``model_save_path``.

    Returns
    -------
    torch.nn.Module
        The trained model.

    Example
    -------
    >>> from im2deep import train
    >>> model = train("train.csv", "my_model.ckpt", training_kwargs={"epochs": 50})

    """
    resolved_config = _resolve_config(config, training_kwargs)
    model_save_path = Path(model_save_path)
    output_dir = Path(output_dir) if output_dir is not None else model_save_path.parent

    LOGGER.info("Building training dataset...")
    train_dataset = _data.build_training_dataset(psm_list, resolved_config)

    if validation_psm_list is not None:
        LOGGER.info("Building validation dataset...")
        validation_dataset = _data.build_training_dataset(validation_psm_list, resolved_config)
        train_subset, validation_subset = train_dataset, validation_dataset
    else:
        train_subset, validation_subset = _data.grouped_split(
            train_dataset, validation_split, seed=resolved_config.get("seed", 0)
        )

    LOGGER.info("Training IM2Deep model...")
    trainer, model, best_checkpoint_path = _model_ops.train(
        train_dataset=train_subset,
        validation_dataset=validation_subset,
        config=resolved_config,
        output_dir=output_dir,
    )

    # Save a Lightning checkpoint, not a pickled module: that is the format
    # predict() and the bundled models use, and it is what carries the config
    # recorded by save_hyperparameters().
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    if best_checkpoint_path and Path(best_checkpoint_path).resolve() != model_save_path.resolve():
        shutil.copyfile(best_checkpoint_path, model_save_path)
    elif not best_checkpoint_path:
        trainer.save_checkpoint(model_save_path)
    LOGGER.info(f"Trained model saved to {model_save_path}")
    return model


def finetune(
    psm_list: PSMList | pd.DataFrame | PathLike | str,
    model_save_path: PathLike | str,
    model: PathLike | str | None = None,
    training_kwargs: dict | None = None,
    validation_psm_list: PSMList | pd.DataFrame | PathLike | str | None = None,
    validation_split: float = 0.1,
    config: dict | PathLike | str | None = None,
    output_dir: PathLike | str | None = None,
) -> torch.nn.Module:
    """
    Fine-tune an existing IM2Deep model on new data.

    Builds an
    :class:`~im2deep._architectures.im2deep_single.IM2DeepTransfer` on top of
    the given backbone. By default the pretrained feature branches are held
    frozen for the first few epochs while the concatenation head adapts, then
    unfrozen at a reduced learning rate, mirroring how DeepLC fine-tunes.

    Parameters
    ----------
    psm_list
        Fine-tuning data, in any form :func:`train` accepts.
    model_save_path
        Path to save the fine-tuned model checkpoint to.
    model
        Backbone checkpoint. Defaults to the bundled IM2Deep model.
    training_kwargs
        Configuration overrides, applied on top of ``config``. ``freeze_epochs``
        and ``unfreeze_lr_scale`` control the warmup.
    validation_psm_list
        Explicit validation set. If None, one is split off ``psm_list``.
    validation_split
        Fraction held out for validation when ``validation_psm_list`` is None.
    config
        Configuration dict or path to a JSON configuration file.
    output_dir
        Directory for Lightning logs and intermediate checkpoints.

    Returns
    -------
    torch.nn.Module
        The fine-tuned model.

    """
    backbone_path = str(model or DEFAULT_MODEL)

    # A transfer model has to match the checkpoint it loads, so the backbone's
    # own architecture and featurisation come first. These sit *under* the
    # caller's config and training_kwargs, which stay authoritative.
    base_overrides = {
        **_model_ops.read_checkpoint_config(backbone_path),
        "backbone_SD_path": backbone_path,
        "model_name": "IM2DeepFinetuned",
        "epochs": 50,
        "freeze_epochs": 5,
        "unfreeze_lr_scale": 0.1,
    }
    resolved_config = _resolve_config(config, training_kwargs, base_overrides)

    return train(
        psm_list=psm_list,
        model_save_path=model_save_path,
        validation_psm_list=validation_psm_list,
        validation_split=validation_split,
        config=resolved_config,
        output_dir=output_dir,
    )


# TODO: finetune_and_predict function?
