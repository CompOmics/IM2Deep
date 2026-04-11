"""IM2Deep core functionality."""

from __future__ import annotations

import logging
from os import PathLike

import numpy as np
import torch
from deeplc.data import DeepLCDataset
from psm_utils.psm_list import PSMList

from im2deep import _model_ops
from im2deep._io_helpers import validate_psm_list
from im2deep.calibration import Calibration, LinearCCSCalibration
from im2deep.constants import DEFAULT_MODEL, DEFAULT_MULTI_MODEL

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
        model=model or (DEFAULT_MODEL if not multi else DEFAULT_MULTI_MODEL),
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


def finetune(
    psm_list: PSMList,
    model_save_path: str | PathLike,
    model: torch.nn.Module | PathLike | str | None = None,
    epochs: int = 5,
    learning_rate: float = 1e-4,
    validation_fraction: float = 0.2,
    batch_size: int = 64,
    device: str = "cpu",
) -> torch.nn.Module:
    """
    Finetune IM2Deep on MALDI reference data using transfer learning.

    Loads a pre-trained IM2Deep model and finetunes all layers on the provided
    MALDI CCS data for a small number of epochs with a low learning rate.

    Parameters
    ----------
    psm_list
        PSMList with observed CCS values (in metadata["CCS"]).
        Must contain peptides with known CCS from MALDI measurements.
    model_save_path
        Path to save the finetuned model checkpoint.
    model
        Pre-trained model or path. If None, uses the default IM2DeepUni.ckpt.
    epochs
        Number of finetuning epochs. Default is 5.
    learning_rate
        Learning rate for finetuning. Default is 1e-4 (10x lower than training).
    validation_fraction
        Fraction of data to hold out for validation. Default is 0.2.
    batch_size
        Batch size for training. Default is 64.
    device
        Device to use ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    torch.nn.Module
        The finetuned model.
    """
    import lightning as L
    from deeplc.data import DeepLCDataset
    from torch.utils.data import DataLoader, random_split

    from im2deep._architectures.im2deep_single import IM2Deep as IM2DeepArch
    from im2deep.constants import DEFAULT_CONFIG, DEFAULT_MODEL

    LOGGER.info(f"Finetuning IM2Deep on {len(psm_list)} PSMs for {epochs} epochs")

    # Validate input
    psm_list = validate_psm_list(psm_list, needs_target=True)

    # Prepare dataset — extract CCS targets from PSM metadata
    ccs_targets = np.array([
        float(psm.metadata.get("CCS", np.nan)) if psm.metadata else np.nan
        for psm in psm_list
    ], dtype=np.float32)
    if np.any(np.isnan(ccs_targets)):
        raise ValueError(
            f"Found {np.sum(np.isnan(ccs_targets))} PSMs without CCS in metadata. "
            "All PSMs must have metadata['CCS'] set for finetuning."
        )
    # DeepLCDataset uses target_retention_times as its target field;
    # we pass CCS values there for IM2Deep finetuning
    dataset = DeepLCDataset(
        peptidoforms=list(psm_list["peptidoform"]),
        target_retention_times=ccs_targets,
        add_ccs_features=True,
    )

    # DeepLCDataset returns (features_tuple, target) where features_tuple is
    # (atom_comp, diatom_comp, global_feats, one_hot). IM2Deep's training_step
    # expects a flat batch: atom_comp, diatom_comp, global_feats, one_hot, y.
    # Use a custom collate_fn to flatten.
    def _collate_fn(batch):
        features_list, targets = zip(*batch)
        stacked_features = [torch.stack([f[i] for f in features_list]) for i in range(len(features_list[0]))]
        stacked_targets = torch.tensor(targets, dtype=torch.float32)
        return (*stacked_features, stacked_targets)

    # Split into train/validation
    n_val = max(1, int(len(dataset) * validation_fraction))
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    LOGGER.info(f"Train: {n_train}, Validation: {n_val}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_collate_fn)

    # Load pre-trained model
    checkpoint_path = model or DEFAULT_MODEL
    config = DEFAULT_CONFIG.copy()
    config["learning_rate"] = learning_rate

    loaded_model = IM2DeepArch.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        config=config,
        criterion=torch.nn.L1Loss(),
    )
    loaded_model.to(device)

    # Finetune with Lightning
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto" if device == "cuda" else "cpu",
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
    )

    trainer.fit(loaded_model, train_loader, val_loader)

    # Save the finetuned model
    trainer.save_checkpoint(str(model_save_path))
    LOGGER.info(f"Finetuned model saved to {model_save_path}")

    return loaded_model


def recommend_calibration_strategy(
    median_error_pct: float,
    n_reference_peptides: int,
    error_threshold_pct: float = 3.0,
    min_spline_peptides: int = 30,
    min_finetune_peptides: int = 100,
) -> str:
    """
    Recommend whether to use linear calibration, spline, or finetuning.

    Based on the error patterns from predictor benchmarking (D2), decide
    which calibration strategy is appropriate given the available data.

    Parameters
    ----------
    median_error_pct
        Median relative CCS prediction error (%) after linear calibration.
    n_reference_peptides
        Number of reference peptides available for calibration.
    error_threshold_pct
        If median error is below this, linear calibration is sufficient.
    min_spline_peptides
        Minimum peptides required for spline calibration. Below this,
        only linear is safe (spline would overfit).
    min_finetune_peptides
        Minimum peptides required for finetuning.

    Returns
    -------
    str
        "linear", "spline", or "finetune"
    """
    if median_error_pct <= error_threshold_pct:
        LOGGER.info(
            f"Median error {median_error_pct:.2f}% <= {error_threshold_pct}%. "
            "Linear calibration is sufficient."
        )
        return "linear"

    if n_reference_peptides < min_spline_peptides:
        LOGGER.info(
            f"Median error {median_error_pct:.2f}% > {error_threshold_pct}% but only "
            f"{n_reference_peptides} reference peptides (< {min_spline_peptides}). "
            "Only linear calibration is safe — too few peptides for spline "
            "(would overfit)."
        )
        return "linear"

    if n_reference_peptides < min_finetune_peptides:
        LOGGER.info(
            f"Median error {median_error_pct:.2f}% > {error_threshold_pct}% with "
            f"{n_reference_peptides} reference peptides (>= {min_spline_peptides}, "
            f"< {min_finetune_peptides}). Recommending spline calibration."
        )
        return "spline"

    LOGGER.info(
        f"Median error {median_error_pct:.2f}% > {error_threshold_pct}% with "
        f"{n_reference_peptides} reference peptides. Recommending finetuning."
    )
    return "finetune"
