# TODO: Evaluate whether these functions can just be imported from DeepLC or use Lightning?
"""Training, predicting, and evaluating using IM2Deep (PyTorch)."""

from __future__ import annotations

import logging
import warnings
from os import PathLike
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ModelSummary,
    RichProgressBar,
)
from rich.progress import track
from torch.utils.data import DataLoader, Dataset

from im2deep._architectures.callbacks import BackboneFreeze, LogLowestMAE
from im2deep._architectures.im2deep_multi import IM2DeepMultiTransfer
from im2deep._architectures.im2deep_single import IM2Deep, IM2DeepTransfer
from im2deep._architectures.losses import FlexibleLossSorted

# Suppress PyTorch padding warning for conv1d with even kernels and odd dilation
warnings.filterwarnings(
    "ignore",
    message="Using padding='same' with even kernel lengths and odd dilation.*",
    category=UserWarning,
    module="torch.nn.modules.conv",
)

LOGGER = logging.getLogger(__name__)


def load_model(
    model: torch.nn.Module | PathLike | str | None = None,
    device: str | None = None,
) -> torch.nn.Module:
    """Load a model from a file or return a randomly initialized model if none is provided."""
    # If device is not specified, use the default device (GPU if available, else CPU)
    selected_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.debug(f"Using device: {selected_device}")

    # Load model from file if a path is provided
    if isinstance(model, (str, Path)):
        checkpoint = torch.load(model, weights_only=False, map_location=selected_device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # If it's a dictionary, it might be a checkpoint with 'model' or 'state_dict' key
            if "model" in checkpoint:
                loaded_model = checkpoint["model"]
            elif "state_dict" in checkpoint:
                # Need to initialize model architecture first, then load state dict
                # For now, just extract the state dict
                LOGGER.error(
                    "Checkpoint contains state_dict but no model architecture. "
                    "This format is not yet supported. Please provide a full model checkpoint."
                )
                raise NotImplementedError(
                    "Loading from state_dict-only checkpoints is not yet implemented."
                )
            else:
                # Assume the entire dict is the model (some formats do this)
                LOGGER.warning(
                    "Checkpoint is a dict but format is unclear. Attempting to use as-is."
                )
                loaded_model = checkpoint
        else:
            # Direct model object
            loaded_model = checkpoint

    elif isinstance(model, torch.nn.Module):
        loaded_model = model
    elif model is None:
        # TODO: Implement randomly initialized model; requires model architecture definition
        raise NotImplementedError("Loading randomly initialized model is not implemented yet.")
    else:
        raise TypeError(f"Expected a PyTorch Module or a file path, got {type(model)} instead.")

    # Ensure the model is on the specified device
    if isinstance(loaded_model, torch.nn.Module):
        loaded_model.to(selected_device)
    else:
        raise TypeError(
            f"Loaded model is not a PyTorch Module, got {type(loaded_model)} instead. "
            f"The checkpoint file may be in an incompatible format."
        )

    return loaded_model


def predict(
    model: torch.nn.Module | PathLike | str | None = None,
    data: Dataset | None = None,
    multi=False,
    device: str = "cpu",
    batch_size: int = 512,
    num_workers: int = 0,
    num_threads: int | None = None,
) -> torch.Tensor:
    """
    Predict using a trained model.

    Parameters
    ----------
    model
        Trained model or path to model file.
    data
        Dataset to predict on.
    device
        Device to use for prediction.
    batch_size
        Batch size for prediction.
    num_workers
        Number of workers for data loading.
    num_threads
        Number of threads for model operations on CPU (ignored if using GPU).

    Returns
    -------
    torch.Tensor
        Predictions.

    """
    # Check data first before loading model
    if data is None:
        raise ValueError("Data must be provided for prediction.")

    torch.set_num_threads(num_threads or torch.get_num_threads())

    # TODO: implement custom model inference
    LOGGER.debug("Loading model for prediction.")

    # TODO: Implement load_model function here (also config) and path to default model?
    model = read_checkpoint_architecture(model, multi=multi).load_from_checkpoint(
        checkpoint_path=model,  # type: ignore # TODO: Match with function signature
        config=read_checkpoint_config(model, multi=multi),
        criterion=_get_loss_function(multi=multi),
    )
    model.to(device)
    model.eval()
    LOGGER.debug(f"Model loaded on device: {device}")

    data_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    LOGGER.debug("DataLoader created for prediction.")
    LOGGER.debug("Starting prediction loop.")
    predictions = _predict_loop(model, data_loader, device)
    return predictions.cpu().detach()


def _predict_loop(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str,
) -> torch.Tensor:
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for features, _ in track(data_loader, description="Predicting", transient=True):
            features = [feature_tensor.to(device) for feature_tensor in features]
            outputs = model(*features)
            if not isinstance(outputs, tuple):
                # Single output
                all_predictions.append(outputs.cpu())
            else:
                # Multi-output: stack both predictions side by side
                stacked = torch.stack([outputs[0], outputs[1]], dim=1)
                all_predictions.append(stacked.cpu())

    return torch.cat(all_predictions, dim=0).squeeze()


def train(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    config: dict,
    output_dir: PathLike | str,
    model: torch.nn.Module | None = None,
) -> tuple[L.Trainer, torch.nn.Module]:
    """
    Train an IM2Deep model on already-featurised datasets.

    Parameters
    ----------
    train_dataset
        Training dataset, yielding the flat
        ``(atom, diatom, global, one_hot, target)`` tuples the architectures'
        ``training_step`` expects. See :class:`im2deep._data.CCSDataset`.
    validation_dataset
        Validation dataset, same layout.
    config
        Training configuration. See
        :data:`im2deep.constants.DEFAULT_TRAINING_CONFIG` for the keys read.
    output_dir
        Directory for Lightning logs and intermediate checkpoints.
    model
        Model to continue training. If None, a fresh model is built from
        ``config``: :class:`~im2deep._architectures.im2deep_single.IM2DeepTransfer`
        when ``config["backbone_SD_path"]`` is set, otherwise
        :class:`~im2deep._architectures.im2deep_single.IM2Deep`.

    Returns
    -------
    tuple
        ``(trainer, model, best_checkpoint_path)``. The model is the best
        checkpoint reloaded when ``config["use_best_model"]`` is set, and
        ``best_checkpoint_path`` is None when no checkpoint was written.

    """
    torch.set_float32_matmul_precision("high")
    output_dir = Path(output_dir)

    criterion = _get_loss_function(multi=False)
    if model is None:
        model = _build_model(config, criterion)
    LOGGER.debug(f"Training {type(model).__name__}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        persistent_workers=bool(config.get("num_workers", 0)),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        persistent_workers=bool(config.get("num_workers", 0)),
    )

    callbacks, checkpoint_callback = _setup_callbacks(config, output_dir)
    trainer = L.Trainer(
        accelerator=config.get("accelerator", "auto"),
        devices=config.get("devices", "auto"),
        max_epochs=config["epochs"],
        callbacks=callbacks,
        logger=_setup_logger(config, model, output_dir),
        default_root_dir=str(output_dir),
        enable_progress_bar=True,
    )
    trainer.fit(model, train_loader, validation_loader)

    best_checkpoint_path = None
    if checkpoint_callback is not None and checkpoint_callback.best_model_path:
        best_checkpoint_path = checkpoint_callback.best_model_path
        LOGGER.info(f"Reloading best checkpoint: {best_checkpoint_path}")
        model = type(model).load_from_checkpoint(
            best_checkpoint_path, config=config, criterion=criterion
        )

    return trainer, model, best_checkpoint_path


def _build_model(config: dict, criterion: torch.nn.Module) -> torch.nn.Module:
    """Build a fresh single-conformer model from a training configuration."""
    if config.get("backbone_SD_path"):
        return IM2DeepTransfer(config, criterion=criterion)
    return IM2Deep(config, criterion=criterion)


def _setup_callbacks(
    config: dict, output_dir: Path
) -> tuple[list[L.Callback], ModelCheckpoint | None]:
    """Build the Lightning callbacks for a training run."""
    callbacks: list[L.Callback] = [
        ModelSummary(),
        RichProgressBar(),
        LogLowestMAE(config),
    ]

    checkpoint_callback = None
    if config.get("use_best_model", True):
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(output_dir / "checkpoint"),
            filename=config["model_name"],
            monitor=config["monitor"],
            mode=config["mode"],
            save_last=False,
        )
        callbacks.append(checkpoint_callback)

    if config.get("patience"):
        callbacks.append(
            EarlyStopping(
                monitor=config["monitor"],
                mode=config["mode"],
                patience=config["patience"],
            )
        )

    if config.get("freeze_epochs"):
        callbacks.append(
            BackboneFreeze(
                freeze_epochs=config["freeze_epochs"],
                unfreeze_lr_scale=config.get("unfreeze_lr_scale", 0.1),
            )
        )

    return callbacks, checkpoint_callback


def _setup_logger(config: dict, model: torch.nn.Module, output_dir: Path | None = None):
    """
    Build a Weights & Biases logger, or None when wandb is not enabled.

    The run is named after ``model_name`` unless the wandb block overrides it,
    so a set of runs differing only in training data or featurisation is
    distinguishable in the dashboard rather than carrying wandb's random names.
    The full training config is logged with the run, which is what makes those
    runs comparable after the fact.
    """
    wandb_config = config.get("wandb") or {}
    if not wandb_config.get("enabled"):
        return None

    try:
        import wandb  # noqa: F401
        from lightning.pytorch.loggers import WandbLogger
    except ImportError as exc:
        raise ImportError(
            "wandb logging was enabled but wandb is not installed. "
            "Install it with `pip install im2deep[wandb]`."
        ) from exc

    logger = WandbLogger(
        project=wandb_config.get("project_name", "IM2Deep"),
        name=wandb_config.get("name") or config.get("model_name"),
        entity=wandb_config.get("entity"),
        tags=wandb_config.get("tags"),
        save_dir=str(output_dir) if output_dir is not None else None,
    )
    # Log the training config so runs can be told apart and compared later.
    logger.experiment.config.update(
        {key: value for key, value in config.items() if key != "wandb"},
        allow_val_change=True,
    )
    logger.watch(model)
    LOGGER.info(
        f"Logging to Weights & Biases project "
        f"'{wandb_config.get('project_name', 'IM2Deep')}' as run "
        f"'{wandb_config.get('name') or config.get('model_name')}'."
    )
    return logger


def _get_architecture(multi: bool) -> type[IM2DeepMultiTransfer] | type[IM2Deep]:
    """Get the model architecture based on whether multi-output is needed."""
    if multi:
        return IM2DeepMultiTransfer
    else:
        return IM2Deep


def _read_checkpoint(model: torch.nn.Module | PathLike | str | None) -> dict | None:
    """Load a checkpoint file as a dict, or None if it is not one."""
    if not isinstance(model, (str, Path)):
        return None

    try:
        checkpoint = torch.load(model, weights_only=False, map_location="cpu")
    except Exception as exc:  # pragma: no cover - re-raised by load_from_checkpoint
        LOGGER.debug(f"Could not pre-read checkpoint: {exc}")
        return None

    if not isinstance(checkpoint, dict):
        LOGGER.debug("Checkpoint is not a Lightning checkpoint.")
        return None
    return checkpoint


def read_checkpoint_config(
    model: torch.nn.Module | PathLike | str | None, multi: bool = False
) -> dict:
    """
    The configuration a checkpoint was trained with, or the package default.

    Checkpoints written by :func:`im2deep.core.train` record their own config,
    which is what makes a model trained with a non-default architecture width
    or featurisation readable back. The bundled checkpoints predate that and
    carry no hyperparameters, so they fall back to the package defaults, which
    is exactly what they were trained with.

    Parameters
    ----------
    model
        Path to a checkpoint. Anything else returns the package default.
    multi
        Whether the multi-conformer default applies.

    Returns
    -------
    dict
        Model configuration.

    """
    default = _get_model_config(multi=multi)
    checkpoint = _read_checkpoint(model)
    if checkpoint is None:
        return default

    config = (checkpoint.get("hyper_parameters") or {}).get("config")
    if not config:
        LOGGER.debug("Checkpoint carries no config; using the package default.")
        return default

    LOGGER.debug("Using the configuration recorded in the checkpoint.")
    return config


def read_checkpoint_architecture(
    model: torch.nn.Module | PathLike | str | None, multi: bool = False
):
    """
    The architecture class a checkpoint was written by.

    A fine-tuned model is an
    :class:`~im2deep._architectures.im2deep_single.IM2DeepTransfer`, whose
    state_dict nests the pretrained weights under ``backbone.``, so it cannot be
    loaded as a plain :class:`~im2deep._architectures.im2deep_single.IM2Deep`.
    The distinction is read off the state_dict rather than the config, so it
    also holds for checkpoints that record no configuration.

    Parameters
    ----------
    model
        Path to a checkpoint.
    multi
        Whether this is a multi-conformer model.

    Returns
    -------
    type
        The architecture class to load the checkpoint with.

    """
    if multi:
        return _get_architecture(multi=True)

    checkpoint = _read_checkpoint(model)
    state_dict = (checkpoint or {}).get("state_dict") or {}
    if any(key.startswith("backbone.") for key in state_dict):
        LOGGER.debug("Checkpoint is a transfer model.")
        return IM2DeepTransfer
    return IM2Deep


def _get_model_config(multi: bool) -> dict:
    """Get the model configuration based on whether multi-output is needed."""
    if multi:
        from im2deep.constants import DEFAULT_MULTI_CONFIG

        return DEFAULT_MULTI_CONFIG
    else:
        from im2deep.constants import DEFAULT_CONFIG

        return DEFAULT_CONFIG


def _get_loss_function(multi: bool) -> FlexibleLossSorted | torch.nn.L1Loss:
    """Get the loss function based on whether multi-output is needed."""
    if multi:
        return FlexibleLossSorted()
    else:
        return torch.nn.L1Loss()
