# TODO: evaluate whether these functions can just be imported from DeepLC
"""Training, predicting, and evaluating using IM2Deep (PyTorch)."""

from __future__ import annotations

import logging
import warnings
from os import PathLike
from pathlib import Path

import lightning as L
import torch
from rich.progress import track
from torch.utils.data import DataLoader, Dataset

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
        loaded_model.eval()  # Set model to evaluation mode
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

    Returns
    -------
    torch.Tensor
        Predictions.

    """
    # Check data first before loading model
    if data is None:
        raise ValueError("Data must be provided for prediction.")

    # TODO: implement custom model inference
    LOGGER.debug("Loading model for prediction.")
    model = _get_architecture(
        multi=multi,
    ).load_from_checkpoint(
        checkpoint_path=model,
        config=_get_model_config(multi=multi),
        criterion=_get_loss_function(multi=multi),
    )
    model.to(device)
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


def _get_architecture(multi: bool) -> L.LightningModule:
    """Get the model architecture based on whether multi-output is needed."""
    if multi:
        from im2deep._architecture import IM2DeepMultiTransfer

        return IM2DeepMultiTransfer
    else:
        from im2deep._architecture import IM2Deep

        return IM2Deep


def _get_model_config(multi: bool) -> dict:
    """Get the model configuration based on whether multi-output is needed."""
    if multi:
        from im2deep.constants import DEFAULT_MULTI_CONFIG

        return DEFAULT_MULTI_CONFIG
    else:
        from im2deep.constants import DEFAULT_CONFIG

        return DEFAULT_CONFIG


def _get_loss_function(multi: bool) -> torch.nn.modules.loss._Loss | torch.nn.Module:
    """Get the loss function based on whether multi-output is needed."""
    if multi:
        from im2deep._architecture import FlexibleLossSorted

        return FlexibleLossSorted()
    else:
        return torch.nn.L1Loss()
