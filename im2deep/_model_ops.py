# TODO: evaluate whether these functions can just be imported from DeepLC
"""Training, predicting, and evaluating using IM2Deep (PyTorch)."""

from __future__ import annotations
import copy
import logging
from os import PathLike
from pathlib import Path

import torch
from rich.progress import track
from torch.utils.data import DataLoader, Dataset

from deeplc.data import split_datasets

LOGGER = logging.getLogger(__name__)


def load_model(
    model: torch.nn.Module | PathLike | str | None = None,
    device: str | None = None,
) -> torch.nn.Module:
    """Load a model from a file or return a randomly initialized model if none is provided."""
    # If device is not specified, use the default device (GPU if available, else CPU)
    selected_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model from file if a path is provided
    if isinstance(model, str | Path):
        loaded_model = torch.load(model, weights_only=False, map_location=selected_device)
    elif isinstance(model, torch.nn.Module):
        loaded_model = model
    elif model is None:
        # TODO: Implement randomly initialized model; requires model architecture definition
        raise NotImplementedError("Loading randomly initialized model is not implemented yet.")
    else:
        raise TypeError(f"Expected a PyTorch Module or a file path, got {type(model)} instead.")

    # Ensure the model is on the specified device
    loaded_model.to(selected_device)

    return loaded_model


def predict(
    model: torch.nn.Module | PathLike | str | None = None,
    data: Dataset | None = None,
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
    LOGGER.debug("Loading model for prediction.")
    model = load_model(model=model, device=device)

    if data is None:
        raise ValueError("Data must be provided for prediction.")

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
            all_predictions.append(outputs.cpu())
    return torch.cat(all_predictions, dim=0).squeeze()
