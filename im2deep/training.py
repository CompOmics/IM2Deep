"""Module for training the model."""

import torch
import torch.nn as nn
import logging
from typing import Dict, Tuple
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from .training_model import (
    IM2Deep,
    IM2DeepMulti,
    LogLowestMAE,
    IM2DeepMultiTransfer,
    IM2DeepTransfer,
)
from .training_utils import FlexibleLossSorted

torch.set_float32_matmul_precision("high")
logger = logging.getLogger(__name__)


def _data_to_dataloaders(
    data: Dict, batch_size: int, shuffle: bool = True
) -> torch.utils.data.DataLoader:
    """
    Converts data to dataloaders

    Args:
        data (Dict): Data dictionary
        batch_size (int): Batch size
        shuffle (bool): Shuffle data

    Returns:
        torch.utils.data.DataLoader: Dataloader object
    """

    tensors = {}
    for key in data.keys():
        tensors[key] = torch.tensor(data[key], dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(*[tensors[key] for key in data.keys()])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    logger.debug(len(dataloader))
    for batch_data in dataloader:
        logger.debug(len(batch_data))
        logger.debug(batch_data[0])
        logger.debug(len(batch_data[0]))
        break
    return dataloader


def _get_dataloaders(data: Dict, batch_size: int) -> Tuple[torch.utils.data.DataLoader, ...]:
    """Get dataloaders for training, validation and testing

    Args:
        data (Dict): Data dictionary
        batch_size (int): Batch size

    Returns:
        Tuple[torch.utils.data.DataLoader, ...]: Tuple of dataloaders
    """
    train_dataloader = _data_to_dataloaders(data["train"], batch_size, shuffle=True)
    valid_dataloader = _data_to_dataloaders(data["valid"], batch_size, shuffle=False)
    test_dataloader = _data_to_dataloaders(data["test"], batch_size, shuffle=False)
    return train_dataloader, valid_dataloader, test_dataloader


def _setup_callbacks(model_config: Dict, output_path: str) -> list:
    """Set up callbacks for the model

    Args:
        model_config (Dict): Model configuration dictionary
        output_path (str): Output path

    Returns:
        list: List of callbacks
    """
    # Use simple progress bar if rich progress is disabled in config
    use_rich_progress = model_config.get("use_rich_progress", True)

    callbacks = [ModelSummary(), LogLowestMAE(model_config)]

    if use_rich_progress:
        callbacks.append(RichProgressBar())

    if model_config["use_best_model"]:
        mcp = ModelCheckpoint(
            output_path + "/checkpoint",
            filename=model_config["model_name"],
            monitor=model_config["monitor"],
            mode=model_config["mode"],
            save_last=False,
        )
        callbacks.append(mcp)

    return callbacks


def _setup_wandb_logger(wandb_config: Dict, model: nn.Module) -> WandbLogger:
    """Set up Weights and Biases logger"""
    wandb_logger = WandbLogger(project=wandb_config["project_name"])
    wandb_logger.watch(model)
    return wandb_logger


def train_model(
    data: Dict, model_config: Dict, output_path: str
) -> Tuple[L.Trainer, nn.Module, torch.utils.data.DataLoader]:
    """Train the model

    Args:
        data (Dict): Data dictionary
        model_config (Dict): Model configuration dictionary
        output_path (str): Output path

    Returns:
        Tuple[L.Trainer, nn.Module, torch.utils.data.DataLoader]: Trainer, model and test dataloader
    """

    wandb_config = model_config["wandb"]
    train_data, valid_data, test_data = _get_dataloaders(data, model_config["batch_size"])
    if model_config.get("multi-output", False) == False:
        if model_config.get("transfer", False) == False:
            model = IM2Deep(model_config, criterion=nn.L1Loss())
        else:
            model = IM2DeepTransfer(model_config, criterion=nn.L1Loss())
    elif (model_config.get("multi-output", False) == True) and (
        model_config.get("transfer", False) == False
    ):
        model = IM2DeepMulti(
            model_config, criterion=FlexibleLossSorted(model_config["diversity_weight"])
        )
    else:
        model = IM2DeepMultiTransfer(
            model_config, criterion=FlexibleLossSorted(model_config["diversity_weight"])
        )

    logger.info(model)

    callbacks = _setup_callbacks(model_config, output_path)
    if wandb_config["enabled"]:
        wandb_logger = _setup_wandb_logger(wandb_config, model)

    # Determine accelerator and devices from config
    accelerator = model_config.get("accelerator", "gpu")
    if accelerator == "cpu":
        devices = 1  # For CPU, use 1 device
    else:
        devices = (
            [model_config["device"]]
            if isinstance(model_config["device"], int)
            else model_config["device"]
        )

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=model_config["epochs"],
        enable_progress_bar=model_config.get("use_rich_progress", True),
        callbacks=callbacks,
        logger=wandb_logger if wandb_config["enabled"] else None,
        default_root_dir=output_path,
    )

    trainer.fit(model, train_data, valid_data)

    # Load best model
    if model_config.get("multi-output", False) == False:
        if model_config["use_best_model"] and model_config.get("transfer", False) == False:
            model = IM2Deep.load_from_checkpoint(
                callbacks[-1].best_model_path, config=model_config, criterion=nn.L1Loss()
            )
        elif model_config.get("transfer", False) and model_config["use_best_model"]:
            model = IM2DeepTransfer.load_from_checkpoint(
                callbacks[-1].best_model_path, config=model_config, criterion=nn.L1Loss()
            )
    elif (
        model_config.get("multi-output", False)
        and model_config["use_best_model"]
        and model_config.get("transfer", False) == False
    ):
        model = IM2DeepMulti.load_from_checkpoint(
            callbacks[-1].best_model_path,
            config=model_config,
            criterion=FlexibleLossSorted(model_config["diversity_weight"]),
        )
    elif model_config["use_best_model"] and model_config["multi-output"]:
        model = IM2DeepMultiTransfer.load_from_checkpoint(
            callbacks[-1].best_model_path,
            config=model_config,
            criterion=FlexibleLossSorted(model_config["diversity_weight"]),
        )

    return trainer, model, test_data
    # TODO: save full model?
