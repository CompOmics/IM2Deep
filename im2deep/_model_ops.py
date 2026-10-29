# TODO: Evaluate whether these functions can just be imported from DeepLC or use Lightning?
"""Training, predicting, and evaluating using IM2Deep (PyTorch)."""

from __future__ import annotations

import logging
import warnings
from collections import Counter
from functools import lru_cache
from os import PathLike
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ModelSummary,
    RichProgressBar,
    StochasticWeightAveraging,
)
from psm_utils.peptidoform import Peptidoform
from rich.progress import track
from torch.utils.data import DataLoader, Dataset, Sampler, Subset, WeightedRandomSampler

from im2deep._architectures.callbacks import BackboneFreeze, LogLowestMAE
from im2deep._architectures.im2deep_multi import IM2DeepMultiTransfer
from im2deep._architectures.im2deep_single import IM2Deep, IM2DeepTransfer
from im2deep._architectures.losses import FlexibleLossSorted
from im2deep._data import stripped_sequence
from im2deep.exceptions import IM2DeepError

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


def _load_model_for_inference(
    model: torch.nn.Module | PathLike | str | None,
    multi: bool,
    device: str,
) -> torch.nn.Module:
    """
    Load a checkpoint (or the bundled default) as an eval-mode model on ``device``.

    Factored out of :func:`predict` so other tooling can load a checkpoint the
    same way - reading its own recorded config/architecture rather than
    guessing - without duplicating this block.
    """
    loaded_model = read_checkpoint_architecture(model, multi=multi).load_from_checkpoint(
        checkpoint_path=model,  # type: ignore # TODO: Match with function signature
        config=read_checkpoint_config(model, multi=multi),
        criterion=_get_loss_function(multi=multi),
        strict=False,
    )
    loaded_model.to(device)
    loaded_model.eval()
    return loaded_model


def predict(
    model: torch.nn.Module | PathLike | str | None = None,
    data: Dataset | None = None,
    multi=False,
    device: str | None = None,
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
        Device to use for prediction. Defaults to GPU if available, else CPU
        (same auto-detection :func:`load_model` and deeplc's own ``predict``
        use) - previously hardcoded to ``"cpu"``, which silently forced
        CPU-only inference even on GPU-equipped machines unless a caller
        explicitly passed ``device="cuda"``.
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

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(num_threads or torch.get_num_threads())

    # TODO: implement custom model inference
    LOGGER.debug("Loading model for prediction.")

    model = _load_model_for_inference(model, multi=multi, device=device)
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
    with torch.inference_mode():
        for features, _ in track(data_loader, description="Predicting", transient=True):
            features = [feature_tensor.to(device) for feature_tensor in features]
            outputs = model(*features)
            if not isinstance(outputs, tuple):
                # Single output
                all_predictions.append(outputs.cpu())
            elif getattr(model, "use_heteroscedastic_head", False):
                # (mu, b) (step 8.5): only mu is a CCS prediction. b (the
                # model's own per-row uncertainty) is not yet exposed
                # through this public API - checked before the generic
                # "stack both" branch below, which is for IM2DeepMulti's
                # genuinely-two-conformer-predictions tuple, a different
                # kind of forward() tuple this would otherwise collide with.
                all_predictions.append(outputs[0].cpu())
            else:
                # Multi-output: stack both predictions side by side
                stacked = torch.stack([outputs[0], outputs[1]], dim=1)
                all_predictions.append(stacked.cpu())

    return torch.cat(all_predictions, dim=0).squeeze()


def _training_targets(dataset: Dataset) -> torch.Tensor:
    """The raw CCS target array for a training dataset, unwrapping a ``Subset`` if needed."""
    if isinstance(dataset, Subset):
        targets = dataset.dataset.target_retention_times[dataset.indices]
    else:
        targets = dataset.target_retention_times
    return torch.as_tensor(targets, dtype=torch.float32)


def _training_peptidoforms(dataset: Dataset) -> list[str]:
    """The ProForma strings for a training dataset, unwrapping a ``Subset`` if needed."""
    if isinstance(dataset, Subset):
        return [dataset.dataset.peptidoforms[i] for i in dataset.indices]
    return list(dataset.peptidoforms)


def _charge_balanced_weights(train_dataset: Dataset) -> torch.Tensor:
    """
    Per-sample weights proportional to ``1 / sqrt(n_z)`` (step 8.3).

    Reweights rare charges (z=1/5/6) toward being sampled as often as the
    dominant ones without discarding any z=2/3 depth, unlike undersampling.
    Cheap: reads charge off each ProForma string's own ``precursor_charge``,
    not a full DeepLC featurisation pass.
    """
    peptidoforms = _training_peptidoforms(train_dataset)
    charges = np.array([Peptidoform(p).precursor_charge for p in peptidoforms])
    counts = Counter(charges.tolist())
    weights = np.array([1.0 / np.sqrt(counts[c]) for c in charges])
    return torch.as_tensor(weights, dtype=torch.double)


def _compute_pair_ids(peptidoforms: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Sibling-group id and theoretical m/z per training row (step 8.3).

    ``group_id`` is an integer code per distinct stripped sequence (matching
    :func:`im2deep._data.grouped_split`'s own grouping), used by the sibling
    delta loss to find pairs that differ only in modification state or
    charge. ``mz`` is each row's theoretical precursor m/z, used by the
    ranking loss to find pairs a quadrupole could not separate.

    Both are computed once per training run, not per epoch: charge-balanced
    weighting already parses every peptidoform once for the same reason (see
    :func:`_charge_balanced_weights`), and re-parsing per batch would be far
    too slow at millions of rows. Modifications change ``theoretical_mz`` but
    not the group id - a phosphorylated and unmodified copy of the same
    backbone are deliberately still siblings.
    """
    stripped = [stripped_sequence(pf) for pf in peptidoforms]
    _, group_ids = np.unique(stripped, return_inverse=True)
    mz = np.array([Peptidoform(pf).theoretical_mz for pf in peptidoforms], dtype=np.float32)
    return group_ids.astype(np.int64), mz


class _PairAugmentedDataset(Dataset):
    """
    Wraps a training dataset to also yield each row's sibling-group id and
    m/z, for the sibling delta and ranking losses (step 8.3).

    Wrapping rather than changing :class:`im2deep._data.CCSDataset` itself
    keeps every other training/prediction path (5-tuple or 6-tuple with
    ``mol_desc``) completely untouched; this only exists on the loader built
    when ``sibling_loss_weight`` or ``ranking_loss_weight`` is set.
    """

    def __init__(self, base: Dataset, group_ids: np.ndarray, mz: np.ndarray):
        self.base = base
        self.group_ids = group_ids
        self.mz = mz

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple:
        item = self.base[idx]
        group_id = torch.as_tensor(self.group_ids[idx], dtype=torch.long)
        mz = torch.as_tensor(self.mz[idx], dtype=torch.float32)
        return (*item, group_id, mz)


class SiblingPairBatchSampler(Sampler):
    """
    Batch sampler that deliberately places sibling pairs inside otherwise
    randomly composed batches (step 8.3's sibling delta loss).

    A plain random batch of 512 has ~0.07 sibling pairs on real training
    data (measured), because most stripped sequences appear only once or
    twice among 4M+ rows - nowhere near enough for a loss term supervising
    pair deltas to see a training signal. Each yielded batch instead draws
    ``pairs_per_batch`` random sibling groups (each contributing one pair,
    sampled with replacement across the epoch, mirroring
    ``WeightedRandomSampler(replacement=True)``'s own approach elsewhere in
    this module) plus ordinary random singleton draws for the remaining
    slots, so composition stays close to the overall data distribution.
    """

    def __init__(
        self,
        group_ids: np.ndarray,
        batch_size: int,
        pairs_per_batch: int,
        num_batches: int,
    ):
        self.group_ids = np.asarray(group_ids)
        self.batch_size = batch_size
        self.pairs_per_batch = pairs_per_batch
        self.num_batches = num_batches
        self.n = len(group_ids)

        groups: dict[int, list[int]] = {}
        for idx, group in enumerate(self.group_ids):
            groups.setdefault(int(group), []).append(idx)
        self.sibling_groups = [idxs for idxs in groups.values() if len(idxs) >= 2]
        if not self.sibling_groups:
            raise IM2DeepError(
                "sibling_loss_weight is set, but no sibling groups (a stripped sequence "
                "observed more than once) exist in the training data."
            )

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng()
        n_groups = len(self.sibling_groups)
        for _ in range(self.num_batches):
            batch: list[int] = []
            for _ in range(self.pairs_per_batch):
                group = self.sibling_groups[rng.integers(n_groups)]
                pair = rng.choice(group, size=2, replace=False)
                batch.extend(int(i) for i in pair)
            remaining = self.batch_size - len(batch)
            batch.extend(int(i) for i in rng.integers(0, self.n, size=remaining))
            rng.shuffle(batch)
            yield batch


def _fit_scalers(
    train_dataset: Dataset, config: dict
) -> tuple[torch.Tensor | None, torch.Tensor | None, float | None, float | None]:
    """
    Fit the optional global-feature and target standardisation buffers (step 8.2).

    Returns ``(global_mean, global_std, target_mean, target_std)``, each pair
    ``None`` when its config flag is off - the caller only calls the matching
    ``set_*_scaler`` when its pair isn't ``None``, so an unfit statistic never
    overwrites a model's no-op default.
    """
    global_mean = global_std = None
    target_mean = target_std = None

    if config.get("standardize_target"):
        targets = _training_targets(train_dataset)
        target_mean = float(targets.mean())
        target_std = float(targets.std())
        if target_std == 0:
            target_std = 1.0  # degenerate all-equal sample; std=0 would divide by zero

    if config.get("standardize_global_features"):
        sample_size = min(len(train_dataset), 50_000)
        generator = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(train_dataset), generator=generator)[:sample_size].tolist()
        loader = DataLoader(
            Subset(train_dataset, idx),
            batch_size=2048,
            num_workers=config.get("num_workers", 0),
        )
        feats = torch.cat([batch[2] for batch in loader], dim=0)
        global_mean = feats.mean(dim=0)
        global_std = feats.std(dim=0)
        global_std[global_std == 0] = 1.0  # a constant feature column; avoid dividing by zero

    return global_mean, global_std, target_mean, target_std


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
    given_model = model
    if model is None:
        model = _build_model(config, criterion)
    # Only for a freshly-built plain model: a model passed in (e.g. finetuning
    # a loaded backbone) keeps whatever scaler state it already has, and so
    # does a transfer model, whose backbone carries its own fitted scalers.
    if model is not given_model and not isinstance(model, IM2DeepTransfer):
        global_mean, global_std, target_mean, target_std = _fit_scalers(train_dataset, config)
        if global_mean is not None:
            LOGGER.info("Fitted global-feature standardisation from a training sample.")
            model.set_global_feature_scaler(global_mean, global_std)
        if target_mean is not None:
            LOGGER.info(
                f"Fitted target standardisation: mean={target_mean:.2f}, std={target_std:.2f}."
            )
            model.set_target_scaler(target_mean, target_std)
    LOGGER.debug(f"Training {type(model).__name__}")

    # Step 8.3: the sibling delta and ranking losses need each row's sibling-
    # group id and/or theoretical m/z, which the plain (atom, diatom, global,
    # one_hot, target) tuple does not carry. Wrap the dataset to add them only
    # when at least one of those losses is enabled, so every other run's
    # batch tuple is unaffected.
    needs_pair_ids = (
        config.get("sibling_loss_weight", 0.0) > 0 or config.get("ranking_loss_weight", 0.0) > 0
    )
    if needs_pair_ids:
        if config.get("mixup_alpha", 0.0) > 0:
            raise IM2DeepError(
                "sibling_loss_weight/ranking_loss_weight are not compatible with mixup_alpha>0: "
                "mixup blends batch rows together, which would silently corrupt the sibling-"
                "group id and m/z each row is supposed to carry."
            )
        group_ids, mz = _compute_pair_ids(_training_peptidoforms(train_dataset))
        loader_dataset: Dataset = _PairAugmentedDataset(train_dataset, group_ids, mz)
    else:
        loader_dataset = train_dataset

    # shuffle and sampler are mutually exclusive on DataLoader, hence the
    # if/else rather than always passing both.
    if config.get("sibling_loss_weight", 0.0) > 0:
        if config.get("charge_balanced_sampling"):
            raise IM2DeepError(
                "sibling_loss_weight and charge_balanced_sampling are both custom batch-"
                "composition strategies and are not yet compatible with each other."
            )
        train_loader = DataLoader(
            loader_dataset,
            batch_sampler=SiblingPairBatchSampler(
                group_ids=group_ids,
                batch_size=config["batch_size"],
                pairs_per_batch=config.get("sibling_pairs_per_batch", 64),
                num_batches=len(loader_dataset) // config["batch_size"],
            ),
            num_workers=config.get("num_workers", 0),
            persistent_workers=bool(config.get("num_workers", 0)),
        )
    elif config.get("charge_balanced_sampling"):
        weights = _charge_balanced_weights(train_dataset)
        train_loader = DataLoader(
            loader_dataset,
            batch_size=config["batch_size"],
            sampler=WeightedRandomSampler(weights, num_samples=len(weights), replacement=True),
            num_workers=config.get("num_workers", 0),
            persistent_workers=bool(config.get("num_workers", 0)),
        )
    else:
        train_loader = DataLoader(
            loader_dataset,
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
        precision=config.get("precision", "32-true"),
    )
    trainer.fit(model, train_loader, validation_loader)

    best_checkpoint_path = None
    if config.get("swa"):
        # StochasticWeightAveraging transfers its averaged weights into
        # `model` in `on_train_end` - after the last validation epoch, since
        # IM2Deep has no BatchNorm layers to trigger Lightning's extra
        # BN-recalibration epoch. No validation ever scores the averaged
        # weights, so `checkpoint_callback.best_model_path` never points at
        # them: reloading it here would silently throw the averaging away.
        # `model` already holds the averaged weights post-fit, so use it
        # as-is instead of reloading anything.
        LOGGER.info(
            "SWA enabled: using the final weight-averaged model, not a reloaded checkpoint."
        )
    elif checkpoint_callback is not None and checkpoint_callback.best_model_path:
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

    if config.get("swa"):
        callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=config.get("swa_lrs", config["learning_rate"]),
                swa_epoch_start=config.get("swa_epoch_start", 0.8),
                annealing_epochs=config.get("swa_annealing_epochs", 10),
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
    logger.log_hyperparams({key: value for key, value in config.items() if key != "wandb"})
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


@lru_cache(maxsize=8)
def _read_checkpoint(model: torch.nn.Module | PathLike | str | None) -> dict | None:
    """Load a checkpoint file as a dict, or None if it is not one.

    Cached: a single ``core.predict()`` call reads the same checkpoint path
    through this function up to three times (once directly, once via
    :func:`read_checkpoint_architecture`, once via :func:`read_checkpoint_config`
    called again inside :func:`predict`), each a full ``torch.load`` of the
    same bytes. Callers never mutate the returned dict (or the sub-dicts,
    e.g. the config, handed back from it), so sharing one cached object
    across those calls is safe.
    """
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
