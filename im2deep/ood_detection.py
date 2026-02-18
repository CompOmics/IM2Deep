"""
Out-of-Distribution (OOD) Detection Module for IM2Deep.

This module provides functionality to extract latent embeddings from trained IM2Deep models
and use them for OOD detection via Mahalanobis distance computation. The latent embeddings
are extracted from the first dense layer after feature concatenation (charge fusion).

Key Components:
    - Latent embedding extraction from trained models
    - Batch processing for large datasets
    - Embedding storage and retrieval
    - Sanity checks and validation

Authors:
    - Robbe Devreese
    - Robbin Bouwmeester

Example:
    >>> from im2deep.ood_detection import extract_embeddings, save_embeddings
    >>> from im2deep.training_model import IM2Deep
    >>>
    >>> # Load trained model
    >>> model = IM2Deep.load_from_checkpoint("model.ckpt")
    >>>
    >>> # Extract embeddings from training data
    >>> embeddings, ids = extract_embeddings(model, train_dataloader, device="cuda")
    >>> save_embeddings("train_embeddings.npz", embeddings, ids)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class LatentExtractorMixin:
    """
    Mixin class to add latent embedding extraction to IM2Deep models.

    This mixin modifies the forward pass to optionally return latent embeddings
    from the first dense layer after feature concatenation, without changing
    model weights or requiring retraining.
    """

    def forward_with_latent(
        self,
        atom_comp: torch.Tensor,
        diatom_comp: torch.Tensor,
        global_feats: torch.Tensor,
        one_hot: torch.Tensor,
        mol_desc: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns both CCS prediction and latent embedding.

        Args:
            atom_comp: Atomic composition features (batch_size, 6, 60)
            diatom_comp: Diatomic composition features (batch_size, 6, 30)
            global_feats: Global features including charge (batch_size, Global_units)
            one_hot: One-hot encoded sequence (batch_size, 20, 60)
            mol_desc: Molecular descriptor features, optional (batch_size, 13, 60)

        Returns:
            output: CCS predictions (batch_size, n_outputs)
            latent: Latent embeddings (batch_size, Concat_units)
        """
        # Permute inputs for conv layers
        atom_comp = atom_comp.permute(0, 2, 1)
        diatom_comp = diatom_comp.permute(0, 2, 1)
        one_hot = one_hot.permute(0, 2, 1)

        # Process through convolutional layers
        for layer in self.ConvAtomComp:
            atom_comp = layer(atom_comp)

        for layer in self.ConvDiatomComp:
            diatom_comp = layer(diatom_comp)

        for layer in self.ConvGlobal:
            global_feats = layer(global_feats)

        for layer in self.OneHot:
            one_hot = layer(one_hot)

        if hasattr(self.config, "__getitem__") and self.config.get("add_X_mol", False):
            if mol_desc is not None:
                for layer in self.MolDesc:
                    mol_desc = layer(mol_desc)

        # Concatenate all features
        concatenated = torch.cat((atom_comp, diatom_comp, one_hot, global_feats), 1)

        if hasattr(self.config, "__getitem__") and self.config.get("add_X_mol", False):
            if mol_desc is not None:
                concatenated = torch.cat((concatenated, mol_desc), 1)

        # Handle optional attention on concatenated features (IM2DeepMultiTransfer)
        if (
            hasattr(self, "SelfAttentionConcat")
            and self.config.get("Use_attention_concat", 0) == 1
        ):
            concatenated = self.SelfAttentionConcat(concatenated.unsqueeze(1)).squeeze(1)

        # Extract latent embedding from first dense layer
        # Handle both self.Concat (ModuleList) and self.concat (list from transfer)
        if hasattr(self, "Concat"):
            # Standard models: IM2Deep, IM2DeepMulti
            concat_layers = self.Concat
            latent = concat_layers[0](concatenated)
            remaining_layers = concat_layers[1:]
        elif hasattr(self, "concat"):
            # Transfer models: IM2DeepMultiTransfer
            concat_layers = self.concat
            latent = concat_layers[0](concatenated)
            remaining_layers = concat_layers[1:]
        else:
            raise AttributeError("Model has neither 'Concat' nor 'concat' attribute")

        # Continue through remaining concat layers
        output = latent
        for layer in remaining_layers:
            output = layer(output)

        # Handle optional attention on output (IM2DeepMultiTransfer)
        if (
            hasattr(self, "SelfAttentionOutput")
            and self.config.get("Use_attention_output", 0) == 1
        ):
            output = self.SelfAttentionOutput(output.unsqueeze(1)).squeeze(1)

        # Handle multi-output models (return tuple)
        if hasattr(self, "branches"):
            # IM2DeepMulti or IM2DeepMultiTransfer - return both outputs
            y_hat1 = self.branches[0](output)
            y_hat2 = self.branches[1](output)
            output = torch.stack([y_hat1.squeeze(), y_hat2.squeeze()], dim=1)

        return output, latent


def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: Union[str, torch.device] = "cpu",
    return_predictions: bool = False,
) -> Union[Tuple[np.ndarray, List], Tuple[np.ndarray, List, np.ndarray]]:
    """
    Extract latent embeddings from a trained IM2Deep model for a dataset.

    This function processes data in batches to handle large datasets efficiently,
    extracting embeddings from the first dense layer after feature concatenation.

    Args:
        model: Trained IM2Deep model (must have forward_with_latent method or be wrapped)
        dataloader: PyTorch DataLoader containing the dataset
        device: Device to run inference on ('cpu', 'cuda', or torch.device)
        return_predictions: If True, also return CCS predictions

    Returns:
        embeddings: Numpy array of shape (N_samples, latent_dim)
        ids: List of sample identifiers (indices)
        predictions: (optional) Numpy array of CCS predictions (N_samples, n_outputs)

    Example:
        >>> model = IM2Deep.load_from_checkpoint("model.ckpt")
        >>> embeddings, ids = extract_embeddings(model, train_loader, device="cuda")
        >>> print(f"Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    """
    # Ensure model is on correct device and in eval mode
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    # Add latent extraction capability if not present
    if not hasattr(model, "forward_with_latent"):
        # Add mixin methods dynamically
        model.forward_with_latent = LatentExtractorMixin.forward_with_latent.__get__(
            model, type(model)
        )
        logger.info("Added latent extraction capability to model")

    embeddings_list = []
    predictions_list = []
    ids_list = []
    sample_idx = 0

    logger.info(f"Extracting embeddings from {len(dataloader)} batches...")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # Handle different batch formats
            if len(batch_data) == 2:  # DeepLCDataset format: (features_list, targets)
                features, targets = batch_data
                # Unpack features list
                if len(features) == 4:  # Without mol_desc
                    atom_comp, diatom_comp, global_feats, one_hot = features
                    mol_desc = None
                elif len(features) == 5:  # With mol_desc
                    atom_comp, diatom_comp, global_feats, one_hot, mol_desc = features
                else:
                    raise ValueError(f"Unexpected features list length: {len(features)}")
            elif len(batch_data) == 5:  # Standard format without mol_desc
                atom_comp, diatom_comp, global_feats, one_hot, targets = batch_data
                mol_desc = None
            elif len(batch_data) == 6:  # Format with mol_desc
                atom_comp, diatom_comp, global_feats, one_hot, targets, mol_desc = batch_data
            else:
                raise ValueError(f"Unexpected batch format with {len(batch_data)} elements")

            # Move data to device
            atom_comp = atom_comp.to(device)
            diatom_comp = diatom_comp.to(device)
            global_feats = global_feats.to(device)
            one_hot = one_hot.to(device)
            if mol_desc is not None:
                mol_desc = mol_desc.to(device)

            # Extract embeddings and predictions
            outputs, latent = model.forward_with_latent(
                atom_comp, diatom_comp, global_feats, one_hot, mol_desc
            )

            # Move to CPU and convert to numpy
            embeddings_batch = latent.cpu().numpy()
            embeddings_list.append(embeddings_batch)

            if return_predictions:
                predictions_batch = outputs.cpu().numpy()
                predictions_list.append(predictions_batch)

            # Generate sample IDs
            batch_size = embeddings_batch.shape[0]
            batch_ids = list(range(sample_idx, sample_idx + batch_size))
            ids_list.extend(batch_ids)
            sample_idx += batch_size

            if (batch_idx + 1) % 100 == 0:
                logger.debug(f"Processed {batch_idx + 1}/{len(dataloader)} batches")

    # Concatenate all batches
    embeddings = np.vstack(embeddings_list)

    logger.info(f"Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    # Sanity checks
    _validate_embeddings(embeddings, "extracted embeddings")

    if return_predictions:
        predictions = np.vstack(predictions_list)
        return embeddings, ids_list, predictions

    return embeddings, ids_list


def embed_peptides(
    model: nn.Module,
    data_dict: Dict[str, np.ndarray],
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """
    Extract embeddings for new peptides from preprocessed feature arrays.

    This function applies the same preprocessing as during training and extracts
    latent embeddings in batch mode for efficiency.

    Args:
        model: Trained IM2Deep model
        data_dict: Dictionary containing preprocessed features:
            - 'X_AtomEnc': Atomic composition (N, 6, 60)
            - 'X_DiAminoAtomEnc': Diatomic composition (N, 6, 30)
            - 'X_GlobalFeatures': Global features (N, Global_units)
            - 'X_OneHot': One-hot encoding (N, 20, 60)
            - 'X_MolEnc': (optional) Molecular descriptors (N, 13, 60)
        device: Device to run inference on
        batch_size: Batch size for processing

    Returns:
        embeddings: Numpy array of shape (N, latent_dim)

    Example:
        >>> from im2deep.training_data import _get_matrices
        >>> # Prepare features using same preprocessing as training
        >>> data = _get_matrices(psm_list, split_name="test", inference=True)
        >>> embeddings = embed_peptides(model, data, device="cuda")
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    # Add latent extraction capability if not present
    if not hasattr(model, "forward_with_latent"):
        model.forward_with_latent = LatentExtractorMixin.forward_with_latent.__get__(
            model, type(model)
        )

    # Extract feature arrays from dictionary
    atom_comp = torch.tensor(data_dict["X_AtomEnc"], dtype=torch.float32)
    diatom_comp = torch.tensor(data_dict["X_DiAminoAtomEnc"], dtype=torch.float32)
    global_feats = torch.tensor(data_dict["X_GlobalFeatures"], dtype=torch.float32)
    one_hot = torch.tensor(data_dict["X_OneHot"], dtype=torch.float32)

    # Check for molecular descriptors
    mol_desc = None
    if "X_MolEnc" in data_dict:
        mol_desc = torch.tensor(data_dict["X_MolEnc"], dtype=torch.float32)
        dataset = TensorDataset(atom_comp, diatom_comp, global_feats, one_hot, mol_desc)
    else:
        dataset = TensorDataset(atom_comp, diatom_comp, global_feats, one_hot)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings_list = []

    logger.info(f"Embedding {len(dataset)} peptides in batches of {batch_size}...")

    with torch.no_grad():
        for batch_data in dataloader:
            if len(batch_data) == 4:
                atom_comp_b, diatom_comp_b, global_feats_b, one_hot_b = batch_data
                mol_desc_b = None
            else:
                atom_comp_b, diatom_comp_b, global_feats_b, one_hot_b, mol_desc_b = batch_data

            # Move to device
            atom_comp_b = atom_comp_b.to(device)
            diatom_comp_b = diatom_comp_b.to(device)
            global_feats_b = global_feats_b.to(device)
            one_hot_b = one_hot_b.to(device)
            if mol_desc_b is not None:
                mol_desc_b = mol_desc_b.to(device)

            # Extract embeddings
            _, latent = model.forward_with_latent(
                atom_comp_b, diatom_comp_b, global_feats_b, one_hot_b, mol_desc_b
            )

            embeddings_list.append(latent.cpu().numpy())

    embeddings = np.vstack(embeddings_list)

    logger.info(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    # Sanity checks
    _validate_embeddings(embeddings, "generated embeddings")

    return embeddings


def save_embeddings(
    path: Union[str, Path],
    embeddings: np.ndarray,
    ids: Optional[List] = None,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save embeddings and associated metadata to disk in NPZ format.

    Args:
        path: Output file path (.npz extension)
        embeddings: Embeddings array (N_samples, latent_dim)
        ids: List of sample identifiers (optional)
        metadata: Additional metadata dictionary (optional)

    Example:
        >>> save_embeddings(
        ...     "train_embeddings.npz",
        ...     embeddings,
        ...     ids=sample_ids,
        ...     metadata={"model": "IM2Deep_v1", "date": "2026-02-17"}
        ... )
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data dictionary
    data = {"embeddings": embeddings}

    if ids is not None:
        data["ids"] = np.array(ids)

    if metadata is not None:
        for key, value in metadata.items():
            # Convert to numpy-compatible format
            if isinstance(value, (list, tuple)):
                data[f"meta_{key}"] = np.array(value)
            elif isinstance(value, (str, int, float)):
                data[f"meta_{key}"] = np.array([value])
            else:
                data[f"meta_{key}"] = np.array(value)

    # Save to compressed NPZ format
    np.savez_compressed(path, **data)

    logger.info(f"Saved {len(embeddings)} embeddings to {path}")
    logger.info(f"File size: {path.stat().st_size / 1024 / 1024:.2f} MB")


def load_embeddings(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Load embeddings and metadata from NPZ file.

    Args:
        path: Path to NPZ file

    Returns:
        Dictionary containing embeddings, ids, and any metadata

    Example:
        >>> data = load_embeddings("train_embeddings.npz")
        >>> embeddings = data["embeddings"]
        >>> ids = data["ids"]
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    data = np.load(path, allow_pickle=True)
    result = {key: data[key] for key in data.files}

    logger.info(f"Loaded {len(result['embeddings'])} embeddings from {path}")

    return result


def _validate_embeddings(embeddings: np.ndarray, name: str = "embeddings") -> None:
    """
    Perform sanity checks on extracted embeddings.

    Validates:
        - No NaN values
        - No infinite values
        - Non-zero variance across dimensions
        - Consistent dimensionality

    Args:
        embeddings: Embeddings array to validate
        name: Name for error messages

    Raises:
        ValueError: If any validation check fails
    """
    # Check for NaN values
    if np.isnan(embeddings).any():
        n_nan = np.isnan(embeddings).sum()
        raise ValueError(f"{name} contains {n_nan} NaN values")

    # Check for infinite values
    if np.isinf(embeddings).any():
        n_inf = np.isinf(embeddings).sum()
        raise ValueError(f"{name} contains {n_inf} infinite values")

    # Check variance
    variances = np.var(embeddings, axis=0)
    zero_var_dims = np.sum(variances == 0)

    if zero_var_dims > 0:
        logger.warning(
            f"{name} has {zero_var_dims} dimensions with zero variance "
            f"(out of {embeddings.shape[1]} total dimensions)"
        )

    # Check if all embeddings are identical (very suspicious)
    if zero_var_dims == embeddings.shape[1]:
        raise ValueError(
            f"{name} has zero variance in all dimensions - all embeddings are identical!"
        )

    # Log statistics
    logger.info(f"Embedding validation for {name}:")
    logger.info(f"  Shape: {embeddings.shape}")
    logger.info(f"  Mean: {np.mean(embeddings):.4f} ± {np.std(embeddings):.4f}")
    logger.info(f"  Range: [{np.min(embeddings):.4f}, {np.max(embeddings):.4f}]")
    logger.info(
        f"  Non-zero variance dimensions: {embeddings.shape[1] - zero_var_dims}/{embeddings.shape[1]}"
    )


def get_embedding_statistics(embeddings: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics for embeddings.

    Args:
        embeddings: Embeddings array (N_samples, latent_dim)

    Returns:
        Dictionary of statistics
    """
    return {
        "n_samples": embeddings.shape[0],
        "embedding_dim": embeddings.shape[1],
        "mean": float(np.mean(embeddings)),
        "std": float(np.std(embeddings)),
        "min": float(np.min(embeddings)),
        "max": float(np.max(embeddings)),
        "median": float(np.median(embeddings)),
        "zero_variance_dims": int(np.sum(np.var(embeddings, axis=0) == 0)),
    }


def _extract_charge_from_id(peptidoform_id: str) -> int:
    """
    Extract charge state from peptidoform ID (format: sequence/charge).

    Args:
        peptidoform_id: Peptidoform string in format "sequence/charge"

    Returns:
        Charge state as integer
    """
    return int(peptidoform_id.split("/")[1])


def _compute_mahalanobis_distances(
    test_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute Mahalanobis distances from test embeddings to reference distribution.

    Args:
        test_embeddings: Test embeddings (N_test, latent_dim)
        reference_embeddings: Reference embeddings (N_ref, latent_dim)

    Returns:
        distances: Array of Mahalanobis distances (N_test,)
    """
    from scipy.spatial import distance

    # Compute covariance matrix and inverse
    cov = np.cov(reference_embeddings, rowvar=False)
    # Add small regularization for numerical stability
    cov += np.eye(cov.shape[0]) * 1e-6
    inv_cov = np.linalg.inv(cov)
    mean_ref = np.mean(reference_embeddings, axis=0)

    # Compute Mahalanobis distance for each test embedding
    distances = np.array(
        [distance.mahalanobis(test_emb, mean_ref, inv_cov) for test_emb in test_embeddings]
    )

    return distances


def _compute_euclidean_distances(
    test_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute Euclidean distances from test embeddings to reference mean.

    Args:
        test_embeddings: Test embeddings (N_test, latent_dim)
        reference_embeddings: Reference embeddings (N_ref, latent_dim)

    Returns:
        distances: Array of Euclidean distances (N_test,)
    """
    from scipy.spatial import distance

    mean_ref = np.mean(reference_embeddings, axis=0)
    distances = np.array([distance.euclidean(test_emb, mean_ref) for test_emb in test_embeddings])

    return distances


def _compute_knn_distances(
    test_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    n_neighbors: int = 1,
) -> np.ndarray:
    """
    Compute k-nearest neighbor distances.

    Args:
        test_embeddings: Test embeddings (N_test, latent_dim)
        reference_embeddings: Reference embeddings (N_ref, latent_dim)
        n_neighbors: Number of neighbors to consider

    Returns:
        distances: Array of k-NN distances (N_test,)
    """
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(reference_embeddings)
    distances, _ = nbrs.kneighbors(test_embeddings)
    distances = distances.flatten() if n_neighbors == 1 else distances.mean(axis=1)

    return distances


def compute_distance(
    test_embeddings: np.ndarray,
    test_ids: List,
    reference_embeddings: Union[np.ndarray, Dict],
    method: str = "mahalanobis",
    charge_conditional: bool = False,
    n_neighbors: int = 1,
) -> np.ndarray:
    """
    Compute distance between test embeddings and reference embeddings for OOD detection.

    Args:
        test_embeddings: Test embeddings (N_test, latent_dim)
        test_ids: List of test peptidoform IDs in format "sequence/charge"
        reference_embeddings: Either:
            - np.ndarray: Reference embeddings (N_ref, latent_dim)
            - Dict: Loaded NPZ file with keys 'embeddings' and 'ids'
        method: Distance method to use ('mahalanobis', 'euclidean', or 'knn')
        charge_conditional: If True, compute distances separately for each charge state
        n_neighbors: Number of neighbors for k-NN method (only used when method='knn')

    Returns:
        distances: Array of distances (N_test,)

    Example:
        >>> # Load reference embeddings
        >>> ref_data = load_embeddings("training_embeddings.npz")
        >>> # Compute charge-conditional Mahalanobis distances
        >>> distances = compute_distance(
        ...     test_embeddings, test_ids, ref_data,
        ...     method="mahalanobis", charge_conditional=True
        ... )
        >>> # Compute k-NN distances with k=5
        >>> distances_knn = compute_distance(
        ...     test_embeddings, test_ids, ref_data,
        ...     method="knn", n_neighbors=5
        ... )
    """
    # Ensure test_embeddings is a numpy array
    if isinstance(test_embeddings, list):
        test_embeddings = np.array(test_embeddings)

    # Handle different input formats for reference embeddings
    if isinstance(reference_embeddings, dict):
        ref_embeddings = reference_embeddings["embeddings"]
        ref_ids = reference_embeddings["ids"]
    else:
        ref_embeddings = reference_embeddings
        ref_ids = None

    # Select distance computation function and prepare kwargs
    distance_functions = {
        "mahalanobis": _compute_mahalanobis_distances,
        "euclidean": _compute_euclidean_distances,
        "knn": _compute_knn_distances,
    }

    if method not in distance_functions:
        raise ValueError(
            f"Unknown distance method: {method}. Choose from {list(distance_functions.keys())}"
        )

    distance_fn = distance_functions[method]

    # Prepare function-specific kwargs
    fn_kwargs = {}
    if method == "knn":
        fn_kwargs["n_neighbors"] = n_neighbors

    # Charge-conditional computation
    if charge_conditional:
        if ref_ids is None:
            raise ValueError(
                "Charge-conditional distance requires reference embeddings with IDs. "
                "Please provide reference_embeddings as a dict with 'embeddings' and 'ids' keys."
            )

        # Extract charge states
        test_charges = np.array([_extract_charge_from_id(id_) for id_ in test_ids])
        ref_charges = np.array([_extract_charge_from_id(id_) for id_ in ref_ids])

        # Get unique charge states
        unique_charges = np.unique(test_charges)

        # Initialize distances array
        distances = np.zeros(len(test_embeddings))

        logger.info(
            f"Computing {method} distances for {len(unique_charges)} charge states: {unique_charges}"
        )

        # Compute distances separately for each charge state
        for charge in unique_charges:
            # Get indices for this charge state
            test_mask = test_charges == charge
            ref_mask = ref_charges == charge

            # Get embeddings for this charge
            test_emb_charge = test_embeddings[test_mask]
            ref_emb_charge = ref_embeddings[ref_mask]

            if len(ref_emb_charge) < 2:
                logger.warning(
                    f"Charge {charge}: Only {len(ref_emb_charge)} reference samples. "
                    f"Using global distribution instead."
                )
                ref_emb_charge = ref_embeddings

            # Compute distances for this charge state
            distances_charge = distance_fn(test_emb_charge, ref_emb_charge, **fn_kwargs)

            # Store distances
            distances[test_mask] = distances_charge

            logger.debug(
                f"Charge {charge}: {np.sum(test_mask)} test samples, "
                f"{len(ref_emb_charge)} reference samples, "
                f"mean distance: {distances_charge.mean():.4f}"
            )

    else:
        # Global distance computation (no charge conditioning)
        distances = distance_fn(test_embeddings, ref_embeddings, **fn_kwargs)

    return distances


def _compute_covariance_and_inverse(
    embeddings: np.ndarray,
    mode: str = "full",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute covariance matrix and its inverse using different methods.

    Args:
        embeddings: Embeddings (N_samples, latent_dim)
        mode: Covariance computation mode
            - 'full': Full covariance with regularization
            - 'shrinkage': Ledoit-Wolf shrinkage estimation
            - 'diagonal': Diagonal approximation (assumes independence)

    Returns:
        cov: Covariance matrix
        inv_cov: Inverse covariance matrix (precision matrix)
    """
    if mode == "full":
        # Standard full covariance with regularization
        cov = np.cov(embeddings, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6
        inv_cov = np.linalg.inv(cov)

    elif mode == "shrinkage":
        # Ledoit-Wolf shrinkage covariance
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf().fit(embeddings)
        cov = lw.covariance_
        inv_cov = lw.precision_

    elif mode == "diagonal":
        # Diagonal covariance (assumes independence)
        var = np.var(embeddings, axis=0)
        cov = np.diag(var)
        inv_cov = np.diag(1.0 / (var + 1e-8))

    else:
        raise ValueError(
            f"Unknown covariance mode: {mode}. Choose from ['full', 'shrinkage', 'diagonal']"
        )

    return cov, inv_cov


def _apply_pca_transform(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    pca_variance: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray, object]:
    """
    Apply PCA dimensionality reduction to embeddings.

    Args:
        train_embeddings: Training embeddings (N_train, latent_dim)
        test_embeddings: Test embeddings (N_test, latent_dim)
        pca_variance: Variance to retain (default 0.99)

    Returns:
        train_reduced: Transformed training embeddings
        test_reduced: Transformed test embeddings
        pca: Fitted PCA object
    """
    from sklearn.decomposition import PCA

    # Fit PCA on training data only
    pca = PCA(n_components=pca_variance, svd_solver="full")
    train_reduced = pca.fit_transform(train_embeddings)
    test_reduced = pca.transform(test_embeddings)

    logger.info(
        f"PCA: Reduced from {train_embeddings.shape[1]} to {train_reduced.shape[1]} dimensions "
        f"(explained variance: {pca.explained_variance_ratio_.sum():.4f})"
    )

    return train_reduced, test_reduced, pca


def _compute_mahalanobis_distances_advanced(
    test_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    covariance_mode: str = "full",
) -> np.ndarray:
    """
    Compute Mahalanobis distances with configurable covariance estimation.

    Args:
        test_embeddings: Test embeddings (N_test, latent_dim)
        reference_embeddings: Reference embeddings (N_ref, latent_dim)
        covariance_mode: Method for covariance estimation

    Returns:
        distances: Array of Mahalanobis distances (N_test,)
    """
    from scipy.spatial import distance

    # Compute covariance and inverse
    cov, inv_cov = _compute_covariance_and_inverse(reference_embeddings, mode=covariance_mode)
    mean_ref = np.mean(reference_embeddings, axis=0)

    # Compute Mahalanobis distance for each test embedding
    distances = np.array(
        [distance.mahalanobis(test_emb, mean_ref, inv_cov) for test_emb in test_embeddings]
    )

    return distances


def compute_ood_scores(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_ids: Optional[List] = None,
    test_ids: Optional[List] = None,
    metric: str = "mahalanobis",
    charge_conditional: bool = False,
    use_pca: bool = False,
    pca_variance: float = 0.99,
    covariance_mode: str = "full",
    n_neighbors: int = 1,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """
    Compute OOD scores with flexible configuration options.

    This function provides a modular interface for computing out-of-distribution
    scores with optional preprocessing (PCA) and various distance metrics.

    Args:
        train_embeddings: Training embeddings (N_train, latent_dim) or list
        test_embeddings: Test embeddings (N_test, latent_dim) or list
        train_ids: Training peptidoform IDs in format "sequence/charge" (optional)
        test_ids: Test peptidoform IDs in format "sequence/charge" (required if charge_conditional=True)
        metric: Distance metric ('mahalanobis', 'euclidean', or 'knn')
        charge_conditional: If True, compute distances separately per charge state
        use_pca: If True, apply PCA dimensionality reduction
        pca_variance: Variance to retain in PCA (default 0.99)
        covariance_mode: Covariance estimation for Mahalanobis distance
            - 'full': Full covariance with regularization (default)
            - 'shrinkage': Ledoit-Wolf shrinkage estimation
            - 'diagonal': Diagonal approximation (fast)
        n_neighbors: Number of neighbors for k-NN method

    Returns:
        scores: Array of OOD scores (N_test,)

        Or if use_pca=True, returns:
        scores: Array of OOD scores
        info: Dictionary with PCA information

    Example:
        >>> # Basic usage
        >>> scores = compute_ood_scores(
        ...     train_embeddings, test_embeddings,
        ...     metric="mahalanobis"
        ... )
        >>>
        >>> # With PCA and shrinkage covariance
        >>> scores, info = compute_ood_scores(
        ...     train_embeddings, test_embeddings,
        ...     metric="mahalanobis",
        ...     use_pca=True,
        ...     pca_variance=0.95,
        ...     covariance_mode="shrinkage"
        ... )
        >>>
        >>> # Charge-conditional k-NN
        >>> scores = compute_ood_scores(
        ...     train_embeddings, test_embeddings,
        ...     train_ids=train_ids, test_ids=test_ids,
        ...     metric="knn", n_neighbors=5,
        ...     charge_conditional=True
        ... )
    """
    # Ensure inputs are numpy arrays
    if isinstance(train_embeddings, list):
        train_embeddings = np.array(train_embeddings)
    if isinstance(test_embeddings, list):
        test_embeddings = np.array(test_embeddings)

    # Store original embeddings for reference
    train_orig = train_embeddings
    test_orig = test_embeddings

    # Apply PCA if requested
    pca_info = None
    if use_pca:
        train_embeddings, test_embeddings, pca = _apply_pca_transform(
            train_embeddings, test_embeddings, pca_variance
        )
        pca_info = {
            "n_components": pca.n_components_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "total_variance": pca.explained_variance_ratio_.sum(),
            "pca_object": pca,
        }

    # For charge-conditional computation, need to handle separately
    if charge_conditional:
        if train_ids is None or test_ids is None:
            raise ValueError("Charge-conditional OOD scoring requires both train_ids and test_ids")

        # Create reference dict
        reference = {
            "embeddings": train_embeddings,
            "ids": np.array(train_ids) if not isinstance(train_ids, np.ndarray) else train_ids,
        }

        # For Mahalanobis with advanced covariance, we need custom handling
        if metric == "mahalanobis" and covariance_mode != "full":
            # Extract charge states
            test_charges = np.array([_extract_charge_from_id(id_) for id_ in test_ids])
            ref_charges = np.array([_extract_charge_from_id(id_) for id_ in reference["ids"]])

            unique_charges = np.unique(test_charges)
            scores = np.zeros(len(test_embeddings))

            logger.info(
                f"Computing {metric} ({covariance_mode}) distances for "
                f"{len(unique_charges)} charge states: {unique_charges}"
            )

            for charge in unique_charges:
                test_mask = test_charges == charge
                ref_mask = ref_charges == charge

                test_emb_charge = test_embeddings[test_mask]
                ref_emb_charge = train_embeddings[ref_mask]

                if len(ref_emb_charge) < 2:
                    logger.warning(
                        f"Charge {charge}: Only {len(ref_emb_charge)} reference samples. "
                        f"Using global distribution."
                    )
                    ref_emb_charge = train_embeddings

                # Compute distances with advanced covariance
                scores_charge = _compute_mahalanobis_distances_advanced(
                    test_emb_charge, ref_emb_charge, covariance_mode
                )
                scores[test_mask] = scores_charge
        else:
            # Use standard compute_distance for other cases
            scores = compute_distance(
                test_embeddings=test_embeddings,
                test_ids=test_ids,
                reference_embeddings=reference,
                method=metric,
                charge_conditional=True,
                n_neighbors=n_neighbors,
            )
    else:
        # Global (non-charge-conditional) computation
        if metric == "mahalanobis":
            scores = _compute_mahalanobis_distances_advanced(
                test_embeddings, train_embeddings, covariance_mode
            )
        elif metric == "euclidean":
            scores = _compute_euclidean_distances(test_embeddings, train_embeddings)
        elif metric == "knn":
            scores = _compute_knn_distances(test_embeddings, train_embeddings, n_neighbors)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    # Return scores with optional PCA info
    if use_pca:
        return scores, pca_info
    else:
        return scores
