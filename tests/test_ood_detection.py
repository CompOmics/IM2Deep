"""
Tests for OOD detection module (latent embedding extraction).
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import tempfile

from im2deep.ood_detection import (
    extract_embeddings,
    embed_peptides,
    save_embeddings,
    load_embeddings,
    get_embedding_statistics,
    _validate_embeddings,
    LatentExtractorMixin,
)


class MockIM2DeepModel(nn.Module):
    """Mock IM2Deep model for testing."""

    def __init__(self, concat_units=94, add_x_mol=False):
        super().__init__()
        self.config = {"Concat_units": concat_units, "add_X_mol": add_x_mol}

        # Mock convolutional layers
        self.ConvAtomComp = nn.ModuleList([nn.Conv1d(6, 64, 3, padding=1), nn.Flatten()])
        self.ConvDiatomComp = nn.ModuleList([nn.Conv1d(6, 32, 3, padding=1), nn.Flatten()])
        self.ConvGlobal = nn.ModuleList([nn.Identity()])
        self.OneHot = nn.ModuleList([nn.Conv1d(20, 16, 3, padding=1), nn.Flatten()])

        # Calculate concat size (simplified)
        self.total_input_size = 64 * 60 + 32 * 30 + 20 + 16 * 60

        # Mock dense layers
        self.Concat = nn.ModuleList(
            [
                nn.Linear(self.total_input_size, concat_units),
                nn.ReLU(),
                nn.Linear(concat_units, concat_units),
                nn.ReLU(),
                nn.Linear(concat_units, 1),
            ]
        )

    def forward(self, atom_comp, diatom_comp, global_feats, one_hot, mol_desc=None):
        atom_comp = atom_comp.permute(0, 2, 1)
        diatom_comp = diatom_comp.permute(0, 2, 1)
        one_hot = one_hot.permute(0, 2, 1)

        atom_comp = self.ConvAtomComp[0](atom_comp)
        atom_comp = self.ConvAtomComp[1](atom_comp)

        diatom_comp = self.ConvDiatomComp[0](diatom_comp)
        diatom_comp = self.ConvDiatomComp[1](diatom_comp)

        global_feats = self.ConvGlobal[0](global_feats)

        one_hot = self.OneHot[0](one_hot)
        one_hot = self.OneHot[1](one_hot)

        concatenated = torch.cat((atom_comp, diatom_comp, one_hot, global_feats), 1)

        for layer in self.Concat:
            concatenated = layer(concatenated)

        return concatenated


class MockIM2DeepTransferModel(nn.Module):
    """Mock IM2DeepMultiTransfer model for testing (uses lowercase 'concat')."""

    def __init__(self, concat_units=94):
        super().__init__()
        self.config = {"Concat_units": concat_units, "add_X_mol": False}

        # Mock convolutional layers (same as regular model)
        self.ConvAtomComp = nn.ModuleList([nn.Conv1d(6, 64, 3, padding=1), nn.Flatten()])
        self.ConvDiatomComp = nn.ModuleList([nn.Conv1d(6, 32, 3, padding=1), nn.Flatten()])
        self.ConvGlobal = nn.ModuleList([nn.Identity()])
        self.OneHot = nn.ModuleList([nn.Conv1d(20, 16, 3, padding=1), nn.Flatten()])

        # Calculate concat size
        self.total_input_size = 64 * 60 + 32 * 30 + 20 + 16 * 60

        # Use lowercase 'concat' (plain list) like IM2DeepMultiTransfer
        self.concat = [
            nn.Linear(self.total_input_size, concat_units),
            nn.ReLU(),
            nn.Linear(concat_units, concat_units),
            nn.ReLU(),
        ]

        # Multi-output branches
        self.branches = nn.ModuleList([nn.Linear(concat_units, 1), nn.Linear(concat_units, 1)])

    def forward(self, atom_comp, diatom_comp, global_feats, one_hot, mol_desc=None):
        atom_comp = atom_comp.permute(0, 2, 1)
        diatom_comp = diatom_comp.permute(0, 2, 1)
        one_hot = one_hot.permute(0, 2, 1)

        atom_comp = self.ConvAtomComp[0](atom_comp)
        atom_comp = self.ConvAtomComp[1](atom_comp)

        diatom_comp = self.ConvDiatomComp[0](diatom_comp)
        diatom_comp = self.ConvDiatomComp[1](diatom_comp)

        global_feats = self.ConvGlobal[0](global_feats)

        one_hot = self.OneHot[0](one_hot)
        one_hot = self.OneHot[1](one_hot)

        concatenated = torch.cat((atom_comp, diatom_comp, one_hot, global_feats), 1)

        for layer in self.concat:
            concatenated = layer(concatenated)

        y_hat1 = self.branches[0](concatenated)
        y_hat2 = self.branches[1](concatenated)

        return y_hat1, y_hat2


@pytest.fixture
def mock_model():
    """Create a mock IM2Deep model."""
    model = MockIM2DeepModel(concat_units=94)
    model.eval()
    return model


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader with dummy data."""
    batch_size = 32
    n_samples = 100

    atom_comp = torch.randn(n_samples, 6, 60)
    diatom_comp = torch.randn(n_samples, 6, 30)
    global_feats = torch.randn(n_samples, 20)
    one_hot = torch.randn(n_samples, 20, 60)
    targets = torch.randn(n_samples, 1)

    dataset = TensorDataset(atom_comp, diatom_comp, global_feats, one_hot, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader


@pytest.fixture
def mock_data_dict():
    """Create mock data dictionary for embed_peptides."""
    n_samples = 50
    return {
        "X_AtomEnc": np.random.randn(n_samples, 6, 60).astype(np.float32),
        "X_DiAminoAtomEnc": np.random.randn(n_samples, 6, 30).astype(np.float32),
        "X_GlobalFeatures": np.random.randn(n_samples, 20).astype(np.float32),
        "X_OneHot": np.random.randn(n_samples, 20, 60).astype(np.float32),
    }


def test_latent_extractor_mixin(mock_model):
    """Test that mixin adds forward_with_latent method."""
    # Add mixin
    mock_model.forward_with_latent = LatentExtractorMixin.forward_with_latent.__get__(
        mock_model, type(mock_model)
    )

    assert hasattr(mock_model, "forward_with_latent")

    # Test forward pass
    atom_comp = torch.randn(4, 6, 60)
    diatom_comp = torch.randn(4, 6, 30)
    global_feats = torch.randn(4, 20)
    one_hot = torch.randn(4, 20, 60)

    output, latent = mock_model.forward_with_latent(atom_comp, diatom_comp, global_feats, one_hot)

    assert output.shape == (4, 1)
    assert latent.shape == (4, 94)  # Concat_units


def test_extract_embeddings(mock_model, mock_dataloader):
    """Test embedding extraction from dataloader."""
    embeddings, ids = extract_embeddings(
        mock_model, mock_dataloader, device="cpu", return_predictions=False
    )

    assert isinstance(embeddings, np.ndarray)
    assert isinstance(ids, list)
    assert embeddings.shape[0] == 100  # n_samples
    assert embeddings.shape[1] == 94  # concat_units
    assert len(ids) == 100


def test_extract_embeddings_with_predictions(mock_model, mock_dataloader):
    """Test embedding extraction with predictions."""
    embeddings, ids, predictions = extract_embeddings(
        mock_model, mock_dataloader, device="cpu", return_predictions=True
    )

    assert embeddings.shape[0] == 100
    assert predictions.shape[0] == 100
    assert len(ids) == 100


def test_embed_peptides(mock_model, mock_data_dict):
    """Test embedding generation for new peptides."""
    embeddings = embed_peptides(mock_model, mock_data_dict, device="cpu", batch_size=16)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 50  # n_samples
    assert embeddings.shape[1] == 94  # concat_units


def test_save_and_load_embeddings():
    """Test saving and loading embeddings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data
        embeddings = np.random.randn(100, 94)
        ids = list(range(100))
        metadata = {"test": "value", "number": 42}

        # Save
        save_path = Path(tmpdir) / "test_embeddings.npz"
        save_embeddings(save_path, embeddings, ids, metadata)

        assert save_path.exists()

        # Load
        loaded_data = load_embeddings(save_path)

        assert "embeddings" in loaded_data
        assert "ids" in loaded_data
        assert np.allclose(loaded_data["embeddings"], embeddings)
        assert np.array_equal(loaded_data["ids"], ids)


def test_validate_embeddings():
    """Test embedding validation."""
    # Valid embeddings
    valid_embeddings = np.random.randn(100, 50)
    _validate_embeddings(valid_embeddings)  # Should not raise

    # NaN values
    nan_embeddings = valid_embeddings.copy()
    nan_embeddings[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        _validate_embeddings(nan_embeddings)

    # Infinite values
    inf_embeddings = valid_embeddings.copy()
    inf_embeddings[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        _validate_embeddings(inf_embeddings)

    # All identical (zero variance)
    zero_var_embeddings = np.ones((100, 50))
    with pytest.raises(ValueError, match="zero variance in all dimensions"):
        _validate_embeddings(zero_var_embeddings)


def test_get_embedding_statistics():
    """Test embedding statistics computation."""
    embeddings = np.random.randn(100, 50)
    stats = get_embedding_statistics(embeddings)

    assert "n_samples" in stats
    assert "embedding_dim" in stats
    assert "mean" in stats
    assert "std" in stats
    assert "min" in stats
    assert "max" in stats
    assert "median" in stats
    assert "zero_variance_dims" in stats

    assert stats["n_samples"] == 100
    assert stats["embedding_dim"] == 50
    assert isinstance(stats["mean"], float)


def test_deterministic_embeddings(mock_model, mock_data_dict):
    """Test that embedding extraction is deterministic."""
    mock_model.eval()

    # Extract twice
    embeddings1 = embed_peptides(mock_model, mock_data_dict, device="cpu", batch_size=16)
    embeddings2 = embed_peptides(mock_model, mock_data_dict, device="cpu", batch_size=16)

    # Should be identical
    assert np.allclose(embeddings1, embeddings2)


def test_batch_size_independence(mock_model, mock_data_dict):
    """Test that different batch sizes give same results."""
    mock_model.eval()

    embeddings_batch8 = embed_peptides(mock_model, mock_data_dict, device="cpu", batch_size=8)
    embeddings_batch16 = embed_peptides(mock_model, mock_data_dict, device="cpu", batch_size=16)
    embeddings_batch32 = embed_peptides(mock_model, mock_data_dict, device="cpu", batch_size=32)

    # Should all be identical
    assert np.allclose(embeddings_batch8, embeddings_batch16)
    assert np.allclose(embeddings_batch16, embeddings_batch32)


def test_no_gradient_computation(mock_model, mock_dataloader):
    """Test that no gradients are computed during extraction."""
    # Enable gradient computation
    torch.set_grad_enabled(True)

    # Extract embeddings
    embeddings, ids = extract_embeddings(mock_model, mock_dataloader, device="cpu")

    # Verify model parameters have no gradients
    for param in mock_model.parameters():
        assert param.grad is None


def test_transfer_model_with_lowercase_concat():
    """Test that transfer models with lowercase 'concat' work correctly."""
    model = MockIM2DeepTransferModel(concat_units=94)
    model.eval()

    # Add latent extraction
    model.forward_with_latent = LatentExtractorMixin.forward_with_latent.__get__(
        model, type(model)
    )

    # Create test data
    atom_comp = torch.randn(4, 6, 60)
    diatom_comp = torch.randn(4, 6, 30)
    global_feats = torch.randn(4, 20)
    one_hot = torch.randn(4, 20, 60)

    # Extract
    output, latent = model.forward_with_latent(atom_comp, diatom_comp, global_feats, one_hot)

    # Verify shapes
    assert latent.shape == (4, 94), f"Wrong latent shape: {latent.shape}"
    assert output.shape == (4, 2), f"Wrong output shape: {output.shape}"  # Multi-output


def test_transfer_model_extraction_from_dataloader():
    """Test embedding extraction from transfer model via dataloader."""
    model = MockIM2DeepTransferModel(concat_units=64)
    model.eval()

    # Create dataloader
    n_samples = 50
    atom_comp = torch.randn(n_samples, 6, 60)
    diatom_comp = torch.randn(n_samples, 6, 30)
    global_feats = torch.randn(n_samples, 20)
    one_hot = torch.randn(n_samples, 20, 60)
    targets = torch.randn(n_samples, 2)  # Multi-output targets

    dataset = TensorDataset(atom_comp, diatom_comp, global_feats, one_hot, targets)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Extract
    embeddings, ids, predictions = extract_embeddings(
        model, dataloader, device="cpu", return_predictions=True
    )

    # Verify
    assert embeddings.shape == (n_samples, 64)
    assert predictions.shape == (n_samples, 2)  # Two conformer predictions
    assert len(ids) == n_samples


def test_both_model_types_produce_consistent_embeddings():
    """Test that both standard and transfer models extract embeddings consistently."""
    # Same config
    concat_units = 64

    # Create both model types
    standard_model = MockIM2DeepModel(concat_units=concat_units)
    transfer_model = MockIM2DeepTransferModel(concat_units=concat_units)

    # Copy weights from standard to transfer (first layer only for comparison)
    with torch.no_grad():
        transfer_model.concat[0].weight.copy_(standard_model.Concat[0].weight)
        transfer_model.concat[0].bias.copy_(standard_model.Concat[0].bias)

    standard_model.eval()
    transfer_model.eval()

    # Same input data
    atom_comp = torch.randn(8, 6, 60)
    diatom_comp = torch.randn(8, 6, 30)
    global_feats = torch.randn(8, 20)
    one_hot = torch.randn(8, 20, 60)

    # Add latent extraction to both
    standard_model.forward_with_latent = LatentExtractorMixin.forward_with_latent.__get__(
        standard_model, type(standard_model)
    )
    transfer_model.forward_with_latent = LatentExtractorMixin.forward_with_latent.__get__(
        transfer_model, type(transfer_model)
    )

    # Extract latent embeddings
    _, standard_latent = standard_model.forward_with_latent(
        atom_comp, diatom_comp, global_feats, one_hot
    )
    _, transfer_latent = transfer_model.forward_with_latent(
        atom_comp, diatom_comp, global_feats, one_hot
    )

    # Latent embeddings should be identical (same first layer)
    assert torch.allclose(
        standard_latent, transfer_latent, atol=1e-5
    ), "Latent embeddings differ between model types"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
