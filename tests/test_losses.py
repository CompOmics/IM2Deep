"""Tests for loss utilities."""

import torch

from im2deep._architectures.losses import MeanPearsonRSorted


def test_mean_pearson_r_sorted_perfect_match_after_sorting():
    """Returns perfect correlation when sorted targets and predictions match."""
    y1 = torch.tensor([1.0, 3.0, 2.0])
    y2 = torch.tensor([2.0, 1.0, 3.0])

    # Intentionally cross-assigned per row; sorting should align them.
    y_hat1 = torch.tensor([2.0, 3.0, 2.0])
    y_hat2 = torch.tensor([1.0, 1.0, 3.0])

    score = MeanPearsonRSorted(y1, y2, y_hat1, y_hat2)

    assert torch.isclose(score, torch.tensor(1.0), atol=1e-6)


def test_mean_pearson_r_sorted_returns_zero_for_undefined_corr():
    """Returns 0 when Pearson correlation is undefined due to zero variance."""
    y1 = torch.tensor([1.0, 1.0, 1.0])
    y2 = torch.tensor([1.0, 1.0, 1.0])
    y_hat1 = torch.tensor([1.0, 1.0, 1.0])
    y_hat2 = torch.tensor([1.0, 1.0, 1.0])

    score = MeanPearsonRSorted(y1, y2, y_hat1, y_hat2)

    assert torch.isclose(score, torch.tensor(0.0), atol=1e-6)
