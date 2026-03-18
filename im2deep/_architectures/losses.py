"""Loss functions for IM2Deep models."""

import torch
from torch import nn

MAE = nn.L1Loss()


class FlexibleLossSorted(nn.Module):
    def __init__(self, diversity_weight=0.1):
        super().__init__()
        self.diversity_weight = diversity_weight

    def forward(self, y1, y2, y_hat1, y_hat2):
        loss_fn = nn.L1Loss()

        # Sort the targets and predictions row-wise
        targets = torch.stack([y1, y2], dim=1)
        predictions = torch.stack([y_hat1, y_hat2], dim=1)
        targets, _ = torch.sort(targets, dim=1)
        predictions, _ = torch.sort(predictions, dim=1)

        target1 = targets[:, 0]

        target2 = targets[:, 1]

        prediction1 = predictions[:, 0]
        prediction1 = prediction1.squeeze()

        prediction2 = predictions[:, 1]

        prediction2 = prediction2.squeeze()

        loss1 = loss_fn(prediction1.float(), target1.float())

        loss2 = loss_fn(prediction2.float(), target2.float())

        target_diff = torch.abs(target1 - target2)

        prediction_diff = torch.abs(prediction1 - prediction2)

        diff_loss = loss_fn(prediction_diff.float(), target_diff.float())

        total_loss = (loss1 + loss2) + (self.diversity_weight * diff_loss)

        return total_loss


class FlexibleLoss(nn.Module):
    def __init__(self, diversity_weight=0.1):
        super().__init__()
        self.diversity_weight = diversity_weight

    def forward(self, y1, y2, y_hat1, y_hat2):
        loss_fn = nn.L1Loss()

        loss1_to_1 = loss_fn(y_hat1, y1)
        loss2_to_2 = loss_fn(y_hat2, y2)
        loss1_to_2 = loss_fn(y_hat1, y2)
        loss2_to_1 = loss_fn(y_hat2, y1)

        loss_dict = {
            "1_to_1": loss1_to_1,
            "2_to_2": loss2_to_2,
            "1_to_2": loss1_to_2,
            "2_to_1": loss2_to_1,
        }
        min_loss_key = min(loss_dict, key=lambda k: loss_dict[k])
        if "1_to" in min_loss_key:
            if "to_1" in min_loss_key:
                loss1 = loss1_to_1
                loss2 = loss2_to_2
            else:
                loss1 = loss1_to_2
                loss2 = loss2_to_1
        else:
            if "to_2" in min_loss_key:
                loss1 = loss2_to_2
                loss2 = loss1_to_1
            else:
                loss1 = loss2_to_1
                loss2 = loss1_to_2

        target_diff = torch.abs(y1 - y2)
        prediction_diff = torch.abs(y_hat1 - y_hat2)

        diff_loss = loss_fn(prediction_diff, target_diff)

        total_loss = (loss1 + loss2) + (self.diversity_weight * diff_loss)

        return total_loss


def MeanMAESorted(y1, y2, y_hat1, y_hat2):
    targets = torch.stack([y1, y2], dim=1)
    predictions = torch.stack([y_hat1, y_hat2], dim=1)
    # predictions is shape [x,2,1] but should be [x,2]
    predictions = predictions.squeeze()

    targets, _ = torch.sort(targets, dim=1)
    predictions, _ = torch.sort(predictions, dim=1)

    target1 = targets[:, 0]
    target2 = targets[:, 1]

    prediction1 = predictions[:, 0]
    prediction2 = predictions[:, 1]

    mae1 = MAE(prediction1, target1)
    mae2 = MAE(prediction2, target2)

    return (mae1 + mae2) / 2


def LowestMAESorted(y1, y2, y_hat1, y_hat2):
    targets = torch.stack([y1, y2], dim=1)
    predictions = torch.stack([y_hat1, y_hat2], dim=1)
    predictions = predictions.squeeze()

    targets, _ = torch.sort(targets, dim=1)
    predictions, _ = torch.sort(predictions, dim=1)

    target1 = targets[:, 0]
    target2 = targets[:, 1]

    prediction1 = predictions[:, 0]
    prediction2 = predictions[:, 1]

    mae1 = MAE(prediction1, target1)
    mae2 = MAE(prediction2, target2)

    return min(mae1, mae2)


def MeanPearsonRSorted(y1, y2, y_hat1, y_hat2):
    """Compute the mean Pearson correlation between sorted targets and predictions."""
    # Reshape and sort the targets and predictions row-wise
    targets = torch.sort(torch.stack((y1, y2), dim=1).float(), dim=1).values
    predictions = torch.sort(
        torch.stack((y_hat1, y_hat2), dim=1).reshape_as(targets).float(),
        dim=1,
    ).values

    # Compute the correlation matrix and extract the mean of the diagonal correlations
    corr = torch.corrcoef(torch.cat((targets.T, predictions.T), dim=0))
    mean_corr = torch.diagonal(corr[:2, 2:]).mean()

    return torch.nan_to_num(mean_corr, nan=0.0)


def MeanMRE(y1, y2, y_hat1, y_hat2):
    """Compute the mean median relative error between predictions and targets."""
    mre1 = torch.median(torch.abs((y_hat1 - y1) / y1))
    mre2 = torch.median(torch.abs((y_hat2 - y2) / y2))
    return (mre1 + mre2) / 2
