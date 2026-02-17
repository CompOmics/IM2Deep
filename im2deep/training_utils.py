import torch
import torch.nn as nn
from scipy.stats import pearsonr
import logging
import sys

logger = logging.getLogger(__name__)

MAE = nn.L1Loss()

BASEMODELCONFIG = {
    "AtomComp_kernel_size": 4,
    "DiatomComp_kernel_size": 4,
    "One_hot_kernel_size": 4,
    "AtomComp_out_channels_start": 356,
    "DiatomComp_out_channels_start": 65,
    "Global_units": 20,
    "OneHot_out_channels": 1,
    "Concat_units": 94,
    "AtomComp_MaxPool_kernel_size": 2,
    "DiatomComp_MaxPool_kernel_size": 2,
    "OneHot_MaxPool_kernel_size": 10,
    "LRelu_negative_slope": 0.013545684190756122,
    "LRelu_saturation": 40,
    "init": "normal",
    "add_X_mol": False,
}


class FlexibleLossSorted(nn.Module):
    def __init__(self, diversity_weight=0.1):
        super(FlexibleLossSorted, self).__init__()
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
        super(FlexibleLoss, self).__init__()
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
        min_loss_key = min(loss_dict, key=loss_dict.get)
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
    targets = torch.stack([y1, y2], dim=1)
    predictions = torch.stack([y_hat1, y_hat2], dim=1)

    targets, _ = torch.sort(targets, dim=1)
    predictions, _ = torch.sort(predictions, dim=1)

    target1 = targets[:, 0]
    target2 = targets[:, 1]

    prediction1 = predictions[:, 0]
    prediction2 = predictions[:, 1]

    r1 = pearsonr(target1, prediction1)[0]
    r2 = pearsonr(target2, prediction2)[0]

    return (r1 + r2) / 2


def MeanMRE(y1, y2, y_hat1, y_hat2):
    mre1 = torch.median(torch.abs((y_hat1 - y1) / y1))
    mre2 = torch.median(torch.abs((y_hat2 - y2) / y2))
    return (mre1 + mre2) / 2


def calculate_concat_shape(config):
    atom_comp_out_shape = (60 // (2 * config["AtomComp_MaxPool_kernel_size"])) * (
        config["AtomComp_out_channels_start"] // 4
    )
    logger.debug(f"AtomComp out shape: {atom_comp_out_shape}")
    diatom_comp_out_shape = (30 // (config["DiatomComp_MaxPool_kernel_size"])) * (
        config["DiatomComp_out_channels_start"] // 2
    )
    logger.debug(f"DiatomComp out shape: {diatom_comp_out_shape}")
    globals_out_shape = config["Global_units"]
    logger.debug(f"Globals out shape: {globals_out_shape}")
    onehot_comp_out_shape = (60 // (config["OneHot_MaxPool_kernel_size"])) * config[
        "OneHot_out_channels"
    ]
    logger.debug(f"OneHot out shape: {onehot_comp_out_shape}")

    if config["add_X_mol"]:
        mol_desc_comp_out_shape = (60 // (2 * config["Mol_MaxPool_kernel_size"])) * (
            config["Mol_out_channels_start"] // 4
        )
        logger.debug(f"MolDesc out shape: {mol_desc_comp_out_shape}")
        total_input_size = (
            atom_comp_out_shape
            + diatom_comp_out_shape
            + globals_out_shape
            + onehot_comp_out_shape
            + mol_desc_comp_out_shape
        )

    else:
        total_input_size = (
            atom_comp_out_shape + diatom_comp_out_shape + globals_out_shape + onehot_comp_out_shape
        )

    return total_input_size
