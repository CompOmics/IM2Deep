"""Architecture definition for the multi-conformer IM2Deep model."""

import logging
from pathlib import Path

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam  # type: ignore[import]

from im2deep._architectures.blocks import Conv1dActivation, DenseActivation
from im2deep._architectures.helpers import calculate_concat_shape
from im2deep._architectures.im2deep_single import IM2Deep
from im2deep._architectures.losses import LowestMAESorted, MeanMAESorted
from im2deep.constants import BASEMODELCONFIG

LOGGER = logging.getLogger(__name__)
PACKAGE_DATA_PATH = Path(__file__).parent / "package_data"


class IM2DeepMulti(L.LightningModule):
    def __init__(self, config, criterion):
        super().__init__()
        self.config = config
        self.criterion = criterion

        initi = self.configure_init()

        self.ConvAtomComp = nn.ModuleList()
        self.ConvAtomComp.append(
            Conv1dActivation(
                6,
                self.config["AtomComp_out_channels_start"],
                self.config["AtomComp_kernel_size"],
                padding="same",
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.ConvAtomComp.append(
            Conv1dActivation(
                self.config["AtomComp_out_channels_start"],
                self.config["AtomComp_out_channels_start"],
                self.config["AtomComp_kernel_size"],
                padding="same",
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.ConvAtomComp.append(
            nn.MaxPool1d(
                self.config["AtomComp_MaxPool_kernel_size"],
                self.config["AtomComp_MaxPool_kernel_size"],
            )
        )
        self.ConvAtomComp.append(
            Conv1dActivation(
                self.config["AtomComp_out_channels_start"],
                self.config["AtomComp_out_channels_start"] // 2,
                self.config["AtomComp_kernel_size"],
                padding="same",
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.ConvAtomComp.append(
            Conv1dActivation(
                self.config["AtomComp_out_channels_start"] // 2,
                self.config["AtomComp_out_channels_start"] // 2,
                self.config["AtomComp_kernel_size"],
                padding="same",
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.ConvAtomComp.append(
            nn.MaxPool1d(
                self.config["AtomComp_MaxPool_kernel_size"],
                self.config["AtomComp_MaxPool_kernel_size"],
            )
        )
        self.ConvAtomComp.append(
            Conv1dActivation(
                self.config["AtomComp_out_channels_start"] // 2,
                self.config["AtomComp_out_channels_start"] // 4,
                self.config["AtomComp_kernel_size"],
                padding="same",
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.ConvAtomComp.append(
            Conv1dActivation(
                self.config["AtomComp_out_channels_start"] // 4,
                self.config["AtomComp_out_channels_start"] // 4,
                self.config["AtomComp_kernel_size"],
                padding="same",
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.ConvAtomComp.append(nn.Flatten())

        self.ConvDiatomComp = nn.ModuleList()
        self.ConvDiatomComp.append(
            Conv1dActivation(
                6,
                self.config["DiatomComp_out_channels_start"],
                self.config["DiatomComp_kernel_size"],
                padding="same",
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.ConvDiatomComp.append(
            Conv1dActivation(
                self.config["DiatomComp_out_channels_start"],
                self.config["DiatomComp_out_channels_start"],
                self.config["DiatomComp_kernel_size"],
                padding="same",
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.ConvDiatomComp.append(
            nn.MaxPool1d(
                self.config["DiatomComp_MaxPool_kernel_size"],
                self.config["DiatomComp_MaxPool_kernel_size"],
            )
        )
        self.ConvDiatomComp.append(
            Conv1dActivation(
                self.config["DiatomComp_out_channels_start"],
                self.config["DiatomComp_out_channels_start"] // 2,
                self.config["DiatomComp_kernel_size"],
                padding="same",
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.ConvDiatomComp.append(
            Conv1dActivation(
                self.config["DiatomComp_out_channels_start"] // 2,
                self.config["DiatomComp_out_channels_start"] // 2,
                self.config["DiatomComp_kernel_size"],
                padding="same",
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.ConvDiatomComp.append(nn.Flatten())

        self.ConvGlobal = nn.ModuleList()
        self.ConvGlobal.append(
            DenseActivation(
                60,
                self.config["Global_units"],
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.ConvGlobal.append(
            DenseActivation(
                self.config["Global_units"],
                self.config["Global_units"],
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.ConvGlobal.append(
            DenseActivation(
                self.config["Global_units"],
                self.config["Global_units"],
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )

        self.OneHot = nn.ModuleList()
        self.OneHot.append(
            Conv1dActivation(
                20,
                self.config["OneHot_out_channels"],
                self.config["One_hot_kernel_size"],
                padding="same",
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.OneHot.append(
            Conv1dActivation(
                self.config["OneHot_out_channels"],
                self.config["OneHot_out_channels"],
                self.config["One_hot_kernel_size"],
                padding="same",
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.OneHot.append(
            nn.MaxPool1d(
                self.config["OneHot_MaxPool_kernel_size"],
                self.config["OneHot_MaxPool_kernel_size"],
            )
        )
        self.OneHot.append(nn.Flatten())

        if config["add_X_mol"]:
            self.MolDesc = nn.ModuleList()
            self.MolDesc.append(
                Conv1dActivation(
                    13,
                    self.config["Mol_out_channels_start"],
                    self.config["Mol_kernel_size"],
                    padding="same",
                    initializer=initi,
                    negative_slope=self.config["LRelu_negative_slope"],
                    saturation=self.config["LRelu_saturation"],
                )
            )
            self.MolDesc.append(
                Conv1dActivation(
                    self.config["Mol_out_channels_start"],
                    self.config["Mol_out_channels_start"],
                    self.config["Mol_kernel_size"],
                    padding="same",
                    initializer=initi,
                    negative_slope=self.config["LRelu_negative_slope"],
                    saturation=self.config["LRelu_saturation"],
                )
            )
            self.MolDesc.append(
                nn.MaxPool1d(
                    self.config["Mol_MaxPool_kernel_size"],
                    self.config["Mol_MaxPool_kernel_size"],
                )
            )
            self.MolDesc.append(
                Conv1dActivation(
                    self.config["Mol_out_channels_start"],
                    self.config["Mol_out_channels_start"] // 2,
                    self.config["Mol_kernel_size"],
                    padding="same",
                    initializer=initi,
                    negative_slope=self.config["LRelu_negative_slope"],
                    saturation=self.config["LRelu_saturation"],
                )
            )
            self.MolDesc.append(
                Conv1dActivation(
                    self.config["Mol_out_channels_start"] // 2,
                    self.config["Mol_out_channels_start"] // 2,
                    self.config["Mol_kernel_size"],
                    padding="same",
                    initializer=initi,
                    negative_slope=self.config["LRelu_negative_slope"],
                    saturation=self.config["LRelu_saturation"],
                )
            )
            self.MolDesc.append(
                nn.MaxPool1d(
                    self.config["Mol_MaxPool_kernel_size"],
                    self.config["Mol_MaxPool_kernel_size"],
                )
            )
            self.MolDesc.append(
                Conv1dActivation(
                    self.config["Mol_out_channels_start"] // 2,
                    self.config["Mol_out_channels_start"] // 4,
                    self.config["Mol_kernel_size"],
                    padding="same",
                    initializer=initi,
                    negative_slope=self.config["LRelu_negative_slope"],
                    saturation=self.config["LRelu_saturation"],
                )
            )
            self.MolDesc.append(
                Conv1dActivation(
                    self.config["Mol_out_channels_start"] // 4,
                    self.config["Mol_out_channels_start"] // 4,
                    self.config["Mol_kernel_size"],
                    padding="same",
                    initializer=initi,
                    negative_slope=self.config["LRelu_negative_slope"],
                    saturation=self.config["LRelu_saturation"],
                )
            )
            self.MolDesc.append(nn.Flatten())

        self.total_input_size = calculate_concat_shape(self.config)
        LOGGER.debug(f"Total input size: {self.total_input_size}")

        self.Concat = nn.ModuleList()
        self.Concat.append(
            DenseActivation(
                self.total_input_size,
                self.config["Concat_units"],
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.Concat.append(
            DenseActivation(
                self.config["Concat_units"],
                self.config["Concat_units"],
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.Concat.append(
            DenseActivation(
                self.config["Concat_units"],
                self.config["Concat_units"],
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.Concat.append(
            DenseActivation(
                self.config["Concat_units"],
                self.config["Concat_units"],
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )
        self.Concat.append(
            DenseActivation(
                self.config["Concat_units"],
                self.config["Concat_units"],
                initializer=initi,
                negative_slope=self.config["LRelu_negative_slope"],
                saturation=self.config["LRelu_saturation"],
            )
        )

        self.concat_input_size = calculate_concat_shape(self.config)
        self.branches = nn.ModuleList(
            [
                _Branch(
                    self.config["Concat_units"],
                    config.get("BranchSize", 0),
                    add_layer=config.get("add_branch_layer", 0),
                ),
                _Branch(
                    self.config["Concat_units"],
                    config.get("BranchSize", 0),
                    add_layer=config.get("add_branch_layer", 0),
                ),
            ]
        )

    def forward(self, atom_comp, diatom_comp, global_feats, one_hot, mol_desc=None):
        atom_comp = atom_comp.permute(0, 2, 1)
        diatom_comp = diatom_comp.permute(0, 2, 1)
        one_hot = one_hot.permute(0, 2, 1)

        for layer in self.ConvAtomComp:
            atom_comp = layer(atom_comp)

        for layer in self.ConvDiatomComp:
            diatom_comp = layer(diatom_comp)

        for layer in self.ConvGlobal:
            global_feats = layer(global_feats)

        for layer in self.OneHot:
            one_hot = layer(one_hot)

        if self.config["add_X_mol"]:
            for layer in self.MolDesc:
                mol_desc = layer(mol_desc)

        concatenated = torch.cat((atom_comp, diatom_comp, one_hot, global_feats), 1)

        if self.config["add_X_mol"] and mol_desc is not None:
            concatenated = torch.cat((concatenated, mol_desc), 1)

        for layer in self.Concat:
            concatenated = layer(concatenated)

        y_hat1 = self.branches[0](concatenated)
        y_hat2 = self.branches[1](concatenated)

        return y_hat1, y_hat2

    def training_step(self, batch, batch_idx):
        if self.config["add_X_mol"]:
            atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
            y_hat1, y_hat2 = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc)
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            y_hat1, y_hat2 = self(atom_comp, diatom_comp, global_feats, one_hot)

        y1, y2 = y[:, 0], y[:, 1]

        loss = self.criterion(y1, y2, y_hat1, y_hat2)

        l1_norm = sum(p.abs().sum() for p in self.parameters())
        total_loss = loss + self.config["L1_alpha"] * l1_norm

        meanmae = MeanMAESorted(y1, y2, y_hat1, y_hat2)
        lowestmae = LowestMAESorted(y1, y2, y_hat1, y_hat2)

        self.log("Train Loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "Train Mean MAE", meanmae, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "Train Lowest MAE", lowestmae, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return total_loss

    def validation_step(self, batch, batch_idx):
        if self.config["add_X_mol"]:
            atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
            y_hat1, y_hat2 = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc)
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            y_hat1, y_hat2 = self(atom_comp, diatom_comp, global_feats, one_hot)

        y1, y2 = y[:, 0], y[:, 1]

        loss = self.criterion(y1, y2, y_hat1, y_hat2)
        meanmae = MeanMAESorted(y1, y2, y_hat1, y_hat2)
        lowestmae = LowestMAESorted(y1, y2, y_hat1, y_hat2)

        self.log("Val Loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("Val Mean MAE", meanmae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "Val Lowest MAE", lowestmae, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        if self.config["add_X_mol"]:
            atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
            y_hat1, y_hat2 = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc)
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            y_hat1, y_hat2 = self(atom_comp, diatom_comp, global_feats, one_hot)

        y1, y2 = y[:, 0], y[:, 1]

        loss = self.criterion(y1, y2, y_hat1, y_hat2)
        meanmae = MeanMAESorted(y1, y2, y_hat1, y_hat2)
        lowestmae = LowestMAESorted(y1, y2, y_hat1, y_hat2)

        self.log("Test Loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "Test Mean MAE", meanmae, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "Test Lowest MAE", lowestmae, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def predict_step(self, batch):
        if self.config["add_X_mol"]:
            atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
            y_hat1, y_hat2 = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc)
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            y_hat1, y_hat2 = self(atom_comp, diatom_comp, global_feats, one_hot)

        return torch.hstack([y_hat1, y_hat2])

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.config["learning_rate"])
        return optimizer

    def configure_init(self):
        if (not self.config["init"]) or (self.config["init"] == "normal"):
            return nn.init.normal_
        if self.config["init"] == "xavier":
            return nn.init.xavier_normal_
        if self.config["init"] == "kaiming":
            return nn.init.kaiming_normal_


class IM2DeepMultiTransfer(L.LightningModule):
    def __init__(self, config, criterion):
        super().__init__()
        # TODO: config should be adapted in config file
        self.config = config
        self.criterion = criterion
        self.l1_alpha = config["L1_alpha"]

        # Load the IM2Deep model
        LOGGER.debug("Loading backbone IM2Deep model")
        self.backbone = IM2Deep.load_from_checkpoint(
            config["backbone_SD_path"], config=config, criterion=criterion
        )

        self.ConvAtomComp = self.backbone.ConvAtomComp
        self.ConvDiatomComp = self.backbone.ConvDiatomComp
        self.ConvGlobal = self.backbone.ConvGlobal
        self.OneHot = self.backbone.OneHot

        if self.config.get("add_X_mol", False) == True:
            self.MolDesc = self.backbone.MolDesc

        self.concat = list(self.backbone.Concat.children())[:-1]

        self.concat_input_size = calculate_concat_shape(self.config)
        try:
            self.output_size = config["Concat_units"]
        except KeyError:
            self.output_size = BASEMODELCONFIG["Concat_units"]

        if self.config.get("Use_attention_concat", False):
            self.SelfAttentionConcat = SelfAttention(
                self.concat_input_size, config.get("Concatheads", 1)
            )
        if self.config.get("Use_attention_output", False):
            self.SelfAttentionOutput = SelfAttention(
                config["Concat_units"], config.get("Outputheads", 1)
            )

        self.branches = nn.ModuleList(
            [
                _Branch(
                    config["Concat_units"],
                    config.get("BranchSize", None),
                    add_layer=config.get("add_branch_layer", 0),
                ),
                _Branch(
                    config["Concat_units"],
                    config.get("BranchSize", None),
                    add_layer=config.get("add_branch_layer", 0),
                ),
            ]
        )

    def forward(self, atom_comp, diatom_comp, global_feats, one_hot, mol_desc=None):
        atom_comp = atom_comp.permute(0, 2, 1)
        diatom_comp = diatom_comp.permute(0, 2, 1)
        one_hot = one_hot.permute(0, 2, 1)

        for layer in self.ConvAtomComp:
            atom_comp = layer(atom_comp)

        for layer in self.ConvDiatomComp:
            diatom_comp = layer(diatom_comp)

        for layer in self.ConvGlobal:
            global_feats = layer(global_feats)

        for layer in self.OneHot:
            one_hot = layer(one_hot)

        if self.config["add_X_mol"]:
            for layer in self.MolDesc:
                mol_desc = layer(mol_desc)

        concatenated = torch.cat((atom_comp, diatom_comp, one_hot, global_feats), 1)

        if self.config["add_X_mol"] and mol_desc is not None:
            concatenated = torch.cat((concatenated, mol_desc), 1)

        if self.config.get("Use_attention_concat", 0) == 1:
            concatenated = self.SelfAttentionConcat(concatenated.unsqueeze(1)).squeeze(1)

        for layer in self.concat:
            concatenated = layer(concatenated)

        if self.config.get("Use_attention_output", 0) == 1:
            concatenated = self.SelfAttentionOutput(concatenated.unsqueeze(1)).squeeze(1)

        y_hat1 = self.branches[0](concatenated)
        y_hat2 = self.branches[1](concatenated)

        return y_hat1, y_hat2

    def training_step(self, batch, batch_idx):
        if self.config["add_X_mol"]:
            atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
            y_hat1, y_hat2 = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc)
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            y_hat1, y_hat2 = self(atom_comp, diatom_comp, global_feats, one_hot)

        y1, y2 = y[:, 0], y[:, 1]

        loss = self.criterion(y1, y2, y_hat1, y_hat2)

        l1_norm = sum(p.abs().sum() for p in self.parameters())
        total_loss = loss + self.l1_alpha * l1_norm

        meanmae = MeanMAESorted(y1, y2, y_hat1, y_hat2)
        lowestmae = LowestMAESorted(y1, y2, y_hat1, y_hat2)

        self.log(
            "Train Loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "Train Mean MAE", meanmae, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "Train Lowest MAE", lowestmae, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return total_loss

    def validation_step(self, batch, batch_idx):
        if self.config["add_X_mol"]:
            atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
            y_hat1, y_hat2 = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc)
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            y_hat1, y_hat2 = self(atom_comp, diatom_comp, global_feats, one_hot)

        y1, y2 = y[:, 0], y[:, 1]

        loss = self.criterion(y1, y2, y_hat1, y_hat2)

        meanmae = MeanMAESorted(y1, y2, y_hat1, y_hat2)
        lowestmae = LowestMAESorted(y1, y2, y_hat1, y_hat2)

        self.log("Val Loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("Val Mean MAE", meanmae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "Val Lowest MAE", lowestmae, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        if self.config["add_X_mol"]:
            atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
            y_hat1, y_hat2 = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc)
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            y_hat1, y_hat2 = self(atom_comp, diatom_comp, global_feats, one_hot)

        y1, y2 = y[:, 0], y[:, 1]

        loss = self.criterion(y1, y2, y_hat1, y_hat2)
        meanmae = MeanMAESorted(y1, y2, y_hat1, y_hat2)
        lowestmae = LowestMAESorted(y1, y2, y_hat1, y_hat2)

        self.log("Test Loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "Test Mean MAE", meanmae, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "Test Lowest MAE", lowestmae, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def predict_step(self, batch, inference=False):
        if self.config["add_X_mol"]:
            if not inference:
                atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
            else:
                atom_comp, diatom_comp, global_feats, one_hot, mol_desc = batch
            y_hat1, y_hat2 = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc)
        else:
            if not inference:
                atom_comp, diatom_comp, global_feats, one_hot, y = batch
            else:
                atom_comp, diatom_comp, global_feats, one_hot = batch
            y_hat1, y_hat2 = self(atom_comp, diatom_comp, global_feats, one_hot)
        return torch.hstack([y_hat1, y_hat2])

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.config["learning_rate"])
        return optimizer


class _Branch(nn.Module):
    def __init__(self, input_size, output_size, add_layer=1, dropout_rate=0.0):
        super().__init__()
        self.add_layer = add_layer
        if self.add_layer:
            self.fc1 = nn.Linear(input_size, output_size)
            # self.dropout = nn.Dropout(dropout_rate)
            self.fcoutput = nn.Linear(output_size, 1)
        else:
            self.fcoutput = nn.Linear(input_size, 1)

    def forward(self, x):
        if self.add_layer == 1:
            x = F.relu(self.fc1(x))
            # x = self.dropout(x)
        x = self.fcoutput(x)

        return x


class SelfAttention(nn.Module):
    def __init__(self, feature_dim, heads=1):
        super().__init__()
        self.feature_dim = feature_dim
        self.heads = heads
        # self.padded_dim = self.feature_dim + (self.feature_dim % self.heads)
        self.query_dim = self.feature_dim // self.heads
        self.extra_dim = self.feature_dim % self.heads

        self.query = nn.Linear(
            self.feature_dim,
            (self.query_dim + (self.extra_dim if self.extra_dim > 0 else 0)) * self.heads,
        )
        self.key = nn.Linear(
            self.feature_dim,
            (self.query_dim + (self.extra_dim if self.extra_dim > 0 else 0)) * self.heads,
        )
        self.value = nn.Linear(
            self.feature_dim,
            (self.query_dim + (self.extra_dim if self.extra_dim > 0 else 0)) * self.heads,
        )

        self.fc_out = nn.Linear(self.feature_dim, self.feature_dim)

    def forward(self, x):

        batch_size, seq_len, feature_dim = x.size()
        queries = self.query(x).view(
            batch_size,
            seq_len,
            self.heads,
            self.query_dim + (self.extra_dim if self.extra_dim > 0 else 0),
        )
        keys = self.key(x).view(
            batch_size,
            seq_len,
            self.heads,
            self.query_dim + (self.extra_dim if self.extra_dim > 0 else 0),
        )
        values = self.value(x).view(
            batch_size,
            seq_len,
            self.heads,
            self.query_dim + (self.extra_dim if self.extra_dim > 0 else 0),
        )

        attention_scores = torch.einsum("bqhd,bkhd->bhqk", [queries, keys]) / (self.query_dim**0.5)
        attention_scores = F.softmax(attention_scores, dim=-1)

        out = torch.einsum("bhqk,bkhd->bqhd", [attention_scores, values])

        out = out.view(
            batch_size,
            seq_len,
            self.heads * (self.query_dim + (self.extra_dim if self.extra_dim > 0 else 0)),
        )
        out = out[:, :, : self.feature_dim]
        out = self.fc_out(out)
        return out
