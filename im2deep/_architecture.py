import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import logging
import wandb

from im2deep.constants import (
    BASEMODELCONFIG,
)

PACKAGE_DATA_PATH = Path(__file__).parent / "package_data"

logger = logging.getLogger(__name__)


class LogLowestMAE(L.Callback):
    def __init__(self, config):
        super(LogLowestMAE, self).__init__()
        self.bestMAE = float("inf")
        self.config = config

    def on_validation_end(self, trainer, pl_module):
        try:
            currentMAE = trainer.callback_metrics["Validation MAE"]
        except KeyError:  # Multi
            currentMAE = trainer.callback_metrics["Val Mean MAE"]
        if currentMAE < self.bestMAE:
            self.bestMAE = currentMAE
        if self.config["wandb"]["enabled"]:
            wandb.log({"Best Val MAE": self.bestMAE})


class LRelu_with_saturation(nn.Module):
    def __init__(self, negative_slope, saturation):
        super(LRelu_with_saturation, self).__init__()
        self.negative_slope = negative_slope
        self.saturation = saturation
        self.leaky_relu = nn.LeakyReLU(self.negative_slope)

    def forward(self, x):
        activated = self.leaky_relu(x)
        return torch.clamp(activated, max=self.saturation)


class Conv1dActivation(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        initializer,
        negative_slope,
        saturation,
    ):
        super(Conv1dActivation, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.initializer = initializer
        self.activation = LRelu_with_saturation(
            negative_slope=negative_slope, saturation=saturation
        )

        initializer(self.conv.weight, 0.0, 0.05)

    def forward(self, x):
        return self.activation(self.conv(x))


class DenseActivation(nn.Module):
    def __init__(self, in_features, out_features, initializer, negative_slope, saturation):
        super(DenseActivation, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.initializer = initializer
        self.activation = LRelu_with_saturation(
            negative_slope=negative_slope, saturation=saturation
        )

        initializer(self.linear.weight, 0.0, 0.05)

    def forward(self, x):
        return self.activation(self.linear(x))


class SelfAttention(nn.Module):
    def __init__(self, feature_dim, heads=1):
        super(SelfAttention, self).__init__()
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


class Branch(nn.Module):
    def __init__(self, input_size, output_size, add_layer=1, dropout_rate=0.0):
        super(Branch, self).__init__()
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


class IM2Deep(L.LightningModule):
    def __init__(self, config, criterion):
        super(IM2Deep, self).__init__()
        self.config = config
        self.criterion = criterion
        self.mae = nn.L1Loss()

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
        logger.debug(f"Total input size: {self.total_input_size}")

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

        self.Concat.append(nn.Linear(self.config["Concat_units"], 1))

    def regularized_loss(self, y_hat, y):
        standard_loss = self.criterion(y_hat, y)
        l1_norm = sum(torch.norm(p, 1) for p in self.parameters())
        return standard_loss + self.config["L1_alpha"] * l1_norm

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

        if self.config["add_X_mol"]:
            concatenated = torch.cat((concatenated, mol_desc), 1)

        for layer in self.Concat:
            concatenated = layer(concatenated)

        output = concatenated
        return output

    def training_step(self, batch, batch_idx):
        if self.config["add_X_mol"]:
            atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc).squeeze(1)
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot).squeeze(1)

        loss = self.regularized_loss(y_hat, y)

        self.log("Train loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "Train MAE",
            self.mae(y_hat, y),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        if self.config["add_X_mol"]:
            atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc).squeeze(1)
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot).squeeze(1)
        loss = self.criterion(y_hat, y)

        self.log("Validation loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "Validation MAE",
            self.mae(y_hat, y),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        if self.config["add_X_mol"]:
            atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc).squeeze(1)
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot).squeeze(1)
        loss = self.criterion(y_hat, y)

        self.log("Test loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "Test MAE",
            self.mae(y_hat, y),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if self.config["add_X_mol"]:
            atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc).squeeze(1)
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot).squeeze(1)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        return optimizer

    def configure_init(self):
        if (not self.config["init"]) or (self.config["init"] == "normal"):
            return nn.init.normal_
        if self.config["init"] == "xavier":
            return nn.init.xavier_normal_
        if self.config["init"] == "kaiming":
            return nn.init.kaiming_normal_


class IM2DeepMulti(L.LightningModule):
    def __init__(self, config, criterion):
        super(IM2DeepMulti, self).__init__()
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
        logger.debug(f"Total input size: {self.total_input_size}")

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
                Branch(
                    self.config["Concat_units"],
                    config.get("BranchSize", 0),
                    add_layer=config.get("add_branch_layer", 0),
                ),
                Branch(
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

        if self.config["add_X_mol"]:
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])
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
        super(IM2DeepMultiTransfer, self).__init__()
        # TODO: config should be adapted in config file
        self.config = config
        self.criterion = criterion
        self.l1_alpha = config["L1_alpha"]

        # Load the IM2Deep model
        logger.debug("Loading backbone IM2Deep model")
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
                Branch(
                    config["Concat_units"],
                    config.get("BranchSize", None),
                    add_layer=config.get("add_branch_layer", 0),
                ),
                Branch(
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

        if self.config["add_X_mol"]:
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        return optimizer


class IM2DeepTransfer(L.LightningModule):
    def __init__(self, config, criterion):
        super(IM2DeepTransfer, self).__init__()

        self.config = config
        self.criterion = criterion
        self.l1_alpha = config["L1_alpha"]
        self.mae = nn.L1Loss()

        # Load the IM2Deep model
        logger.debug("Loading backbone IM2Deep model")
        self.backbone = IM2Deep.load_from_checkpoint(
            config["backbone_SD_path"], config=config, criterion=criterion
        )

        self.ConvAtomComp = self.backbone.ConvAtomComp
        self.ConvDiatomComp = self.backbone.ConvDiatomComp
        self.ConvGlobal = self.backbone.ConvGlobal
        self.OneHot = self.backbone.OneHot

        if self.config.get("add_X_mol", False) == True:
            self.MolDesc = self.backbone.MolDesc

        self.concat = self.backbone.Concat

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

        if self.config["add_X_mol"]:
            concatenated = torch.cat((concatenated, mol_desc), 1)

        for layer in self.concat:
            concatenated = layer(concatenated)

        y_hat = concatenated
        return y_hat

    def training_step(self, batch, batch_idx):
        if self.config["add_X_mol"]:
            atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc)
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot).squeeze(1)

        loss = self.criterion(y_hat, y)

        l1_norm = sum(p.abs().sum() for p in self.parameters())
        total_loss = loss + self.l1_alpha * l1_norm

        self.log(
            "Train Loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "Train MAE",
            self.mae(y_hat, y),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return total_loss

    def validation_step(self, batch, batch_idx):
        if self.config["add_X_mol"]:
            atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc)
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot).squeeze(1)

        loss = self.criterion(y_hat, y)

        self.log("Validation Loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "Validation MAE",
            self.mae(y_hat, y),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        if self.config["add_X_mol"]:
            atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc)
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot).squeeze(1)

        loss = self.criterion(y_hat, y)

        self.log("Test Loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "Test MAE",
            self.mae(y_hat, y),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def predict_step(self, batch, inference=False):
        if self.config["add_X_mol"]:
            if not inference:
                atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
            else:
                atom_comp, diatom_comp, global_feats, one_hot, mol_desc = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc).squeeze(1)
        else:
            if not inference:
                atom_comp, diatom_comp, global_feats, one_hot, y = batch
            else:
                atom_comp, diatom_comp, global_feats, one_hot = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot).squeeze(1)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        return optimizer


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
