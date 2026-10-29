"""Architecture definition for the single-conformer IM2Deep model."""

import logging

import lightning as L
import torch
from torch import Tensor, nn
from torch.optim import Adam  # type: ignore[import]

from im2deep._architectures.blocks import Conv1dActivation, DenseActivation
from im2deep._architectures.helpers import calculate_concat_shape
from im2deep.constants import DEFAULT_GLOBAL_FEATURES

LOGGER = logging.getLogger(__name__)


class IM2Deep(L.LightningModule):
    def __init__(self, config, criterion):
        super().__init__()
        self.config = config
        self.criterion = criterion
        self.mae = nn.L1Loss()
        # Record the config in the checkpoint so a model can be read back with
        # the architecture and featurisation it was trained on. The bundled
        # checkpoints predate this and carry no hyperparameters, which is why
        # _model_ops still falls back to DEFAULT_CONFIG.
        self.save_hyperparameters("config")

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
                # Number of DeepLC global features, which depends on the
                # featurisation flags. The default matches
                # `add_ccs_features=True, add_terminal_composition=False`,
                # i.e. what every bundled checkpoint was trained with.
                self.config.get("Global_features", DEFAULT_GLOBAL_FEATURES),
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

        self.Concat.append(nn.Linear(self.config["Concat_units"], 1))

    def regularized_loss(self, y_hat, y):
        standard_loss = self.criterion(y_hat, y)
        l1_norm = sum(torch.norm(p, 1) for p in self.parameters())
        return standard_loss + self.config["L1_alpha"] * l1_norm

    def forward(
        self,
        atom_comp,
        diatom_comp,
        global_feats,
        one_hot,
        mol_desc: Tensor | None = None,
    ):
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
            if mol_desc is None:
                raise ValueError("`mol_desc` is required when `add_X_mol` is enabled.")
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
        optimizer = Adam(self.parameters(), lr=self.config["learning_rate"])
        return optimizer

    def configure_init(self):
        if (not self.config["init"]) or (self.config["init"] == "normal"):
            return nn.init.normal_
        if self.config["init"] == "xavier":
            return nn.init.xavier_normal_
        if self.config["init"] == "kaiming":
            return nn.init.kaiming_normal_


class IM2DeepTransfer(L.LightningModule):
    def __init__(self, config, criterion):
        super().__init__()

        self.config = config
        self.criterion = criterion
        self.l1_alpha = config["L1_alpha"]
        self.mae = nn.L1Loss()
        self.save_hyperparameters("config")

        # Load the IM2Deep model
        LOGGER.debug("Loading backbone IM2Deep model")
        self.backbone = IM2Deep.load_from_checkpoint(
            config["backbone_SD_path"], config=config, criterion=criterion
        )

        self.ConvAtomComp = self.backbone.ConvAtomComp
        self.ConvDiatomComp = self.backbone.ConvDiatomComp
        self.ConvGlobal = self.backbone.ConvGlobal
        self.OneHot = self.backbone.OneHot

        if bool(self.config.get("add_X_mol", False)):
            self.MolDesc = self.backbone.MolDesc

        self.concat = self.backbone.Concat

    def forward(
        self,
        atom_comp,
        diatom_comp,
        global_feats,
        one_hot,
        mol_desc: Tensor | None = None,
    ):
        atom_comp = atom_comp.permute(0, 2, 1)
        diatom_comp = diatom_comp.permute(0, 2, 1)
        one_hot = one_hot.permute(0, 2, 1)
        mol_desc_tensor: Tensor | None = None

        for layer in self.ConvAtomComp:
            atom_comp = layer(atom_comp)

        for layer in self.ConvDiatomComp:
            diatom_comp = layer(diatom_comp)

        for layer in self.ConvGlobal:
            global_feats = layer(global_feats)

        for layer in self.OneHot:
            one_hot = layer(one_hot)

        if self.config["add_X_mol"]:
            if mol_desc is None:
                raise ValueError("`mol_desc` is required when `add_X_mol` is enabled.")
            mol_desc_tensor = mol_desc
            for layer in self.MolDesc:
                mol_desc_tensor = layer(mol_desc_tensor)

        concatenated = torch.cat((atom_comp, diatom_comp, one_hot, global_feats), 1)

        if self.config["add_X_mol"]:
            if mol_desc_tensor is None:
                raise ValueError("`mol_desc` is required when `add_X_mol` is enabled.")
            concatenated = torch.cat((concatenated, mol_desc_tensor), 1)

        for layer in self.concat:
            concatenated = layer(concatenated)

        y_hat = concatenated
        return y_hat

    def training_step(self, batch, batch_idx):
        if self.config["add_X_mol"]:
            atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc).squeeze(1)
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
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc).squeeze(1)
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
            y_hat = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc).squeeze(1)
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
        optimizer = Adam(self.parameters(), lr=self.config["learning_rate"])
        return optimizer
