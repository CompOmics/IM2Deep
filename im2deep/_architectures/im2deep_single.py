"""Architecture definition for the single-conformer IM2Deep model."""

import logging
from functools import partial

import lightning as L
import torch
from torch import Tensor, nn
from torch.optim import Adam, AdamW  # type: ignore[import]
from torch.optim.lr_scheduler import OneCycleLR

from im2deep._architectures.blocks import (
    AttentionPool1d,
    Conv1dActivation,
    DenseActivation,
    FlexTrunk,
)
from im2deep._architectures.helpers import calculate_concat_shape, calculate_seq_concat_shape
from im2deep._architectures.transformer_trunk import TransformerTrunk
from im2deep.constants import DEFAULT_GLOBAL_FEATURES

LOGGER = logging.getLogger(__name__)


CHARGE_FEATURE_INDEX = 11
LOG_B_CLAMP = 10.0
NUM_CHARGE_HEADS = 6


class IM2Deep(L.LightningModule):
    def __init__(self, config, criterion):
        super().__init__()
        self.config = config
        self.criterion = criterion
        self.mae = nn.L1Loss()
        self.save_hyperparameters("config")

        n_global = self.config.get("Global_features", DEFAULT_GLOBAL_FEATURES)
        self.register_buffer("global_feat_mean", torch.zeros(n_global))
        self.register_buffer("global_feat_std", torch.ones(n_global))
        self.register_buffer("target_mean", torch.zeros(()))
        self.register_buffer("target_std", torch.ones(()))

        initi = self.configure_init()

        self.use_attention_pooling = bool(self.config.get("use_attention_pooling", False))
        self.use_flex_trunk = bool(self.config.get("use_flex_trunk", False))
        self.use_transformer_trunk = bool(self.config.get("use_transformer_trunk", False))
        if self.use_flex_trunk and self.use_transformer_trunk:
            raise ValueError(
                "use_flex_trunk and use_transformer_trunk are mutually exclusive: both "
                "replace ConvAtomComp/ConvDiatomComp with a different mechanism."
            )
        if self.use_flex_trunk:
            self.flex_trunk = FlexTrunk(
                embed_dim=self.config.get("flex_trunk_embed_dim", 16),
                channels=self.config.get("flex_trunk_channels", (192, 192)),
                kernel_size=self.config.get("flex_trunk_kernel_size", 5),
                stem_channels=self.config.get("flex_trunk_stem_channels", 64),
                stem_layers=self.config.get("flex_trunk_stem_layers", 2),
            )
        elif self.use_transformer_trunk:
            if not self.config.get("add_ccs_features"):
                raise ValueError(
                    "use_transformer_trunk requires add_ccs_features=True: the charge "
                    "register token needs CHARGE_FEATURE_INDEX from matrix_global."
                )
            self.transformer_trunk = TransformerTrunk(
                d_model=self.config.get("transformer_d_model", 128),
                num_layers=self.config.get("transformer_num_layers", 4),
                num_heads=self.config.get("transformer_num_heads", 4),
                ffn_dim=self.config.get("transformer_ffn_dim", 512),
                max_relative_distance=self.config.get("transformer_max_relative_distance", 32),
                max_terminus_distance=self.config.get("transformer_max_terminus_distance", 16),
                num_readout_queries=self.config.get("transformer_readout_queries", 1),
                dropout=self.config.get("transformer_dropout", 0.0),
            )
        else:
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
            if self.use_attention_pooling:
                self.ConvAtomComp.append(
                    AttentionPool1d(self.config["AtomComp_out_channels_start"] // 4, initi)
                )
            else:
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
            if self.use_attention_pooling:
                self.ConvDiatomComp.append(
                    AttentionPool1d(self.config["DiatomComp_out_channels_start"] // 2, initi)
                )
            else:
                self.ConvDiatomComp.append(nn.Flatten())

        self.ConvGlobal = nn.ModuleList()
        self.ConvGlobal.append(
            DenseActivation(
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

        self.use_one_hot = bool(self.config.get("use_one_hot", True)) and not self.use_flex_trunk
        if self.use_one_hot:
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

        self.use_charge_embedding = bool(self.config.get("use_charge_embedding", False))
        if self.use_charge_embedding:
            if not self.config.get("add_ccs_features"):
                raise ValueError(
                    "use_charge_embedding requires add_ccs_features=True: "
                    "charge is only present in matrix_global, at a fixed "
                    "index, when that flag is set."
                )
            charge_embedding_dim = self.config.get("charge_embedding_dim", 16)
            seq_concat_shape = calculate_seq_concat_shape(self.config)
            self.charge_embedding = nn.Embedding(7, charge_embedding_dim)
            self.charge_film = nn.Linear(charge_embedding_dim, 2 * seq_concat_shape)
            nn.init.zeros_(self.charge_film.weight)
            nn.init.zeros_(self.charge_film.bias)

        self.use_per_charge_heads = bool(self.config.get("use_per_charge_heads", False))
        if self.use_per_charge_heads and not self.config.get("add_ccs_features"):
            raise ValueError(
                "use_per_charge_heads requires add_ccs_features=True: selecting the right "
                "charge head needs CHARGE_FEATURE_INDEX from matrix_global."
            )

        if self.config.get("mixup_alpha", 0.0) > 0 and (
            self.config.get("sibling_loss_weight", 0.0) > 0
            or self.config.get("ranking_loss_weight", 0.0) > 0
            or self.use_per_charge_heads
        ):
            raise ValueError(
                "mixup_alpha>0 is not compatible with sibling_loss_weight/ranking_loss_weight/"
                "use_per_charge_heads: mixup blends batch rows together, corrupting the "
                "per-row identity those depend on."
            )

        self.use_heteroscedastic_head = bool(self.config.get("use_heteroscedastic_head", False))

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

        # Per-row output width: mu only, or (mu, log_b) when
        # use_heteroscedastic_head, repeated once per charge head when
        # use_per_charge_heads - see forward() for the destandardisation and
        # (for use_per_charge_heads) the per-row head selection.
        per_row_width = 2 if self.use_heteroscedastic_head else 1
        if self.use_per_charge_heads:
            per_row_width *= NUM_CHARGE_HEADS
        self.Concat.append(nn.Linear(self.config["Concat_units"], per_row_width))

    def set_global_feature_scaler(self, mean: Tensor, std: Tensor) -> None:
        """Set the global-feature standardisation buffers from a fitted sample."""
        self.global_feat_mean.copy_(mean)  # ty: ignore[call-non-callable]
        self.global_feat_std.copy_(std)  # ty: ignore[call-non-callable]

    def set_target_scaler(self, mean: float, std: float) -> None:
        """Set the target standardisation buffers from the training target distribution."""
        self.target_mean.fill_(mean)  # ty: ignore[call-non-callable]
        self.target_std.fill_(std)  # ty: ignore[call-non-callable]

    def _base_loss(self, y_hat, y):
        """
        The training objective, in whichever space ``config["loss"]`` selects
        (step 8.3). Always given raw-CCS ``y_hat``/``y`` - only the
        functional form changes, not the units - so this composes cleanly
        with target standardisation, which already returns raw CCS from
        ``forward()`` regardless of whether it's enabled. The reported/
        monitored "Validation MAE" always stays ``self.mae(y_hat, y)``
        (plain L1 in raw Å²), so runs on different loss choices stay
        comparable; only this training-objective value differs.
        """
        loss_name = self.config.get("loss", "l1")
        if loss_name == "l1":
            return self.criterion(y_hat, y)
        if loss_name == "mape":
            return 100.0 * (torch.abs(y_hat - y) / y).mean()
        if loss_name == "huber":
            return nn.functional.huber_loss(y_hat, y, delta=self.config.get("huber_delta", 5.0))
        raise ValueError(f"Unknown loss {loss_name!r}")

    def regularized_loss(self, y_hat, y):
        standard_loss = self._base_loss(y_hat, y)
        l1_norm = sum(torch.norm(p, 1) for p in self.parameters())
        return standard_loss + self.config["L1_alpha"] * l1_norm

    def _pair_mask(self, same: Tensor) -> tuple[Tensor, Tensor]:
        """Row indices of each unique unordered pair where ``same`` holds."""
        return torch.triu(same, diagonal=1).nonzero(as_tuple=True)  # ty: ignore[invalid-return-type]

    def _sibling_delta_loss(self, y_hat: Tensor, y: Tensor, group_id: Tensor) -> Tensor:
        """
        Sibling delta loss: ``L1(pred_a - pred_b, obs_a - obs_b)``
        over batch pairs sharing a stripped sequence (same ``group_id``),
        i.e. pairs differing only in modification state or charge.

        Always plain L1 on the delta regardless of ``config["loss"]`` - a
        deliberate simplification, not an oversight: this term exists to
        supervise the modification/charge *effect* directly (IM2Deep's own
        selling point), which the absolute-CCS loss can hide behind two
        errors that cancel. ``group_id`` only ever contains real matches
        because ``SiblingPairBatchSampler`` deliberately placed them; a
        batch with none (e.g. every filler slot missed by chance) safely
        contributes zero.
        """
        weight = self.config.get("sibling_loss_weight", 0.0)
        if weight <= 0:
            return y_hat.new_zeros(())
        idx_a, idx_b = self._pair_mask(group_id.unsqueeze(0) == group_id.unsqueeze(1))
        if idx_a.numel() == 0:
            return y_hat.new_zeros(())
        pred_delta = y_hat[idx_a] - y_hat[idx_b]
        obs_delta = y[idx_a] - y[idx_b]
        return weight * nn.functional.l1_loss(pred_delta, obs_delta)

    def _ranking_loss(self, y_hat: Tensor, y: Tensor, mz: Tensor) -> Tensor:
        """
        Pairwise ranking loss: a margin ranking loss on the sign
        of predicted vs observed CCS difference, over batch pairs whose
        theoretical m/z are within ``ranking_mz_window`` Th of each other -
        the discrimination that actually matters for precursors a
        quadrupole cannot isolate separately. Pairs with a tied observed CCS
        (``target_sign == 0``) are dropped: there is no ordering to
        supervise, and 0 is not a valid ``margin_ranking_loss`` target.
        """
        weight = self.config.get("ranking_loss_weight", 0.0)
        if weight <= 0:
            return y_hat.new_zeros(())
        window = self.config.get("ranking_mz_window", 3.0)
        idx_a, idx_b = self._pair_mask((mz.unsqueeze(0) - mz.unsqueeze(1)).abs() <= window)
        if idx_a.numel() == 0:
            return y_hat.new_zeros(())
        target_sign = torch.sign(y[idx_a] - y[idx_b])
        keep = target_sign != 0
        if keep.sum() == 0:
            return y_hat.new_zeros(())
        return weight * nn.functional.margin_ranking_loss(
            y_hat[idx_a][keep],
            y_hat[idx_b][keep],
            target_sign[keep],
            margin=self.config.get("ranking_margin", 0.0),
        )

    def _laplace_nll(self, mu: Tensor, y: Tensor, b: Tensor) -> Tensor:
        """
        Laplace negative log-likelihood (step 8.5's heteroscedastic head):
        ``log(2b) + |y-mu|/b``. For a fixed absolute error ``e = |y-mu|``,
        this is minimised exactly at ``b = e`` (the Laplace MLE scale), so a
        row the network cannot fit is automatically assigned a larger ``b``
        rather than forcing ``mu`` to keep chasing it - loss attenuation by
        construction, not a hand-picked per-row reweighting rule. Chosen
        over the Gaussian NLL specifically for consistency with this
        codebase's existing default (`loss: "l1"`, whose point estimate is
        also the median, matching Laplace's) and because CCS label noise
        here is residual disagreement between calibrated mass-spec runs -
        occasional large residuals from a still-disagreeing run, not
        symmetric Gaussian jitter - which Laplace's heavier tail tolerates
        better than Gaussian's quadratic penalty would.
        """
        return (torch.log(2.0 * b) + torch.abs(y - mu) / b).mean()

    def _in_heteroscedastic_warmup(self) -> bool:
        """
        True for the first ``heteroscedastic_warmup_epochs`` epochs (default
        0, i.e. never) when ``use_heteroscedastic_head`` is on.
        """
        warmup_epochs = self.config.get("heteroscedastic_warmup_epochs", 0)
        return self.use_heteroscedastic_head and self.current_epoch < warmup_epochs

    def _point_estimate_loss(self, y_hat: Tensor, y: Tensor, b: Tensor) -> Tensor:
        """
        Plain L1 on ``y_hat`` during the heteroscedastic warm-up (``b`` is
        computed by ``forward()`` regardless, but does not enter the loss,
        so its output row gets no gradient from this term - it starts
        learning from an already-well-trained shared trunk once warm-up
        ends, rather than trying to calibrate from a cold, random one at
        the same time ``mu`` is still finding its footing). The Laplace NLL
        otherwise. Used identically by training/validation/test so
        "Validation loss" stays meaningful for whichever phase training is
        actually in - unlike "Validation MAE", which always reports plain
        L1 on ``mu`` regardless (the established invariant across every
        ``loss``/head choice in this codebase).
        """
        if self._in_heteroscedastic_warmup():
            return nn.functional.l1_loss(y_hat, y)
        return self._laplace_nll(y_hat, y, b)

    def forward(
        self,
        atom_comp,
        diatom_comp,
        global_feats,
        one_hot,
        mol_desc: Tensor | None = None,
        return_attention: bool = False,
    ):
        if return_attention and not self.use_transformer_trunk:
            raise ValueError(
                "return_attention=True requires use_transformer_trunk=True: "
                "no other branch exposes attention weights."
            )

        # TransformerTrunk needs the RAW, un-permuted matrix tensor (it does
        # its own layout internally, unlike every Conv1d branch here) -
        # captured before the permute below.
        atom_comp_raw = atom_comp

        atom_comp = atom_comp.permute(0, 2, 1)
        diatom_comp = diatom_comp.permute(0, 2, 1)
        one_hot = one_hot.permute(0, 2, 1)

        if self.use_charge_embedding or self.use_transformer_trunk or self.use_per_charge_heads:
            charge_idx = global_feats[:, CHARGE_FEATURE_INDEX].round().long().clamp(0, 6)

        global_feats = (global_feats - self.global_feat_mean) / self.global_feat_std

        if self.use_flex_trunk:
            # atom_comp/one_hot are already permuted to (batch, channels,
            # length) above - FlexTrunk expects exactly that layout, same
            # as every other branch here. diatom_comp is unused: matrix_sum
            # is redundant once the trunk sees matrix directly.
            seq_feats = self.flex_trunk(atom_comp, one_hot)
        elif self.use_transformer_trunk:
            # diatom_comp is unused here too, for the same reason as
            # flex_trunk - the transformer sees matrix directly.
            if return_attention:
                seq_feats, attention = self.transformer_trunk(
                    atom_comp_raw, charge_idx, return_attention=True
                )
            else:
                seq_feats = self.transformer_trunk(atom_comp_raw, charge_idx)
        else:
            for layer in self.ConvAtomComp:
                atom_comp = layer(atom_comp)
            for layer in self.ConvDiatomComp:
                diatom_comp = layer(diatom_comp)
            seq_feats = torch.cat((atom_comp, diatom_comp), 1)

        for layer in self.ConvGlobal:
            global_feats = layer(global_feats)
        if self.use_one_hot:
            for layer in self.OneHot:
                one_hot = layer(one_hot)

        if self.config["add_X_mol"]:
            for layer in self.MolDesc:
                mol_desc = layer(mol_desc)

        if self.use_charge_embedding:
            charge_embed = self.charge_embedding(charge_idx)
            gamma, beta = self.charge_film(charge_embed).chunk(2, dim=-1)
            seq_feats = seq_feats * (1.0 + gamma) + beta

        if self.use_one_hot:
            concatenated = torch.cat((seq_feats, one_hot, global_feats), 1)
        else:
            concatenated = torch.cat((seq_feats, global_feats), 1)

        if self.config["add_X_mol"]:
            if mol_desc is None:
                raise ValueError("`mol_desc` is required when `add_X_mol` is enabled.")
            concatenated = torch.cat((concatenated, mol_desc), 1)

        for layer in self.Concat:
            concatenated = layer(concatenated)

        if self.use_per_charge_heads:
            sub_width = concatenated.shape[1] // NUM_CHARGE_HEADS
            concatenated = concatenated.view(-1, NUM_CHARGE_HEADS, sub_width)
            head_idx = (charge_idx - 1).clamp(0, NUM_CHARGE_HEADS - 1)
            concatenated = concatenated[
                torch.arange(concatenated.size(0), device=concatenated.device), head_idx
            ]

        if self.use_heteroscedastic_head:
            mu_std, log_b_std = concatenated[:, 0], concatenated[:, 1]
            mu = mu_std * self.target_std + self.target_mean  # ty: ignore[unsupported-operator]
            b = torch.exp(log_b_std.clamp(-LOG_B_CLAMP, LOG_B_CLAMP)) * self.target_std  # ty: ignore[unsupported-operator]
            b = b.clamp(min=1e-3)
            if return_attention:
                return mu, b, attention
            return mu, b

        # De-standardise back to raw CCS.
        output = concatenated * self.target_std + self.target_mean  # ty: ignore[unsupported-operator]
        if return_attention:
            return output, attention
        return output

    def _apply_mixup(self, atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc=None):
        """
        Composition mixup: mix each training batch with a random
        permutation of itself via ``lambda ~ Beta(mixup_alpha, mixup_alpha)``,
        applied identically to every input tensor and the target.

        The atomic-composition encoding is additive, so a convex combination
        of two peptides' `matrix`/`matrix_global` is a chemically
        interpretable point rather than nonsense, and CCS is smooth in
        composition, so interpolating the targets the same way is
        approximately valid. Training only - never called for validation/
        test - and "Train MAE" is measured against the mixed target while
        this is on, so it stops being directly comparable to a non-mixup
        run's; "Validation MAE" is unaffected and remains the number to
        compare across configs.

        NOT YET COMPATIBLE with ``use_charge_embedding``: ``global_feats``'
        charge value (see ``CHARGE_FEATURE_INDEX``) would get linearly
        blended into a non-integer, physically meaningless "charge", which
        ``forward()``'s ``charge_idx = ....round().long()`` would then round
        arbitrarily - don't combine them until that's addressed.

        A no-op when ``config["mixup_alpha"]`` is 0 (the default).
        """
        mixup_alpha = self.config.get("mixup_alpha", 0.0)
        if mixup_alpha <= 0:
            return atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc

        lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
        perm = torch.randperm(atom_comp.size(0), device=atom_comp.device)
        atom_comp = lam * atom_comp + (1 - lam) * atom_comp[perm]
        diatom_comp = lam * diatom_comp + (1 - lam) * diatom_comp[perm]
        global_feats = lam * global_feats + (1 - lam) * global_feats[perm]
        one_hot = lam * one_hot + (1 - lam) * one_hot[perm]
        y = lam * y + (1 - lam) * y[perm]
        if mol_desc is not None:
            mol_desc = lam * mol_desc + (1 - lam) * mol_desc[perm]
        return atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc

    def _forward_point_estimate(
        self, atom_comp, diatom_comp, global_feats, one_hot, mol_desc=None
    ) -> tuple[Tensor, Tensor | None]:
        """
        Runs ``forward()`` and normalises its return to ``(y_hat, b)``
        regardless of ``use_heteroscedastic_head`` - ``b`` is ``None`` when
        disabled. Centralises the ``add_X_mol`` branching and the
        heteroscedastic tuple-vs-plain-tensor difference so
        training_step/validation_step/test_step/predict_step don't each
        repeat it; ``y_hat`` plays the same "point estimate" role in both
        cases for MAE logging and the sibling/ranking losses.
        """
        if self.config["add_X_mol"]:
            output = self(atom_comp, diatom_comp, global_feats, one_hot, mol_desc)
        else:
            output = self(atom_comp, diatom_comp, global_feats, one_hot)
        if self.use_heteroscedastic_head:
            mu, b = output
            return mu, b
        return output.squeeze(1), None

    def training_step(self, batch, batch_idx):
        needs_pair_ids = (
            self.config.get("sibling_loss_weight", 0.0) > 0
            or self.config.get("ranking_loss_weight", 0.0) > 0
        )
        if needs_pair_ids:
            *batch, group_id, mz = batch

        if self.config["add_X_mol"]:
            atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = batch
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            mol_desc = None

        atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc = self._apply_mixup(
            atom_comp, diatom_comp, global_feats, one_hot, y, mol_desc
        )

        y_hat, b = self._forward_point_estimate(
            atom_comp, diatom_comp, global_feats, one_hot, mol_desc
        )

        if self.use_heteroscedastic_head:
            l1_norm = sum(torch.norm(p, 1) for p in self.parameters())
            loss = self._point_estimate_loss(y_hat, y, b) + self.config["L1_alpha"] * l1_norm  # ty: ignore[invalid-argument-type]
        else:
            loss = self.regularized_loss(y_hat, y)
        if self.config.get("sibling_loss_weight", 0.0) > 0:
            loss = loss + self._sibling_delta_loss(y_hat, y, group_id)
        if self.config.get("ranking_loss_weight", 0.0) > 0:
            loss = loss + self._ranking_loss(y_hat, y, mz)

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
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            mol_desc = None
        y_hat, b = self._forward_point_estimate(
            atom_comp, diatom_comp, global_feats, one_hot, mol_desc
        )
        loss = (
            self._point_estimate_loss(y_hat, y, b)  # ty: ignore[invalid-argument-type]
            if self.use_heteroscedastic_head
            else self._base_loss(y_hat, y)
        )

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
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            mol_desc = None
        y_hat, b = self._forward_point_estimate(
            atom_comp, diatom_comp, global_feats, one_hot, mol_desc
        )
        loss = (
            self._point_estimate_loss(y_hat, y, b)  # ty: ignore[invalid-argument-type]
            if self.use_heteroscedastic_head
            else self._base_loss(y_hat, y)
        )

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
        else:
            atom_comp, diatom_comp, global_feats, one_hot, y = batch
            mol_desc = None
        y_hat, _ = self._forward_point_estimate(
            atom_comp, diatom_comp, global_feats, one_hot, mol_desc
        )
        return y_hat

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config.get("weight_decay", 0.0),
        )
        scheduler_name = self.config.get("lr_scheduler")
        if not scheduler_name:
            return optimizer
        if scheduler_name == "one_cycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.config["learning_rate"],
                total_steps=self.trainer.estimated_stepping_batches,  # ty: ignore[invalid-argument-type]
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        raise ValueError(f"Unknown lr_scheduler {scheduler_name!r}")

    def configure_init(self):
        if (not self.config["init"]) or (self.config["init"] == "normal"):
            return partial(nn.init.normal_, mean=0.0, std=0.05)
        if self.config["init"] == "xavier":
            return partial(nn.init.xavier_normal_, gain=1.0)
        if self.config["init"] == "kaiming":
            return partial(
                nn.init.kaiming_normal_,
                a=self.config["LRelu_negative_slope"],
                nonlinearity="leaky_relu",
            )


# TODO: update to new TransformerTrun architectures
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
            config["backbone_SD_path"], config=config, criterion=criterion, strict=False
        )

        if self.backbone.config.get("use_heteroscedastic_head", False):
            raise ValueError(
                "Cannot finetune from a use_heteroscedastic_head=True backbone: its Concat "
                "head ends in a 2-wide (mu, log_b) layer, which IM2DeepTransfer.forward() has "
                "no path for. Finetuning support for this wasn't added here."
            )

        if self.backbone.config.get("use_per_charge_heads", False):
            raise ValueError(
                "Cannot finetune from a use_per_charge_heads=True backbone: its Concat head "
                "ends in a per-charge-head layer, which IM2DeepTransfer.forward() has no "
                "selection path for. Finetuning support for this wasn't added here."
            )

        # A use_flex_trunk/use_transformer_trunk backbone has no
        # ConvAtomComp/ConvDiatomComp, and its forward() also carries the
        # charge register token and both standardisation scalers. Rather than
        # re-implementing that here, forward() delegates to the backbone; the
        # branches are only exposed as attributes so BackboneFreeze finds them.
        self.delegate_forward = bool(
            self.backbone.use_transformer_trunk or self.backbone.use_flex_trunk
        )
        if self.delegate_forward:
            for name in ("transformer_trunk", "flex_trunk", "ConvGlobal", "OneHot", "MolDesc"):
                if hasattr(self.backbone, name):
                    setattr(self, name, getattr(self.backbone, name))
            self.concat = self.backbone.Concat
            return

        # KNOWN LIMITATION (CNN backbones only): forward() below skips the
        # backbone's global-feature and target standardisation, so a CNN
        # backbone trained with standardize_* gives wrongly scaled output.
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
        if self.delegate_forward:
            return self.backbone(atom_comp, diatom_comp, global_feats, one_hot, mol_desc)

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
