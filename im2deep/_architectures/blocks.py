"""Building blocks for the IM2Deep architecture."""

import torch
from torch import nn

from im2deep._architectures.activations import LRelu_with_saturation


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
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.initializer = initializer
        self.activation = LRelu_with_saturation(
            negative_slope=negative_slope, saturation=saturation
        )

        initializer(self.conv.weight)

    def forward(self, x):
        return self.activation(self.conv(x))


class DenseActivation(nn.Module):
    def __init__(self, in_features, out_features, initializer, negative_slope, saturation):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.initializer = initializer
        self.activation = LRelu_with_saturation(
            negative_slope=negative_slope, saturation=saturation
        )

        initializer(self.linear.weight)

    def forward(self, x):
        return self.activation(self.linear(x))


class AttentionPool1d(nn.Module):
    """
    Learned attention pooling over the position axis: ``(B, C, L) -> (B, C)``.

    Step 8.4 item 2's replacement for the position-index-dependent
    ``nn.Flatten()`` used elsewhere in the architecture. A 1x1 conv produces
    one raw score per position, softmax-normalised over positions (``L``),
    then a weighted sum over ``L`` - a small, standard attention-pooling
    design. Unlike ``Flatten()``, the output width depends only on the
    channel count, not on the (``padding_length``-derived) sequence length,
    so a branch ending in this is length-agnostic rather than hard-wired to
    whatever ``padding_length`` produced.
    """

    def __init__(self, channels, initializer):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)
        initializer(self.score.weight)

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=-1)  # (B, 1, L)
        return (x * weights).sum(dim=-1)  # (B, C)


class _ConvSiLU(nn.Module):
    """A single ``Conv1d`` with ``same`` padding followed by SiLU, matching
    DeepLC's ``_ConvSiLU`` (``deeplc/_architecture.py``) exactly - no
    LeakyReLU/saturation, no learnable-initializer hook, since this block is
    only ever used inside :class:`FlexTrunk`, ported to be faithful to the
    reference implementation rather than to match IM2Deep's other blocks."""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding="same")
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.conv(x))


class FlexTrunk(nn.Module):
    """
    Fused convolutional trunk over atomic composition and a learned residue
    embedding, with length-masked sum+max pooling (step 8.4 items 8+9).

    Ported from DeepLC's ``_FlexCNNEncoder`` (``deeplc/_architecture.py``,
    lines 988-1079), whose own docstring claims roughly a third of the
    four-branch backbone's error on DeepLC's retention-time task, on the
    same head. Replaces ``ConvAtomComp``, ``ConvDiatomComp`` and ``OneHot``
    together when enabled: residue identity reaches the network through an
    embedding fused with atom counts *before* any neighbour-mixing
    convolution, rather than through a separate, late-concatenated 92-
    parameter one-hot branch, and ``matrix_sum`` (``ConvDiatomComp``'s input)
    is redundant once the trunk sees ``matrix`` directly - DeepLC's own
    docstring calls it exactly that.

    Deliberately does **not** port DeepLC's ``InputNorm``/``residue_counts``/
    final dense ``self.net`` - those fuse the global feature branch and a
    residue-count vector into this same module, which is step 8.4 item 10,
    not this one. IM2Deep keeps ``ConvGlobal`` as its own already-tuned
    branch feeding the shared ``Concat`` head, so this module's output is
    just the pooled sequence feature vector - the same role
    ``torch.cat((atom_comp, diatom_comp), 1)`` played before, sized by
    :func:`im2deep._architectures.helpers.calculate_seq_concat_shape`.

    Channel widths (``channels``, ``stem_channels``) are **not** copied from
    DeepLC's own RT-tuned defaults (512/512 channels, 128-wide stem) - those
    were tuned for a different task and a differently-sized encoder; IM2Deep
    sizes them down to stay proportionate to its own current best config's
    scale (~1.5-2M total parameters). Only ``embed_dim`` (16) and
    ``stem_layers``/``kernel_size`` are carried over unchanged, since those
    are small, task-agnostic choices rather than capacity knobs.
    """

    #: Index reserved for padding positions, matching the one-hot vocabulary
    #: size (20 canonical amino acids, indices 0-19). A residue DeepLC's own
    #: featurisation could not encode (e.g. selenocysteine - see the
    #: "One-hot skip" warnings) is an all-zero one-hot row, indistinguishable
    #: from real padding by this same detection - an approximation inherited
    #: unchanged from DeepLC's own ``residue_indices``, not introduced here.
    PAD_INDEX = 20

    def __init__(
        self,
        embed_dim: int,
        channels: "tuple[int, ...] | list[int]",
        kernel_size: int,
        stem_channels: int,
        stem_layers: int,
    ):
        super().__init__()
        self.embed = nn.Embedding(self.PAD_INDEX + 1, embed_dim, padding_idx=self.PAD_INDEX)
        self.stem = self._pointwise_stem(stem_channels, stem_layers) if stem_channels > 0 else None

        in_channels = (stem_channels if stem_channels > 0 else 6) + embed_dim
        blocks = []
        for out_channels in channels:
            blocks.append(_ConvSiLU(in_channels, out_channels, kernel_size))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = in_channels

        # Sum (length-extensive) and max (length-intensive) pooling, masked
        # to the valid length - item 9. Concatenated so the trunk carries
        # both a peptide-size-sensitive and a peptide-size-insensitive view.
        self.pool_norm = nn.LayerNorm(in_channels * 2)

    @staticmethod
    def _pointwise_stem(channels: int, layers: int) -> nn.Sequential:
        """
        Width-1 convolutions decoding each position's atom counts alone,
        before any neighbour mixing - atomic composition determines the
        residue uniquely except for the Leu/Ile pair, which is a lookup
        rather than a linear map, so one layer is generally not enough.
        """
        return nn.Sequential(
            *[_ConvSiLU(6 if i == 0 else channels, channels, kernel_size=1) for i in range(layers)]
        )

    def residue_indices(self, one_hot: torch.Tensor) -> torch.Tensor:
        """
        Integer residue index per position from an already-permuted
        ``(batch, 20, length)`` one-hot tensor. Padding positions are
        all-zero columns, for which ``argmax`` would return 0 and collide
        with the first residue, so they are set to :attr:`PAD_INDEX`
        explicitly - the same fix DeepLC's own version applies.
        """
        idx = one_hot.argmax(dim=1)
        return idx.masked_fill(one_hot.sum(dim=1) == 0, self.PAD_INDEX)

    def forward(self, atom_comp: torch.Tensor, one_hot: torch.Tensor) -> torch.Tensor:
        """
        ``atom_comp``: ``(batch, 6, length)``; ``one_hot``: ``(batch, 20, length)``
        - both already permuted to Conv1d's channels-first layout by the
        caller, matching every other branch in this architecture.
        """
        idx = self.residue_indices(one_hot)
        valid = (idx != self.PAD_INDEX).unsqueeze(1)  # (batch, 1, length)

        hidden = atom_comp
        if self.stem is not None:
            hidden = self.stem(hidden)
        hidden = torch.cat([hidden, self.embed(idx).transpose(1, 2)], dim=1)

        for block in self.blocks:
            hidden = block(hidden)
        hidden = hidden * valid

        summed = hidden.sum(dim=2)
        maxed = torch.nan_to_num(
            hidden.masked_fill(~valid, float("-inf")).max(dim=2).values, neginf=0.0
        )
        return self.pool_norm(torch.cat([summed, maxed], dim=1))
