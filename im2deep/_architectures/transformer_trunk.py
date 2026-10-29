"""
Transformer encoder trunk over atomic-composition tokens (step 8.4 item 3).

The main architecture arm of step 8.4: token ``i`` is residue ``i``'s 6-vector
atomic composition, taken straight from ``matrix`` and projected to
``d_model`` by a single linear layer, so the atomic encoding is preserved
exactly and no featuriser changes are needed. Three additions distinguish
this from a generic transformer encoder, each addressing a specific,
previously-diagnosed gap:

* **Charge as a prepended register token** (:class:`TransformerTrunk`), not a
  late-concatenated scalar (the four-branch design's own approach) or a
  FiLM scale-and-shift (``use_charge_embedding``, step 8.4 item 1). Every
  residue attends to charge at every layer, and charge attends back -
  giving charge a data flow the FiLM version cannot: a kernel can only
  rescale a feature it has already computed, while self-attention lets the
  network learn a charge-dependent contribution from the first layer.
* **Pair-biased attention** (:class:`RelativePairBias`): a learned bias on
  the attention logits as a function of relative sequence separation and
  proximity to either terminus, giving the model an explicit, learnable
  notion of sequence distance - the AlphaFold/T5 pattern, and a much
  stronger inductive bias for a task fundamentally about intramolecular
  contact than a sinusoidal positional encoding would be. Deliberately
  **not** T5's logarithmic bucketing: peptides here are at most
  ``padding_length`` (60) residues, so a plain clamped linear table over
  the true distance range is both simpler and strictly more expressive at
  this scale (log-bucketing exists to keep a table small for very long
  sequences, which is not the situation here).
* **Learned-query read-out** (:class:`PerceiverReadout`): ``k`` learned
  query vectors cross-attend to the residue tokens, replacing pooling.
  ``k=1`` (the only mode implemented here) is attention pooling with a
  masked softmax; ``k=2``/``3`` would be a route into the multi-conformer
  problem (step 9) but is out of scope for this arm as built.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RelativePairBias(nn.Module):
    """
    Per-head additive attention bias from relative sequence separation and
    terminus proximity, over residue positions only (the register token
    gets no positional bias - it has no sequence position of its own).

    Two small, fully-learnable tables rather than DeepLC-style hand
    engineering: ``relative`` indexed by the *signed*, clamped distance
    ``i - j`` (asymmetric by design, so the model can learn direction-
    dependent effects, e.g. "N-terminal side" vs "C-terminal side" of a
    contact); ``terminus`` indexed by how close either position of the pair
    is to either terminus of the *real* (unpadded) peptide, which is
    symmetric by construction (a ``min`` over both positions and both
    termini).
    """

    def __init__(self, num_heads: int, max_relative_distance: int, max_terminus_distance: int):
        super().__init__()
        self.max_relative_distance = max_relative_distance
        self.max_terminus_distance = max_terminus_distance
        self.relative = nn.Embedding(2 * max_relative_distance + 1, num_heads)
        self.terminus = nn.Embedding(max_terminus_distance + 1, num_heads)
        # (cache_key, tensor) for the batch-invariant half of forward() - see
        # _relative_bias. Not a buffer: its shape varies with seq_len, and it
        # must never be part of the state_dict.
        self._relative_bias_cache: tuple[tuple, Tensor] | None = None

    def _relative_bias(self, seq_len: int, device: torch.device) -> Tensor:
        """
        ``(seq_len, seq_len, num_heads)`` relative-position bias.

        Depends only on ``seq_len`` and the ``relative`` embedding's
        weights, not on the batch's actual data - safe to compute once and
        reuse across every forward call in an ``eval()``/no-grad prediction
        run. Caching is skipped whenever a gradient could flow through this
        call (``self.training`` or autograd enabled), since then a fresh
        computation graph is required on every call; it is also skipped,
        i.e. recomputed, if ``seq_len``/``device`` change or the embedding
        weights have since been updated (``_version`` guards the latter).
        """
        cacheable = not self.training and not torch.is_grad_enabled()
        weight_version = self.relative.weight._version
        cache_key = (seq_len, device, weight_version)
        if (
            cacheable
            and self._relative_bias_cache is not None
            and self._relative_bias_cache[0] == cache_key
        ):
            return self._relative_bias_cache[1]

        positions = torch.arange(seq_len, device=device)
        relative_distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # (L, L), [i,j] = i-j
        relative_idx = (
            relative_distance.clamp(-self.max_relative_distance, self.max_relative_distance)
            + self.max_relative_distance
        )
        relative_bias = self.relative(relative_idx)  # (L, L, num_heads)

        if cacheable:
            self._relative_bias_cache = (cache_key, relative_bias)
        return relative_bias

    def forward(self, real_lengths: Tensor, seq_len: int) -> Tensor:
        """
        ``real_lengths``: ``(batch,)`` true (unpadded) peptide lengths.
        ``seq_len``: the array length (``padding_length``), not the real
        length - positions past a peptide's real length still get a
        relative/terminus bias here (they are masked out separately via
        ``key_padding_mask``; this module only computes the bias values).

        Returns ``(batch, num_heads, seq_len, seq_len)``.
        """
        device = real_lengths.device
        relative_bias = self._relative_bias(seq_len, device)

        dist_to_n_term = torch.arange(seq_len, device=device)  # (L,)
        # (batch, L): distance to the C-terminus depends on each peptide's
        # own real length, not the shared padded array length.
        dist_to_c_term = real_lengths.unsqueeze(1) - 1 - dist_to_n_term.unsqueeze(0)
        closest_term = torch.minimum(dist_to_n_term.unsqueeze(0), dist_to_c_term)  # (batch, L)
        # Pairwise minimum over both positions: symmetric by construction.
        terminus_distance = torch.minimum(
            closest_term.unsqueeze(2), closest_term.unsqueeze(1)
        )  # (batch, L, L)
        terminus_idx = terminus_distance.clamp(0, self.max_terminus_distance)
        terminus_bias = self.terminus(terminus_idx)  # (batch, L, L, num_heads)

        combined = relative_bias.unsqueeze(0) + terminus_bias  # (batch, L, L, num_heads)
        return combined.permute(0, 3, 1, 2)  # (batch, num_heads, L, L)


def _fold_padding_into_bias(bias: Tensor, key_padding_mask: Tensor) -> Tensor:
    """
    Fold a boolean key-padding mask into an additive attention bias, so a
    single float ``attn_mask`` can be passed to ``nn.MultiheadAttention``
    instead of separately supplying a bool ``key_padding_mask`` -
    PyTorch deprecates mixing the two types on one call, and this changes
    nothing behaviourally: an ignored key still gets ``-inf`` either way.

    ``bias``: ``(batch, heads, Lq, Lk)`` float. ``key_padding_mask``:
    ``(batch, Lk)`` bool, ``True`` = ignore that key. Returns
    ``(batch*heads, Lq, Lk)`` float, ready for ``attn_mask``.
    """
    batch, heads, seq_q, seq_k = bias.shape
    penalty = torch.zeros(batch, seq_k, dtype=bias.dtype, device=bias.device)
    penalty = penalty.masked_fill(key_padding_mask, float("-inf"))
    combined = bias + penalty.view(batch, 1, 1, seq_k)
    return combined.reshape(batch * heads, seq_q, seq_k)


def _sdpa_attention(
    mha: nn.MultiheadAttention, query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor
) -> Tensor:
    """
    Equivalent to ``mha(query, key, value, attn_mask=attn_mask, need_weights=False)[0]``
    for how ``mha`` is always constructed in this module - ``batch_first=True``,
    default ``kdim``/``vdim`` (so a single fused ``in_proj_weight``), no
    ``bias_k``/``bias_v``, ``add_zero_attn=False`` - calling
    ``scaled_dot_product_attention`` directly instead of going through
    ``nn.MultiheadAttention.forward``'s generic path.

    That generic path reshapes q/k/v between its ``batch_first`` layout and
    its internal layout via a transpose-and-``.contiguous()`` copy around
    every call, *in addition to* whatever ``scaled_dot_product_attention``
    itself costs - `torch.profiler` on this trunk measured that reshaping at
    ~21% of total self-CPU time, on top of the fused-kernel cost the
    ``need_weights=False`` fix already secured. This reads the same
    ``in_proj_weight``/``in_proj_bias``/``out_proj`` parameters ``mha``
    already owns, so an existing checkpoint's state_dict loads unchanged -
    no retraining, no shape change, nothing to migrate.

    Only handles the two shapes of call this file makes: self-attention
    (``query is key is value``) and cross-attention with a shared key/value
    (``key is value``, ``query`` a different tensor) - the same two branches
    ``torch.nn.functional._in_projection_packed`` special-cases, reproduced
    here rather than delegated to, so this can skip straight to
    ``scaled_dot_product_attention`` afterwards.

    Not bit-identical to ``mha(...)`` - a different kernel call order, same
    numerical class as the ``need_weights=False`` change this sits on top of.
    Verified to the same tight float32 tolerance that change was verified
    against (see the CPU-timing investigation this accompanies).
    """
    embed_dim = query.size(-1)
    num_heads = mha.num_heads
    head_dim = embed_dim // num_heads
    batch = query.size(0)

    if query is key and key is value:
        q, k, v = F.linear(query, mha.in_proj_weight, mha.in_proj_bias).chunk(3, dim=-1)
    else:
        assert key is value, "_sdpa_attention only supports self-attention or a shared key/value"
        w_q, w_kv = mha.in_proj_weight.split([embed_dim, embed_dim * 2])
        if mha.in_proj_bias is None:
            b_q, b_kv = None, None
        else:
            b_q, b_kv = mha.in_proj_bias.split([embed_dim, embed_dim * 2])
        q = F.linear(query, w_q, b_q)
        k, v = F.linear(key, w_kv, b_kv).chunk(2, dim=-1)

    def split_heads(t: Tensor) -> Tensor:
        return t.view(batch, -1, num_heads, head_dim).transpose(1, 2)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)
    # attn_mask arrives as _fold_padding_into_bias's (batch*heads, Lq, Lk) -
    # a view back to 4D, exactly undoing that function's own final reshape
    # (same contiguous memory, no copy), which is the shape
    # scaled_dot_product_attention wants directly.
    mask = attn_mask.view(batch, num_heads, attn_mask.size(-2), attn_mask.size(-1))
    dropout_p = mha.dropout if mha.training else 0.0
    attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
    attended = attended.transpose(1, 2).reshape(batch, -1, embed_dim)
    return F.linear(attended, mha.out_proj.weight, mha.out_proj.bias)


class PreNormTransformerBlock(nn.Module):
    """Pre-norm self-attention + GELU FFN block, both with a residual connection."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(
        self, x: Tensor, attn_mask: Tensor, return_attention: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """``attn_mask``: ``(batch*num_heads, L, L)`` float, already carrying
        both the pair/terminus bias and padding (as ``-inf``) - see
        :func:`_fold_padding_into_bias`. Passed only as ``attn_mask``, not
        also as ``key_padding_mask``: PyTorch deprecates mixing a float
        ``attn_mask`` with a bool ``key_padding_mask`` on the same call.

        ``return_attention`` (default ``False``, never set by the training/
        predict paths - for attention inspection only): when
        ``True``, also returns the per-head self-attention weights,
        ``(batch, num_heads, L, L)``, via ``nn.MultiheadAttention`` itself
        (``need_weights=True``) rather than :func:`_sdpa_attention` - the
        weights aren't otherwise available, and requesting them forces
        ``nn.MultiheadAttention`` onto its slower, unfused eager path
        regardless (see the CPU-timing investigation this accompanies for
        why the default path avoids that path entirely instead). Only pay
        that cost when attention is actually wanted."""
        normed = self.attn_norm(x)
        if return_attention:
            attended, attn_weights = self.self_attn(
                normed,
                normed,
                normed,
                attn_mask=attn_mask,
                need_weights=True,
                average_attn_weights=False,
            )
        else:
            attended = _sdpa_attention(self.self_attn, normed, normed, normed, attn_mask)
        x = x + attended
        x = x + self.ffn(self.ffn_norm(x))
        if return_attention:
            return x, attn_weights
        return x


class PerceiverReadout(nn.Module):
    """
    ``num_queries`` learned query vectors cross-attending to residue tokens.

    ``num_queries=1`` is plain (masked) attention pooling: a single learned
    query reads a weighted combination of the valid residue positions.
    Higher ``k`` is exposed for future multi-conformer work (step 9) but is
    not otherwise treated specially here - the caller decides what to do
    with a ``(batch, k, d_model)`` output when ``k>1``.
    """

    def __init__(self, d_model: int, num_heads: int, num_queries: int, dropout: float):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, d_model) * (1.0 / math.sqrt(d_model)))
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self, residue_hidden: Tensor, attn_mask: Tensor, return_attention: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """``attn_mask``: ``(batch*num_heads, num_queries, residue_length)``
        float, carrying only padding (as ``-inf``) - the read-out has no
        pair/terminus bias of its own, see :func:`_fold_padding_into_bias`.

        ``return_attention``: see ``PreNormTransformerBlock.forward`` for why
        this is opt-in and costs the fused kernel when requested. Returns the
        cross-attention weights, ``(batch, num_heads, num_queries,
        residue_length)`` - directly, per 8.10, "how much the model relied
        on each residue to predict CCS"."""
        batch_size = residue_hidden.size(0)
        query = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        if return_attention:
            out, attn_weights = self.cross_attn(
                query,
                residue_hidden,
                residue_hidden,
                attn_mask=attn_mask,
                need_weights=True,
                average_attn_weights=False,
            )
        else:
            # See PreNormTransformerBlock.forward and _sdpa_attention for why
            # this bypasses nn.MultiheadAttention.forward entirely.
            out = _sdpa_attention(
                self.cross_attn, query, residue_hidden, residue_hidden, attn_mask
            )
        pooled = out.squeeze(1) if self.num_queries == 1 else out
        if return_attention:
            return pooled, attn_weights
        return pooled


class TransformerTrunk(nn.Module):
    """
    Fused transformer encoder replacing ``ConvAtomComp``/``ConvDiatomComp``.

    Unlike :class:`~im2deep._architectures.blocks.FlexTrunk` (step 8.4 items
    8+9), this does **not** also retire ``OneHot`` - the roadmap's own text
    for this item describes only the atomic-composition token stream, not a
    residue-identity fusion, and folding one-hot identity in here would be
    testing an extra, untested idea under this item's name rather than the
    item as specified. ``use_one_hot`` stays a live, independent switch.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        max_relative_distance: int,
        max_terminus_distance: int,
        num_readout_queries: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(6, d_model)
        self.charge_embed = nn.Embedding(7, d_model)
        self.pair_bias = RelativePairBias(num_heads, max_relative_distance, max_terminus_distance)
        self.blocks = nn.ModuleList(
            [
                PreNormTransformerBlock(d_model, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.readout = PerceiverReadout(d_model, num_heads, num_readout_queries, dropout)
        self.num_heads = num_heads

    def forward(
        self, atom_comp_raw: Tensor, charge_idx: Tensor, return_attention: bool = False
    ) -> Tensor | tuple[Tensor, dict]:
        """
        ``atom_comp_raw``: ``(batch, padding_length, 6)`` - the RAW, un-
        permuted ``matrix`` tensor (unlike every conv branch in this
        architecture, this module does its own layout, so the caller must
        capture this before ``forward()``'s usual ``.permute(0, 2, 1)``).
        ``charge_idx``: ``(batch,)`` long, 0-6 - the same raw charge index
        ``use_charge_embedding`` already derives from ``CHARGE_FEATURE_INDEX``.

        ``return_attention`` (default ``False``, for attention inspection only):
        when ``True``, also returns a dict with
        ``"self_attention"`` (a list of per-layer ``(batch, heads, L+1, L+1)``
        tensors, register token included at index 0) and
        ``"readout_attention"`` (``(batch, heads, num_queries, L)``, register
        token excluded - see ``test_register_token_excluded_from_readout_keys``).
        """
        batch_size, seq_len, _ = atom_comp_raw.shape
        padding_mask = atom_comp_raw.sum(dim=-1) == 0  # (batch, seq_len); True = padding

        tokens = self.input_proj(atom_comp_raw)  # (batch, seq_len, d_model)
        charge_token = self.charge_embed(charge_idx).unsqueeze(1)  # (batch, 1, d_model)
        hidden = torch.cat([charge_token, tokens], dim=1)  # (batch, seq_len+1, d_model)

        register_valid = torch.zeros(batch_size, 1, dtype=torch.bool, device=padding_mask.device)
        key_padding_mask = torch.cat([register_valid, padding_mask], dim=1)  # (batch, seq_len+1)

        real_lengths = (~padding_mask).sum(dim=1)  # (batch,)
        residue_bias = self.pair_bias(real_lengths, seq_len)  # (batch, heads, seq_len, seq_len)
        # Register token gets no pair/terminus bias of its own (row/column 0
        # stay exactly zero) - F.pad's constant-0 fill is the zero-alloc +
        # slice-assign this replaces, in one fused op instead of two.
        bias_full = F.pad(residue_bias, (1, 0, 1, 0))
        attn_mask = _fold_padding_into_bias(bias_full, key_padding_mask)

        self_attn_weights = [] if return_attention else None
        for block in self.blocks:
            if return_attention:
                hidden, attn_weights = block(hidden, attn_mask=attn_mask, return_attention=True)
                self_attn_weights.append(attn_weights)
            else:
                hidden = block(hidden, attn_mask=attn_mask)
        hidden = self.final_norm(hidden)

        residue_hidden = hidden[:, 1:, :]  # register token excluded from read-out
        residue_padding_mask = key_padding_mask[:, 1:]
        # The read-out has no pair/terminus bias of its own: a size-1 zero
        # tensor broadcasts through _fold_padding_into_bias's addition to
        # the same result as a fully materialized zero bias, without
        # allocating (batch, heads, num_queries, seq_len) of zeros for it.
        readout_bias = residue_bias.new_zeros(batch_size, self.num_heads, 1, 1).expand(
            batch_size, self.num_heads, self.readout.num_queries, seq_len
        )
        readout_attn_mask = _fold_padding_into_bias(readout_bias, residue_padding_mask)
        if return_attention:
            pooled, readout_attn = self.readout(
                residue_hidden, readout_attn_mask, return_attention=True
            )
            return pooled, {
                "self_attention": self_attn_weights,
                "readout_attention": readout_attn,
            }
        return self.readout(residue_hidden, readout_attn_mask)
