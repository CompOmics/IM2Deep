"""Tests for the IM2Deep architecture itself (im2deep/_architectures/im2deep_single.py)."""

import copy

import pytest
import torch
from torch import nn

from im2deep._architectures.im2deep_single import CHARGE_FEATURE_INDEX, IM2Deep
from im2deep.constants import DEFAULT_TRAINING_CONFIG


def _model(**config_overrides):
    config = {**copy.deepcopy(DEFAULT_TRAINING_CONFIG), "Global_features": 60, **config_overrides}
    return IM2Deep(config, nn.L1Loss())


def _dummy_batch(batch_size=2, global_features=60, charges=None):
    global_feats = torch.randn(batch_size, global_features) * 50 + 100
    if charges is not None:
        global_feats[:, CHARGE_FEATURE_INDEX] = torch.as_tensor(charges, dtype=torch.float32)
    return (
        torch.randn(batch_size, 60, 6),
        torch.randn(batch_size, 30, 6),
        global_feats,
        torch.randn(batch_size, 60, 20),  # (seq_len, one-hot vocab), permuted inside forward()
    )


class TestStandardizationBuffers:
    """Step 8.2: global-feature and target standardisation, opt-in via buffers."""

    def test_default_buffers_are_a_no_op(self):
        """A model that never calls set_*_scaler has mean=0/std=1 buffers."""
        model = _model()

        assert torch.equal(model.global_feat_mean, torch.zeros(60))
        assert torch.equal(model.global_feat_std, torch.ones(60))
        assert model.target_mean.item() == 0.0
        assert model.target_std.item() == 1.0

    def test_explicit_no_op_scaler_matches_default(self):
        """Setting mean=0/std=1 explicitly reproduces the untouched default output."""
        model = _model()
        batch = _dummy_batch()
        before = model(*batch)

        model.set_global_feature_scaler(torch.zeros(60), torch.ones(60))
        model.set_target_scaler(0.0, 1.0)
        after = model(*batch)

        assert torch.allclose(before, after)

    def test_target_scaler_destandardizes_output(self):
        """forward() always returns raw-scale predictions, regardless of the fitted scaler."""
        model = _model()
        batch = _dummy_batch()

        model.set_target_scaler(0.0, 1.0)
        raw_space_output = model(*batch)

        model.set_target_scaler(500.0, 200.0)
        scaled_output = model(*batch)

        # Only the final affine step differs: scaled = raw * 200 + 500. The
        # two calls otherwise share every weight, so this isolates exactly
        # the transform forward() is supposed to apply.
        assert torch.allclose(scaled_output, raw_space_output * 200.0 + 500.0, atol=1e-4)

    def test_global_feature_scaler_changes_output(self):
        """A non-trivial global-feature scaler measurably changes the prediction."""
        model = _model()
        batch = _dummy_batch()
        before = model(*batch)

        model.set_global_feature_scaler(torch.full((60,), 100.0), torch.full((60,), 50.0))
        after = model(*batch)

        assert not torch.allclose(before, after)

    def test_buffers_survive_a_strict_checkpoint_round_trip(self, tmp_path):
        """A freshly-trained checkpoint carries the fitted buffers, not the no-op default."""
        model = _model()
        model.set_global_feature_scaler(torch.arange(60, dtype=torch.float32), torch.full((60,), 2.0))
        model.set_target_scaler(555.5, 123.4)

        path = tmp_path / "buffers.ckpt"
        torch.save({"state_dict": model.state_dict(), "hyper_parameters": {"config": model.config}}, path)

        reloaded = IM2Deep(model.config, nn.L1Loss())
        missing, unexpected = reloaded.load_state_dict(
            torch.load(path, weights_only=False)["state_dict"], strict=True
        )
        assert not missing and not unexpected
        assert torch.equal(reloaded.global_feat_mean, torch.arange(60, dtype=torch.float32))
        assert reloaded.target_mean.item() == 555.5

    def test_old_checkpoint_missing_buffers_loads_with_strict_false(self):
        """A checkpoint predating these buffers (e.g. the bundled models) still loads."""
        model = _model()
        state_dict = model.state_dict()
        for key in ("global_feat_mean", "global_feat_std", "target_mean", "target_std"):
            del state_dict[key]

        reloaded = _model()
        missing, unexpected = reloaded.load_state_dict(state_dict, strict=False)

        assert set(missing) == {"global_feat_mean", "global_feat_std", "target_mean", "target_std"}
        assert not unexpected
        # Falls back to the no-op default rather than crashing or zeroing.
        assert torch.equal(reloaded.global_feat_std, torch.ones(60))
        assert reloaded.target_std.item() == 1.0


class TestLossVariants:
    """Step 8.3: the training objective (config['loss']), always reported as raw-Å² MAE."""

    def test_default_is_plain_l1(self):
        """"l1" (the default) is byte-identical to the pre-8.3 behaviour: self.criterion(y_hat, y)."""
        model = _model()
        y_hat = torch.tensor([510.0, 300.0])
        y = torch.tensor([500.0, 320.0])

        assert model._base_loss(y_hat, y).item() == pytest.approx(nn.L1Loss()(y_hat, y).item())

    def test_mape_is_mean_relative_error_as_a_percentage(self):
        """Reported as a percentage (x100), matching benchmark_models.py's mean_abs_pct_error.

        Not cosmetic: the unscaled fraction's gradient carries a 1/y factor
        that measured ~200x smaller than plain l1's at the same
        initialisation and left training stuck near its starting point
        within a normal epoch budget - see test_mape_gradient_is_not_starved.
        """
        model = _model(loss="mape")
        y_hat = torch.tensor([110.0, 190.0])
        y = torch.tensor([100.0, 200.0])

        # |10|/100 = 0.10, |10|/200 = 0.05 -> mean 0.075 -> x100 = 7.5%
        assert model._base_loss(y_hat, y).item() == pytest.approx(7.5)

    def test_mape_gradient_is_not_starved(self):
        """
        The x100 scaling keeps mape's gradient within the same order of
        magnitude as plain l1's, unlike the raw (unscaled) fraction.

        Measured directly on real proxy data: the unscaled mape's last-layer
        gradient norm was ~200x smaller than l1's at identical
        initialisation, and log_l1's (which has no equivalent rescaling
        available) was ~500x smaller - both left training stuck at a
        roughly constant MAE for the entire run. This is a coarser but
        cheap proxy for that same effect: mape's gradient should not be
        starved by more than about one order of magnitude relative to l1's.
        """
        y_hat = torch.tensor([510.0, 300.0, 700.0], requires_grad=True)
        y = torch.tensor([500.0, 320.0, 650.0])

        model = _model(loss="l1")
        model._base_loss(y_hat, y).backward()
        l1_grad_norm = y_hat.grad.norm().item()

        y_hat.grad = None
        model_mape = _model(loss="mape")
        model_mape._base_loss(y_hat, y).backward()
        mape_grad_norm = y_hat.grad.norm().item()

        assert mape_grad_norm > l1_grad_norm / 10

    def test_huber_matches_torch_reference(self):
        model = _model(loss="huber")
        y_hat = torch.tensor([510.0, 300.0])
        y = torch.tensor([500.0, 320.0])

        expected = nn.functional.huber_loss(y_hat, y, delta=5.0)
        assert model._base_loss(y_hat, y).item() == pytest.approx(expected.item())

    def test_huber_delta_is_configurable(self):
        model = _model(loss="huber", huber_delta=1.0)
        y_hat = torch.tensor([510.0])
        y = torch.tensor([500.0])

        expected = nn.functional.huber_loss(y_hat, y, delta=1.0)
        assert model._base_loss(y_hat, y).item() == pytest.approx(expected.item())

    def test_huber_gradient_is_not_starved(self):
        """Unlike log_l1/mape, huber has no division, so no scale pathology to guard against."""
        y_hat = torch.tensor([510.0, 300.0, 700.0], requires_grad=True)
        y = torch.tensor([500.0, 320.0, 650.0])

        model = _model(loss="l1")
        model._base_loss(y_hat, y).backward()
        l1_grad_norm = y_hat.grad.norm().item()

        y_hat.grad = None
        model_huber = _model(loss="huber")
        model_huber._base_loss(y_hat, y).backward()
        huber_grad_norm = y_hat.grad.norm().item()

        assert huber_grad_norm > l1_grad_norm / 10

    def test_unknown_loss_raises(self):
        model = _model(loss="not_a_real_loss")

        with pytest.raises(ValueError, match="not_a_real_loss"):
            model._base_loss(torch.tensor([1.0]), torch.tensor([1.0]))

    def test_validation_mae_stays_raw_l1_regardless_of_loss_choice(self):
        """The monitored metric must be comparable across configs with different `loss` values."""
        batch = (*_dummy_batch(batch_size=4), torch.rand(4) * 500 + 300)

        for loss_name in ("l1", "mape", "huber"):
            model = _model(loss=loss_name)
            atom, diatom, global_feats, one_hot, y = batch
            y_hat = model(atom, diatom, global_feats, one_hot).squeeze(1)
            assert model.mae(y_hat, y).item() == pytest.approx(nn.L1Loss()(y_hat, y).item())


class TestChargeEmbedding:
    """Step 8.4 item 1: charge as an embedding that FiLM-conditions the sequence branches."""

    def test_disabled_by_default_no_modules_created(self):
        model = _model()

        assert model.use_charge_embedding is False
        assert not hasattr(model, "charge_embedding")
        assert not hasattr(model, "charge_film")

    def test_enabled_creates_correctly_shaped_modules(self):
        model = _model(use_charge_embedding=True, charge_embedding_dim=8)

        assert model.charge_embedding.weight.shape == (7, 8)
        # film maps the embedding to (gamma, beta) for the full seq-concat width
        from im2deep._architectures.helpers import calculate_seq_concat_shape

        seq_concat_shape = calculate_seq_concat_shape(model.config)
        assert model.charge_film.out_features == 2 * seq_concat_shape

    def test_requires_add_ccs_features(self):
        with pytest.raises(ValueError, match="add_ccs_features"):
            _model(use_charge_embedding=True, add_ccs_features=False)

    def test_zero_init_is_an_exact_identity(self):
        """FiLM starts as gamma=0/beta=0, so enabling it changes nothing before training."""
        batch = _dummy_batch(charges=[2, 4])

        model_off = _model()
        model_on = _model(use_charge_embedding=True)
        # Same everything except the new (zero-initialised) charge modules,
        # so the rest of the network must share weights for this to be a
        # fair comparison - copy the shared submodules' state across.
        model_on.load_state_dict(model_off.state_dict(), strict=False)

        out_off = model_off(*batch)
        out_on = model_on(*batch)

        assert torch.allclose(out_off, out_on)

    def test_trained_film_changes_output(self):
        """Once charge_film moves away from zero-init, it measurably affects the output."""
        batch = _dummy_batch(charges=[2, 4])
        model = _model(use_charge_embedding=True)
        before = model(*batch)

        with torch.no_grad():
            model.charge_film.weight.add_(0.1)
            model.charge_film.bias.add_(0.1)
        after = model(*batch)

        assert not torch.allclose(before, after)

    def test_different_charges_reach_different_embeddings(self):
        """Two otherwise-identical inputs differing only in charge get different FiLM parameters."""
        model = _model(use_charge_embedding=True)
        with torch.no_grad():
            model.charge_film.weight.add_(0.1)
            model.charge_film.bias.add_(0.1)

        atom, diatom, global_feats, one_hot = _dummy_batch(batch_size=1, charges=[2])
        out_z2 = model(atom, diatom, global_feats.clone(), one_hot)

        global_feats_z5 = global_feats.clone()
        global_feats_z5[:, CHARGE_FEATURE_INDEX] = 5.0
        out_z5 = model(atom, diatom, global_feats_z5, one_hot)

        assert not torch.allclose(out_z2, out_z5)

    def test_charge_index_matches_deeplc_output(self):
        """CHARGE_FEATURE_INDEX must match where DeepLC actually puts charge, not just its docstring."""
        from deeplc._features import encode_peptidoform
        from psm_utils.peptidoform import Peptidoform

        for charge in (1, 3, 6):
            pf = Peptidoform(f"ACDEFGHIK/{charge}")
            feats = encode_peptidoform(
                pf, padding_length=60, add_ccs_features=True, add_terminal_composition=False
            )
            assert feats["matrix_global"][CHARGE_FEATURE_INDEX] == charge


class TestOneHotAblation:
    """Step 8.1/8.4: use_one_hot=False removes the one-hot branch entirely."""

    def test_enabled_by_default(self):
        model = _model()

        assert model.use_one_hot is True
        assert hasattr(model, "OneHot")

    def test_disabled_removes_the_module(self):
        model = _model(use_one_hot=False)

        assert model.use_one_hot is False
        assert not hasattr(model, "OneHot")

    def test_disabled_model_runs_and_ignores_one_hot_input(self):
        """forward() must not touch `one_hot` at all when the branch is off."""
        model = _model(use_one_hot=False)
        atom, diatom, global_feats, one_hot = _dummy_batch()

        out = model(atom, diatom, global_feats, one_hot)

        assert out.shape == (2, 1)
        assert torch.all(torch.isfinite(out))

    def test_concat_shape_shrinks_when_disabled(self):
        from im2deep._architectures.helpers import calculate_concat_shape

        config_with = {**copy.deepcopy(DEFAULT_TRAINING_CONFIG), "Global_features": 60}
        config_without = {**config_with, "use_one_hot": False}

        shape_with = calculate_concat_shape(config_with)
        shape_without = calculate_concat_shape(config_without)

        onehot_out_shape = (60 // config_with["OneHot_MaxPool_kernel_size"]) * config_with[
            "OneHot_out_channels"
        ]
        assert shape_with - shape_without == onehot_out_shape

    def test_head_input_size_matches_actual_concatenation(self):
        """The Concat head's first layer must accept exactly what forward() feeds it."""
        model = _model(use_one_hot=False)
        atom, diatom, global_feats, one_hot = _dummy_batch()

        # Would raise a shape-mismatch RuntimeError if total_input_size (used
        # to size Concat[0]) didn't shrink in lockstep with the ablation.
        model(atom, diatom, global_feats, one_hot)


class TestAttentionPooling:
    """Step 8.4 item 2: AttentionPool1d replaces Flatten(), length-agnostic output width."""

    def test_disabled_by_default_uses_flatten(self):
        model = _model()

        assert model.use_attention_pooling is False
        assert isinstance(model.ConvAtomComp[-1], nn.Flatten)
        assert isinstance(model.ConvDiatomComp[-1], nn.Flatten)

    def test_enabled_uses_attention_pool(self):
        from im2deep._architectures.blocks import AttentionPool1d

        model = _model(use_attention_pooling=True)

        assert isinstance(model.ConvAtomComp[-1], AttentionPool1d)
        assert isinstance(model.ConvDiatomComp[-1], AttentionPool1d)

    def test_runs_and_produces_finite_output(self):
        model = _model(use_attention_pooling=True)
        atom, diatom, global_feats, one_hot = _dummy_batch()

        out = model(atom, diatom, global_feats, one_hot)

        assert out.shape == (2, 1)
        assert torch.all(torch.isfinite(out))

    def test_output_width_is_length_agnostic(self):
        """Unlike Flatten, the branch's output width shouldn't depend on padding_length."""
        from im2deep._architectures.helpers import calculate_seq_concat_shape

        config = {**copy.deepcopy(DEFAULT_TRAINING_CONFIG), "Global_features": 60}
        config_pooled = {**config, "use_attention_pooling": True}

        # Flattened width scales with sequence length (60 here); pooled width
        # is just the channel count, independent of it.
        flattened_width = calculate_seq_concat_shape(config)
        pooled_width = calculate_seq_concat_shape(config_pooled)
        expected_pooled = (
            config["AtomComp_out_channels_start"] // 4 + config["DiatomComp_out_channels_start"] // 2
        )
        assert pooled_width == expected_pooled
        assert pooled_width < flattened_width

    def test_composes_with_charge_film(self):
        """FiLM's expected width must track attention pooling's smaller seq-concat shape."""
        model = _model(use_attention_pooling=True, use_charge_embedding=True)
        atom, diatom, global_feats, one_hot = _dummy_batch(charges=[2, 4])

        out = model(atom, diatom, global_feats, one_hot)

        assert torch.all(torch.isfinite(out))

    def test_attention_weights_sum_to_one_over_positions(self):
        from im2deep._architectures.blocks import AttentionPool1d

        pool = AttentionPool1d(channels=4, initializer=lambda w: nn.init.normal_(w, std=0.05))
        x = torch.randn(2, 4, 15)

        weights = torch.softmax(pool.score(x), dim=-1)

        assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 1), atol=1e-5)


class TestFlexTrunkBlock:
    """Step 8.4 items 8+9: FlexTrunk itself, ported from DeepLC's _FlexCNNEncoder."""

    def _trunk(self, **overrides):
        from im2deep._architectures.blocks import FlexTrunk

        kwargs = {"embed_dim": 4, "channels": (8, 8), "kernel_size": 3, "stem_channels": 6, "stem_layers": 2}
        kwargs.update(overrides)
        return FlexTrunk(**kwargs)

    def _one_hot(self, batch_size, length, real_length):
        """(batch, 20, length) one-hot, real for the first real_length positions, zero after."""
        one_hot = torch.zeros(batch_size, 20, length)
        for pos in range(real_length):
            one_hot[:, pos % 20, pos] = 1.0
        return one_hot

    def test_residue_indices_matches_argmax_on_real_positions(self):
        trunk = self._trunk()
        one_hot = self._one_hot(batch_size=2, length=10, real_length=10)

        idx = trunk.residue_indices(one_hot)

        assert torch.equal(idx, torch.arange(10).unsqueeze(0).expand(2, 10))

    def test_residue_indices_flags_all_zero_columns_as_padding(self):
        trunk = self._trunk()
        one_hot = self._one_hot(batch_size=2, length=10, real_length=6)

        idx = trunk.residue_indices(one_hot)

        assert torch.all(idx[:, 6:] == trunk.PAD_INDEX)
        assert torch.all(idx[:, :6] != trunk.PAD_INDEX)

    def test_forward_runs_and_produces_finite_output(self):
        trunk = self._trunk()
        atom = torch.randn(3, 6, 12)
        one_hot = self._one_hot(batch_size=3, length=12, real_length=8)

        out = trunk(atom, one_hot)

        assert out.shape == (3, 2 * 8)  # sum + max, each channels[-1] wide
        assert torch.all(torch.isfinite(out))

    def test_padding_positions_do_not_change_the_pooled_output(self):
        """Masking means only the real positions should affect sum/max pooling."""
        torch.manual_seed(0)
        trunk = self._trunk()
        real_length = 7
        length_a, length_b = 10, 20

        atom_a = torch.randn(1, 6, length_a)
        atom_b = torch.zeros(1, 6, length_b)
        atom_b[:, :, :length_a] = atom_a
        one_hot_a = self._one_hot(batch_size=1, length=length_a, real_length=real_length)
        one_hot_b = self._one_hot(batch_size=1, length=length_b, real_length=real_length)

        out_a = trunk(atom_a, one_hot_a)
        out_b = trunk(atom_b, one_hot_b)

        assert torch.allclose(out_a, out_b, atol=1e-5)

    def test_no_stem_feeds_raw_atom_counts(self):
        trunk = self._trunk(stem_channels=0)

        assert trunk.stem is None
        atom = torch.randn(2, 6, 10)
        one_hot = self._one_hot(batch_size=2, length=10, real_length=10)
        out = trunk(atom, one_hot)
        assert torch.all(torch.isfinite(out))


class TestFlexTrunkModel:
    """Step 8.4 items 8+9: FlexTrunk wired into IM2Deep, replacing three branches."""

    def test_disabled_by_default(self):
        model = _model()

        assert model.use_flex_trunk is False
        assert hasattr(model, "ConvAtomComp")
        assert hasattr(model, "ConvDiatomComp")
        assert not hasattr(model, "flex_trunk")

    def test_enabled_retires_the_three_branches(self):
        model = _model(use_flex_trunk=True)

        assert hasattr(model, "flex_trunk")
        assert not hasattr(model, "ConvAtomComp")
        assert not hasattr(model, "ConvDiatomComp")
        assert not hasattr(model, "OneHot")

    def test_use_one_hot_is_silently_moot_when_enabled(self):
        # use_one_hot defaults to True; enabling flex_trunk must not raise
        # and must not build the (now redundant) OneHot branch anyway.
        model = _model(use_flex_trunk=True, use_one_hot=True)

        assert model.use_one_hot is False
        assert not hasattr(model, "OneHot")

    def test_runs_and_produces_finite_output(self):
        model = _model(use_flex_trunk=True)
        atom, diatom, global_feats, one_hot = _dummy_batch()

        out = model(atom, diatom, global_feats, one_hot)

        assert out.shape == (2, 1)
        assert torch.all(torch.isfinite(out))

    def test_concat_shape_matches_actual_seq_feats_width(self):
        from im2deep._architectures.helpers import calculate_seq_concat_shape

        config = {
            **copy.deepcopy(DEFAULT_TRAINING_CONFIG),
            "Global_features": 60,
            "use_flex_trunk": True,
            "flex_trunk_channels": [12, 10],
        }
        model = IM2Deep(config, nn.L1Loss())
        atom, diatom, global_feats, one_hot = _dummy_batch()

        seq_feats = model.flex_trunk(
            atom.permute(0, 2, 1), one_hot.permute(0, 2, 1)
        )

        assert seq_feats.shape[1] == calculate_seq_concat_shape(config)
        assert calculate_seq_concat_shape(config) == 2 * 10

    def test_uses_far_fewer_parameters_than_the_four_branch_default(self):
        default_model = _model()
        flex_model = _model(use_flex_trunk=True)

        default_params = sum(p.numel() for p in default_model.parameters())
        flex_params = sum(p.numel() for p in flex_model.parameters())

        assert flex_params < default_params

    def test_composes_with_charge_film(self):
        model = _model(use_flex_trunk=True, use_charge_embedding=True)
        atom, diatom, global_feats, one_hot = _dummy_batch(charges=[2, 4])

        out = model(atom, diatom, global_feats, one_hot)

        assert torch.all(torch.isfinite(out))

    def test_diatom_comp_is_ignored(self):
        """matrix_sum is redundant once the trunk sees matrix directly (item 8)."""
        model = _model(use_flex_trunk=True)
        atom, diatom, global_feats, one_hot = _dummy_batch()
        different_diatom = torch.randn_like(diatom)

        out_a = model(atom, diatom, global_feats, one_hot)
        out_b = model(atom, different_diatom, global_feats, one_hot)

        assert torch.allclose(out_a, out_b)


class TestTransformerTrunkBlock:
    """Step 8.4 item 3: RelativePairBias, PerceiverReadout, TransformerTrunk."""

    def _pair_bias(self, **overrides):
        from im2deep._architectures.transformer_trunk import RelativePairBias

        kwargs = {"num_heads": 2, "max_relative_distance": 5, "max_terminus_distance": 3}
        kwargs.update(overrides)
        return RelativePairBias(**kwargs)

    def _trunk(self, **overrides):
        from im2deep._architectures.transformer_trunk import TransformerTrunk

        kwargs = {
            "d_model": 8,
            "num_layers": 2,
            "num_heads": 2,
            "ffn_dim": 16,
            "max_relative_distance": 5,
            "max_terminus_distance": 3,
        }
        kwargs.update(overrides)
        return TransformerTrunk(**kwargs)

    def _atom_comp(self, batch_size, length, real_length):
        """(batch, length, 6): nonzero for the first real_length positions, zero after."""
        atom = torch.zeros(batch_size, length, 6)
        atom[:, :real_length, :] = torch.randn(batch_size, real_length, 6)
        return atom

    def test_pair_bias_output_shape(self):
        pair_bias = self._pair_bias(num_heads=3)

        out = pair_bias(real_lengths=torch.tensor([10, 5]), seq_len=10)

        assert out.shape == (2, 3, 10, 10)

    def test_relative_component_is_asymmetric_by_design(self):
        """i-j and j-i map to different buckets - direction should matter."""
        pair_bias = self._pair_bias()
        nn.init.zeros_(pair_bias.terminus.weight)

        out = pair_bias(real_lengths=torch.tensor([10]), seq_len=10)

        assert not torch.allclose(out, out.transpose(-1, -2))

    def test_relative_component_diagonal_is_a_fixed_bucket(self):
        """i==j always means distance 0, regardless of where i is."""
        pair_bias = self._pair_bias()
        nn.init.zeros_(pair_bias.terminus.weight)

        out = pair_bias(real_lengths=torch.tensor([10]), seq_len=10)
        diagonal = torch.diagonal(out[0, 0])

        assert torch.allclose(diagonal, diagonal[0] * torch.ones_like(diagonal))

    def test_relative_component_clamps_beyond_max_distance(self):
        pair_bias = self._pair_bias(max_relative_distance=3)
        nn.init.zeros_(pair_bias.terminus.weight)

        out = pair_bias(real_lengths=torch.tensor([10]), seq_len=10)

        # i-j=3 and i-j=5 both clamp to the same (boundary) bucket.
        assert torch.allclose(out[0, :, 5, 2], out[0, :, 7, 2])

    def test_terminus_component_is_symmetric(self):
        pair_bias = self._pair_bias()
        nn.init.zeros_(pair_bias.relative.weight)

        out = pair_bias(real_lengths=torch.tensor([10]), seq_len=10)

        assert torch.allclose(out, out.transpose(-1, -2))

    def test_terminus_distance_formula_matches_hand_computation(self):
        pair_bias = self._pair_bias(num_heads=1, max_terminus_distance=10)
        nn.init.zeros_(pair_bias.relative.weight)
        # Craft the terminus table so its output value equals the bucket index.
        with torch.no_grad():
            pair_bias.terminus.weight.copy_(torch.arange(11).float().unsqueeze(1))

        out = pair_bias(real_lengths=torch.tensor([6]), seq_len=6)

        # length 6: closest-to-either-terminus distance per position is
        # [0, 1, 2, 2, 1, 0]; terminus_distance(i,j) = min(closest[i], closest[j]).
        assert out[0, 0, 0, 0].item() == 0
        assert out[0, 0, 2, 3].item() == 2
        assert out[0, 0, 0, 5].item() == 0

    def test_readout_output_shape_single_query(self):
        from im2deep._architectures.transformer_trunk import PerceiverReadout

        readout = PerceiverReadout(d_model=8, num_heads=2, num_queries=1, dropout=0.0)
        residue_hidden = torch.randn(2, 10, 8)
        attn_mask = torch.zeros(2 * 2, 1, 10)

        out = readout(residue_hidden, attn_mask)

        assert out.shape == (2, 8)

    def test_readout_output_shape_multiple_queries(self):
        from im2deep._architectures.transformer_trunk import PerceiverReadout

        readout = PerceiverReadout(d_model=8, num_heads=2, num_queries=3, dropout=0.0)
        residue_hidden = torch.randn(2, 10, 8)
        attn_mask = torch.zeros(2 * 2, 3, 10)

        out = readout(residue_hidden, attn_mask)

        assert out.shape == (2, 3, 8)

    def test_trunk_runs_and_produces_finite_output(self):
        trunk = self._trunk()
        atom = self._atom_comp(batch_size=3, length=12, real_length=8)
        charge_idx = torch.tensor([1, 2, 4])

        out = trunk(atom, charge_idx)

        assert out.shape == (3, 8)
        assert torch.all(torch.isfinite(out))

    def test_padding_positions_do_not_change_the_pooled_output(self):
        """Same real peptide, different amounts of trailing padding -> identical output."""
        torch.manual_seed(0)
        trunk = self._trunk()
        real_length = 7
        length_a, length_b = 10, 20

        atom_a = torch.randn(1, length_a, 6)
        atom_a[:, real_length:, :] = 0.0
        atom_b = torch.zeros(1, length_b, 6)
        atom_b[:, :length_a, :] = atom_a
        charge_idx = torch.tensor([2])

        out_a = trunk(atom_a, charge_idx)
        out_b = trunk(atom_b, charge_idx)

        assert torch.allclose(out_a, out_b, atol=1e-5)

    def test_register_token_excluded_from_readout_keys(self):
        # The readout's key/value go straight to _sdpa_attention rather than
        # through cross_attn.forward() (see _sdpa_attention's docstring for
        # why), so a forward hook on cross_attn no longer observes the call -
        # patch _sdpa_attention itself instead. PerceiverReadout.forward
        # looks it up as a module global at call time, so this redirect
        # applies regardless of which module imported PerceiverReadout.
        import im2deep._architectures.transformer_trunk as transformer_trunk_module

        trunk = self._trunk()
        atom = self._atom_comp(batch_size=2, length=10, real_length=10)
        charge_idx = torch.tensor([1, 3])
        captured = {}
        original = transformer_trunk_module._sdpa_attention

        def spy(mha, query, key, value, attn_mask):
            captured["key_len"] = key.shape[1]
            return original(mha, query, key, value, attn_mask)

        transformer_trunk_module._sdpa_attention = spy
        try:
            trunk(atom, charge_idx)
        finally:
            transformer_trunk_module._sdpa_attention = original

        assert captured["key_len"] == 10  # residue positions only, not 10 + 1 register token

    def test_charge_sensitivity(self):
        """Different charge, same residues -> different output ('charge attends back')."""
        torch.manual_seed(0)
        trunk = self._trunk()
        atom = self._atom_comp(batch_size=1, length=10, real_length=10)

        out_low = trunk(atom, torch.tensor([1]))
        out_high = trunk(atom, torch.tensor([5]))

        assert not torch.allclose(out_low, out_high)

    # -- step 8.10: return_attention is opt-in and additive --

    def test_block_return_attention_shape_and_same_output(self):
        """need_weights=True must not change the block's own output."""
        from im2deep._architectures.transformer_trunk import PreNormTransformerBlock

        torch.manual_seed(0)
        block = PreNormTransformerBlock(d_model=8, num_heads=2, ffn_dim=16, dropout=0.0)
        block.eval()
        x = torch.randn(2, 5, 8)
        attn_mask = torch.zeros(2 * 2, 5, 5)

        out_plain = block(x, attn_mask)
        out_with_attn, attn_weights = block(x, attn_mask, return_attention=True)

        assert torch.allclose(out_plain, out_with_attn, atol=1e-4)
        assert attn_weights.shape == (2, 2, 5, 5)

    def test_readout_return_attention_shape_and_same_output(self):
        from im2deep._architectures.transformer_trunk import PerceiverReadout

        torch.manual_seed(0)
        readout = PerceiverReadout(d_model=8, num_heads=2, num_queries=1, dropout=0.0)
        readout.eval()
        residue_hidden = torch.randn(2, 10, 8)
        attn_mask = torch.zeros(2 * 2, 1, 10)

        out_plain = readout(residue_hidden, attn_mask)
        out_with_attn, attn_weights = readout(residue_hidden, attn_mask, return_attention=True)

        assert torch.allclose(out_plain, out_with_attn, atol=1e-4)
        assert attn_weights.shape == (2, 2, 1, 10)

    def test_trunk_return_attention_shapes_and_same_pooled_output(self):
        torch.manual_seed(0)
        trunk = self._trunk()  # d_model=8, num_layers=2, num_heads=2
        trunk.eval()
        atom = self._atom_comp(batch_size=2, length=10, real_length=7)
        charge_idx = torch.tensor([1, 3])

        out_plain = trunk(atom, charge_idx)
        out_with_attn, attention = trunk(atom, charge_idx, return_attention=True)

        assert torch.allclose(out_plain, out_with_attn, atol=1e-4)
        assert set(attention.keys()) == {"self_attention", "readout_attention"}
        assert len(attention["self_attention"]) == 2  # num_layers
        # seq_len=10 (the padded array length, not real_length) + register token.
        assert attention["self_attention"][0].shape == (2, 2, 11, 11)
        assert attention["readout_attention"].shape == (2, 2, 1, 10)


class TestTransformerTrunkModel:
    """Step 8.4 item 3: TransformerTrunk wired into IM2Deep."""

    def test_disabled_by_default(self):
        model = _model()

        assert model.use_transformer_trunk is False
        assert hasattr(model, "ConvAtomComp")
        assert not hasattr(model, "transformer_trunk")

    def test_enabled_retires_conv_branches_but_keeps_one_hot(self):
        """Unlike use_flex_trunk, this arm does not fold in residue identity."""
        model = _model(use_transformer_trunk=True)

        assert hasattr(model, "transformer_trunk")
        assert not hasattr(model, "ConvAtomComp")
        assert not hasattr(model, "ConvDiatomComp")
        assert hasattr(model, "OneHot")
        assert model.use_one_hot is True

    def test_mutual_exclusion_with_flex_trunk_raises(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            _model(use_transformer_trunk=True, use_flex_trunk=True)

    def test_requires_add_ccs_features(self):
        with pytest.raises(ValueError, match="add_ccs_features"):
            _model(use_transformer_trunk=True, add_ccs_features=False)

    def test_runs_and_produces_finite_output(self):
        model = _model(use_transformer_trunk=True)
        atom, diatom, global_feats, one_hot = _dummy_batch(charges=[2, 4])

        out = model(atom, diatom, global_feats, one_hot)

        assert out.shape == (2, 1)
        assert torch.all(torch.isfinite(out))

    def test_composes_with_charge_embedding(self):
        model = _model(use_transformer_trunk=True, use_charge_embedding=True)
        atom, diatom, global_feats, one_hot = _dummy_batch(charges=[2, 4])

        out = model(atom, diatom, global_feats, one_hot)

        assert torch.all(torch.isfinite(out))

    def test_use_attention_pooling_is_silently_moot(self):
        model = _model(use_transformer_trunk=True, use_attention_pooling=True)

        assert not hasattr(model, "ConvAtomComp")
        atom, diatom, global_feats, one_hot = _dummy_batch(charges=[2, 4])
        out = model(atom, diatom, global_feats, one_hot)
        assert torch.all(torch.isfinite(out))

    def test_concat_shape_matches_actual_seq_feats_width(self):
        from im2deep._architectures.helpers import calculate_seq_concat_shape

        config = {
            **copy.deepcopy(DEFAULT_TRAINING_CONFIG),
            "Global_features": 60,
            "use_transformer_trunk": True,
            "transformer_d_model": 12,
        }
        model = IM2Deep(config, nn.L1Loss())
        atom, diatom, global_feats, one_hot = _dummy_batch(charges=[2, 4])

        atom_raw = atom
        charge_idx = global_feats[:, CHARGE_FEATURE_INDEX].round().long().clamp(0, 6)
        seq_feats = model.transformer_trunk(atom_raw, charge_idx)

        assert seq_feats.shape[1] == calculate_seq_concat_shape(config)
        assert calculate_seq_concat_shape(config) == 12

    def test_return_attention_requires_transformer_trunk(self):
        model = _model()  # use_transformer_trunk=False by default
        atom, diatom, global_feats, one_hot = _dummy_batch(charges=[2, 4])

        with pytest.raises(ValueError, match="use_transformer_trunk"):
            model(atom, diatom, global_feats, one_hot, return_attention=True)

    def test_return_attention_matches_default_output(self):
        """return_attention=True must be additive: same prediction either way."""
        torch.manual_seed(0)
        model = _model(use_transformer_trunk=True)
        model.eval()
        atom, diatom, global_feats, one_hot = _dummy_batch(charges=[2, 4])

        out_plain = model(atom, diatom, global_feats, one_hot)
        out_with_attn, attention = model(
            atom, diatom, global_feats, one_hot, return_attention=True
        )

        assert torch.allclose(out_plain, out_with_attn, atol=1e-4)
        assert set(attention.keys()) == {"self_attention", "readout_attention"}


class TestMixup:
    """Step 8.3: composition mixup, applied inside training_step only."""

    def _batch(self, batch_size=4):
        atom, diatom, global_feats, one_hot = _dummy_batch(batch_size=batch_size)
        y = torch.rand(batch_size) * 500 + 300
        return atom, diatom, global_feats, one_hot, y

    def test_disabled_by_default_is_a_no_op(self):
        model = _model()
        batch = self._batch()

        result = model._apply_mixup(*batch)

        # batch is 5-tuple (no mol_desc); result is 6-tuple (mol_desc=None
        # appended) - intentionally comparing only the first 5.
        for original, returned in zip(batch, result, strict=False):
            assert torch.equal(original, returned)

    def test_alpha_zero_is_a_no_op(self):
        model = _model(mixup_alpha=0.0)
        batch = self._batch()

        result = model._apply_mixup(*batch)

        # batch is 5-tuple (no mol_desc); result is 6-tuple (mol_desc=None
        # appended) - intentionally comparing only the first 5.
        for original, returned in zip(batch, result, strict=False):
            assert torch.equal(original, returned)

    def test_enabled_mixes_every_tensor_with_the_same_permutation(self):
        model = _model(mixup_alpha=1.0)
        atom, diatom, global_feats, one_hot, y = self._batch()

        mixed_atom, mixed_diatom, mixed_global, mixed_one_hot, mixed_y, mixed_mol = model._apply_mixup(
            atom, diatom, global_feats, one_hot, y
        )

        assert mixed_mol is None
        # lambda ~ Beta(1,1) = Uniform(0,1) essentially never lands exactly on
        # 0 or 1, so every mixed tensor should differ from the original.
        assert not torch.equal(mixed_atom, atom)
        assert not torch.equal(mixed_y, y)
        # Every value must remain within the convex hull of the two inputs it
        # came from (a basic sanity check that this is a real interpolation).
        assert (mixed_y.min() >= min(y.min(), y.max()) - 1e-4) and (
            mixed_y.max() <= max(y.min(), y.max()) + 1e-4
        )

    def test_mol_desc_is_mixed_when_present(self):
        # _apply_mixup only branches on whether mol_desc is passed, not on
        # config["add_X_mol"], so a plain model is enough here.
        model = _model(mixup_alpha=1.0)
        atom, diatom, global_feats, one_hot, y = self._batch()
        mol_desc = torch.randn(4, 13, 60)

        *_, mixed_mol = model._apply_mixup(atom, diatom, global_feats, one_hot, y, mol_desc)

        assert mixed_mol is not None
        assert not torch.equal(mixed_mol, mol_desc)


class TestSiblingDeltaLoss:
    """Step 8.3: sibling delta loss, L1 on (pred_a - pred_b) vs (obs_a - obs_b)."""

    def test_disabled_by_default_is_zero(self):
        model = _model()
        y_hat = torch.tensor([400.0, 410.0, 420.0])
        y = torch.tensor([405.0, 408.0, 425.0])
        group_id = torch.tensor([0, 0, 1])

        loss = model._sibling_delta_loss(y_hat, y, group_id)

        assert loss.item() == 0.0

    def test_zero_weight_is_zero(self):
        model = _model(sibling_loss_weight=0.0)
        y_hat = torch.tensor([400.0, 410.0])
        y = torch.tensor([405.0, 408.0])
        group_id = torch.tensor([0, 0])

        assert model._sibling_delta_loss(y_hat, y, group_id).item() == 0.0

    def test_no_matching_groups_is_zero(self):
        model = _model(sibling_loss_weight=1.0)
        y_hat = torch.tensor([400.0, 410.0, 420.0])
        y = torch.tensor([405.0, 408.0, 425.0])
        group_id = torch.tensor([0, 1, 2])  # every row its own group

        assert model._sibling_delta_loss(y_hat, y, group_id).item() == 0.0

    def test_matches_manual_l1_on_the_one_sibling_pair(self):
        model = _model(sibling_loss_weight=1.0)
        y_hat = torch.tensor([400.0, 410.0, 500.0])
        y = torch.tensor([405.0, 408.0, 550.0])
        group_id = torch.tensor([0, 0, 1])  # only rows 0/1 are siblings

        loss = model._sibling_delta_loss(y_hat, y, group_id)

        pred_delta = y_hat[0] - y_hat[1]
        obs_delta = y[0] - y[1]
        expected = torch.nn.functional.l1_loss(pred_delta, obs_delta)
        assert torch.allclose(loss, expected)

    def test_weight_scales_the_loss(self):
        model_a = _model(sibling_loss_weight=1.0)
        model_b = _model(sibling_loss_weight=3.0)
        y_hat = torch.tensor([400.0, 410.0])
        y = torch.tensor([405.0, 408.0])
        group_id = torch.tensor([0, 0])

        loss_a = model_a._sibling_delta_loss(y_hat, y, group_id)
        loss_b = model_b._sibling_delta_loss(y_hat, y, group_id)

        assert torch.allclose(loss_b, 3.0 * loss_a)

    def test_ignores_config_loss_choice(self):
        # The delta term is documented as always plain L1, independent of
        # config["loss"] - verify a non-l1 choice does not change it.
        model = _model(sibling_loss_weight=1.0, loss="huber")
        y_hat = torch.tensor([400.0, 410.0])
        y = torch.tensor([405.0, 408.0])
        group_id = torch.tensor([0, 0])

        loss = model._sibling_delta_loss(y_hat, y, group_id)
        expected = torch.nn.functional.l1_loss(y_hat[0] - y_hat[1], y[0] - y[1])
        assert torch.allclose(loss, expected)

    def test_groups_larger_than_two_cover_every_pair(self):
        model = _model(sibling_loss_weight=1.0)
        y_hat = torch.tensor([100.0, 110.0, 120.0])
        y = torch.tensor([100.0, 130.0, 160.0])
        group_id = torch.tensor([0, 0, 0])  # one group of three -> 3 pairs

        loss = model._sibling_delta_loss(y_hat, y, group_id)

        pairs = [(0, 1), (0, 2), (1, 2)]
        pred_deltas = torch.stack([y_hat[a] - y_hat[b] for a, b in pairs])
        obs_deltas = torch.stack([y[a] - y[b] for a, b in pairs])
        expected = torch.nn.functional.l1_loss(pred_deltas, obs_deltas)
        assert torch.allclose(loss, expected)


class TestRankingLoss:
    """Step 8.3: pairwise ranking loss within an m/z window."""

    def test_disabled_by_default_is_zero(self):
        model = _model()
        y_hat = torch.tensor([400.0, 410.0])
        y = torch.tensor([405.0, 408.0])
        mz = torch.tensor([500.0, 501.0])

        assert model._ranking_loss(y_hat, y, mz).item() == 0.0

    def test_pairs_outside_window_are_ignored(self):
        model = _model(ranking_loss_weight=1.0, ranking_mz_window=3.0)
        y_hat = torch.tensor([400.0, 999.0])  # badly mis-ordered if paired
        y = torch.tensor([405.0, 408.0])
        mz = torch.tensor([500.0, 600.0])  # 100 Th apart, outside the window

        assert model._ranking_loss(y_hat, y, mz).item() == 0.0

    def test_correctly_ordered_pair_within_margin_is_zero(self):
        model = _model(ranking_loss_weight=1.0, ranking_mz_window=3.0, ranking_margin=0.0)
        y_hat = torch.tensor([400.0, 410.0])  # same order as y
        y = torch.tensor([405.0, 408.0])
        mz = torch.tensor([500.0, 501.0])

        assert model._ranking_loss(y_hat, y, mz).item() == pytest.approx(0.0, abs=1e-6)

    def test_misordered_pair_within_window_is_penalized(self):
        model = _model(ranking_loss_weight=1.0, ranking_mz_window=3.0, ranking_margin=1.0)
        # y says row 0 < row 1, y_hat says the opposite.
        y_hat = torch.tensor([410.0, 400.0])
        y = torch.tensor([405.0, 408.0])
        mz = torch.tensor([500.0, 501.0])

        loss = model._ranking_loss(y_hat, y, mz)

        assert loss.item() > 0.0

    def test_tied_observed_ccs_pair_is_ignored(self):
        model = _model(ranking_loss_weight=1.0, ranking_mz_window=3.0)
        y_hat = torch.tensor([400.0, 999.0])
        y = torch.tensor([405.0, 405.0])  # exact tie -> no ordering to supervise
        mz = torch.tensor([500.0, 501.0])

        assert model._ranking_loss(y_hat, y, mz).item() == 0.0

    def test_weight_scales_the_loss(self):
        model_a = _model(ranking_loss_weight=1.0, ranking_mz_window=3.0, ranking_margin=1.0)
        model_b = _model(ranking_loss_weight=4.0, ranking_mz_window=3.0, ranking_margin=1.0)
        y_hat = torch.tensor([410.0, 400.0])
        y = torch.tensor([405.0, 408.0])
        mz = torch.tensor([500.0, 501.0])

        loss_a = model_a._ranking_loss(y_hat, y, mz)
        loss_b = model_b._ranking_loss(y_hat, y, mz)

        assert torch.allclose(loss_b, 4.0 * loss_a)


class TestPairwiseLossMixupIncompatibility:
    """Step 8.3: mixup blends rows, corrupting group_id/mz - must raise, not silently combine."""

    def test_sibling_loss_with_mixup_raises(self):
        with pytest.raises(ValueError, match="mixup_alpha"):
            _model(mixup_alpha=0.5, sibling_loss_weight=1.0)

    def test_ranking_loss_with_mixup_raises(self):
        with pytest.raises(ValueError, match="mixup_alpha"):
            _model(mixup_alpha=0.5, ranking_loss_weight=1.0)

    def test_either_alone_with_mixup_is_fine(self):
        _model(mixup_alpha=0.5)
        _model(sibling_loss_weight=1.0)
        _model(ranking_loss_weight=1.0)


class TestHeteroscedasticHead:
    """Step 8.5: a (mu, b) head trained with Laplace NLL instead of a single output."""

    def test_disabled_by_default(self):
        model = _model()

        assert model.use_heteroscedastic_head is False
        assert model.Concat[-1].out_features == 1

    def test_enabled_changes_final_layer_to_two_outputs(self):
        model = _model(use_heteroscedastic_head=True)

        assert model.Concat[-1].out_features == 2

    def test_disabled_forward_returns_a_single_tensor(self):
        model = _model()
        batch = _dummy_batch()

        out = model(*batch)

        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 1)

    def test_enabled_forward_returns_mu_b_tuple(self):
        model = _model(use_heteroscedastic_head=True)
        batch = _dummy_batch()

        out = model(*batch)

        assert isinstance(out, tuple) and len(out) == 2
        mu, b = out
        assert mu.shape == (2,)
        assert b.shape == (2,)

    def test_b_is_always_positive_and_finite(self):
        model = _model(use_heteroscedastic_head=True)
        batch = _dummy_batch(batch_size=8)

        _, b = model(*batch)

        assert torch.all(b > 0)
        assert torch.all(torch.isfinite(b))

    def test_b_scales_with_target_std_not_shifted_by_target_mean(self):
        """b is a scale parameter: only the multiplicative half of destandardisation applies."""
        model = _model(use_heteroscedastic_head=True)
        batch = _dummy_batch()

        model.set_target_scaler(0.0, 1.0)
        _, b_unit = model(*batch)
        model.set_target_scaler(500.0, 200.0)  # same std ratio check, now with a nonzero mean
        _, b_scaled = model(*batch)

        assert torch.allclose(b_scaled, b_unit * 200.0, rtol=1e-4)

    def test_mu_destandardizes_the_same_way_as_the_single_output_head(self):
        model = _model(use_heteroscedastic_head=True)
        batch = _dummy_batch()

        model.set_target_scaler(0.0, 1.0)
        mu_raw, _ = model(*batch)
        model.set_target_scaler(500.0, 200.0)
        mu_scaled, _ = model(*batch)

        assert torch.allclose(mu_scaled, mu_raw * 200.0 + 500.0, atol=1e-4)

    def test_laplace_nll_matches_hand_computation(self):
        model = _model(use_heteroscedastic_head=True)
        mu = torch.tensor([490.0, 495.0])
        y = torch.tensor([500.0, 510.0])
        b = torch.tensor([10.0, 15.0])

        loss = model._laplace_nll(mu, y, b)

        expected = (torch.log(2.0 * b) + torch.abs(y - mu) / b).mean()
        assert torch.allclose(loss, expected)

    def test_laplace_nll_is_minimized_when_b_equals_the_error(self):
        """The Laplace MLE scale for a fixed error e=|y-mu| is exactly b=e."""
        model = _model(use_heteroscedastic_head=True)
        mu = torch.tensor([490.0, 495.0])
        y = torch.tensor([500.0, 510.0])
        error = torch.abs(y - mu)

        nll_at_error = model._laplace_nll(mu, y, error)
        nll_too_small = model._laplace_nll(mu, y, error / 10)
        nll_too_large = model._laplace_nll(mu, y, error * 10)

        assert nll_at_error < nll_too_small
        assert nll_at_error < nll_too_large

    def test_training_step_runs_and_produces_a_finite_loss(self):
        model = _model(use_heteroscedastic_head=True)
        atom, diatom, global_feats, one_hot = _dummy_batch(batch_size=4)
        y = torch.rand(4) * 500 + 300
        batch = (atom, diatom, global_feats, one_hot, y)

        loss = model.training_step(batch, 0)

        assert torch.isfinite(loss)
        assert loss.requires_grad

    def test_validation_mae_uses_only_mu_not_b(self):
        """The established invariant (Validation MAE is always plain L1 on the point
        estimate, regardless of config["loss"]) extends to this head too."""
        model = _model(use_heteroscedastic_head=True)
        atom, diatom, global_feats, one_hot = _dummy_batch(batch_size=4)
        y = torch.rand(4) * 500 + 300
        batch = (atom, diatom, global_feats, one_hot, y)

        mu, _ = model(atom, diatom, global_feats, one_hot)
        expected_mae = torch.nn.functional.l1_loss(mu, y)

        logged = {}
        model.log = lambda name, value, **kwargs: logged.__setitem__(name, value)
        model.validation_step(batch, 0)

        assert torch.allclose(logged["Validation MAE"], expected_mae)

    def test_predict_step_returns_only_mu(self):
        model = _model(use_heteroscedastic_head=True)
        atom, diatom, global_feats, one_hot = _dummy_batch(batch_size=4)
        y = torch.rand(4) * 500 + 300
        batch = (atom, diatom, global_feats, one_hot, y)

        out = model.predict_step(batch, 0)

        assert isinstance(out, torch.Tensor)
        assert out.shape == (4,)

    def test_composes_with_transformer_trunk(self):
        model = _model(
            use_heteroscedastic_head=True, use_transformer_trunk=True, add_ccs_features=True
        )
        atom, diatom, global_feats, one_hot = _dummy_batch(charges=[2, 4])

        mu, b = model(atom, diatom, global_feats, one_hot)

        assert torch.all(torch.isfinite(mu))
        assert torch.all(b > 0)

    def test_composes_with_transformer_trunk_and_return_attention(self):
        """The heteroscedastic head's (mu, b) return still gets the attention
        dict appended as a third element, not swallowed - see forward()'s
        two return_attention branches."""
        model = _model(
            use_heteroscedastic_head=True, use_transformer_trunk=True, add_ccs_features=True
        )
        atom, diatom, global_feats, one_hot = _dummy_batch(charges=[2, 4])

        mu, b, attention = model(atom, diatom, global_feats, one_hot, return_attention=True)

        assert torch.all(torch.isfinite(mu))
        assert torch.all(b > 0)
        assert set(attention.keys()) == {"self_attention", "readout_attention"}

    def test_composes_with_sibling_delta_loss(self):
        """y_hat (here, mu) plays the same role for sibling/ranking losses either way."""
        model = _model(use_heteroscedastic_head=True, sibling_loss_weight=1.0)
        mu = torch.tensor([400.0, 410.0])
        y = torch.tensor([405.0, 408.0])
        group_id = torch.tensor([0, 0])

        loss = model._sibling_delta_loss(mu, y, group_id)

        assert torch.isfinite(loss)


class _FakeTrainer:
    """Minimal stand-in so a bare (Trainer-less) LightningModule's
    current_epoch reads a chosen value, without spinning up a real Trainer."""

    def __init__(self, current_epoch):
        self.current_epoch = current_epoch


class TestHeteroscedasticWarmup:
    """Step 8.5: mu warm-start before the Laplace NLL kicks in."""

    def test_disabled_by_default(self):
        model = _model(use_heteroscedastic_head=True)

        assert model._in_heteroscedastic_warmup() is False

    def test_no_op_when_heteroscedastic_head_is_disabled(self):
        model = _model(heteroscedastic_warmup_epochs=5)
        model.trainer = _FakeTrainer(current_epoch=0)

        assert model._in_heteroscedastic_warmup() is False

    def test_in_warmup_before_the_boundary_epoch(self):
        model = _model(use_heteroscedastic_head=True, heteroscedastic_warmup_epochs=3)
        model.trainer = _FakeTrainer(current_epoch=2)

        assert model._in_heteroscedastic_warmup() is True

    def test_not_in_warmup_at_the_boundary_epoch(self):
        model = _model(use_heteroscedastic_head=True, heteroscedastic_warmup_epochs=3)
        model.trainer = _FakeTrainer(current_epoch=3)

        assert model._in_heteroscedastic_warmup() is False

    def test_point_estimate_loss_is_plain_l1_during_warmup(self):
        model = _model(use_heteroscedastic_head=True, heteroscedastic_warmup_epochs=3)
        model.trainer = _FakeTrainer(current_epoch=0)
        mu = torch.tensor([490.0, 495.0])
        y = torch.tensor([500.0, 510.0])
        b = torch.tensor([1.0, 1.0])  # deliberately bad b - must be ignored during warmup

        loss = model._point_estimate_loss(mu, y, b)

        assert torch.allclose(loss, torch.nn.functional.l1_loss(mu, y))

    def test_point_estimate_loss_is_laplace_nll_after_warmup(self):
        model = _model(use_heteroscedastic_head=True, heteroscedastic_warmup_epochs=3)
        model.trainer = _FakeTrainer(current_epoch=3)
        mu = torch.tensor([490.0, 495.0])
        y = torch.tensor([500.0, 510.0])
        b = torch.tensor([10.0, 15.0])

        loss = model._point_estimate_loss(mu, y, b)

        assert torch.allclose(loss, model._laplace_nll(mu, y, b))

    def test_b_receives_no_gradient_during_warmup(self):
        """The whole point: b's output row must not be trained while mu warms up.

        L1_alpha=0 here: the L1 weight-decay-style regularisation term in
        training_step touches every parameter unconditionally (including
        b's row), so a nonzero L1_alpha would give it a small, real but
        unrelated gradient - isolating this to the main loss path only.
        """
        model = _model(use_heteroscedastic_head=True, heteroscedastic_warmup_epochs=3, L1_alpha=0.0)
        model.trainer = _FakeTrainer(current_epoch=0)
        model.log = lambda name, value, **kwargs: None  # avoid needing a real Trainer
        atom, diatom, global_feats, one_hot = _dummy_batch(batch_size=4)
        y = torch.rand(4) * 500 + 300
        batch = (atom, diatom, global_feats, one_hot, y)

        loss = model.training_step(batch, 0)
        loss.backward()

        # Concat[-1] is the final 2-wide Linear layer: row 0 is mu, row 1 is log_b.
        b_row_grad = model.Concat[-1].weight.grad[1]
        assert torch.all(b_row_grad == 0)

    def test_b_does_receive_gradient_after_warmup(self):
        model = _model(use_heteroscedastic_head=True, heteroscedastic_warmup_epochs=3)
        model.trainer = _FakeTrainer(current_epoch=3)
        model.log = lambda name, value, **kwargs: None  # avoid needing a real Trainer
        atom, diatom, global_feats, one_hot = _dummy_batch(batch_size=4)
        y = torch.rand(4) * 500 + 300
        batch = (atom, diatom, global_feats, one_hot, y)

        loss = model.training_step(batch, 0)
        loss.backward()

        b_row_grad = model.Concat[-1].weight.grad[1]
        assert not torch.all(b_row_grad == 0)


class TestHeteroscedasticTransferGuard:
    """A use_heteroscedastic_head=True backbone's Concat is 2-wide, which
    IM2DeepTransfer.forward() has no path for - must raise, not silently
    misinterpret a (batch, 2) tensor as a single CCS value."""

    def test_finetuning_from_heteroscedastic_backbone_raises(self, tmp_path):
        from im2deep._architectures.im2deep_single import IM2DeepTransfer

        model = _model(use_heteroscedastic_head=True)
        ckpt_path = tmp_path / "heteroscedastic.ckpt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hyper_parameters": {"config": model.config},
                "pytorch-lightning_version": "2.0.0",
            },
            ckpt_path,
        )
        transfer_config = {**model.config, "backbone_SD_path": str(ckpt_path)}

        with pytest.raises(ValueError, match="heteroscedastic"):
            IM2DeepTransfer(transfer_config, nn.L1Loss())


class TestTransformerTransfer:
    """Finetuning a use_transformer_trunk backbone: IM2DeepTransfer delegates
    forward() to the backbone, so the trunk, the charge register token and
    both standardisation scalers are all applied."""

    def _transfer(self, tmp_path):
        from im2deep._architectures.im2deep_single import IM2DeepTransfer

        backbone = _model(use_transformer_trunk=True)
        backbone.set_global_feature_scaler(torch.full((60,), 90.0), torch.full((60,), 40.0))
        backbone.set_target_scaler(420.0, 80.0)
        ckpt_path = tmp_path / "transformer.ckpt"
        torch.save(
            {
                "state_dict": backbone.state_dict(),
                "hyper_parameters": {"config": backbone.config},
                "pytorch-lightning_version": "2.0.0",
            },
            ckpt_path,
        )
        transfer_config = {**backbone.config, "backbone_SD_path": str(ckpt_path)}
        return backbone, IM2DeepTransfer(transfer_config, nn.L1Loss())

    def test_output_matches_backbone(self, tmp_path):
        backbone, transfer = self._transfer(tmp_path)
        batch = _dummy_batch(batch_size=4, charges=[1, 2, 3, 4])
        backbone.eval()
        transfer.eval()

        with torch.no_grad():
            assert torch.allclose(transfer(*batch), backbone(*batch))

    def test_scalers_are_the_backbones(self, tmp_path):
        _, transfer = self._transfer(tmp_path)

        assert transfer.backbone.target_mean.item() == pytest.approx(420.0)
        assert transfer.backbone.target_std.item() == pytest.approx(80.0)
        assert torch.all(transfer.backbone.global_feat_mean == 90.0)

    def test_backbone_freeze_covers_the_trunk(self, tmp_path):
        from im2deep._architectures.callbacks import BackboneFreeze

        _, transfer = self._transfer(tmp_path)
        BackboneFreeze(freeze_epochs=1)._set_requires_grad(transfer, False)

        assert not any(p.requires_grad for p in transfer.backbone.transformer_trunk.parameters())
        assert not any(p.requires_grad for p in transfer.backbone.ConvGlobal.parameters())
        assert all(p.requires_grad for p in transfer.backbone.Concat.parameters())


class TestPerChargeHeads:
    """Step 8.6 item 1: one output head per charge, selected per row."""

    def test_disabled_by_default(self):
        model = _model()

        assert model.use_per_charge_heads is False
        assert model.Concat[-1].out_features == 1

    def test_enabled_widens_final_layer_to_six(self):
        model = _model(use_per_charge_heads=True)

        assert model.Concat[-1].out_features == 6

    def test_enabled_with_heteroscedastic_widens_to_twelve(self):
        model = _model(use_per_charge_heads=True, use_heteroscedastic_head=True)

        assert model.Concat[-1].out_features == 12

    def test_requires_add_ccs_features(self):
        with pytest.raises(ValueError, match="add_ccs_features"):
            _model(use_per_charge_heads=True, add_ccs_features=False)

    def test_incompatible_with_mixup(self):
        with pytest.raises(ValueError, match="use_per_charge_heads"):
            _model(use_per_charge_heads=True, mixup_alpha=0.5)

    def test_runs_and_produces_finite_output(self):
        model = _model(use_per_charge_heads=True)
        atom, diatom, global_feats, one_hot = _dummy_batch(batch_size=3, charges=[1, 3, 5])

        out = model(atom, diatom, global_feats, one_hot)

        assert out.shape == (3, 1)
        assert torch.all(torch.isfinite(out))

    def test_selects_the_correct_head_per_charge(self):
        """Zero the final layer's weights so its output is exactly its bias,
        regardless of input - then each charge's selected output is
        predictable exactly, not just "different"."""
        model = _model(use_per_charge_heads=True)
        with torch.no_grad():
            model.Concat[-1].weight.zero_()
            model.Concat[-1].bias.copy_(torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))
        atom, diatom, global_feats, one_hot = _dummy_batch(
            batch_size=6, charges=[1, 2, 3, 4, 5, 6]
        )

        out = model(atom, diatom, global_feats, one_hot).squeeze(1)

        assert torch.allclose(out, torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))

    def test_out_of_range_charge_clamps_to_the_nearest_head(self):
        """CHARGE_FEATURE_INDEX itself is already clamped to [0, 6] upstream
        (forward()'s existing charge_idx line) - charge 0 is the only
        out-of-range value reachable here, and it must map to charge 1's
        head (index 0), not raise or wrap around."""
        model = _model(use_per_charge_heads=True)
        with torch.no_grad():
            model.Concat[-1].weight.zero_()
            model.Concat[-1].bias.copy_(torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))
        atom, diatom, global_feats, one_hot = _dummy_batch(batch_size=1, charges=[0])

        out = model(atom, diatom, global_feats, one_hot).squeeze(1)

        assert torch.allclose(out, torch.tensor([10.0]))

    def test_composes_with_heteroscedastic_head_selection(self):
        """The same per-row head selection must apply to both mu and b, not
        just mu - checked by giving each of the 6 heads a distinct,
        predictable (mu, b) pair via a zeroed weight matrix."""
        model = _model(use_per_charge_heads=True, use_heteroscedastic_head=True)
        with torch.no_grad():
            model.Concat[-1].weight.zero_()
            # 12 = 6 heads x (mu, log_b), interleaved as this model's own
            # concatenated[:, 0::2]=mu, [:, 1::2]=log_b per head.
            bias = torch.zeros(12)
            bias[0::2] = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])  # mu per head
            bias[1::2] = 0.0  # log_b=0 -> b=1 for every head
            model.Concat[-1].bias.copy_(bias)
        atom, diatom, global_feats, one_hot = _dummy_batch(
            batch_size=6, charges=[1, 2, 3, 4, 5, 6]
        )

        mu, b = model(atom, diatom, global_feats, one_hot)

        assert torch.allclose(mu, torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))
        assert torch.allclose(b, torch.ones(6))

    def test_finetuning_from_per_charge_backbone_raises(self, tmp_path):
        from im2deep._architectures.im2deep_single import IM2DeepTransfer

        model = _model(use_per_charge_heads=True)
        ckpt_path = tmp_path / "per_charge.ckpt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hyper_parameters": {"config": model.config},
                "pytorch-lightning_version": "2.0.0",
            },
            ckpt_path,
        )
        transfer_config = {**model.config, "backbone_SD_path": str(ckpt_path)}

        with pytest.raises(ValueError, match="use_per_charge_heads"):
            IM2DeepTransfer(transfer_config, nn.L1Loss())
