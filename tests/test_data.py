"""Tests for the training dataset layer (``im2deep._data``)."""

import numpy as np
import pandas as pd
import pytest
import torch

from im2deep._data import (
    CCSDataset,
    build_training_dataset,
    check_global_features,
    expected_global_features,
    grouped_split,
    stripped_sequence,
)
from im2deep.constants import DEFAULT_TRAINING_CONFIG
from im2deep.exceptions import IM2DeepError


class TestStrippedSequence:
    """Tests for stripping ProForma strings down to bare sequences."""

    @pytest.mark.parametrize(
        "peptidoform,expected",
        [
            ("PEPTIDE/2", "PEPTIDE"),
            ("PEPT[UNIMOD:21]IDEK/2", "PEPTIDEK"),
            ("[UNIMOD:1]-PEPTIDEK/3", "PEPTIDEK"),
            ("PEPTIDEK-[UNIMOD:2]/2", "PEPTIDEK"),
            ("PEPTIDE", "PEPTIDE"),
        ],
    )
    def test_strips_modifications_and_charge(self, peptidoform, expected):
        """Modification tags, terminal mods and the charge suffix are removed."""
        assert stripped_sequence(peptidoform) == expected

    def test_charge_states_share_a_key(self):
        """The same peptide at two charges maps to one grouping key."""
        assert stripped_sequence("PEPTIDE/2") == stripped_sequence("PEPTIDE/3")


class TestExpectedGlobalFeatures:
    """Tests for the global-feature width bookkeeping."""

    @pytest.mark.parametrize(
        "add_ccs,add_terminal,expected",
        [(False, False, 55), (True, False, 60), (False, True, 67), (True, True, 72)],
    )
    def test_matches_deeplc(self, add_ccs, add_terminal, expected):
        """The recorded widths match what DeepLC actually yields."""
        from deeplc._features import encode_peptidoform

        assert expected_global_features(add_ccs, add_terminal) == expected
        features = encode_peptidoform(
            "PEPT[UNIMOD:21]IDEK/2",
            add_ccs_features=add_ccs,
            add_terminal_composition=add_terminal,
        )
        assert features["matrix_global"].shape[0] == expected

    def test_check_passes_for_default_config(self):
        """The shipped training defaults are self-consistent."""
        check_global_features(dict(DEFAULT_TRAINING_CONFIG))

    def test_check_rejects_mismatch(self):
        """A width that disagrees with the featurisation flags is an error."""
        config = {**DEFAULT_TRAINING_CONFIG, "add_terminal_composition": True}
        with pytest.raises(IM2DeepError, match="Global_features"):
            check_global_features(config)

    def test_check_accepts_corrected_mismatch(self):
        """Setting the width to match the flags resolves the error."""
        config = {
            **DEFAULT_TRAINING_CONFIG,
            "add_terminal_composition": True,
            "Global_features": 72,
        }
        check_global_features(config)


class TestBuildTrainingDataset:
    """Tests for building a dataset from tabular input or a PSMList."""

    def test_returns_flat_tuple(self, sample_training_df):
        """Items unpack as the architectures' training_step expects."""
        dataset = build_training_dataset(sample_training_df, dict(DEFAULT_TRAINING_CONFIG))
        item = dataset[0]

        assert isinstance(dataset, CCSDataset)
        assert len(item) == 5
        atom, diatom, global_feats, one_hot, target = item
        assert atom.shape == (60, 6)
        assert diatom.shape == (30, 6)
        assert global_feats.shape == (60,)
        assert one_hot.shape == (60, 20)
        assert float(target) > 0

    def test_batch_unpacks_for_training_step(self, sample_training_df):
        """A default-collated batch is the flat 5-tuple, not a nested one."""
        dataset = build_training_dataset(sample_training_df, dict(DEFAULT_TRAINING_CONFIG))
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        batch = next(iter(loader))

        atom, diatom, global_feats, one_hot, targets = batch
        assert atom.shape == (4, 60, 6)
        assert targets.shape == (4,)

    @pytest.mark.parametrize("column", ["ccs", "CCS"])
    def test_accepts_both_ccs_column_spellings(self, sample_training_df, column):
        """The collated files use lowercase 'ccs'; parse_input uses 'CCS'."""
        df = sample_training_df.rename(columns={"ccs": column})
        dataset = build_training_dataset(df, dict(DEFAULT_TRAINING_CONFIG))
        assert len(dataset) == len(df)

    def test_accepts_peptidoform_column(self):
        """A peptidoform column is used directly when present."""
        df = pd.DataFrame({"peptidoform": ["PEPTIDE/2", "SEQUENCE/3"], "ccs": [400.0, 500.0]})
        dataset = build_training_dataset(df, dict(DEFAULT_TRAINING_CONFIG))
        assert len(dataset) == 2

    def test_accepts_psm_list(self, sample_training_psm_list):
        """A PSMList target is read from metadata['CCS']."""
        dataset = build_training_dataset(sample_training_psm_list, dict(DEFAULT_TRAINING_CONFIG))
        assert len(dataset) == len(sample_training_psm_list)

    def test_reads_from_path(self, sample_training_df, tmp_path):
        """A path to a delimited file is read directly."""
        csv_path = tmp_path / "train.csv"
        sample_training_df.to_csv(csv_path, index=False)
        dataset = build_training_dataset(csv_path, dict(DEFAULT_TRAINING_CONFIG))
        assert len(dataset) == len(sample_training_df)

    def test_drops_over_length_peptides(self):
        """Peptides beyond the featurisation window are dropped, not truncated."""
        df = pd.DataFrame(
            {
                "seq": ["A" * 65, "PEPTIDEK"],
                "modifications": ["", ""],
                "charge": [2, 2],
                "ccs": [900.0, 400.0],
            }
        )
        dataset = build_training_dataset(df, dict(DEFAULT_TRAINING_CONFIG))
        assert len(dataset) == 1

    def test_drops_missing_targets(self, sample_training_df):
        """Rows with a non-finite CCS are dropped."""
        df = sample_training_df.copy()
        df.loc[0, "ccs"] = np.nan
        dataset = build_training_dataset(df, dict(DEFAULT_TRAINING_CONFIG))
        assert len(dataset) == len(df) - 1

    def test_missing_ccs_column_raises(self, sample_training_df):
        """A table with no target column is an error, not a silent NaN target."""
        df = sample_training_df.drop(columns=["ccs"])
        with pytest.raises(IM2DeepError, match="target CCS column"):
            build_training_dataset(df, dict(DEFAULT_TRAINING_CONFIG))

    def test_missing_sequence_columns_raises(self):
        """A table with a target but no sequence is an error."""
        df = pd.DataFrame({"ccs": [400.0]})
        with pytest.raises(IM2DeepError, match="peptidoform"):
            build_training_dataset(df, dict(DEFAULT_TRAINING_CONFIG))

    def test_terminal_composition_widens_global_features(self, sample_training_df):
        """The terminal-composition variant yields 72 global features."""
        config = {
            **DEFAULT_TRAINING_CONFIG,
            "add_terminal_composition": True,
            "Global_features": 72,
        }
        dataset = build_training_dataset(sample_training_df, config)
        assert dataset[0][2].shape == (72,)


class TestGroupedSplit:
    """Tests for the leakage-free train/validation split."""

    def test_no_sequence_in_both_halves(self, sample_training_df):
        """A peptide at two charge states cannot straddle the split."""
        dataset = build_training_dataset(sample_training_df, dict(DEFAULT_TRAINING_CONFIG))
        train_subset, validation_subset = grouped_split(dataset, 0.2, seed=0)

        train_sequences = {
            stripped_sequence(dataset.peptidoforms[i]) for i in train_subset.indices
        }
        validation_sequences = {
            stripped_sequence(dataset.peptidoforms[i]) for i in validation_subset.indices
        }
        assert not train_sequences & validation_sequences

    def test_covers_every_row_exactly_once(self, sample_training_df):
        """The split is a partition, not a sample."""
        dataset = build_training_dataset(sample_training_df, dict(DEFAULT_TRAINING_CONFIG))
        train_subset, validation_subset = grouped_split(dataset, 0.2, seed=0)

        combined = sorted(train_subset.indices + validation_subset.indices)
        assert combined == list(range(len(dataset)))

    def test_is_deterministic(self, sample_training_df):
        """The same seed gives the same split."""
        dataset = build_training_dataset(sample_training_df, dict(DEFAULT_TRAINING_CONFIG))
        first, _ = grouped_split(dataset, 0.2, seed=7)
        second, _ = grouped_split(dataset, 0.2, seed=7)
        assert first.indices == second.indices

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_invalid_fraction(self, sample_training_df, fraction):
        """A split fraction outside (0, 1) is rejected."""
        dataset = build_training_dataset(sample_training_df, dict(DEFAULT_TRAINING_CONFIG))
        with pytest.raises(ValueError, match="validation_split"):
            grouped_split(dataset, fraction)

    def test_rejects_split_that_empties_a_half(self):
        """A dataset dominated by one sequence cannot be split."""
        df = pd.DataFrame(
            {
                "seq": ["PEPTIDEK", "PEPTIDEK"],
                "modifications": ["", ""],
                "charge": [2, 3],
                "ccs": [400.0, 500.0],
            }
        )
        dataset = build_training_dataset(df, dict(DEFAULT_TRAINING_CONFIG))
        with pytest.raises(IM2DeepError, match="at least one of"):
            grouped_split(dataset, 0.5)
