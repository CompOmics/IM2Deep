"""
Tests pinning the bundled single-conformer model to its calibration reference.

IM2Deep's per-charge shift is ``observed_CCS - reference_CCS`` over peptidoforms
shared with the reference, so the reference only calibrates a model correctly if
it sits on that model's own CCS scale. The two files are replaced together; these
tests fail if one is swapped without the other.
"""

import numpy as np
import pytest
from psm_utils import PSM, Peptidoform, PSMList

from im2deep import core
from im2deep._model_ops import read_checkpoint_config
from im2deep.calibration import get_default_reference
from im2deep.constants import DEFAULT_MODEL

#: Charges with enough reference rows to sample from in a unit test.
CHARGES = (1, 2, 3, 4)


def _psm_list(peptidoforms, ccs=None):
    psms = [
        PSM(
            peptidoform=Peptidoform(peptidoform),
            spectrum_id=f"spectrum_{i}",
            run="run",
            metadata={} if ccs is None else {"CCS": float(ccs[i])},
        )
        for i, peptidoform in enumerate(peptidoforms)
    ]
    return PSMList(psm_list=psms)


@pytest.fixture(scope="module")
def reference():
    return get_default_reference()


class TestBundledModelConfig:
    """The checkpoint records the architecture and featurisation it was trained with."""

    def test_records_transformer_architecture_and_featurisation(self):
        config = read_checkpoint_config(DEFAULT_MODEL)

        assert config["use_transformer_trunk"] is True
        assert config["standardize_target"] is True
        assert config["standardize_global_features"] is True
        assert config["add_ccs_features"] is True
        assert config["add_terminal_composition"] is True
        assert config["legacy_positional_deltas"] is False
        assert config["Global_features"] == 72

    def test_carries_no_training_run_state(self):
        """``finetune`` copies the backbone's config, so it must not hold run-specific keys."""
        config = read_checkpoint_config(DEFAULT_MODEL)

        assert config["wandb"]["enabled"] is False
        assert config["num_workers"] == 0


class TestDefaultReference:
    """The bundled reference is well formed and on the bundled model's scale."""

    def test_format(self, reference):
        assert list(reference.columns) == ["peptidoform", "CCS"]
        assert reference["peptidoform"].str.contains(r"/[1-6]$").all()
        assert not reference["peptidoform"].duplicated().any()
        assert (reference["CCS"] > 0).all()

    def test_covers_every_charge_and_modified_peptides(self, reference):
        charges = reference["peptidoform"].str.rsplit("/", n=1).str[1].astype(int)

        assert set(charges) == {1, 2, 3, 4, 5, 6}
        assert reference["peptidoform"].str.contains(r"\[UNIMOD:").any()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_sits_on_the_default_models_scale(self, reference):
        """
        The model's raw predictions agree with the reference per charge.

        Calibrating against the previous reference, which sat ~3.7% below this
        model's scale (+14 to +25 A^2 by charge), would fail here.
        """
        charges = reference["peptidoform"].str.rsplit("/", n=1).str[1].astype(int)
        for charge in CHARGES:
            sample = reference[charges == charge].sample(150, random_state=0)
            predicted = core.predict(_psm_list(sample["peptidoform"]))

            offset = float(np.mean(sample["CCS"].to_numpy() - predicted))
            assert abs(offset) < 4.0, f"z={charge}: reference - prediction = {offset:.2f} A^2"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_default_calibration_recovers_a_known_shift(self, reference):
        """
        Calibration against the default reference matches modified and unmodified
        peptides alike: anchors observed 10 A^2 above the reference shift the
        predictions by exactly 10 A^2 in every charge.
        """
        charges = reference["peptidoform"].str.rsplit("/", n=1).str[1].astype(int)
        anchors = np.concatenate(
            [
                reference[charges == charge].sample(60, random_state=1).index.to_numpy()
                for charge in CHARGES
            ]
        )
        calibration_rows = reference.loc[anchors]
        psm_list_cal = _psm_list(
            calibration_rows["peptidoform"], ccs=calibration_rows["CCS"].to_numpy() + 10.0
        )
        # Peptides to predict on: other rows of the reference, same charges.
        eval_rows = reference[charges.isin(CHARGES) & ~reference.index.isin(anchors)].sample(
            40, random_state=2
        )
        psm_list = _psm_list(eval_rows["peptidoform"])

        calibrated = np.asarray(
            core.predict_and_calibrate(psm_list, psm_list_cal=psm_list_cal), dtype=float
        )
        uncalibrated = np.asarray(core.predict(psm_list), dtype=float)

        assert np.allclose(calibrated - uncalibrated, 10.0, atol=1e-2)
