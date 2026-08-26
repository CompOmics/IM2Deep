"""
Dataset construction for IM2Deep training.

Inference builds its dataset with :meth:`deeplc.data.DeepLCDataset.from_psm_list`
(see :func:`im2deep.core.predict`), but training cannot reuse that path for two
reasons:

* ``DeepLCDataset.__getitem__`` returns a nested ``((atom, diatom, global,
  one_hot), target)``, while the architectures' ``training_step`` unpacks a flat
  ``atom_comp, diatom_comp, global_feats, one_hot, y``. Inference gets away with
  the nested form because :func:`im2deep._model_ops._predict_loop` unpacks it
  itself.
* ``from_psm_list`` reads its target from ``psm_list["retention_time"]`` and
  discards all targets if a single PSM lacks one, whereas IM2Deep carries the
  target CCS in ``psm.metadata["CCS"]``.

:class:`CCSDataset` fixes the first and :func:`build_training_dataset` the
second. Both keep the featurisation itself in DeepLC's hands, so a model trained
here sees exactly what ``predict()`` will feed it later.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
import psm_utils.io
from deeplc.data import DeepLCDataset
from psm_utils.psm_list import PSMList
from torch.utils.data import Subset

from im2deep._io_helpers import validate_psm_list
from im2deep.constants import GLOBAL_FEATURE_COUNTS
from im2deep.exceptions import IM2DeepError

LOGGER = logging.getLogger(__name__)

#: Column names accepted as the target CCS, in order of preference. The
#: collated training sets produced by the dataset pipeline use lowercase
#: ``ccs``; ``parse_input`` only recognises the uppercase form.
CCS_COLUMNS = ("CCS", "ccs")


class CCSDataset(DeepLCDataset):
    """
    A :class:`deeplc.data.DeepLCDataset` that yields a flat feature tuple.

    ``DeepLCDataset`` returns ``((atom, diatom, global, one_hot), target)``,
    which cannot be unpacked by the architectures' ``training_step``. This
    subclass flattens it to ``(atom, diatom, global, one_hot, target)``.

    Subclassing rather than passing a ``collate_fn`` keeps
    :class:`torch.utils.data.Subset` and PyTorch's default collation working
    unchanged, and leaves the nested inference path untouched.
    """

    def __getitem__(self, idx: int) -> tuple:  # type: ignore[override]
        """Return ``(atom_comp, diatom_comp, global_feats, one_hot, target)``."""
        features, target = super().__getitem__(idx)
        return (*features, target)


def expected_global_features(
    add_ccs_features: bool = True, add_terminal_composition: bool = False
) -> int:
    """
    Number of global features DeepLC yields for a featurisation flag pair.

    Parameters
    ----------
    add_ccs_features
        Whether the five CCS-specific global features are included.
    add_terminal_composition
        Whether the twelve terminal-composition values are included.

    Returns
    -------
    int
        Length of DeepLC's ``matrix_global`` for those flags.

    """
    return GLOBAL_FEATURE_COUNTS[(bool(add_ccs_features), bool(add_terminal_composition))]


def check_global_features(config: dict) -> None:
    """
    Check that ``Global_features`` matches the configured featurisation.

    The architectures' global branch input width is fixed at construction, so a
    mismatch with what :class:`CCSDataset` actually yields would otherwise
    surface as an opaque shape error inside the first dense layer.

    Parameters
    ----------
    config
        Training configuration.

    Raises
    ------
    IM2DeepError
        If ``config["Global_features"]`` does not match the featurisation flags.

    """
    expected = expected_global_features(
        config.get("add_ccs_features", True),
        config.get("add_terminal_composition", False),
    )
    configured = config.get("Global_features", expected)
    if configured != expected:
        raise IM2DeepError(
            f"Global_features is {configured}, but add_ccs_features="
            f"{config.get('add_ccs_features', True)} and add_terminal_composition="
            f"{config.get('add_terminal_composition', False)} make DeepLC yield "
            f"{expected} global features. Set Global_features to {expected}, or "
            "change the featurisation flags to match."
        )


def _featurisation_kwargs(config: dict) -> dict:
    """Extract the DeepLC featurisation keyword arguments from a config."""
    return {
        "add_ccs_features": config.get("add_ccs_features", True),
        "add_terminal_composition": config.get("add_terminal_composition", False),
        "padding_length": config.get("padding_length", 60),
        "legacy_positional_deltas": config.get("legacy_positional_deltas", True),
    }


def _from_psm_list(psm_list: PSMList) -> tuple[list[str], np.ndarray]:
    """ProForma strings and target CCS values from a PSMList."""
    # validate_psm_list drops charge None/>6 and normalises ion_mobility into
    # metadata["CCS"], so afterwards metadata["CCS"] is the single target source.
    psm_list = validate_psm_list(psm_list, needs_target=True)
    peptidoforms = [str(psm.peptidoform) for psm in psm_list]
    targets = np.array([float((psm.metadata or {})["CCS"]) for psm in psm_list], dtype=np.float32)
    return peptidoforms, targets


def _from_dataframe(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """
    ProForma strings and target CCS values from a DataFrame.

    Deliberately does not go through :func:`im2deep._io_helpers.parse_input`:
    that builds a :class:`~psm_utils.psm.PSM` per row, which costs roughly 36x
    the memory of the equivalent ProForma strings and is the difference between
    a multi-million-row training set fitting in memory comfortably or not.
    """
    ccs_column = next((c for c in CCS_COLUMNS if c in df.columns), None)
    if ccs_column is None:
        raise IM2DeepError(
            f"No target CCS column found; expected one of {list(CCS_COLUMNS)}, "
            f"got {list(df.columns)}."
        )

    if "peptidoform" in df.columns:
        peptidoform_strings = df["peptidoform"].astype(str).tolist()
    elif all(col in df.columns for col in ("seq", "modifications", "charge")):
        modifications = df["modifications"].fillna("").astype(str)
        peptidoform_strings = [
            psm_utils.io.peptide_record.peprec_to_proforma(
                peptide=seq, modifications=mods, charge=int(charge)
            ).proforma
            for seq, mods, charge in zip(df["seq"], modifications, df["charge"], strict=True)
        ]
    else:
        raise IM2DeepError(
            "Expected either a 'peptidoform' column or 'seq', 'modifications' "
            f"and 'charge' columns, got {list(df.columns)}."
        )

    targets = df[ccs_column].to_numpy(dtype=np.float32)
    return peptidoform_strings, targets


def _drop_long_and_missing(
    peptidoforms: list[str], targets: np.ndarray, padding_length: int
) -> tuple[list[str], np.ndarray]:
    """
    Drop peptides longer than ``padding_length`` and rows with no target.

    DeepLC silently truncates over-length peptides to the first
    ``padding_length`` residues and then emits one warning per token beyond it,
    so on a large training set these rows cost both correctness and a very large
    warning stream. Dropping them is both cheaper and more honest.
    """
    lengths = np.array([_stripped_length(pf) for pf in peptidoforms])
    keep = (lengths <= padding_length) & np.isfinite(targets)

    n_long = int((lengths > padding_length).sum())
    if n_long:
        LOGGER.warning(
            f"Dropped {n_long} peptide(s) longer than the {padding_length}-residue "
            "featurisation window; DeepLC would otherwise truncate them silently."
        )
    n_missing = int((~np.isfinite(targets)).sum())
    if n_missing:
        LOGGER.warning(f"Dropped {n_missing} row(s) with a missing or non-finite CCS.")

    if not keep.any():
        raise IM2DeepError("No usable training rows remain after filtering.")

    if keep.all():
        return peptidoforms, targets
    return [pf for pf, k in zip(peptidoforms, keep, strict=True) if k], targets[keep]


def _stripped_length(peptidoform: str) -> int:
    """
    Residue count of a ProForma string, ignoring modifications and charge.

    Counting upper-case residue letters outside bracketed modification tags is
    enough here and avoids parsing several million peptidoforms just to measure
    them.
    """
    length = 0
    depth = 0
    for char in peptidoform:
        if char in "[(":
            depth += 1
        elif char in "])":
            depth = max(0, depth - 1)
        elif char == "/" and depth == 0:
            break
        elif depth == 0 and char.isalpha() and char.isupper():
            length += 1
    return length


def stripped_sequence(peptidoform: str) -> str:
    """
    The bare amino-acid sequence of a ProForma string.

    Strips modification tags, the charge suffix and any terminal-modification
    prefix, so that ``PEPT[UNIMOD:21]IDEK/2`` and ``PEPTIDEK/3`` map to the same
    key. Used to keep a peptide out of both halves of a train/validation split.
    """
    residues = []
    depth = 0
    for char in peptidoform:
        if char in "[(":
            depth += 1
        elif char in "])":
            depth = max(0, depth - 1)
        elif char == "/" and depth == 0:
            break
        elif depth == 0 and char.isalpha() and char.isupper():
            residues.append(char)
    return "".join(residues)


def build_training_dataset(
    source: PSMList | pd.DataFrame | PathLike | str,
    config: dict,
) -> CCSDataset:
    """
    Build a :class:`CCSDataset` from a PSMList, DataFrame or delimited file.

    Parameters
    ----------
    source
        A :class:`~psm_utils.psm_list.PSMList` with a target CCS on every PSM
        (either ``psm.ion_mobility`` or ``psm.metadata["CCS"]``), a
        :class:`pandas.DataFrame`, or a path to a delimited file. Tabular input
        needs a target column (``CCS`` or ``ccs``) plus either a ``peptidoform``
        column or ``seq``/``modifications``/``charge`` columns.
    config
        Training configuration; the featurisation keys (``add_ccs_features``,
        ``add_terminal_composition``, ``padding_length``,
        ``legacy_positional_deltas``) and ``Global_features`` are read from it.

    Returns
    -------
    CCSDataset
        Dataset yielding ``(atom, diatom, global, one_hot, target)`` per item.

    """
    check_global_features(config)

    if isinstance(source, PSMList):
        peptidoforms, targets = _from_psm_list(source)
    else:
        if isinstance(source, pd.DataFrame):
            df = source
        elif isinstance(source, (str, Path, PathLike)):
            LOGGER.info(f"Reading training data from {source}")
            df = pd.read_csv(source, sep=None, engine="python")
        else:
            raise TypeError(
                f"source must be a PSMList, DataFrame or path, got {type(source).__name__}."
            )
        peptidoforms, targets = _from_dataframe(df)

    featurisation = _featurisation_kwargs(config)
    peptidoforms, targets = _drop_long_and_missing(
        peptidoforms, targets, featurisation["padding_length"]
    )
    LOGGER.info(f"Built training dataset with {len(peptidoforms):,} precursors.")

    # Peptidoforms are passed as ProForma strings rather than Peptidoform
    # objects: DeepLC re-parses them per item, which costs throughput but keeps
    # resident memory roughly 36x lower and avoids each DataLoader worker
    # forking a large object graph.
    return CCSDataset(
        peptidoforms=peptidoforms,  # type: ignore[arg-type]
        target_retention_times=targets,
        **featurisation,
    )


def grouped_split(
    dataset: CCSDataset,
    validation_split: float,
    seed: int = 0,
) -> tuple[Subset, Subset]:
    """
    Split a dataset into train and validation, grouped by stripped sequence.

    A peptide observed at two charge states or in two modification states must
    not land in both halves, or the validation loss reports memorisation as
    generalisation. :func:`deeplc.data.split_datasets` uses a plain
    :func:`~torch.utils.data.random_split` and carries a TODO for exactly this;
    this is that grouped version.

    Parameters
    ----------
    dataset
        Dataset to split.
    validation_split
        Approximate fraction of precursors to hold out. The realised fraction
        differs slightly because whole sequence groups move together.
    seed
        Seed for the group shuffle.

    Returns
    -------
    tuple of Subset
        ``(train_subset, validation_subset)``.

    """
    if not 0 < validation_split < 1:
        raise ValueError(f"validation_split must be in (0, 1), got {validation_split}.")

    peptidoforms = _peptidoform_strings(dataset)
    groups: dict[str, list[int]] = {}
    for idx, peptidoform in enumerate(peptidoforms):
        groups.setdefault(stripped_sequence(peptidoform), []).append(idx)

    group_keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(group_keys)  # type: ignore[arg-type]

    target_validation = validation_split * len(peptidoforms)
    validation_indices: list[int] = []
    for key in group_keys:
        if len(validation_indices) >= target_validation:
            break
        validation_indices.extend(groups[key])

    validation_set = set(validation_indices)
    train_indices = [idx for idx in range(len(peptidoforms)) if idx not in validation_set]

    if not train_indices or not validation_indices:
        raise IM2DeepError(
            f"validation_split={validation_split} left {len(train_indices)} training "
            f"and {len(validation_indices)} validation precursors; at least one of "
            "each is required. The dataset may be too small or dominated by a "
            "single sequence."
        )

    LOGGER.info(
        f"Split {len(peptidoforms):,} precursors into {len(train_indices):,} train "
        f"and {len(validation_indices):,} validation "
        f"({len(validation_indices) / len(peptidoforms):.1%}), grouped over "
        f"{len(group_keys):,} stripped sequences."
    )
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


def _peptidoform_strings(dataset: CCSDataset) -> Sequence[str]:
    """Peptidoform strings of a dataset, as strings whatever was passed in."""
    return [
        peptidoform if isinstance(peptidoform, str) else str(peptidoform)
        for peptidoform in dataset.peptidoforms
    ]
