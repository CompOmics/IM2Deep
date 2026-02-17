"""Data extraction module for the im2deep training functionality."""

from typing import Dict, Tuple
from deeplcretrainer.cnn_functions import get_feat_df, get_feat_matrix
import os
import random
import pandas as pd
import pickle
import logging
import numpy as np
from psm_utils.io.peptide_record import peprec_to_proforma
from psm_utils import PSM, PSMList
from pyteomics import proforma

logger = logging.getLogger(__name__)
random.seed(42)

MOL_FEATS = pd.read_csv(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "training_package_data", "aa_mol_desc_feats.csv"
    )
)


def _train_test_split(ccs_df: pd.DataFrame, test_split=0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into a training and testing set.

    Args:
        ccs_df (pd.DataFrame): The dataframe containing the data.
        test_split (float, optional): The fraction of the data to use for testing. Defaults to 0.1.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and testing dataframes.
    """
    # Get all the indices of the dataframe
    indices = list(ccs_df.index)
    # Shuffle the indices
    random.shuffle(indices)
    # Split the indices in a training and testing set
    train_idx = indices[0 : int(len(indices) * (1 - test_split))]
    test_idx = indices[int(len(indices) * (1 - test_split)) :]

    # Get the train and test indices and point to new variables
    ccs_df_train = ccs_df.loc[train_idx, :]
    ccs_df_test = ccs_df.loc[test_idx, :]
    return ccs_df_train, ccs_df_test


def _aa_chemical_features(features: pd.DataFrame, mask=None) -> Dict:
    """Get the chemical features for the amino acids.

    Args:
        features (pd.DataFrame): The dataframe containing the features.
        mask (list, optional): The mask to apply to the features. Defaults to None.

    Returns:
        dict: The dictionary containing the amino acids.
    """
    aa_features = features.iloc[:20]
    if mask:
        aa_features[aa_features.columns[[mask]]] = 0
    amino_acids = aa_features.set_index("Elements").T.to_dict("list")
    features_arrays = {
        aa: np.array(aa_features, dtype=np.float32) for aa, aa_features in amino_acids.items()
    }
    return features_arrays


def _mod_chemical_features(features: pd.DataFrame, mask=None) -> Dict:
    """Get the chemical features for the modifications.

    Args:
        features (pd.DataFrame): The dataframe containing the features.
        mask (list, optional): The mask to apply to the features. Defaults to None.

    Returns:
        dict: The dictionary containing the modifications."""

    mod_features = features.iloc[20:]
    if mask:
        mod_features[mod_features.columns[mask]] = 0
    mod_features = mod_features.set_index("Elements").T
    modified = mod_features.to_dict("list")
    dic = {}
    for key in modified:
        main_key, sub_key = key.split("#")
        dic.setdefault(main_key, {})[sub_key] = dict(zip(mod_features.index, modified[key]))
    return dic


def _empty_array():
    """Create an empty array to store the encoded sequence."""
    return np.zeros((13, 60), dtype=np.float32)


def _peptide_parser(peptide: str) -> Tuple[list, dict, str, str]:
    """Parse the peptide sequence and modifications.

    Args:
        peptide (str): The peptide sequence.

    Returns:
        parsed_sequence (list): The parsed sequence.
        modifiers (dict): Proforma modifiers.
        sequence (str): The sequence.
        modifications (str): The modifications.
    """
    modifications = []
    parsed_sequence, modifiers = proforma.parse(peptide)

    sequence = "".join([aa for aa, _ in parsed_sequence])
    for loc, (_, mods) in enumerate(parsed_sequence):
        if mods:
            modifications.append(":".join([str(loc + 1), mods[0].name]))
    modifications = "|".join(modifications)
    return parsed_sequence, modifiers, sequence, modifications


def encode_sequence_and_modification(
    sequence: str,
    parsed_sequence: list,
    modifications_dict: Dict,
    aa_to_feature: Dict,
    n_term=None,
) -> np.ndarray:
    """Encode the sequence and modifications using the given dictionaries.

    Args:
        sequence (str): The sequence to encode.
        parsed_sequence (list): The parsed sequence.
        modifications_dict (Dict): The dictionary containing the modifications.
        aa_to_feature (Dict): The dictionary containing the amino acid features.
        n_term ([type], optional): The N-terminal modifications. Defaults to None.

    Returns:
        np.ndarray: The encoded sequence.
    """

    encoded = _empty_array()

    for i, aa in enumerate(sequence):
        encoded[:, i] = aa_to_feature[aa]

    for loc, (aa, mods) in enumerate(parsed_sequence):
        if mods:
            for mod in mods:
                name = mod.name
                encoded[:, loc] = list(modifications_dict[name][aa].values())

    if n_term:
        for mod in n_term:
            name = mod.name
            name = name + "(N-T)"
            encoded[:, 0] = list(modifications_dict[name][sequence[0]].values())

    return encoded


def _get_mol_matrix(psmlist: PSMList, features: pd.DataFrame = MOL_FEATS) -> np.ndarray:
    """Get the molecular features for the given PSMList.

    Args:
        psmlist (PSMList): The PSMList containing the PSMs.
        features (pd.DataFrame, optional): The dataframe containing the molecular features. Defaults to MOL_FEATS.

    Returns:
        np.ndarray: The molecular features.
    """
    mol_feats = []

    aa_to_feature = _aa_chemical_features(features)
    mod_dict = _mod_chemical_features(features)

    for psm in psmlist:
        parsed_sequence, modifiers, sequence, modifications = _peptide_parser(
            psm.peptidoform.proforma
        )

        encoded = encode_sequence_and_modification(
            sequence, parsed_sequence, mod_dict, aa_to_feature, n_term=modifiers.get("n_term")
        )

        mol_feats.append(encoded)

    return np.array(mol_feats)


def _get_matrices(
    psm_list: PSMList,
    split_name: str = "test",
    add_X_mol: bool = False,
    multi_output: bool = False,
    inference: bool = False,
) -> Dict:
    """Get the feature matrices for the given PSMList.

    Args:
        psm_list (PSMList): The PSMList containing the PSMs.
        split_name (str, optional): The name of the split. Defaults to "test".
        add_X_mol (bool, optional): Whether to add the molecular features to the data. Defaults to False.

    Returns:
        Dict: A dictionary containing the feature matrices.
    """

    feat_df = get_feat_df(psm_list=psm_list, predict_ccs=True)

    if not inference:
        if multi_output:
            logger.debug("Multi-output")
            y1 = []
            y2 = []
            for psm in psm_list:
                y1.append(float(psm.metadata["CCS1"]))
                y2.append(float(psm.metadata["CCS2"]))

    feat_df["tr"] = 0

    X, X_sum, X_global, X_hc, y = get_feat_matrix(feat_df)

    if not inference:
        if multi_output:
            y = np.array(list(zip(y1, y2)))
        else:
            y = np.array([float(psm.metadata["CCS"]) for psm in psm_list])

        data = {
            f"X_{split_name}_AtomEnc": X,
            f"X_{split_name}_DiAminoAtomEnc": X_sum,
            f"X_{split_name}_GlobalFeatures": X_global,
            f"X_{split_name}_OneHot": X_hc,
            f"y_{split_name}": y,
        }

        if add_X_mol:
            X_mol = _get_mol_matrix(psm_list)
            data[f"X_{split_name}_MolEnc"] = X_mol

    else:
        data = {
            f"X_{split_name}_AtomEnc": X,
            f"X_{split_name}_DiAminoAtomEnc": X_sum,
            f"X_{split_name}_GlobalFeatures": X_global,
            f"X_{split_name}_OneHot": X_hc,
        }

        if add_X_mol:
            X_mol = _get_mol_matrix(psm_list)
            data[f"X_{split_name}_MolEnc"] = X_mol

    return data


def data_extraction(config: Dict) -> Tuple[Dict, pd.DataFrame]:
    """Extract data from the given paths and return the data in the correct format for training.

    Args:
        config (Dict): The configuration dictionary.

    Returns:
        Tuple[Dict, pd.DataFrame]: A tuple containing the data in the correct format for training and the test dataframe.
    """

    try:
        data = pd.read_csv(config["data_path"])
    except UnicodeDecodeError:
        data = pd.read_pickle(config["data_path"])
    logger.debug(len(data))

    if config["remove_charge_dupes"]:
        data = data.drop_duplicates(subset=["seq", "modifications"], keep="first")
        logger.debug(len(data))

    try:
        ccs_df_test = pd.read_csv(config["test_data_path"])
        ccs_df_train = data
    except UnicodeDecodeError:
        ccs_df_test = pd.read_pickle(config["test_data_path"])
        ccs_df_train = data
    except KeyError:
        ccs_df_train, ccs_df_test = _train_test_split(data, config["test_split"])

    ccs_df_train, ccs_df_valid = _train_test_split(ccs_df_train, config["val_split"])

    if config["save_dfs"]:
        ccs_df_train.to_pickle(
            f"{config['output_path']}/train_data_{config['model_params']['model_name']}.pkl"
        )
        ccs_df_valid.to_pickle(
            f"{config['output_path']}/valid_data_{config['model_params']['model_name']}.pkl"
        )
        ccs_df_test.to_pickle(
            f"{config['output_path']}/test_data_{config['model_params']['model_name']}.pkl"
        )

    logger.debug(
        f"Train: {ccs_df_train.shape}, Valid: {ccs_df_valid.shape}, Test: {ccs_df_test.shape}"
    )

    train_psm = []

    if config["model_params"].get("multi-output", False):
        try:
            for seq, mod, charge, ccs, ident in zip(
                ccs_df_train["seq"],
                ccs_df_train["modifications"],
                ccs_df_train["charge"],
                ccs_df_train["CCS"],
                ccs_df_train.index,
            ):
                train_psm.append(
                    PSM(
                        peptidoform=peprec_to_proforma(seq, mod, charge),
                        spectrum_id=ident,
                        metadata={"CCS1": str(ccs[0]), "CCS2": str(ccs[1])},
                    )
                )
        except KeyError:
            for proforma, ccs, ident in zip(
                ccs_df_train["proforma"], ccs_df_train["Conformer_CCS_list"], ccs_df_train.index
            ):
                train_psm.append(
                    PSM(
                        peptidoform=proforma,
                        spectrum_id=ident,
                        metadata={"CCS1": str(ccs[0]), "CCS2": str(ccs[1])},
                    )
                )
    else:
        try:
            for seq, mod, charge, ccs, ident in zip(
                ccs_df_train["seq"],
                ccs_df_train["modifications"],
                ccs_df_train["charge"],
                ccs_df_train["CCS"],
                ccs_df_train.index,
            ):
                train_psm.append(
                    PSM(
                        peptidoform=peprec_to_proforma(seq, mod, charge),
                        spectrum_id=ident,
                        metadata={"CCS": str(ccs)},
                    )
                )
        except KeyError:
            for proforma, ccs, ident in zip(
                ccs_df_train["proforma"], ccs_df_train["CCS"], ccs_df_train.index
            ):
                train_psm.append(
                    PSM(
                        peptidoform=proforma,
                        spectrum_id=ident,
                        metadata={"CCS": str(ccs)},
                    )
                )

    train_psmlist = PSMList(psm_list=train_psm)

    valid_psm = []

    if config["model_params"].get("multi-output"):
        try:
            for seq, mod, charge, ccs, ident in zip(
                ccs_df_valid["seq"],
                ccs_df_valid["modifications"],
                ccs_df_valid["charge"],
                ccs_df_valid["CCS"],
                ccs_df_valid.index,
            ):
                valid_psm.append(
                    PSM(
                        peptidoform=peprec_to_proforma(seq, mod, charge),
                        spectrum_id=ident,
                        metadata={"CCS1": str(ccs[0]), "CCS2": str(ccs[1])},
                    )
                )
        except KeyError:
            for proforma, ccs, ident in zip(
                ccs_df_valid["proforma"], ccs_df_valid["Conformer_CCS_list"], ccs_df_valid.index
            ):
                valid_psm.append(
                    PSM(
                        peptidoform=proforma,
                        spectrum_id=ident,
                        metadata={"CCS1": str(ccs[0]), "CCS2": str(ccs[1])},
                    )
                )
    else:
        try:
            for seq, mod, charge, ccs, ident in zip(
                ccs_df_valid["seq"],
                ccs_df_valid["modifications"],
                ccs_df_valid["charge"],
                ccs_df_valid["CCS"],
                ccs_df_valid.index,
            ):
                valid_psm.append(
                    PSM(
                        peptidoform=peprec_to_proforma(seq, mod, charge),
                        spectrum_id=ident,
                        metadata={"CCS": str(ccs)},
                    )
                )
        except KeyError:
            for proforma, ccs, ident in zip(
                ccs_df_valid["proforma"], ccs_df_valid["CCS"], ccs_df_valid.index
            ):
                valid_psm.append(
                    PSM(
                        peptidoform=proforma,
                        spectrum_id=ident,
                        metadata={"CCS": str(ccs)},
                    )
                )

    valid_psmlist = PSMList(psm_list=valid_psm)

    test_psm = []
    if config["model_params"].get("multi-output"):
        try:
            for seq, mod, charge, ccs, ident in zip(
                ccs_df_test["seq"],
                ccs_df_test["modifications"],
                ccs_df_test["charge"],
                ccs_df_test["CCS"],
                ccs_df_test.index,
            ):
                test_psm.append(
                    PSM(
                        peptidoform=peprec_to_proforma(seq, mod, charge),
                        spectrum_id=ident,
                        metadata={"CCS1": str(ccs[0]), "CCS2": str(ccs[1])},
                    )
                )
        except KeyError:
            for proforma, ccs, ident in zip(
                ccs_df_test["proforma"], ccs_df_test["Conformer_CCS_list"], ccs_df_test.index
            ):
                test_psm.append(
                    PSM(
                        peptidoform=proforma,
                        spectrum_id=ident,
                        metadata={"CCS1": str(ccs[0]), "CCS2": str(ccs[1])},
                    )
                )
    else:
        try:
            for seq, mod, charge, ccs, ident in zip(
                ccs_df_test["seq"],
                ccs_df_test["modifications"],
                ccs_df_test["charge"],
                ccs_df_test["CCS"],
                ccs_df_test.index,
            ):
                test_psm.append(
                    PSM(
                        peptidoform=peprec_to_proforma(seq, mod, charge),
                        spectrum_id=ident,
                        metadata={"CCS": str(ccs)},
                    )
                )
        except KeyError:
            for proforma, ccs, ident in zip(
                ccs_df_test["proforma"], ccs_df_test["CCS"], ccs_df_test.index
            ):
                test_psm.append(
                    PSM(
                        peptidoform=proforma,
                        spectrum_id=ident,
                        metadata={"CCS": str(ccs)},
                    )
                )
    test_psmlist = PSMList(psm_list=test_psm)

    train_data = _get_matrices(
        train_psmlist,
        "train",
        add_X_mol=config["model_params"]["add_X_mol"],
        multi_output=config["model_params"].get("multi-output", False),
    )

    valid_data = _get_matrices(
        valid_psmlist,
        "valid",
        add_X_mol=config["model_params"]["add_X_mol"],
        multi_output=config["model_params"].get("multi-output", False),
    )

    test_data = _get_matrices(
        test_psmlist,
        "test",
        add_X_mol=config["model_params"]["add_X_mol"],
        multi_output=config["model_params"].get("multi-output", False),
    )

    if config["save_data_tensors"]:
        for split, data_dict in zip(
            ["train", "valid", "test"], [train_data, valid_data, test_data]
        ):
            for key, value in data_dict.items():
                pickle.dump(value, open(f"{config['output_path']}/{split}_{key}.pkl", "wb"))

    return {"train": train_data, "valid": valid_data, "test": test_data}, ccs_df_test
