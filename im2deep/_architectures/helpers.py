"""Helper functions for IM2Deep architecture and model operations."""

import logging

LOGGER = logging.getLogger(__name__)


def calculate_concat_shape(config):
    atom_comp_out_shape = (60 // (2 * config["AtomComp_MaxPool_kernel_size"])) * (
        config["AtomComp_out_channels_start"] // 4
    )
    LOGGER.debug(f"AtomComp out shape: {atom_comp_out_shape}")
    diatom_comp_out_shape = (30 // (config["DiatomComp_MaxPool_kernel_size"])) * (
        config["DiatomComp_out_channels_start"] // 2
    )
    LOGGER.debug(f"DiatomComp out shape: {diatom_comp_out_shape}")
    globals_out_shape = config["Global_units"]
    LOGGER.debug(f"Globals out shape: {globals_out_shape}")
    onehot_comp_out_shape = (60 // (config["OneHot_MaxPool_kernel_size"])) * config[
        "OneHot_out_channels"
    ]
    LOGGER.debug(f"OneHot out shape: {onehot_comp_out_shape}")

    if config["add_X_mol"]:
        mol_desc_comp_out_shape = (60 // (2 * config["Mol_MaxPool_kernel_size"])) * (
            config["Mol_out_channels_start"] // 4
        )
        LOGGER.debug(f"MolDesc out shape: {mol_desc_comp_out_shape}")
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
