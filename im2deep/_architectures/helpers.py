"""Helper functions for IM2Deep architecture and model operations."""

import logging

LOGGER = logging.getLogger(__name__)


def calculate_seq_concat_shape(config):
    """
    Flattened width of ``torch.cat((atom_comp, diatom_comp), 1)``.

    This is the "sequence branch" -
    everything derived from the peptide sequence itself, excluding globals,
    one-hot and mol-desc.

    With ``use_attention_pooling``, each branch ends in
    ``AttentionPool1d`` instead of ``nn.Flatten()``, so its output width is
    just its channel count - independent of sequence length, unlike
    ``Flatten()``'s ``channels * positions``.

    With ``use_flex_trunk``, ``ConvAtomComp`` and
    ``ConvDiatomComp`` are retired entirely in favour of
    :class:`~im2deep._architectures.blocks.FlexTrunk`, whose masked sum+max
    pooling produces ``2 * flex_trunk_channels[-1]`` regardless of
    ``use_attention_pooling`` - checked first and takes priority over it.

    """
    if config.get("use_flex_trunk", False):
        channels = config.get("flex_trunk_channels", (128, 128))
        return 2 * channels[-1]
    if config.get("use_transformer_trunk", False):
        return config["transformer_d_model"]
    if config.get("use_attention_pooling", False):
        atom_comp_out_shape = config["AtomComp_out_channels_start"] // 4
        diatom_comp_out_shape = config["DiatomComp_out_channels_start"] // 2
    else:
        atom_comp_out_shape = (60 // (2 * config["AtomComp_MaxPool_kernel_size"])) * (
            config["AtomComp_out_channels_start"] // 4
        )
        diatom_comp_out_shape = (30 // (config["DiatomComp_MaxPool_kernel_size"])) * (
            config["DiatomComp_out_channels_start"] // 2
        )
    return atom_comp_out_shape + diatom_comp_out_shape


def calculate_concat_shape(config):
    seq_concat_shape = calculate_seq_concat_shape(config)
    LOGGER.debug(f"Sequence (atom+diatom) concat shape: {seq_concat_shape}")
    globals_out_shape = config["Global_units"]
    LOGGER.debug(f"Globals out shape: {globals_out_shape}")
    if config.get("use_flex_trunk", False):
        onehot_comp_out_shape = 0
    elif config.get("use_one_hot", True):
        onehot_comp_out_shape = (60 // (config["OneHot_MaxPool_kernel_size"])) * config[
            "OneHot_out_channels"
        ]
    else:
        onehot_comp_out_shape = 0
    LOGGER.debug(f"OneHot out shape: {onehot_comp_out_shape}")

    if config["add_X_mol"]:
        mol_desc_comp_out_shape = (60 // (2 * config["Mol_MaxPool_kernel_size"])) * (
            config["Mol_out_channels_start"] // 4
        )
        LOGGER.debug(f"MolDesc out shape: {mol_desc_comp_out_shape}")
        total_input_size = (
            seq_concat_shape + globals_out_shape + onehot_comp_out_shape + mol_desc_comp_out_shape
        )

    else:
        total_input_size = seq_concat_shape + globals_out_shape + onehot_comp_out_shape

    return total_input_size
