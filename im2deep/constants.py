from pathlib import Path

# Paths and names for default models and reference datasets
DEFAULT_MODEL_NAME = "IM2DeepUni.ckpt"
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "TIMS" / DEFAULT_MODEL_NAME
DEFAULT_MULTI_MODEL_NAME = "IM2DeepMulti.ckpt"
DEFAULT_MULTI_MODEL = (
    Path(__file__).resolve().parent / "models" / "TIMS" / DEFAULT_MULTI_MODEL_NAME
)
DEFAULT_REFERENCE_DATASET_PATH = (
    Path(__file__).resolve().parent / "reference_data" / "reference_ccs.csv.gz"
)
DEFAULT_MULTI_REFERENCE_DATASET_PATH = (
    Path(__file__).parent / "reference_data" / "multi_reference_ccs.csv.gz"
)
MULTI_BACKBONE_PATH = Path(__file__).parent / "models" / "TIMS" / "multi_output_backbone.ckpt"

# Constant values
SUMMARY_CONSTANT = 18509.8632163405
MASS_GAS_N2 = 28.013
TEMP = 31.85
T_DIFF = 273.15

# Default model configuration
DEFAULT_MULTI_CONFIG = {
    "model_name": "IM2DeepMulti",
    "batch_size": 16,
    "learning_rate": 0.0001,
    "AtomComp_kernel_size": 4,
    "DiatomComp_kernel_size": 2,
    "One_hot_kernel_size": 2,
    "AtomComp_out_channels_start": 256,
    "DiatomComp_out_channels_start": 128,
    "Global_units": 16,
    "OneHot_out_channels": 2,
    "Concat_units": 128,
    "AtomComp_MaxPool_kernel_size": 2,
    "DiatomComp_MaxPool_kernel_size": 2,
    "OneHot_MaxPool_kernel_size": 10,
    "LRelu_negative_slope": 0.1,
    "LRelu_saturation": 20,
    "L1_alpha": 0.00001,
    "delta": 0,
    "device": 0,
    "add_X_mol": False,
    "init": "normal",
    "backbone_SD_path": MULTI_BACKBONE_PATH,
}

DEFAULT_CONFIG = {
    "model_name": "IM2DeepTorch2026ChargeDupes",
    "batch_size": 512,
    "learning_rate": 0.001,
    "AtomComp_kernel_size": 4,
    "DiatomComp_kernel_size": 2,
    "One_hot_kernel_size": 2,
    "AtomComp_out_channels_start": 256,
    "DiatomComp_out_channels_start": 128,
    "Global_units": 16,
    "OneHot_out_channels": 2,
    "Concat_units": 128,
    "AtomComp_MaxPool_kernel_size": 2,
    "DiatomComp_MaxPool_kernel_size": 2,
    "OneHot_MaxPool_kernel_size": 10,
    "LRelu_negative_slope": 0.1,
    "LRelu_saturation": 20,
    "L1_alpha": 0.000005,
    "delta": 0,
    "device": 0,
    "add_X_mol": False,
    "init": "normal",
}

# Number of global features DeepLC yields per featurisation flag combination.
# `matrix_global` is 6 summed atom counts + sequence length + 48 flattened
# positional-composition values = 55, plus 5 CCS features
# (`add_ccs_features`) and/or 12 terminal-composition values
# (`add_terminal_composition`). The architectures' global branch input width
# must match, which is what `Global_features` in the config carries.
GLOBAL_FEATURE_COUNTS = {
    # (add_ccs_features, add_terminal_composition): n_global_features
    (False, False): 55,
    (True, False): 60,
    (False, True): 67,
    (True, True): 72,
}

# Default global-feature width, matching `add_ccs_features=True` and
# `add_terminal_composition=False`, which is what every bundled checkpoint was
# trained with.
DEFAULT_GLOBAL_FEATURES = GLOBAL_FEATURE_COUNTS[(True, False)]

# Default configuration for training. `DEFAULT_CONFIG` above describes the
# bundled checkpoints' architecture and is deliberately left alone so they keep
# loading unchanged; this adds the keys the training loop needs on top.
DEFAULT_TRAINING_CONFIG = {
    **DEFAULT_CONFIG,
    "model_name": "IM2Deep",
    # Training loop
    "epochs": 100,
    "patience": 10,
    "use_best_model": True,
    # "Validation MAE" is the metric name `LogLowestMAE` reads and the one the
    # bundled IM2DeepUni.ckpt's own ModelCheckpoint monitored; changing it
    # means changing both.
    "monitor": "Validation MAE",
    "mode": "min",
    "num_workers": 0,
    "accelerator": "auto",
    "devices": "auto",
    "wandb": {"enabled": False, "project_name": "IM2Deep"},
    # Featurisation. These are recorded in the checkpoint so `predict()` reads
    # a model back with the encoding it was trained on.
    "add_ccs_features": True,
    "add_terminal_composition": False,
    # DeepLC 4.1.0 defaults this to True, reproducing the pre-4.0.1 positional
    # modification indexing the bundled checkpoints were trained with. Set to
    # False to train against the 4.0.1 fix instead.
    "legacy_positional_deltas": True,
    "padding_length": 60,
    "Global_features": DEFAULT_GLOBAL_FEATURES,
}

BASEMODELCONFIG = {
    "AtomComp_kernel_size": 4,
    "DiatomComp_kernel_size": 4,
    "One_hot_kernel_size": 4,
    "AtomComp_out_channels_start": 356,
    "DiatomComp_out_channels_start": 65,
    "Global_units": 20,
    "OneHot_out_channels": 1,
    "Concat_units": 94,
    "AtomComp_MaxPool_kernel_size": 2,
    "DiatomComp_MaxPool_kernel_size": 2,
    "OneHot_MaxPool_kernel_size": 10,
    "LRelu_negative_slope": 0.013545684190756122,
    "LRelu_saturation": 40,
    "init": "normal",
    "add_X_mol": False,
}
