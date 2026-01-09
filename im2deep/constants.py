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
)  # TODO: Remake the dataset
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
