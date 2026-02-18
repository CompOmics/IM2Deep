"""
IM2Deep: Deep learning framework for peptide collisional cross section prediction.

IM2Deep is a Python package that provides accurate CCS (Collisional Cross Section)
prediction for peptides and modified peptides using deep learning models trained
specifically for TIMS (Trapped Ion Mobility Spectrometry) data.

Key Features:
    - Single-conformer CCS prediction using ensemble of neural networks
    - Multi-conformer CCS prediction for peptides with multiple conformations
    - Linear calibration using reference datasets
    - Support for modified peptides
    - Ion mobility conversion utilities
    - Model training capabilities for custom datasets
    - Command-line interface for easy usage

Example:
    Basic usage for CCS prediction:

    >>> from im2deep import predict, predict_and_calibrate
    >>> from psm_utils.psm_list import PSMList
    >>> predictions = predict(psm_list)

    Training a new model:

    >>> from im2deep.training_data import data_extraction
    >>> from im2deep.training import train_model
    >>> # See CLI documentation for config structure
    >>> data, test_df = data_extraction(config)
    >>> trainer, model, test_loader = train_model(data, model_config, output_path)

Dependencies:
    - deeplc: For deep learning model infrastructure
    - psm_utils: For peptide and PSM handling
    - pandas: For data manipulation
    - numpy: For numerical computations
    - torch & lightning: For neural network training and inference
    - click: For command-line interface

Authors:
    - Robbe Devreese
    - Robbin Bouwmeester
    - Ralf Gabriels

License:
    Apache License 2.0
"""

__version__ = "2.0.0-beta"

# Import main functionality for easier access
from importlib.metadata import version
from im2deep.utils import ccs2im, im2ccs
from im2deep.core import predict, predict_and_calibrate

# Training functionality (optional imports - may require additional dependencies)
try:
    from im2deep.training_data import data_extraction
    from im2deep.training import train_model
    from im2deep.training_evaluate import evaluate_and_plot

    _training_available = True
except ImportError:
    _training_available = False

__version__: str = version("im2deep")
__all__ = [
    "predict",
    "predict_and_calibrate",
    "ccs2im",
    "im2ccs",
]

# Add training functions to __all__ if available
if _training_available:
    __all__.extend(["data_extraction", "train_model", "evaluate_and_plot"])
