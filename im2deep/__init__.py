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
    - Command-line interface for easy usage

Example:
    Basic usage for CCS prediction:

    >>> from im2deep import predict
    >>> from psm_utils.psm_list import PSMList
    >>> predictions = predict(psm_list)

"""

from importlib.metadata import PackageNotFoundError, version

from im2deep.core import finetune, predict, predict_and_calibrate, train
from im2deep.utils import ccs2im, im2ccs

try:
    __version__: str = version("im2deep")
except PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for version in pyproject.toml
__all__ = [
    "predict",
    "predict_and_calibrate",
    "train",
    "finetune",
    "ccs2im",
    "im2ccs",
]
