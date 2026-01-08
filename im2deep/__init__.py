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

    >>> from im2deep.im2deep import predict_ccs
    >>> from psm_utils.psm_list import PSMList
    >>> predictions = predict_ccs(psm_list, calibration_data)

Dependencies:
    - deeplc: For deep learning model infrastructure
    - psm_utils: For peptide and PSM handling
    - pandas: For data manipulation
    - numpy: For numerical computations
    - click: For command-line interface

Authors:
    - Robbe Devreese
    - Robbin Bouwmeester
    - Ralf Gabriels

License:
    Apache License 2.0
"""

__version__ = "1.2.0"

# Import main functionality for easier access
from im2deep.im2deep import predict_ccs
from im2deep.calibrate import linear_calibration
from im2deep.utils import ccs2im, im2ccs
from im2deep._exceptions import IM2DeepError, CalibrationError

__all__ = [
    "predict_ccs",
    "linear_calibration",
    "ccs2im",
    "im2ccs",
    "IM2DeepError",
    "CalibrationError",
]
