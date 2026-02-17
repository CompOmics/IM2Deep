# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Integrated training functionality from IM2DeepTrainer package**
  - New `train` CLI command for training custom models
  - Training modules: `training.py`, `training_data.py`, `training_evaluate.py`, `training_model.py`, `training_utils.py`
  - Support for single-conformer and multi-conformer model training
  - Transfer learning capabilities from pre-trained backbones
  - Weights & Biases (wandb) integration for experiment tracking
  - Training data extraction and preprocessing utilities
  - Model evaluation and visualization tools
- Training package data including amino acid molecular descriptors and transfer learning backbone
- Comprehensive training documentation in README
- Python API for programmatic model training
- Dependencies: wandb, matplotlib, scipy, deeplcretrainer, pyteomics
- This CHANGELOG file
- Comprehensive documentation with API reference, development guide, and tutorial
- Enhanced error handling with custom exceptions
- Input validation throughout the codebase
- Type hints for better code clarity
- Detailed logging with different verbosity levels

### Changed
- Improved function signatures with better parameter validation
- Enhanced docstrings with NumPy style documentation
- Better error messages with more context
- Improved CLI with better help text and validation
- More robust file handling with proper encoding
- Enhanced calibration functions with edge case handling

### Fixed
- Import error handling for optional dependencies
- File path validation in CLI
- Memory management improvements
- Edge cases in CCS/ion mobility conversions
- Calibration edge cases with insufficient data
