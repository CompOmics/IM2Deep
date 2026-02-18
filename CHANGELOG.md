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
- **Out-of-Distribution (OOD) detection module** (`ood_detection.py`)
  - Latent embedding extraction from first dense layer after feature concatenation
  - Support for both training and inference data
  - Efficient batch processing for large datasets
  - Comprehensive validation and sanity checks
  - Save/load functionality for embeddings (compressed NPZ format)
  - Deterministic extraction without model modification
  - Works with all model variants (IM2Deep, IM2DeepMulti, transfer models)
  - Example scripts and comprehensive documentation
- Training package data including amino acid molecular descriptors and transfer learning backbone
- Comprehensive training documentation in README
- OOD detection documentation (docs/OOD_DETECTION.md)
- Python API for programmatic model training and embedding extraction
- Dependencies: wandb, matplotlib, scipy, deeplcretrainer, pyteomics
- Test suite for OOD detection functionality
- Example script for embedding extraction (examples/extract_embeddings_example.py)
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
