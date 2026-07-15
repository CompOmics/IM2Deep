# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.1] - 2026-07-15

### Fixed

- Deepcopy fixed that resulted in big memory inflation (and even OOM)

## [2.0.0] - 2026-07-13

### Added
- New `core` module with `predict` and `predict_and_calibrate` as main public API
- New `calibration` module replacing `calibrate`, with `Calibration` and `LinearCCSCalibration` classes
- New `constants` module for default model paths and configuration
- New `exceptions` module (public, replacing private `_exceptions`)
- New `_model_ops` module for PyTorch model loading and inference
- New `_io_helpers` module (split from `utils`) for file parsing and I/O
- Architecture subpackage (`_architectures`) with modular components: activations, blocks, callbacks, losses, helpers
- Multi-conformer CCS prediction support (`IM2DeepMulti`)
- PyTorch Lightning-based model architectures replacing Keras models
- `num_threads` argument to control Torch CPU parallelization
- Profiling support in CLI (`--profile` flag)
- Test suite with tests for calibration, CLI, constants, core, exceptions, integration, losses, model ops, and utils
- CI workflow for tests and type checking
- This CHANGELOG file

### Changed
- **Breaking:** Public API changed from `predict_ccs`/`linear_calibration` to `predict`/`predict_and_calibrate`
- **Breaking:** Minimum Python version raised from 3.10 to 3.11
- **Breaking:** Switched from Keras/TensorFlow models to PyTorch Lightning models (`.ckpt`)
- Switched build system from setuptools to uv
- CLI restructured as click command group with `predict` as default command
- Renamed reference data files for clarity
- Linting moved from black/isort to ruff
- Upgrade `deeplc` dependency to `deeplc>=4.0.0b1,<5`
- Added explicit lower/upper version bounds to all core dependencies.

### Removed
- `im2deep.py` module (replaced by `core.py`)
- `calibrate.py` module (replaced by `calibration.py`)
- `predict_multi.py` module (functionality merged into architecture subpackage)
- Keras model files
- Removed pinned `numpy==1.26.0` dependency
- Removed `im2deeptrainer` optional dependency
