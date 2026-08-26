# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `core.train()` and `core.finetune()`, replacing the `NotImplementedError` stub that
  pointed at the separate `im2deeptrainer` package. Both featurise through
  `DeepLCDataset`, the same path `predict()` uses, so a trained checkpoint is directly
  usable for prediction
- `im2deep train` and `im2deep finetune` CLI commands
- New `_data` module with `CCSDataset` (flattens DeepLC's nested feature tuple for
  training), `build_training_dataset` (accepts a PSMList, DataFrame or delimited file,
  and reads the target from `CCS` or `ccs`) and `grouped_split` (train/validation split
  grouped by stripped sequence, so a peptide cannot appear in both halves)
- `DEFAULT_TRAINING_CONFIG` with the training-loop and featurisation keys, leaving
  `DEFAULT_CONFIG` untouched
- `Global_features` configuration key, making the architectures' global branch width
  configurable so featurisations other than the default 60 can be trained. Defaults to
  60, so existing checkpoints and configurations are unaffected
- `BackboneFreeze` callback, freezing a transfer model's pretrained feature branches for
  a warmup before unfreezing at a reduced learning rate
- Trained checkpoints now record their own configuration, so a model is read back with
  the architecture and featurisation it was trained on
- Weights & Biases logging, off by default. Runs are named after `model_name` (or an
  explicit `wandb.name`) and carry the full training config, so a set of runs differing
  only in training data or featurisation is comparable. `--wandb`, `--wandb-project` and
  `--wandb-name` on both CLI commands; `entity` and `tags` are settable from a config
  file

### Changed
- Widened the `deeplc` dependency to `>=4.1.0,<5`. From 4.1.0 the 4.0.1 feature-encoding
  change sits behind `legacy_positional_deltas`, which defaults to `True`, so the
  encoding matches what the bundled checkpoints were trained with. Verified
  bit-identical to 4.0.0 across 4,000 peptidoforms, 3,000 of them modified
- `predict()` now selects the architecture and featurisation from the checkpoint rather
  than assuming the package defaults, so fine-tuned models and models trained with
  other featurisations load correctly
- `wandb` is now an optional extra (`pip install im2deep[wandb]`) rather than an
  undeclared import

### Fixed
- `IM2DeepTransfer`'s training, validation and test steps did not squeeze the model
  output in the `add_X_mol` branch, so the loss broadcast to a `(batch, batch)` matrix
  instead of comparing element-wise

## [2.0.2] - 2026-07-16

### Added
- Sphinx documentation integration with CHANGELOG.md included in docs

### Changed
- Updated code examples in docstrings and documentation to reflect current API
- Optimized PSM list filtering to use shallow copy instead of full deepcopy for better memory efficiency on large datasets

### Fixed
- CLI argument passing to `predict` function when calibration data is not provided

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
