# IM2Deep Test Suite

This directory contains comprehensive tests for the IM2Deep package.

## Test Structure

- `conftest.py`: Pytest configuration and shared fixtures
- `test_calibration.py`: Tests for calibration module
- `test_utils.py`: Tests for utility functions
- `test_model_ops.py`: Tests for model operations
- `test_core.py`: Tests for core functionality
- `test_cli.py`: Tests for command-line interface
- `test_exceptions.py`: Tests for custom exceptions
- `test_integration.py`: Integration tests for end-to-end workflows

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_calibration.py
```

### Run with coverage
```bash
pytest --cov=im2deep --cov-report=html
```

### Run with verbose output
```bash
pytest -v
```

### Run only fast tests (skip integration tests)
```bash
pytest -m "not integration"
```

### Run only integration tests
```bash
pytest -m integration
```

## Test Categories

### Unit Tests
- `test_calibration.py`: LinearCCSCalibration class methods
- `test_utils.py`: Input parsing, validation, and conversion functions
- `test_model_ops.py`: Model loading and prediction functions
- `test_core.py`: High-level prediction and calibration functions
- `test_cli.py`: Command-line interface and argument parsing
- `test_exceptions.py`: Custom exception classes

### Integration Tests
- `test_integration.py`: End-to-end workflows and data consistency

## Fixtures

Common fixtures are defined in `conftest.py`:

- `sample_psm_list`: Basic PSMList for testing
- `sample_psm_list_with_ccs`: PSMList with CCS values for calibration
- `sample_reference_psm_list`: Reference PSMList for calibration
- `sample_peptidoforms`: Array of Peptidoform objects
- `sample_ccs_values`: Array of CCS values
- `sample_predicted_ccs`: Array of predicted CCS values (single-output)
- `sample_predicted_ccs_multi`: Array of predicted CCS values (multi-output)
- `temp_model_path`: Temporary file path for model testing
- `sample_legacy_format_df`: DataFrame in legacy format
- `sample_peprec_format_df`: DataFrame in PEPREC format

## Test Coverage

The test suite aims to cover:

- ✅ Input parsing and validation
- ✅ CCS calibration (per-charge and global)
- ✅ Single-output and multi-output predictions
- ✅ Model loading from various checkpoint formats
- ✅ Command-line interface and argument handling
- ✅ Error handling and custom exceptions
- ✅ Data consistency across pipeline
- ✅ Edge cases (single peptide, modified peptides, high charges, etc.)

## Notes

- Some integration tests require trained models and are skipped by default
- Tests use mocking for external dependencies (PyTorch Lightning, DeepLC)
- Multi-output prediction tests verify proper handling of tuple outputs
- Calibration tests verify broadcasting for both single and multi-output cases

## Adding New Tests

When adding new tests:

1. Use appropriate fixtures from `conftest.py`
2. Group related tests in classes
3. Use descriptive test names starting with `test_`
4. Add docstrings explaining what each test verifies
5. Use `@pytest.mark.integration` for tests requiring trained models
6. Mock external dependencies when possible
7. Test both success and failure cases
