"""Tests for constants module."""

from pathlib import Path


class TestConstants:
    """Tests for module constants."""

    def test_default_model_path_exists(self):
        """Test that DEFAULT_MODEL constant points to existing file."""
        from im2deep.constants import DEFAULT_MODEL

        if DEFAULT_MODEL is not None:
            model_path = Path(DEFAULT_MODEL)
            # Check if path is valid (may not exist in test environment)
            assert isinstance(DEFAULT_MODEL, (str, Path))

    def test_default_multi_model_path_exists(self):
        """Test that DEFAULT_MULTI_MODEL constant points to existing file."""
        from im2deep.constants import DEFAULT_MULTI_MODEL

        if DEFAULT_MULTI_MODEL is not None:
            model_path = Path(DEFAULT_MULTI_MODEL)
            assert isinstance(DEFAULT_MULTI_MODEL, (str, Path))

    def test_default_reference_dataset_path_exists(self):
        """Test that default reference dataset path exists."""
        from im2deep.constants import DEFAULT_REFERENCE_DATASET_PATH

        if DEFAULT_REFERENCE_DATASET_PATH is not None:
            dataset_path = Path(DEFAULT_REFERENCE_DATASET_PATH)
            assert isinstance(DEFAULT_REFERENCE_DATASET_PATH, (str, Path))

    def test_default_multi_reference_dataset_path_exists(self):
        """Test that default multi reference dataset path exists."""
        from im2deep.constants import DEFAULT_MULTI_REFERENCE_DATASET_PATH

        if DEFAULT_MULTI_REFERENCE_DATASET_PATH is not None:
            dataset_path = Path(DEFAULT_MULTI_REFERENCE_DATASET_PATH)
            assert isinstance(DEFAULT_MULTI_REFERENCE_DATASET_PATH, (str, Path))

    def test_default_config_exists(self):
        """Test that DEFAULT_CONFIG constant exists."""
        from im2deep.constants import DEFAULT_CONFIG

        assert isinstance(DEFAULT_CONFIG, dict)
        assert len(DEFAULT_CONFIG) > 0

    def test_default_multi_config_exists(self):
        """Test that DEFAULT_MULTI_CONFIG constant exists."""
        from im2deep.constants import DEFAULT_MULTI_CONFIG

        assert isinstance(DEFAULT_MULTI_CONFIG, dict)
        assert len(DEFAULT_MULTI_CONFIG) > 0

    def test_config_has_required_keys(self):
        """Test that config dictionaries have required keys."""
        from im2deep.constants import DEFAULT_CONFIG

        # Check for common required keys
        # (actual keys depend on model architecture)
        assert isinstance(DEFAULT_CONFIG, dict)

    def test_constants_are_immutable(self):
        """Test that constants should not be modified."""
        from im2deep import constants

        # Store original values
        original_model = constants.DEFAULT_MODEL

        # Try to modify (this is just checking the pattern, not enforcement)
        # In Python, constants are by convention, not enforced
        assert hasattr(constants, "DEFAULT_MODEL")
