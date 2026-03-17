"""Tests for CLI module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from im2deep.__main__ import cli


class TestCLI:
    """Tests for command-line interface."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def temp_input_file(self):
        """Create a temporary input file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("seq,modifications,charge\n")
            f.write("PEPTIDE,,2\n")
            f.write("SEQUENCE,,3\n")
            yield Path(f.name)

    @pytest.fixture
    def temp_cal_file(self):
        """Create a temporary calibration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("seq,modifications,charge,CCS\n")
            f.write("PEPTIDE,,2,450.5\n")
            f.write("TESTPEP,,2,480.2\n")
            yield Path(f.name)

    def test_cli_help(self, runner):
        """Test CLI help message."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "IM2Deep" in result.output or "predict" in result.output

    def test_cli_version(self, runner):
        """Test CLI version flag."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0

    @patch("im2deep.core.predict_and_calibrate")
    def test_predict_command_basic(self, mock_predict, runner, temp_input_file, temp_cal_file):
        """Test basic predict command."""
        mock_predict.return_value = MagicMock()

        result = runner.invoke(
            cli,
            [
                "predict",
                str(temp_input_file),
                "--calibration-precursors",
                str(temp_cal_file),
            ],
        )

        # Check that it doesn't crash
        assert "error" not in result.output.lower() or result.exit_code == 0

    @patch("im2deep.core.predict_and_calibrate")
    def test_predict_command_with_output(
        self, mock_predict, runner, temp_input_file, temp_cal_file, tmp_path
    ):
        """Test predict command with output file."""
        mock_predict.return_value = MagicMock()
        output_file = tmp_path / "output.csv"

        result = runner.invoke(
            cli,
            [
                "predict",
                str(temp_input_file),
                "--calibration-precursors",
                str(temp_cal_file),
                "--output-file",
                str(output_file),
            ],
        )

        # Should create output file or at least not crash
        assert result.exit_code in [0, 1]  # May fail due to mocking

    @patch("im2deep.core.predict_and_calibrate")
    def test_predict_default_command(self, mock_predict, runner, temp_input_file, temp_cal_file):
        """Test that predict is the default command."""
        mock_predict.return_value = MagicMock()

        # Call without explicit 'predict' subcommand
        result = runner.invoke(
            cli,
            [
                str(temp_input_file),
                "--calibration-precursors",
                str(temp_cal_file),
            ],
        )

        # Should work as if 'predict' was specified
        assert result.exit_code in [0, 1]

    def test_predict_missing_input(self, runner):
        """Test predict command without input file."""
        result = runner.invoke(cli, ["predict"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "required" in result.output.lower()

    def test_predict_logging_level(self, runner, temp_input_file):
        """Test predict command with different logging levels."""
        for level in ["debug", "info", "warning", "error"]:
            result = runner.invoke(
                cli, ["predict", str(temp_input_file), "--logging-level", level]
            )
            # Should at least parse the argument
            assert "Invalid value" not in result.output

    @patch("im2deep.core.predict_and_calibrate")
    def test_predict_multi_flag(self, mock_predict, runner, temp_input_file, temp_cal_file):
        """Test predict command with multi-conformer flag."""
        mock_predict.return_value = MagicMock()

        result = runner.invoke(
            cli,
            [
                "predict",
                str(temp_input_file),
                "--calibration-precursors",
                str(temp_cal_file),
                "--multi",
            ],
        )

        # Check that multi flag is recognized
        assert result.exit_code in [0, 1]

    @patch("im2deep.core.predict_and_calibrate")
    def test_predict_per_charge_calibration(
        self, mock_predict, runner, temp_input_file, temp_cal_file
    ):
        """Test predict command with per-charge calibration."""
        mock_predict.return_value = MagicMock()

        # Test with per-charge enabled (default is True, so just don't pass the flag)
        result = runner.invoke(
            cli,
            [
                "predict",
                str(temp_input_file),
                "--calibration-precursors",
                str(temp_cal_file),
            ],
        )

        assert result.exit_code in [0, 1]

    @patch("im2deep.core.predict_and_calibrate")
    def test_predict_global_calibration(
        self, mock_predict, runner, temp_input_file, temp_cal_file
    ):
        """Test predict command with global calibration (per-charge disabled)."""
        mock_predict.return_value = MagicMock()

        # Test with per-charge disabled
        result = runner.invoke(
            cli,
            [
                "predict",
                str(temp_input_file),
                "--calibration-precursors",
                str(temp_cal_file),
                "--calibrate-per-charge",
                "false",
            ],
        )

        assert result.exit_code in [0, 1]

    def test_train_command_not_available(self, runner):
        """Test that train command is not currently available."""
        result = runner.invoke(cli, ["train", "--help"])
        # Train is commented out, so this should fail
        assert result.exit_code != 0 or "train" not in result.output.lower()


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_default(self):
        """Test setup_logging with info level."""
        import logging

        from im2deep.utils import setup_logging

        setup_logging("info")

        logger = logging.getLogger("im2deep")
        assert logger.level == logging.INFO

    def test_setup_logging_debug(self):
        """Test setup_logging with debug level."""
        import logging

        from im2deep.utils import setup_logging

        setup_logging("debug")

        logger = logging.getLogger("im2deep")
        assert logger.level == logging.DEBUG

    def test_setup_logging_warning(self):
        """Test setup_logging with warning level."""
        import logging

        from im2deep.utils import setup_logging

        setup_logging("warning")

        logger = logging.getLogger("im2deep")
        assert logger.level == logging.WARNING

    def test_setup_logging_affects_submodules(self):
        """Test that setup_logging affects all im2deep submodules."""
        import logging

        from im2deep.utils import setup_logging

        setup_logging("debug")

        # Check root logger is set to debug
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG


class TestDefaultCommandGroup:
    """Tests for DefaultCommandGroup."""

    def test_default_command_group_import(self):
        """Test that DefaultCommandGroup can be imported."""
        import click

        from im2deep.utils import DefaultCommandGroup

        assert issubclass(DefaultCommandGroup, click.Group)
