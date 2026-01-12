"""Tests for exceptions module."""

import pytest
from im2deep._exceptions import (
    IM2DeepError,
    CalibrationError,
)


class TestExceptions:
    """Tests for custom exception classes."""

    def test_im2deep_error(self):
        """Test IM2DeepError can be raised and caught."""
        with pytest.raises(IM2DeepError, match="test error"):
            raise IM2DeepError("test error")

    def test_calibration_error(self):
        """Test CalibrationError inherits from IM2DeepError."""
        with pytest.raises(IM2DeepError):
            raise CalibrationError("calibration failed")
        
        with pytest.raises(CalibrationError, match="calibration failed"):
            raise CalibrationError("calibration failed")

    def test_exception_inheritance(self):
        """Test that CalibrationError inherits from IM2DeepError."""
        assert issubclass(CalibrationError, IM2DeepError)
        assert issubclass(CalibrationError, Exception)

    def test_exception_with_cause(self):
        """Test exceptions can wrap other exceptions."""
        original_error = ValueError("original error")
        
        with pytest.raises(CalibrationError) as exc_info:
            try:
                raise original_error
            except ValueError as e:
                raise CalibrationError("wrapped error") from e
        
        assert exc_info.value.__cause__ is original_error

    def test_exception_messages(self):
        """Test that exception messages are preserved."""
        message = "detailed error message with context"
        
        try:
            raise IM2DeepError(message)
        except IM2DeepError as e:
            assert str(e) == message
