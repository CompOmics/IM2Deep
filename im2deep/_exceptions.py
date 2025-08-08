"""
Custom exceptions for IM2Deep package.

This module defines custom exception classes used throughout the IM2Deep package
for better error handling and debugging.

Classes:
    IM2DeepError: Base exception class for all IM2Deep-related errors
    CalibrationError: Exception raised when calibration-related errors occur
"""


class IM2DeepError(Exception):
    """
    Base exception class for all IM2Deep-related errors.
    
    This exception serves as the base class for all custom exceptions 
    in the IM2Deep package, allowing users to catch all package-specific
    errors with a single except clause.
    
    Attributes:
        message (str): Human readable string describing the exception.
        
    Example:
        >>> try:
        ...     predict_ccs(invalid_data)
        ... except IM2DeepError as e:
        ...     print(f"IM2Deep error occurred: {e}")
    """
    pass


class CalibrationError(IM2DeepError):
    """
    Exception raised when calibration-related errors occur.
    
    This exception is raised when there are issues with calibration data,
    reference datasets, or calibration procedures that prevent successful
    CCS calibration.
    
    Common scenarios:
        - Insufficient overlapping peptides between calibration and reference data
        - Invalid calibration file format
        - Missing required columns in calibration data
        - Numerical issues during calibration calculation
        
    Example:
        >>> try:
        ...     linear_calibration(pred_df, cal_df, ref_df)
        ... except CalibrationError as e:
        ...     print(f"Calibration failed: {e}")
    """
    pass
