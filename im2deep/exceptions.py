"""
Custom exceptions for the IM2Deep package.

This module defines package-specific exception classes for clearer error
handling and debugging.

Classes
-------
IM2DeepError
    Base exception class for all IM2Deep-related errors.
CalibrationError
    Exception raised when calibration-related errors occur.
"""


class IM2DeepError(Exception):
    """
    Base exception class for all IM2Deep-related errors.

    This class is the root for all custom IM2Deep exceptions, allowing users to
    catch package-specific errors with a single ``except`` clause.

    Attributes
    ----------
    args : tuple
        Positional arguments passed to ``Exception``.

    Examples
    --------
    >>> try:
    ...     predict_ccs(invalid_data)
    ... except IM2DeepError as exc:
    ...     print(f"IM2Deep error occurred: {exc}")
    """

    pass


class CalibrationError(IM2DeepError):
    """
    Exception raised when calibration-related errors occur.

    Raised when issues in calibration inputs or procedures prevent successful
    CCS calibration.

    Notes
    -----
    Typical scenarios include:

    - Insufficient overlap between calibration and reference peptides.
    - Invalid calibration file format.
    - Missing required columns in calibration data.
    - Numerical instability during calibration calculation.

    Examples
    --------
    >>> try:
    ...     linear_calibration(pred_df, cal_df, ref_df)
    ... except CalibrationError as exc:
    ...     print(f"Calibration failed: {exc}")
    """

    pass
