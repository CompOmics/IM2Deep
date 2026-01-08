"""
Utility functions for IM2Deep package.

This module provides utility functions for converting between different
mobility measurements and configuration settings for multi-conformer models.

Functions:
    im2ccs: Convert ion mobility to collisional cross section
    ccs2im: Convert collisional cross section to ion mobility

Constants:
    multi_config: Configuration dictionary for multi-conformer model
    MULTI_BACKBONE_PATH: Path to the multi-conformer model backbone
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

MULTI_BACKBONE_PATH = (
    Path(__file__).parent / "models" / "TIMS_multi" / "multi_output_backbone.ckpt"
)


def im2ccs(
    reverse_im: float | np.ndarray,
    mz: float | np.ndarray,
    charge: int | np.ndarray,
    mass_gas: float = 28.013,
    temp: float = 31.85,
    t_diff: float = 273.15,
) -> float | np.ndarray:
    """
    Convert reduced ion mobility to collisional cross section.

    This function converts reduced ion mobility (1/K0) values to collisional
    cross section (CCS) using the Mason-Schamp equation. The conversion is
    temperature and gas-dependent.

    Parameters
    ----------
    reverse_im : float or array-like
        Reduced ion mobility (1/K0) in V⋅s/cm².
    mz : float or array-like
        Precursor m/z ratio.
    charge : int or array-like
        Precursor charge state.
    mass_gas : float, optional
        Mass of drift gas in atomic mass units. Default is 28.013 (N₂).
    temp : float, optional
        Temperature in Celsius. Default is 31.85°C
    t_diff : float, optional
        Temperature conversion factor (°C to K). Default is 273.15.

    Returns
    -------
    float or np.ndarray
        Collisional cross section in Ų (square Angstroms).

    Notes
    -----
    The conversion uses the Mason-Schamp equation:
    CCS = (18509.8632163405 * z) / (sqrt(μ * T) * K0)

    Where:
    - z is the charge
    - μ is the reduced mass
    - T is temperature in Kelvin
    - K0 is the ion mobility

    References
    ----------
    Adapted from theGreatHerrLebert/ionmob
    (https://doi.org/10.1093/bioinformatics/btad486)

    Examples
    --------
    >>> im2ccs(0.7, 500.0, 2)
    425.3

    >>> # For arrays
    >>> import numpy as np
    >>> ims = np.array([0.7, 0.8, 0.9])
    >>> mzs = np.array([500.0, 600.0, 700.0])
    >>> charges = np.array([2, 2, 3])
    >>> ccs_values = im2ccs(ims, mzs, charges)
    """
    # Validate inputs
    if np.any(reverse_im <= 0):
        raise ValueError("Reduced ion mobility must be positive")
    if np.any(mz <= 0):
        raise ValueError("m/z must be positive")
    if np.any(charge <= 0):
        raise ValueError("Charge must be positive")
    if mass_gas <= 0:
        raise ValueError("Gas mass must be positive")
    if temp <= -t_diff:
        raise ValueError("Temperature must be above absolute zero")

    SUMMARY_CONSTANT = 18509.8632163405
    reduced_mass = (mz * charge * mass_gas) / (mz * charge + mass_gas)
    return (SUMMARY_CONSTANT * charge) / (np.sqrt(reduced_mass * (temp + t_diff)) * 1 / reverse_im)


def ccs2im(
    ccs: float | np.ndarray,
    mz: float | np.ndarray,
    charge: int | np.ndarray,
    mass_gas: float = 28.013,
    temp: float = 31.85,
    t_diff: float = 273.15,
) -> float | np.ndarray:
    """
    Convert collisional cross section to reduced ion mobility.

    This function converts collisional cross section (CCS) values to reduced
    ion mobility (1/K0) using the inverse of the Mason-Schamp equation.

    Parameters
    ----------
    ccs : float or array-like
        Collisional cross section in Ų (square Angstroms).
    mz : float or array-like
        Precursor m/z ratio.
    charge : int or array-like
        Precursor charge state.
    mass_gas : float, optional
        Mass of drift gas in atomic mass units. Default is 28.013 (N₂).
    temp : float, optional
        Temperature in Celsius. Default is 31.85°C (typical for TIMS).
    t_diff : float, optional
        Temperature conversion factor (°C to K). Default is 273.15.

    Returns
    -------
    float or np.ndarray
        Reduced ion mobility (1/K0) in V⋅s/cm².

    Notes
    -----
    The conversion uses the inverse Mason-Schamp equation:
    1/K0 = (sqrt(μ * T) * CCS) / (18509.8632163405 * z)

    Where:
    - μ is the reduced mass
    - T is temperature in Kelvin
    - z is the charge

    References
    ----------
    Adapted from theGreatHerrLebert/ionmob
    (https://doi.org/10.1093/bioinformatics/btad486)

    Examples
    --------
    >>> ccs2im(425.3, 500.0, 2)
    0.7

    >>> # For arrays
    >>> import numpy as np
    >>> ccs_values = np.array([425.3, 510.2, 680.5])
    >>> mzs = np.array([500.0, 600.0, 700.0])
    >>> charges = np.array([2, 2, 3])
    >>> ims = ccs2im(ccs_values, mzs, charges)
    """
    # Validate inputs
    if np.any(ccs <= 0):
        raise ValueError("CCS must be positive")
    if np.any(mz <= 0):
        raise ValueError("m/z must be positive")
    if np.any(charge <= 0):
        raise ValueError("Charge must be positive")
    if mass_gas <= 0:
        raise ValueError("Gas mass must be positive")
    if temp <= -t_diff:
        raise ValueError("Temperature must be above absolute zero")

    SUMMARY_CONSTANT = 18509.8632163405
    reduced_mass = (mz * charge * mass_gas) / (mz * charge + mass_gas)
    return ((np.sqrt(reduced_mass * (temp + t_diff))) * ccs) / (SUMMARY_CONSTANT * charge)


# Configuration for multi-conformer model
multi_config: dict[str, Any] = {
    "model_name": "IM2DeepMulti",
    "batch_size": 16,
    "learning_rate": 0.0001,
    "AtomComp_kernel_size": 4,
    "DiatomComp_kernel_size": 2,
    "One_hot_kernel_size": 2,
    "AtomComp_out_channels_start": 256,
    "DiatomComp_out_channels_start": 128,
    "Global_units": 16,
    "OneHot_out_channels": 2,
    "Concat_units": 128,
    "AtomComp_MaxPool_kernel_size": 2,
    "DiatomComp_MaxPool_kernel_size": 2,
    "Mol_MaxPool_kernel_size": 2,
    "OneHot_MaxPool_kernel_size": 10,
    "LRelu_negative_slope": 0.1,
    "LRelu_saturation": 20,
    "L1_alpha": 0.00001,
    "delta": 0,
    "device": 0,
    "add_X_mol": False,
    "init": "normal",
    "backbone_SD_path": MULTI_BACKBONE_PATH,
}
