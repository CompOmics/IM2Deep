"""Pytest configuration and fixtures for IM2Deep tests."""


import numpy as np
import pandas as pd
import pytest
from psm_utils import PSM, Peptidoform, PSMList


@pytest.fixture
def sample_psm_list():
    """Create a sample PSMList for testing."""
    psms = [
        PSM(
            peptidoform=Peptidoform("PEPTIDE/2"),
            spectrum_id="test_001",
            run="test_run",
            collection="test_collection",
            is_decoy=False,
            score=0.95,
        ),
        PSM(
            peptidoform=Peptidoform("SEQUENCE/3"),
            spectrum_id="test_002",
            run="test_run",
            collection="test_collection",
            is_decoy=False,
            score=0.92,
        ),
        PSM(
            peptidoform=Peptidoform("TESTPEPTIDE/2"),
            spectrum_id="test_003",
            run="test_run",
            collection="test_collection",
            is_decoy=False,
            score=0.88,
        ),
    ]
    return PSMList(psm_list=psms)


@pytest.fixture
def sample_psm_list_with_ccs():
    """Create a sample PSMList with CCS values for calibration testing."""
    psms = [
        PSM(
            peptidoform=Peptidoform("PEPTIDE/2"),
            spectrum_id="cal_001",
            run="cal_run",
            collection="cal_collection",
            is_decoy=False,
            score=0.95,
            retention_time=100.5,
            metadata={"CCS": 450.5},
        ),
        PSM(
            peptidoform=Peptidoform("SEQUENCE/3"),
            spectrum_id="cal_002",
            run="cal_run",
            collection="cal_collection",
            is_decoy=False,
            score=0.92,
            retention_time=120.3,
            metadata={"CCS": 520.8},
        ),
        PSM(
            peptidoform=Peptidoform("TESTPEPTIDE/2"),
            spectrum_id="cal_003",
            run="cal_run",
            collection="cal_collection",
            is_decoy=False,
            score=0.88,
            retention_time=135.7,
            metadata={"CCS": 480.2},
        ),
        PSM(
            peptidoform=Peptidoform("ANOTHER/3"),
            spectrum_id="cal_004",
            run="cal_run",
            collection="cal_collection",
            is_decoy=False,
            score=0.90,
            retention_time=142.1,
            metadata={"CCS": 510.5},
        ),
    ]
    return PSMList(psm_list=psms)


@pytest.fixture
def sample_reference_psm_list():
    """Create a sample reference PSMList for calibration."""
    psms = [
        PSM(
            peptidoform=Peptidoform("PEPTIDE/2"),
            spectrum_id="ref_001",
            run="ref_run",
            collection="ref_collection",
            is_decoy=False,
            metadata={"CCS": 455.0},
        ),
        PSM(
            peptidoform=Peptidoform("SEQUENCE/3"),
            spectrum_id="ref_002",
            run="ref_run",
            collection="ref_collection",
            is_decoy=False,
            metadata={"CCS": 525.0},
        ),
        PSM(
            peptidoform=Peptidoform("TESTPEPTIDE/2"),
            spectrum_id="ref_003",
            run="ref_run",
            collection="ref_collection",
            is_decoy=False,
            metadata={"CCS": 485.0},
        ),
        PSM(
            peptidoform=Peptidoform("REFERENCE/4"),
            spectrum_id="ref_004",
            run="ref_run",
            collection="ref_collection",
            is_decoy=False,
            metadata={"CCS": 600.0},
        ),
    ]
    return PSMList(psm_list=psms)


@pytest.fixture
def sample_peptidoforms():
    """Create sample peptidoforms list."""
    return [
        Peptidoform("PEPTIDE/2"),
        Peptidoform("SEQUENCE/3"),
        Peptidoform("TESTPEPTIDE/2"),
    ]


@pytest.fixture
def sample_ccs_values():
    """Create sample CCS values array."""
    return np.array([450.5, 520.8, 480.2], dtype=np.float32)


@pytest.fixture
def sample_predicted_ccs():
    """Create sample predicted CCS values."""
    return np.array([448.0, 516.0, 478.0], dtype=np.float32)


@pytest.fixture
def sample_predicted_ccs_multi():
    """Create sample multi-conformer predicted CCS values."""
    return np.array([[448.0, 452.0], [516.0, 524.0], [478.0, 482.0]], dtype=np.float32)


@pytest.fixture
def temp_model_path(tmp_path):
    """Create a temporary model file path."""
    return tmp_path / "test_model.ckpt"


@pytest.fixture
def sample_legacy_format_df():
    """Create a sample DataFrame in legacy format."""
    return pd.DataFrame(
        {
            "seq": ["PEPTIDE", "SEQUENCE", "TESTPEPTIDE"],
            "modifications": ["", "", ""],
            "charge": [2, 3, 2],
            "CCS": [450.5, 520.8, 480.2],
        }
    )


@pytest.fixture
def sample_peprec_format_df():
    """Create a sample DataFrame in PEPREC format."""
    return pd.DataFrame(
        {
            "spec_id": ["test_001", "test_002", "test_003"],
            "peptide": ["PEPTIDE", "SEQUENCE", "TESTPEPTIDE"],
            "modifications": ["", "", ""],
            "charge": [2, 3, 2],
        }
    )
