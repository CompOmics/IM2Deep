"""Tests for utils module."""


import numpy as np
import pandas as pd
import pytest
from psm_utils import PSMList

from im2deep.exceptions import IM2DeepError
from im2deep.utils import (
    ccs2im,
    im2ccs,
    parse_input,
    validate_psm_list,
)


class TestValidatePSMList:
    """Tests for validate_psm_list function."""

    def test_validate_psm_list_valid(self, sample_psm_list):
        """Test validation with valid PSMList."""
        result = validate_psm_list(sample_psm_list)
        assert isinstance(result, PSMList)
        assert len(result) == len(sample_psm_list)

    def test_validate_psm_list_with_ccs(self, sample_psm_list_with_ccs):
        """Test validation with PSMList containing CCS values."""
        result = validate_psm_list(sample_psm_list_with_ccs, needs_target=True)
        assert isinstance(result, PSMList)
        for psm in result:
            assert "CCS" in psm.metadata
            # CCS should always be stored as float
            assert isinstance(psm.metadata["CCS"], float)

    def test_validate_psm_list_missing_ccs(self, sample_psm_list):
        """Test validation fails when CCS values are required but missing."""
        with pytest.raises(IM2DeepError, match="ion_mobility.*CCS.*metadata"):
            validate_psm_list(sample_psm_list, needs_target=True)

    def test_validate_psm_list_empty(self):
        """Test validation with empty PSMList."""
        empty_list = PSMList(psm_list=[])
        with pytest.raises(IM2DeepError, match="No PSMs present"):
            validate_psm_list(empty_list)

    def test_validate_psm_list_not_psm_list(self):
        """Test validation fails with non-PSMList input."""
        with pytest.raises(IM2DeepError, match="PSMList"):
            validate_psm_list([1, 2, 3])


class TestParseInput:
    """Tests for parse_input function."""

    def test_parse_input_psm_list(self, sample_psm_list):
        """Test parsing PSMList input."""
        result = parse_input(sample_psm_list)
        assert isinstance(result, PSMList)
        assert len(result) == len(sample_psm_list)

    def test_parse_input_csv_file(self, tmp_path, sample_legacy_format_df):
        """Test parsing CSV file."""
        csv_path = tmp_path / "test.csv"
        sample_legacy_format_df.to_csv(csv_path, index=False)

        result = parse_input(csv_path)
        assert isinstance(result, PSMList)
        assert len(result) == len(sample_legacy_format_df)

    def test_parse_input_tsv_file(self, tmp_path, sample_legacy_format_df):
        """Test parsing TSV file."""
        tsv_path = tmp_path / "test.tsv"
        sample_legacy_format_df.to_csv(tsv_path, sep="\t", index=False)

        result = parse_input(tsv_path)
        assert isinstance(result, PSMList)
        assert len(result) == len(sample_legacy_format_df)

    def test_parse_input_peprec_format(self, tmp_path, sample_peprec_format_df):
        """Test parsing PEPREC format file."""
        csv_path = tmp_path / "peprec.csv"
        sample_peprec_format_df.to_csv(csv_path, index=False)

        result = parse_input(csv_path)
        assert isinstance(result, PSMList)
        assert len(result) == len(sample_peprec_format_df)

    def test_parse_input_with_modifications(self, tmp_path):
        """Test parsing file with modifications."""
        df = pd.DataFrame(
            {
                "seq": ["PEPTIDE", "SEQUENCE"],
                "modifications": ["1|Oxidation", ""],
                "charge": [2, 3],
            }
        )
        csv_path = tmp_path / "test_mods.csv"
        df.to_csv(csv_path, index=False)

        result = parse_input(csv_path)
        assert isinstance(result, PSMList)
        assert len(result) == 2

    def test_parse_input_invalid_file(self, tmp_path):
        """Test parsing non-existent file raises error."""
        fake_path = tmp_path / "nonexistent.csv"
        with pytest.raises((FileNotFoundError, IM2DeepError)):
            parse_input(fake_path)

    def test_parse_input_dataframe(self, sample_legacy_format_df):
        """Test parsing DataFrame directly."""
        result = parse_input(sample_legacy_format_df)
        assert isinstance(result, PSMList)
        assert len(result) == len(sample_legacy_format_df)

    def test_parse_input_legacy_format_detection(self, tmp_path):
        """Test that legacy format is properly detected."""
        df = pd.DataFrame(
            {
                "seq": ["PEPTIDE", "SEQUENCE"],
                "modifications": ["", ""],
                "charge": [2, 3],
                "CCS": [450.5, 520.8],
            }
        )
        csv_path = tmp_path / "legacy.csv"
        df.to_csv(csv_path, index=False)

        result = parse_input(csv_path)
        assert isinstance(result, PSMList)
        assert len(result) == 2
        # Check that CCS values are preserved in metadata
        for psm in result:
            assert "CCS" in psm.metadata


class TestCCSConversions:
    """Tests for CCS and ion mobility conversion functions."""

    def test_ccs2im_basic(self):
        """Test basic CCS to ion mobility conversion."""
        ccs = 450.0
        charge = 2
        mz = 500.0
        im = ccs2im(ccs, charge, mz)

        assert isinstance(im, float)
        assert im > 0

    def test_ccs2im_array(self):
        """Test CCS to ion mobility conversion with arrays."""
        ccs = np.array([450.0, 520.0, 480.0])
        charge = np.array([2, 3, 2])
        mz = np.array([500.0, 600.0, 550.0])

        im = ccs2im(ccs, charge, mz)

        assert isinstance(im, np.ndarray)
        assert len(im) == len(ccs)
        assert np.all(im > 0)

    def test_im2ccs_basic(self):
        """Test basic ion mobility to CCS conversion."""
        im = 1.0
        charge = 2
        mz = 500.0
        ccs = im2ccs(im, charge, mz)

        assert isinstance(ccs, float)
        assert ccs > 0

    def test_im2ccs_array(self):
        """Test ion mobility to CCS conversion with arrays."""
        im = np.array([1.0, 1.2, 0.9])
        charge = np.array([2, 3, 2])
        mz = np.array([500.0, 600.0, 550.0])

        ccs = im2ccs(im, charge, mz)

        assert isinstance(ccs, np.ndarray)
        assert len(ccs) == len(im)
        assert np.all(ccs > 0)

    def test_ccs2im_im2ccs_roundtrip(self):
        """Test that CCS -> IM -> CCS conversion is consistent."""
        ccs_original = 450.0
        charge = 2
        mz = 500.0

        im = ccs2im(ccs_original, charge, mz)
        ccs_roundtrip = im2ccs(im, charge, mz)

        assert abs(ccs_roundtrip - ccs_original) < 0.01

    def test_ccs2im_zero_values(self):
        """Test handling of zero values."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            ccs2im(0, 2, 500.0)

    def test_im2ccs_zero_values(self):
        """Test handling of zero values."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            im2ccs(0, 2, 500.0)

    def test_ccs2im_negative_values(self):
        """Test handling of negative values."""
        # Function should raise ValueError for negative CCS values
        with pytest.raises(ValueError, match="CCS must be positive"):
            ccs2im(-450.0, 2, 500.0)

    def test_im2ccs_different_charges(self):
        """Test conversions with different charge states."""
        im = 1.0
        mz = 500.0

        ccs_z2 = im2ccs(im, 2, mz)
        ccs_z3 = im2ccs(im, 3, mz)

        assert ccs_z2 != ccs_z3
        assert ccs_z2 > 0 and ccs_z3 > 0
