"""Tests for file readers."""

from pathlib import Path

import polars as pl
import pytest

import oepolars as oep

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestReadSDF:
    def test_read_sdf_returns_dataframe(self):
        df = oep.read_sdf(FIXTURES_DIR / "sample.sdf")
        assert isinstance(df, pl.DataFrame)

    def test_read_sdf_has_molecule_column(self):
        df = oep.read_sdf(FIXTURES_DIR / "sample.sdf")
        assert "Molecule" in df.columns
        assert df.schema["Molecule"].ext_name() == "molecule"

    def test_read_sdf_has_title_column(self):
        df = oep.read_sdf(FIXTURES_DIR / "sample.sdf")
        assert "Title" in df.columns
        assert df["Title"].to_list() == ["ethanol", "benzene"]

    def test_read_sdf_reads_sd_data(self):
        df = oep.read_sdf(FIXTURES_DIR / "sample.sdf")
        assert "MW" in df.columns
        assert "LogP" in df.columns

    def test_read_sdf_custom_molecule_column_name(self):
        df = oep.read_sdf(FIXTURES_DIR / "sample.sdf", molecule_column="Mol")
        assert "Mol" in df.columns
        assert df.schema["Mol"].ext_name() == "molecule"

    def test_read_sdf_no_title_column(self):
        df = oep.read_sdf(FIXTURES_DIR / "sample.sdf", title_column=None)
        assert "Title" not in df.columns

    def test_read_sdf_file_not_found_raises(self):
        from oepolars.exceptions import FileError
        with pytest.raises(FileError):
            oep.read_sdf("nonexistent.sdf")

    def test_read_sdf_molecule_count(self):
        df = oep.read_sdf(FIXTURES_DIR / "sample.sdf")
        assert len(df) == 2

    def test_read_sdf_empty_file(self):
        df = oep.read_sdf(FIXTURES_DIR / "empty.sdf")
        assert len(df) == 0
        assert "Molecule" in df.columns
