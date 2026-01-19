"""Tests for lazy/scan file readers."""

from pathlib import Path

import polars as pl
import pytest

import oepolars as oep


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestScanSDF:
    def test_scan_sdf_returns_lazyframe(self):
        lf = oep.scan_sdf(FIXTURES_DIR / "sample.sdf")
        assert isinstance(lf, pl.LazyFrame)

    def test_scan_sdf_collect_returns_dataframe(self):
        lf = oep.scan_sdf(FIXTURES_DIR / "sample.sdf")
        df = lf.collect()
        assert isinstance(df, pl.DataFrame)

    def test_scan_sdf_has_molecule_column(self):
        lf = oep.scan_sdf(FIXTURES_DIR / "sample.sdf")
        df = lf.collect()
        assert "Molecule" in df.columns
        assert df.schema["Molecule"].ext_name() == "molecule"

    def test_scan_sdf_has_title_column(self):
        lf = oep.scan_sdf(FIXTURES_DIR / "sample.sdf")
        df = lf.collect()
        assert "Title" in df.columns

    def test_scan_sdf_reads_sd_data(self):
        lf = oep.scan_sdf(FIXTURES_DIR / "sample.sdf")
        df = lf.collect()
        assert "MW" in df.columns
        assert "LogP" in df.columns

    def test_scan_sdf_supports_select(self):
        lf = oep.scan_sdf(FIXTURES_DIR / "sample.sdf")
        # Select only Title column (lazy operation)
        lf_selected = lf.select("Title")
        df = lf_selected.collect()
        assert list(df.columns) == ["Title"]

    def test_scan_sdf_supports_filter(self):
        lf = oep.scan_sdf(FIXTURES_DIR / "sample.sdf")
        # Filter operation (lazy)
        lf_filtered = lf.filter(pl.col("Title") == "ethanol")
        df = lf_filtered.collect()
        assert len(df) == 1

    def test_scan_sdf_file_not_found_raises(self):
        from oepolars.exceptions import FileError
        with pytest.raises(FileError):
            oep.scan_sdf("nonexistent.sdf")

    def test_scan_sdf_molecule_count(self):
        lf = oep.scan_sdf(FIXTURES_DIR / "sample.sdf")
        df = lf.collect()
        assert len(df) == 2
