from pathlib import Path

import polars as pl
import pytest

import oepolars as oep
from oepolars.exceptions import LazyOperationError

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestLazyFrameNamespaceRegistration:
    def test_chem_namespace_exists_on_lazyframe(self):
        lf = pl.LazyFrame({"x": [1, 2, 3]})
        assert hasattr(lf, "chem")

    def test_chem_namespace_accessible(self):
        lf = oep.scan_sdf(FIXTURES_DIR / "sample.sdf")
        ns = lf.chem
        assert ns is not None


class TestLazyFrameMoleculeColumnDetection:
    def test_has_molecule_columns_returns_true(self):
        lf = oep.scan_sdf(FIXTURES_DIR / "sample.sdf")
        assert lf.chem.has_molecule_columns() is True

    def test_has_molecule_columns_returns_false(self):
        lf = pl.LazyFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        assert lf.chem.has_molecule_columns() is False

    def test_molecule_columns_returns_list(self):
        lf = oep.scan_sdf(FIXTURES_DIR / "sample.sdf")
        cols = lf.chem.molecule_columns()
        assert "Molecule" in cols


class TestLazyFrameRestrictedOperations:
    def test_to_smiles_raises_lazy_operation_error(self):
        lf = oep.scan_sdf(FIXTURES_DIR / "sample.sdf")
        with pytest.raises(LazyOperationError, match="collect"):
            lf.chem.to_smiles("Molecule")

    def test_substructure_search_raises_lazy_operation_error(self):
        lf = oep.scan_sdf(FIXTURES_DIR / "sample.sdf")
        with pytest.raises(LazyOperationError, match="collect"):
            lf.chem.substructure_search("Molecule", "c1ccccc1")

    def test_substructure_filter_raises_lazy_operation_error(self):
        lf = oep.scan_sdf(FIXTURES_DIR / "sample.sdf")
        with pytest.raises(LazyOperationError, match="collect"):
            lf.chem.substructure_filter("Molecule", "c1ccccc1")

    def test_filter_valid_raises_lazy_operation_error(self):
        lf = oep.scan_sdf(FIXTURES_DIR / "sample.sdf")
        with pytest.raises(LazyOperationError, match="collect"):
            lf.chem.filter_valid("Molecule")

    def test_as_molecule_raises_lazy_operation_error(self):
        lf = pl.LazyFrame({"smiles": ["CCO", "c1ccccc1"]})
        with pytest.raises(LazyOperationError, match="collect"):
            lf.chem.as_molecule("smiles")

    def test_detect_molecule_columns_raises_lazy_operation_error(self):
        lf = pl.LazyFrame({"smiles": ["CCO", "c1ccccc1"]})
        with pytest.raises(LazyOperationError, match="collect"):
            lf.chem.detect_molecule_columns()
