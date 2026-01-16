import polars as pl
import pytest
from openeye import oechem

# Import to register namespace
import oepolars  # noqa: F401
from oepolars import MoleculeType


class TestSeriesNamespaceRegistration:
    def test_oe_namespace_exists_on_series(self):
        s = pl.Series("x", [1, 2, 3])
        assert hasattr(s, "oe")

    def test_oe_namespace_accessible(self):
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")
        s = pl.Series("mol", [mol], dtype=MoleculeType())
        ns = s.oe
        assert ns is not None


class TestSeriesIsValid:
    @pytest.fixture
    def molecule_series_with_nulls(self):
        mols = []
        for smi in ["CCO", "c1ccccc1"]:
            mol = oechem.OEMol()
            oechem.OEParseSmiles(mol, smi)
            mols.append(mol)
        data = [mols[0], None, mols[1]]
        return pl.Series("mol", data, dtype=MoleculeType())

    def test_is_valid_returns_series(self, molecule_series_with_nulls):
        result = molecule_series_with_nulls.oe.is_valid()
        assert isinstance(result, pl.Series)

    def test_is_valid_dtype_is_bool(self, molecule_series_with_nulls):
        result = molecule_series_with_nulls.oe.is_valid()
        assert result.dtype == pl.Boolean

    def test_is_valid_detects_nulls(self, molecule_series_with_nulls):
        result = molecule_series_with_nulls.oe.is_valid()
        expected = [True, False, True]
        assert result.to_list() == expected

    def test_is_valid_all_valid(self):
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")
        s = pl.Series("mol", [mol, mol], dtype=MoleculeType())
        result = s.oe.is_valid()
        assert result.to_list() == [True, True]

    def test_is_valid_wrong_type_raises(self):
        s = pl.Series("x", ["a", "b", "c"])
        with pytest.raises(TypeError, match="MoleculeType or DesignUnitType"):
            s.oe.is_valid()

    def test_is_valid_empty_series(self):
        s = pl.Series("mol", [], dtype=MoleculeType())
        result = s.oe.is_valid()
        assert result.to_list() == []

    def test_is_valid_all_nulls(self):
        s = pl.Series("mol", [None, None], dtype=MoleculeType())
        result = s.oe.is_valid()
        assert result.to_list() == [False, False]

    def test_is_valid_with_design_unit_type(self):
        from oepolars import DesignUnitType
        du = oechem.OEDesignUnit()
        s = pl.Series("du", [du, None], dtype=DesignUnitType())
        result = s.oe.is_valid()
        assert result.to_list() == [True, False]
