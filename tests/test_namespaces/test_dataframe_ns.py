import polars as pl
import pytest
from openeye import oechem

import oepolars  # noqa: F401
from oepolars import MoleculeType


class TestDataFrameNamespaceRegistration:
    def test_oe_namespace_exists_on_dataframe(self):
        df = pl.DataFrame({"x": [1, 2, 3]})
        assert hasattr(df, "oe")

    def test_oe_namespace_accessible(self):
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")
        df = pl.DataFrame({
            "mol": pl.Series([mol], dtype=MoleculeType()),
        })
        ns = df.oe
        assert ns is not None


class TestDataFrameAsMolecule:
    def test_as_molecule_converts_smiles_column(self):
        df = pl.DataFrame({"smiles": ["CCO", "c1ccccc1"]})
        result = df.oe.as_molecule("smiles")
        assert result.schema["smiles"].ext_name() == "molecule"

    def test_as_molecule_multiple_columns(self):
        df = pl.DataFrame({
            "smi1": ["CCO", "c1ccccc1"],
            "smi2": ["CC", "CCC"],
        })
        result = df.oe.as_molecule(["smi1", "smi2"])
        assert result.schema["smi1"].ext_name() == "molecule"
        assert result.schema["smi2"].ext_name() == "molecule"

    def test_as_molecule_preserves_other_columns(self):
        df = pl.DataFrame({
            "smiles": ["CCO", "c1ccccc1"],
            "name": ["ethanol", "benzene"],
        })
        result = df.oe.as_molecule("smiles")
        assert result.schema["name"] == pl.String

    def test_as_molecule_handles_invalid_smiles(self):
        df = pl.DataFrame({"smiles": ["CCO", "invalid_smiles", "c1ccccc1"]})
        result = df.oe.as_molecule("smiles")
        # Invalid SMILES should become null
        assert result["smiles"].null_count() == 1

    def test_as_molecule_column_not_found_raises(self):
        df = pl.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(KeyError):
            df.oe.as_molecule("nonexistent")

    def test_as_molecule_already_molecule_no_change(self):
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")
        df = pl.DataFrame({"mol": pl.Series([mol], dtype=MoleculeType())})
        result = df.oe.as_molecule("mol")
        assert result.schema["mol"].ext_name() == "molecule"
