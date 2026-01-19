import polars as pl
import pytest
from openeye import oechem

import oepolars  # noqa: F401
from oepolars import MoleculeType


class TestDataFrameNamespaceRegistration:
    def test_chem_namespace_exists_on_dataframe(self):
        df = pl.DataFrame({"x": [1, 2, 3]})
        assert hasattr(df, "chem")

    def test_chem_namespace_accessible(self):
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")
        df = pl.DataFrame({
            "mol": pl.Series([mol], dtype=MoleculeType()),
        })
        ns = df.chem
        assert ns is not None


class TestDataFrameAsMolecule:
    def test_as_molecule_converts_smiles_column(self):
        df = pl.DataFrame({"smiles": ["CCO", "c1ccccc1"]})
        result = df.chem.as_molecule("smiles")
        assert result.schema["smiles"].ext_name() == "molecule"

    def test_as_molecule_multiple_columns(self):
        df = pl.DataFrame({
            "smi1": ["CCO", "c1ccccc1"],
            "smi2": ["CC", "CCC"],
        })
        result = df.chem.as_molecule(["smi1", "smi2"])
        assert result.schema["smi1"].ext_name() == "molecule"
        assert result.schema["smi2"].ext_name() == "molecule"

    def test_as_molecule_preserves_other_columns(self):
        df = pl.DataFrame({
            "smiles": ["CCO", "c1ccccc1"],
            "name": ["ethanol", "benzene"],
        })
        result = df.chem.as_molecule("smiles")
        assert result.schema["name"] == pl.String

    def test_as_molecule_handles_invalid_smiles(self):
        df = pl.DataFrame({"smiles": ["CCO", "invalid_smiles", "c1ccccc1"]})
        result = df.chem.as_molecule("smiles")
        # Invalid SMILES should become null
        assert result["smiles"].null_count() == 1

    def test_as_molecule_column_not_found_raises(self):
        df = pl.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(KeyError):
            df.chem.as_molecule("nonexistent")

    def test_as_molecule_already_molecule_no_change(self):
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")
        df = pl.DataFrame({"mol": pl.Series([mol], dtype=MoleculeType())})
        result = df.chem.as_molecule("mol")
        assert result.schema["mol"].ext_name() == "molecule"


class TestDataFrameFilterValid:
    def test_filter_valid_removes_null_molecules(self):
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")
        df = pl.DataFrame({
            "mol": pl.Series([mol, None, mol], dtype=MoleculeType()),
            "name": ["a", "b", "c"],
        })
        result = df.chem.filter_valid("mol")
        assert len(result) == 2

    def test_filter_valid_multiple_columns(self):
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")
        df = pl.DataFrame({
            "mol1": pl.Series([mol, None, mol], dtype=MoleculeType()),
            "mol2": pl.Series([mol, mol, None], dtype=MoleculeType()),
        })
        result = df.chem.filter_valid(["mol1", "mol2"])
        # Only first row has both valid
        assert len(result) == 1

    def test_filter_valid_preserves_data(self):
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")
        df = pl.DataFrame({
            "mol": pl.Series([mol, None], dtype=MoleculeType()),
            "name": ["ethanol", "invalid"],
        })
        result = df.chem.filter_valid("mol")
        assert result["name"].to_list() == ["ethanol"]

    def test_filter_valid_column_not_found_raises(self):
        df = pl.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(KeyError):
            df.chem.filter_valid("nonexistent")


class TestDataFrameDetectMoleculeColumns:
    def test_detect_converts_smiles_columns(self):
        df = pl.DataFrame({
            "smiles": ["CCO", "c1ccccc1"],
            "name": ["ethanol", "benzene"],
        })
        result = df.chem.detect_molecule_columns()
        # SMILES strings should be detected and converted
        assert result.schema["smiles"].ext_name() == "molecule"
        assert result.schema["name"] == pl.String

    def test_detect_converts_molecule_object_columns(self):
        mol1 = oechem.OEMol()
        oechem.OEParseSmiles(mol1, "CCO")
        mol2 = oechem.OEMol()
        oechem.OEParseSmiles(mol2, "c1ccccc1")
        df = pl.DataFrame({
            "mol": pl.Series([mol1, mol2], dtype=pl.Object),
            "name": ["ethanol", "benzene"],
        })
        result = df.chem.detect_molecule_columns()
        assert result.schema["mol"].ext_name() == "molecule"

    def test_detect_leaves_non_molecule_columns(self):
        df = pl.DataFrame({
            "x": [1, 2, 3],
            "y": ["a", "b", "c"],
        })
        result = df.chem.detect_molecule_columns()
        assert result.schema["x"] == pl.Int64
        assert result.schema["y"] == pl.String
