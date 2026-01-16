import polars as pl
import pytest
from openeye import oechem

from oepolars.types import MoleculeType


class TestMoleculeTypeCreation:
    def test_molecule_type_is_extension(self):
        mol_type = MoleculeType()
        assert isinstance(mol_type, pl.BaseExtension)

    def test_molecule_type_name(self):
        mol_type = MoleculeType()
        assert mol_type.ext_name() == "molecule"

    def test_molecule_type_storage(self):
        mol_type = MoleculeType()
        assert mol_type.ext_storage() == pl.Object


class TestMoleculeTypeSeries:
    @pytest.fixture
    def sample_molecules(self):
        mols = []
        for smi in ["CCO", "c1ccccc1", "CC(=O)O"]:
            mol = oechem.OEMol()
            oechem.OEParseSmiles(mol, smi)
            mols.append(mol)
        return mols

    def test_create_series_from_molecules(self, sample_molecules):
        s = pl.Series("mol", sample_molecules, dtype=MoleculeType())
        assert len(s) == 3
        assert s.dtype.ext_name() == "molecule"

    def test_series_displays_as_ext_molecule(self, sample_molecules):
        s = pl.Series("mol", sample_molecules, dtype=MoleculeType())
        repr_str = repr(s)
        assert "ext[molecule]" in repr_str

    def test_series_with_none_values(self, sample_molecules):
        data = [sample_molecules[0], None, sample_molecules[2]]
        s = pl.Series("mol", data, dtype=MoleculeType())
        assert s.null_count() == 1

    def test_series_item_retrieval(self, sample_molecules):
        s = pl.Series("mol", sample_molecules, dtype=MoleculeType())
        mol = s[0]
        assert isinstance(mol, oechem.OEMolBase)


class TestMoleculeTypeInDataFrame:
    @pytest.fixture
    def sample_molecules(self):
        mols = []
        for smi in ["CCO", "c1ccccc1"]:
            mol = oechem.OEMol()
            oechem.OEParseSmiles(mol, smi)
            mols.append(mol)
        return mols

    def test_dataframe_with_molecule_column(self, sample_molecules):
        df = pl.DataFrame({
            "mol": pl.Series(sample_molecules, dtype=MoleculeType()),
            "name": ["ethanol", "benzene"],
        })
        assert df.schema["mol"].ext_name() == "molecule"
        assert df.schema["name"] == pl.String

    def test_dataframe_select_molecule_column(self, sample_molecules):
        df = pl.DataFrame({
            "mol": pl.Series(sample_molecules, dtype=MoleculeType()),
            "name": ["ethanol", "benzene"],
        })
        result = df.select("mol")
        assert result.schema["mol"].ext_name() == "molecule"
