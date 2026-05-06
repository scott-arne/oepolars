import polars as pl
import pytest
from openeye import oechem

from oepolars.types import MoleculeType, MoleculeTypeKind
from oepolars.types.molecule import (
    _coerce_molecule,
    _copy_molecule,
    _molecule_class_for_type,
    _new_molecule,
    _normalize_molecule_type,
)


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


class TestMoleculeTypeKind:
    def test_normalizes_none_to_default(self):
        assert _normalize_molecule_type(None) == MoleculeTypeKind.DEFAULT

    def test_normalizes_enum(self):
        assert _normalize_molecule_type(MoleculeTypeKind.OEMOL) == MoleculeTypeKind.OEMOL

    def test_normalizes_flexible_string(self):
        assert _normalize_molecule_type("oe_graph_mol") == MoleculeTypeKind.OEGRAPHMOL

    def test_normalizes_openeye_class(self):
        assert _normalize_molecule_type(oechem.OEQMol) == MoleculeTypeKind.OEQMOL

    def test_rejects_invalid_selector(self):
        with pytest.raises(ValueError, match="Unsupported molecule_type"):
            _normalize_molecule_type("rdkit")

    def test_molecule_class_for_type(self):
        assert _molecule_class_for_type("oeqmol") is oechem.OEQMol

    def test_new_molecule_uses_selected_class(self):
        assert isinstance(_new_molecule("oe_graph_mol"), oechem.OEGraphMol)

    def test_coerce_molecule_uses_selected_class(self):
        mol = oechem.OEMol()
        assert oechem.OEParseSmiles(mol, "CCO")

        coerced = _coerce_molecule(mol, "oeqmol")

        assert isinstance(coerced, oechem.OEQMol)
        assert oechem.OEMolToSmiles(coerced) == "CCO"

    def test_copy_molecule_can_clear_titles_without_mutating_source(self):
        mol = oechem.OEMol()
        assert oechem.OEParseSmiles(mol, "CCO")
        mol.SetTitle("parent")
        mol.GetActive().SetTitle("active")

        copied = _copy_molecule(mol, no_title=True)

        assert isinstance(copied, oechem.OEMol)
        assert copied is not mol
        assert copied.GetTitle() == ""
        assert copied.GetActive().GetTitle() == ""
        assert mol.GetTitle() == "parent"
        assert mol.GetActive().GetTitle() == "active"


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
