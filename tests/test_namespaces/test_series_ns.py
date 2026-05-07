import polars as pl
import pytest
from openeye import oechem

# Import to register namespace
import oepolars  # noqa: F401
from oepolars import DesignUnitType, MoleculeType
from oepolars.exceptions import InvalidSMARTS


class TestSeriesNamespaceRegistration:
    def test_chem_namespace_exists_on_series(self):
        s = pl.Series("x", [1, 2, 3])
        assert hasattr(s, "chem")

    def test_chem_namespace_accessible(self):
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")
        s = pl.Series("mol", [mol], dtype=MoleculeType())
        ns = s.chem
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
        result = molecule_series_with_nulls.chem.is_valid()
        assert isinstance(result, pl.Series)

    def test_is_valid_dtype_is_bool(self, molecule_series_with_nulls):
        result = molecule_series_with_nulls.chem.is_valid()
        assert result.dtype == pl.Boolean

    def test_is_valid_detects_nulls(self, molecule_series_with_nulls):
        result = molecule_series_with_nulls.chem.is_valid()
        expected = [True, False, True]
        assert result.to_list() == expected

    def test_is_valid_all_valid(self):
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")
        s = pl.Series("mol", [mol, mol], dtype=MoleculeType())
        result = s.chem.is_valid()
        assert result.to_list() == [True, True]

    def test_is_valid_wrong_type_raises(self):
        s = pl.Series("x", ["a", "b", "c"])
        with pytest.raises(TypeError, match="MoleculeType or DesignUnitType"):
            s.chem.is_valid()

    def test_is_valid_empty_series(self):
        s = pl.Series("mol", [], dtype=MoleculeType())
        result = s.chem.is_valid()
        assert result.to_list() == []

    def test_is_valid_all_nulls(self):
        s = pl.Series("mol", [None, None], dtype=MoleculeType())
        result = s.chem.is_valid()
        assert result.to_list() == [False, False]

    def test_is_valid_with_design_unit_type(self):
        from oepolars import DesignUnitType
        du = oechem.OEDesignUnit()
        s = pl.Series("du", [du, None], dtype=DesignUnitType())
        result = s.chem.is_valid()
        assert result.to_list() == [True, False]


class TestSeriesToSmiles:
    @pytest.fixture
    def molecule_series(self):
        mols = []
        for smi in ["CCO", "c1ccccc1", "CC(=O)O"]:
            mol = oechem.OEMol()
            oechem.OEParseSmiles(mol, smi)
            mols.append(mol)
        return pl.Series("mol", mols, dtype=MoleculeType())

    def test_to_smiles_returns_series(self, molecule_series):
        result = molecule_series.chem.to_smiles()
        assert isinstance(result, pl.Series)

    def test_to_smiles_dtype_is_string(self, molecule_series):
        result = molecule_series.chem.to_smiles()
        assert result.dtype == pl.String

    def test_to_smiles_generates_valid_smiles(self, molecule_series):
        result = molecule_series.chem.to_smiles()
        # Check that we can parse the generated SMILES back
        for smi in result.to_list():
            mol = oechem.OEMol()
            assert oechem.OEParseSmiles(mol, smi)

    def test_to_smiles_with_nulls(self):
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")
        s = pl.Series("mol", [mol, None], dtype=MoleculeType())
        result = s.chem.to_smiles()
        assert result[0] is not None
        assert result[1] is None

    def test_to_smiles_with_flavor(self, molecule_series):
        # Test with isomeric flag
        result = molecule_series.chem.to_smiles(flavor=oechem.OESMILESFlag_Canonical)
        assert isinstance(result, pl.Series)

    def test_to_smiles_wrong_type_raises(self):
        s = pl.Series("x", ["a", "b", "c"])
        with pytest.raises(TypeError, match="MoleculeType"):
            s.chem.to_smiles()


class TestSeriesSubstructureSearch:
    @pytest.fixture
    def molecule_series(self):
        mols = []
        smiles_list = ["CCO", "c1ccccc1", "CC(=O)O", "c1ccc(O)cc1"]  # ethanol, benzene, acetic acid, phenol
        for smi in smiles_list:
            mol = oechem.OEMol()
            oechem.OEParseSmiles(mol, smi)
            mols.append(mol)
        return pl.Series("mol", mols, dtype=MoleculeType())

    def test_substructure_search_returns_series(self, molecule_series):
        result = molecule_series.chem.substructure_search("c1ccccc1")
        assert isinstance(result, pl.Series)

    def test_substructure_search_dtype_is_bool(self, molecule_series):
        result = molecule_series.chem.substructure_search("c1ccccc1")
        assert result.dtype == pl.Boolean

    def test_substructure_search_finds_benzene_ring(self, molecule_series):
        result = molecule_series.chem.substructure_search("c1ccccc1")
        # benzene and phenol have benzene rings
        expected = [False, True, False, True]
        assert result.to_list() == expected

    def test_substructure_search_finds_hydroxyl(self, molecule_series):
        result = molecule_series.chem.substructure_search("[OH]")
        # ethanol, acetic acid (carboxylic OH), and phenol have hydroxyl
        expected = [True, False, True, True]
        assert result.to_list() == expected

    def test_substructure_search_with_oesub_search_object(self, molecule_series):
        ss = oechem.OESubSearch("c1ccccc1")
        result = molecule_series.chem.substructure_search(ss)
        expected = [False, True, False, True]
        assert result.to_list() == expected

    def test_substructure_search_with_nulls(self):
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "c1ccccc1")
        s = pl.Series("mol", [mol, None], dtype=MoleculeType())
        result = s.chem.substructure_search("c1ccccc1")
        assert result.to_list() == [True, False]

    def test_substructure_search_invalid_smarts_raises(self, molecule_series):
        with pytest.raises(InvalidSMARTS):
            molecule_series.chem.substructure_search("invalid[[[smarts")

    def test_substructure_search_wrong_type_raises(self):
        s = pl.Series("x", ["a", "b", "c"])
        with pytest.raises(TypeError, match="MoleculeType"):
            s.chem.substructure_search("c1ccccc1")

    def test_substructure_filter_is_inverse(self, molecule_series):
        search = molecule_series.chem.substructure_search("c1ccccc1")
        filtered = molecule_series.chem.substructure_filter("c1ccccc1")
        assert filtered.to_list() == [not v for v in search.to_list()]


class TestSeriesCopyMolecules:
    @pytest.fixture
    def molecule_series(self):
        mols = []
        for smi in ["CCO", "c1ccccc1"]:
            mol = oechem.OEMol()
            oechem.OEParseSmiles(mol, smi)
            mols.append(mol)
        return pl.Series("mol", mols, dtype=MoleculeType())

    def test_copy_molecules_returns_series(self, molecule_series):
        result = molecule_series.chem.copy_molecules()
        assert isinstance(result, pl.Series)

    def test_copy_molecules_preserves_dtype(self, molecule_series):
        result = molecule_series.chem.copy_molecules()
        assert result.dtype.ext_name() == "molecule"

    def test_copy_molecules_creates_independent_copies(self, molecule_series):
        result = molecule_series.chem.copy_molecules()
        # Modify original
        original_mol = molecule_series[0]
        original_mol.SetTitle("modified")
        # Check copy is unaffected
        copied_mol = result[0]
        assert copied_mol.GetTitle() != "modified"

    def test_copy_molecules_with_nulls(self):
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")
        s = pl.Series("mol", [mol, None], dtype=MoleculeType())
        result = s.chem.copy_molecules()
        assert result[0] is not None
        assert result[1] is None

    def test_copy_molecules_wrong_type_raises(self):
        s = pl.Series("x", ["a", "b", "c"])
        with pytest.raises(TypeError, match="MoleculeType"):
            s.chem.copy_molecules()


class TestSeriesDesignUnitOperations:
    def test_copy_design_units_returns_series(self):
        du = oechem.OEDesignUnit()
        s = pl.Series("du", [du], dtype=DesignUnitType())
        result = s.chem.copy_design_units()
        assert isinstance(result, pl.Series)

    def test_copy_design_units_preserves_dtype(self):
        du = oechem.OEDesignUnit()
        s = pl.Series("du", [du], dtype=DesignUnitType())
        result = s.chem.copy_design_units()
        assert result.dtype.ext_name() == "design_unit"

    def test_copy_design_units_wrong_type_raises(self):
        s = pl.Series("x", ["a", "b", "c"])
        with pytest.raises(TypeError, match="DesignUnitType"):
            s.chem.copy_design_units()

    def test_get_ligands_returns_molecule_series(self):
        # Create a design unit with a ligand
        du = oechem.OEDesignUnit()
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")
        # Note: This is a simplified test - real DUs come from files
        s = pl.Series("du", [du], dtype=DesignUnitType())
        result = s.chem.get_ligands()
        assert result.dtype.ext_name() == "molecule"

    def test_get_proteins_returns_molecule_series(self):
        du = oechem.OEDesignUnit()
        s = pl.Series("du", [du], dtype=DesignUnitType())
        result = s.chem.get_proteins()
        assert result.dtype.ext_name() == "molecule"

    def test_get_components_returns_molecule_series(self):
        du = oechem.OEDesignUnit()
        s = pl.Series("du", [du], dtype=DesignUnitType())
        result = s.chem.get_components(oechem.OEDesignUnitComponents_Ligand)
        assert result.dtype.ext_name() == "molecule"

    def test_copy_design_units_creates_independent_copies(self):
        du = oechem.OEDesignUnit()
        du.SetTitle("original")
        s = pl.Series("du", [du], dtype=DesignUnitType())
        result = s.chem.copy_design_units()
        # Modify original
        du.SetTitle("modified")
        # Verify copy is unaffected
        assert result[0].GetTitle() == "original"

    def test_copy_design_units_with_nulls(self):
        du = oechem.OEDesignUnit()
        s = pl.Series("du", [du, None], dtype=DesignUnitType())
        result = s.chem.copy_design_units()
        assert result[0] is not None
        assert result[1] is None

    def test_get_ligands_wrong_type_raises(self):
        s = pl.Series("x", ["a", "b", "c"])
        with pytest.raises(TypeError, match="DesignUnitType"):
            s.chem.get_ligands()

    def test_get_proteins_wrong_type_raises(self):
        s = pl.Series("x", ["a", "b", "c"])
        with pytest.raises(TypeError, match="DesignUnitType"):
            s.chem.get_proteins()

    def test_get_components_wrong_type_raises(self):
        s = pl.Series("x", ["a", "b", "c"])
        with pytest.raises(TypeError, match="DesignUnitType"):
            s.chem.get_components(oechem.OEDesignUnitComponents_Ligand)

    def test_get_ligands_with_nulls(self):
        du = oechem.OEDesignUnit()
        s = pl.Series("du", [du, None], dtype=DesignUnitType())
        result = s.chem.get_ligands()
        # Empty DU has no ligand, and null DU also returns null
        assert result[1] is None

    def test_get_proteins_with_nulls(self):
        du = oechem.OEDesignUnit()
        s = pl.Series("du", [du, None], dtype=DesignUnitType())
        result = s.chem.get_proteins()
        assert result[1] is None
