import polars as pl
import pytest
from openeye import oechem

# Import to register namespace
import oepolars  # noqa: F401
from oepolars import DesignUnitType, MoleculeType, MoleculeTypeKind, QueryFormat, QueryType
from oepolars.exceptions import InvalidSMARTS


def _make_titled_molecule(smiles: str, title: str) -> oechem.OEMol:
    mol = oechem.OEMol()
    assert oechem.OEParseSmiles(mol, smiles)
    mol.SetTitle(title)
    mol.GetActive().SetTitle(f"{title} active")
    return mol


def _copy_molecule_into(
    target: oechem.OEMolBase,
    source: oechem.OEMolBase,
) -> bool:
    target.Clear()
    oechem.OEAddMols(target, source)
    target.SetTitle(source.GetTitle())
    if isinstance(target, oechem.OEMol) and isinstance(source, oechem.OEMol):
        target.GetActive().SetTitle(source.GetActive().GetTitle())
    return True


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
        with pytest.raises(TypeError, match="MoleculeType, QueryType, or DesignUnitType"):
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

    def test_is_valid_with_query_type(self):
        query = oechem.OEQMol()
        assert oechem.OEParseSmarts(query, "[#6]")
        s = pl.Series("query", [query, None], dtype=QueryType())

        result = s.chem.is_valid()

        assert result.to_list() == [True, False]


class TestSeriesAsMolecule:
    def test_as_molecule_existing_molecule_type_returns_same_series(self):
        mol = oechem.OEMol()
        assert oechem.OEParseSmiles(mol, "CCO")
        s = pl.Series("mol", [mol], dtype=MoleculeType())

        result = s.chem.as_molecule()

        assert result is s

    def test_as_molecule_with_molecule_type_returns_graph_mol_values(self):
        s = pl.Series("smiles", ["CCO"])

        result = s.chem.as_molecule(molecule_type=MoleculeTypeKind.OEGRAPHMOL)

        assert result.dtype.ext_name() == "molecule"
        assert isinstance(result[0], oechem.OEGraphMol)

    def test_as_molecule_preserves_existing_graph_mol_from_object_series(self):
        mol = oechem.OEGraphMol()
        assert oechem.OEParseSmiles(mol, "CCO")
        s = pl.Series("mol", [mol], dtype=pl.Object)

        result = s.chem.as_molecule()

        assert result.dtype.ext_name() == "molecule"
        assert isinstance(result[0], oechem.OEGraphMol)

    def test_as_molecule_no_title_clears_copy_not_source(self):
        mol = oechem.OEMol()
        assert oechem.OEParseSmiles(mol, "CCO")
        mol.SetTitle("parent")
        mol.GetActive().SetTitle("active")
        s = pl.Series("mol", [mol], dtype=MoleculeType())

        result = s.chem.as_molecule(no_title=True)

        assert result[0] is not mol
        assert result[0].GetTitle() == ""
        assert result[0].GetActive().GetTitle() == ""
        assert mol.GetTitle() == "parent"
        assert mol.GetActive().GetTitle() == "active"


class TestSeriesAsQuery:
    def test_as_query_from_smarts(self):
        s = pl.Series("query", ["[#6]-[#8]", None])

        result = s.chem.as_query()

        assert result.dtype.ext_name() == "query"
        assert isinstance(result[0], oechem.OEQMol)
        assert result[0].IsValid()
        assert result[1] is None

    def test_as_query_from_smirks(self):
        s = pl.Series("query", ["[#6:1]-[#8:2]>>[#6:1]=[#8:2]"])

        result = s.chem.as_query(query_format=QueryFormat.SMIRKS)

        assert result.dtype.ext_name() == "query"
        assert isinstance(result[0], oechem.OEQMol)
        assert result[0].IsValid()

    def test_as_query_existing_query_type_returns_same_series(self):
        query = oechem.OEQMol()
        assert oechem.OEParseSmarts(query, "[#6]")
        s = pl.Series("query", [query], dtype=QueryType())

        result = s.chem.as_query()

        assert result is s

    def test_as_query_no_title_clears_copy_not_source(self):
        query = oechem.OEQMol()
        assert oechem.OEParseSmarts(query, "[#6]")
        query.SetTitle("source")
        s = pl.Series("query", [query], dtype=QueryType())

        result = s.chem.as_query(no_title=True)

        assert result[0] is not query
        assert result[0].GetTitle() == ""
        assert query.GetTitle() == "source"


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


class TestSeriesSubsearch:
    @pytest.fixture
    def molecule_series(self):
        mols = []
        smiles_list = ["CCO", "c1ccccc1", "CC(=O)O", "c1ccc(O)cc1"]  # ethanol, benzene, acetic acid, phenol
        for smi in smiles_list:
            mol = oechem.OEMol()
            oechem.OEParseSmiles(mol, smi)
            mols.append(mol)
        return pl.Series("mol", mols, dtype=MoleculeType())

    def test_subsearch_returns_series(self, molecule_series):
        result = molecule_series.chem.subsearch("c1ccccc1")
        assert isinstance(result, pl.Series)

    def test_subsearch_dtype_is_bool(self, molecule_series):
        result = molecule_series.chem.subsearch("c1ccccc1")
        assert result.dtype == pl.Boolean

    def test_subsearch_finds_benzene_ring(self, molecule_series):
        result = molecule_series.chem.subsearch("c1ccccc1")
        # benzene and phenol have benzene rings
        expected = [False, True, False, True]
        assert result.to_list() == expected

    def test_subsearch_finds_hydroxyl(self, molecule_series):
        result = molecule_series.chem.subsearch("[OH]")
        # ethanol, acetic acid (carboxylic OH), and phenol have hydroxyl
        expected = [True, False, True, True]
        assert result.to_list() == expected

    def test_subsearch_with_oesub_search_object(self, molecule_series):
        ss = oechem.OESubSearch("c1ccccc1")
        result = molecule_series.chem.subsearch(ss)
        expected = [False, True, False, True]
        assert result.to_list() == expected

    def test_substructure_search_accepts_oeqmol(self, molecule_series):
        query = oechem.OEQMol()
        assert oechem.OEParseSmarts(query, "c1ccccc1")

        result = molecule_series.chem.substructure_search(query)

        expected = [False, True, False, True]
        assert result.to_list() == expected

    def test_subsearch_with_nulls(self):
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "c1ccccc1")
        s = pl.Series("mol", [mol, None], dtype=MoleculeType())
        result = s.chem.subsearch("c1ccccc1")
        assert result.to_list() == [True, False]

    def test_subsearch_invalid_smarts_raises(self, molecule_series):
        with pytest.raises(InvalidSMARTS):
            molecule_series.chem.subsearch("invalid[[[smarts")

    def test_subsearch_wrong_type_raises(self):
        s = pl.Series("x", ["a", "b", "c"])
        with pytest.raises(TypeError, match="MoleculeType"):
            s.chem.subsearch("c1ccccc1")


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

    def test_copy_molecules_preserves_concrete_molecule_class(self):
        mol = oechem.OEGraphMol()
        assert oechem.OEParseSmiles(mol, "CCO")
        s = pl.Series("mol", [mol], dtype=MoleculeType())

        result = s.chem.copy_molecules()

        assert isinstance(result[0], oechem.OEGraphMol)
        assert result[0] is not mol

    def test_copy_molecules_wrong_type_raises(self):
        s = pl.Series("x", ["a", "b", "c"])
        with pytest.raises(TypeError, match="MoleculeType"):
            s.chem.copy_molecules()


class TestSeriesDesignUnitOperations:
    @pytest.fixture
    def design_unit_with_titled_components(self):
        class DesignUnitWithTitledComponents:
            def __init__(self) -> None:
                self.ligand = _make_titled_molecule("CCO", "ligand")
                self.protein = _make_titled_molecule("NCC", "protein")

            def GetLigand(self, mol: oechem.OEMolBase) -> bool:
                return _copy_molecule_into(mol, self.ligand)

            def GetProtein(self, mol: oechem.OEMolBase) -> bool:
                return _copy_molecule_into(mol, self.protein)

        return DesignUnitWithTitledComponents()

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

    def test_get_ligands_rejects_conflicting_title_options(self):
        du = oechem.OEDesignUnit()
        s = pl.Series("du", [du], dtype=DesignUnitType())

        with pytest.raises(ValueError, match="clear_titles"):
            s.chem.get_ligands(no_title=True, clear_titles=False)

    def test_get_ligands_no_title_clears_titles(
        self,
        design_unit_with_titled_components,
    ):
        s = pl.Series("du", [design_unit_with_titled_components], dtype=DesignUnitType())

        result = s.chem.get_ligands(no_title=True)

        assert result[0].GetTitle() == ""
        assert result[0].GetActive().GetTitle() == ""

    def test_get_proteins_clear_titles_alias_clears_titles(
        self,
        design_unit_with_titled_components,
    ):
        s = pl.Series("du", [design_unit_with_titled_components], dtype=DesignUnitType())

        result = s.chem.get_proteins(clear_titles=True)

        assert result[0].GetTitle() == ""
        assert result[0].GetActive().GetTitle() == ""

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
