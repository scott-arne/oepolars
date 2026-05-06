"""Tests for pandas interoperability."""

import oepandas as oepd
import pandas as pd
import polars as pl
from openeye import oechem

import oepolars as oepl
from oepolars.types import QueryType

FIXTURES_DIR = "tests/fixtures"


class TestFromPandas:
    """Tests for oepl.from_pandas function."""

    def test_from_pandas_returns_polars_dataframe(self):
        """Test that from_pandas returns a Polars DataFrame."""
        pd_df = pd.DataFrame({"a": [1, 2, 3]})
        result = oepl.from_pandas(pd_df)
        assert isinstance(result, pl.DataFrame)

    def test_from_pandas_preserves_non_molecule_columns(self):
        """Test that non-molecule columns are preserved."""
        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = oepl.from_pandas(pd_df)
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["x", "y", "z"]

    def test_from_pandas_converts_molecule_column(self):
        """Test that OEPandas molecule columns are converted to MoleculeType."""
        pd_df = oepd.read_smi(f"{FIXTURES_DIR}/sample.smi", title_column="Title")

        result = oepl.from_pandas(pd_df)

        # Check the molecule column has MoleculeType
        assert hasattr(result.schema["Molecule"], "ext_name")
        assert result.schema["Molecule"].ext_name() == "molecule"

    def test_from_pandas_molecule_values_are_valid(self):
        """Test that converted molecules are valid OEMol objects."""
        pd_df = oepd.read_smi(f"{FIXTURES_DIR}/sample.smi", title_column="Title")

        result = oepl.from_pandas(pd_df)

        # Check molecules are valid
        for mol in result["Molecule"].to_list():
            if mol is not None:
                assert isinstance(mol, oechem.OEMol)
                assert mol.NumAtoms() > 0

    def test_from_pandas_handles_null_molecules(self):
        """Test that null molecules are handled correctly."""
        # Create molecules
        mol1 = oechem.OEMol()
        mol2 = oechem.OEMol()
        oechem.OEParseSmiles(mol1, "CCO")
        oechem.OEParseSmiles(mol2, "c1ccccc1")

        pd_df = pd.DataFrame({
            "Molecule": oepd.MoleculeArray([mol1, None, mol2]),
            "Name": ["ethanol", "null", "benzene"],
        })

        result = oepl.from_pandas(pd_df)

        assert result["Molecule"][0] is not None
        assert result["Molecule"][1] is None
        assert result["Molecule"][2] is not None

    def test_from_pandas_preserves_smiles_structure(self):
        """Test that molecular structures are preserved during conversion."""
        pd_df = oepd.read_smi(f"{FIXTURES_DIR}/sample.smi", title_column="Title")

        # Get SMILES from pandas
        pd_smiles = pd_df["Molecule"].chem.to_smiles()

        result = oepl.from_pandas(pd_df)

        # Get SMILES from polars
        pl_smiles = result["Molecule"].chem.to_smiles().to_list()

        # Compare
        for i, (ps, pls) in enumerate(zip(pd_smiles, pl_smiles, strict=False)):
            assert ps == pls, f"SMILES mismatch at index {i}: {ps} != {pls}"

    def test_from_pandas_with_multiple_molecule_columns(self):
        """Test conversion with multiple molecule columns."""
        mol1 = oechem.OEMol()
        mol2 = oechem.OEMol()
        oechem.OEParseSmiles(mol1, "CCO")
        oechem.OEParseSmiles(mol2, "c1ccccc1")

        pd_df = pd.DataFrame({
            "Mol1": oepd.MoleculeArray([mol1]),
            "Mol2": oepd.MoleculeArray([mol2]),
            "Name": ["pair"],
        })

        result = oepl.from_pandas(pd_df)

        assert result.schema["Mol1"].ext_name() == "molecule"
        assert result.schema["Mol2"].ext_name() == "molecule"
        assert result.schema["Name"] == pl.String

    def test_from_pandas_chem_accessor_works(self):
        """Test that .chem accessor works on converted DataFrame."""
        pd_df = oepd.read_smi(f"{FIXTURES_DIR}/sample.smi", title_column="Title")

        result = oepl.from_pandas(pd_df)

        # Test that chem accessor methods work
        smiles = result["Molecule"].chem.to_smiles()
        assert len(smiles) == len(result)
        assert all(isinstance(s, str) for s in smiles.to_list())

    def test_from_pandas_empty_dataframe(self):
        """Test conversion of empty DataFrame."""
        pd_df = pd.DataFrame({"Molecule": oepd.MoleculeArray([])})
        result = oepl.from_pandas(pd_df)
        assert len(result) == 0
        assert result.schema["Molecule"].ext_name() == "molecule"

    def test_from_pandas_no_molecule_columns(self):
        """Test conversion when no molecule columns exist."""
        pd_df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = oepl.from_pandas(pd_df)
        assert result["a"].to_list() == [1, 2]
        assert result["b"].to_list() == ["x", "y"]

    def test_from_pandas_preserves_column_order(self):
        """Test that column order is preserved during conversion."""
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")

        pd_df = pd.DataFrame({
            "A": [1],
            "Molecule": oepd.MoleculeArray([mol]),
            "B": ["x"],
            "C": [3.14],
        })

        result = oepl.from_pandas(pd_df)

        assert result.columns == ["A", "Molecule", "B", "C"]

    def test_from_pandas_with_sdf_data(self):
        """Test conversion of DataFrame read from SDF."""
        pd_df = oepd.read_sdf(f"{FIXTURES_DIR}/sample.sdf")

        result = oepl.from_pandas(pd_df)

        assert result.schema["Molecule"].ext_name() == "molecule"
        assert len(result) > 0

    def test_from_pandas_converts_query_column(self):
        """Test that OEPandas query columns are converted to QueryType."""
        query_array = oepd.QueryArray.from_sequence_of_strings(["[#6]"])
        pd_df = pd.DataFrame({"Query": query_array})
        source_query = pd_df["Query"].iloc[0]

        result = oepl.from_pandas(pd_df)

        assert result.schema["Query"] == QueryType()
        assert isinstance(result["Query"][0], oechem.OEQMol)
        assert result["Query"][0] is not source_query
