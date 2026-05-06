import polars as pl
import pytest
from openeye import oechem

import oepolars  # noqa: F401
from oepolars import MoleculeType, QueryFormat, QueryType


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

    def test_as_molecule_with_molecule_type_returns_oegraphmol_values(self):
        df = pl.DataFrame({"smiles": ["CCO", "c1ccccc1"]})

        result = df.chem.as_molecule("smiles", molecule_type=oechem.OEGraphMol)

        assert result.schema["smiles"].ext_name() == "molecule"
        assert all(
            isinstance(mol, oechem.OEGraphMol)
            for mol in result["smiles"].to_list()
        )

    def test_as_molecule_no_title_clears_copy_not_source(self):
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")
        mol.SetTitle("source")
        df = pl.DataFrame({"mol": pl.Series([mol], dtype=MoleculeType())})

        result = df.chem.as_molecule("mol", no_title=True)

        assert df["mol"][0].GetTitle() == "source"
        assert result["mol"][0].GetTitle() == ""


class TestDataFrameAsQuery:
    def test_as_query_converts_smarts_column(self):
        df = pl.DataFrame({"query": ["[#6]", "c1ccccc1"]})

        result = df.chem.as_query("query")

        assert result.schema["query"].ext_name() == "query"
        assert all(
            isinstance(query, oechem.OEQMol)
            for query in result["query"].to_list()
        )

    def test_as_query_converts_smirks_column(self):
        df = pl.DataFrame({"query": ["[C:1]>>[C:1][O]"]})

        result = df.chem.as_query("query", query_format=QueryFormat.SMIRKS)

        assert result.schema["query"].ext_name() == "query"
        assert isinstance(result["query"][0], oechem.OEQMol)

    def test_as_query_no_title_clears_copy_not_source(self):
        query = oechem.OEQMol()
        assert oechem.OEParseSmarts(query, "[#6]")
        query.SetTitle("source")
        df = pl.DataFrame({"query": pl.Series([query], dtype=QueryType())})

        result = df.chem.as_query("query", no_title=True)

        assert df["query"][0].GetTitle() == "source"
        assert result["query"][0].GetTitle() == ""

    def test_as_query_column_not_found_raises(self):
        df = pl.DataFrame({"x": [1, 2, 3]})

        with pytest.raises(KeyError):
            df.chem.as_query("nonexistent")


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


class TestDataFrameSubstructureSearch:
    def test_substructure_search_accepts_oeqmol(self):
        mols = []
        for smiles in ["CCO", "c1ccccc1", "c1ccc(O)cc1"]:
            mol = oechem.OEMol()
            oechem.OEParseSmiles(mol, smiles)
            mols.append(mol)
        df = pl.DataFrame({"mol": pl.Series(mols, dtype=MoleculeType())})
        query = oechem.OEQMol()
        assert oechem.OEParseSmarts(query, "c1ccccc1")

        result = df.chem.substructure_search("mol", query)

        assert result.to_list() == [False, True, True]


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


class TestDataFrameWriteSDF:
    @pytest.fixture
    def molecule_dataframe(self):
        mols = []
        for smiles in ["CCO", "c1ccccc1"]:
            mol = oechem.OEMol()
            oechem.OEParseSmiles(mol, smiles)
            mols.append(mol)
        return pl.DataFrame({
            "Molecule": pl.Series(mols, dtype=MoleculeType()),
            "name": ["ethanol", "benzene"],
            "MW": [46.07, 78.11],
        })

    def test_write_sdf_creates_file(self, molecule_dataframe, tmp_path):
        output_path = tmp_path / "output.sdf"
        molecule_dataframe.chem.write_sdf(output_path, molecule_column="Molecule")
        assert output_path.exists()

    def test_write_sdf_readable(self, molecule_dataframe, tmp_path):
        output_path = tmp_path / "output.sdf"
        molecule_dataframe.chem.write_sdf(output_path, molecule_column="Molecule")

        # Read it back
        import oepolars as oep
        df_back = oep.read_sdf(output_path)
        assert len(df_back) == 2

    def test_write_sdf_with_sd_data(self, molecule_dataframe, tmp_path):
        output_path = tmp_path / "output.sdf"
        molecule_dataframe.chem.write_sdf(
            output_path,
            molecule_column="Molecule",
            sd_columns=["name", "MW"]
        )

        import oepolars as oep
        df_back = oep.read_sdf(output_path)
        assert "name" in df_back.columns
        assert "MW" in df_back.columns

    def test_write_sdf_with_title_column(self, molecule_dataframe, tmp_path):
        output_path = tmp_path / "output.sdf"
        molecule_dataframe.chem.write_sdf(
            output_path,
            molecule_column="Molecule",
            title_column="name"
        )

        import oepolars as oep
        df_back = oep.read_sdf(output_path)
        assert df_back["Title"].to_list() == ["ethanol", "benzene"]

    def test_write_sdf_invalid_molecule_column_raises(self, molecule_dataframe, tmp_path):
        output_path = tmp_path / "output.sdf"
        with pytest.raises(KeyError):
            molecule_dataframe.chem.write_sdf(output_path, molecule_column="nonexistent")

    def test_write_sdf_wrong_column_type_raises(self, tmp_path):
        df = pl.DataFrame({"x": [1, 2, 3]})
        output_path = tmp_path / "output.sdf"
        with pytest.raises(TypeError):
            df.chem.write_sdf(output_path, molecule_column="x")


class TestDataFrameWriteSMI:
    @pytest.fixture
    def molecule_dataframe(self):
        mols = []
        for smiles in ["CCO", "c1ccccc1"]:
            mol = oechem.OEMol()
            oechem.OEParseSmiles(mol, smiles)
            mols.append(mol)
        return pl.DataFrame({
            "Molecule": pl.Series(mols, dtype=MoleculeType()),
            "name": ["ethanol", "benzene"],
        })

    def test_write_smi_creates_file(self, molecule_dataframe, tmp_path):
        output_path = tmp_path / "output.smi"
        molecule_dataframe.chem.write_smi(output_path, molecule_column="Molecule")
        assert output_path.exists()

    def test_write_smi_readable(self, molecule_dataframe, tmp_path):
        output_path = tmp_path / "output.smi"
        molecule_dataframe.chem.write_smi(output_path, molecule_column="Molecule")

        import oepolars as oep
        df_back = oep.read_smi(output_path)
        assert len(df_back) == 2

    def test_write_smi_with_title_column(self, molecule_dataframe, tmp_path):
        output_path = tmp_path / "output.smi"
        molecule_dataframe.chem.write_smi(
            output_path,
            molecule_column="Molecule",
            title_column="name"
        )

        import oepolars as oep
        df_back = oep.read_smi(output_path)
        assert df_back["Title"].to_list() == ["ethanol", "benzene"]

    def test_write_smi_invalid_molecule_column_raises(self, molecule_dataframe, tmp_path):
        output_path = tmp_path / "output.smi"
        with pytest.raises(KeyError):
            molecule_dataframe.chem.write_smi(output_path, molecule_column="nonexistent")

    def test_write_smi_wrong_column_type_raises(self, tmp_path):
        df = pl.DataFrame({"x": [1, 2, 3]})
        output_path = tmp_path / "output.smi"
        with pytest.raises(TypeError):
            df.chem.write_smi(output_path, molecule_column="x")


class TestDataFrameWriteMoleculeCSV:
    @pytest.fixture
    def molecule_dataframe(self):
        mols = []
        for smiles in ["CCO", "c1ccccc1"]:
            mol = oechem.OEMol()
            oechem.OEParseSmiles(mol, smiles)
            mols.append(mol)
        return pl.DataFrame({
            "Molecule": pl.Series(mols, dtype=MoleculeType()),
            "name": ["ethanol", "benzene"],
            "MW": [46.07, 78.11],
        })

    def test_write_molecule_csv_creates_file(self, molecule_dataframe, tmp_path):
        output_path = tmp_path / "output.csv"
        molecule_dataframe.chem.write_molecule_csv(output_path, molecule_column="Molecule")
        assert output_path.exists()

    def test_write_molecule_csv_readable(self, molecule_dataframe, tmp_path):
        output_path = tmp_path / "output.csv"
        molecule_dataframe.chem.write_molecule_csv(output_path, molecule_column="Molecule")

        # Read it back as regular CSV
        df_back = pl.read_csv(output_path)
        assert len(df_back) == 2
        assert "smiles" in df_back.columns  # Default SMILES column name

    def test_write_molecule_csv_custom_smiles_column(self, molecule_dataframe, tmp_path):
        output_path = tmp_path / "output.csv"
        molecule_dataframe.chem.write_molecule_csv(
            output_path,
            molecule_column="Molecule",
            smiles_column="SMILES"
        )

        df_back = pl.read_csv(output_path)
        assert "SMILES" in df_back.columns

    def test_write_molecule_csv_preserves_other_columns(self, molecule_dataframe, tmp_path):
        output_path = tmp_path / "output.csv"
        molecule_dataframe.chem.write_molecule_csv(output_path, molecule_column="Molecule")

        df_back = pl.read_csv(output_path)
        assert "name" in df_back.columns
        assert "MW" in df_back.columns
        assert df_back["name"].to_list() == ["ethanol", "benzene"]

    def test_write_molecule_csv_invalid_molecule_column_raises(self, molecule_dataframe, tmp_path):
        output_path = tmp_path / "output.csv"
        with pytest.raises(KeyError):
            molecule_dataframe.chem.write_molecule_csv(output_path, molecule_column="nonexistent")

    def test_write_molecule_csv_wrong_column_type_raises(self, tmp_path):
        df = pl.DataFrame({"x": [1, 2, 3]})
        output_path = tmp_path / "output.csv"
        with pytest.raises(TypeError):
            df.chem.write_molecule_csv(output_path, molecule_column="x")


class TestDataFrameWriteOEB:
    @pytest.fixture
    def molecule_dataframe(self):
        mols = []
        for smiles in ["CCO", "c1ccccc1"]:
            mol = oechem.OEMol()
            oechem.OEParseSmiles(mol, smiles)
            mols.append(mol)
        return pl.DataFrame({
            "Molecule": pl.Series(mols, dtype=MoleculeType()),
            "name": ["ethanol", "benzene"],
            "MW": [46.07, 78.11],
        })

    def test_write_oeb_creates_file(self, molecule_dataframe, tmp_path):
        output_path = tmp_path / "output.oeb"
        molecule_dataframe.chem.write_oeb(output_path, molecule_column="Molecule")
        assert output_path.exists()

    def test_write_oeb_readable(self, molecule_dataframe, tmp_path):
        output_path = tmp_path / "output.oeb"
        molecule_dataframe.chem.write_oeb(output_path, molecule_column="Molecule")

        import oepolars as oep
        df_back = oep.read_oeb(output_path)
        assert len(df_back) == 2

    def test_write_oeb_with_sd_data(self, molecule_dataframe, tmp_path):
        output_path = tmp_path / "output.oeb"
        molecule_dataframe.chem.write_oeb(
            output_path,
            molecule_column="Molecule",
            sd_columns=["name", "MW"]
        )

        import oepolars as oep
        df_back = oep.read_oeb(output_path)
        assert "name" in df_back.columns
        assert "MW" in df_back.columns

    def test_write_oeb_with_title_column(self, molecule_dataframe, tmp_path):
        output_path = tmp_path / "output.oeb"
        molecule_dataframe.chem.write_oeb(
            output_path,
            molecule_column="Molecule",
            title_column="name"
        )

        import oepolars as oep
        df_back = oep.read_oeb(output_path)
        assert df_back["Title"].to_list() == ["ethanol", "benzene"]

    def test_write_oeb_invalid_molecule_column_raises(self, molecule_dataframe, tmp_path):
        output_path = tmp_path / "output.oeb"
        with pytest.raises(KeyError):
            molecule_dataframe.chem.write_oeb(output_path, molecule_column="nonexistent")

    def test_write_oeb_wrong_column_type_raises(self, tmp_path):
        df = pl.DataFrame({"x": [1, 2, 3]})
        output_path = tmp_path / "output.oeb"
        with pytest.raises(TypeError):
            df.chem.write_oeb(output_path, molecule_column="x")


class TestDataFrameWriteParquet:
    @pytest.fixture
    def molecule_dataframe(self):
        mols = []
        for smiles in ["CCO", "c1ccccc1"]:
            mol = oechem.OEMol()
            oechem.OEParseSmiles(mol, smiles)
            mols.append(mol)
        return pl.DataFrame({
            "Molecule": pl.Series(mols, dtype=MoleculeType()),
            "name": ["ethanol", "benzene"],
            "MW": [46.07, 78.11],
        })

    def test_write_parquet_creates_file(self, molecule_dataframe, tmp_path):
        """Verify that write_parquet creates a file."""
        output_path = tmp_path / "output.parquet"
        molecule_dataframe.chem.write_parquet(output_path)
        assert output_path.exists()

    def test_write_parquet_serializes_molecules(self, molecule_dataframe, tmp_path):
        """Verify molecules are converted to binary bytes."""
        output_path = tmp_path / "output.parquet"
        molecule_dataframe.chem.write_parquet(output_path)

        # Read back the parquet file
        df_back = pl.read_parquet(output_path)

        # Molecule column should be Binary type
        assert df_back.schema["Molecule"] == pl.Binary

        # Verify we can deserialize back to molecules
        from oepolars.util import bytes_to_mol
        mol_bytes = df_back["Molecule"][0]
        assert isinstance(mol_bytes, bytes)
        mol = bytes_to_mol(mol_bytes)
        assert mol is not None
        assert mol.NumAtoms() > 0

    def test_write_parquet_preserves_other_columns(self, molecule_dataframe, tmp_path):
        """Verify non-molecule columns are unchanged."""
        output_path = tmp_path / "output.parquet"
        molecule_dataframe.chem.write_parquet(output_path)

        df_back = pl.read_parquet(output_path)

        # Non-molecule columns should be preserved
        assert df_back["name"].to_list() == ["ethanol", "benzene"]
        assert df_back["MW"].to_list() == [46.07, 78.11]

    def test_write_parquet_handles_nulls(self, tmp_path):
        """Verify null molecules become null bytes."""
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")
        df = pl.DataFrame({
            "Molecule": pl.Series([mol, None], dtype=MoleculeType()),
            "name": ["ethanol", "none"],
        })

        output_path = tmp_path / "output.parquet"
        df.chem.write_parquet(output_path)

        df_back = pl.read_parquet(output_path)
        assert df_back["Molecule"][0] is not None  # First mol serialized
        assert df_back["Molecule"][1] is None  # Null stays null

    def test_write_parquet_auto_detects_molecule_columns(self, tmp_path):
        """Verify auto-detection of molecule columns works."""
        mol1 = oechem.OEMol()
        oechem.OEParseSmiles(mol1, "CCO")
        mol2 = oechem.OEMol()
        oechem.OEParseSmiles(mol2, "c1ccccc1")

        df = pl.DataFrame({
            "Mol1": pl.Series([mol1, mol2], dtype=MoleculeType()),
            "Mol2": pl.Series([mol2, mol1], dtype=MoleculeType()),
            "name": ["a", "b"],
        })

        output_path = tmp_path / "output.parquet"
        # Don't specify molecule_columns - should auto-detect both
        df.chem.write_parquet(output_path)

        df_back = pl.read_parquet(output_path)
        # Both molecule columns should be serialized to Binary
        assert df_back.schema["Mol1"] == pl.Binary
        assert df_back.schema["Mol2"] == pl.Binary
        assert df_back.schema["name"] == pl.String

    def test_write_parquet_explicit_molecule_columns(self, molecule_dataframe, tmp_path):
        """Verify explicit column specification works."""
        output_path = tmp_path / "output.parquet"
        # Explicitly specify the molecule column
        molecule_dataframe.chem.write_parquet(output_path, molecule_columns=["Molecule"])

        df_back = pl.read_parquet(output_path)
        # Molecule column should be Binary
        assert df_back.schema["Molecule"] == pl.Binary
        # Other columns preserved
        assert df_back["name"].to_list() == ["ethanol", "benzene"]

    def test_write_parquet_passes_kwargs(self, molecule_dataframe, tmp_path):
        """Verify parquet kwargs are passed through."""
        output_path = tmp_path / "output.parquet"
        # Test with compression kwarg
        molecule_dataframe.chem.write_parquet(output_path, compression="snappy")
        assert output_path.exists()

    def test_write_parquet_invalid_column_raises(self, molecule_dataframe, tmp_path):
        """Verify KeyError when non-existent column specified."""
        output_path = tmp_path / "output.parquet"
        with pytest.raises(KeyError):
            molecule_dataframe.chem.write_parquet(output_path, molecule_columns=["nonexistent"])

    def test_write_parquet_wrong_column_type_raises(self, tmp_path):
        """Verify TypeError when non-MoleculeType column specified."""
        df = pl.DataFrame({"x": [1, 2, 3]})
        output_path = tmp_path / "output.parquet"
        with pytest.raises(TypeError):
            df.chem.write_parquet(output_path, molecule_columns=["x"])


class TestDataFrameWriteOEDB:
    @pytest.fixture
    def molecule_dataframe(self):
        mols = []
        for smiles in ["CCO", "c1ccccc1"]:
            mol = oechem.OEMol()
            oechem.OEParseSmiles(mol, smiles)
            mols.append(mol)
        return pl.DataFrame({
            "Molecule": pl.Series(mols, dtype=MoleculeType()),
            "name": ["ethanol", "benzene"],
            "MW": [46.07, 78.11],
        })

    def test_write_oedb_creates_file(self, molecule_dataframe, tmp_path):
        """Verify that write_oedb creates a file."""
        output_path = tmp_path / "output.oedb"
        molecule_dataframe.chem.write_oedb(output_path, molecule_column="Molecule")
        assert output_path.exists()

    def test_write_oedb_readable(self, molecule_dataframe, tmp_path):
        """Verify OEDB file can be read back."""
        output_path = tmp_path / "output.oedb"
        molecule_dataframe.chem.write_oedb(output_path, molecule_column="Molecule")

        import oepolars as oep
        df_back = oep.read_oedb(output_path)
        assert len(df_back) == 2

    def test_write_oedb_with_sd_data(self, molecule_dataframe, tmp_path):
        """Verify SD data columns are written."""
        output_path = tmp_path / "output.oedb"
        molecule_dataframe.chem.write_oedb(
            output_path,
            molecule_column="Molecule",
            sd_columns=["name", "MW"]
        )

        import oepolars as oep
        df_back = oep.read_oedb(output_path)
        assert "name" in df_back.columns
        assert "MW" in df_back.columns

    def test_write_oedb_with_title_column(self, molecule_dataframe, tmp_path):
        """Verify title column is written."""
        output_path = tmp_path / "output.oedb"
        molecule_dataframe.chem.write_oedb(
            output_path,
            molecule_column="Molecule",
            title_column="name"
        )

        import oepolars as oep
        df_back = oep.read_oedb(output_path)
        assert df_back["Title"].to_list() == ["ethanol", "benzene"]

    def test_write_oedb_invalid_column_raises(self, molecule_dataframe, tmp_path):
        """Verify KeyError for nonexistent column."""
        output_path = tmp_path / "output.oedb"
        with pytest.raises(KeyError):
            molecule_dataframe.chem.write_oedb(output_path, molecule_column="nonexistent")

    def test_write_oedb_wrong_column_type_raises(self, tmp_path):
        """Verify TypeError for wrong column type."""
        df = pl.DataFrame({"x": [1, 2, 3]})
        output_path = tmp_path / "output.oedb"
        with pytest.raises(TypeError):
            df.chem.write_oedb(output_path, molecule_column="x")


class TestDataFrameWriteOEDU:
    @pytest.fixture
    def design_unit_dataframe(self):
        from oepolars.types import DesignUnitType
        # Create simple design units
        dus = []
        for title in ["complex1", "complex2"]:
            du = oechem.OEDesignUnit()
            du.SetTitle(title)
            dus.append(du)
        return pl.DataFrame({
            "DesignUnit": pl.Series(dus, dtype=DesignUnitType()),
            "name": ["complex1", "complex2"],
        })

    def test_write_oedu_creates_file(self, design_unit_dataframe, tmp_path):
        """Verify that write_oedu creates a file."""
        output_path = tmp_path / "output.oedu"
        design_unit_dataframe.chem.write_oedu(output_path, design_unit_column="DesignUnit")
        assert output_path.exists()

    def test_write_oedu_readable(self, design_unit_dataframe, tmp_path):
        """Verify OEDU file can be read back."""
        output_path = tmp_path / "output.oedu"
        design_unit_dataframe.chem.write_oedu(output_path, design_unit_column="DesignUnit")

        import oepolars as oep
        df_back = oep.read_oedu(output_path)
        assert len(df_back) == 2

    def test_write_oedu_with_title_column(self, design_unit_dataframe, tmp_path):
        """Verify title column is written."""
        output_path = tmp_path / "output.oedu"
        design_unit_dataframe.chem.write_oedu(
            output_path,
            design_unit_column="DesignUnit",
            title_column="name"
        )

        import oepolars as oep
        df_back = oep.read_oedu(output_path)
        assert df_back["Title"].to_list() == ["complex1", "complex2"]

    def test_write_oedu_invalid_column_raises(self, design_unit_dataframe, tmp_path):
        """Verify KeyError for nonexistent column."""
        output_path = tmp_path / "output.oedu"
        with pytest.raises(KeyError):
            design_unit_dataframe.chem.write_oedu(output_path, design_unit_column="nonexistent")

    def test_write_oedu_wrong_column_type_raises(self, tmp_path):
        """Verify TypeError for wrong column type."""
        df = pl.DataFrame({"x": [1, 2, 3]})
        output_path = tmp_path / "output.oedu"
        with pytest.raises(TypeError):
            df.chem.write_oedu(output_path, design_unit_column="x")
