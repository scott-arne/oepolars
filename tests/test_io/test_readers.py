"""Tests for file readers."""

from pathlib import Path

import polars as pl
import pytest

import oepolars as oep

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestReadSDF:
    def test_read_sdf_returns_dataframe(self):
        df = oep.read_sdf(FIXTURES_DIR / "sample.sdf")
        assert isinstance(df, pl.DataFrame)

    def test_read_sdf_has_molecule_column(self):
        df = oep.read_sdf(FIXTURES_DIR / "sample.sdf")
        assert "Molecule" in df.columns
        assert df.schema["Molecule"].ext_name() == "molecule"

    def test_read_sdf_has_title_column(self):
        df = oep.read_sdf(FIXTURES_DIR / "sample.sdf")
        assert "Title" in df.columns
        assert df["Title"].to_list() == ["ethanol", "benzene"]

    def test_read_sdf_reads_sd_data(self):
        df = oep.read_sdf(FIXTURES_DIR / "sample.sdf")
        assert "MW" in df.columns
        assert "LogP" in df.columns

    def test_read_sdf_custom_molecule_column_name(self):
        df = oep.read_sdf(FIXTURES_DIR / "sample.sdf", molecule_column="Mol")
        assert "Mol" in df.columns
        assert df.schema["Mol"].ext_name() == "molecule"

    def test_read_sdf_no_title_column(self):
        df = oep.read_sdf(FIXTURES_DIR / "sample.sdf", title_column=None)
        assert "Title" not in df.columns

    def test_read_sdf_file_not_found_raises(self):
        from oepolars.exceptions import FileError
        with pytest.raises(FileError):
            oep.read_sdf("nonexistent.sdf")

    def test_read_sdf_molecule_count(self):
        df = oep.read_sdf(FIXTURES_DIR / "sample.sdf")
        assert len(df) == 2

    def test_read_sdf_empty_file(self):
        df = oep.read_sdf(FIXTURES_DIR / "empty.sdf")
        assert len(df) == 0
        assert "Molecule" in df.columns


class TestReadOEB:
    @pytest.fixture
    def oeb_file(self, tmp_path):
        """Create a temporary OEB file for testing."""
        from openeye import oechem

        oeb_path = tmp_path / "test.oeb"
        ofs = oechem.oemolostream(str(oeb_path))

        for smiles, title in [("CCO", "ethanol"), ("c1ccccc1", "benzene")]:
            mol = oechem.OEMol()
            oechem.OEParseSmiles(mol, smiles)
            mol.SetTitle(title)
            oechem.OESetSDData(mol, "MW", "46.07" if smiles == "CCO" else "78.11")
            oechem.OEWriteMolecule(ofs, mol)

        ofs.close()
        return oeb_path

    def test_read_oeb_returns_dataframe(self, oeb_file):
        df = oep.read_oeb(oeb_file)
        assert isinstance(df, pl.DataFrame)

    def test_read_oeb_has_molecule_column(self, oeb_file):
        df = oep.read_oeb(oeb_file)
        assert "Molecule" in df.columns
        assert df.schema["Molecule"].ext_name() == "molecule"

    def test_read_oeb_has_title_column(self, oeb_file):
        df = oep.read_oeb(oeb_file)
        assert "Title" in df.columns
        assert df["Title"].to_list() == ["ethanol", "benzene"]

    def test_read_oeb_reads_sd_data(self, oeb_file):
        df = oep.read_oeb(oeb_file)
        assert "MW" in df.columns

    def test_read_oeb_custom_molecule_column_name(self, oeb_file):
        df = oep.read_oeb(oeb_file, molecule_column="Mol")
        assert "Mol" in df.columns
        assert df.schema["Mol"].ext_name() == "molecule"

    def test_read_oeb_file_not_found_raises(self):
        from oepolars.exceptions import FileError

        with pytest.raises(FileError):
            oep.read_oeb("nonexistent.oeb")

    def test_read_oeb_molecule_count(self, oeb_file):
        df = oep.read_oeb(oeb_file)
        assert len(df) == 2

    def test_read_oeb_empty_file(self, tmp_path):
        """Empty OEB file returns empty DataFrame."""
        from openeye import oechem

        oeb_path = tmp_path / "empty.oeb"
        ofs = oechem.oemolostream(str(oeb_path))
        ofs.close()

        df = oep.read_oeb(oeb_path)
        assert len(df) == 0
        assert "Molecule" in df.columns

    def test_read_oeb_no_title_column(self, oeb_file):
        df = oep.read_oeb(oeb_file, title_column=None)
        assert "Title" not in df.columns


class TestReadSMI:
    def test_read_smi_returns_dataframe(self):
        df = oep.read_smi(FIXTURES_DIR / "sample.smi")
        assert isinstance(df, pl.DataFrame)

    def test_read_smi_has_molecule_column(self):
        df = oep.read_smi(FIXTURES_DIR / "sample.smi")
        assert "Molecule" in df.columns
        assert df.schema["Molecule"].ext_name() == "molecule"

    def test_read_smi_has_title_column(self):
        df = oep.read_smi(FIXTURES_DIR / "sample.smi")
        assert "Title" in df.columns
        assert df["Title"].to_list() == ["ethanol", "benzene", "acetic_acid"]

    def test_read_smi_custom_molecule_column_name(self):
        df = oep.read_smi(FIXTURES_DIR / "sample.smi", molecule_column="Mol")
        assert "Mol" in df.columns
        assert df.schema["Mol"].ext_name() == "molecule"

    def test_read_smi_no_title_column(self):
        df = oep.read_smi(FIXTURES_DIR / "sample.smi", title_column=None)
        assert "Title" not in df.columns

    def test_read_smi_file_not_found_raises(self):
        from oepolars.exceptions import FileError
        with pytest.raises(FileError):
            oep.read_smi("nonexistent.smi")

    def test_read_smi_molecule_count(self):
        df = oep.read_smi(FIXTURES_DIR / "sample.smi")
        assert len(df) == 3

    def test_read_smi_empty_file(self):
        import os
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".smi", delete=False) as f:
            pass
        df = oep.read_smi(f.name)
        assert len(df) == 0
        assert "Molecule" in df.columns
        os.unlink(f.name)


class TestReadMoleculeCSV:
    def test_read_molecule_csv_returns_dataframe(self):
        df = oep.read_molecule_csv(FIXTURES_DIR / "sample.csv", smiles_column="smiles")
        assert isinstance(df, pl.DataFrame)

    def test_read_molecule_csv_has_molecule_column(self):
        df = oep.read_molecule_csv(FIXTURES_DIR / "sample.csv", smiles_column="smiles")
        assert "Molecule" in df.columns
        assert df.schema["Molecule"].ext_name() == "molecule"

    def test_read_molecule_csv_custom_molecule_column_name(self):
        df = oep.read_molecule_csv(FIXTURES_DIR / "sample.csv", smiles_column="smiles", molecule_column="Mol")
        assert "Mol" in df.columns
        assert df.schema["Mol"].ext_name() == "molecule"

    def test_read_molecule_csv_preserves_other_columns(self):
        df = oep.read_molecule_csv(FIXTURES_DIR / "sample.csv", smiles_column="smiles")
        assert "name" in df.columns
        assert "MW" in df.columns
        assert df["name"].to_list() == ["ethanol", "benzene", "acetic_acid"]

    def test_read_molecule_csv_removes_smiles_column(self):
        df = oep.read_molecule_csv(FIXTURES_DIR / "sample.csv", smiles_column="smiles")
        assert "smiles" not in df.columns

    def test_read_molecule_csv_keep_smiles_column(self):
        df = oep.read_molecule_csv(FIXTURES_DIR / "sample.csv", smiles_column="smiles", drop_smiles=False)
        assert "smiles" in df.columns

    def test_read_molecule_csv_file_not_found_raises(self):
        from oepolars.exceptions import FileError
        with pytest.raises(FileError):
            oep.read_molecule_csv("nonexistent.csv", smiles_column="smiles")

    def test_read_molecule_csv_molecule_count(self):
        df = oep.read_molecule_csv(FIXTURES_DIR / "sample.csv", smiles_column="smiles")
        assert len(df) == 3


class TestReadOEDB:
    @pytest.fixture
    def oedb_file(self, tmp_path):
        """Create a temporary OEDB file for testing.

        Note: We use .oeb extension since OpenEye's oemolistream auto-detects
        format from extension, and .oedb is not a standard recognized extension.
        The read_oedb function works with any format that oemolistream supports.
        """
        from openeye import oechem

        oedb_path = tmp_path / "test.oeb"
        ofs = oechem.oemolostream(str(oedb_path))

        for smiles, title in [("CCO", "ethanol"), ("c1ccccc1", "benzene")]:
            mol = oechem.OEMol()
            oechem.OEParseSmiles(mol, smiles)
            mol.SetTitle(title)
            oechem.OESetSDData(mol, "MW", "46.07" if smiles == "CCO" else "78.11")
            oechem.OEWriteMolecule(ofs, mol)

        ofs.close()
        return oedb_path

    def test_read_oedb_returns_dataframe(self, oedb_file):
        df = oep.read_oedb(oedb_file)
        assert isinstance(df, pl.DataFrame)

    def test_read_oedb_has_molecule_column(self, oedb_file):
        df = oep.read_oedb(oedb_file)
        assert "Molecule" in df.columns
        assert df.schema["Molecule"].ext_name() == "molecule"

    def test_read_oedb_has_title_column(self, oedb_file):
        df = oep.read_oedb(oedb_file)
        assert "Title" in df.columns
        assert df["Title"].to_list() == ["ethanol", "benzene"]

    def test_read_oedb_reads_sd_data(self, oedb_file):
        df = oep.read_oedb(oedb_file)
        assert "MW" in df.columns

    def test_read_oedb_custom_molecule_column_name(self, oedb_file):
        df = oep.read_oedb(oedb_file, molecule_column="Mol")
        assert "Mol" in df.columns

    def test_read_oedb_file_not_found_raises(self):
        from oepolars.exceptions import FileError
        with pytest.raises(FileError):
            oep.read_oedb("nonexistent.oedb")

    def test_read_oedb_molecule_count(self, oedb_file):
        df = oep.read_oedb(oedb_file)
        assert len(df) == 2

    def test_read_oedb_no_title_column(self, oedb_file):
        df = oep.read_oedb(oedb_file, title_column=None)
        assert "Title" not in df.columns

    def test_read_oedb_empty_file(self, tmp_path):
        """Empty OEDB file returns empty DataFrame."""
        from openeye import oechem

        oedb_path = tmp_path / "empty.oeb"
        ofs = oechem.oemolostream(str(oedb_path))
        ofs.close()

        df = oep.read_oedb(oedb_path)
        assert len(df) == 0
        assert "Molecule" in df.columns


class TestReadOEDU:
    @pytest.fixture
    def oedu_file(self, tmp_path):
        """Create a temporary OEDU file for testing."""
        from openeye import oechem

        oedu_path = tmp_path / "test.oedu"
        ofs = oechem.oeofstream(str(oedu_path))

        # Create a simple design unit
        du = oechem.OEDesignUnit()
        du.SetTitle("test_du")
        oechem.OEWriteDesignUnit(ofs, du)

        du2 = oechem.OEDesignUnit()
        du2.SetTitle("test_du_2")
        oechem.OEWriteDesignUnit(ofs, du2)

        ofs.close()
        return oedu_path

    def test_read_oedu_returns_dataframe(self, oedu_file):
        df = oep.read_oedu(oedu_file)
        assert isinstance(df, pl.DataFrame)

    def test_read_oedu_has_design_unit_column(self, oedu_file):
        df = oep.read_oedu(oedu_file)
        assert "DesignUnit" in df.columns
        assert df.schema["DesignUnit"].ext_name() == "design_unit"

    def test_read_oedu_has_title_column(self, oedu_file):
        df = oep.read_oedu(oedu_file)
        assert "Title" in df.columns
        assert df["Title"].to_list() == ["test_du", "test_du_2"]

    def test_read_oedu_custom_du_column_name(self, oedu_file):
        df = oep.read_oedu(oedu_file, design_unit_column="DU")
        assert "DU" in df.columns
        assert df.schema["DU"].ext_name() == "design_unit"

    def test_read_oedu_no_title_column(self, oedu_file):
        df = oep.read_oedu(oedu_file, title_column=None)
        assert "Title" not in df.columns

    def test_read_oedu_file_not_found_raises(self):
        from oepolars.exceptions import FileError
        with pytest.raises(FileError):
            oep.read_oedu("nonexistent.oedu")

    def test_read_oedu_design_unit_count(self, oedu_file):
        df = oep.read_oedu(oedu_file)
        assert len(df) == 2

    def test_read_oedu_empty_file(self, tmp_path):
        """Empty OEDU file returns empty DataFrame."""
        from openeye import oechem

        oedu_path = tmp_path / "empty.oedu"
        ofs = oechem.oeofstream(str(oedu_path))
        ofs.close()

        df = oep.read_oedu(oedu_path)
        assert len(df) == 0
        assert "DesignUnit" in df.columns


class TestReadParquet:
    """Tests for read_parquet function that reconstructs molecules from binary."""

    @pytest.fixture
    def molecule_dataframe(self):
        """Create a DataFrame with molecule columns for testing."""
        from openeye import oechem

        from oepolars import MoleculeType

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

    def test_read_parquet_reconstructs_molecules(self, molecule_dataframe, tmp_path):
        """Test roundtrip: write then read back reconstructs molecules."""
        from openeye import oechem

        parquet_path = tmp_path / "test.parquet"
        molecule_dataframe.chem.write_parquet(parquet_path)

        # Read back with molecule_columns specified
        df_back = oep.read_parquet(parquet_path, molecule_columns=["Molecule"])

        # Check that the Molecule column is MoleculeType
        assert df_back.schema["Molecule"].ext_name() == "molecule"

        # Check that we can access the molecules
        mol = df_back["Molecule"][0]
        assert mol is not None
        assert isinstance(mol, oechem.OEMolBase)
        assert mol.NumAtoms() > 0

    def test_read_parquet_preserves_other_columns(self, molecule_dataframe, tmp_path):
        """Test that non-molecule columns are unchanged."""
        parquet_path = tmp_path / "test.parquet"
        molecule_dataframe.chem.write_parquet(parquet_path)

        df_back = oep.read_parquet(parquet_path, molecule_columns=["Molecule"])

        # Verify non-molecule columns preserved
        assert df_back["name"].to_list() == ["ethanol", "benzene"]
        assert df_back["MW"].to_list() == [46.07, 78.11]

    def test_read_parquet_handles_nulls(self, tmp_path):
        """Test that null bytes become null molecules."""
        from openeye import oechem

        from oepolars import MoleculeType

        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")
        df = pl.DataFrame({
            "Molecule": pl.Series([mol, None], dtype=MoleculeType()),
            "name": ["ethanol", "none"],
        })

        parquet_path = tmp_path / "test.parquet"
        df.chem.write_parquet(parquet_path)

        df_back = oep.read_parquet(parquet_path, molecule_columns=["Molecule"])

        # First molecule should be reconstructed
        assert df_back["Molecule"][0] is not None
        # Second molecule should be null
        assert df_back["Molecule"][1] is None

    def test_read_parquet_with_multiple_molecule_columns(self, tmp_path):
        """Test that multiple molecule columns work correctly."""
        from openeye import oechem

        from oepolars import MoleculeType

        mol1 = oechem.OEMol()
        oechem.OEParseSmiles(mol1, "CCO")
        mol2 = oechem.OEMol()
        oechem.OEParseSmiles(mol2, "c1ccccc1")

        df = pl.DataFrame({
            "Mol1": pl.Series([mol1, mol2], dtype=MoleculeType()),
            "Mol2": pl.Series([mol2, mol1], dtype=MoleculeType()),
            "name": ["a", "b"],
        })

        parquet_path = tmp_path / "test.parquet"
        df.chem.write_parquet(parquet_path)

        df_back = oep.read_parquet(parquet_path, molecule_columns=["Mol1", "Mol2"])

        # Both columns should be MoleculeType
        assert df_back.schema["Mol1"].ext_name() == "molecule"
        assert df_back.schema["Mol2"].ext_name() == "molecule"

        # Molecules should be accessible
        assert df_back["Mol1"][0] is not None
        assert df_back["Mol2"][0] is not None

    def test_read_parquet_passes_kwargs(self, molecule_dataframe, tmp_path):
        """Test that kwargs are passed through to pl.read_parquet."""
        parquet_path = tmp_path / "test.parquet"
        molecule_dataframe.chem.write_parquet(parquet_path)

        # Test with columns kwarg to read only specific columns
        df_back = oep.read_parquet(
            parquet_path,
            molecule_columns=["Molecule"],
            columns=["Molecule", "name"],  # Exclude MW column
        )

        assert "Molecule" in df_back.columns
        assert "name" in df_back.columns
        assert "MW" not in df_back.columns

    def test_read_parquet_requires_molecule_columns(self, molecule_dataframe, tmp_path):
        """Test that molecule_columns parameter is required."""
        parquet_path = tmp_path / "test.parquet"
        molecule_dataframe.chem.write_parquet(parquet_path)

        # Without molecule_columns, should raise an error
        with pytest.raises(ValueError):
            oep.read_parquet(parquet_path)

    def test_read_parquet_empty_molecule_columns(self, molecule_dataframe, tmp_path):
        """Test with empty molecule_columns list reads as regular parquet."""
        parquet_path = tmp_path / "test.parquet"
        molecule_dataframe.chem.write_parquet(parquet_path)

        # With empty molecule_columns, should read but Molecule stays Binary
        df_back = oep.read_parquet(parquet_path, molecule_columns=[])

        # Molecule should still be Binary (not converted)
        assert df_back.schema["Molecule"] == pl.Binary

    def test_read_parquet_file_not_found_raises(self):
        from oepolars.exceptions import FileError
        with pytest.raises(FileError):
            oep.read_parquet("nonexistent.parquet", molecule_columns=["Molecule"])

    def test_read_parquet_invalid_column_raises(self, molecule_dataframe, tmp_path):
        parquet_path = tmp_path / "test.parquet"
        molecule_dataframe.chem.write_parquet(parquet_path)
        with pytest.raises(KeyError, match="NonExistent"):
            oep.read_parquet(parquet_path, molecule_columns=["NonExistent"])

    def test_read_parquet_empty_file(self, tmp_path):
        # Create empty DataFrame with molecule column
        from oepolars import MoleculeType
        df = pl.DataFrame({"Molecule": pl.Series([], dtype=MoleculeType())})
        parquet_path = tmp_path / "empty.parquet"
        df.chem.write_parquet(parquet_path)
        result = oep.read_parquet(parquet_path, molecule_columns=["Molecule"])
        assert len(result) == 0
