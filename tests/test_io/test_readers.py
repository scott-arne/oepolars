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
        import tempfile
        import os
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