"""Tests for lazy/scan file readers."""

import inspect
from pathlib import Path

import polars as pl
import pytest
from openeye import oechem

import oepolars as oep

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestScanSDF:
    def test_scan_sdf_returns_lazyframe(self):
        lf = oep.scan_sdf(FIXTURES_DIR / "sample.sdf")
        assert isinstance(lf, pl.LazyFrame)

    def test_scan_sdf_collect_returns_dataframe(self):
        lf = oep.scan_sdf(FIXTURES_DIR / "sample.sdf")
        df = lf.collect()
        assert isinstance(df, pl.DataFrame)

    def test_scan_sdf_has_molecule_column(self):
        lf = oep.scan_sdf(FIXTURES_DIR / "sample.sdf")
        df = lf.collect()
        assert "Molecule" in df.columns
        assert df.schema["Molecule"].ext_name() == "molecule"

    def test_scan_sdf_has_title_column(self):
        lf = oep.scan_sdf(FIXTURES_DIR / "sample.sdf")
        df = lf.collect()
        assert "Title" in df.columns

    def test_scan_sdf_reads_sd_data(self):
        lf = oep.scan_sdf(FIXTURES_DIR / "sample.sdf")
        df = lf.collect()
        assert "MW" in df.columns
        assert "LogP" in df.columns

    def test_scan_sdf_supports_select(self):
        lf = oep.scan_sdf(FIXTURES_DIR / "sample.sdf")
        # Select only Title column (lazy operation)
        lf_selected = lf.select("Title")
        df = lf_selected.collect()
        assert list(df.columns) == ["Title"]

    def test_scan_sdf_supports_filter(self):
        lf = oep.scan_sdf(FIXTURES_DIR / "sample.sdf")
        # Filter operation (lazy)
        lf_filtered = lf.filter(pl.col("Title") == "ethanol")
        df = lf_filtered.collect()
        assert len(df) == 1

    def test_scan_sdf_file_not_found_raises(self):
        from oepolars.exceptions import FileError
        with pytest.raises(FileError):
            oep.scan_sdf("nonexistent.sdf")

    def test_scan_sdf_molecule_count(self):
        lf = oep.scan_sdf(FIXTURES_DIR / "sample.sdf")
        df = lf.collect()
        assert len(df) == 2


class TestScanOEB:
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

    def test_scan_oeb_returns_lazyframe(self, oeb_file):
        lf = oep.scan_oeb(oeb_file)
        assert isinstance(lf, pl.LazyFrame)

    def test_scan_oeb_collect_returns_dataframe(self, oeb_file):
        lf = oep.scan_oeb(oeb_file)
        df = lf.collect()
        assert isinstance(df, pl.DataFrame)

    def test_scan_oeb_has_molecule_column(self, oeb_file):
        lf = oep.scan_oeb(oeb_file)
        df = lf.collect()
        assert "Molecule" in df.columns
        assert df.schema["Molecule"].ext_name() == "molecule"

    def test_scan_oeb_supports_filter(self, oeb_file):
        lf = oep.scan_oeb(oeb_file)
        lf_filtered = lf.filter(pl.col("Title") == "ethanol")
        df = lf_filtered.collect()
        assert len(df) == 1

    def test_scan_oeb_file_not_found_raises(self):
        from oepolars.exceptions import FileError

        with pytest.raises(FileError):
            oep.scan_oeb("nonexistent.oeb")


class TestScanSMI:
    def test_scan_smi_returns_lazyframe(self):
        lf = oep.scan_smi(FIXTURES_DIR / "sample.smi")
        assert isinstance(lf, pl.LazyFrame)

    def test_scan_smi_collect_returns_dataframe(self):
        lf = oep.scan_smi(FIXTURES_DIR / "sample.smi")
        df = lf.collect()
        assert isinstance(df, pl.DataFrame)

    def test_scan_smi_has_molecule_column(self):
        lf = oep.scan_smi(FIXTURES_DIR / "sample.smi")
        df = lf.collect()
        assert "Molecule" in df.columns
        assert df.schema["Molecule"].ext_name() == "molecule"

    def test_scan_smi_molecule_type_class_returns_graph_mols(self):
        lf = oep.scan_smi(
            FIXTURES_DIR / "sample.smi",
            molecule_type=oechem.OEGraphMol,
        )
        df = lf.collect()

        assert all(
            isinstance(mol, oechem.OEGraphMol)
            for mol in df["Molecule"].to_list()
        )

    def test_scan_smi_has_title_column(self):
        lf = oep.scan_smi(FIXTURES_DIR / "sample.smi")
        df = lf.collect()
        assert "Title" in df.columns

    def test_scan_smi_no_title_clears_molecule_titles(self):
        lf = oep.scan_smi(FIXTURES_DIR / "sample.smi", no_title=True)
        df = lf.collect()

        assert "Title" not in df.columns
        assert all(mol.GetTitle() == "" for mol in df["Molecule"].to_list())
        assert all(
            mol.GetActive().GetTitle() == ""
            for mol in df["Molecule"].to_list()
        )

    def test_scan_smi_supports_filter(self):
        lf = oep.scan_smi(FIXTURES_DIR / "sample.smi")
        lf_filtered = lf.filter(pl.col("Title") == "ethanol")
        df = lf_filtered.collect()
        assert len(df) == 1

    def test_scan_smi_file_not_found_raises(self):
        from oepolars.exceptions import FileError

        with pytest.raises(FileError):
            oep.scan_smi("nonexistent.smi")


class TestScanParquet:
    @pytest.fixture
    def parquet_file(self, tmp_path):
        """Create a temporary Parquet file with molecules for testing."""
        # Create a DataFrame with molecules and write to Parquet
        df = oep.read_sdf(FIXTURES_DIR / "sample.sdf")
        parquet_path = tmp_path / "test.parquet"
        df.chem.write_parquet(parquet_path)
        return parquet_path

    def test_scan_parquet_returns_lazyframe(self, parquet_file):
        lf = oep.scan_parquet(parquet_file, molecule_columns=["Molecule"])
        assert isinstance(lf, pl.LazyFrame)

    def test_scan_parquet_collects_to_dataframe(self, parquet_file):
        lf = oep.scan_parquet(parquet_file, molecule_columns=["Molecule"])
        df = lf.collect()
        assert isinstance(df, pl.DataFrame)

    def test_scan_parquet_has_molecule_column(self, parquet_file):
        lf = oep.scan_parquet(parquet_file, molecule_columns=["Molecule"])
        df = lf.collect()
        assert "Molecule" in df.columns
        assert df.schema["Molecule"].ext_name() == "molecule"

    def test_scan_parquet_preserves_data(self, parquet_file):
        lf = oep.scan_parquet(parquet_file, molecule_columns=["Molecule"])
        df = lf.collect()
        assert len(df) == 2
        assert "Title" in df.columns

    def test_scan_parquet_supports_filter(self, parquet_file):
        lf = oep.scan_parquet(parquet_file, molecule_columns=["Molecule"])
        lf_filtered = lf.filter(pl.col("Title") == "ethanol")
        df = lf_filtered.collect()
        assert len(df) == 1

    def test_scan_parquet_file_not_found_raises(self):
        from oepolars.exceptions import FileError

        with pytest.raises(FileError):
            oep.scan_parquet("nonexistent.parquet", molecule_columns=["Molecule"])


class TestScanOEDB:
    @pytest.fixture
    def oedb_file(self, tmp_path):
        """Create a temporary OEDB file for testing.

        Note: We use .oeb extension because oemolostream auto-detects
        format from extension, and .oedb is not a standard recognized extension.
        The scan_oedb function works with any format that oemolistream supports.
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

    def test_scan_oedb_returns_lazyframe(self, oedb_file):
        lf = oep.scan_oedb(oedb_file)
        assert isinstance(lf, pl.LazyFrame)

    def test_scan_oedb_collects_to_dataframe(self, oedb_file):
        lf = oep.scan_oedb(oedb_file)
        df = lf.collect()
        assert isinstance(df, pl.DataFrame)

    def test_scan_oedb_has_molecule_column(self, oedb_file):
        lf = oep.scan_oedb(oedb_file)
        df = lf.collect()
        assert "Molecule" in df.columns
        assert df.schema["Molecule"].ext_name() == "molecule"

    def test_scan_oedb_supports_filter(self, oedb_file):
        lf = oep.scan_oedb(oedb_file)
        lf_filtered = lf.filter(pl.col("Title") == "ethanol")
        df = lf_filtered.collect()
        assert len(df) == 1

    def test_scan_oedb_file_not_found_raises(self):
        from oepolars.exceptions import FileError

        with pytest.raises(FileError):
            oep.scan_oedb("nonexistent.oedb")


class TestScanOEDU:
    @pytest.fixture
    def oedu_file(self, tmp_path):
        """Create a temporary OEDU file for testing."""
        from openeye import oechem

        oedu_path = tmp_path / "test.oedu"
        ofs = oechem.oeofstream(str(oedu_path))

        # Create minimal design units with just a title
        for title in ["complex1", "complex2"]:
            du = oechem.OEDesignUnit()
            du.SetTitle(title)
            oechem.OEWriteDesignUnit(ofs, du)

        ofs.close()
        return oedu_path

    def test_scan_oedu_returns_lazyframe(self, oedu_file):
        lf = oep.scan_oedu(oedu_file)
        assert isinstance(lf, pl.LazyFrame)

    def test_scan_oedu_collects_to_dataframe(self, oedu_file):
        lf = oep.scan_oedu(oedu_file)
        df = lf.collect()
        assert isinstance(df, pl.DataFrame)

    def test_scan_oedu_has_design_unit_column(self, oedu_file):
        lf = oep.scan_oedu(oedu_file)
        df = lf.collect()
        assert "DesignUnit" in df.columns
        assert df.schema["DesignUnit"].ext_name() == "design_unit"

    def test_scan_oedu_no_title_clears_design_unit_titles(self, oedu_file):
        lf = oep.scan_oedu(oedu_file, no_title=True)
        df = lf.collect()

        assert "Title" not in df.columns
        assert all(du.GetTitle() == "" for du in df["DesignUnit"].to_list())

    def test_scan_oedu_supports_filter(self, oedu_file):
        lf = oep.scan_oedu(oedu_file)
        lf_filtered = lf.filter(pl.col("Title") == "complex1")
        df = lf_filtered.collect()
        assert len(df) == 1

    def test_scan_oedu_file_not_found_raises(self):
        from oepolars.exceptions import FileError

        with pytest.raises(FileError):
            oep.scan_oedu("nonexistent.oedu")


class TestScanMoleculeCSV:
    def test_scan_molecule_csv_exposes_parity_parameters(self):
        signature = inspect.signature(oep.scan_molecule_csv)

        assert "molecule_type" in signature.parameters
        assert "no_title" in signature.parameters

    def test_scan_molecule_csv_returns_lazyframe(self):
        lf = oep.scan_molecule_csv(FIXTURES_DIR / "sample.csv", smiles_column="smiles")
        assert isinstance(lf, pl.LazyFrame)

    def test_scan_molecule_csv_collects_to_dataframe(self):
        lf = oep.scan_molecule_csv(FIXTURES_DIR / "sample.csv", smiles_column="smiles")
        df = lf.collect()
        assert isinstance(df, pl.DataFrame)

    def test_scan_molecule_csv_has_molecule_column(self):
        lf = oep.scan_molecule_csv(FIXTURES_DIR / "sample.csv", smiles_column="smiles")
        df = lf.collect()
        assert "Molecule" in df.columns
        assert df.schema["Molecule"].ext_name() == "molecule"

    def test_scan_molecule_csv_molecule_type_returns_graph_mols(self):
        lf = oep.scan_molecule_csv(
            FIXTURES_DIR / "sample.csv",
            smiles_column="smiles",
            molecule_type=oechem.OEGraphMol,
        )
        df = lf.collect()

        assert all(
            isinstance(mol, oechem.OEGraphMol)
            for mol in df["Molecule"].to_list()
        )

    def test_scan_molecule_csv_no_title_clears_molecule_titles(self, tmp_path):
        csv_path = tmp_path / "titled_smiles.csv"
        csv_path.write_text('smiles\n"CCO ethanol"\n')

        lf = oep.scan_molecule_csv(
            csv_path,
            smiles_column="smiles",
            no_title=True,
        )
        df = lf.collect()

        mol = df["Molecule"][0]
        assert mol.GetTitle() == ""
        assert mol.GetActive().GetTitle() == ""

    def test_scan_molecule_csv_preserves_other_columns(self):
        lf = oep.scan_molecule_csv(FIXTURES_DIR / "sample.csv", smiles_column="smiles")
        df = lf.collect()
        assert "name" in df.columns
        assert "MW" in df.columns
        assert len(df) == 3

    def test_scan_molecule_csv_supports_filter(self):
        lf = oep.scan_molecule_csv(FIXTURES_DIR / "sample.csv", smiles_column="smiles")
        lf_filtered = lf.filter(pl.col("name") == "ethanol")
        df = lf_filtered.collect()
        assert len(df) == 1

    def test_scan_molecule_csv_file_not_found_raises(self):
        from oepolars.exceptions import FileError

        with pytest.raises(FileError):
            oep.scan_molecule_csv("nonexistent.csv", smiles_column="smiles")
