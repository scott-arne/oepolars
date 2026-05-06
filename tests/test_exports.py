from pathlib import Path


class TestPackageExports:
    def test_version_exported(self):
        from oepolars import __version__
        assert __version__ is not None

    def test_molecule_type_exported(self):
        from oepolars import MoleculeType
        assert MoleculeType is not None

    def test_query_type_exported(self):
        from oepolars import QueryType
        assert QueryType is not None

    def test_query_format_exported(self):
        from oepolars import QueryFormat
        assert QueryFormat.SMARTS == "smarts"

    def test_molecule_type_kind_exported(self):
        from oepolars import MoleculeTypeKind
        assert MoleculeTypeKind.OEQMOL == "oeqmol"

    def test_design_unit_type_exported(self):
        from oepolars import DesignUnitType
        assert DesignUnitType is not None

    def test_display_type_exported(self):
        from oepolars import DisplayType
        assert DisplayType is not None

    def test_exceptions_exported(self):
        from oepolars import (
            FileError,
            OEPolarsError,
        )
        assert issubclass(FileError, OEPolarsError)

    def test_from_pandas_exported(self):
        from oepolars import from_pandas
        assert callable(from_pandas)

    def test_py_typed_marker_in_imported_package(self):
        import oepolars

        package_dir = Path(oepolars.__file__).parent

        assert (package_dir / "py.typed").is_file()
