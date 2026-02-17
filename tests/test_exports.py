

class TestPackageExports:
    def test_version_exported(self):
        from oepolars import __version__
        assert __version__ is not None

    def test_molecule_type_exported(self):
        from oepolars import MoleculeType
        assert MoleculeType is not None

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
