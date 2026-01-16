import pytest

from oepolars.exceptions import (
    FileError,
    InvalidMolecule,
    InvalidSMARTS,
    LazyOperationError,
    OEPolarsError,
    UnsupportedFileFormat,
)


class TestExceptions:
    def test_oepolars_error_is_base_exception(self):
        assert issubclass(OEPolarsError, Exception)

    def test_file_error_inherits_from_base(self):
        assert issubclass(FileError, OEPolarsError)

    def test_unsupported_file_format_inherits_from_base(self):
        assert issubclass(UnsupportedFileFormat, OEPolarsError)

    def test_invalid_smarts_inherits_from_base(self):
        assert issubclass(InvalidSMARTS, OEPolarsError)

    def test_invalid_molecule_inherits_from_base(self):
        assert issubclass(InvalidMolecule, OEPolarsError)

    def test_lazy_operation_error_inherits_from_base(self):
        assert issubclass(LazyOperationError, OEPolarsError)

    def test_exceptions_can_be_raised_with_message(self):
        with pytest.raises(FileError, match="test message"):
            raise FileError("test message")

    def test_exceptions_can_be_caught_by_base_class(self):
        with pytest.raises(OEPolarsError):
            raise InvalidSMARTS("bad pattern")
