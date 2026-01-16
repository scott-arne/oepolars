"""Custom exceptions for OEPolars."""


class OEPolarsError(Exception):
    """Base exception for all OEPolars errors."""


class FileError(OEPolarsError):
    """File I/O errors (not found, permission denied, invalid format)."""


class UnsupportedFileFormat(OEPolarsError):
    """Unrecognized or unsupported molecule file format."""


class InvalidSMARTS(OEPolarsError):
    """Invalid SMARTS pattern for substructure search."""


class InvalidMolecule(OEPolarsError):
    """Molecule parsing or validation error."""


class LazyOperationError(OEPolarsError):
    """Molecule operation attempted on LazyFrame without collect()."""
