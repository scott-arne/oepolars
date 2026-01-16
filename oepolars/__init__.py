"""OEPolars: OpenEye Toolkits integration with Polars DataFrames."""

import logging

__version__ = "0.1.0"

# Configure logging
log = logging.getLogger("oepolars")


class LevelSpecificFormatter(logging.Formatter):
    """A logging formatter that prefixes every level but INFO."""

    NORMAL_FORMAT = "%(message)s"
    LEVEL_SPECIFIC_FORMAT = "%(levelname)s: %(message)s"

    def __init__(self) -> None:
        super().__init__(fmt=self.NORMAL_FORMAT, datefmt=None, style="%")

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.INFO:
            self._style._fmt = self.NORMAL_FORMAT
        else:
            self._style._fmt = self.LEVEL_SPECIFIC_FORMAT
        return logging.Formatter.format(self, record)


# Configure the logger
_handler = logging.StreamHandler()
_handler.setLevel(logging.DEBUG)
_handler.setFormatter(LevelSpecificFormatter())
log.addHandler(_handler)
log.setLevel(logging.INFO)

# Export extension types
# Export exceptions
# Register namespaces (import for side effects)
from oepolars import namespaces as _namespaces  # noqa: E402, F401
from oepolars.exceptions import (  # noqa: E402
    FileError,
    InvalidMolecule,
    InvalidSMARTS,
    LazyOperationError,
    OEPolarsError,
    UnsupportedFileFormat,
)
from oepolars.types import DesignUnitType, DisplayType, MoleculeType  # noqa: E402

__all__ = [
    # Version
    "__version__",
    # Extension types
    "MoleculeType",
    "DesignUnitType",
    "DisplayType",
    # Exceptions
    "OEPolarsError",
    "FileError",
    "UnsupportedFileFormat",
    "InvalidSMARTS",
    "InvalidMolecule",
    "LazyOperationError",
]
