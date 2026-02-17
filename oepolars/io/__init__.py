"""File I/O functions for OEPolars."""

from oepolars.io.pandas import from_pandas
from oepolars.io.readers import (
    read_molecule_csv,
    read_oeb,
    read_oedb,
    read_oedu,
    read_parquet,
    read_sdf,
    read_smi,
)
from oepolars.io.scanners import (
    scan_molecule_csv,
    scan_oeb,
    scan_oedb,
    scan_oedu,
    scan_parquet,
    scan_sdf,
    scan_smi,
)

__all__ = [
    "from_pandas",
    "read_molecule_csv",
    "read_oeb",
    "read_oedb",
    "read_oedu",
    "read_parquet",
    "read_sdf",
    "read_smi",
    "scan_molecule_csv",
    "scan_oeb",
    "scan_oedb",
    "scan_oedu",
    "scan_parquet",
    "scan_sdf",
    "scan_smi",
]
