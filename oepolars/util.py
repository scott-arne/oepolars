"""Utility functions for OEPolars."""

from dataclasses import dataclass
from pathlib import Path

from openeye import oechem

from oepolars.exceptions import UnsupportedFileFormat


@dataclass
class OEFormatInfo:
    """Container for OpenEye format information."""

    oeformat: int
    gzip: bool


# Mapping of string format names to OEFormat constants
FORMAT_MAP: dict[str, int] = {
    "sdf": oechem.OEFormat_SDF,
    "sd": oechem.OEFormat_SDF,
    "mol": oechem.OEFormat_SDF,
    "mol2": oechem.OEFormat_MOL2,
    "oeb": oechem.OEFormat_OEB,
    "oez": oechem.OEFormat_OEZ,
    "smi": oechem.OEFormat_SMI,
    "smiles": oechem.OEFormat_SMI,
    "ism": oechem.OEFormat_ISM,
    "usm": oechem.OEFormat_USM,
    "can": oechem.OEFormat_CAN,
    "cxsmiles": oechem.OEFormat_CXSMILES,
    "pdb": oechem.OEFormat_PDB,
    "xyz": oechem.OEFormat_XYZ,
    "csv": oechem.OEFormat_CSV,
}


def is_gz(filepath: str | Path) -> bool:
    """
    Check if a file path indicates gzip compression.

    Parameters
    ----------
    filepath
        Path to check for gzip extension.

    Returns
    -------
    bool
        True if the file has a .gz or .gzip extension.
    """
    path = Path(filepath) if isinstance(filepath, str) else filepath
    suffixes = [s.lower() for s in path.suffixes]
    return ".gz" in suffixes or ".gzip" in suffixes


def get_oeformat(fmt: str | int, gzip: bool) -> OEFormatInfo:
    """
    Convert a format specification to OEFormat info.

    Parameters
    ----------
    fmt
        Format as string (e.g., "sdf", "oeb") or OEFormat constant.
    gzip
        Whether gzip compression is enabled.

    Returns
    -------
    OEFormatInfo
        Container with oeformat constant and gzip flag.

    Raises
    ------
    UnsupportedFileFormat
        If the format string is not recognized.
    """
    if isinstance(fmt, int):
        return OEFormatInfo(oeformat=fmt, gzip=gzip)

    fmt_lower = fmt.lower().strip(".")
    if fmt_lower not in FORMAT_MAP:
        raise UnsupportedFileFormat(f"Unsupported molecule format: {fmt}")

    return OEFormatInfo(oeformat=FORMAT_MAP[fmt_lower], gzip=gzip)
