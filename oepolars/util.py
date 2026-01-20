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
    """Check if a file path indicates gzip compression.

    :param filepath: Path to check for gzip extension.
    :returns: True if the file has a .gz or .gzip extension.
    """
    path = Path(filepath) if isinstance(filepath, str) else filepath
    suffixes = [s.lower() for s in path.suffixes]
    return ".gz" in suffixes or ".gzip" in suffixes


def mol_to_bytes(mol: oechem.OEMolBase | None) -> bytes | None:
    """Serialize a molecule to OEB binary format.

    :param mol: The molecule to serialize, or None.
    :returns: Binary representation of the molecule, or None if mol is None.

    Example::

        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")
        data = mol_to_bytes(mol)
        isinstance(data, bytes)  # Returns True
    """
    if mol is None:
        return None
    return oechem.OEWriteMolToBytes(".oeb", mol)


def bytes_to_mol(data: bytes | None) -> oechem.OEMol | None:
    """Deserialize a molecule from OEB binary format.

    :param data: Binary data in OEB format, or None.
    :returns: The deserialized molecule, or None if data is None/empty/invalid.

    Example::

        data = mol_to_bytes(mol)
        mol2 = bytes_to_mol(data)
        mol2.GetTitle()  # Returns 'ethanol'
    """
    if data is None or len(data) == 0:
        return None

    mol = oechem.OEMol()
    if oechem.OEReadMolFromBytes(mol, ".oeb", data):
        return mol
    return None


def get_oeformat(fmt: str | int, gzip: bool) -> OEFormatInfo:
    """Convert a format specification to OEFormat info.

    :param fmt: Format as string (e.g., "sdf", "oeb") or OEFormat constant.
    :param gzip: Whether gzip compression is enabled.
    :returns: Container with oeformat constant and gzip flag.
    :raises UnsupportedFileFormat: If the format string is not recognized.
    """
    if isinstance(fmt, int):
        return OEFormatInfo(oeformat=fmt, gzip=gzip)

    fmt_lower = fmt.lower().strip(".")
    if fmt_lower not in FORMAT_MAP:
        raise UnsupportedFileFormat(f"Unsupported molecule format: {fmt}")

    return OEFormatInfo(oeformat=FORMAT_MAP[fmt_lower], gzip=gzip)
