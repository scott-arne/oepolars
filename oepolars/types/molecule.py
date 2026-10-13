"""MoleculeType extension type for OpenEye molecules."""

from enum import StrEnum
from typing import TypeAlias

import polars as pl
from openeye import oechem


class MoleculeTypeKind(StrEnum):
    """Molecule implementation to use when constructing molecule objects."""

    DEFAULT = "default"
    OEMOL = "oemol"
    OEGRAPHMOL = "oegraphmol"
    OEQMOL = "oeqmol"


MoleculeTypeInput: TypeAlias = MoleculeTypeKind | str | type[oechem.OEMolBase] | None

_MOLECULE_CLASS_BY_TYPE: dict[MoleculeTypeKind, type[oechem.OEMolBase] | None] = {
    MoleculeTypeKind.DEFAULT: None,
    MoleculeTypeKind.OEMOL: oechem.OEMol,
    MoleculeTypeKind.OEGRAPHMOL: oechem.OEGraphMol,
    MoleculeTypeKind.OEQMOL: oechem.OEQMol,
}

_MOLECULE_TYPE_BY_CLASS: dict[type[oechem.OEMolBase], MoleculeTypeKind] = {
    oechem.OEMol: MoleculeTypeKind.OEMOL,
    oechem.OEGraphMol: MoleculeTypeKind.OEGRAPHMOL,
    oechem.OEQMol: MoleculeTypeKind.OEQMOL,
}

_MOLECULE_TYPE_BY_STRING: dict[str, MoleculeTypeKind] = {
    molecule_type.value: molecule_type for molecule_type in MoleculeTypeKind
}


class MoleculeType(pl.BaseExtension):
    """Polars extension type for OpenEye molecule objects.

    Stores OEMol objects using Polars' Object dtype as underlying storage.
    OEGraphMol objects are automatically converted to OEMol on input.

    :param token: Opaque per-column token stored in the dtype metadata; it survives
        polars frame operations and lets clients key column-level settings.

    Example::

        from openeye import oechem
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")
        s = pl.Series("mol", [mol], dtype=MoleculeType())
        s.dtype.ext_name()  # Returns 'molecule'
    """

    def __init__(self, token: str | None = None) -> None:
        super().__init__("molecule", pl.Object, token)

    @property
    def token(self) -> str | None:
        """The per-column token, or ``None``."""
        return self.ext_metadata()


def _normalize_molecule_type(molecule_type: MoleculeTypeInput) -> MoleculeTypeKind:
    """Normalize user molecule type input.

    :param molecule_type: Molecule type selector.
    :returns: Normalized molecule type.
    :raises ValueError: When molecule type is unsupported.
    """
    if molecule_type is None:
        return MoleculeTypeKind.DEFAULT

    if isinstance(molecule_type, MoleculeTypeKind):
        return molecule_type

    if isinstance(molecule_type, str):
        key = molecule_type.casefold().replace("-", "").replace("_", "").replace(" ", "")
        if key in _MOLECULE_TYPE_BY_STRING:
            return _MOLECULE_TYPE_BY_STRING[key]

    if isinstance(molecule_type, type) and molecule_type in _MOLECULE_TYPE_BY_CLASS:
        return _MOLECULE_TYPE_BY_CLASS[molecule_type]

    raise ValueError(f"Unsupported molecule_type: {molecule_type!r}")


def _molecule_class_for_type(molecule_type: MoleculeTypeInput) -> type[oechem.OEMolBase] | None:
    """Return the OpenEye molecule class for a molecule type selector.

    :param molecule_type: Molecule type selector.
    :returns: Molecule class, or ``None`` for default preservation mode.
    """
    return _MOLECULE_CLASS_BY_TYPE[_normalize_molecule_type(molecule_type)]


def _new_molecule(molecule_type: MoleculeTypeInput = None) -> oechem.OEMolBase:
    """Create a new molecule for the requested molecule type.

    :param molecule_type: Molecule type selector.
    :returns: New molecule instance.
    """
    molecule_cls = _molecule_class_for_type(molecule_type) or oechem.OEMol
    return molecule_cls()


def _coerce_molecule(
    mol: oechem.OEMolBase,
    molecule_type: MoleculeTypeInput = None,
) -> oechem.OEMolBase:
    """Coerce a molecule to the requested molecule type.

    :param mol: Molecule to coerce.
    :param molecule_type: Molecule type selector.
    :returns: Coerced molecule, or the original molecule in default mode.
    """
    molecule_cls = _molecule_class_for_type(molecule_type)
    if molecule_cls is None:
        return mol
    return molecule_cls(mol)


def _clear_mol_titles(mol: oechem.OEMolBase) -> None:
    """Clear molecule titles in place.

    :param mol: Molecule to update.
    """
    mol.SetTitle("")
    if isinstance(mol, oechem.OEMol):
        mol.GetActive().SetTitle("")


def _copy_molecule(
    mol: oechem.OEMolBase,
    molecule_type: MoleculeTypeInput = None,
    *,
    no_title: bool = False,
) -> oechem.OEMolBase:
    """Copy a molecule, optionally selecting the returned molecule implementation.

    :param mol: Molecule to copy.
    :param molecule_type: Molecule type selector.
    :param no_title: Clear molecule titles on the returned copy.
    :returns: Copied molecule.
    """
    molecule_cls = _molecule_class_for_type(molecule_type)
    copied = (type(mol) if molecule_cls is None else molecule_cls)(mol)
    if no_title:
        _clear_mol_titles(copied)
    return copied


# Register the extension type with Polars
pl.register_extension_type("molecule", MoleculeType)
