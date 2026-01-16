"""MoleculeType extension type for OpenEye molecules."""

import polars as pl


class MoleculeType(pl.BaseExtension):
    """
    Polars extension type for OpenEye molecule objects.

    Stores OEMol objects using Polars' Object dtype as underlying storage.
    OEGraphMol objects are automatically converted to OEMol on input.

    Examples
    --------
    >>> from openeye import oechem
    >>> mol = oechem.OEMol()
    >>> oechem.OEParseSmiles(mol, "CCO")
    True
    >>> s = pl.Series("mol", [mol], dtype=MoleculeType())
    >>> s.dtype.ext_name()
    'molecule'
    """

    def __init__(self) -> None:
        super().__init__("molecule", pl.Object, None)


# Register the extension type with Polars
pl.register_extension_type("molecule", MoleculeType)
