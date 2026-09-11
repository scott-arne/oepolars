"""MoleculeType extension type for OpenEye molecules."""

import polars as pl


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


# Register the extension type with Polars
pl.register_extension_type("molecule", MoleculeType)
