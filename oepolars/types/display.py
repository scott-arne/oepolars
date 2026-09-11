"""DisplayType extension type for OpenEye 2D molecule displays."""

import polars as pl


class DisplayType(pl.BaseExtension):
    """Polars extension type for OpenEye 2D molecule display objects.

    Stores OE2DMolDisplay objects for molecular visualization using
    Polars' Object dtype as underlying storage.

    :param token: Opaque per-column token stored in the dtype metadata; it survives
        polars frame operations and lets clients key column-level settings.

    Example::

        from openeye import oechem, oedepict
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")
        oedepict.OEPrepareDepiction(mol)
        opts = oedepict.OE2DMolDisplayOptions()
        disp = oedepict.OE2DMolDisplay(mol, opts)
        s = pl.Series("disp", [disp], dtype=DisplayType())
        s.dtype.ext_name()  # Returns 'display'
    """

    def __init__(self, token: str | None = None) -> None:
        super().__init__("display", pl.Object, token)

    @property
    def token(self) -> str | None:
        """The per-column token, or ``None``."""
        return self.ext_metadata()


# Register the extension type with Polars
pl.register_extension_type("display", DisplayType)
