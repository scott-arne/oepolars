"""DesignUnitType extension type for OpenEye design units."""

import polars as pl


class DesignUnitType(pl.BaseExtension):
    """Polars extension type for OpenEye design unit objects.

    Stores OEDesignUnit objects (protein-ligand complexes) using
    Polars' Object dtype as underlying storage.

    :param token: Opaque per-column token stored in the dtype metadata; it survives
        polars frame operations and lets clients key column-level settings.

    Example::

        from openeye import oechem
        du = oechem.OEDesignUnit()
        s = pl.Series("du", [du], dtype=DesignUnitType())
        s.dtype.ext_name()  # Returns 'design_unit'
    """

    def __init__(self, token: str | None = None) -> None:
        super().__init__("design_unit", pl.Object, token)

    @property
    def token(self) -> str | None:
        """The per-column token, or ``None``."""
        return self.ext_metadata()


# Register the extension type with Polars
pl.register_extension_type("design_unit", DesignUnitType)
