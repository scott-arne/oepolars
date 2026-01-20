"""DesignUnitType extension type for OpenEye design units."""

import polars as pl


class DesignUnitType(pl.BaseExtension):
    """Polars extension type for OpenEye design unit objects.

    Stores OEDesignUnit objects (protein-ligand complexes) using
    Polars' Object dtype as underlying storage.

    Example::

        from openeye import oechem
        du = oechem.OEDesignUnit()
        s = pl.Series("du", [du], dtype=DesignUnitType())
        s.dtype.ext_name()  # Returns 'design_unit'
    """

    def __init__(self) -> None:
        super().__init__("design_unit", pl.Object, None)


# Register the extension type with Polars
pl.register_extension_type("design_unit", DesignUnitType)
