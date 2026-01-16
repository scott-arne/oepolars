"""DisplayType extension type for OpenEye 2D molecule displays."""

import polars as pl


class DisplayType(pl.BaseExtension):
    """
    Polars extension type for OpenEye 2D molecule display objects.

    Stores OE2DMolDisplay objects for molecular visualization using
    Polars' Object dtype as underlying storage.

    Examples
    --------
    >>> from openeye import oechem, oedepict
    >>> mol = oechem.OEMol()
    >>> oechem.OEParseSmiles(mol, "CCO")
    True
    >>> oedepict.OEPrepareDepiction(mol)
    True
    >>> opts = oedepict.OE2DMolDisplayOptions()
    >>> disp = oedepict.OE2DMolDisplay(mol, opts)
    >>> s = pl.Series("disp", [disp], dtype=DisplayType())
    >>> s.dtype.ext_name()
    'display'
    """

    def __init__(self) -> None:
        super().__init__("display", pl.Object, None)


# Register the extension type with Polars
pl.register_extension_type("display", DisplayType)
