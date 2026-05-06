"""FingerprintType extension type for OpenEye fingerprints."""

import polars as pl


class FingerprintType(pl.BaseExtension):
    """Polars extension type for OpenEye fingerprint objects.

    Stores OEFingerPrint objects using Polars' Object dtype as underlying storage.
    Provides integration with OpenEye's fingerprint generation and similarity
    calculation functionality.

    Example::

        from openeye import oechem, oegraphsim
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")
        fp = oegraphsim.OEFingerPrint()
        oegraphsim.OEMakePathFP(fp, mol)
        s = pl.Series("fp", [fp], dtype=FingerprintType())
        s.dtype.ext_name()  # Returns 'fingerprint'
    """

    def __init__(self) -> None:
        super().__init__("fingerprint", pl.Object, None)


# Register the extension type with Polars
pl.register_extension_type("fingerprint", FingerprintType)
