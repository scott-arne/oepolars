import polars as pl
from openeye import oechem

# Import to register namespace
import oepolars  # noqa: F401
from oepolars import MoleculeType


class TestSeriesNamespaceRegistration:
    def test_oe_namespace_exists_on_series(self):
        s = pl.Series("x", [1, 2, 3])
        assert hasattr(s, "oe")

    def test_oe_namespace_accessible(self):
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")
        s = pl.Series("mol", [mol], dtype=MoleculeType())
        ns = s.oe
        assert ns is not None
