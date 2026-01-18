import polars as pl
from openeye import oechem

import oepolars  # noqa: F401
from oepolars import MoleculeType


class TestDataFrameNamespaceRegistration:
    def test_oe_namespace_exists_on_dataframe(self):
        df = pl.DataFrame({"x": [1, 2, 3]})
        assert hasattr(df, "oe")

    def test_oe_namespace_accessible(self):
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")
        df = pl.DataFrame({
            "mol": pl.Series([mol], dtype=MoleculeType()),
        })
        ns = df.oe
        assert ns is not None
