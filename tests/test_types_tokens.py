import polars as pl
import pytest
from openeye import oechem

import oepolars as oeplr


def _mols(n):
    out = []
    for _ in range(n):
        m = oechem.OEGraphMol()
        oechem.OESmilesToMol(m, "CCO")
        out.append(m)
    return out


def _frame(token, n=3):
    return pl.DataFrame([
        pl.Series("mol", _mols(n), dtype=oeplr.MoleculeType(token)),
        pl.Series("x", list(range(n))),
    ])


def test_token_property_and_default():
    assert oeplr.MoleculeType().token is None
    assert oeplr.MoleculeType("abc").token == "abc"
    assert oeplr.DisplayType("d").token == "d" and oeplr.DesignUnitType("u").token == "u"
    assert isinstance(oeplr.MoleculeType("abc"), oeplr.MoleculeType)


@pytest.mark.parametrize(
    "op",
    [
        lambda df: df.filter(pl.col("x") > 0),
        lambda df: df.select(pl.col("mol").alias("m2")).rename({"m2": "mol"}),
        lambda df: df.with_columns(y=pl.col("x") * 2),
        lambda df: df.sort("x", descending=True),
        lambda df: df.head(2),
        lambda df: pl.concat([df, df]),
        lambda df: df.join(pl.DataFrame({"x": [0, 1, 2], "z": [1, 2, 3]}), on="x"),
        lambda df: pl.DataFrame([df.get_column("mol")]),
    ],
)
def test_token_survives_frame_operations(op):
    out = op(_frame("tok"))
    dtype = out.schema["mol"]
    assert isinstance(dtype, oeplr.MoleculeType) and dtype.token == "tok"


def test_series_token_and_with_token():
    df = _frame("one")
    assert df.get_column("mol").chem.token == "one"
    retagged = df.chem.with_token("mol", "two")
    assert retagged.schema["mol"].token == "two" and df.schema["mol"].token == "one"
    assert retagged.get_column("mol")[0] is df.get_column("mol")[0]
    cleared = retagged.chem.with_token("mol", None)
    assert cleared.schema["mol"].token is None
    with pytest.raises(TypeError):
        df.chem.with_token("x", "nope")


def test_mixed_tokens_cannot_concat_but_cleared_frames_can():
    with pytest.raises(pl.exceptions.SchemaError):
        pl.concat([_frame("a", 1), _frame("b", 1)])
    a = _frame("a", 1).chem.with_token("mol", None)
    b = _frame("b", 1).chem.with_token("mol", None)
    assert len(pl.concat([a, b])) == 2


def _queries(n):
    out = []
    for _ in range(n):
        q = oechem.OEQMol()
        oechem.OEParseSmarts(q, "c1ccccc1")
        out.append(q)
    return out


def test_query_token_property_and_default():
    assert oeplr.QueryType().token is None
    assert oeplr.QueryType("q").token == "q"


def test_query_token_survives_frame_operations():
    df = pl.DataFrame([
        pl.Series("query", _queries(3), dtype=oeplr.QueryType("tok")),
        pl.Series("x", list(range(3))),
    ])
    dtype = df.filter(pl.col("x") > 0).schema["query"]
    assert isinstance(dtype, oeplr.QueryType) and dtype.token == "tok"


def test_with_token_on_query_column():
    df = pl.DataFrame([pl.Series("query", _queries(2), dtype=oeplr.QueryType())])
    retagged = df.chem.with_token("query", "two")
    assert retagged.get_column("query").chem.token == "two"
    assert retagged.get_column("query")[0] is df.get_column("query")[0]
