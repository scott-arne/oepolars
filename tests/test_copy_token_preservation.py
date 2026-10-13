import polars as pl
from openeye import oechem

import oepolars as oeplr


def _mols(n):
    out = []
    for _ in range(n):
        m = oechem.OEGraphMol()
        oechem.OESmilesToMol(m, "CCO")
        out.append(m)
    return out


def _design_units(n):
    out = []
    for _ in range(n):
        du = oechem.OEDesignUnit()
        out.append(du)
    return out


def test_deepcopy_preserves_molecule_token():
    s = pl.Series("mol", _mols(2), dtype=oeplr.MoleculeType("tok"))
    copied = s.chem.deepcopy()
    assert copied.chem.token == "tok"
    assert copied[0] is not s[0]


def test_deepcopy_preserves_no_token_on_molecule():
    s = pl.Series("mol", _mols(2), dtype=oeplr.MoleculeType())
    copied = s.chem.deepcopy()
    assert copied.chem.token is None


def test_copy_molecules_preserves_token():
    s = pl.Series("mol", _mols(2), dtype=oeplr.MoleculeType("keep"))
    copied = s.chem.copy_molecules()
    assert copied.chem.token == "keep"
    assert copied[0] is not s[0]


def test_copy_molecules_preserves_no_token():
    s = pl.Series("mol", _mols(2), dtype=oeplr.MoleculeType())
    copied = s.chem.copy_molecules()
    assert copied.chem.token is None


def test_copy_design_units_preserves_token():
    s = pl.Series("du", _design_units(2), dtype=oeplr.DesignUnitType("keep"))
    copied = s.chem.copy_design_units()
    assert copied.chem.token == "keep"
    assert copied[0] is not s[0]


def test_copy_design_units_preserves_no_token():
    s = pl.Series("du", _design_units(2), dtype=oeplr.DesignUnitType())
    copied = s.chem.copy_design_units()
    assert copied.chem.token is None


def test_as_molecule_recopy_preserves_token():
    s = pl.Series("mol", _mols(2), dtype=oeplr.MoleculeType("keep"))
    converted = s.chem.as_molecule(no_title=True)
    assert converted.chem.token == "keep"
    assert converted[0] is not s[0]


def test_as_query_recopy_preserves_token():
    queries = []
    for _ in range(2):
        q = oechem.OEQMol()
        oechem.OEParseSmarts(q, "c1ccccc1")
        queries.append(q)
    s = pl.Series("query", queries, dtype=oeplr.QueryType("keep"))
    converted = s.chem.as_query(no_title=True)
    assert converted.chem.token == "keep"
    assert converted[0] is not s[0]


def _queries(n):
    out = []
    for _ in range(n):
        q = oechem.OEQMol()
        oechem.OEParseSmarts(q, "c1ccccc1")
        out.append(q)
    return out


def test_deepcopy_copies_queries_and_preserves_token():
    s = pl.Series("query", _queries(2), dtype=oeplr.QueryType("tok"))
    copied = s.chem.deepcopy()
    assert isinstance(copied.dtype, oeplr.QueryType)
    assert copied.chem.token == "tok"
    assert isinstance(copied[0], oechem.OEQMol)
    assert copied[0] is not s[0]


def test_deepcopy_keeps_null_queries():
    s = pl.Series("query", [*_queries(1), None], dtype=oeplr.QueryType())
    copied = s.chem.deepcopy()
    assert copied[1] is None
