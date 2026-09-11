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
