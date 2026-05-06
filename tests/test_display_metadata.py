import polars as pl
from openeye import oechem, oedepict

# Import to register namespace
import oepolars  # noqa: F401
from oepolars import DisplayType


class TestDisplayMetadata:
    """Tests for display series metadata support."""

    def test_metadata_exists_on_display_series(self):
        """Display series should have a metadata dictionary."""
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, "CCO")
        oedepict.OEPrepareDepiction(mol)

        opts = oedepict.OE2DMolDisplayOptions(200, 200, oedepict.OEScale_AutoScale)
        disp = oedepict.OE2DMolDisplay(mol, opts)

        s = pl.Series("disp", [disp], dtype=DisplayType())

        metadata = s.chem.metadata
        assert isinstance(metadata, dict)

    def test_display_metadata_stores_values(self):
        """Display metadata should store and retrieve values."""
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, "CCO")
        oedepict.OEPrepareDepiction(mol)

        opts = oedepict.OE2DMolDisplayOptions(200, 200, oedepict.OEScale_AutoScale)
        disp = oedepict.OE2DMolDisplay(mol, opts)

        s = pl.Series("disp", [disp], dtype=DisplayType())

        s.chem.metadata["callbacks"] = ["func1", "func2"]
        assert s.chem.metadata["callbacks"] == ["func1", "func2"]

    def test_display_deepcopy_preserves_metadata(self):
        """Deep copy of display series should preserve metadata."""
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, "CCO")
        oedepict.OEPrepareDepiction(mol)

        opts = oedepict.OE2DMolDisplayOptions(200, 200, oedepict.OEScale_AutoScale)
        disp = oedepict.OE2DMolDisplay(mol, opts)

        s = pl.Series("disp", [disp], dtype=DisplayType())
        s.chem.metadata["key"] = "value"

        copied = s.chem.deepcopy()
        assert copied.chem.metadata.get("key") == "value"

    def test_display_deepcopy_creates_independent_displays(self):
        """Deep copy should create independent display objects."""
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, "CCO")
        oedepict.OEPrepareDepiction(mol)

        opts = oedepict.OE2DMolDisplayOptions(200, 200, oedepict.OEScale_AutoScale)
        disp = oedepict.OE2DMolDisplay(mol, opts)

        s = pl.Series("disp", [disp], dtype=DisplayType())
        copied = s.chem.deepcopy()

        # Verify both are valid display objects
        assert s[0].IsValid()
        assert copied[0].IsValid()

        # They should be different objects
        assert id(s[0]) != id(copied[0])
