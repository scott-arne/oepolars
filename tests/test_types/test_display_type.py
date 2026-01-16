import polars as pl
import pytest
from openeye import oechem, oedepict

from oepolars.types import DisplayType


class TestDisplayTypeCreation:
    def test_display_type_is_extension(self):
        disp_type = DisplayType()
        assert isinstance(disp_type, pl.BaseExtension)

    def test_display_type_name(self):
        disp_type = DisplayType()
        assert disp_type.ext_name() == "display"

    def test_display_type_storage(self):
        disp_type = DisplayType()
        assert disp_type.ext_storage() == pl.Object


class TestDisplayTypeSeries:
    @pytest.fixture
    def sample_display(self):
        # Create a minimal 2D display for testing
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")
        oedepict.OEPrepareDepiction(mol)
        opts = oedepict.OE2DMolDisplayOptions()
        disp = oedepict.OE2DMolDisplay(mol, opts)
        return disp

    def test_create_series_from_displays(self, sample_display):
        s = pl.Series("disp", [sample_display], dtype=DisplayType())
        assert len(s) == 1
        assert s.dtype.ext_name() == "display"

    def test_series_displays_as_ext_display(self, sample_display):
        s = pl.Series("disp", [sample_display], dtype=DisplayType())
        repr_str = repr(s)
        assert "ext[display]" in repr_str

    def test_series_with_none_values(self, sample_display):
        data = [sample_display, None]
        s = pl.Series("disp", data, dtype=DisplayType())
        assert s.null_count() == 1
