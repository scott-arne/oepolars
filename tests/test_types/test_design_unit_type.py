import polars as pl
import pytest
from openeye import oechem

from oepolars.types import DesignUnitType


class TestDesignUnitTypeCreation:
    def test_design_unit_type_is_extension(self):
        du_type = DesignUnitType()
        assert isinstance(du_type, pl.BaseExtension)

    def test_design_unit_type_name(self):
        du_type = DesignUnitType()
        assert du_type.ext_name() == "design_unit"

    def test_design_unit_type_storage(self):
        du_type = DesignUnitType()
        assert du_type.ext_storage() == pl.Object


class TestDesignUnitTypeSeries:
    @pytest.fixture
    def sample_design_unit(self):
        # Create a minimal design unit for testing
        du = oechem.OEDesignUnit()
        return du

    def test_create_series_from_design_units(self, sample_design_unit):
        s = pl.Series("du", [sample_design_unit], dtype=DesignUnitType())
        assert len(s) == 1
        assert s.dtype.ext_name() == "design_unit"

    def test_series_displays_as_ext_design_unit(self, sample_design_unit):
        s = pl.Series("du", [sample_design_unit], dtype=DesignUnitType())
        repr_str = repr(s)
        assert "ext[design_unit]" in repr_str

    def test_series_with_none_values(self, sample_design_unit):
        data = [sample_design_unit, None]
        s = pl.Series("du", data, dtype=DesignUnitType())
        assert s.null_count() == 1
