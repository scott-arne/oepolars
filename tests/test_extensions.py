import polars as pl
import pytest

import oepolars as oeplr
from oepolars import extensions


def test_register_methods_and_overwrite_guard():
    def greet(self):
        return f"hello {self._series.name}"

    extensions.register_series_method("greet_probe", greet)
    s = pl.Series("m", [], dtype=oeplr.MoleculeType())
    assert s.chem.greet_probe() == "hello m"
    with pytest.raises(ValueError):
        extensions.register_series_method("greet_probe", greet)
    extensions.register_series_method("greet_probe", lambda self: "again", overwrite=True)
    assert s.chem.greet_probe() == "again"

    def width(self):
        return self._df.width

    extensions.register_dataframe_method("width_probe", width)
    assert pl.DataFrame({"a": [1]}).chem.width_probe() == 1
