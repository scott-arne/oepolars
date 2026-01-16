"""Series namespace for OpenEye operations."""

import polars as pl


@pl.api.register_series_namespace("oe")
class SeriesOENamespace:
    """
    Namespace for OpenEye operations on Polars Series.

    Accessed via `series.oe.*`.

    Examples
    --------
    >>> import oepolars as oep
    >>> s = pl.Series("mol", [mol1, mol2], dtype=oep.MoleculeType())
    >>> s.oe.to_smiles()
    """

    def __init__(self, series: pl.Series) -> None:
        self._series = series
