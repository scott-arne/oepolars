"""DataFrame namespace for OpenEye operations."""

import polars as pl


@pl.api.register_dataframe_namespace("oe")
class DataFrameOENamespace:
    """
    Namespace for OpenEye operations on Polars DataFrames.

    Accessed via `df.oe.*`.

    Examples
    --------
    >>> import oepolars as oep
    >>> df = oep.read_sdf("molecules.sdf")
    >>> df.oe.filter_valid("Molecule")
    """

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df
