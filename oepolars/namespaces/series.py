"""Series namespace for OpenEye operations."""

import polars as pl


def _is_molecule_type(dtype: pl.DataType) -> bool:
    """Check if dtype is MoleculeType."""
    return hasattr(dtype, "ext_name") and dtype.ext_name() == "molecule"


def _is_design_unit_type(dtype: pl.DataType) -> bool:
    """Check if dtype is DesignUnitType."""
    return hasattr(dtype, "ext_name") and dtype.ext_name() == "design_unit"


@pl.api.register_series_namespace("oe")
class SeriesOENamespace:
    """
    Namespace for OpenEye operations on Polars Series.

    Accessed via `series.oe.*`.
    """

    def __init__(self, series: pl.Series) -> None:
        self._series = series

    def is_valid(self) -> pl.Series:
        """
        Return a boolean Series indicating valid (non-null) molecules.

        Returns
        -------
        pl.Series
            Boolean Series where True indicates a valid molecule.

        Raises
        ------
        TypeError
            If the Series is not a MoleculeType or DesignUnitType.

        Examples
        --------
        >>> s.oe.is_valid()
        shape: (3,)
        Series: 'mol' [bool]
        [
            true
            false
            true
        ]
        """
        dtype = self._series.dtype
        if not (_is_molecule_type(dtype) or _is_design_unit_type(dtype)):
            raise TypeError(
                f"is_valid() requires MoleculeType or DesignUnitType, got {dtype}"
            )

        return self._series.is_not_null()
