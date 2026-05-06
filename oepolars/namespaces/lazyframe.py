"""LazyFrame namespace for OpenEye chemistry operations."""

import polars as pl

from oepolars.exceptions import LazyOperationError
from oepolars.types.query import QueryFormat, QueryFormatInput


def _is_molecule_type(dtype: pl.DataType) -> bool:
    """Check if dtype is MoleculeType."""
    return hasattr(dtype, "ext_name") and dtype.ext_name() == "molecule"


def _is_design_unit_type(dtype: pl.DataType) -> bool:
    """Check if dtype is DesignUnitType."""
    return hasattr(dtype, "ext_name") and dtype.ext_name() == "design_unit"


@pl.api.register_lazyframe_namespace("chem")
class LazyFrameChemNamespace:
    """Namespace for OpenEye chemistry operations on Polars LazyFrames.

    Most molecule operations require eager evaluation. Use ``.collect()`` first
    to convert to a DataFrame, then use ``df.chem.*`` methods.

    Accessed via ``lf.chem.*``.
    """

    def __init__(self, lf: pl.LazyFrame) -> None:
        self._lf = lf

    def has_molecule_columns(self) -> bool:
        """Check if the LazyFrame has any molecule columns.

        :returns: True if there are molecule columns, False otherwise.

        Example::

            lf = oep.scan_sdf("molecules.sdf")
            lf.chem.has_molecule_columns()  # Returns True
        """
        for dtype in self._lf.collect_schema().values():
            if _is_molecule_type(dtype) or _is_design_unit_type(dtype):
                return True
        return False

    def molecule_columns(self) -> list[str]:
        """Get list of molecule column names.

        :returns: Names of columns with MoleculeType or DesignUnitType.

        Example::

            lf = oep.scan_sdf("molecules.sdf")
            lf.chem.molecule_columns()  # Returns ['Molecule']
        """
        cols = []
        for name, dtype in self._lf.collect_schema().items():
            if _is_molecule_type(dtype) or _is_design_unit_type(dtype):
                cols.append(name)
        return cols

    def to_smiles(self, column: str) -> None:
        """Convert molecules to SMILES - NOT SUPPORTED on LazyFrames.

        :raises LazyOperationError: Always raised. Use ``.collect()`` first.
        """
        raise LazyOperationError(
            "to_smiles() requires eager evaluation. "
            "Use lf.collect().chem.to_smiles() instead, or "
            "use the Series method: lf.collect()['column'].chem.to_smiles()"
        )

    def subsearch(self, column: str, pattern: str) -> None:
        """Substructure search - NOT SUPPORTED on LazyFrames.

        :raises LazyOperationError: Always raised. Use ``.collect()`` first.
        """
        raise LazyOperationError(
            "subsearch() requires eager evaluation. "
            "Use lf.collect().chem.subsearch() instead, or "
            "use the Series method: lf.collect()['column'].chem.subsearch()"
        )

    def filter_valid(self, columns: str | list[str]) -> None:
        """Filter valid molecules - NOT SUPPORTED on LazyFrames.

        :raises LazyOperationError: Always raised. Use ``.collect()`` first.
        """
        raise LazyOperationError(
            "filter_valid() requires eager evaluation. "
            "Use lf.collect().chem.filter_valid() instead."
        )

    def as_molecule(self, columns: str | list[str]) -> None:
        """Convert columns to molecules - NOT SUPPORTED on LazyFrames.

        :raises LazyOperationError: Always raised. Use ``.collect()`` first.
        """
        raise LazyOperationError(
            "as_molecule() requires eager evaluation. "
            "Use lf.collect().chem.as_molecule() instead."
        )

    def as_query(
        self,
        columns: str | list[str],
        *,
        query_format: QueryFormatInput = QueryFormat.SMARTS,
        no_title: bool = False,
    ) -> None:
        """Convert columns to queries - NOT SUPPORTED on LazyFrames.

        :raises LazyOperationError: Always raised. Use ``.collect()`` first.
        """
        raise LazyOperationError(
            "as_query() requires eager evaluation. "
            "Use lf.collect().chem.as_query() instead."
        )

    def detect_molecule_columns(self) -> None:
        """Detect and convert molecule columns - NOT SUPPORTED on LazyFrames.

        :raises LazyOperationError: Always raised. Use ``.collect()`` first.
        """
        raise LazyOperationError(
            "detect_molecule_columns() requires eager evaluation. "
            "Use lf.collect().chem.detect_molecule_columns() instead."
        )
