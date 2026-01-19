"""Lazy/scan functions for molecular data formats."""

from collections.abc import Iterable
from pathlib import Path

import polars as pl

from oepolars.io.readers import read_sdf

# Type alias for file paths
FilePath = str | Path


def scan_sdf(
    filepath: FilePath,
    *,
    flavor: int | None = None,
    molecule_column: str = "Molecule",
    title_column: str | None = "Title",
    sd_data: bool = True,
    usecols: Iterable[str] | None = None,
    numeric_columns: Iterable[str] | None = None,
) -> pl.LazyFrame:
    """
    Lazily read molecules from an SDF file.

    Returns a LazyFrame for query optimization. Molecules are read eagerly
    but downstream operations (select, filter, etc.) are optimized lazily.

    Parameters
    ----------
    filepath
        Path to the SDF file.
    flavor
        OpenEye flavor flags for reading.
    molecule_column
        Name for the molecule column.
    title_column
        Name for the title column, or None to skip.
    sd_data
        Whether to read SD data tags as columns.
    usecols
        Specific SD tags to read (None = all).
    numeric_columns
        SD tag columns to convert to numeric types.

    Returns
    -------
    pl.LazyFrame
        LazyFrame with molecule column and optional SD data.

    Raises
    ------
    FileError
        If the file cannot be read.

    Examples
    --------
    >>> lf = oep.scan_sdf("molecules.sdf")
    >>> df = lf.filter(pl.col("MW") > 100).collect()
    """
    df = read_sdf(
        filepath,
        flavor=flavor,
        molecule_column=molecule_column,
        title_column=title_column,
        sd_data=sd_data,
        usecols=usecols,
        numeric_columns=numeric_columns,
    )
    return df.lazy()
