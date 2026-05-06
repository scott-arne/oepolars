"""Lazy/scan functions for molecular data formats."""

from collections.abc import Iterable
from pathlib import Path

import polars as pl

from oepolars.io.readers import (
    read_molecule_csv,
    read_oeb,
    read_oedb,
    read_oedu,
    read_parquet,
    read_sdf,
    read_smi,
)
from oepolars.types.molecule import MoleculeTypeInput

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
    molecule_type: MoleculeTypeInput = None,
    no_title: bool = False,
) -> pl.LazyFrame:
    """Lazily read molecules from an SDF file.

    Returns a LazyFrame for query optimization. Molecules are read eagerly
    but downstream operations (select, filter, etc.) are optimized lazily.

    :param filepath: Path to the SDF file.
    :param flavor: OpenEye flavor flags for reading.
    :param molecule_column: Name for the molecule column.
    :param title_column: Name for the title column, or None to skip.
    :param sd_data: Whether to read SD data tags as columns.
    :param usecols: Specific SD tags to read (None = all).
    :param numeric_columns: SD tag columns to convert to numeric types.
    :param molecule_type: Molecule implementation to produce.
    :param no_title: Do not read molecule titles.
    :returns: LazyFrame with molecule column and optional SD data.
    :raises FileError: If the file cannot be read.

    Example::

        lf = oep.scan_sdf("molecules.sdf")
        df = lf.filter(pl.col("MW") > 100).collect()
    """
    df = read_sdf(
        filepath,
        flavor=flavor,
        molecule_column=molecule_column,
        title_column=title_column,
        sd_data=sd_data,
        usecols=usecols,
        numeric_columns=numeric_columns,
        molecule_type=molecule_type,
        no_title=no_title,
    )
    return df.lazy()


def scan_oeb(
    filepath: FilePath,
    *,
    molecule_column: str = "Molecule",
    title_column: str | None = "Title",
    sd_data: bool = True,
    usecols: Iterable[str] | None = None,
    numeric_columns: Iterable[str] | None = None,
    molecule_type: MoleculeTypeInput = None,
    no_title: bool = False,
) -> pl.LazyFrame:
    """Lazily read molecules from an OEB file.

    Returns a LazyFrame for query optimization. Molecules are read eagerly
    but downstream operations (select, filter, etc.) are optimized lazily.

    :param filepath: Path to the OEB file.
    :param molecule_column: Name for the molecule column.
    :param title_column: Name for the title column, or None to skip.
    :param sd_data: Whether to read SD data tags as columns.
    :param usecols: Specific SD tags to read (None = all).
    :param numeric_columns: SD tag columns to convert to numeric types.
    :param molecule_type: Molecule implementation to produce.
    :param no_title: Do not read molecule titles.
    :returns: LazyFrame with molecule column and optional SD data.
    :raises FileError: If the file cannot be read.

    Example::

        lf = oep.scan_oeb("molecules.oeb")
        df = lf.collect()
    """
    df = read_oeb(
        filepath,
        molecule_column=molecule_column,
        title_column=title_column,
        sd_data=sd_data,
        usecols=usecols,
        numeric_columns=numeric_columns,
        molecule_type=molecule_type,
        no_title=no_title,
    )
    return df.lazy()


def scan_smi(
    filepath: FilePath,
    *,
    molecule_column: str = "Molecule",
    title_column: str | None = "Title",
    molecule_type: MoleculeTypeInput = None,
    no_title: bool = False,
) -> pl.LazyFrame:
    """Lazily read molecules from a SMILES file.

    Returns a LazyFrame for query optimization. Molecules are read eagerly
    but downstream operations (select, filter, etc.) are optimized lazily.

    :param filepath: Path to the SMILES file.
    :param molecule_column: Name for the molecule column.
    :param title_column: Name for the title column, or None to skip.
    :param molecule_type: Molecule implementation to produce.
    :param no_title: Do not read molecule titles.
    :returns: LazyFrame with molecule column and optional title.
    :raises FileError: If the file cannot be read.

    Example::

        lf = oep.scan_smi("molecules.smi")
        df = lf.collect()
    """
    df = read_smi(
        filepath,
        molecule_column=molecule_column,
        title_column=title_column,
        molecule_type=molecule_type,
        no_title=no_title,
    )
    return df.lazy()


def scan_parquet(
    filepath: FilePath,
    *,
    molecule_columns: Iterable[str] | None = None,
    **parquet_kwargs,
) -> pl.LazyFrame:
    """Lazily read molecules from a Parquet file.

    Returns a LazyFrame for query optimization. Molecules are read eagerly
    but downstream operations (select, filter, etc.) are optimized lazily.

    :param filepath: Path to the Parquet file.
    :param molecule_columns: Column names containing serialized molecule data (Binary type).
        This parameter is required because we cannot auto-detect which
        Binary columns were originally molecules.
    :param parquet_kwargs: Additional arguments passed to ``pl.read_parquet()``.
    :returns: LazyFrame with specified binary columns converted to MoleculeType.
    :raises ValueError: If molecule_columns is not specified (None).
    :raises FileError: If the file cannot be read.

    Example::

        lf = oep.scan_parquet("molecules.parquet", molecule_columns=["Molecule"])
        df = lf.collect()
    """
    df = read_parquet(
        filepath,
        molecule_columns=molecule_columns,
        **parquet_kwargs,
    )
    return df.lazy()


def scan_oedb(
    filepath: FilePath,
    *,
    molecule_column: str = "Molecule",
    title_column: str | None = "Title",
    sd_data: bool = True,
    usecols: Iterable[str] | None = None,
    numeric_columns: Iterable[str] | None = None,
    molecule_type: MoleculeTypeInput = None,
    no_title: bool = False,
) -> pl.LazyFrame:
    """Lazily read molecules from an OEDB file.

    Returns a LazyFrame for query optimization. Molecules are read eagerly
    but downstream operations (select, filter, etc.) are optimized lazily.

    :param filepath: Path to the OEDB file.
    :param molecule_column: Name for the molecule column.
    :param title_column: Name for the title column, or None to skip.
    :param sd_data: Whether to read SD data tags as columns.
    :param usecols: Specific SD tags to read (None = all).
    :param numeric_columns: SD tag columns to convert to numeric types.
    :param molecule_type: Molecule implementation to produce.
    :param no_title: Do not read molecule titles.
    :returns: LazyFrame with molecule column and optional SD data.
    :raises FileError: If the file cannot be read.

    Example::

        lf = oep.scan_oedb("molecules.oedb")
        df = lf.collect()
    """
    df = read_oedb(
        filepath,
        molecule_column=molecule_column,
        title_column=title_column,
        sd_data=sd_data,
        usecols=usecols,
        numeric_columns=numeric_columns,
        molecule_type=molecule_type,
        no_title=no_title,
    )
    return df.lazy()


def scan_oedu(
    filepath: FilePath,
    *,
    design_unit_column: str = "DesignUnit",
    title_column: str | None = "Title",
) -> pl.LazyFrame:
    """Lazily read design units from an OEDU file.

    Returns a LazyFrame for query optimization. Design units are read eagerly
    but downstream operations (select, filter, etc.) are optimized lazily.

    :param filepath: Path to the OEDU file.
    :param design_unit_column: Name for the design unit column.
    :param title_column: Name for the title column, or None to skip.
    :returns: LazyFrame with design unit column and optional title.
    :raises FileError: If the file cannot be read.

    Example::

        lf = oep.scan_oedu("complexes.oedu")
        df = lf.collect()
    """
    df = read_oedu(
        filepath,
        design_unit_column=design_unit_column,
        title_column=title_column,
    )
    return df.lazy()


def scan_molecule_csv(
    filepath: FilePath,
    *,
    smiles_column: str,
    molecule_column: str = "Molecule",
    drop_smiles: bool = True,
    **csv_kwargs,
) -> pl.LazyFrame:
    """Lazily read a CSV file and convert a SMILES column to molecules.

    Returns a LazyFrame for query optimization. Molecules are read eagerly
    but downstream operations (select, filter, etc.) are optimized lazily.

    :param filepath: Path to the CSV file.
    :param smiles_column: Name of the column containing SMILES strings.
    :param molecule_column: Name for the new molecule column.
    :param drop_smiles: If True (default), remove the original SMILES column.
    :param csv_kwargs: Additional arguments passed to ``pl.read_csv()``.
    :returns: LazyFrame with molecule column and other CSV data.
    :raises FileError: If the file cannot be read.

    Example::

        lf = oep.scan_molecule_csv("data.csv", smiles_column="SMILES")
        df = lf.collect()
    """
    df = read_molecule_csv(
        filepath,
        smiles_column=smiles_column,
        molecule_column=molecule_column,
        drop_smiles=drop_smiles,
        **csv_kwargs,
    )
    return df.lazy()
