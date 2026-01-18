"""File readers for molecular data formats."""

import logging
from collections.abc import Iterable
from pathlib import Path

import polars as pl
from openeye import oechem

from oepolars.exceptions import FileError
from oepolars.types import MoleculeType

log = logging.getLogger("oepolars")

# Type alias for file paths
FilePath = str | Path


def read_sdf(
    filepath: FilePath,
    *,
    flavor: int | None = None,
    molecule_column: str = "Molecule",
    title_column: str | None = "Title",
    sd_data: bool = True,
    usecols: Iterable[str] | None = None,
    numeric_columns: Iterable[str] | None = None,
) -> pl.DataFrame:
    """
    Read molecules from an SDF file into a DataFrame.

    Parameters
    ----------
    filepath
        Path to the SDF file.
    flavor
        OpenEye flavor flags for reading (OEIFlavor_SDF_*).
    molecule_column
        Name for the molecule column.
    title_column
        Name for the title column, or None to skip.
    sd_data
        Whether to read SD data tags as columns.
    usecols
        Specific SD tags to read (None = all).
    numeric_columns
        SD tag columns to convert to numeric types. Invalid values that cannot
        be converted to numeric are silently converted to null.

    Returns
    -------
    pl.DataFrame
        DataFrame with molecule column and optional SD data.

    Raises
    ------
    FileError
        If the file cannot be read.

    Examples
    --------
    >>> df = oep.read_sdf("molecules.sdf")
    >>> df = oep.read_sdf("molecules.sdf", molecule_column="Mol")
    """
    path = Path(filepath)
    if not path.exists():
        raise FileError(f"File not found: {filepath}")

    # Prepare usecols as a set if provided
    usecols_set = set(usecols) if usecols is not None else None

    # Read molecules
    molecules = []
    titles = []
    sd_data_dict: dict[str, list] = {}

    ifs = oechem.oemolistream(str(path))
    if flavor is not None:
        ifs.SetFlavor(oechem.OEFormat_SDF, flavor)

    mol = oechem.OEGraphMol()
    while oechem.OEReadMolecule(ifs, mol):
        # Make a copy to store (OEMol preserves SD data)
        molecules.append(oechem.OEMol(mol))
        titles.append(mol.GetTitle())

        # Read SD data
        if sd_data:
            for dp in oechem.OEGetSDDataPairs(mol):
                tag = dp.GetTag()
                value = dp.GetValue()

                # Skip if not in usecols
                if usecols_set is not None and tag not in usecols_set:
                    continue

                if tag not in sd_data_dict:
                    # Backfill with None for previous molecules
                    sd_data_dict[tag] = [None] * (len(molecules) - 1)

                sd_data_dict[tag].append(value)

            # Fill missing tags with None
            for tag in sd_data_dict:
                if len(sd_data_dict[tag]) < len(molecules):
                    sd_data_dict[tag].append(None)

    ifs.close()

    log.debug(f"Read {len(molecules)} molecules from {filepath}")

    # Build DataFrame
    data: dict[str, pl.Series] = {
        molecule_column: pl.Series(molecules, dtype=MoleculeType()),
    }

    if title_column is not None:
        data[title_column] = pl.Series(titles, dtype=pl.String)

    # Add SD data columns
    for tag, values in sd_data_dict.items():
        data[tag] = pl.Series(values, dtype=pl.String)

    df = pl.DataFrame(data)

    # Convert numeric columns
    if numeric_columns is not None:
        for col in numeric_columns:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).cast(pl.Float64, strict=False)
                )

    return df


def read_oeb(
    filepath: FilePath,
    *,
    molecule_column: str = "Molecule",
    title_column: str | None = "Title",
    sd_data: bool = True,
    usecols: Iterable[str] | None = None,
    numeric_columns: Iterable[str] | None = None,
) -> pl.DataFrame:
    """
    Read molecules from an OEB file into a DataFrame.

    OEB is OpenEye's binary format - fast and lossless.

    Parameters
    ----------
    filepath
        Path to the OEB file.
    molecule_column
        Name for the molecule column.
    title_column
        Name for the title column, or None to skip.
    sd_data
        Whether to read SD data tags as columns.
    usecols
        Specific SD tags to read (None = all).
    numeric_columns
        SD tag columns to convert to numeric types. Invalid values that cannot
        be converted to numeric are silently converted to null.

    Returns
    -------
    pl.DataFrame
        DataFrame with molecule column and optional SD data.

    Raises
    ------
    FileError
        If the file cannot be read.

    Examples
    --------
    >>> df = oep.read_oeb("molecules.oeb")
    """
    path = Path(filepath)
    if not path.exists():
        raise FileError(f"File not found: {filepath}")

    # Prepare usecols as a set if provided
    usecols_set = set(usecols) if usecols is not None else None

    # Read molecules
    molecules = []
    titles = []
    sd_data_dict: dict[str, list] = {}

    ifs = oechem.oemolistream(str(path))

    mol = oechem.OEGraphMol()
    while oechem.OEReadMolecule(ifs, mol):
        # Make a copy to store (OEMol preserves SD data)
        molecules.append(oechem.OEMol(mol))
        titles.append(mol.GetTitle())

        # Read SD data
        if sd_data:
            for dp in oechem.OEGetSDDataPairs(mol):
                tag = dp.GetTag()
                value = dp.GetValue()

                # Skip if not in usecols
                if usecols_set is not None and tag not in usecols_set:
                    continue

                if tag not in sd_data_dict:
                    # Backfill with None for previous molecules
                    sd_data_dict[tag] = [None] * (len(molecules) - 1)

                sd_data_dict[tag].append(value)

            # Fill missing tags with None
            for tag in sd_data_dict:
                if len(sd_data_dict[tag]) < len(molecules):
                    sd_data_dict[tag].append(None)

    ifs.close()

    log.debug(f"Read {len(molecules)} molecules from {filepath}")

    # Build DataFrame
    data: dict[str, pl.Series] = {
        molecule_column: pl.Series(molecules, dtype=MoleculeType()),
    }

    if title_column is not None:
        data[title_column] = pl.Series(titles, dtype=pl.String)

    # Add SD data columns
    for tag, values in sd_data_dict.items():
        data[tag] = pl.Series(values, dtype=pl.String)

    df = pl.DataFrame(data)

    # Convert numeric columns
    if numeric_columns is not None:
        for col in numeric_columns:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).cast(pl.Float64, strict=False)
                )

    return df


def read_smi(
    filepath: FilePath,
    *,
    molecule_column: str = "Molecule",
    title_column: str | None = "Title",
) -> pl.DataFrame:
    """
    Read molecules from a SMILES file into a DataFrame.

    SMILES files contain one molecule per line with optional title.
    Format: SMILES [whitespace] [title]

    Parameters
    ----------
    filepath
        Path to the SMILES file.
    molecule_column
        Name for the molecule column.
    title_column
        Name for the title column, or None to skip.

    Returns
    -------
    pl.DataFrame
        DataFrame with molecule column and optional title.

    Raises
    ------
    FileError
        If the file cannot be read.

    Examples
    --------
    >>> df = oep.read_smi("molecules.smi")
    """
    path = Path(filepath)
    if not path.exists():
        raise FileError(f"File not found: {filepath}")

    molecules = []
    titles = []

    ifs = oechem.oemolistream(str(path))

    mol = oechem.OEGraphMol()
    while oechem.OEReadMolecule(ifs, mol):
        molecules.append(oechem.OEMol(mol))
        titles.append(mol.GetTitle())

    ifs.close()

    log.debug(f"Read {len(molecules)} molecules from {filepath}")

    data: dict[str, pl.Series] = {
        molecule_column: pl.Series(molecules, dtype=MoleculeType()),
    }

    if title_column is not None:
        data[title_column] = pl.Series(titles, dtype=pl.String)

    return pl.DataFrame(data)
