"""File readers for molecular data formats."""

import logging
from collections.abc import Iterable
from pathlib import Path

import polars as pl
from openeye import oechem

from oepolars.exceptions import FileError
from oepolars.types import DesignUnitType, MoleculeType
from oepolars.util import bytes_to_mol

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


def read_molecule_csv(
    filepath: FilePath,
    *,
    smiles_column: str,
    molecule_column: str = "Molecule",
    drop_smiles: bool = True,
    **csv_kwargs,
) -> pl.DataFrame:
    """
    Read a CSV file and convert a SMILES column to molecules.

    Uses Polars' native CSV reading for efficiency, then converts
    the specified SMILES column to molecules.

    Parameters
    ----------
    filepath
        Path to the CSV file.
    smiles_column
        Name of the column containing SMILES strings.
    molecule_column
        Name for the new molecule column.
    drop_smiles
        If True (default), remove the original SMILES column.
    **csv_kwargs
        Additional arguments passed to pl.read_csv().

    Returns
    -------
    pl.DataFrame
        DataFrame with molecule column and other CSV data.

    Raises
    ------
    FileError
        If the file cannot be read.

    Examples
    --------
    >>> df = oep.read_molecule_csv("data.csv", smiles_column="SMILES")
    """
    path = Path(filepath)
    if not path.exists():
        raise FileError(f"File not found: {filepath}")

    # Read CSV with Polars
    df = pl.read_csv(path, **csv_kwargs)

    # Parse SMILES strings to molecules
    smiles_values = df[smiles_column].to_list()
    molecules = []
    for smiles in smiles_values:
        if smiles is None:
            molecules.append(None)
        else:
            mol = oechem.OEMol()
            if oechem.OEParseSmiles(mol, smiles):
                molecules.append(mol)
            else:
                log.warning("Failed to parse SMILES: %s", smiles)
                molecules.append(None)

    # Add the molecule column
    mol_series = pl.Series(molecule_column, molecules, dtype=MoleculeType())
    df = df.with_columns(mol_series)

    # Optionally drop the SMILES column
    if drop_smiles and smiles_column in df.columns:
        df = df.drop(smiles_column)

    log.debug(f"Read {len(df)} rows from {filepath}")

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


def read_oedb(
    filepath: FilePath,
    *,
    molecule_column: str = "Molecule",
    title_column: str | None = "Title",
    sd_data: bool = True,
    usecols: Iterable[str] | None = None,
    numeric_columns: Iterable[str] | None = None,
) -> pl.DataFrame:
    """
    Read molecules from an OEDB file into a DataFrame.

    OEDB is OpenEye's database format for molecular records.

    Parameters
    ----------
    filepath
        Path to the OEDB file.
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
    >>> df = oep.read_oedb("molecules.oedb")
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


def read_oedu(
    filepath: FilePath,
    *,
    design_unit_column: str = "DesignUnit",
    title_column: str | None = "Title",
) -> pl.DataFrame:
    """
    Read design units from an OEDU file into a DataFrame.

    OEDU files contain OpenEye design units (protein-ligand complexes).

    Parameters
    ----------
    filepath
        Path to the OEDU file.
    design_unit_column
        Name for the design unit column.
    title_column
        Name for the title column, or None to skip.

    Returns
    -------
    pl.DataFrame
        DataFrame with design unit column and optional title.

    Raises
    ------
    FileError
        If the file cannot be read.

    Examples
    --------
    >>> df = oep.read_oedu("complexes.oedu")
    """
    path = Path(filepath)
    if not path.exists():
        raise FileError(f"File not found: {filepath}")

    design_units = []
    titles = []

    ifs = oechem.oeifstream(str(path))

    du = oechem.OEDesignUnit()
    while oechem.OEReadDesignUnit(ifs, du):
        design_units.append(oechem.OEDesignUnit(du))
        titles.append(du.GetTitle())

    ifs.close()

    log.debug(f"Read {len(design_units)} design units from {filepath}")

    data: dict[str, pl.Series] = {
        design_unit_column: pl.Series(design_units, dtype=DesignUnitType()),
    }

    if title_column is not None:
        data[title_column] = pl.Series(titles, dtype=pl.String)

    return pl.DataFrame(data)


def read_parquet(
    filepath: FilePath,
    *,
    molecule_columns: Iterable[str] | None = None,
    **parquet_kwargs,
) -> pl.DataFrame:
    """
    Read a Parquet file and reconstruct molecule columns from serialized binary data.

    This function reads Parquet files that were written with
    `df.chem.write_parquet()`, which serializes MoleculeType columns
    to binary OEB format. This function converts those binary columns
    back to MoleculeType.

    Parameters
    ----------
    filepath
        Path to the Parquet file.
    molecule_columns
        Column names containing serialized molecule data (Binary type).
        This parameter is required because we cannot auto-detect which
        Binary columns were originally molecules.
    **parquet_kwargs
        Additional arguments passed to pl.read_parquet().

    Returns
    -------
    pl.DataFrame
        DataFrame with specified binary columns converted to MoleculeType.

    Raises
    ------
    ValueError
        If molecule_columns is not specified (None).
    FileError
        If the file cannot be read.

    Examples
    --------
    >>> # Write molecules to Parquet
    >>> df.chem.write_parquet("molecules.parquet")
    >>> # Read them back
    >>> df_back = oep.read_parquet("molecules.parquet", molecule_columns=["Molecule"])
    >>> df_back.schema["Molecule"].ext_name()
    'molecule'
    """
    if molecule_columns is None:
        raise ValueError(
            "molecule_columns must be specified. Cannot auto-detect which "
            "Binary columns were originally molecules. Pass an empty list "
            "if no columns should be converted."
        )

    path = Path(filepath)
    if not path.exists():
        raise FileError(f"File not found: {filepath}")

    # Read the Parquet file
    df = pl.read_parquet(path, **parquet_kwargs)

    # Convert molecule_columns to list
    mol_cols = list(molecule_columns)

    # If empty list, just return the DataFrame as-is
    if not mol_cols:
        return df

    # Reconstruct each molecule column
    for col in mol_cols:
        if col not in df.columns:
            raise KeyError(f"Column not found: {col}")

        # Get the binary data and convert to molecules
        binary_values = df[col].to_list()
        molecules = [bytes_to_mol(data) for data in binary_values]

        # Replace the column with MoleculeType
        df = df.with_columns(
            pl.Series(col, molecules, dtype=MoleculeType())
        )

    log.debug(f"Read {len(df)} rows from {filepath}")

    return df
