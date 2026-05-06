"""File readers for molecular data formats."""

import logging
from collections.abc import Iterable
from pathlib import Path

import polars as pl
from openeye import oechem

from oepolars.exceptions import FileError
from oepolars.types import DesignUnitType, MoleculeType
from oepolars.types.molecule import (
    MoleculeTypeInput,
    _clear_mol_titles,
    _coerce_molecule,
    _new_molecule,
    _normalize_molecule_type,
)
from oepolars.util import bytes_to_mol

log = logging.getLogger("oepolars")

# Type alias for file paths
FilePath = str | Path


def _validate_path(filepath: FilePath) -> Path:
    """Validate that a file path exists and return as Path object.

    :param filepath: Path to validate.
    :returns: Validated Path object.
    :raises FileError: If the file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileError(f"File not found: {filepath}")
    return path


def _collect_sd_data(
    mol: oechem.OEMolBase,
    sd_data_dict: dict[str, list],
    molecule_count: int,
    usecols_set: set[str] | None,
) -> None:
    """Collect SD data from a molecule into the sd_data_dict.

    SD data may be stored on the molecule (OEB format) or on the active
    conformer (SDF format), so we check both locations.

    :param mol: Molecule to extract SD data from.
    :param sd_data_dict: Dictionary to store SD data (modified in place).
    :param molecule_count: Current count of molecules (for backfilling).
    :param usecols_set: Set of column names to include, or None for all.
    """
    # Try molecule-level SD data first (OEB format), then conformer-level (SDF format)
    sd_pairs = list(oechem.OEGetSDDataPairs(mol))
    if not sd_pairs and isinstance(mol, oechem.OEMol):
        sd_pairs = list(oechem.OEGetSDDataPairs(mol.GetActive()))

    for dp in sd_pairs:
        tag = dp.GetTag()
        value = dp.GetValue()

        # Skip if not in usecols
        if usecols_set is not None and tag not in usecols_set:
            continue

        if tag not in sd_data_dict:
            # Backfill with None for previous molecules
            sd_data_dict[tag] = [None] * (molecule_count - 1)

        sd_data_dict[tag].append(value)

    # Fill missing tags with None
    for tag in sd_data_dict:
        if len(sd_data_dict[tag]) < molecule_count:
            sd_data_dict[tag].append(None)


def _copy_stream_molecule(
    mol: oechem.OEMolBase,
    molecule_type: MoleculeTypeInput,
    *,
    no_title: bool,
) -> oechem.OEMolBase:
    """Copy a stream-owned molecule for DataFrame storage.

    :param mol: Stream-owned molecule to copy.
    :param molecule_type: Molecule type selector.
    :param no_title: Clear molecule titles on the returned value.
    :returns: Independent molecule instance for DataFrame storage.
    """
    copied = oechem.OEMol(mol)
    result = _coerce_molecule(copied, molecule_type)
    if no_title:
        _clear_mol_titles(result)
    return result


def _build_molecule_dataframe(
    molecules: list,
    titles: list | None,
    sd_data_dict: dict[str, list],
    molecule_column: str,
    title_column: str | None,
    numeric_columns: Iterable[str] | None,
) -> pl.DataFrame:
    """Build a DataFrame from molecules, titles, and SD data.

    :param molecules: List of molecule objects.
    :param titles: List of titles, or None if not reading titles.
    :param sd_data_dict: Dictionary of SD tag data.
    :param molecule_column: Name for the molecule column.
    :param title_column: Name for the title column, or None to skip.
    :param numeric_columns: Columns to convert to numeric types.
    :returns: Constructed DataFrame.
    """
    data: dict[str, pl.Series] = {
        molecule_column: pl.Series(molecules, dtype=MoleculeType()),
    }

    if titles is not None and title_column is not None:
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


def read_sdf(
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
) -> pl.DataFrame:
    """Read molecules from an SDF file into a DataFrame.

    :param filepath: Path to the SDF file.
    :param flavor: OpenEye flavor flags for reading (OEIFlavor_SDF_*).
    :param molecule_column: Name for the molecule column.
    :param title_column: Name for the title column, or None to skip.
    :param sd_data: Whether to read SD data tags as columns.
    :param usecols: Specific SD tags to read (None = all).
    :param numeric_columns: SD tag columns to convert to numeric types. Invalid values
        that cannot be converted to numeric are silently converted to null.
    :param molecule_type: Molecule implementation to produce.
    :param no_title: Do not read molecule titles.
    :returns: DataFrame with molecule column and optional SD data.
    :raises FileError: If the file cannot be read.

    Example::

        df = oep.read_sdf("molecules.sdf")
        df = oep.read_sdf("molecules.sdf", molecule_column="Mol")
    """
    path = _validate_path(filepath)
    usecols_set = set(usecols) if usecols is not None else None
    normalized_molecule_type = _normalize_molecule_type(molecule_type)

    molecules: list[oechem.OEMolBase] = []
    titles: list[str] | None = None if no_title else []
    sd_data_dict: dict[str, list] = {}

    with oechem.oemolistream(str(path)) as ifs:
        if flavor is not None:
            ifs.SetFlavor(oechem.OEFormat_SDF, flavor)

        for mol in ifs.GetOEMols():
            molecules.append(
                _copy_stream_molecule(
                    mol,
                    normalized_molecule_type,
                    no_title=no_title,
                )
            )

            if titles is not None:
                titles.append(mol.GetTitle())

            if sd_data:
                _collect_sd_data(mol, sd_data_dict, len(molecules), usecols_set)

    log.debug(f"Read {len(molecules)} molecules from {filepath}")

    return _build_molecule_dataframe(
        molecules, titles, sd_data_dict, molecule_column, title_column, numeric_columns
    )


def read_oeb(
    filepath: FilePath,
    *,
    molecule_column: str = "Molecule",
    title_column: str | None = "Title",
    sd_data: bool = True,
    usecols: Iterable[str] | None = None,
    numeric_columns: Iterable[str] | None = None,
    molecule_type: MoleculeTypeInput = None,
    no_title: bool = False,
) -> pl.DataFrame:
    """Read molecules from an OEB file into a DataFrame.

    OEB is OpenEye's binary format - fast and lossless.

    :param filepath: Path to the OEB file.
    :param molecule_column: Name for the molecule column.
    :param title_column: Name for the title column, or None to skip.
    :param sd_data: Whether to read SD data tags as columns.
    :param usecols: Specific SD tags to read (None = all).
    :param numeric_columns: SD tag columns to convert to numeric types. Invalid values
        that cannot be converted to numeric are silently converted to null.
    :param molecule_type: Molecule implementation to produce.
    :param no_title: Do not read molecule titles.
    :returns: DataFrame with molecule column and optional SD data.
    :raises FileError: If the file cannot be read.

    Example::

        df = oep.read_oeb("molecules.oeb")
    """
    path = _validate_path(filepath)
    usecols_set = set(usecols) if usecols is not None else None
    normalized_molecule_type = _normalize_molecule_type(molecule_type)

    molecules: list[oechem.OEMolBase] = []
    titles: list[str] | None = None if no_title else []
    sd_data_dict: dict[str, list] = {}

    with oechem.oemolistream(str(path)) as ifs:
        for mol in ifs.GetOEMols():
            molecules.append(
                _copy_stream_molecule(
                    mol,
                    normalized_molecule_type,
                    no_title=no_title,
                )
            )

            if titles is not None:
                titles.append(mol.GetTitle())

            if sd_data:
                _collect_sd_data(mol, sd_data_dict, len(molecules), usecols_set)

    log.debug(f"Read {len(molecules)} molecules from {filepath}")

    return _build_molecule_dataframe(
        molecules, titles, sd_data_dict, molecule_column, title_column, numeric_columns
    )


def read_molecule_csv(
    filepath: FilePath,
    *,
    smiles_column: str,
    molecule_column: str = "Molecule",
    drop_smiles: bool = True,
    molecule_type: MoleculeTypeInput = None,
    no_title: bool = False,
    **csv_kwargs,
) -> pl.DataFrame:
    """Read a CSV file and convert a SMILES column to molecules.

    Uses Polars' native CSV reading for efficiency, then converts
    the specified SMILES column to molecules.

    :param filepath: Path to the CSV file.
    :param smiles_column: Name of the column containing SMILES strings.
    :param molecule_column: Name for the new molecule column.
    :param drop_smiles: If True (default), remove the original SMILES column.
    :param molecule_type: Molecule implementation to produce.
    :param no_title: Clear molecule titles on returned values.
    :param csv_kwargs: Additional arguments passed to ``pl.read_csv()``.
    :returns: DataFrame with molecule column and other CSV data.
    :raises FileError: If the file cannot be read.

    Example::

        df = oep.read_molecule_csv("data.csv", smiles_column="SMILES")
    """
    path = _validate_path(filepath)
    normalized_molecule_type = _normalize_molecule_type(molecule_type)

    # Read CSV with Polars
    df = pl.read_csv(path, **csv_kwargs)

    # Parse SMILES strings to molecules
    smiles_values = df[smiles_column].to_list()
    molecules: list[oechem.OEMolBase | None] = []
    for smiles in smiles_values:
        if smiles is None:
            molecules.append(None)
        else:
            mol = _new_molecule(normalized_molecule_type)
            if oechem.OEParseSmiles(mol, smiles):
                if no_title:
                    _clear_mol_titles(mol)
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
    molecule_type: MoleculeTypeInput = None,
    no_title: bool = False,
) -> pl.DataFrame:
    """Read molecules from a SMILES file into a DataFrame.

    SMILES files contain one molecule per line with optional title.
    Format: SMILES [whitespace] [title]

    :param filepath: Path to the SMILES file.
    :param molecule_column: Name for the molecule column.
    :param title_column: Name for the title column, or None to skip.
    :param molecule_type: Molecule implementation to produce.
    :param no_title: Do not read molecule titles.
    :returns: DataFrame with molecule column and optional title.
    :raises FileError: If the file cannot be read.

    Example::

        df = oep.read_smi("molecules.smi")
    """
    path = _validate_path(filepath)
    normalized_molecule_type = _normalize_molecule_type(molecule_type)

    molecules: list[oechem.OEMolBase] = []
    titles: list[str] | None = None if no_title else []

    with oechem.oemolistream(str(path)) as ifs:
        for mol in ifs.GetOEMols():
            molecules.append(
                _copy_stream_molecule(
                    mol,
                    normalized_molecule_type,
                    no_title=no_title,
                )
            )

            if titles is not None:
                titles.append(mol.GetTitle())

    log.debug(f"Read {len(molecules)} molecules from {filepath}")

    return _build_molecule_dataframe(
        molecules, titles, {}, molecule_column, title_column, None
    )


def read_oedb(
    filepath: FilePath,
    *,
    molecule_column: str = "Molecule",
    title_column: str | None = "Title",
    sd_data: bool = True,
    usecols: Iterable[str] | None = None,
    numeric_columns: Iterable[str] | None = None,
    molecule_type: MoleculeTypeInput = None,
    no_title: bool = False,
) -> pl.DataFrame:
    """Read molecules from an OEDB file into a DataFrame.

    OEDB is OpenEye's database format for molecular records.

    :param filepath: Path to the OEDB file.
    :param molecule_column: Name for the molecule column.
    :param title_column: Name for the title column, or None to skip.
    :param sd_data: Whether to read SD data tags as columns.
    :param usecols: Specific SD tags to read (None = all).
    :param numeric_columns: SD tag columns to convert to numeric types. Invalid values
        that cannot be converted to numeric are silently converted to null.
    :param molecule_type: Molecule implementation to produce.
    :param no_title: Do not read molecule titles.
    :returns: DataFrame with molecule column and optional SD data.
    :raises FileError: If the file cannot be read.

    Example::

        df = oep.read_oedb("molecules.oedb")
    """
    path = _validate_path(filepath)
    usecols_set = set(usecols) if usecols is not None else None
    normalized_molecule_type = _normalize_molecule_type(molecule_type)

    molecules: list[oechem.OEMolBase] = []
    titles: list[str] | None = None if no_title else []
    sd_data_dict: dict[str, list] = {}

    with oechem.oemolistream() as ifs:
        # Set format explicitly to OEB for OEDB files (same binary format)
        ifs.SetFormat(oechem.OEFormat_OEB)
        if not ifs.open(str(path)):
            raise FileError(f"Failed to open file: {filepath}")

        for mol in ifs.GetOEMols():
            molecules.append(
                _copy_stream_molecule(
                    mol,
                    normalized_molecule_type,
                    no_title=no_title,
                )
            )

            if titles is not None:
                titles.append(mol.GetTitle())

            if sd_data:
                _collect_sd_data(mol, sd_data_dict, len(molecules), usecols_set)

    log.debug(f"Read {len(molecules)} molecules from {filepath}")

    return _build_molecule_dataframe(
        molecules, titles, sd_data_dict, molecule_column, title_column, numeric_columns
    )


def read_oedu(
    filepath: FilePath,
    *,
    design_unit_column: str = "DesignUnit",
    title_column: str | None = "Title",
    no_title: bool = False,
) -> pl.DataFrame:
    """Read design units from an OEDU file into a DataFrame.

    OEDU files contain OpenEye design units (protein-ligand complexes).

    :param filepath: Path to the OEDU file.
    :param design_unit_column: Name for the design unit column.
    :param title_column: Name for the title column, or None to skip.
    :param no_title: Do not read design unit titles.
    :returns: DataFrame with design unit column and optional title.
    :raises FileError: If the file cannot be read.

    Example::

        df = oep.read_oedu("complexes.oedu")
    """
    path = _validate_path(filepath)

    design_units: list[oechem.OEDesignUnit] = []
    titles: list[str] | None = None if no_title else []

    ifs = oechem.oeifstream(str(path))
    try:
        du = oechem.OEDesignUnit()
        while oechem.OEReadDesignUnit(ifs, du):
            if no_title:
                du.SetTitle("")

            design_units.append(oechem.OEDesignUnit(du))

            if titles is not None:
                titles.append(du.GetTitle())
    finally:
        ifs.close()

    log.debug(f"Read {len(design_units)} design units from {filepath}")

    data: dict[str, pl.Series] = {
        design_unit_column: pl.Series(design_units, dtype=DesignUnitType()),
    }

    if titles is not None and title_column is not None:
        data[title_column] = pl.Series(titles, dtype=pl.String)

    return pl.DataFrame(data)


def read_parquet(
    filepath: FilePath,
    *,
    molecule_columns: Iterable[str] | None = None,
    **parquet_kwargs,
) -> pl.DataFrame:
    """Read a Parquet file and reconstruct molecule columns from serialized binary data.

    This function reads Parquet files that were written with
    ``df.chem.write_parquet()``, which serializes MoleculeType columns
    to binary OEB format. This function converts those binary columns
    back to MoleculeType.

    :param filepath: Path to the Parquet file.
    :param molecule_columns: Column names containing serialized molecule data (Binary type).
        This parameter is required because we cannot auto-detect which
        Binary columns were originally molecules.
    :param parquet_kwargs: Additional arguments passed to ``pl.read_parquet()``.
    :returns: DataFrame with specified binary columns converted to MoleculeType.
    :raises ValueError: If molecule_columns is not specified (None).
    :raises FileError: If the file cannot be read.

    Example::

        # Write molecules to Parquet
        df.chem.write_parquet("molecules.parquet")
        # Read them back
        df_back = oep.read_parquet("molecules.parquet", molecule_columns=["Molecule"])
        df_back.schema["Molecule"].ext_name()  # Returns 'molecule'
    """
    if molecule_columns is None:
        raise ValueError(
            "molecule_columns must be specified. Cannot auto-detect which "
            "Binary columns were originally molecules. Pass an empty list "
            "if no columns should be converted."
        )

    path = _validate_path(filepath)

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
