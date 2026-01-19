"""DataFrame namespace for OpenEye operations."""

import logging
from collections.abc import Iterable
from pathlib import Path

import polars as pl
from openeye import oechem

from oepolars.types import MoleculeType

log = logging.getLogger("oepolars")


def _is_molecule_type(dtype: pl.DataType) -> bool:
    """Check if dtype is MoleculeType."""
    return hasattr(dtype, "ext_name") and dtype.ext_name() == "molecule"


def _parse_smiles_to_mol(smiles: str | None) -> oechem.OEMol | None:
    """Parse a SMILES string to an OEMol."""
    if smiles is None:
        return None
    mol = oechem.OEMol()
    if oechem.OEParseSmiles(mol, smiles):
        return mol
    log.warning("Failed to parse SMILES: %s", smiles)
    return None


def _parse_bytes_to_mol(data: bytes | None, mol_format: int) -> oechem.OEMol | None:
    """Parse bytes to an OEMol using the specified format."""
    if data is None:
        return None
    mol = oechem.OEMol()
    if oechem.OEReadMolFromBytes(mol, mol_format, False, data):
        return mol
    return None


def _is_likely_smiles(val: str) -> bool:
    """
    Check if a string is likely a SMILES string.

    Returns True if the string:
    - Parses as a valid SMILES
    - AND (contains SMILES-specific characters OR has multiple atoms)

    This prevents false positives on short text strings like "a", "b", "c".
    """
    mol = oechem.OEMol()
    if not oechem.OEParseSmiles(mol, val):
        return False

    # Check for SMILES-specific characters
    smiles_chars = set("0123456789()[]=#@+\\/-.")
    if any(c in smiles_chars for c in val):
        return True

    # Check if molecule has more than one atom (e.g., "CCO" vs "C")
    return mol.NumAtoms() > 1


@pl.api.register_dataframe_namespace("chem")
class DataFrameChemNamespace:
    """
    Namespace for OpenEye chemistry operations on Polars DataFrames.

    Accessed via `df.chem.*`.
    """

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def as_molecule(
        self,
        columns: str | Iterable[str],
        *,
        molecule_format: int | None = None,
    ) -> pl.DataFrame:
        """
        Convert column(s) to MoleculeType.

        Parameters
        ----------
        columns
            Column name or list of column names to convert.
        molecule_format
            OpenEye format constant for parsing. Default assumes SMILES.

        Returns
        -------
        pl.DataFrame
            DataFrame with specified columns converted to MoleculeType.

        Raises
        ------
        KeyError
            If a specified column does not exist.

        Examples
        --------
        >>> df = pl.DataFrame({"smiles": ["CCO", "c1ccccc1"]})
        >>> df.chem.as_molecule("smiles")
        """
        columns = [columns] if isinstance(columns, str) else list(columns)

        # Validate columns exist
        for col in columns:
            if col not in self._df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")

        result = self._df.clone()

        for col in columns:
            # Skip if already MoleculeType
            if _is_molecule_type(result.schema[col]):
                continue

            col_values = result[col].to_list()

            # Check if Object column contains OEMolBase objects
            if result.schema[col] == pl.Object:
                # Check first non-null value
                first_val = next((v for v in col_values if v is not None), None)
                if first_val is not None and isinstance(first_val, oechem.OEMolBase):
                    # Already molecule objects, just wrap in MoleculeType
                    molecules = [
                        oechem.OEMol(v) if isinstance(v, oechem.OEMolBase) else None
                        for v in col_values
                    ]
                    new_col = pl.Series(col, molecules, dtype=MoleculeType())
                    result = result.with_columns(new_col)
                    continue

            # Convert column from string/bytes
            mol_format = (
                molecule_format if molecule_format is not None else oechem.OEFormat_SMI
            )

            if mol_format == oechem.OEFormat_SMI:
                molecules = [_parse_smiles_to_mol(val) for val in col_values]
            else:
                # For other formats, parse from bytes
                molecules = [
                    _parse_bytes_to_mol(val, mol_format)
                    for val in col_values
                ]

            new_col = pl.Series(col, molecules, dtype=MoleculeType())
            result = result.with_columns(new_col)

        return result

    def filter_valid(self, columns: str | Iterable[str]) -> pl.DataFrame:
        """
        Filter out rows with null molecules in specified columns.

        Parameters
        ----------
        columns
            Column name or list of column names to check for validity.

        Returns
        -------
        pl.DataFrame
            DataFrame with rows containing null molecules removed.

        Raises
        ------
        KeyError
            If a specified column does not exist.

        Examples
        --------
        >>> df.chem.filter_valid("Molecule")
        >>> df.chem.filter_valid(["Molecule", "Ligand"])
        """
        columns = [columns] if isinstance(columns, str) else list(columns)

        # Validate columns exist
        for col in columns:
            if col not in self._df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")

        # Build filter expression: all specified columns must be not null
        filter_expr = pl.lit(True)
        for col in columns:
            filter_expr = filter_expr & pl.col(col).is_not_null()

        return self._df.filter(filter_expr)

    def detect_molecule_columns(self, *, sample_size: int = 25) -> pl.DataFrame:
        """
        Auto-detect and convert molecule columns.

        Samples values from Object and String columns to detect:
        - OpenEye molecule objects (OEMolBase subclasses)
        - SMILES strings (validated by parsing)

        Parameters
        ----------
        sample_size
            Maximum number of non-null values to sample per column.

        Returns
        -------
        pl.DataFrame
            DataFrame with detected molecule columns converted.

        Examples
        --------
        >>> df = pl.DataFrame({"smiles": ["CCO", "c1ccccc1"]})
        >>> df.chem.detect_molecule_columns()
        """
        result = self._df.clone()
        columns_to_convert: list[str] = []

        for col in result.columns:
            dtype = result.schema[col]

            # Skip if already a molecule type
            if _is_molecule_type(dtype):
                continue

            # Check Object columns for OEMolBase objects
            if dtype == pl.Object:
                sample = result[col].drop_nulls().head(sample_size)
                if len(sample) > 0:
                    # Check if majority are molecule objects
                    mol_count = sum(
                        1 for val in sample.to_list()
                        if isinstance(val, oechem.OEMolBase)
                    )
                    if mol_count > len(sample) / 2:
                        columns_to_convert.append(col)

            # Check String columns for SMILES
            elif dtype == pl.String:
                sample = result[col].drop_nulls().head(sample_size)
                if len(sample) > 0:
                    # Try to parse as SMILES with additional heuristics
                    valid_count = sum(
                        1 for val in sample.to_list()
                        if _is_likely_smiles(val)
                    )
                    # If majority parse as valid SMILES, convert
                    if valid_count > len(sample) / 2:
                        columns_to_convert.append(col)

        # Convert detected columns
        if columns_to_convert:
            result = result.chem.as_molecule(columns_to_convert)

        return result

    def write_sdf(
        self,
        filepath: str | Path,
        *,
        molecule_column: str,
        title_column: str | None = None,
        sd_columns: Iterable[str] | None = None,
        flavor: int | None = None,
    ) -> None:
        """
        Write molecules to an SDF file.

        Parameters
        ----------
        filepath
            Path to output SDF file.
        molecule_column
            Name of the column containing molecules.
        title_column
            Column to use for molecule titles (optional).
        sd_columns
            Columns to write as SD data tags (optional).
        flavor
            OpenEye flavor flags for writing.

        Raises
        ------
        KeyError
            If molecule_column does not exist.
        TypeError
            If molecule_column is not MoleculeType.

        Examples
        --------
        >>> df.chem.write_sdf("output.sdf", molecule_column="Molecule")
        >>> df.chem.write_sdf("output.sdf", molecule_column="Mol", sd_columns=["MW", "LogP"])
        """
        if molecule_column not in self._df.columns:
            raise KeyError(f"Column not found: {molecule_column}")

        if not _is_molecule_type(self._df.schema[molecule_column]):
            raise TypeError(
                f"Column '{molecule_column}' is not MoleculeType, "
                f"got {self._df.schema[molecule_column]}"
            )

        ofs = oechem.oemolostream(str(Path(filepath)))

        try:
            if flavor is not None:
                ofs.SetFlavor(oechem.OEFormat_SDF, flavor)

            sd_cols_list = list(sd_columns) if sd_columns else []

            for row_idx in range(len(self._df)):
                mol = self._df[molecule_column][row_idx]
                if mol is None:
                    continue

                # Make a copy to avoid modifying original
                mol_copy = oechem.OEMol(mol)

                # Set title if specified
                if title_column and title_column in self._df.columns:
                    title = self._df[title_column][row_idx]
                    if title is not None:
                        mol_copy.SetTitle(str(title))

                # Add SD data
                for sd_col in sd_cols_list:
                    if sd_col in self._df.columns:
                        value = self._df[sd_col][row_idx]
                        if value is not None:
                            oechem.OESetSDData(mol_copy, sd_col, str(value))

                oechem.OEWriteMolecule(ofs, mol_copy)
        finally:
            ofs.close()

    def write_smi(
        self,
        filepath: str | Path,
        *,
        molecule_column: str,
        title_column: str | None = None,
        flavor: int | None = None,
    ) -> None:
        """
        Write molecules to a SMILES file.

        Parameters
        ----------
        filepath
            Path to output SMILES file.
        molecule_column
            Name of the column containing molecules.
        title_column
            Column to use for molecule titles (optional).
        flavor
            OpenEye flavor flags for SMILES writing.

        Raises
        ------
        KeyError
            If molecule_column does not exist.
        TypeError
            If molecule_column is not MoleculeType.

        Examples
        --------
        >>> df.chem.write_smi("output.smi", molecule_column="Molecule")
        """
        if molecule_column not in self._df.columns:
            raise KeyError(f"Column not found: {molecule_column}")

        if not _is_molecule_type(self._df.schema[molecule_column]):
            raise TypeError(
                f"Column '{molecule_column}' is not MoleculeType, "
                f"got {self._df.schema[molecule_column]}"
            )

        ofs = oechem.oemolostream(str(Path(filepath)))

        try:
            if flavor is not None:
                ofs.SetFlavor(oechem.OEFormat_SMI, flavor)

            for row_idx in range(len(self._df)):
                mol = self._df[molecule_column][row_idx]
                if mol is None:
                    continue

                mol_copy = oechem.OEMol(mol)

                if title_column and title_column in self._df.columns:
                    title = self._df[title_column][row_idx]
                    if title is not None:
                        mol_copy.SetTitle(str(title))

                oechem.OEWriteMolecule(ofs, mol_copy)
        finally:
            ofs.close()
