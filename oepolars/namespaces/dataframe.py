"""DataFrame namespace for OpenEye operations."""

import logging
from collections.abc import Iterable
from pathlib import Path

import polars as pl
from openeye import oechem

from oepolars.types import MoleculeType
from oepolars.util import mol_to_bytes

log = logging.getLogger("oepolars")


def _is_molecule_type(dtype: pl.DataType) -> bool:
    """Check if dtype is MoleculeType."""
    return hasattr(dtype, "ext_name") and dtype.ext_name() == "molecule"


def _is_design_unit_type(dtype: pl.DataType) -> bool:
    """Check if dtype is DesignUnitType."""
    return hasattr(dtype, "ext_name") and dtype.ext_name() == "design_unit"


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
    """Namespace for OpenEye chemistry operations on Polars DataFrames.

    Accessed via ``df.chem.*``.
    """

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def as_molecule(
        self,
        columns: str | Iterable[str],
        *,
        molecule_format: int | None = None,
    ) -> pl.DataFrame:
        """Convert column(s) to MoleculeType.

        :param columns: Column name or list of column names to convert.
        :param molecule_format: OpenEye format constant for parsing. Default assumes SMILES.
        :returns: DataFrame with specified columns converted to MoleculeType.
        :raises KeyError: If a specified column does not exist.

        Example::

            df = pl.DataFrame({"smiles": ["CCO", "c1ccccc1"]})
            df.chem.as_molecule("smiles")
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
        """Filter out rows with null molecules in specified columns.

        :param columns: Column name or list of column names to check for validity.
        :returns: DataFrame with rows containing null molecules removed.
        :raises KeyError: If a specified column does not exist.

        Example::

            df.chem.filter_valid("Molecule")
            df.chem.filter_valid(["Molecule", "Ligand"])
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
        """Auto-detect and convert molecule columns.

        Samples values from Object and String columns to detect OpenEye molecule
        objects (OEMolBase subclasses) and SMILES strings (validated by parsing).

        :param sample_size: Maximum number of non-null values to sample per column.
        :returns: DataFrame with detected molecule columns converted.

        Example::

            df = pl.DataFrame({"smiles": ["CCO", "c1ccccc1"]})
            df.chem.detect_molecule_columns()
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
        """Write molecules to an SDF file.

        :param filepath: Path to output SDF file.
        :param molecule_column: Name of the column containing molecules.
        :param title_column: Column to use for molecule titles.
        :param sd_columns: Columns to write as SD data tags.
        :param flavor: OpenEye flavor flags for writing.
        :raises KeyError: If molecule_column does not exist.
        :raises TypeError: If molecule_column is not MoleculeType.

        Example::

            df.chem.write_sdf("output.sdf", molecule_column="Molecule")
            df.chem.write_sdf("output.sdf", molecule_column="Mol", sd_columns=["MW", "LogP"])
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
        """Write molecules to a SMILES file.

        :param filepath: Path to output SMILES file.
        :param molecule_column: Name of the column containing molecules.
        :param title_column: Column to use for molecule titles.
        :param flavor: OpenEye flavor flags for SMILES writing.
        :raises KeyError: If molecule_column does not exist.
        :raises TypeError: If molecule_column is not MoleculeType.

        Example::

            df.chem.write_smi("output.smi", molecule_column="Molecule")
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

    def write_molecule_csv(
        self,
        filepath: str | Path,
        *,
        molecule_column: str,
        smiles_column: str = "smiles",
        smiles_flavor: int = oechem.OESMILESFlag_ISOMERIC,
        drop_molecule: bool = True,
        **csv_kwargs,
    ) -> None:
        """Write DataFrame to CSV with molecules as SMILES strings.

        Converts the molecule column to SMILES and writes as CSV.

        :param filepath: Path to output CSV file.
        :param molecule_column: Name of the column containing molecules.
        :param smiles_column: Name for the SMILES column in output.
        :param smiles_flavor: OpenEye SMILES flavor flags.
        :param drop_molecule: If True, remove the molecule column from output.
        :param csv_kwargs: Additional arguments passed to ``pl.DataFrame.write_csv()``.
        :raises KeyError: If molecule_column does not exist.
        :raises TypeError: If molecule_column is not MoleculeType.

        Example::

            df.chem.write_molecule_csv("output.csv", molecule_column="Molecule")
        """
        if molecule_column not in self._df.columns:
            raise KeyError(f"Column not found: {molecule_column}")

        if not _is_molecule_type(self._df.schema[molecule_column]):
            raise TypeError(
                f"Column '{molecule_column}' is not MoleculeType, "
                f"got {self._df.schema[molecule_column]}"
            )

        # Convert molecules to SMILES
        smiles_list = []
        for mol in self._df[molecule_column].to_list():
            if mol is None:
                smiles_list.append(None)
            else:
                smiles_list.append(oechem.OECreateSmiString(mol, smiles_flavor))

        # Create output DataFrame
        df_out = self._df.clone()

        # Add SMILES column
        df_out = df_out.with_columns(
            pl.Series(smiles_column, smiles_list, dtype=pl.String)
        )

        # Drop molecule column if requested
        if drop_molecule:
            df_out = df_out.drop(molecule_column)

        # Write CSV
        df_out.write_csv(str(Path(filepath)), **csv_kwargs)

    def write_oeb(
        self,
        filepath: str | Path,
        *,
        molecule_column: str,
        title_column: str | None = None,
        sd_columns: Iterable[str] | None = None,
    ) -> None:
        """Write molecules to an OEB (OpenEye binary) file.

        :param filepath: Path to output OEB file.
        :param molecule_column: Name of the column containing molecules.
        :param title_column: Column to use for molecule titles.
        :param sd_columns: Columns to write as SD data tags.
        :raises KeyError: If molecule_column does not exist.
        :raises TypeError: If molecule_column is not MoleculeType.

        Example::

            df.chem.write_oeb("output.oeb", molecule_column="Molecule")
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
            sd_cols_list = list(sd_columns) if sd_columns else []

            for row_idx in range(len(self._df)):
                mol = self._df[molecule_column][row_idx]
                if mol is None:
                    continue

                mol_copy = oechem.OEMol(mol)

                if title_column and title_column in self._df.columns:
                    title = self._df[title_column][row_idx]
                    if title is not None:
                        mol_copy.SetTitle(str(title))

                for sd_col in sd_cols_list:
                    if sd_col in self._df.columns:
                        value = self._df[sd_col][row_idx]
                        if value is not None:
                            oechem.OESetSDData(mol_copy, sd_col, str(value))

                oechem.OEWriteMolecule(ofs, mol_copy)
        finally:
            ofs.close()

    def write_parquet(
        self,
        filepath: str | Path,
        *,
        molecule_columns: Iterable[str] | None = None,
        **parquet_kwargs,
    ) -> None:
        """Write DataFrame to Parquet with molecules serialized to binary OEB format.

        Molecule columns are converted to binary (bytes) using OEB format before
        writing, allowing the data to be stored in standard Parquet files.

        :param filepath: Path to output Parquet file.
        :param molecule_columns: Column names to serialize as molecules. If None,
            auto-detects all MoleculeType columns.
        :param parquet_kwargs: Additional arguments passed to ``pl.DataFrame.write_parquet()``.

        Example::

            df.chem.write_parquet("output.parquet")
            df.chem.write_parquet("output.parquet", molecule_columns=["Mol1"])
            df.chem.write_parquet("output.parquet", compression="snappy")
        """
        # Auto-detect molecule columns if not specified
        if molecule_columns is None:
            mol_cols = [
                col for col in self._df.columns
                if _is_molecule_type(self._df.schema[col])
            ]
        else:
            mol_cols = list(molecule_columns)
            # Validate explicitly specified columns exist and are MoleculeType
            for col in mol_cols:
                if col not in self._df.columns:
                    raise KeyError(f"Column not found: {col}")
                if not _is_molecule_type(self._df.schema[col]):
                    raise TypeError(
                        f"Column '{col}' is not MoleculeType, "
                        f"got {self._df.schema[col]}"
                    )

        # Start with a clone of the DataFrame
        df_out = self._df.clone()

        # Serialize each molecule column to binary
        for col in mol_cols:
            mol_values = df_out[col].to_list()
            serialized = [mol_to_bytes(mol) for mol in mol_values]
            df_out = df_out.with_columns(
                pl.Series(col, serialized, dtype=pl.Binary)
            )

        # Write to Parquet
        df_out.write_parquet(str(Path(filepath)), **parquet_kwargs)

    def write_oedb(
        self,
        filepath: str | Path,
        *,
        molecule_column: str,
        title_column: str | None = None,
        sd_columns: Iterable[str] | None = None,
    ) -> None:
        """Write molecules to an OEDB (OpenEye database) file.

        :param filepath: Path to output OEDB file.
        :param molecule_column: Name of the column containing molecules.
        :param title_column: Column to use for molecule titles.
        :param sd_columns: Columns to write as SD data tags.
        :raises KeyError: If molecule_column does not exist.
        :raises TypeError: If molecule_column is not MoleculeType.

        Example::

            df.chem.write_oedb("output.oedb", molecule_column="Molecule")
        """
        if molecule_column not in self._df.columns:
            raise KeyError(f"Column not found: {molecule_column}")

        if not _is_molecule_type(self._df.schema[molecule_column]):
            raise TypeError(
                f"Column '{molecule_column}' is not MoleculeType, "
                f"got {self._df.schema[molecule_column]}"
            )

        ofs = oechem.oemolostream()
        # Set format explicitly to OEB for OEDB files (same binary format)
        ofs.SetFormat(oechem.OEFormat_OEB)
        if not ofs.open(str(Path(filepath))):
            raise IOError(f"Failed to open file for writing: {filepath}")

        try:
            sd_cols_list = list(sd_columns) if sd_columns else []

            for row_idx in range(len(self._df)):
                mol = self._df[molecule_column][row_idx]
                if mol is None:
                    continue

                mol_copy = oechem.OEMol(mol)

                if title_column and title_column in self._df.columns:
                    title = self._df[title_column][row_idx]
                    if title is not None:
                        mol_copy.SetTitle(str(title))

                for sd_col in sd_cols_list:
                    if sd_col in self._df.columns:
                        value = self._df[sd_col][row_idx]
                        if value is not None:
                            oechem.OESetSDData(mol_copy, sd_col, str(value))

                oechem.OEWriteMolecule(ofs, mol_copy)
        finally:
            ofs.close()

    def write_oedu(
        self,
        filepath: str | Path,
        *,
        design_unit_column: str,
        title_column: str | None = None,
    ) -> None:
        """Write design units to an OEDU file.

        :param filepath: Path to output OEDU file.
        :param design_unit_column: Name of the column containing design units.
        :param title_column: Column to use for design unit titles.
        :raises KeyError: If design_unit_column does not exist.
        :raises TypeError: If design_unit_column is not DesignUnitType.

        Example::

            df.chem.write_oedu("output.oedu", design_unit_column="DesignUnit")
        """
        if design_unit_column not in self._df.columns:
            raise KeyError(f"Column not found: {design_unit_column}")

        if not _is_design_unit_type(self._df.schema[design_unit_column]):
            raise TypeError(
                f"Column '{design_unit_column}' is not DesignUnitType, "
                f"got {self._df.schema[design_unit_column]}"
            )

        ofs = oechem.oeofstream(str(Path(filepath)))

        try:
            for row_idx in range(len(self._df)):
                du = self._df[design_unit_column][row_idx]
                if du is None:
                    continue

                du_copy = oechem.OEDesignUnit(du)

                if title_column and title_column in self._df.columns:
                    title = self._df[title_column][row_idx]
                    if title is not None:
                        du_copy.SetTitle(str(title))

                oechem.OEWriteDesignUnit(ofs, du_copy)
        finally:
            ofs.close()
