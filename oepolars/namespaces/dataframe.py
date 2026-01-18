"""DataFrame namespace for OpenEye operations."""

import logging
from collections.abc import Iterable

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


@pl.api.register_dataframe_namespace("oe")
class DataFrameOENamespace:
    """
    Namespace for OpenEye operations on Polars DataFrames.

    Accessed via `df.oe.*`.
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
        >>> df.oe.as_molecule("smiles")
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

            # Convert column
            mol_format = (
                molecule_format if molecule_format is not None else oechem.OEFormat_SMI
            )

            if mol_format == oechem.OEFormat_SMI:
                molecules = [_parse_smiles_to_mol(val) for val in result[col].to_list()]
            else:
                # For other formats, parse from bytes
                molecules = [
                    _parse_bytes_to_mol(val, mol_format)
                    for val in result[col].to_list()
                ]

            new_col = pl.Series(col, molecules, dtype=MoleculeType())
            result = result.with_columns(new_col)

        return result
