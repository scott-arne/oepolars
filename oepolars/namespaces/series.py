"""Series namespace for OpenEye operations."""

import polars as pl
from openeye import oechem

from oepolars.exceptions import InvalidSMARTS


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

    def to_smiles(self, flavor: int = oechem.OESMILESFlag_ISOMERIC) -> pl.Series:
        """
        Generate SMILES strings from molecules.

        Parameters
        ----------
        flavor
            SMILES generation options (bitmask from oechem.OESMILESFlag_*).
            Default is OESMILESFlag_ISOMERIC for isomeric SMILES.

        Returns
        -------
        pl.Series
            String Series containing SMILES representations.

        Raises
        ------
        TypeError
            If the Series is not a MoleculeType.

        Examples
        --------
        >>> s.oe.to_smiles()
        shape: (3,)
        Series: 'mol' [str]
        [
            "CCO"
            "c1ccccc1"
            "CC(=O)O"
        ]
        """
        if not _is_molecule_type(self._series.dtype):
            raise TypeError(
                f"to_smiles() requires MoleculeType, got {self._series.dtype}"
            )

        def mol_to_smiles(mol: oechem.OEMolBase | None) -> str | None:
            if mol is None:
                return None
            return oechem.OECreateSmiString(mol, flavor)

        return self._series.map_elements(
            mol_to_smiles,
            return_dtype=pl.String,
        )

    def subsearch(
        self,
        pattern: str | oechem.OESubSearch,
        *,
        adjustH: bool = False,
    ) -> pl.Series:
        """
        Perform substructure search on molecules.

        Parameters
        ----------
        pattern
            SMARTS pattern string or OESubSearch object.
        adjustH
            If True, adjust implicit/explicit hydrogens to match query.

        Returns
        -------
        pl.Series
            Boolean Series where True indicates a substructure match.

        Raises
        ------
        TypeError
            If the Series is not a MoleculeType.
        InvalidSMARTS
            If the pattern is an invalid SMARTS string.

        Examples
        --------
        >>> s.oe.subsearch("c1ccccc1")  # Find molecules with benzene ring
        shape: (4,)
        Series: 'mol' [bool]
        [
            false
            true
            false
            true
        ]
        """
        if not _is_molecule_type(self._series.dtype):
            raise TypeError(
                f"subsearch() requires MoleculeType, got {self._series.dtype}"
            )

        # Create OESubSearch object if string pattern provided
        if isinstance(pattern, str):
            ss = oechem.OESubSearch()
            if not ss.Init(pattern):
                raise InvalidSMARTS(f"Invalid SMARTS pattern: {pattern}")
        else:
            ss = pattern

        def match_mol(mol: oechem.OEMolBase) -> bool:
            if adjustH:
                oechem.OEPrepareSearch(mol, ss)
            return ss.SingleMatch(mol)

        return self._series.map_elements(
            match_mol,
            return_dtype=pl.Boolean,
        ).fill_null(False)
