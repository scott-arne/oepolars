"""Series namespace for OpenEye operations."""

import copy
from typing import Any

import polars as pl
from openeye import oechem, oedepict

from oepolars.exceptions import InvalidSMARTS
from oepolars.types import DesignUnitType, DisplayType, MoleculeType


# Dictionary to store metadata for Series objects, keyed by object id.
# Note: This is keyed by id() which means metadata persists as long as
# the Series object exists (no automatic cleanup when Series is garbage collected).
_series_metadata: dict[int, dict[str, Any]] = {}


def _is_molecule_type(dtype: pl.DataType) -> bool:
    """Check if dtype is MoleculeType."""
    return hasattr(dtype, "ext_name") and dtype.ext_name() == "molecule"


def _is_design_unit_type(dtype: pl.DataType) -> bool:
    """Check if dtype is DesignUnitType."""
    return hasattr(dtype, "ext_name") and dtype.ext_name() == "design_unit"


def _is_display_type(dtype: pl.DataType) -> bool:
    """Check if dtype is DisplayType."""
    return hasattr(dtype, "ext_name") and dtype.ext_name() == "display"


@pl.api.register_series_namespace("chem")
class SeriesChemNamespace:
    """
    Namespace for OpenEye chemistry operations on Polars Series.

    Accessed via `series.chem.*`.
    """

    def __init__(self, series: pl.Series) -> None:
        self._series = series

    @property
    def metadata(self) -> dict[str, Any]:
        """
        Get the metadata dictionary for this Series.

        Metadata is stored per-Series and can be used to attach arbitrary
        data (such as rendering callbacks, highlight patterns, etc.) to
        molecule Series.

        Returns
        -------
        dict[str, Any]
            A dictionary for storing metadata. Created on first access.

        Examples
        --------
        >>> s.chem.metadata["highlight"] = "c1ccccc1"
        >>> s.chem.metadata["highlight"]
        'c1ccccc1'
        """
        series_id = id(self._series)
        if series_id not in _series_metadata:
            _series_metadata[series_id] = {}
        return _series_metadata[series_id]

    def deepcopy(self) -> pl.Series:
        """
        Create a deep copy of the Series with independent object copies.

        This method copies both the objects (molecules or displays) and the
        associated metadata, creating a fully independent copy of the Series.

        Returns
        -------
        pl.Series
            A new Series with deep copies of all objects and a copy
            of the metadata dictionary.

        Raises
        ------
        TypeError
            If the Series is not a MoleculeType or DisplayType.

        Examples
        --------
        >>> copied = s.chem.deepcopy()
        >>> # Modifying original doesn't affect copy
        >>> # Metadata is also preserved
        >>> s.chem.metadata["key"] = "value"
        >>> copied = s.chem.deepcopy()
        >>> copied.chem.metadata["key"]
        'value'
        """
        dtype = self._series.dtype
        is_molecule = _is_molecule_type(dtype)
        is_display = _is_display_type(dtype)

        if not (is_molecule or is_display):
            raise TypeError(
                f"deepcopy() requires MoleculeType or DisplayType, got {dtype}"
            )

        if is_molecule:

            def copy_mol(mol: oechem.OEMolBase | None) -> oechem.OEMol | None:
                if mol is None:
                    return None
                return oechem.OEMol(mol)

            copied = [copy_mol(mol) for mol in self._series.to_list()]
            new_series = pl.Series(self._series.name, copied, dtype=MoleculeType())

        else:  # is_display

            def copy_disp(
                disp: oedepict.OE2DMolDisplay | None,
            ) -> oedepict.OE2DMolDisplay | None:
                if disp is None:
                    return None
                return oedepict.OE2DMolDisplay(disp)

            copied = [copy_disp(disp) for disp in self._series.to_list()]
            new_series = pl.Series(self._series.name, copied, dtype=DisplayType())

        # Copy metadata to the new series
        series_id = id(self._series)
        if series_id in _series_metadata:
            _series_metadata[id(new_series)] = copy.deepcopy(
                _series_metadata[series_id]
            )

        return new_series

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
        >>> s.chem.is_valid()
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
        >>> s.chem.to_smiles()
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
        >>> s.chem.subsearch("c1ccccc1")  # Find molecules with benzene ring
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

    def copy_molecules(self) -> pl.Series:
        """
        Create deep copies of all molecules in the Series.

        Returns
        -------
        pl.Series
            New Series with independent copies of each molecule.

        Raises
        ------
        TypeError
            If the Series is not a MoleculeType.

        Examples
        --------
        >>> copied = s.chem.copy_molecules()
        >>> # Modifying original doesn't affect copy
        """
        if not _is_molecule_type(self._series.dtype):
            raise TypeError(
                f"copy_molecules() requires MoleculeType, got {self._series.dtype}"
            )

        def copy_mol(mol: oechem.OEMolBase | None) -> oechem.OEMol | None:
            if mol is None:
                return None
            return oechem.OEMol(mol)

        copied = [copy_mol(mol) for mol in self._series.to_list()]
        return pl.Series(self._series.name, copied, dtype=MoleculeType())

    def copy_design_units(self) -> pl.Series:
        """
        Create deep copies of all design units in the Series.

        Returns
        -------
        pl.Series
            New Series with independent copies of each design unit.

        Raises
        ------
        TypeError
            If the Series is not a DesignUnitType.
        """
        if not _is_design_unit_type(self._series.dtype):
            raise TypeError(
                f"copy_design_units() requires DesignUnitType, got {self._series.dtype}"
            )

        def copy_du(du: oechem.OEDesignUnit | None) -> oechem.OEDesignUnit | None:
            if du is None:
                return None
            return oechem.OEDesignUnit(du)

        copied = [copy_du(du) for du in self._series.to_list()]
        return pl.Series(self._series.name, copied, dtype=DesignUnitType())

    def get_ligands(self, *, clear_titles: bool = False) -> pl.Series:
        """
        Extract ligand molecules from design units.

        Parameters
        ----------
        clear_titles
            If True, clear the titles of extracted ligands.

        Returns
        -------
        pl.Series
            MoleculeType Series containing ligand molecules.

        Raises
        ------
        TypeError
            If the Series is not a DesignUnitType.
        """
        if not _is_design_unit_type(self._series.dtype):
            raise TypeError(
                f"get_ligands() requires DesignUnitType, got {self._series.dtype}"
            )

        def extract_ligand(du: oechem.OEDesignUnit | None) -> oechem.OEMol | None:
            if du is None:
                return None
            mol = oechem.OEMol()
            if du.GetLigand(mol):
                if clear_titles:
                    mol.SetTitle("")
                return mol
            return None

        ligands = [extract_ligand(du) for du in self._series.to_list()]
        return pl.Series(self._series.name, ligands, dtype=MoleculeType())

    def get_proteins(self, *, clear_titles: bool = False) -> pl.Series:
        """
        Extract protein molecules from design units.

        Parameters
        ----------
        clear_titles
            If True, clear the titles of extracted proteins.

        Returns
        -------
        pl.Series
            MoleculeType Series containing protein molecules.

        Raises
        ------
        TypeError
            If the Series is not a DesignUnitType.
        """
        if not _is_design_unit_type(self._series.dtype):
            raise TypeError(
                f"get_proteins() requires DesignUnitType, got {self._series.dtype}"
            )

        def extract_protein(du: oechem.OEDesignUnit | None) -> oechem.OEMol | None:
            if du is None:
                return None
            mol = oechem.OEMol()
            if du.GetProtein(mol):
                if clear_titles:
                    mol.SetTitle("")
                return mol
            return None

        proteins = [extract_protein(du) for du in self._series.to_list()]
        return pl.Series(self._series.name, proteins, dtype=MoleculeType())

    def get_components(self, mask: int) -> pl.Series:
        """
        Extract components from design units by mask.

        Parameters
        ----------
        mask
            Component mask from oechem.OEDesignUnitComponents_*.

        Returns
        -------
        pl.Series
            MoleculeType Series containing extracted components.

        Raises
        ------
        TypeError
            If the Series is not a DesignUnitType.
        """
        if not _is_design_unit_type(self._series.dtype):
            raise TypeError(
                f"get_components() requires DesignUnitType, got {self._series.dtype}"
            )

        def extract_components(du: oechem.OEDesignUnit | None) -> oechem.OEMol | None:
            if du is None:
                return None
            mol = oechem.OEMol()
            if du.GetComponents(mol, mask):
                return mol
            return None

        components = [extract_components(du) for du in self._series.to_list()]
        return pl.Series(self._series.name, components, dtype=MoleculeType())
