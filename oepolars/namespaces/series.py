"""Series namespace for OpenEye operations."""

import copy
import logging
import math
from typing import Any, cast

import numpy as np
import polars as pl
from openeye import oechem, oedepict, oegraphsim

from oepolars.exceptions import InvalidSMARTS
from oepolars.fingerprint import (
    OEJaccard,
    fpdist,
    get_openeye_fp_info,
    make_fingerprints,
    to_numpy_fp,
)
from oepolars.types import DesignUnitType, DisplayType, FingerprintType, MoleculeType
from oepolars.types.molecule import (
    MoleculeTypeInput,
    _clear_mol_titles,
    _copy_molecule,
    _new_molecule,
)
from oepolars.types.query import (
    QueryFormat,
    QueryFormatInput,
    QueryType,
    _coerce_query,
    _copy_query,
)
from oepolars.util import get_oeformat

log = logging.getLogger("oepolars")

# Dictionary to store metadata for Series objects, keyed by object id.
# Note: This is keyed by id() which means metadata persists as long as
# the Series object exists (no automatic cleanup when Series is garbage collected).
_series_metadata: dict[int, dict[str, Any]] = {}


def _extension_name(dtype: pl.DataType) -> str | None:
    """Return a Polars extension dtype name when present."""
    ext_name = getattr(dtype, "ext_name", None)
    if callable(ext_name):
        return str(ext_name())
    return None


def _is_molecule_type(dtype: pl.DataType) -> bool:
    """Check if dtype is MoleculeType."""
    return _extension_name(dtype) == "molecule"


def _is_query_type(dtype: pl.DataType) -> bool:
    """Check if dtype is QueryType."""
    return _extension_name(dtype) == "query"


def _is_design_unit_type(dtype: pl.DataType) -> bool:
    """Check if dtype is DesignUnitType."""
    return _extension_name(dtype) == "design_unit"


def _is_display_type(dtype: pl.DataType) -> bool:
    """Check if dtype is DisplayType."""
    return _extension_name(dtype) == "display"


def _is_fingerprint_type(dtype: pl.DataType) -> bool:
    """Check if dtype is FingerprintType."""
    return _extension_name(dtype) == "fingerprint"


def _resolve_no_title(no_title: bool, clear_titles: bool | None) -> bool:
    """Resolve canonical no_title and deprecated clear_titles options."""
    if clear_titles is None:
        return no_title
    if no_title and not clear_titles:
        raise ValueError("clear_titles conflicts with no_title")
    return no_title or clear_titles


def _is_missing_value(value: object) -> bool:
    """Check whether a scalar should be treated as missing."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _parse_molecule_value(
    value: object,
    *,
    molecule_format: int,
    molecule_type: MoleculeTypeInput,
    no_title: bool,
) -> oechem.OEMolBase | None:
    """Parse or copy a scalar value into molecule storage."""
    if _is_missing_value(value):
        return None

    if isinstance(value, oechem.OEMolBase):
        if molecule_type is None and not no_title:
            return value
        return _copy_molecule(value, molecule_type, no_title=no_title)

    mol = _new_molecule(molecule_type)
    if isinstance(value, str):
        if molecule_format == oechem.OEFormat_SMI:
            if oechem.OEParseSmiles(mol, value):
                if no_title:
                    _clear_mol_titles(mol)
                return mol
            log.warning("Failed to parse SMILES: %s", value)
            return None

        if oechem.OEReadMolFromBytes(
            mol,
            molecule_format,
            False,
            value.encode("utf-8"),
        ):
            if no_title:
                _clear_mol_titles(mol)
            return mol
        log.warning("Failed to read molecule from string")
        return None

    if isinstance(value, bytes):
        if oechem.OEReadMolFromBytes(mol, molecule_format, False, value):
            if no_title:
                _clear_mol_titles(mol)
            return mol
        log.warning("Failed to read molecule from bytes")
        return None

    raise TypeError(f"Cannot create a molecule from {type(value).__name__}")


def _valid_openeye_object(value: object) -> bool:
    """Return whether an OpenEye object is present and valid."""
    if value is None:
        return False
    is_valid = getattr(value, "IsValid", None)
    if callable(is_valid):
        return bool(is_valid())
    return True


def _prepare_subsearch(
    pattern: str | oechem.OEQMol | oechem.OESubSearch,
) -> oechem.OESubSearch:
    """Create and validate an OpenEye substructure search object."""
    ss = (
        oechem.OESubSearch(pattern)
        if isinstance(pattern, str | oechem.OEQMol)
        else pattern
    )

    if not ss.IsValid():
        if isinstance(pattern, str):
            raise InvalidSMARTS(f"Invalid SMARTS pattern: {pattern}")
        raise InvalidSMARTS("Invalid substructure search pattern")

    return ss


@pl.api.register_series_namespace("chem")
class SeriesChemNamespace:
    """Namespace for OpenEye chemistry operations on Polars Series.

    Accessed via ``series.chem.*``.
    """

    def __init__(self, series: pl.Series) -> None:
        self._series = series

    @property
    def metadata(self) -> dict[str, Any]:
        """Get the metadata dictionary for this Series.

        Metadata is stored per-Series and can be used to attach arbitrary
        data (such as rendering callbacks, highlight patterns, etc.) to
        molecule Series.

        :returns: A dictionary for storing metadata. Created on first access.

        Example::

            s.chem.metadata["highlight"] = "c1ccccc1"
            s.chem.metadata["highlight"]  # Returns 'c1ccccc1'
        """
        series_id = id(self._series)
        if series_id not in _series_metadata:
            _series_metadata[series_id] = {}
        return _series_metadata[series_id]

    @property
    def token(self) -> str | None:
        """The per-column token from the dtype metadata, or ``None``."""
        return getattr(self._series.dtype, "token", None)

    def deepcopy(self) -> pl.Series:
        """Create a deep copy of the Series with independent object copies.

        This method copies both the objects (molecules, displays, or fingerprints)
        and the associated metadata, creating a fully independent copy of the Series.

        :returns: A new Series with deep copies of all objects and a copy
            of the metadata dictionary.
        :raises TypeError: If the Series is not a MoleculeType, DisplayType,
            or FingerprintType.

        Example::

            copied = s.chem.deepcopy()
            # Modifying original doesn't affect copy
            # Metadata is also preserved
            s.chem.metadata["key"] = "value"
            copied = s.chem.deepcopy()
            copied.chem.metadata["key"]  # Returns 'value'
        """
        dtype = self._series.dtype
        is_molecule = _is_molecule_type(dtype)
        is_display = _is_display_type(dtype)
        is_fingerprint = _is_fingerprint_type(dtype)

        if not (is_molecule or is_display or is_fingerprint):
            raise TypeError(
                f"deepcopy() requires MoleculeType, DisplayType, or FingerprintType, "
                f"got {dtype}"
            )

        if is_molecule:

            def copy_mol(
                mol: oechem.OEMolBase | None,
            ) -> oechem.OEMolBase | None:
                if mol is None:
                    return None
                return _copy_molecule(mol)

            copied = [copy_mol(mol) for mol in self._series.to_list()]
            new_series = pl.Series(self._series.name, copied, dtype=MoleculeType(dtype.token))

        elif is_display:

            def copy_disp(
                disp: oedepict.OE2DMolDisplay | None,
            ) -> oedepict.OE2DMolDisplay | None:
                if disp is None:
                    return None
                return oedepict.OE2DMolDisplay(disp)

            copied = [copy_disp(disp) for disp in self._series.to_list()]
            new_series = pl.Series(self._series.name, copied, dtype=DisplayType(dtype.token))

        else:  # is_fingerprint

            def copy_fp(
                fp: oegraphsim.OEFingerPrint | None,
            ) -> oegraphsim.OEFingerPrint | None:
                if fp is None:
                    return None
                return oegraphsim.OEFingerPrint(fp)

            copied = [copy_fp(fp) for fp in self._series.to_list()]
            new_series = pl.Series(self._series.name, copied, dtype=FingerprintType())

        # Copy metadata to the new series
        series_id = id(self._series)
        if series_id in _series_metadata:
            _series_metadata[id(new_series)] = copy.deepcopy(
                _series_metadata[series_id]
            )

        return new_series

    def as_molecule(
        self,
        molecule_format: str | int | None = None,
        molecule_type: MoleculeTypeInput = None,
        *,
        no_title: bool = False,
    ) -> pl.Series:
        """Convert this Series to molecule values.

        :param molecule_format: OpenEye format constant or name for parsing.
            Defaults to SMILES.
        :param molecule_type: Molecule implementation to produce.
        :param no_title: Clear molecule titles on returned values.
        :returns: MoleculeType Series.
        """
        if _is_molecule_type(self._series.dtype) and molecule_type is None and not no_title:
            return self._series

        fmt = get_oeformat(molecule_format or oechem.OEFormat_SMI, gzip=False).oeformat
        molecules = [
            _parse_molecule_value(
                value,
                molecule_format=fmt,
                molecule_type=molecule_type,
                no_title=no_title,
            )
            for value in self._series.to_list()
        ]
        dtype = self._series.dtype
        token = dtype.token if _is_molecule_type(dtype) else None
        return pl.Series(self._series.name, molecules, dtype=MoleculeType(token))

    def as_query(
        self,
        query_format: QueryFormatInput = QueryFormat.SMARTS,
        *,
        no_title: bool = False,
    ) -> pl.Series:
        """Convert this Series to query molecule values.

        :param query_format: Query string format for string values.
        :param no_title: Clear query titles on returned values.
        :returns: QueryType Series.
        """
        if _is_query_type(self._series.dtype) and not no_title:
            return self._series

        def convert(value: object) -> oechem.OEQMol | None:
            query = _coerce_query(value, query_format)
            if query is None:
                return None
            if no_title:
                return _copy_query(query, no_title=True)
            return query

        queries = [convert(value) for value in self._series.to_list()]
        dtype = self._series.dtype
        token = dtype.token if _is_query_type(dtype) else None  # pyright: ignore[reportAttributeAccessIssue]
        return pl.Series(self._series.name, queries, dtype=QueryType(token))

    def is_valid(self) -> pl.Series:
        """Return a boolean Series indicating valid OpenEye objects.

        :returns: Boolean Series where True indicates a valid object.
        :raises TypeError: If the Series is not a MoleculeType, QueryType, or
            DesignUnitType.

        Example::

            validity = s.chem.is_valid()
            # Series: 'mol' [bool]
            # [true, false, true]
        """
        dtype = self._series.dtype
        is_molecule = _is_molecule_type(dtype)
        is_query = _is_query_type(dtype)
        is_design_unit = _is_design_unit_type(dtype)

        if not (is_molecule or is_query or is_design_unit):
            raise TypeError(
                "is_valid() requires MoleculeType, QueryType, or DesignUnitType, "
                f"got {dtype}"
            )

        if is_design_unit:
            return self._series.is_not_null()

        return self._series.map_elements(
            _valid_openeye_object,
            return_dtype=pl.Boolean,
        ).fill_null(False)


    def to_smiles(self, flavor: int = oechem.OESMILESFlag_ISOMERIC) -> pl.Series:
        """Generate SMILES strings from molecules.

        :param flavor: SMILES generation options (bitmask from oechem.OESMILESFlag_*).
            Default is OESMILESFlag_ISOMERIC for isomeric SMILES.
        :returns: String Series containing SMILES representations.
        :raises TypeError: If the Series is not a MoleculeType.

        Example::

            smiles = s.chem.to_smiles()
            # Series: 'mol' [str]
            # ["CCO", "c1ccccc1", "CC(=O)O"]
        """
        if not _is_molecule_type(self._series.dtype):
            raise TypeError(
                f"to_smiles() requires MoleculeType, got {self._series.dtype}"
            )

        def mol_to_smiles(mol: oechem.OEMolBase | None) -> str | None:
            if mol is None:
                return None
            return cast(str, oechem.OECreateSmiString(mol, flavor))

        return self._series.map_elements(
            mol_to_smiles,
            return_dtype=pl.String,
        )

    def substructure_search(
        self,
        pattern: str | oechem.OEQMol | oechem.OESubSearch,
        *,
        adjustH: bool = False,
    ) -> pl.Series:
        """Perform substructure search on molecules.

        :param pattern: SMARTS pattern string, OEQMol, or OESubSearch object.
        :param adjustH: If True, adjust implicit/explicit hydrogens to match query.
        :returns: Boolean Series where True indicates a substructure match.
        :raises TypeError: If the Series is not a MoleculeType.
        :raises InvalidSMARTS: If the pattern is an invalid SMARTS string.

        Example::

            # Find molecules with benzene ring
            has_benzene = s.chem.substructure_search("c1ccccc1")
            # Series: 'mol' [bool]
            # [false, true, false, true]
        """
        if not _is_molecule_type(self._series.dtype):
            raise TypeError(
                f"substructure_search() requires MoleculeType, got {self._series.dtype}"
            )

        ss = _prepare_subsearch(pattern)

        def match_mol(mol: oechem.OEMolBase | None) -> bool:
            if mol is None or not mol.IsValid():
                return False
            try:
                oechem.OEPrepareSearch(mol, ss, adjustH)
                return bool(ss.SingleMatch(mol))
            except Exception:
                return False

        return self._series.map_elements(
            match_mol,
            return_dtype=pl.Boolean,
        ).fill_null(False)

    def substructure_filter(
        self,
        pattern: str | oechem.OEQMol | oechem.OESubSearch,
        *,
        adjustH: bool = False,
    ) -> pl.Series:
        """Return inverse of :meth:`substructure_search`.

        :param pattern: SMARTS pattern string, OEQMol, or OESubSearch object.
        :param adjustH: If True, adjust implicit/explicit hydrogens to match query.
        :returns: Boolean Series where True indicates the molecule does NOT match.
        :raises TypeError: If the Series is not a MoleculeType.
        :raises InvalidSMARTS: If the pattern is an invalid SMARTS string.

        Example::

            # Find molecules without a benzene ring
            no_benzene = s.chem.substructure_filter("c1ccccc1")
        """
        return ~self.substructure_search(pattern, adjustH=adjustH)

    def copy_molecules(self) -> pl.Series:
        """Create deep copies of all molecules in the Series.

        :returns: New Series with independent copies of each molecule.
        :raises TypeError: If the Series is not a MoleculeType.

        Example::

            copied = s.chem.copy_molecules()
            # Modifying original doesn't affect copy
        """
        if not _is_molecule_type(self._series.dtype):
            raise TypeError(
                f"copy_molecules() requires MoleculeType, got {self._series.dtype}"
            )

        def copy_mol(mol: oechem.OEMolBase | None) -> oechem.OEMolBase | None:
            if mol is None:
                return None
            return _copy_molecule(mol)

        copied = [copy_mol(mol) for mol in self._series.to_list()]
        dtype = self._series.dtype
        return pl.Series(self._series.name, copied, dtype=MoleculeType(dtype.token))

    def copy_design_units(self) -> pl.Series:
        """Create deep copies of all design units in the Series.

        :returns: New Series with independent copies of each design unit.
        :raises TypeError: If the Series is not a DesignUnitType.
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
        dtype = self._series.dtype
        return pl.Series(self._series.name, copied, dtype=DesignUnitType(dtype.token))

    def get_ligands(
        self,
        *,
        no_title: bool = False,
        clear_titles: bool | None = None,
    ) -> pl.Series:
        """Extract ligand molecules from design units.

        :param no_title: If True, clear titles on extracted ligands.
        :param clear_titles: Deprecated alias for ``no_title``.
        :returns: MoleculeType Series containing ligand molecules.
        :raises TypeError: If the Series is not a DesignUnitType.
        :raises ValueError: If ``no_title`` conflicts with ``clear_titles``.
        """
        resolved_no_title = _resolve_no_title(no_title, clear_titles)
        if not _is_design_unit_type(self._series.dtype):
            raise TypeError(
                f"get_ligands() requires DesignUnitType, got {self._series.dtype}"
            )

        def extract_ligand(du: oechem.OEDesignUnit | None) -> oechem.OEMol | None:
            if du is None:
                return None
            mol = oechem.OEMol()
            if du.GetLigand(mol):
                if resolved_no_title:
                    _clear_mol_titles(mol)
                return mol
            return None

        ligands = [extract_ligand(du) for du in self._series.to_list()]
        return pl.Series(self._series.name, ligands, dtype=MoleculeType())

    def get_proteins(
        self,
        *,
        no_title: bool = False,
        clear_titles: bool | None = None,
    ) -> pl.Series:
        """Extract protein molecules from design units.

        :param no_title: If True, clear titles on extracted proteins.
        :param clear_titles: Deprecated alias for ``no_title``.
        :returns: MoleculeType Series containing protein molecules.
        :raises TypeError: If the Series is not a DesignUnitType.
        :raises ValueError: If ``no_title`` conflicts with ``clear_titles``.
        """
        resolved_no_title = _resolve_no_title(no_title, clear_titles)
        if not _is_design_unit_type(self._series.dtype):
            raise TypeError(
                f"get_proteins() requires DesignUnitType, got {self._series.dtype}"
            )

        def extract_protein(du: oechem.OEDesignUnit | None) -> oechem.OEMol | None:
            if du is None:
                return None
            mol = oechem.OEMol()
            if du.GetProtein(mol):
                if resolved_no_title:
                    _clear_mol_titles(mol)
                return mol
            return None

        proteins = [extract_protein(du) for du in self._series.to_list()]
        return pl.Series(self._series.name, proteins, dtype=MoleculeType())

    def get_components(self, mask: int) -> pl.Series:
        """Extract components from design units by mask.

        :param mask: Component mask from oechem.OEDesignUnitComponents_*.
        :returns: MoleculeType Series containing extracted components.
        :raises TypeError: If the Series is not a DesignUnitType.
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

    # =========================================================================
    # Fingerprint operations
    # =========================================================================

    def create_fingerprints(
        self,
        fptype: str,
        num_bits: int,
        min_distance: int,
        max_distance: int,
        atom_type: str | int,
        bond_type: str | int,
        *,
        desalt: bool = True,
        quiet: bool = True,
    ) -> pl.Series:
        """Generate fingerprints from molecules in this series.

        :param fptype: OpenEye fingerprint type (path, tree, circular, lingo, maccs).
        :param num_bits: Number of bits in the fingerprint.
        :param min_distance: Minimum distance (path, tree) or radius (circular).
        :param max_distance: Maximum distance (path, tree) or radius (circular).
        :param atom_type: "|" separated OEFPAtomType members or bitmask.
        :param bond_type: "|" separated OEFPBondType members or bitmask.
        :param desalt: Remove salts by keeping only the largest component.
        :param quiet: Silence the progress bar.
        :returns: FingerprintType Series containing the generated fingerprints.
        :raises TypeError: If the Series is not a MoleculeType.

        Example::

            fps = s.chem.create_fingerprints(
                "circular", 2048, 0, 5, "DefaultAtom", "DefaultBond"
            )
        """
        if not _is_molecule_type(self._series.dtype):
            raise TypeError(
                f"create_fingerprints() requires MoleculeType, got {self._series.dtype}"
            )

        # Generate fingerprints
        fps = list(
            make_fingerprints(
                self._series.to_list(),
                fptype=fptype,
                num_bits=num_bits,
                min_distance=min_distance,
                max_distance=max_distance,
                atom_type=atom_type,
                bond_type=bond_type,
                desalt=desalt,
                quiet=quiet,
            )
        )

        # Create new series with metadata about fingerprint parameters
        new_series = pl.Series(self._series.name, fps, dtype=FingerprintType())

        # Store fingerprint parameters in metadata
        _series_metadata[id(new_series)] = {
            "fptype": fptype,
            "num_bits": num_bits,
            "min_distance": min_distance,
            "max_distance": max_distance,
            "atom_type": atom_type,
            "bond_type": bond_type,
            "desalt": desalt,
        }

        return new_series

    def create_numpy_fingerprints(
        self,
        fptype: str,
        num_bits: int,
        min_distance: int,
        max_distance: int,
        atom_type: str | int,
        bond_type: str | int,
        *,
        desalt: bool = True,
        quiet: bool = True,
    ) -> pl.Series:
        """Generate numpy fingerprints from molecules in this series.

        :param fptype: OpenEye fingerprint type (path, tree, circular, lingo, maccs).
        :param num_bits: Number of bits in the fingerprint.
        :param min_distance: Minimum distance (path, tree) or radius (circular).
        :param max_distance: Maximum distance (path, tree) or radius (circular).
        :param atom_type: "|" separated OEFPAtomType members or bitmask.
        :param bond_type: "|" separated OEFPBondType members or bitmask.
        :param desalt: Remove salts by keeping only the largest component.
        :param quiet: Silence the progress bar.
        :returns: Series of numpy boolean arrays (one per molecule).
        :raises TypeError: If the Series is not a MoleculeType.

        Example::

            fps = s.chem.create_numpy_fingerprints(
                "circular", 2048, 0, 5, "DefaultAtom", "DefaultBond"
            )
        """
        if not _is_molecule_type(self._series.dtype):
            raise TypeError(
                f"create_numpy_fingerprints() requires MoleculeType, "
                f"got {self._series.dtype}"
            )

        from oepolars.fingerprint import make_numpy_fingerprints

        fps = make_numpy_fingerprints(
            self._series.to_list(),
            fptype=fptype,
            num_bits=num_bits,
            min_distance=min_distance,
            max_distance=max_distance,
            atom_type=atom_type,
            bond_type=bond_type,
            desalt=desalt,
            quiet=quiet,
        )

        return pl.Series(self._series.name, fps, dtype=pl.Object)

    def copy_fingerprints(self) -> pl.Series:
        """Create deep copies of all fingerprints in the Series.

        :returns: New Series with independent copies of each fingerprint.
        :raises TypeError: If the Series is not a FingerprintType.

        Example::

            copied = fps.chem.copy_fingerprints()
            # Modifying original doesn't affect copy
        """
        if not _is_fingerprint_type(self._series.dtype):
            raise TypeError(
                f"copy_fingerprints() requires FingerprintType, got {self._series.dtype}"
            )

        def copy_fp(
            fp: oegraphsim.OEFingerPrint | None,
        ) -> oegraphsim.OEFingerPrint | None:
            if fp is None:
                return None
            return oegraphsim.OEFingerPrint(fp)

        copied = [copy_fp(fp) for fp in self._series.to_list()]
        new_series = pl.Series(self._series.name, copied, dtype=FingerprintType())

        # Copy metadata to the new series
        series_id = id(self._series)
        if series_id in _series_metadata:
            _series_metadata[id(new_series)] = copy.deepcopy(
                _series_metadata[series_id]
            )

        return new_series

    def fp_num_bits(self) -> int | None:
        """Get the number of bits in the fingerprints.

        :returns: Number of bits, or None if array is empty or has no valid
            fingerprints.
        :raises TypeError: If the Series is not a FingerprintType.

        Example::

            n_bits = fps.chem.fp_num_bits()
            # Returns 2048 (or whatever the fingerprint size is)
        """
        if not _is_fingerprint_type(self._series.dtype):
            raise TypeError(
                f"fp_num_bits() requires FingerprintType, got {self._series.dtype}"
            )

        for fp in self._series.to_list():
            if fp is not None and fp.IsValid():
                return cast(int, fp.GetSize())
        return None

    def to_hex_strings(self) -> pl.Series:
        """Convert fingerprints to hex strings.

        Invalid/missing fingerprints are represented as empty strings.

        :returns: String Series containing hex representations.
        :raises TypeError: If the Series is not a FingerprintType.

        Example::

            hex_strings = fps.chem.to_hex_strings()
            # Series: 'fp' [str]
            # ["A1B2C3...", "D4E5F6...", ...]
        """
        if not _is_fingerprint_type(self._series.dtype):
            raise TypeError(
                f"to_hex_strings() requires FingerprintType, got {self._series.dtype}"
            )

        hex_strings = []
        for fp in self._series.to_list():
            if fp is not None and fp.IsValid():
                hex_strings.append(fp.ToHexString())
            else:
                hex_strings.append("")

        return pl.Series(self._series.name, hex_strings, dtype=pl.String)

    def fp_to_numpy(self, quiet: bool = True) -> np.ndarray:
        """Convert fingerprints to a 2D numpy boolean array.

        Each row is a fingerprint, each column is a bit position.
        Invalid/missing fingerprints are represented as rows of False.

        :param quiet: Silence the progress bar.
        :returns: 2D numpy array of shape (n_fingerprints, n_bits).
        :raises TypeError: If the Series is not a FingerprintType.

        Example::

            arr = fps.chem.fp_to_numpy()
            # array([[True, False, True, ...],
            #        [False, True, False, ...],
            #        ...], dtype=bool)
        """
        if not _is_fingerprint_type(self._series.dtype):
            raise TypeError(
                f"fp_to_numpy() requires FingerprintType, got {self._series.dtype}"
            )

        fps_list = self._series.to_list()

        # Get fingerprint size from first valid fingerprint
        n_bits = self.fp_num_bits()
        if n_bits is None:
            return np.array([], dtype=bool).reshape(0, 0)

        # Convert each fingerprint
        result = []
        fp_info = None

        for fp in fps_list:
            if fp is None or not fp.IsValid():
                result.append(np.zeros(n_bits, dtype=bool))
            else:
                if fp_info is None:
                    fp_info = get_openeye_fp_info(fp)
                result.append(cast(np.ndarray, to_numpy_fp(fp, fp_info)))

        return np.vstack(result) if result else np.array([], dtype=bool).reshape(
            0, n_bits
        )

    def tanimoto(
        self,
        other: "pl.Series | oegraphsim.OEFingerPrint",
    ) -> pl.Series:
        """Calculate Tanimoto similarity against another fingerprint or Series.

        :param other: Single fingerprint or FingerprintType Series.
        :returns: Float Series of Tanimoto similarities.
        :raises TypeError: If the Series is not a FingerprintType.
        :raises ValueError: If comparing Series of different lengths.

        Example::

            # Compare all fingerprints to a single reference
            sims = fps.chem.tanimoto(reference_fp)

            # Element-wise comparison between two Series
            sims = fps1.chem.tanimoto(fps2)
        """
        if not _is_fingerprint_type(self._series.dtype):
            raise TypeError(
                f"tanimoto() requires FingerprintType, got {self._series.dtype}"
            )

        fps_list = self._series.to_list()

        if isinstance(other, oegraphsim.OEFingerPrint):
            # Compare all fingerprints to a single reference
            similarities = []
            for fp in fps_list:
                if fp is None or not fp.IsValid() or not other.IsValid():
                    similarities.append(np.nan)
                else:
                    similarities.append(oegraphsim.OETanimoto(fp, other))
            return pl.Series(self._series.name, similarities, dtype=pl.Float64)

        elif isinstance(other, pl.Series):
            if not _is_fingerprint_type(other.dtype):
                raise TypeError(
                    f"other must be FingerprintType Series, got {other.dtype}"
                )
            if len(self._series) != len(other):
                raise ValueError(
                    f"Cannot compare Series of different lengths: "
                    f"{len(self._series)} vs {len(other)}"
                )
            other_list = other.to_list()
            similarities = []
            for fp1, fp2 in zip(fps_list, other_list, strict=True):
                if fp1 is None or fp2 is None or not fp1.IsValid() or not fp2.IsValid():
                    similarities.append(np.nan)
                else:
                    similarities.append(oegraphsim.OETanimoto(fp1, fp2))
            return pl.Series(self._series.name, similarities, dtype=pl.Float64)

        else:
            raise TypeError(
                f"Cannot compare FingerprintType with {type(other).__name__}"
            )

    def jaccard(
        self,
        other: "pl.Series | oegraphsim.OEFingerPrint",
    ) -> pl.Series:
        """Calculate Jaccard distance against another fingerprint or Series.

        :param other: Single fingerprint or FingerprintType Series.
        :returns: Float Series of Jaccard distances.
        :raises TypeError: If the Series is not a FingerprintType.
        :raises ValueError: If comparing Series of different lengths.

        Example::

            # Compare all fingerprints to a single reference
            dists = fps.chem.jaccard(reference_fp)

            # Element-wise comparison between two Series
            dists = fps1.chem.jaccard(fps2)
        """
        if not _is_fingerprint_type(self._series.dtype):
            raise TypeError(
                f"jaccard() requires FingerprintType, got {self._series.dtype}"
            )

        fps_list = self._series.to_list()

        if isinstance(other, oegraphsim.OEFingerPrint):
            # Compare all fingerprints to a single reference
            distances = []
            for fp in fps_list:
                if fp is None or not fp.IsValid() or not other.IsValid():
                    distances.append(np.nan)
                else:
                    distances.append(OEJaccard(fp, other))
            return pl.Series(self._series.name, distances, dtype=pl.Float64)

        elif isinstance(other, pl.Series):
            if not _is_fingerprint_type(other.dtype):
                raise TypeError(
                    f"other must be FingerprintType Series, got {other.dtype}"
                )
            if len(self._series) != len(other):
                raise ValueError(
                    f"Cannot compare Series of different lengths: "
                    f"{len(self._series)} vs {len(other)}"
                )
            other_list = other.to_list()
            distances = []
            for fp1, fp2 in zip(fps_list, other_list, strict=True):
                if fp1 is None or fp2 is None or not fp1.IsValid() or not fp2.IsValid():
                    distances.append(np.nan)
                else:
                    distances.append(OEJaccard(fp1, fp2))
            return pl.Series(self._series.name, distances, dtype=pl.Float64)

        else:
            raise TypeError(
                f"Cannot compare FingerprintType with {type(other).__name__}"
            )

    def pdist(
        self,
        metric: str = "tanimoto",
    ) -> np.ndarray:
        """Calculate pairwise distances/similarities between all fingerprints.

        :param metric: Comparison metric name (tanimoto, dice, cosine, euclidean,
            manhattan, jaccard).
        :returns: Compressed pairwise distance matrix (scipy pdist format).
        :raises TypeError: If the Series is not a FingerprintType.

        Example::

            # Get pairwise Tanimoto similarities
            dist_matrix = fps.chem.pdist(metric="tanimoto")

            # Convert to square matrix with scipy
            from scipy.spatial.distance import squareform
            square = squareform(dist_matrix)
        """
        if not _is_fingerprint_type(self._series.dtype):
            raise TypeError(
                f"pdist() requires FingerprintType, got {self._series.dtype}"
            )

        # Filter to valid fingerprints for pdist calculation
        valid_fps = [
            fp
            for fp in self._series.to_list()
            if fp is not None and fp.IsValid()
        ]
        return fpdist(valid_fps, metric)
