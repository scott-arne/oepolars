"""Fingerprint utility functions for OpenEye fingerprints.

This module provides functions for fingerprint conversion, generation,
and similarity calculations.
"""

import itertools
import math
import typing
from collections import namedtuple
from collections.abc import Iterable

import numpy as np
import scipy.special
from openeye import oechem, oegraphsim
from tqdm.auto import tqdm

# =============================================================================
# Fingerprint conversion
# =============================================================================

OEFingerprintInfo = namedtuple("OEFingerPrintInfo", ("hex_length", "size", "ngroups"))


def get_openeye_fp_info(fp: oegraphsim.OEFingerPrint | str) -> OEFingerprintInfo:
    """Get information on an OpenEye fingerprint for conversion.

    :param fp: OEFingerPrint or OEFingerPrint hex string.
    :returns: Named tuple with hex_length, size, and ngroups.
    :raises TypeError: If fp is not a string or OEFingerPrint.
    """
    if isinstance(fp, str):
        fp_hex = fp
    elif isinstance(fp, oegraphsim.OEFingerPrint):
        fp_hex = fp.ToHexString()
    else:
        raise TypeError(
            f"Fingerprint must be either hex string or OEFingerPrint not {type(fp).__name__}"
        )

    # Size of the fingerprint
    hex_length = len(fp_hex) - 1
    fp_size = 4 * hex_length

    # Number of bits to nibble off at the end
    if (nibble_char := fp_hex[-1]) == "G":
        fp_size -= 1
    elif nibble_char == "H":
        fp_size -= 2
    elif nibble_char == "I":
        fp_size -= 3

    # Padding at end that may result given that numpy uses uint8 but OpenEye is using uint4
    # We are grouping two uint4 FPs into a uint8 for decoding so "ngroups" is the number
    # of byte groups.
    return OEFingerprintInfo(
        hex_length=hex_length, size=fp_size, ngroups=math.ceil(hex_length / 2)
    )


def to_numpy_fp(
    fp: oegraphsim.OEFingerPrint | str | None,
    fp_info: OEFingerprintInfo | None = None,
) -> np.ndarray | None:
    """Convert an OpenEye fingerprint to a numpy boolean array.

    :param fp: OpenEye fingerprint or hex string.
    :param fp_info: Optional fingerprint information (for efficiency).
    :returns: Numpy boolean array or None if fp is None/invalid.
    :raises TypeError: If fp is not a string or OEFingerPrint.
    """
    # Missing or invalid fingerprints
    if fp is None:
        return None
    elif isinstance(fp, str):
        fp_hex = fp
    elif isinstance(fp, oegraphsim.OEFingerPrint):
        fp_hex = fp.ToHexString()
    else:
        raise TypeError(
            f"Fingerprint must be either hex string or OEFingerPrint not {type(fp).__name__}"
        )

    # Get the fingerprint info (if necessary)
    if fp_info is None:
        fp_info = get_openeye_fp_info(fp_hex)

    # OpenEye BitVector.ToHexString() starts from the least significant bit,
    # so we need to reverse the string and truncate out the nibble character
    fp_hex = fp_hex[-2::-1]

    packed = np.array(
        [
            int(
                f"{fp_hex[2 * i]}{fp_hex[j] if (j := 2 * i + 1) < fp_info.hex_length else 0}",
                16,
            )
            for i in range(fp_info.ngroups)
        ],
        dtype=np.uint8,
    )

    return np.array(np.unpackbits(packed, count=fp_info.size), dtype=bool)


def iterable_to_numpy_fps(
    fps: Iterable[oegraphsim.OEFingerPrint | str | None],
    quiet: bool = True,
) -> list[np.ndarray | None]:
    """Convert an iterable of OpenEye fingerprints to numpy bit arrays.

    :param fps: Iterable of OpenEye fingerprints.
    :param quiet: Disable the progress bar.
    :returns: List of numpy boolean arrays.
    """
    bitvecs = []
    # Assume every fingerprint is the same type, so we only need to calculate these
    # things once
    fp_info = None

    for fp in tqdm(fps, desc="Converting Fingerprints", unit="fp", disable=quiet):
        # If we need to calculate the fingerprint processing items
        # This is done once for efficiency - we assume every fingerprint is the same
        # type and length
        if fp_info is None and fp is not None:
            fp_info = get_openeye_fp_info(fp)

        # Convert and track
        bitvecs.append(to_numpy_fp(fp, fp_info))

    return bitvecs


# =============================================================================
# Fingerprint generation
# =============================================================================

# Dynamic creation of a typemap for OpenEye atom type fingerprints
atom_fp_typemap = dict(
    (x.replace("OEFPAtomType_", "").lower(), getattr(oegraphsim, x))
    for x in list(filter(lambda x: x.startswith("OEFPAtomType_"), dir(oegraphsim)))
)

# Dynamic creation of a typemap for OpenEye bond type fingerprints
bond_fp_typemap = dict(
    (x.replace("OEFPBondType_", "").lower(), getattr(oegraphsim, x))
    for x in list(filter(lambda x: x.startswith("OEFPBondType_"), dir(oegraphsim)))
)


def get_atom_mask(atom_type: str) -> int:
    """Get the OEFingerprint atom type masks from "|" delimited strings.

    The atom_type string is composed of "|" delimited members from the
    OEFPAtomType_ namespace. These are case-insensitive and only optionally
    need to be prefixed by "OEFPAtomType_".

    :param atom_type: Delimited string of OEFPAtomTypes.
    :returns: Bitmask for OpenEye fingerprint atom types.
    :raises KeyError: If an unknown atom type is specified.
    :raises ValueError: If no valid atom types are configured.
    """
    atom_mask = oegraphsim.OEFPAtomType_None
    for m in atom_type.split("|"):
        mask = atom_fp_typemap.get(m.strip().lower().replace("oefpatomtype_", ""))
        if mask is None:
            raise KeyError(f"{m} is not a known OEAtomFPType")
        atom_mask |= mask
    # Check validity
    if atom_mask == oegraphsim.OEFPAtomType_None:
        raise ValueError("No atom fingerprint types configured")
    return atom_mask


def get_bond_mask(bond_type: str) -> int:
    """Get the OEFingerprint bond type masks from "|" delimited strings.

    The bond_type string is composed of "|" delimited members from the
    OEFPBondType_ namespace. These are case-insensitive and only optionally
    need to be prefixed by "OEFPBondType_".

    :param bond_type: Delimited string of OEFPBondTypes.
    :returns: Bitmask for OpenEye fingerprint bond types.
    :raises KeyError: If an unknown bond type is specified.
    :raises ValueError: If no valid bond types are configured.
    """
    bond_mask = oegraphsim.OEFPBondType_None
    for m in bond_type.split("|"):
        mask = bond_fp_typemap.get(m.strip().lower().replace("oefpbondtype_", ""))
        if mask is None:
            raise KeyError(f"{m} is not a known OEBondFPType")
        bond_mask |= mask
    # Check validity
    if bond_mask == oegraphsim.OEFPBondType_None:
        raise ValueError("No bond fingerprint types configured")
    return bond_mask


def fingerprint_maker(
    fptype: str,
    num_bits: int,
    min_distance: int,
    max_distance: int,
    atom_type: str | int,
    bond_type: str | int,
) -> typing.Callable[[oechem.OEMolBase], oegraphsim.OEFingerPrint]:
    """Create a function that generates a fingerprint from a molecule.

    :param fptype: Fingerprint type (path, tree, circular, maccs, lingo).
    :param num_bits: Number of bits in the fingerprint.
    :param min_distance: Minimum distance/radius for path/circular/tree.
    :param max_distance: Maximum distance/radius for path/circular/tree.
    :param atom_type: Atom type string delimited by "|" or int bitmask.
    :param bond_type: Bond type string delimited by "|" or int bitmask.
    :returns: Function that generates a fingerprint from a molecule.
    :raises KeyError: If an unknown fingerprint type is specified.
    """
    # Be forgiving with case
    _fptype = fptype.lower()

    # Convert atom type and bond type strings to masks if necessary
    atom_mask = get_atom_mask(atom_type) if isinstance(atom_type, str) else atom_type
    bond_mask = get_bond_mask(bond_type) if isinstance(bond_type, str) else bond_type

    if _fptype == "path":

        def _make_path_fp(mol):
            fp = oegraphsim.OEFingerPrint()
            oegraphsim.OEMakePathFP(
                fp, mol, num_bits, min_distance, max_distance, atom_mask, bond_mask
            )
            return fp

        return _make_path_fp

    elif _fptype == "circular":

        def _make_circular_fp(mol):
            fp = oegraphsim.OEFingerPrint()
            oegraphsim.OEMakeCircularFP(
                fp, mol, num_bits, min_distance, max_distance, atom_mask, bond_mask
            )
            return fp

        return _make_circular_fp

    elif _fptype == "tree":

        def _make_tree_fp(mol):
            fp = oegraphsim.OEFingerPrint()
            oegraphsim.OEMakeTreeFP(
                fp, mol, num_bits, min_distance, max_distance, atom_mask, bond_mask
            )
            return fp

        return _make_tree_fp

    elif _fptype == "maccs":

        def _make_maccs(mol):
            fp = oegraphsim.OEFingerPrint()
            oegraphsim.OEMakeMACCS166FP(fp, mol)
            return fp

        return _make_maccs

    elif _fptype == "lingo":

        def _make_lingo(mol):
            fp = oegraphsim.OEFingerPrint()
            oegraphsim.OEMakeLingoFP(fp, mol)
            return fp

        return _make_lingo

    raise KeyError(
        f"Unknown fingerprint type {fptype} (valid: path / tree / circular / maccs / lingo)"
    )


def make_fingerprints(
    data: Iterable[oechem.OEMolBase],
    fptype: str,
    num_bits: int,
    min_distance: int,
    max_distance: int,
    atom_type: str | int,
    bond_type: str | int,
    desalt: bool = True,
    quiet: bool = True,
) -> Iterable[oegraphsim.OEFingerPrint | None]:
    """Generate fingerprints from an iterable of molecules.

    :param data: Iterable of OpenEye molecules.
    :param fptype: OpenEye fingerprint type (path, tree, circular, lingo, maccs).
    :param num_bits: Number of bits in the fingerprint.
    :param min_distance: Minimum distance (path, tree) or radius (circular).
    :param max_distance: Maximum distance (path, tree) or radius (circular).
    :param atom_type: List of "|" separated OEFPAtomType members or bitmask.
    :param bond_type: List of "|" separated OEFPBondType members or bitmask.
    :param desalt: Delete everything but the largest component in the molecule.
    :param quiet: Silence the progress bar.
    :returns: Iterator over OEFingerPrint objects, or None for invalid molecules.
    """
    # Fingerprint maker
    make_fp = fingerprint_maker(
        fptype, num_bits, min_distance, max_distance, atom_type, bond_type
    )

    for mol in tqdm(data, desc="Generate Fingerprints", unit="mol", disable=quiet):
        # Handle None molecules
        if mol is None:
            yield None
            continue

        # If getting rid of salts and other nonsense
        if desalt:
            oechem.OEDeleteEverythingExceptTheFirstLargestComponent(mol)

        # Sanity check
        if not isinstance(mol, oechem.OEMolBase):
            raise TypeError(
                f"Expected list of molecule objects but encountered a {type(mol).__name__}"
            )

        # Only generate fingerprints for valid molecules
        if mol.IsValid():
            fp = make_fp(mol)
            if fp.IsValid():
                yield fp
            else:
                yield None
        else:
            yield None


def make_numpy_fingerprints(
    data: Iterable[oechem.OEMolBase],
    fptype: str,
    num_bits: int,
    min_distance: int,
    max_distance: int,
    atom_type: str | int,
    bond_type: str | int,
    desalt: bool = True,
    quiet: bool = True,
) -> list[np.ndarray | None]:
    """Convenience function to make a list of Numpy fingerprints from molecules.

    :param data: Iterable of OpenEye molecules.
    :param fptype: OpenEye fingerprint type (path, tree, circular, lingo, maccs).
    :param num_bits: Number of bits in the fingerprint.
    :param min_distance: Minimum distance (path, tree) or radius (circular).
    :param max_distance: Maximum distance (path, tree) or radius (circular).
    :param atom_type: List of "|" separated OEFPAtomType members or bitmask.
    :param bond_type: List of "|" separated OEFPBondType members or bitmask.
    :param desalt: Delete everything but the largest component in the molecule.
    :param quiet: Silence the progress bar.
    :returns: List of Numpy fingerprints as boolean arrays.
    """
    return iterable_to_numpy_fps(
        list(
            make_fingerprints(
                data,
                fptype,
                num_bits,
                min_distance,
                max_distance,
                atom_type,
                bond_type,
                desalt=desalt,
                quiet=quiet,
            )
        ),
        quiet=quiet,
    )


# =============================================================================
# Similarity calculations
# =============================================================================


def OEJaccard(
    fpA: oegraphsim.OEFingerPrint,
    fpB: oegraphsim.OEFingerPrint,
) -> float:
    """Calculate Jaccard distance between two fingerprints.

    :param fpA: First fingerprint.
    :param fpB: Second fingerprint.
    :returns: Jaccard distance.
    """
    onlyA, onlyB, bothAB, _ = oechem.OEGetBitCounts(fpA, fpB)
    union = onlyA + onlyB + bothAB
    return float((union - bothAB) / union)


# Fingerprint similarity metric stored as (function, is_similarity_metric)
openeye_comparison_metric = {
    "tanimoto": (oegraphsim.OETanimoto, True),
    "euclidean": (oegraphsim.OEEuclid, True),
    "dice": (oegraphsim.OEDice, True),
    "cosine": (oegraphsim.OECosine, True),
    "manhattan": (oegraphsim.OEManhattan, False),
    "jaccard": (OEJaccard, False),
}


def get_openeye_comparison_metric(
    name: str,
) -> tuple[typing.Callable, bool]:
    """Get a comparison function by name.

    :param name: Name of the comparison function.
    :returns: Tuple of (comparison function, is_similarity_metric).
    :raises KeyError: If the comparison function is unknown.
    """
    cf = openeye_comparison_metric.get(name.lower())
    if cf is None:
        raise KeyError(f"Unknown comparison function: {name}")
    return cf


def fpdist(
    fps: list[oegraphsim.OEFingerPrint],
    metric: str | typing.Callable,
) -> np.ndarray:
    """Create a compressed pairwise distance matrix for a set of fingerprints.

    :param fps: List of fingerprints.
    :param metric: OpenEye comparison metric name or function.
    :returns: Compressed pairwise distance matrix (scipy pdist format).
    """
    if isinstance(metric, str):
        cfunc, _ = get_openeye_comparison_metric(metric)
    else:
        cfunc = metric

    compressed = np.zeros(shape=scipy.special.comb(len(fps), 2, exact=True))
    for idx, (fp1, fp2) in enumerate(itertools.combinations(fps, 2)):
        compressed[idx] = cfunc(fp1, fp2)
    return compressed
