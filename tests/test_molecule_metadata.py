import polars as pl
from openeye import oechem

# Import to register namespace
import oepolars  # noqa: F401
from oepolars import MoleculeType


class TestMoleculeMetadata:
    """Tests for molecule series metadata support."""

    def test_metadata_exists_on_molecule_series(self):
        """Molecule series should have a metadata dictionary."""
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, "CCO")

        s = pl.Series("mol", [mol], dtype=MoleculeType())

        # Access via chem accessor
        metadata = s.chem.metadata
        assert isinstance(metadata, dict)

    def test_metadata_can_store_values(self):
        """Metadata should store and retrieve values."""
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, "CCO")

        s = pl.Series("mol", [mol], dtype=MoleculeType())

        s.chem.metadata["test_key"] = "test_value"
        assert s.chem.metadata["test_key"] == "test_value"

    def test_deepcopy_preserves_metadata(self):
        """Deep copy should preserve metadata."""
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, "CCO")

        s = pl.Series("mol", [mol], dtype=MoleculeType())

        s.chem.metadata["key"] = "value"
        copied = s.chem.deepcopy()

        assert copied.chem.metadata.get("key") == "value"

    def test_deepcopy_creates_independent_molecules(self):
        """Deep copy should create independent molecule objects."""
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, "CCO")
        mol.SetTitle("original")

        s = pl.Series("mol", [mol], dtype=MoleculeType())

        copied = s.chem.deepcopy()

        # Modify original
        s[0].SetTitle("modified")

        # Copy should be unchanged
        assert copied[0].GetTitle() == "original"