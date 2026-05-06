from openeye import oechem

from oepolars.util import bytes_to_mol, mol_to_bytes


class TestMoleculeSerializationRoundTrip:
    def test_simple_molecule_roundtrip(self):
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "CCO")
        mol.SetTitle("ethanol")

        # Serialize
        data = mol_to_bytes(mol)
        assert isinstance(data, bytes)
        assert len(data) > 0

        # Deserialize
        mol2 = bytes_to_mol(data)
        assert mol2 is not None
        assert mol2.GetTitle() == "ethanol"

        # SMILES should match
        smi1 = oechem.OECreateSmiString(mol)
        smi2 = oechem.OECreateSmiString(mol2)
        assert smi1 == smi2

    def test_complex_molecule_roundtrip(self):
        mol = oechem.OEMol()
        oechem.OEParseSmiles(mol, "c1ccc(cc1)C(=O)O")  # benzoic acid
        mol.SetTitle("benzoic_acid")
        oechem.OESetSDData(mol, "MW", "122.12")

        data = mol_to_bytes(mol)
        mol2 = bytes_to_mol(data)

        assert mol2.GetTitle() == "benzoic_acid"
        assert oechem.OEGetSDData(mol2, "MW") == "122.12"

    def test_none_molecule_returns_none(self):
        data = mol_to_bytes(None)
        assert data is None

    def test_none_bytes_returns_none(self):
        mol = bytes_to_mol(None)
        assert mol is None

    def test_empty_bytes_returns_none(self):
        mol = bytes_to_mol(b"")
        assert mol is None

    def test_invalid_bytes_returns_none(self):
        mol = bytes_to_mol(b"invalid data")
        assert mol is None
