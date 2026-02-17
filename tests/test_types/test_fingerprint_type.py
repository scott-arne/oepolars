import numpy as np
import polars as pl
import pytest
from openeye import oechem, oegraphsim

from oepolars.types import FingerprintType, MoleculeType


class TestFingerprintTypeCreation:
    def test_fingerprint_type_is_extension(self):
        fp_type = FingerprintType()
        assert isinstance(fp_type, pl.BaseExtension)

    def test_fingerprint_type_name(self):
        fp_type = FingerprintType()
        assert fp_type.ext_name() == "fingerprint"

    def test_fingerprint_type_storage(self):
        fp_type = FingerprintType()
        assert fp_type.ext_storage() == pl.Object


class TestFingerprintTypeSeries:
    @pytest.fixture
    def sample_fingerprints(self):
        fps = []
        for smi in ["CCO", "c1ccccc1", "CC(=O)O"]:
            mol = oechem.OEMol()
            oechem.OEParseSmiles(mol, smi)
            fp = oegraphsim.OEFingerPrint()
            oegraphsim.OEMakeCircularFP(fp, mol)
            fps.append(fp)
        return fps

    @pytest.fixture
    def sample_molecules(self):
        mols = []
        for smi in ["CCO", "c1ccccc1", "CC(=O)O"]:
            mol = oechem.OEMol()
            oechem.OEParseSmiles(mol, smi)
            mols.append(mol)
        return mols

    def test_create_series_from_fingerprints(self, sample_fingerprints):
        s = pl.Series("fp", sample_fingerprints, dtype=FingerprintType())
        assert len(s) == 3
        assert s.dtype.ext_name() == "fingerprint"

    def test_series_displays_as_ext_fingerprint(self, sample_fingerprints):
        s = pl.Series("fp", sample_fingerprints, dtype=FingerprintType())
        repr_str = repr(s)
        assert "ext[fingerprint]" in repr_str

    def test_series_with_none_values(self, sample_fingerprints):
        data = [sample_fingerprints[0], None, sample_fingerprints[2]]
        s = pl.Series("fp", data, dtype=FingerprintType())
        assert s.null_count() == 1

    def test_series_item_retrieval(self, sample_fingerprints):
        s = pl.Series("fp", sample_fingerprints, dtype=FingerprintType())
        fp = s[0]
        assert isinstance(fp, oegraphsim.OEFingerPrint)


class TestFingerprintTypeInDataFrame:
    @pytest.fixture
    def sample_fingerprints(self):
        fps = []
        for smi in ["CCO", "c1ccccc1"]:
            mol = oechem.OEMol()
            oechem.OEParseSmiles(mol, smi)
            fp = oegraphsim.OEFingerPrint()
            oegraphsim.OEMakeCircularFP(fp, mol)
            fps.append(fp)
        return fps

    def test_dataframe_with_fingerprint_column(self, sample_fingerprints):
        df = pl.DataFrame({
            "fp": pl.Series(sample_fingerprints, dtype=FingerprintType()),
            "name": ["ethanol", "benzene"],
        })
        assert df.schema["fp"].ext_name() == "fingerprint"
        assert df.schema["name"] == pl.String

    def test_dataframe_select_fingerprint_column(self, sample_fingerprints):
        df = pl.DataFrame({
            "fp": pl.Series(sample_fingerprints, dtype=FingerprintType()),
            "name": ["ethanol", "benzene"],
        })
        result = df.select("fp")
        assert result.schema["fp"].ext_name() == "fingerprint"


class TestFingerprintGeneration:
    @pytest.fixture
    def sample_molecules(self):
        mols = []
        for smi in ["CCO", "c1ccccc1", "CC(=O)O"]:
            mol = oechem.OEMol()
            oechem.OEParseSmiles(mol, smi)
            mols.append(mol)
        return mols

    def test_create_fingerprints_circular(self, sample_molecules):
        s = pl.Series("mol", sample_molecules, dtype=MoleculeType())
        fps = s.chem.create_fingerprints(
            "circular", 2048, 0, 5, "DefaultAtom", "DefaultBond"
        )
        assert len(fps) == 3
        assert fps.dtype.ext_name() == "fingerprint"

    def test_create_fingerprints_path(self, sample_molecules):
        s = pl.Series("mol", sample_molecules, dtype=MoleculeType())
        fps = s.chem.create_fingerprints(
            "path", 1024, 0, 5, "DefaultAtom", "DefaultBond"
        )
        assert len(fps) == 3
        assert fps.dtype.ext_name() == "fingerprint"

    def test_create_fingerprints_maccs(self, sample_molecules):
        s = pl.Series("mol", sample_molecules, dtype=MoleculeType())
        # MACCS doesn't use num_bits, min/max distance, atom/bond type but we still
        # need to pass them
        fps = s.chem.create_fingerprints(
            "maccs", 166, 0, 0, "DefaultAtom", "DefaultBond"
        )
        assert len(fps) == 3
        assert fps.dtype.ext_name() == "fingerprint"

    def test_create_fingerprints_metadata(self, sample_molecules):
        s = pl.Series("mol", sample_molecules, dtype=MoleculeType())
        fps = s.chem.create_fingerprints(
            "circular", 2048, 0, 5, "DefaultAtom", "DefaultBond"
        )
        assert fps.chem.metadata["fptype"] == "circular"
        assert fps.chem.metadata["num_bits"] == 2048

    def test_create_fingerprints_wrong_type(self):
        s = pl.Series("names", ["a", "b", "c"])
        with pytest.raises(TypeError):
            s.chem.create_fingerprints(
                "circular", 2048, 0, 5, "DefaultAtom", "DefaultBond"
            )

    def test_create_numpy_fingerprints(self, sample_molecules):
        s = pl.Series("mol", sample_molecules, dtype=MoleculeType())
        fps = s.chem.create_numpy_fingerprints(
            "circular", 2048, 0, 5, "DefaultAtom", "DefaultBond"
        )
        assert len(fps) == 3
        assert fps.dtype == pl.Object
        # Each element should be a numpy array
        for fp in fps.to_list():
            assert isinstance(fp, np.ndarray)
            assert fp.dtype == bool

    def test_create_numpy_fingerprints_wrong_type(self):
        s = pl.Series("names", ["a", "b", "c"])
        with pytest.raises(TypeError):
            s.chem.create_numpy_fingerprints(
                "circular", 2048, 0, 5, "DefaultAtom", "DefaultBond"
            )


class TestFingerprintOperations:
    @pytest.fixture
    def sample_fingerprints(self):
        fps = []
        for smi in ["CCO", "c1ccccc1", "CC(=O)O"]:
            mol = oechem.OEMol()
            oechem.OEParseSmiles(mol, smi)
            fp = oegraphsim.OEFingerPrint()
            oegraphsim.OEMakeCircularFP(fp, mol)
            fps.append(fp)
        return fps

    def test_fp_num_bits(self, sample_fingerprints):
        s = pl.Series("fp", sample_fingerprints, dtype=FingerprintType())
        n_bits = s.chem.fp_num_bits()
        assert isinstance(n_bits, int)
        assert n_bits > 0

    def test_to_hex_strings(self, sample_fingerprints):
        s = pl.Series("fp", sample_fingerprints, dtype=FingerprintType())
        hex_strs = s.chem.to_hex_strings()
        assert hex_strs.dtype == pl.String
        assert len(hex_strs) == 3
        # Hex strings should be non-empty
        for h in hex_strs.to_list():
            assert len(h) > 0

    def test_fp_to_numpy(self, sample_fingerprints):
        s = pl.Series("fp", sample_fingerprints, dtype=FingerprintType())
        arr = s.chem.fp_to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == bool
        assert arr.shape[0] == 3
        assert arr.shape[1] == s.chem.fp_num_bits()

    def test_copy_fingerprints(self, sample_fingerprints):
        s = pl.Series("fp", sample_fingerprints, dtype=FingerprintType())
        copied = s.chem.copy_fingerprints()
        assert len(copied) == len(s)
        assert copied.dtype.ext_name() == "fingerprint"
        # Verify they are different objects
        for orig, cp in zip(s.to_list(), copied.to_list()):
            assert orig is not cp


class TestFingerprintSimilarity:
    @pytest.fixture
    def sample_fingerprints(self):
        fps = []
        for smi in ["CCO", "CCCO", "CCCCO"]:  # Similar molecules
            mol = oechem.OEMol()
            oechem.OEParseSmiles(mol, smi)
            fp = oegraphsim.OEFingerPrint()
            oegraphsim.OEMakeCircularFP(fp, mol)
            fps.append(fp)
        return fps

    def test_tanimoto_single_reference(self, sample_fingerprints):
        s = pl.Series("fp", sample_fingerprints, dtype=FingerprintType())
        reference = sample_fingerprints[0]
        sims = s.chem.tanimoto(reference)
        assert sims.dtype == pl.Float64
        assert len(sims) == 3
        # First one should be 1.0 (same fingerprint)
        assert sims[0] == 1.0

    def test_tanimoto_series(self, sample_fingerprints):
        s1 = pl.Series("fp1", sample_fingerprints, dtype=FingerprintType())
        s2 = pl.Series("fp2", sample_fingerprints, dtype=FingerprintType())
        sims = s1.chem.tanimoto(s2)
        assert sims.dtype == pl.Float64
        assert len(sims) == 3
        # Comparing same fingerprints should all be 1.0
        for sim in sims.to_list():
            assert sim == 1.0

    def test_tanimoto_different_lengths(self, sample_fingerprints):
        s1 = pl.Series("fp1", sample_fingerprints, dtype=FingerprintType())
        s2 = pl.Series("fp2", sample_fingerprints[:2], dtype=FingerprintType())
        with pytest.raises(ValueError):
            s1.chem.tanimoto(s2)

    def test_jaccard_single_reference(self, sample_fingerprints):
        s = pl.Series("fp", sample_fingerprints, dtype=FingerprintType())
        reference = sample_fingerprints[0]
        dists = s.chem.jaccard(reference)
        assert dists.dtype == pl.Float64
        assert len(dists) == 3
        # First one should be 0.0 (same fingerprint, zero distance)
        assert dists[0] == 0.0

    def test_jaccard_series(self, sample_fingerprints):
        s1 = pl.Series("fp1", sample_fingerprints, dtype=FingerprintType())
        s2 = pl.Series("fp2", sample_fingerprints, dtype=FingerprintType())
        dists = s1.chem.jaccard(s2)
        assert dists.dtype == pl.Float64
        assert len(dists) == 3
        # Comparing same fingerprints should all be 0.0 (zero distance)
        for dist in dists.to_list():
            assert dist == 0.0

    def test_jaccard_different_lengths(self, sample_fingerprints):
        s1 = pl.Series("fp1", sample_fingerprints, dtype=FingerprintType())
        s2 = pl.Series("fp2", sample_fingerprints[:2], dtype=FingerprintType())
        with pytest.raises(ValueError):
            s1.chem.jaccard(s2)

    def test_pdist(self, sample_fingerprints):
        s = pl.Series("fp", sample_fingerprints, dtype=FingerprintType())
        dist = s.chem.pdist(metric="tanimoto")
        assert isinstance(dist, np.ndarray)
        # For 3 fingerprints, pdist returns 3 values (3 choose 2)
        assert len(dist) == 3


class TestFingerprintWithNulls:
    @pytest.fixture
    def fingerprints_with_nulls(self):
        fps = []
        for smi in ["CCO", None, "CC(=O)O"]:
            if smi is None:
                fps.append(None)
            else:
                mol = oechem.OEMol()
                oechem.OEParseSmiles(mol, smi)
                fp = oegraphsim.OEFingerPrint()
                oegraphsim.OEMakeCircularFP(fp, mol)
                fps.append(fp)
        return fps

    def test_tanimoto_with_nulls(self, fingerprints_with_nulls):
        s = pl.Series("fp", fingerprints_with_nulls, dtype=FingerprintType())
        reference = fingerprints_with_nulls[0]
        sims = s.chem.tanimoto(reference)
        assert sims[0] == 1.0
        assert np.isnan(sims[1])
        assert not np.isnan(sims[2])

    def test_fp_to_numpy_with_nulls(self, fingerprints_with_nulls):
        s = pl.Series("fp", fingerprints_with_nulls, dtype=FingerprintType())
        arr = s.chem.fp_to_numpy()
        # Should have 3 rows
        assert arr.shape[0] == 3
        # Row 1 (the null) should be all zeros
        assert arr[1].sum() == 0

    def test_to_hex_strings_with_nulls(self, fingerprints_with_nulls):
        s = pl.Series("fp", fingerprints_with_nulls, dtype=FingerprintType())
        hex_strs = s.chem.to_hex_strings()
        assert len(hex_strs[0]) > 0  # First should have content
        assert hex_strs[1] == ""  # Null should be empty string
        assert len(hex_strs[2]) > 0  # Third should have content
