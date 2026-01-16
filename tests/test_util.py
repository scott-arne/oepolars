import pytest
from openeye import oechem

from oepolars.util import (
    get_oeformat,
    is_gz,
)


class TestIsGz:
    def test_gz_extension(self):
        assert is_gz("file.sdf.gz") is True

    def test_gzip_extension(self):
        assert is_gz("file.sdf.gzip") is True

    def test_no_gz_extension(self):
        assert is_gz("file.sdf") is False

    def test_path_object(self):
        from pathlib import Path
        assert is_gz(Path("file.oeb.gz")) is True


class TestGetOEFormat:
    def test_sdf_string(self):
        info = get_oeformat("sdf", gzip=False)
        assert info.oeformat == oechem.OEFormat_SDF
        assert info.gzip is False

    def test_oeb_string(self):
        info = get_oeformat("oeb", gzip=False)
        assert info.oeformat == oechem.OEFormat_OEB

    def test_smi_string(self):
        info = get_oeformat("smi", gzip=False)
        assert info.oeformat == oechem.OEFormat_SMI

    def test_smiles_string(self):
        info = get_oeformat("smiles", gzip=False)
        assert info.oeformat == oechem.OEFormat_SMI

    def test_integer_format(self):
        info = get_oeformat(oechem.OEFormat_SDF, gzip=False)
        assert info.oeformat == oechem.OEFormat_SDF

    def test_gzip_flag(self):
        info = get_oeformat("sdf", gzip=True)
        assert info.gzip is True

    def test_unknown_format_raises(self):
        from oepolars.exceptions import UnsupportedFileFormat
        with pytest.raises(UnsupportedFileFormat):
            get_oeformat("xyz123", gzip=False)
