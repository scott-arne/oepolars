import logging

import polars as pl
import pytest
from openeye import oechem

from oepolars.types import QueryFormat, QueryType
from oepolars.types.query import (
    _coerce_query,
    _copy_query,
    _is_missing_query_value,
    _normalize_query_format,
    _parse_query_string,
)


class TestQueryTypeCreation:
    def test_query_type_is_extension(self):
        query_type = QueryType()
        assert isinstance(query_type, pl.BaseExtension)

    def test_query_type_name(self):
        query_type = QueryType()
        assert query_type.ext_name() == "query"

    def test_query_type_storage(self):
        query_type = QueryType()
        assert query_type.ext_storage() == pl.Object


class TestQueryFormat:
    def test_normalizes_enum(self):
        assert _normalize_query_format(QueryFormat.SMIRKS) == QueryFormat.SMIRKS

    def test_normalizes_string(self):
        assert _normalize_query_format("SMARTS") == QueryFormat.SMARTS

    def test_rejects_invalid_query_format(self):
        with pytest.raises(ValueError, match="Unsupported query_format"):
            _normalize_query_format("smi")


class TestQueryHelpers:
    def test_parse_smarts_string(self):
        query = _parse_query_string("[#6]-[#8]", query_format=QueryFormat.SMARTS)
        assert isinstance(query, oechem.OEQMol)
        assert query.IsValid()
        assert query.NumAtoms() == 2

    def test_parse_smirks_string(self):
        query = _parse_query_string("[#6:1]-[#8:2]>>[#6:1]=[#8:2]", query_format="smirks")
        assert isinstance(query, oechem.OEQMol)
        assert query.IsValid()
        assert query.NumAtoms() == 4

    def test_invalid_query_string_returns_none_and_logs_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger="oepolars")
        query = _parse_query_string("not smarts", query_format="smarts")
        assert query is None
        assert "Could not convert query from 'smarts'" in caplog.text

    def test_missing_query_values_include_none_and_nan(self):
        assert _is_missing_query_value(None)
        assert _is_missing_query_value(float("nan"))

    def test_coerce_oemol_to_oeqmol(self):
        mol = oechem.OEMol()
        assert oechem.OEParseSmiles(mol, "CCO")
        query = _coerce_query(mol)
        assert isinstance(query, oechem.OEQMol)
        assert oechem.OEMolToSmiles(query) == "CCO"

    def test_copy_query_can_clear_titles_without_mutating_source(self):
        query = _parse_query_string("[#6]", query_format="smarts")
        assert query is not None
        query.SetTitle("source")

        copied = _copy_query(query, no_title=True)

        assert isinstance(copied, oechem.OEQMol)
        assert copied is not query
        assert copied.GetTitle() == ""
        assert query.GetTitle() == "source"


class TestQueryTypeSeries:
    def test_create_series_from_queries(self):
        query = _parse_query_string("[#6]", query_format="smarts")
        s = pl.Series("query", [query], dtype=QueryType())
        assert len(s) == 1
        assert s.dtype.ext_name() == "query"

    def test_series_with_none_values(self):
        query = _parse_query_string("[#6]", query_format="smarts")
        s = pl.Series("query", [query, None], dtype=QueryType())
        assert s.null_count() == 1
