"""QueryType extension type for OpenEye query molecules."""

import logging
import math
from enum import StrEnum
from typing import TypeAlias

import polars as pl
from openeye import oechem

log = logging.getLogger("oepolars")


class QueryType(pl.BaseExtension):
    """Polars extension type for OpenEye query molecule objects.

    Stores OEQMol objects using Polars' Object dtype as underlying storage.

    :param token: Opaque per-column token stored in the dtype metadata; it survives
        polars frame operations and lets clients key column-level settings.

    Example::

        from openeye import oechem
        query = oechem.OEQMol()
        oechem.OEParseSmarts(query, "[#6]")
        s = pl.Series("query", [query], dtype=QueryType())
        s.dtype.ext_name()  # Returns 'query'
    """

    def __init__(self, token: str | None = None) -> None:
        super().__init__("query", pl.Object, token)

    @property
    def token(self) -> str | None:
        """The per-column token, or ``None``."""
        return self.ext_metadata()


class QueryFormat(StrEnum):
    """Query string format to use when constructing query molecules."""

    SMARTS = "smarts"
    SMIRKS = "smirks"


QueryFormatInput: TypeAlias = QueryFormat | str

_QUERY_FORMAT_BY_STRING: dict[str, QueryFormat] = {
    query_format.value: query_format for query_format in QueryFormat
}


def _normalize_query_format(query_format: QueryFormatInput) -> QueryFormat:
    """Normalize user query format input.

    :param query_format: Query format selector.
    :returns: Normalized query format.
    :raises ValueError: When query format is unsupported.
    """
    if isinstance(query_format, QueryFormat):
        return query_format

    if isinstance(query_format, str):
        key = query_format.casefold()
        if key in _QUERY_FORMAT_BY_STRING:
            return _QUERY_FORMAT_BY_STRING[key]

    raise ValueError(f"Unsupported query_format: {query_format!r}")


def _parse_query_string(
    value: str,
    query_format: QueryFormatInput = QueryFormat.SMARTS,
) -> oechem.OEQMol | None:
    """Parse a query string into an OpenEye query molecule.

    :param value: Query string to parse.
    :param query_format: Query string format.
    :returns: Parsed query molecule, or ``None`` when parsing fails.
    """
    normalized_query_format = _normalize_query_format(query_format)
    query = oechem.OEQMol()

    if normalized_query_format == QueryFormat.SMARTS:
        parsed = oechem.OEParseSmarts(query, value)
    else:
        parsed = oechem.OEParseSmirks(query, value)

    if not parsed:
        log.warning(
            "Could not convert query from '%s': %s",
            normalized_query_format.value,
            value,
        )
        return None

    return query


def _is_missing_query_value(value: object) -> bool:
    """Check whether a value should be treated as a missing query value.

    :param value: Value to check.
    :returns: ``True`` when value is ``None`` or a floating-point NaN.
    """
    return value is None or (isinstance(value, float) and math.isnan(value))


def _coerce_query(
    value: object,
    query_format: QueryFormatInput = QueryFormat.SMARTS,
) -> oechem.OEQMol | None:
    """Coerce a scalar value into query molecule storage.

    :param value: Value to coerce.
    :param query_format: Query format for string values.
    :returns: Query molecule or ``None``.
    :raises TypeError: When value cannot be represented as a query.
    """
    if isinstance(value, str):
        return _parse_query_string(value.strip(), query_format)

    if isinstance(value, oechem.OEQMol):
        return value

    if isinstance(value, oechem.OEMolBase):
        return oechem.OEQMol(value)

    if _is_missing_query_value(value):
        return None

    raise TypeError(f"Cannot create a query from {type(value).__name__}")


def _copy_query(query: oechem.OEQMol, *, no_title: bool = False) -> oechem.OEQMol:
    """Copy a query molecule.

    :param query: Query molecule to copy.
    :param no_title: Clear molecule titles on the returned copy.
    :returns: Copied query molecule.
    """
    copied = oechem.OEQMol(query)
    if no_title:
        copied.SetTitle("")
    return copied


# Register the extension type with Polars
pl.register_extension_type("query", QueryType)
