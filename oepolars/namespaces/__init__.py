"""Polars namespace extensions for OpenEye operations."""

from oepolars.namespaces.dataframe import DataFrameOENamespace
from oepolars.namespaces.series import SeriesOENamespace

__all__ = ["SeriesOENamespace", "DataFrameOENamespace"]
