"""Polars namespace extensions for OpenEye chemistry operations."""

from oepolars.namespaces.dataframe import DataFrameChemNamespace
from oepolars.namespaces.lazyframe import LazyFrameChemNamespace
from oepolars.namespaces.series import SeriesChemNamespace

__all__ = ["SeriesChemNamespace", "DataFrameChemNamespace", "LazyFrameChemNamespace"]
