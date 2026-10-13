"""Public hooks for packages that add methods to the ``chem`` namespaces."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from oepolars.namespaces.dataframe import DataFrameChemNamespace
from oepolars.namespaces.series import SeriesChemNamespace


def _register(cls: type, name: str, func: Callable[..., Any], overwrite: bool) -> None:
    if hasattr(cls, name) and not overwrite:
        raise ValueError(f"{cls.__name__} already defines {name!r}; pass overwrite=True to replace it")
    setattr(cls, name, func)


def register_series_method(name: str, func: Callable[..., Any], *, overwrite: bool = False) -> None:
    """Expose *func* as ``series.chem.<name>``.

    :param name: Method name.
    :param func: Function taking the namespace instance (``self._series`` is the Series).
    :param overwrite: Replace an existing attribute instead of raising.
    :raises ValueError: If *name* exists and *overwrite* is false.
    """
    _register(SeriesChemNamespace, name, func, overwrite)


def register_dataframe_method(name: str, func: Callable[..., Any], *, overwrite: bool = False) -> None:
    """Expose *func* as ``df.chem.<name>`` (``self._df`` is the DataFrame)."""
    _register(DataFrameChemNamespace, name, func, overwrite)
