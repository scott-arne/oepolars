"""Pandas interoperability for OEPolars."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from openeye import oechem, oedepict

from oepolars.types import DesignUnitType, DisplayType, MoleculeType

if TYPE_CHECKING:
    import pandas as pd


def from_pandas(df: pd.DataFrame) -> pl.DataFrame:
    """Convert a Pandas DataFrame to a Polars DataFrame, preserving OEPandas extension types.

    Automatically detects and converts OEPandas extension dtype columns:
    - ``MoleculeDtype`` -> ``MoleculeType``
    - ``DesignUnitDtype`` -> ``DesignUnitType``
    - ``DisplayDtype`` -> ``DisplayType``

    :param df: Pandas DataFrame, optionally containing OEPandas extension dtype columns.
    :returns: Polars DataFrame with OEPolars extension types for detected columns.

    Example::

        import pandas as pd
        import oepandas as oepd
        import oepolars as oepl

        # Create OEPandas DataFrame
        pd_df = oepd.read_sdf("molecules.sdf")

        # Convert to OEPolars DataFrame
        pl_df = oepl.from_pandas(pd_df)
    """
    # Detect OEPandas extension dtype columns
    molecule_cols: set[str] = set()
    design_unit_cols: set[str] = set()
    display_cols: set[str] = set()

    for col in df.columns:
        dtype_name = getattr(df[col].dtype, "name", None)
        if dtype_name == "molecule":
            molecule_cols.add(col)
        elif dtype_name == "design_unit":
            design_unit_cols.add(col)
        elif dtype_name == "display":
            display_cols.add(col)

    openeye_cols = molecule_cols | design_unit_cols | display_cols

    # Build Polars DataFrame column by column
    polars_series: list[pl.Series] = []

    for col in df.columns:
        if col in molecule_cols:
            # Convert OEPandas molecule column to OEPolars MoleculeType
            values = df[col].tolist()
            molecules = [
                oechem.OEMol(v) if isinstance(v, oechem.OEMolBase) else None
                for v in values
            ]
            polars_series.append(pl.Series(col, molecules, dtype=MoleculeType()))

        elif col in design_unit_cols:
            # Convert OEPandas design unit column to OEPolars DesignUnitType
            values = df[col].tolist()
            design_units = [
                oechem.OEDesignUnit(v) if isinstance(v, oechem.OEDesignUnit) else None
                for v in values
            ]
            polars_series.append(pl.Series(col, design_units, dtype=DesignUnitType()))

        elif col in display_cols:
            # Convert OEPandas display column to OEPolars DisplayType
            values = df[col].tolist()
            displays = [
                oedepict.OE2DMolDisplay(v)
                if isinstance(v, oedepict.OE2DMolDisplay)
                else None
                for v in values
            ]
            polars_series.append(pl.Series(col, displays, dtype=DisplayType()))

        else:
            # Regular column - convert via Polars
            polars_series.append(pl.Series(col, df[col].tolist()))

    return pl.DataFrame(polars_series)
