# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0]

### Added

- `QueryType` extension dtype for OpenEye query molecules, with a `QueryFormat` enum (`SMARTS`, `SMIRKS`) selecting how string values are parsed.
- `series.chem.as_query()` and `df.chem.as_query()` to convert columns to `QueryType`. `lf.chem.as_query()` raises `LazyOperationError` like the other eager-only operations.
- `MoleculeTypeKind` enum (`DEFAULT`, `OEMOL`, `OEGRAPHMOL`, `OEQMOL`) and a `molecule_type` parameter to choose the OpenEye molecule class produced by `series.chem.as_molecule()`, `df.chem.as_molecule()`, `read_sdf()`, `read_oeb()`, `read_smi()`, `read_oedb()`, `read_molecule_csv()`, and the matching `scan_*` functions.
- `series.chem.as_molecule()` to convert a Series to `MoleculeType`. Re-copying an existing molecule column keeps its token.
- `no_title` parameter on `read_molecule_csv()`, `get_ligands()`, and `get_proteins()`.
- `substructure_search()` and `substructure_filter()` accept `OEQMol` patterns in addition to SMARTS strings and `OESubSearch` objects.
- `from_pandas()` converts OEPandas `QueryDtype` columns to `QueryType`.
- `py.typed` marker so type checkers use the package's inline annotations.

### Deprecated

- `clear_titles` on `get_ligands()` and `get_proteins()`; use `no_title`. Passing `no_title=True` with `clear_titles=False` raises `ValueError`.

## [0.4.0]

### Added

- Extension dtypes (`MoleculeType`, `DisplayType`, `DesignUnitType`) now accept an optional `token` parameter that stores per-column metadata. The token survives polars frame operations (filter, select, join, concat with matching tokens, etc.). Copy methods (`deepcopy()`, `copy_molecules()`, `copy_design_units()`) preserve the source column's token.
- `df.chem.with_token(column, token)` method to set or clear the token on an extension column.
- `series.chem.token` property to read the token from an extension column's dtype.
- `oepolars.extensions` module with `register_series_method()` and `register_dataframe_method()` hooks for packages that extend the `chem` namespaces.

### Changed

- Minimum required polars version raised to `1.44.1` (from `1.37.1`) to ensure reliable token metadata round-tripping through frame operations.
- Frames with extension columns carrying different tokens cannot be concatenated with `pl.concat()` (polars raises `SchemaError` because the dtypes differ). Clear the tokens with `with_token(column, None)` before concatenating if the token distinction is not meaningful for the concat operation.
