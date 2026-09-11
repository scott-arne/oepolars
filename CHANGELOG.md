# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0]

### Added

- Extension dtypes (`MoleculeType`, `DisplayType`, `DesignUnitType`) now accept an optional `token` parameter that stores per-column metadata. The token survives polars frame operations (filter, select, join, concat with matching tokens, etc.).
- `df.chem.with_token(column, token)` method to set or clear the token on an extension column.
- `series.chem.token` property to read the token from an extension column's dtype.
- `oepolars.extensions` module with `register_series_method()` and `register_dataframe_method()` hooks for packages that extend the `chem` namespaces.

### Changed

- Frames with extension columns carrying different tokens cannot be concatenated with `pl.concat()` (polars raises `SchemaError` because the dtypes differ). Clear the tokens with `with_token(column, None)` before concatenating if the token distinction is not meaningful for the concat operation.
