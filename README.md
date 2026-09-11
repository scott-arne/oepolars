# OEPolars

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![OpenEye Toolkits](https://img.shields.io/badge/OpenEye-2023.1.0+-green.svg)](https://www.eyesopen.com/toolkits)
[![Polars 1.44.1+](https://img.shields.io/badge/polars-1.44.1+-orange.svg)](https://pola.rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Deep integration of OpenEye objects into Polars DataFrames with native support for molecules and design units.

OEPolars extends Polars with custom extension types that store OpenEye `OEMol` and `OEDesignUnit` objects as first-class DataFrame column types. This enables seamless interoperability between OpenEye's cheminformatics capabilities and Polars' high-performance data analysis workflows, including lazy evaluation for large-scale datasets.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Basic Usage](#basic-usage)
  - [Reading Molecular Data](#reading-molecular-data)
  - [Lazy Reading with Scanners](#lazy-reading-with-scanners)
  - [Working with Molecules](#working-with-molecules)
  - [Design Units](#design-units)
  - [Data Quality and Filtering](#data-quality-and-filtering)
- [Writing Data](#writing-data)
- [Parquet Serialization](#parquet-serialization)
- [API Reference](#api-reference)
  - [File Readers](#file-readers)
  - [File Scanners (Lazy)](#file-scanners-lazy)
  - [DataFrame Accessor Methods](#dataframe-accessor-methods-dfchem)
  - [LazyFrame Accessor Methods](#lazyframe-accessor-methods-lfchem)
  - [Series Accessor Methods](#series-accessor-methods-serieschem)
  - [Extension Types](#extension-types)
- [Examples](#examples)
- [Development](#development)
- [License](#license)

---

## Installation

### Requirements

| Package | Version |
|---------|---------|
| Python | 3.11+ |
| polars | 1.44.1+ |
| OpenEye Toolkits | 2023.1.0+ |

### OpenEye Toolkits License

OpenEye Toolkits requires a commercial license. However, **free licenses are available for academic and non-profit institutions**. Visit [OpenEye Scientific](https://www.eyesopen.com/academic-licensing) to request an academic license.

### Install from PyPI

```bash
pip install oepolars
```

### Development Installation

```bash
git clone https://github.com/scott-arne/oepolars.git
cd oepolars
pip install -e ".[dev]"
```

---

## Quick Start

```python
import oepolars as oepl
from openeye import oechem

# Load molecule data from various formats
df = oepl.read_sdf("molecules.sdf")
df = oepl.read_oeb("molecules.oeb.gz")
df = oepl.read_molecule_csv("data.csv", smiles_column="SMILES")

# Use Polars normally with molecules
df = df.with_columns(
    num_oxygens=df["Molecule"].map_elements(
        lambda mol: oechem.OECount(mol, oechem.OEIsOxygen()),
        return_dtype=pl.Int64
    )
)

# Generate SMILES strings
smiles = df["Molecule"].chem.to_smiles()

# Filter invalid molecules
df_valid = df.chem.filter_valid("Molecule")

# Write to file
df.chem.write_sdf("output.sdf", molecule_column="Molecule")
```

---

## Basic Usage

### Reading Molecular Data

OEPolars provides readers for all major chemical file formats supported by the OpenEye Toolkits:

```python
import oepolars as oepl

# SDF files - molecules with SD data as columns
df = oepl.read_sdf("molecules.sdf")

# OEB files (binary format, supports conformers)
df = oepl.read_oeb("molecules.oeb.gz")

# SMILES files
df = oepl.read_smi("molecules.smi")

# CSV files with SMILES column
df = oepl.read_molecule_csv("data.csv", smiles_column="SMILES")

# OERecord databases
df = oepl.read_oedb("records.oedb")

# Design unit files (protein-ligand complexes)
df = oepl.read_oedu("complexes.oedu")

# Parquet files with serialized molecules
df = oepl.read_parquet("molecules.parquet", molecule_columns=["Molecule"])
```

### Lazy Reading with Scanners

OEPolars provides lazy scanners for query optimization on large datasets:

```python
import oepolars as oepl

# Lazy reading - operations are optimized before execution
lf = oepl.scan_sdf("large_dataset.sdf")
lf = oepl.scan_oeb("large_dataset.oeb.gz")
lf = oepl.scan_smi("large_dataset.smi")
lf = oepl.scan_molecule_csv("large_dataset.csv", smiles_column="SMILES")
lf = oepl.scan_oedb("records.oedb")
lf = oepl.scan_oedu("complexes.oedu")
lf = oepl.scan_parquet("molecules.parquet", molecule_columns=["Molecule"])

# Apply filters before collecting
result = (
    lf
    .filter(pl.col("MolWt") > 300)
    .select(["Molecule", "Title", "MolWt"])
    .collect()
)
```

### Working with Molecules

Once loaded, molecules are stored as `MoleculeType` columns. Standard Polars operations work seamlessly:

```python
import polars as pl
from openeye import oechem

# Standard Polars operations
filtered_df = df.filter(pl.col("MolWt") > 200)
sorted_df = df.sort("Title")

# Apply OpenEye functions directly
df = df.with_columns(
    MW=pl.col("Molecule").map_elements(oechem.OECalculateMolecularWeight, return_dtype=pl.Float64),
    LogP=pl.col("Molecule").map_elements(oechem.OEGetXLogP, return_dtype=pl.Float64),
    HBD=pl.col("Molecule").map_elements(
        lambda m: oechem.OECount(m, oechem.OEIsHBondDonor()),
        return_dtype=pl.Int64
    ),
)

# Use the .chem accessor for molecular operations
smiles_series = df["Molecule"].chem.to_smiles()
copies = df["Molecule"].chem.copy_molecules()

# Substructure searching with SMARTS
has_carboxylic_acid = df["Molecule"].chem.substructure_search("C(=O)O")
df_acids = df.filter(has_carboxylic_acid)
```

### Design Units

Work with protein-ligand complexes stored as `DesignUnitType`:

```python
# Read design unit file
df = oepl.read_oedu("protein_ligand_complexes.oedu")

# Extract components using .chem accessor
df = df.with_columns(
    Ligand=df["Design_Unit"].chem.get_ligands(),
    Protein=df["Design_Unit"].chem.get_proteins(),
)

# Analyze components
df = df.with_columns(
    ligand_mw=pl.col("Ligand").map_elements(oechem.OECalculateMolecularWeight, return_dtype=pl.Float64)
)

# Deep copy design units
df = df.with_columns(
    DU_copy=df["Design_Unit"].chem.copy_design_units()
)
```

### Data Quality and Filtering

OEPolars provides methods to check and filter molecule validity:

```python
# Check which molecules are valid
validity = df["Molecule"].chem.is_valid()
print(f"Valid molecules: {validity.sum()}")
print(f"Invalid molecules: {(~validity).sum()}")

# Filter to keep only valid molecules
df_valid = df.chem.filter_valid("Molecule")

# Filter multiple columns at once
df_valid = df.chem.filter_valid(["Molecule", "Product"])

# Add validity as a column for inspection
df = df.with_columns(
    is_valid=df["Molecule"].chem.is_valid()
)
```

---

## Writing Data

Export DataFrames to various molecular file formats using the `.chem` accessor:

```python
# Write to SDF (columns become SD tags)
df.chem.write_sdf(
    "output.sdf",
    molecule_column="Molecule",
    title_column="Name",
    sd_columns=["Activity", "MW"]  # Include as SD tags
)

# Write to SMILES file
df.chem.write_smi(
    "output.smi",
    molecule_column="Molecule",
    title_column="Name"
)

# Write to OEB format
df.chem.write_oeb(
    "output.oeb",
    molecule_column="Molecule",
    title_column="Name"
)

# Write to CSV (molecules as SMILES strings)
df.chem.write_molecule_csv(
    "output.csv",
    molecule_column="Molecule",
    smiles_column="SMILES"
)

# Write to OERecord database
df.chem.write_oedb(
    "output.oedb",
    molecule_column="Molecule"
)

# Write design units
df.chem.write_oedu(
    "output.oedu",
    design_unit_column="Design_Unit"
)
```

---

## Parquet Serialization

OEPolars supports Parquet format with automatic molecule serialization, enabling efficient storage and retrieval of molecular data:

```python
# Write to Parquet (molecules serialized to binary OEB format)
df.chem.write_parquet(
    "molecules.parquet",
    molecule_columns=["Molecule"]  # Optional: auto-detects MoleculeType columns
)

# Read from Parquet (reconstruct molecules from binary)
df = oepl.read_parquet(
    "molecules.parquet",
    molecule_columns=["Molecule"]  # Required: specify which columns contain molecules
)

# Lazy reading from Parquet
lf = oepl.scan_parquet(
    "molecules.parquet",
    molecule_columns=["Molecule"]
)
```

---

## API Reference

### File Readers

#### `read_sdf()`

Read SD (Structure Data) files into a DataFrame.

```python
oepl.read_sdf(
    filepath,
    *,
    flavor=oechem.OEIFlavor_SDF_Default,
    molecule_column="Molecule",
    title_column="Title",
    sd_data=True,
    usecols=None,
    numeric_columns=None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | str, Path | required | Path to SDF file |
| `flavor` | int | `OEIFlavor_SDF_Default` | OpenEye SDF reader flavor |
| `molecule_column` | str | `"Molecule"` | Name of molecule column |
| `title_column` | str, None | `"Title"` | Name of title column (None to skip) |
| `sd_data` | bool | `True` | Read SD data into columns |
| `usecols` | str, list | `None` | SD tags to read (None for all) |
| `numeric_columns` | str, list | `None` | Columns to convert to numeric |

#### `read_oeb()`

Read OpenEye Binary (OEB) files into a DataFrame.

```python
oepl.read_oeb(
    filepath,
    *,
    flavor=oechem.OEIFlavor_SDF_Default,
    molecule_column="Molecule",
    title_column="Title",
    sd_data=True,
    usecols=None,
    numeric_columns=None
)
```

*Parameters same as `read_sdf()`*

#### `read_smi()`

Read SMILES files into a DataFrame.

```python
oepl.read_smi(
    filepath,
    *,
    molecule_column="Molecule",
    title_column="Title"
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | str, Path | required | Path to SMILES file |
| `molecule_column` | str | `"Molecule"` | Name of molecule column |
| `title_column` | str | `"Title"` | Name of title column |

#### `read_molecule_csv()`

Read CSV files with molecule columns.

```python
oepl.read_molecule_csv(
    filepath,
    smiles_column,
    *,
    molecule_column="Molecule",
    drop_smiles=False,
    **csv_kwargs
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | str, Path | required | Path to CSV file |
| `smiles_column` | str | required | Column containing SMILES strings |
| `molecule_column` | str | `"Molecule"` | Name of new molecule column |
| `drop_smiles` | bool | `False` | Drop original SMILES column |
| `**csv_kwargs` | | | Additional arguments passed to `pl.read_csv()` |

#### `read_oedb()`

Read OpenEye Database (OERecord) files into a DataFrame.

```python
oepl.read_oedb(
    filepath,
    *,
    molecule_column="Molecule",
    title_column="Title",
    sd_data=True,
    usecols=None,
    numeric_columns=None
)
```

*Parameters same as `read_sdf()`*

#### `read_oedu()`

Read Design Unit files into a DataFrame.

```python
oepl.read_oedu(
    filepath,
    *,
    design_unit_column="Design_Unit",
    title_column="Title"
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | str, Path | required | Path to OEDU file |
| `design_unit_column` | str | `"Design_Unit"` | Name of design unit column |
| `title_column` | str | `"Title"` | Name of title column |

#### `read_parquet()`

Read Parquet files with molecule column reconstruction.

```python
oepl.read_parquet(
    filepath,
    molecule_columns,
    **parquet_kwargs
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | str, Path | required | Path to Parquet file |
| `molecule_columns` | str, list | required | Column(s) containing serialized molecules |
| `**parquet_kwargs` | | | Additional arguments passed to `pl.read_parquet()` |

---

### File Scanners (Lazy)

All scanners return `pl.LazyFrame` for query optimization. Parameters match their eager counterparts.

| Scanner | Description |
|---------|-------------|
| `scan_sdf()` | Lazy reading of SDF files |
| `scan_oeb()` | Lazy reading of OEB files |
| `scan_smi()` | Lazy reading of SMILES files |
| `scan_molecule_csv()` | Lazy reading of CSV with SMILES |
| `scan_oedb()` | Lazy reading of OEDB files |
| `scan_oedu()` | Lazy reading of OEDU files |
| `scan_parquet()` | Lazy reading of Parquet files |

---

### DataFrame Accessor Methods (`df.chem.*`)

Access these methods via `df.chem.<method>()`:

#### `as_molecule()`

Convert column(s) to MoleculeType.

```python
df.chem.as_molecule(
    columns,
    *,
    molecule_format=None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `columns` | str, list | required | Column name(s) to convert |
| `molecule_format` | str, int | `None` | Format for parsing (default: SMILES) |

#### `filter_valid()`

Filter rows to keep only those with valid molecules.

```python
df.chem.filter_valid(columns)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `columns` | str, list | required | MoleculeType column(s) to check |

#### `detect_molecule_columns()`

Auto-detect and convert molecule columns based on content.

```python
df.chem.detect_molecule_columns(*, sample_size=25)
```

#### `with_token()`

Set or clear the token in an extension column's dtype metadata.

```python
df.chem.with_token(column, token)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `column` | str | required | Name of a MoleculeType, DisplayType, or DesignUnitType column |
| `token` | str, None | required | Token to set, or `None` to clear |

The token is stored in the dtype metadata and survives polars frame operations. Extension columns cannot be cast, so the column is rebuilt from its values with the new token; the Python objects are shared, not copied.

**Note:** Frames with extension columns carrying different tokens cannot be concatenated with `pl.concat()` (polars raises `SchemaError` because the dtypes differ).

#### `write_sdf()`

Write DataFrame to SDF file.

```python
df.chem.write_sdf(
    filepath,
    molecule_column,
    *,
    title_column=None,
    sd_columns=None,
    flavor=None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | str, Path | required | Output file path |
| `molecule_column` | str | required | Column with molecules |
| `title_column` | str | `None` | Column for titles |
| `sd_columns` | str, list | `None` | Columns to include as SD tags |
| `flavor` | int | `None` | OpenEye output flavor |

#### `write_smi()`

Write DataFrame to SMILES file.

```python
df.chem.write_smi(
    filepath,
    molecule_column,
    *,
    title_column=None,
    flavor=None
)
```

#### `write_oeb()`

Write DataFrame to OEB file.

```python
df.chem.write_oeb(
    filepath,
    molecule_column,
    *,
    title_column=None,
    sd_columns=None
)
```

#### `write_molecule_csv()`

Write DataFrame to CSV with molecules as SMILES strings.

```python
df.chem.write_molecule_csv(
    filepath,
    molecule_column,
    *,
    smiles_column="smiles",
    smiles_flavor=oechem.OESMILESFlag_ISOMERIC,
    drop_molecule=True,
    **csv_kwargs
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `smiles_column` | str | `"smiles"` | Name of SMILES column in output |
| `smiles_flavor` | int | `OESMILESFlag_ISOMERIC` | SMILES generation flavor |
| `drop_molecule` | bool | `True` | Drop molecule column from output |
| `**csv_kwargs` | | | Additional arguments passed to CSV writer |

#### `write_oedb()`

Write DataFrame to OERecord database.

```python
df.chem.write_oedb(
    filepath,
    molecule_column,
    *,
    title_column=None,
    sd_columns=None
)
```

#### `write_oedu()`

Write DataFrame to Design Unit file.

```python
df.chem.write_oedu(
    filepath,
    design_unit_column,
    *,
    title_column=None
)
```

#### `write_parquet()`

Write DataFrame to Parquet with molecule serialization.

```python
df.chem.write_parquet(
    filepath,
    molecule_columns=None,
    **parquet_kwargs
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | str, Path | required | Output file path |
| `molecule_columns` | str, list | `None` | Columns to serialize (None auto-detects) |
| `**parquet_kwargs` | | | Additional arguments passed to `write_parquet()` |

---

### LazyFrame Accessor Methods (`lf.chem.*`)

Access these methods via `lf.chem.<method>()`. Most operations require `.collect()` first.

| Method | Returns | Description |
|--------|---------|-------------|
| `has_molecule_columns()` | `bool` | Check if LazyFrame has molecule columns |
| `molecule_columns()` | `list[str]` | Get names of molecule columns |

**Note:** Operations like `to_smiles()`, `substructure_search()`, `filter_valid()`, and `as_molecule()` raise `LazyOperationError` on LazyFrames. Use `.collect()` first or apply these on eager DataFrames.

---

### Series Accessor Methods (`series.chem.*`)

Access these methods via `series.chem.<method>()`:

#### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `token` | `str, None` | The per-column token from the dtype metadata, or `None` |

#### Molecule Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `copy_molecules()` | `Series[MoleculeType]` | Deep copy all molecules |
| `is_valid()` | `Series[bool]` | Boolean mask of valid (non-null) molecules |
| `to_smiles(flavor=OESMILESFlag_ISOMERIC)` | `Series[str]` | Convert to SMILES strings |
| `substructure_search(pattern, adjustH=False)` | `Series[bool]` | Substructure search with SMARTS pattern |

#### Design Unit Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `copy_design_units()` | `Series[DesignUnitType]` | Deep copy all design units |
| `get_ligands(clear_titles=False)` | `Series[MoleculeType]` | Extract ligand molecules |
| `get_proteins(clear_titles=False)` | `Series[MoleculeType]` | Extract protein molecules |
| `get_components(mask)` | `Series[MoleculeType]` | Extract components by mask |

---

### Extension Types

OEPolars provides three custom Polars extension types:

| Type Class | Extension Name | Underlying Type | Description |
|------------|----------------|-----------------|-------------|
| `MoleculeType` | `"molecule"` | `oechem.OEMol` | Stores molecular structures |
| `DesignUnitType` | `"design_unit"` | `oechem.OEDesignUnit` | Stores protein-ligand complexes |
| `DisplayType` | `"display"` | `oedepict.OE2DMolDisplay` | Stores 2D molecular depictions |

Each extension type now accepts an optional `token` parameter that stores per-column metadata:

```python
import oepolars as oepl

# Create a column with a token
dtype = oepl.MoleculeType(token="my_molecules")
s = pl.Series("mol", molecules, dtype=dtype)

# Read the token
s.chem.token  # Returns "my_molecules"
```

---

### Extension Hooks

The `oepolars.extensions` module provides hooks for packages that extend the `chem` namespaces:

#### `register_series_method()`

```python
from oepolars import extensions

def custom_method(self):
    # self._series is the Polars Series
    return self._series.name

extensions.register_series_method("my_method", custom_method)

# Now callable as:
s.chem.my_method()
```

#### `register_dataframe_method()`

```python
def custom_df_method(self):
    # self._df is the Polars DataFrame
    return self._df.width

extensions.register_dataframe_method("my_method", custom_df_method)

# Now callable as:
df.chem.my_method()
```

Both functions accept an `overwrite=True` parameter to replace existing methods.

---

## Examples

Comprehensive Jupyter notebooks are available in the `examples/` directory:

- **01_getting_started.ipynb** - Basic usage, molecular calculations, data manipulation, validity checking
- **02_advanced_features.ipynb** - File I/O, design units, data quality filtering, performance optimization, ML integration

### Example: Complete Workflow

```python
import oepolars as oepl
import polars as pl
from openeye import oechem

# 1. Load data
df = oepl.read_sdf("molecules.sdf")

# 2. Filter invalid molecules
df = df.chem.filter_valid("Molecule")

# 3. Calculate properties
df = df.with_columns(
    MW=pl.col("Molecule").map_elements(oechem.OECalculateMolecularWeight, return_dtype=pl.Float64),
    LogP=pl.col("Molecule").map_elements(oechem.OEGetXLogP, return_dtype=pl.Float64),
    HBD=pl.col("Molecule").map_elements(
        lambda m: oechem.OECount(m, oechem.OEIsHBondDonor()),
        return_dtype=pl.Int64
    ),
    HBA=pl.col("Molecule").map_elements(
        lambda m: oechem.OECount(m, oechem.OEIsHBondAcceptor()),
        return_dtype=pl.Int64
    ),
)

# 4. Filter by Lipinski's Rule of Five
df_druglike = df.filter(
    (pl.col("MW") <= 500) &
    (pl.col("LogP") <= 5) &
    (pl.col("HBD") <= 5) &
    (pl.col("HBA") <= 10)
)

# 5. Substructure search for carboxylic acids
has_acid = df_druglike["Molecule"].chem.substructure_search("C(=O)O")
df_acids = df_druglike.filter(has_acid)

# 6. Export results
df_acids.chem.write_sdf(
    "druglike_acids.sdf",
    molecule_column="Molecule",
    title_column="Title",
    sd_columns=["MW", "LogP"]
)
```

---

## Development

### Running Tests

```bash
invoke test
# or
pytest
```

### Building Package

```bash
invoke build
# or
python -m build
```

### Project Structure

```
oepolars/
├── oepolars/
│   ├── __init__.py              # Public API exports
│   ├── exceptions.py            # Custom exception hierarchy
│   ├── util.py                  # Utility functions
│   ├── types/
│   │   ├── __init__.py
│   │   ├── molecule.py          # MoleculeType extension
│   │   ├── design_unit.py       # DesignUnitType extension
│   │   └── display.py           # DisplayType extension
│   ├── io/
│   │   ├── __init__.py
│   │   ├── readers.py           # Eager file readers
│   │   └── scanners.py          # Lazy file scanners
│   └── namespaces/
│       ├── __init__.py
│       ├── dataframe.py         # DataFrame.chem accessor
│       ├── lazyframe.py         # LazyFrame.chem accessor
│       └── series.py            # Series.chem accessor
├── tests/                       # Test suite
├── examples/                    # Jupyter notebooks
└── pyproject.toml              # Project configuration
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author

**Scott Arne Johnson**
- Email: [scott.arne.johnson@gmail.com](mailto:scott.arne.johnson@gmail.com)

---

## Related Projects

- [OEPandas](https://github.com/scott-arne/oepandas) - Sister project providing OpenEye integration with Pandas DataFrames
- [OpenEye Toolkits](https://www.eyesopen.com/toolkits) - The underlying cheminformatics toolkit
- [Polars](https://pola.rs/) - Lightning-fast DataFrame library that OEPolars extends
