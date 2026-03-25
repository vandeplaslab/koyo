# Kōyō

A utility library of reusable Python functions used throughout various projects in the Van de Plas lab.

[![License](https://img.shields.io/pypi/l/koyo.svg?color=green)](https://github.com/vandeplaslab/koyo/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/koyo.svg?color=green)](https://pypi.org/project/koyo)
[![Python Version](https://img.shields.io/pypi/pyversions/koyo.svg?color=green)](https://python.org)
[![CI](https://github.com/vandeplaslab/koyo/actions/workflows/ci.yml/badge.svg)](https://github.com/vandeplaslab/koyo/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/vandeplaslab/koyo/branch/main/graph/badge.svg)](https://codecov.io/gh/vandeplaslab/koyo)

## Features

| Module | Description |
|---|---|
| `koyo.click` | CLI helpers: path parsing, plugin loading, parameter tables, timed iterators |
| `koyo.system` | Platform detection, environment variable helpers, version checks, disk-space checks |
| `koyo.utilities` | Generic utils: nearest-index search (Numba-accelerated), array rescaling, chunking, size formatting |
| `koyo.timer` | Human-readable timing and `measure_time` context manager |
| `koyo.multicore` | `joblib`-backed multi-core helpers with RAM-aware CPU count |
| `koyo.logging` | `loguru`-based logging configuration |
| `koyo.color` | RGB/hex conversions and matplotlib colour utilities |
| `koyo.path` | File-system helpers: zero-copy `sendfile`, glob utilities, URI handling |
| `koyo.zarr` | Zarr-backed compressed array storage (Blosc/zip) |
| `koyo.sparse` | `scipy.sparse` convenience wrappers |
| `koyo.compression` | Generic data-compression utilities |
| `koyo.fig_mixin` | `FigureMixin` for composable figure export |
| `koyo.pdf_mixin` | PDF export backend |
| `koyo.pptx_mixin` | PowerPoint export backend (requires `pptx` extra) |
| `koyo.spectrum` | Spectroscopy (mass-spectrometry) utilities |
| `koyo.mosaic` | Rectangle-packing algorithm for image/figure mosaics |
| `koyo.decorators` | `retry` decorator with exponential back-off |
| `koyo.json` | JSON serialisation helpers |
| `koyo.toml` | TOML file handling |

## Installation

```bash
pip install koyo
```

### Optional dependencies

```bash
# PowerPoint export
pip install koyo[pptx]

# Image-processing utilities (scikit-learn)
pip install koyo[image]

# Clipboard support
pip install koyo[clipboard]

# Windows shortcut support
pip install koyo[win]

# All extras
pip install koyo[pptx,image,clipboard,win]
```

## Quick start

```python
from koyo.utilities import find_nearest_index, rescale, format_size
from koyo.system import is_installed, get_version
from koyo.timer import measure_time

# Find the closest index in a sorted array
import numpy as np
arr = np.linspace(0, 100, 1000)
idx = find_nearest_index(arr, 42.7)

# Rescale an array to [0, 1]
scaled = rescale(arr, 0, 1)

# Human-readable byte size
print(format_size(1_500_000))  # "1.4M"

# Check whether a package is installed
if is_installed("zarr"):
    print(f"zarr {get_version('zarr')} is available")

# Time a block of code
with measure_time() as t:
    result = some_expensive_function()
print(f"Finished in {t():.3f}s")
```

## Contributing

Contributions are always welcome. Please feel free to submit PRs with new features, bug fixes, or documentation improvements.

```bash
git clone https://github.com/vandeplaslab/koyo.git
cd koyo
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

Run linting:

```bash
ruff check src/
```
