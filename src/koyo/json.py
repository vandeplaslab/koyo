"""All utility functions that deal with JSON files."""

import typing as ty
from pathlib import Path

import numpy as np

from koyo.typing import PathLike
from koyo.utilities import _remove_duplicates_from_dict

try:
    import ujson as json
except ImportError:
    import json


def default(o: ty.Any) -> ty.Any:
    """Fixes writing of objects containing numpy dtypes.

    Parameters
    ----------
    o : any
        object to be serialized

    Returns
    -------
    int / float
        converted object

    Raises
    ------
    TypeError
    """
    if isinstance(o, (np.int64, np.int32, np.int16, np.integer)):
        return int(o)
    elif isinstance(o, (np.float64, np.float32, np.float16, np.floating)):
        return float(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    elif isinstance(o, np.bool_):
        return bool(o)
    elif isinstance(o, Path):
        return str(o)
    raise TypeError("Could not convert {} of type {}".format(*o), type(o))


def read_json(filepath: PathLike) -> ty.Any:
    """Read JSON data and metadata.

    Parameters
    ----------
    filepath : PathLike
        path to the JSON file

    Returns
    -------
    loaded json data
    """
    with open(filepath) as f_ptr:
        obj = json.load(f_ptr)
    return obj


def write_json(filepath: PathLike, obj, indent: int = 4, check_existing: bool = False, compress: bool = False) -> Path:
    """Write data to JSON file.

    Parameters
    ----------
    filepath : PathLike
        path to JSON file
    obj : any
        object containing data
    indent : int, optional
        number of spaces to indent by, by default 4
    check_existing : bool, optional
        if True, existing JSON file data will be merged with the new data object
    compress : bool, optional
    """
    kws = {} if not compress else {"separators": (",", ":")}
    indent = 1 if compress else indent

    filepath = Path(filepath)
    if not check_existing or not filepath.exists():
        with open(filepath, "w") as f_ptr:
            json.dump(obj, f_ptr, indent=indent, default=default, **kws)
    else:
        with open(filepath, "r+") as f_ptr:
            data = json.load(f_ptr)

            if isinstance(data, list):
                data.extend(obj)
            elif isinstance(data, dict):
                data.update(obj)

            # remove duplicates
            data = _remove_duplicates_from_dict(data)

            f_ptr.seek(0)  # rewind
            json.dump(data, f_ptr, indent=indent, default=default, **kws)
            f_ptr.truncate()
    return filepath


def write_json_gzip(filepath: PathLike, obj: ty.Any) -> Path:
    """Write gzip compressed JSON data."""
    import gzip

    filepath = Path(filepath)
    if not filepath.with_suffix(".gz"):
        filepath = filepath.with_suffix(".gz")
    with gzip.open(filepath, "w") as f_ptr:
        f_ptr.write(json.dumps(obj, default=default).encode("utf-8"))
    return filepath


def read_json_gzip(filepath: PathLike) -> ty.Any:
    """Read gzip compressed JSON data."""
    import gzip

    with gzip.open(filepath, "r") as f_ptr:
        return json.loads(f_ptr.read().decode("utf-8"))


# compatibility
read_json_data = read_json
write_json_data = write_json
