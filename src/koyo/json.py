"""All utility functions that deal with JSON files."""
import os
from pathlib import Path

import numpy as np

from koyo.typing import PathLike
from koyo.utilities import _remove_duplicates_from_dict

try:
    import ujson as json
except ImportError:
    import json


def default(o):
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
    if isinstance(o, (np.int64, np.int32, np.int16)):
        return int(o)
    elif isinstance(o, (np.float64, np.float32, np.float16)):
        return float(o)
    elif isinstance(o, Path):
        return str(o)
    raise TypeError("Could not convert {} of type {}".format(*o), type(o))


def read_json_data(filepath: PathLike):
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
        json_data = json.load(f_ptr)

    return json_data


def write_json_data(filepath: PathLike, obj, indent=4, check_existing=False):
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
    """
    if not check_existing or not os.path.exists(filepath):
        with open(filepath, "w") as f_ptr:
            json.dump(obj, f_ptr, indent=indent, default=default)
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
            json.dump(data, f_ptr, indent=indent, default=default)
            f_ptr.truncate()
    return filepath
