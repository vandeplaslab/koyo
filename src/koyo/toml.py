"""TOML utility functions."""
import os
import typing as ty

import toml

from koyo.typing import PathLike
from koyo.utilities import _remove_duplicates_from_dict


def read_toml_data(filepath: PathLike):
    """Read TOML data and metadata.

    Parameters
    ----------
    filepath : str
        path to the TOML file

    Returns
    -------
    data : dict
        dictionary with TOML data
    """
    with open(filepath) as f_ptr:
        data = toml.load(f_ptr)
    return data


def write_toml_data(filepath: PathLike, obj: ty.Dict, check_existing: bool = False):
    """Write data to TOML file.

    Parameters
    ----------
    filepath : PathLike
        path to TOML file
    obj : dict
        object containing data
    check_existing : bool, optional
        if True, existing JSON file data will be merged with the new data object
    """
    if not check_existing or not os.path.exists(filepath):
        with open(filepath, "w") as f_ptr:
            toml.dump(obj, f_ptr)
    else:
        with open(filepath, "r+") as f_ptr:
            try:
                data = toml.load(f_ptr)
            except toml.decoder.TomlDecodeError:
                data = {}

            if isinstance(data, list):
                data.extend(obj)
            elif isinstance(data, dict):
                data.update(obj)

            # remove duplicates
            data = _remove_duplicates_from_dict(data)

            f_ptr.seek(0)  # rewind
            toml.dump(data, f_ptr)
            f_ptr.truncate()
