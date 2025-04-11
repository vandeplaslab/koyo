import hashlib
import typing as ty
import uuid
from pathlib import Path

import numpy as np
from natsort import natsorted


def get_unique_str() -> str:
    """Gives random, unique name."""
    return str(uuid.uuid4().hex)


def get_short_hash(n_in_hash: int = 0) -> str:
    """Get short hash."""
    value = get_unique_str()
    return value[0:n_in_hash] if n_in_hash else value


def get_uuid() -> str:
    """Get unique id."""
    return str(uuid.uuid4())


def uuid_obj(data: ty.Union[ty.Iterable, ty.List, ty.Dict, ty.Tuple, Path, str, int, float]) -> str:
    """Hash python object."""
    return str(uuid.UUID(hash_obj(data)))


def uuid_iterable(iterable: ty.Iterable) -> str:
    """Hash iterable object."""
    return str(uuid.UUID(hash_iterable(iterable)))


def uuid_parameters(**kwargs) -> str:
    """Hash iterable object."""
    return str(uuid.UUID(hash_parameters(**kwargs)))


def hash_obj(data: ty.Union[ty.Iterable, ty.List, ty.Dict, ty.Tuple, Path, str, int, float], n_in_hash: int = 0) -> str:
    """Hash python object."""
    hash_id = hashlib.md5()
    hash_id.update(repr(data).encode("utf-8"))
    value = hash_id.hexdigest()
    return value[0:n_in_hash] if n_in_hash else value


def hash_iterable(iterable: ty.Iterable, n_in_hash: int = 0) -> str:
    """Hash iterable object."""
    iterable = list(iterable)
    hash_id = hash_obj(_natsort_if_iterable(iterable))
    return hash_id[0:n_in_hash] if n_in_hash else hash_id


def _natsort_if_iterable(value: ty.Any) -> ty.Any:
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return list(natsorted(value))
    elif isinstance(value, dict):
        return {key: _natsort_if_iterable(value[key]) for key in value}
    return value


def hash_parameters(n_in_hash: int = 0, exclude_keys: tuple[str, ...] = (), **kwargs: ty.Any) -> str:
    """Hash parameters."""
    if exclude_keys is None:
        exclude_keys = ()
    hash_id = hashlib.md5()
    for key in natsorted(kwargs.keys()):
        if key in exclude_keys:
            continue
        hash_id.update(repr(_natsort_if_iterable(kwargs[key])).encode("utf-8"))
    value = hash_id.hexdigest()
    return value[0:n_in_hash] if n_in_hash else value
