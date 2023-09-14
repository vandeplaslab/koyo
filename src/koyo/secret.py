import hashlib
import typing as ty
import uuid

from pathlib import Path
from natsort import natsorted


def get_unique_str():
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


def hash_obj(data: ty.Union[ty.Iterable, ty.List, ty.Dict, ty.Tuple, Path, str, int, float]) -> str:
    """Hash python object."""
    hash_id = hashlib.md5()
    hash_id.update(repr(data).encode("utf-8"))
    return hash_id.hexdigest()

def uuid_iterable(iterable) -> str:
    """Hash iterable object."""
    return str(uuid.UUID(hash_iterable(iterable)))


def hash_iterable(iterable, n_in_hash: int = 0) -> str:
    """Hash iterable object."""
    hash_id = hash_obj(natsorted(iterable))
    return hash_id[0:n_in_hash] if n_in_hash else hash_id

def uuid_parameters(**kwargs) -> str:
    """Hash iterable object."""
    p = hash_parameters(**kwargs)
    return str(uuid.UUID(hash_parameters(**kwargs)))

def hash_parameters(n_in_hash: int = 0, **kwargs) -> str:
    """Hash parameters."""
    hash_id = hashlib.md5()
    for key in natsorted(kwargs.keys()):
        hash_id.update(repr(kwargs[key]).encode("utf-8"))
    value = hash_id.hexdigest()
    return value[0:n_in_hash] if n_in_hash else value
