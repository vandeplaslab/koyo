import hashlib
import typing as ty
import uuid

from natsort import natsorted

__all__ = ("get_short_hash", "hash_iterable", "hash_obj", "hash_parameters")


def get_unique_str():
    """Gives random, unique name."""
    return str(uuid.uuid4().hex)


def get_short_hash(n: int = 0) -> str:
    """Get short hash."""
    value = str(uuid.uuid4().hex)
    if n:
        return value[0:n]
    return value


def hash_obj(data: ty.Union[ty.Iterable, ty.List, ty.Dict, ty.Tuple, str, int, float]) -> str:
    """Hash python object."""
    hash_id = hashlib.md5()
    hash_id.update(repr(data).encode("utf-8"))
    return hash_id.hexdigest()


def hash_iterable(iterable, n: int = 0) -> str:
    """Hash iterable object."""
    hash_id = hash_obj(natsorted(iterable))
    if n:
        return hash_id[0:n]
    return hash_id


def hash_parameters(**kwargs) -> str:
    """Hash parameters."""
    hash_id = hashlib.md5()
    for key in natsorted(kwargs.keys()):
        hash_id.update(repr(kwargs[key]).encode("utf-8"))
    return hash_id.hexdigest()
