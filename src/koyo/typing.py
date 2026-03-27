"""Definitions of types."""

from __future__ import annotations

import builtins
import sys
import typing as ty
from enum import EnumMeta
from pathlib import Path

if ty.TYPE_CHECKING:
    import numpy as np
    from scipy import sparse


__all__ = ("ArrayLike", "CompressedSparseArray", "PathLike", "SparseArray")

ArrayLike = ty.Union[ty.List, ty.Tuple, "np.ndarray", "sparse.csc_matrix", "sparse.csr_matrix"]
SimpleArrayLike = ty.TypeVar("SimpleArrayLike", ty.List, "np.ndarray", ty.Iterable)
CompressedSparseArray = ty.Union["sparse.csc_matrix", "sparse.csr_matrix"]
SparseArray = ty.Union["sparse.csc_matrix", "sparse.csr_matrix", "sparse.coo_matrix"]
PathLike = ty.Union[str, Path]


if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    # in 3.11+, using the below class in an f-string would put the enum name instead of its value
    from enum import Enum

    class StrEnum(str, Enum):
        pass


class StringEnumMeta(EnumMeta):
    def __getitem__(self, item):
        """Set the item name case to uppercase for name lookup."""
        if isinstance(item, str):
            item = item.upper()

        return super().__getitem__(item)

    def __call__(
        cls,
        value,
        names=None,
        *,
        module=None,
        qualname=None,
        type=None,  # noqa: A002
        start=1,
    ):
        """Set the item value case to lowercase for value lookup."""
        # simple value lookup
        if names is None:
            if isinstance(value, str):
                return super().__call__(value.lower())
            if isinstance(value, cls):
                return value

            raise ValueError(
                f"{cls} may only be called with a `str` or an instance of {cls}. Got {builtins.type(value)}",
            )

        # otherwise create new Enum class
        return cls._create_(
            value,
            names,
            module=module,
            qualname=qualname,
            type=type,
            start=start,
        )

    def keys(self) -> list[str]:
        return list(map(str, self))


class StringEnum(Enum, metaclass=StringEnumMeta):
    @staticmethod
    def _generate_next_value_(name: str, start, count, last_values) -> str:
        """Autonaming function assigns each value its own name as a value."""
        return name.lower()

    def __str__(self) -> str:
        """String representation: The string method returns the lowercase
        string of the Enum name.
        """
        return self.value

    def __eq__(self, other: object) -> bool:
        if type(self) is type(other):
            return self is other
        if isinstance(other, str):
            return str(self) == other
        return False

    def __hash__(self) -> int:
        return hash(str(self))
