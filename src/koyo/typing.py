"""Definitions of types."""

import sys
import typing as ty
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
