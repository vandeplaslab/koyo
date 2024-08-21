"""Definitions of types."""

import typing as ty
from pathlib import Path

if ty.TYPE_CHECKING:
    import numpy as np
    from scipy import sparse


__all__ = ("ArrayLike", "PathLike", "SparseArray", "CompressedSparseArray")

ArrayLike = ty.Union[ty.List, ty.Tuple, "np.ndarray", "sparse.csc_matrix", "sparse.csr_matrix"]
SimpleArrayLike = ty.TypeVar("SimpleArrayLike", ty.List, "np.ndarray", ty.Iterable)
CompressedSparseArray = ty.Union["sparse.csc_matrix", "sparse.csr_matrix"]
SparseArray = ty.Union["sparse.csc_matrix", "sparse.csr_matrix", "sparse.coo_matrix"]
PathLike = ty.Union[str, Path]
