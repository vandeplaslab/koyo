"""Various utility functions speed up using numba."""

import numba as nb
import numpy as np


def nanquantile_along_axis(data: np.ndarray, q1: float, q2: float, axis: int = 0) -> np.ndarray:
    """Numba quantile."""
    return np.vstack(_nanquantile_along_axis(data, q1, q2, axis == 0)).T


@nb.njit(fastmath=True, nogil=True, cache=True)
def _nanquantile_along_axis(data: np.ndarray, q1: float, q2: float, transpose: bool = True):
    """Numba quantile."""
    return [np.nanquantile(d, (q1, q2)) for d in (data.T if transpose else data)]


def quantile_along_axis(data: np.ndarray, q1: float, q2: float, axis: int = 0) -> np.ndarray:
    """Numba quantile."""
    return np.vstack(_quantile_along_axis(data, q1, q2, axis == 0)).T


@nb.njit(fastmath=True, nogil=True, cache=True)
def _quantile_along_axis(data: np.ndarray, q1: float, q2: float, transpose: bool = True):
    """Numba quantile."""
    return [np.quantile(d, (q1, q2)) for d in (data.T if transpose else data)]


def _precompile() -> None:
    """Precompile numba functions."""
    import os

    if os.environ.get("KOYO_JIT_PRE", "0") == "0":
        return
    _nanquantile_along_axis(np.random.rand(100, 10), 0.25, 0.75)
    _quantile_along_axis(np.random.rand(100, 10), 0.25, 0.75)


_precompile()
