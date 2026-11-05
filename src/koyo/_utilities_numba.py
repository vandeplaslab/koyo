"""Numba-jitted helpers re-exported by `koyo.utilities`.

They live in a separate module so that `import koyo.utilities` doesn't have to import numba (~0.15 s).
`koyo.utilities` loads this module lazily on first access to any of these names, and the objects returned are the
real numba dispatchers, so they can still be called from other `@numba.njit` functions.
"""

from __future__ import annotations

import typing as ty

import numba as nb
import numpy as np

from koyo.typing import SimpleArrayLike

__all__ = ["filter_array", "find_nearest_index_array", "find_nearest_index_single", "get_array_mask_"]


@nb.njit()
def find_nearest_index_array(data: SimpleArrayLike, value: np.ndarray | ty.Iterable) -> np.ndarray:
    """Find nearest index of asked value.

    Parameters
    ----------
    data : ArrayLike
        input array (e.g. m/z values)
    value : Union[int, float, np.ndarray]
        asked value

    Returns
    -------
    index :
        index value
    """
    data = np.asarray(data)
    return np.asarray([np.argmin(np.abs(data - _value)) for _value in value])


@nb.njit()
def find_nearest_index_single(data: SimpleArrayLike, value: float):
    """Find nearest index of a scalar value.

    Parameters
    ----------
    data : ArrayLike
        input array (e.g. m/z values)
    value : float
        scalar value to search for

    Returns
    -------
    index :
        index value
    """
    return np.argmin(np.abs(data - value))


@nb.njit(parallel=True)
def get_array_mask_(array: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Return mask for array."""
    mask = np.empty(array.shape, dtype=np.bool_)
    for i in nb.prange(array.size):
        val = array[i]
        mask[i] = min_val <= val <= max_val
    return mask


@nb.njit(parallel=True)
def filter_array(
    array: np.ndarray,
    array_to_mask: np.ndarray,
    min_val: float,
    max_val: float,
) -> np.ndarray:
    """Filter array based on another array."""
    # First, count how many elements pass the condition
    count = 0
    for i in nb.prange(array_to_mask.size):
        val = array_to_mask[i]
        if min_val <= val <= max_val:
            count += 1

    # Allocate an output array of the required size
    result = np.zeros(count, dtype=array.dtype)

    # Second pass: fill the result with filtered values (must be serial to preserve order)
    idx = 0
    for i in range(array_to_mask.size):
        val = array_to_mask[i]
        if min_val <= val <= max_val:
            result[idx] = array[i]
            idx += 1
    return result
