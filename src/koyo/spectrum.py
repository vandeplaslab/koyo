"""Utilities for spectrum analysis."""
import typing as ty
from bisect import bisect_left, bisect_right

import numba
import numpy as np

from koyo.typing import SimpleArrayLike


@numba.njit(fastmath=True, cache=True)
def ppm_error(x: ty.Union[float, np.ndarray], y: ty.Union[float, np.ndarray]) -> ty.Union[float, np.ndarray]:
    """Calculate ppm error."""
    return ((x - y) / y) * 1e6


@numba.njit(fastmath=True, cache=True)
def get_window_for_ppm(mz: float, ppm: float) -> float:
    """Calculate window size for specified peak at specified ppm."""
    step = mz * 1e-6  # calculate appropriate step size for specified mz value
    peak_x_ppm = mz
    is_subtract = ppm < 0
    ppm = abs(ppm)
    while True:
        if ((peak_x_ppm - mz) / mz) * 1e6 >= ppm:
            break
        peak_x_ppm += step
    value = peak_x_ppm - mz
    return value if not is_subtract else -value


def ppm_diff(a: np.ndarray, axis=-1) -> np.ndarray:
    """Calculate the ppm difference between set of values in array.

    This function is inspired by `np.diff` which very efficiently computes the difference between adjacent points.
    """
    a = np.asarray(a, dtype=np.float)
    nd = a.ndim
    axis = np.core.multiarray.normalize_axis_index(axis, nd)
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    return (np.subtract(a[slice1], a[slice2]) / a[slice2]) * 1e6


@numba.njit()
def find_between(data: SimpleArrayLike, min_value: float, max_value: float) -> np.ndarray:
    """Find indices between windows."""
    return np.where(np.logical_and(data >= min_value, data <= max_value))[0]


@numba.njit()
def find_between_ppm(data: SimpleArrayLike, value: float, ppm: float):
    """Find indices between window and ppm."""
    window = get_window_for_ppm(value, abs(ppm))
    return find_between(data, value - window, value + window)


def get_peaklist_window_for_ppm(peaklist: np.ndarray, ppm: float) -> ty.List[ty.Tuple[float, float]]:
    """Retrieve peaklist + tolerance."""
    _peaklist = []
    for mz in peaklist:
        _peaklist.append((mz, get_window_for_ppm(mz, ppm)))
    return _peaklist


def get_peaklist_window_for_da(peaklist: np.ndarray, da: float) -> ty.List[ty.Tuple[float, float]]:
    """Retrieve peaklist + tolerance."""
    _peaklist = []
    for mz in peaklist:
        _peaklist.append((mz, da))
    return _peaklist


def bisect_spectrum(x_spectrum, mz_value, tol: float) -> ty.Tuple[int, int]:
    """Get left/right window of extraction for peak."""
    ix_l, ix_u = (
        bisect_left(x_spectrum, mz_value - tol),
        bisect_right(x_spectrum, mz_value + tol) - 1,
    )
    if ix_l == len(x_spectrum):
        return len(x_spectrum), len(x_spectrum)
    if ix_u < 1:
        return 0, 0
    if ix_u == len(x_spectrum):
        ix_u -= 1
    if x_spectrum[ix_l] < (mz_value - tol):
        ix_l += 1
    if x_spectrum[ix_u] > (mz_value + tol):
        ix_u -= 1
    return ix_l, ix_u
