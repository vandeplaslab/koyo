"""Utilities for spectrum analysis."""
import typing as ty
from bisect import bisect_left, bisect_right

import numba
import numpy as np

from koyo.typing import SimpleArrayLike


@numba.njit(fastmath=True, cache=True)
def ppm_to_delta_mass(mz: ty.Union[float, np.ndarray], ppm: ty.Union[float, np.ndarray]) -> ty.Union[float, np.ndarray]:
    """Converts a ppm error range to a delta mass in th (da?).

    Parameters
    ----------
    mz : float
        Observed m/z
    ppm : float
        mass range in ppm

    Example
    -------
    ppm_to_delta_mass(1234.567, 50)
    """
    return ppm * mz / 1_000_000.0


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


@numba.njit(cache=True, fastmath=True)
def find_between(data: SimpleArrayLike, min_value: float, max_value: float):
    """Find indices between windows."""
    return np.where(np.logical_and(data >= min_value, data <= max_value))[0]


@numba.njit(cache=True, fastmath=True)
def find_between_tol(data: np.ndarray, value: float, tol: float):
    """Find indices between window and ppm."""
    return find_between(data, value - tol, value + tol)


@numba.njit(cache=True, fastmath=True)
def find_between_ppm(data: np.ndarray, value: float, ppm: float):
    """Find indices between window and ppm."""
    window = get_window_for_ppm(value, abs(ppm))
    return find_between(data, value - window, value + window)


@numba.njit(cache=True, fastmath=True)
def find_between_batch(array: np.ndarray, min_value: np.ndarray, max_value: np.ndarray):
    """Find indices between specified boundaries for many items."""
    min_indices = np.searchsorted(array, min_value, side="left")
    max_indices = np.searchsorted(array, max_value, side="right")

    res = []
    for i in range(len(min_value)):
        _array = array[min_indices[i] : max_indices[i]]
        res.append(min_indices[i] + find_between(_array, min_value[i], max_value[i]))
    return res


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


def get_mzs_for_tol(mzs: np.ndarray, tol: ty.Optional[float] = None, ppm: ty.Optional[float] = None):
    """Get min/max values for specified tolerance or ppm."""
    if tol is None and ppm is None or tol == 0 and ppm == 0:
        raise ValueError("Please specify `tol` or `ppm`.")
    elif tol is not None and ppm is not None:
        raise ValueError("Please only specify `tol` or `ppm`.")

    mzs = np.asarray(mzs)
    if tol:
        mzs_min = mzs - tol
        mzs_max = mzs + tol
    else:
        tol = np.asarray([get_window_for_ppm(mz, ppm) for mz in mzs])
        mzs_min = mzs - tol
        mzs_max = mzs + tol
    return mzs_min, mzs_max


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
