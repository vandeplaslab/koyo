"""Utilities for spectrum analysis."""
import typing as ty
from bisect import bisect_left, bisect_right

import numba
import numpy as np
import scipy.signal
import scipy.ndimage

from koyo.typing import SimpleArrayLike
from koyo.utilities import find_nearest_index, find_nearest_index_batch


def running_average(x: np.ndarray, size: int) -> np.ndarray:
    """Running average."""
    return scipy.ndimage.uniform_filter1d(x, size, mode="nearest")
    # return np.convolve(x, np.ones(size), "valid") / size


def _cluster_within_ppm_with_index(array: np.ndarray, ppm: float):
    """Cluster results within ppm tolerance."""
    tmp = array.copy()
    indices = np.arange(tmp.size)
    groups = []
    index_groups = []
    while len(tmp):
        # select seed
        seed = tmp.min()
        mask = np.abs(ppm_error(tmp, seed)) <= ppm
        groups.append(tmp[mask])
        index_groups.append(indices[mask])
        tmp = tmp[~mask]
        indices = indices[~mask]
    return groups, index_groups


def cluster_within_ppm(array: np.ndarray, ppm: float):
    """Cluster results within ppm tolerance."""
    tmp = array.copy()
    groups = []
    while len(tmp):
        # select seed
        seed = tmp.min()
        mask = np.abs(ppm_error(tmp, seed)) <= ppm
        groups.append(tmp[mask])
        tmp = tmp[~mask]
    return groups


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
def ppm_error(
    measured_mz: ty.Union[float, np.ndarray], theoretical_mz: ty.Union[float, np.ndarray]
) -> ty.Union[float, np.ndarray]:
    """Calculate ppm error."""
    return ((measured_mz - theoretical_mz) / theoretical_mz) * 1e6


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
    a = np.asarray(a, dtype=np.float32)
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


def parabolic_centroid(
    mzs: np.ndarray, intensities: np.ndarray, peak_threshold: float = 0
) -> ty.Tuple[np.ndarray, np.ndarray]:
    """Calculate centroid position.

    This function was taken from msiwarp package available on GitHub
    """
    peak_indices, _ = scipy.signal.find_peaks(intensities, height=peak_threshold)
    peak_left = peak_indices - 1
    peak_right = peak_indices + 1

    n = len(peak_indices)

    x = np.zeros((n, 3))
    y = np.zeros((n, 3))

    x[:, 0] = mzs[peak_left]
    x[:, 1] = mzs[peak_indices]
    x[:, 2] = mzs[peak_right]

    y[:, 0] = intensities[peak_left]
    y[:, 1] = intensities[peak_indices]
    y[:, 2] = intensities[peak_right]

    a = ((y[:, 2] - y[:, 1]) / (x[:, 2] - x[:, 1]) - (y[:, 1] - y[:, 0]) / (x[:, 1] - x[:, 0])) / (x[:, 2] - x[:, 0])

    b = (
        (y[:, 2] - y[:, 1]) / (x[:, 2] - x[:, 1]) * (x[:, 1] - x[:, 0])
        + (y[:, 1] - y[:, 0]) / (x[:, 1] - x[:, 0]) * (x[:, 2] - x[:, 1])
    ) / (x[:, 2] - x[:, 0])

    mzs_parabolic = (1 / 2) * (-b + 2 * a * x[:, 1]) / a
    intensities_parabolic = a * (mzs_parabolic - x[:, 1]) ** 2 + b * (mzs_parabolic - x[:, 1]) + y[:, 1]
    mask = ~np.isnan(mzs_parabolic)
    return mzs_parabolic[mask], intensities_parabolic[mask]


def get_ppm_axis(mz_start: float, mz_end: float, ppm: float):
    """Compute sequence of m/z values at a particular ppm."""
    import math

    if mz_start == 0 or mz_end == 0 or ppm == 0:
        raise ValueError("Input values cannot be equal to 0.")
    length = (np.log(mz_end) - np.log(mz_start)) / np.log((1 + 1e-6 * ppm) / (1 - 1e-6 * ppm))
    length = math.floor(length) + 1
    mz = mz_start * np.power(((1 + 1e-6 * ppm) / (1 - 1e-6 * ppm)), (np.arange(length)))
    return mz


@numba.njit()
def trim_axis(x: np.ndarray, y: np.ndarray, min_val: float, max_val: float):
    """Trim axis to prevent accumulation of edges."""
    mask = np.where((x >= min_val) & (x <= max_val))
    return x[mask], y[mask]


@numba.njit()
def set_ppm_axis(mz_x: np.ndarray, mz_y: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Set values for axis."""
    mz_idx = np.digitize(x, mz_x, True)
    for i, idx in enumerate(mz_idx):
        mz_y[idx] += y[i]
    return mz_y


def get_ppm_offsets(mz_x: np.ndarray, ppm: float, min_spacing: float = 1e-5, every_n: int = 100) -> np.ndarray:
    """Generate correction map of specified ppm."""
    spacing = min_spacing  # if ppm > 0 else -min_spacing
    is_subtract = ppm < 0
    ppm = abs(ppm)
    _mzx = mz_x[::every_n]
    result = np.zeros_like(_mzx)
    mzx = mz_x
    full_result = np.zeros_like(mzx)
    if ppm == 0:
        return full_result

    n = 10
    index_offset = 0
    for i, val in enumerate(_mzx):
        while True:
            offsets = np.full(n, spacing) * np.arange(index_offset, index_offset + n)
            errors = ppm_error(val, val - offsets)
            index = find_nearest_index(errors, ppm)
            nearest = errors[index]
            if nearest >= ppm:
                offset = offsets[index]
                break
            index_offset += n
        result[i] = offset
        index_offset -= n * 2

    start_idx = 0
    indices = find_nearest_index_batch(mzx, _mzx)
    indices[-1] = len(mzx)
    for i, end_idx in enumerate(indices):
        full_result[start_idx:end_idx] = result[i]
        start_idx = end_idx
    return full_result if not is_subtract else -full_result


def get_multi_ppm_offsets(mz_x: np.ndarray, ppm_ranges, min_spacing: float = 1e-5, every_n: int = 100) -> np.ndarray:
    """Generate correction map of specified ppm."""
    start_offset = 0
    _mzx = mz_x[::every_n]
    offsets = np.zeros_like(_mzx)
    mzx = mz_x
    full_offsets = np.zeros_like(mzx)
    if all(_ppm[2] == 0 for _ppm in ppm_ranges):
        return full_offsets

    ppm_ = []
    for x_min, x_max, ppm in ppm_ranges:
        spacing = min_spacing if ppm > 0 else -min_spacing
        ppm_.append((x_min, x_max, ppm, spacing))

    for i, val in enumerate(_mzx):
        offset = start_offset
        for x_min, x_max, ppm, spacing in ppm_:
            if x_min <= val <= x_max:
                if ppm > 0:
                    while ppm_error(val, val - offset) <= ppm:
                        offset += spacing
                else:
                    while ppm_error(val, val - offset) >= ppm:
                        offset += spacing
                break
        offsets[i] = offset

    start_idx = 0
    indices = find_nearest_index_batch(mzx, _mzx)
    indices[-1] = len(mzx)
    for i, end_idx in enumerate(indices):
        full_offsets[start_idx:end_idx] = offsets[i]
        start_idx = end_idx
    return full_offsets
