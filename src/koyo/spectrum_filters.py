"""Filters.

Majority of these filters had been taken from ms_peak_picker
https://github.com/mobiusklein/ms_peak_picker/blob/master/ms_peak_picker/scan_filter.py
"""
import typing as ty

import numpy as np

from koyo.utilities import find_nearest_index

#: Global register of all named scan filters
FILTER_REGISTER = {}


def register(name, *args, **kwargs):
    """Decorate a class to register a name for it, optionally with a set of associated initialization parameters.

    Parameters
    ----------
    name : str
        The name to register the filter under.
    *args
        Positional arguments forwarded to the decorated class's
        initialization method
    **kwargs
        Keyword arguments forwarded to the decorated class's
        initialization method

    Returns
    -------
    function
        A decorating function which will carry out the registration
        process on the decorated class.
    """

    def _wrap(cls):
        FILTER_REGISTER[name] = cls(*args, **kwargs)
        return cls

    return _wrap


class FilterBase:
    """A base type for Filters over raw signal arrays.

    All subtypes should provide a :meth:`filter` method
    which takes arguments *mz_array* and *intensity_array*
    which will be NumPy Arrays.
    """

    def __repr__(self) -> str:
        return f"Filter<{self.__class__.__name__}>"

    def filter(self, mz_array: np.ndarray, intensity_array: np.ndarray) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Filter array."""
        return mz_array, intensity_array

    def __call__(self, mz_array: np.ndarray, intensity_array: np.ndarray) -> ty.Tuple[np.ndarray, np.ndarray]:
        return self.filter(mz_array, intensity_array)


@register("median")
class MedianIntensityFilter(FilterBase):
    """Filter signal below the median signal."""

    def filter(self, mz_array: np.ndarray, intensity_array: np.ndarray) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Filter."""
        mask = intensity_array < np.median(intensity_array)
        intensity_array = np.array(intensity_array)
        intensity_array[mask] = 0.0
        return mz_array, intensity_array


@register("mean_below_mean")
class MeanBelowMeanFilter(FilterBase):
    """Filter signal below the mean below the mean."""

    def filter(self, mz_array: np.ndarray, intensity_array: np.ndarray) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Filter."""
        mean = intensity_array.mean()
        mean_below_mean = (intensity_array < mean).mean()
        mask = intensity_array < mean_below_mean
        intensity_array[mask] = 0.0
        return mz_array, intensity_array


@register("savitzky_golay")
@register("sav_gol")
class SavitzkyGolayFilter(FilterBase):
    """Apply `Savitsky-Golay smoothing <https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter>` to the signal.

    Attributes
    ----------
    deriv : int
        Number of derivatives to take
    poly_order : int
        Order of the polynomial to construct
    window_length : int
        Number of data points to include around the current point
    """

    def __init__(self, window_length=5, poly_order=3, deriv=0):
        self.window_length = window_length
        self.poly_order = poly_order
        self.deriv = deriv

    def filter(self, mz_array: np.ndarray, intensity_array: np.ndarray) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Filter."""
        from scipy.signal import savgol_filter

        if len(intensity_array) <= self.window_length:
            return mz_array, intensity_array
        smoothed = savgol_filter(
            intensity_array, window_length=self.window_length, polyorder=self.poly_order, deriv=self.deriv
        ).clip(0)
        mask = smoothed > 0
        smoothed[~mask] = 0
        return mz_array, smoothed


@register("gaussian", 0.02)
class GaussianSmoothFilter(FilterBase):
    """Gaussian smoothing."""

    def __init__(self, width=0.02):
        self.width = width

    def filter(self, mz_array: np.ndarray, intensity_array: np.ndarray) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Filter."""
        from scipy.ndimage import gaussian_filter

        intensity_array = gaussian_filter(intensity_array, self.width)
        return mz_array, intensity_array


@register("mov_avg", 20)
class MovingAverageFilter(FilterBase):
    """Gaussian smoothing."""

    def __init__(self, size=20):
        self.size = size

    def filter(self, mz_array: np.ndarray, intensity_array: np.ndarray) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Filter."""
        from scipy.ndimage import uniform_filter1d

        intensity_array = uniform_filter1d(intensity_array, self.size)
        return mz_array, intensity_array


@register("tenth_percent_of_max")
@register("one_percent_of_max", 0.01)
class NPercentOfMaxFilter(FilterBase):
    """Filter-out N-percent."""

    def __init__(self, p=0.001):
        self.p = p

    def filter(self, mz_array: np.ndarray, intensity_array: np.ndarray) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Filter."""
        mask = (intensity_array / np.max(intensity_array)) < self.p
        intensity_array_clone = np.array(intensity_array)
        intensity_array_clone[mask] = 0.0
        return mz_array, intensity_array_clone


class ConstantThreshold(FilterBase):
    """Threshold."""

    def __init__(self, constant):
        self.constant = constant

    def filter(self, mz_array: np.ndarray, intensity_array: np.ndarray) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Filter."""
        mask = intensity_array < self.constant
        intensity_array = intensity_array.copy()
        intensity_array[mask] = 0
        return mz_array, intensity_array


class MaximumScaler(FilterBase):
    """Scaler."""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def filter(self, mz_array: np.ndarray, intensity_array: np.ndarray) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Filter."""
        max_value = np.max(intensity_array)
        if max_value > self.threshold:
            intensity_array = intensity_array / max_value * self.threshold
        return mz_array, intensity_array


class IntensityScaler(FilterBase):
    """Intensity scaler."""

    def __init__(self, scale):
        self.scale = scale

    def filter(self, mz_array: np.ndarray, intensity_array: np.ndarray) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Filter."""
        intensity_array = intensity_array * self.scale
        return mz_array, intensity_array


@register("linear", 0.005)
class LinearResampling(FilterBase):
    """Perform linear resampling."""

    def __init__(self, spacing, mz_start=None, mz_end=None):
        self.spacing = spacing
        self.mz_start = mz_start
        self.mz_end = mz_end

    def filter(self, mz_array: np.ndarray, intensity_array: np.ndarray) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Filter."""
        if self.mz_start is None:
            self.mz_start = np.min(mz_array)
        if self.mz_end is None:
            self.mz_end = np.max(mz_array)
        new_mz = np.arange(self.mz_start, self.mz_end + self.spacing, self.spacing)
        new_intensity = np.interp(new_mz, mz_array, intensity_array)
        return new_mz, new_intensity


class PpmResampling(FilterBase):
    """Parts-per-million resampling."""

    def __init__(self, ppm: float, mz_start: float, mz_end: float):
        from koyo.spectrum import get_ppm_axis

        if ppm <= 0:
            raise ValueError("Please specify value of ppm that is larger than 0.")
        if mz_start == mz_end or mz_start > mz_end:
            raise ValueError("Please specify `mz_start` and `mz_end` that are not equal and where `mz_start > mz_end`")

        self.ppm = ppm
        self.mz_start = float(mz_start)
        self.mz_end = float(mz_end)
        self.mz_new = get_ppm_axis(self.mz_start, self.mz_end, self.ppm)

    def filter(self, mz_array: np.ndarray, intensity_array: np.ndarray) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Filter."""
        # from scipy.interpolate import interp1d
        return self.mz_new, np.interp(self.mz_new, mz_array, intensity_array)
        # func = interp1d(mz_array, intensity_array, fill_value=0, bounds_error=False)
        # return self.mz_new, func(self.mz_new)


class Crop(FilterBase):
    """Crop spectrum."""

    def __init__(self, mz_start: float, mz_end: float):
        self.mz_start = mz_start
        self.mz_end = mz_end

    def filter(self, mz_array: np.ndarray, intensity_array: np.ndarray) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Filter."""
        start_idx, end_idx = find_nearest_index(mz_array, [self.mz_start, self.mz_end])
        return mz_array[start_idx:end_idx], intensity_array[start_idx:end_idx]


class ConstantRecalibrate(FilterBase):
    """Recalibrate mass."""

    def __init__(self, offset: float):
        self.offset = offset

    def filter(self, mz_array: np.ndarray, intensity_array: np.ndarray) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Filter."""
        return mz_array + self.offset, intensity_array


class IndexRecalibrate(FilterBase):
    """Recalibrate mass axis using mass bins."""

    def __init__(self, shift: int):
        self.shift = shift

    def filter(self, mz_array: np.ndarray, intensity_array: np.ndarray) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Filter."""
        from msalign.utilities import shift

        _mz_array = shift(mz_array, self.shift, np.nan)
        indices = np.where(np.isnan(_mz_array))[0]
        imax = indices.max()
        # fill needs to happen at the front
        if imax + 2 < len(_mz_array):
            spacing = _mz_array[imax + 2] - _mz_array[imax + 1]
            fill = _mz_array[imax + 1] - np.arange(len(indices)) * spacing
        else:
            index_min = np.min(indices)
            spacing = _mz_array[index_min - 1] - _mz_array[index_min - 2]
            fill = _mz_array[index_min - 1] - np.arange(len(indices)) * spacing
        _mz_array[indices] = fill[::-1]
        return _mz_array, intensity_array


class PpmRecalibrate(FilterBase):
    """Recalibrate mass axis using ppm offsets."""

    def __init__(self, ppm: float, every_n: int = 100):
        self.ppm = ppm
        self.every_n = every_n

    def filter(self, mz_array: np.ndarray, intensity_array: np.ndarray) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Filter."""
        from koyo.spectrum import get_ppm_offsets

        return mz_array - get_ppm_offsets(mz_array, self.ppm, every_n=self.every_n), intensity_array


class MzResampling(FilterBase):
    """Resample spectrum to specified number of points."""

    def __init__(self, mz: np.ndarray):
        self.mz = mz

    def filter(self, mz_array: np.ndarray, intensity_array: np.ndarray) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Filter."""
        # from scipy.interpolate import interp1d
        return self.mz, np.interp(self.mz, mz_array, intensity_array)
        # func = interp1d(mz_array, intensity_array, fill_value=0, bounds_error=False)
        # return self.mz, func(self.mz)


class PpmPeakRecalibrate(FilterBase):
    """Recalibrate mass axis using ppm offsets."""

    def __init__(self, ppm: float):
        self.ppm = ppm

    def filter(self, mz_array: np.ndarray, intensity_array: np.ndarray) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Filter."""
        from koyo.spectrum import get_window_for_ppm

        offsets = np.asarray([get_window_for_ppm(p, self.ppm) for p in mz_array])
        return mz_array - offsets, intensity_array


def transform(
    mz_array: np.ndarray, intensity_array: np.ndarray, filters: ty.Optional[ty.Iterable[str]] = None
) -> ty.Tuple[np.ndarray, np.ndarray]:
    """Apply a series of *filters* to the paired m/z and intensity arrays.

    The `filters` argument should be an iterable of either strings,
    callables, or instances of :class:`FilterBase`-derived classes.
    If they are strings, they must be registered names, as created by
    :func:`register`.

    Parameters
    ----------
    mz_array : np.ndarray
        The m/z array to filter
    intensity_array : np.ndarray
        The intensity array to filter
    filters : Iterable
        An Iterable of callables.

    Returns
    -------
    mz_array : np.ndarray[float64]:
        The m/z array after filtering
    intensity_array : np.ndarray[float64]:
        The intensity array after filtering
    """
    if filters is None:
        filters = []

    for _filter in filters:
        if isinstance(_filter, str):
            _filter = FILTER_REGISTER[_filter]
        mz_array, intensity_array = _filter(mz_array, intensity_array)
    return mz_array, intensity_array
