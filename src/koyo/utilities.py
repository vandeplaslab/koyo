"""Generic utilities."""
import re
import typing as ty
import unicodedata
from collections.abc import Iterable
from math import ceil

import numba as nb
import numpy as np

from koyo.typing import SimpleArrayLike

def reraise_exception_if_debug(exc, message: str, env_key: str = "DEV_MODE"):
    """Reraise exception if debug mode is enabled and jump into the debugger."""
    if os.environ.get(env_key, "0") == "1":
        raise exc
    logger.exception(message)

def pluralize(word: str, n: int, with_e: bool = False) -> str:
    """Give plural form to a word."""
    if word in ["is", "are"]:
        return "is" if n == 1 else "are"
    if word in ["was", "were"]:
        return "was" if n == 1 else "were"
    elif word in ["has", "have"]:
        return "has" if n == 1 else "have"
    extra = "s" if not with_e else "es"
    return word if n == 1 else word + extra

def is_valid_python_name(name):
    from keyword import iskeyword
    return name.isidentifier() and not iskeyword(name)

def is_between(value: float, lower: float, upper: float, inclusive: bool = True) -> bool:
    """Check if value is between lower and upper."""
    if inclusive:
        return lower <= value <= upper
    return lower < value < upper


def flatten_nested_list(list_of_lists: ty.List[ty.List]) -> ty.List:
    """Flatten nested list of lists."""
    return [item for sublist in list_of_lists for item in sublist]


def get_list_difference(li1: ty.List, li2: ty.List):
    """Get difference between two lists."""
    # get difference between two lists while keeping the order
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif


def find_nearest_index(data: SimpleArrayLike, value: ty.Union[int, float, np.ndarray, Iterable]):
    """Find nearest index of asked value.

    Parameters
    ----------
    data : np.array
        input array (e.g. m/z values)
    value : Union[int, float, np.ndarray]
        asked value

    Returns
    -------
    index :
        index value
    """
    data = np.asarray(data)
    if isinstance(value, Iterable):
        return np.asarray([np.argmin(np.abs(data - _value)) for _value in value], dtype=np.int64)
    return np.argmin(np.abs(data - value))


@nb.njit()
def find_nearest_index_array(data: SimpleArrayLike, value: ty.Union[np.ndarray, ty.Iterable]) -> np.ndarray:
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
def find_nearest_index_single(data: SimpleArrayLike, value: ty.Union[int, float]):
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
    return np.argmin(np.abs(data - value))


def find_nearest_value_single(data: SimpleArrayLike, value: ty.Union[int, float]) -> ty.Union[int, float]:
    """Find nearest value."""
    data = np.asarray(data)
    idx = find_nearest_index_single(data, value)
    return data[idx]


def find_nearest_index_batch(array: SimpleArrayLike, values: SimpleArrayLike) -> np.ndarray:
    """Find nearest index."""
    # make sure array is a numpy array
    array = np.asarray(array)
    values = np.asarray(values)
    if not array.size or not values.size:
        return np.array([])

    # get insert positions
    indices = np.searchsorted(array, values, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = (indices == len(array)) | (
        np.fabs(values - array[np.maximum(indices - 1, 0)])
        < np.fabs(values - array[np.minimum(indices, len(array) - 1)])
    )
    indices[prev_idx_is_less] -= 1
    return indices


def find_nearest_value(data: ty.Iterable, value: ty.Union[int, float, np.ndarray, Iterable]):
    """Find nearest value."""
    data = np.asarray(data)
    idx = find_nearest_index(data, value)
    return data[idx]


def get_kws(func: ty.Callable, **kwargs) -> ty.Dict:
    """Get kwargs."""
    import inspect

    args = inspect.getfullargspec(func).args

    kws = {}
    for kw in args:
        if kw in kwargs:
            kws[kw] = kwargs[kw]
    return kws


def format_count(value):
    """Format count."""
    if value < 1e3:
        return f"{value:.0f}"
    elif value < 1e6:
        return f"{value/1e3:.1f}K"
    elif value < 1e9:
        return f"{value/1e6:.1f}M"
    return f"{value/1e9:.1f}B"


def format_size(size: int) -> str:
    """Convert bytes to nicer format."""
    if size < 2**10:
        return "%s" % size
    elif size < 2**20:
        return "%.1fK" % (size / float(2**10))
    elif size < 2**30:
        return "%.1fM" % (size / float(2**20))
    elif size < 2**40:
        return "%.1fG" % (size / float(2**30))
    elif size < 2**50:
        return "%.1fT" % (size / float(2**40))
    return "%.1fP" % (size / float(2**50))


def is_number(value):
    """Quick and easy way to check if input is a number.

    Parameters
    ----------
    value : any
        input value

    Returns
    -------
    bool
        returns True/False, depending if a value is a number
    """
    return isinstance(value, (int, float, complex))


def check_value_order(value_min, value_max):
    """Check whether the value order is correct (min -> max).

    Parameters
    ----------
    value_min : int or float or complex
        presumed minimal value
    value_max : int or float or complex
        presumed maximal value

    Returns
    -------
    value_min : int or float or complex
        true minimal value
    value_max : int or float or complex
        true maximal value
    """
    if not is_number(value_min) or not is_number(value_max):
        return value_min, value_max

    if value_max < value_min:
        value_max, value_min = value_min, value_max
    return value_min, value_max


def get_value(new_value, current_value):
    """Get value."""
    if new_value is None:
        return current_value
    return new_value


def rescale(values: ty.Union[np.ndarray, ty.List], new_min: float, new_max: float, dtype=None) -> np.ndarray:
    """Rescale values from one range to another.

    Parameters
    ----------
    values : Union[np.ndarray, List]
        input range
    new_min : float
        new minimum value
    new_max : float
        new maximum value
    dtype :
        data type

    Returns
    -------
    new_values : np.ndarray
        rescaled range
    """
    values = np.asarray(values)
    if dtype is None:
        dtype = values.dtype
    old_min, old_max = get_min_max(values)
    new_values = ((values - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    return new_values.astype(dtype)


def rescale_value(
    value: float,
    old_min: float,
    old_max: float,
    new_min: float = 0.0,
    new_max: float = 1.0,
    clip: bool = True,
) -> float:
    """Rescale value to new range.

    Parameters
    ----------
    value : float
        value that needs to be rescaled
    old_min : float
        minimum value of the original range
    old_max : float
        maximum value of the original range
    new_min : float
        minimum value of the new range
    new_max : float
        maximum value of the new range
    clip : bool
        if True, the new value will be clipped to fit inside of the new maximum range
    """
    if clip:
        if value < old_min:
            value = old_min
        if value > old_max:
            value = old_max
    return ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min


def chunks(item_list, n_items: int = 0, n_tasks: int = 0):
    """Yield successive n-sized chunks from `item_list`."""
    if n_items == 0 and n_tasks == 0:
        raise ValueError("You must specified either 'n_items' or 'n_tasks'.")
    if n_tasks:
        n_items = ceil(len(item_list) / n_tasks)
    for i in range(0, len(item_list), n_items):
        yield item_list[i : i + n_items]


def zip_chunks(item_list, n_items: int, *items):
    """Yield successive n-sized chunks from `item_list`."""
    yield from zip(chunks(item_list, n_items), *[chunks(x, n_items) for x in items])


def sequential_chunks(item_list, n_items: int):
    """Create multiple lists of chunks that will be subsequent in nature.

    For instance if the `item_list` has
    the following values: [0, 1, 2, 3] and `n_items=2`, it will generate two lists of [0, 2] and [1, 3]. This can
    be helpful when doing tasks in chunks but want to see results of subsequent chunks.
    """
    n = len(item_list)
    n_pots = ceil(n / n_items)
    pots = [[] for _ in range(n_pots)]
    pot, current = 0, 0
    while current < n:
        pots[pot].append(item_list[current])
        pot += 1
        if pot >= n_pots:
            pot = 0
        current += 1
    yield from pots


def get_min_max(values):
    """Get the minimum and maximum value of an array."""
    return [np.min(values), np.max(values)]


def check_image_orientation(array):
    """Transpose image if the primary size is larger than the secondary size in order to improve images.

    Parameters
    ----------
    array: np.array
        2D heatmap

    Returns
    -------
    zvals: np.array
        (potentially) transposed 2D heatmap
    """
    shape = array.shape
    if len(shape) == 3:
        return np.swapaxes(array, 1, 2) if shape[1] > shape[2] else array
    return array.T if shape[0] > shape[1] else array


def slugify(value, allow_unicode=False):
    """Convert to ASCII if 'allow_unicode' is False.

    Convert spaces or repeated dashes to single dashes. Remove characters that aren't alphanumerics, underscores,
    or hyphens. Convert to lowercase. Also strip leading and trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^.=\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def get_array_mask(array: np.ndarray, min_val: float, max_val: float):
    """Return mask for array."""
    return np.logical_and(array >= min_val, array <= max_val)


def get_array_window(array: np.ndarray, min_val: float, max_val: float, *arrays):
    """Get narrower view of array based on upper and lower limits.

    The first array is the one that is used to create mask.
    """
    mask = np.logical_and(array >= min_val, array <= max_val)
    _arrays = [array[mask]]
    for _array in arrays:
        if _array.shape[0] != mask.shape[0]:
            raise ValueError("Incorrect shape of the input arrays")
        _arrays.append(_array[mask])
    return _arrays


def split_array(array: np.ndarray, chunk_size: int, axis=0):
    """Split array according to chunk size."""
    n = array.shape[axis]
    while True:
        if n <= chunk_size:
            yield array
            break
        else:
            if axis == 0:
                yield array.iloc[:chunk_size]
                array = array.iloc[chunk_size:]
            else:
                yield array[:, :chunk_size]
                array = array[:, chunk_size:]
            n -= chunk_size


def _remove_duplicates_from_dict(data):
    """Remove duplicates from list of dictionaries."""
    # list of dictionaries
    if isinstance(data, list):
        # check if list of dictionaries
        if all(isinstance(d, dict) for d in data):
            return [dict(t) for t in {tuple(d.items()) for d in data}]
    return data


class Cycler:
    """Cycling class similar to itertools.cycle with the addition of `previous` functionality."""

    def __init__(self, c):
        self._c = c
        self._index = -1

    def __len__(self):
        return len(self._c)

    def __next__(self):
        self._index += 1
        if self._index >= len(self._c):
            self._index = 0
        return self._c[self._index]

    def next(self):
        """Go forward."""
        return self.__next__()

    def previous(self):
        """Go backwards."""
        self._index -= 1
        if self._index < 0:
            self._index = len(self._c) - 1
        return self._c[self._index]

    def current(self):
        """Get current index."""
        return self._c[self._index]

    def set_current(self, index: int):
        """Set current index."""
        self._index = index


def difference_matrix(a: np.ndarray) -> np.ndarray:
    """Difference matrix.

    Compute difference between each element in a 1d array.
    """
    a = np.asarray(a)
    assert a.ndim == 1, "Input must be 1d array."

    x = np.reshape(a, (len(a), 1))
    return x - x.transpose()


def running_as_pyinstaller_app() -> bool:
    """Infer whether we are running pyinstaller bundle."""
    import sys

    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")
