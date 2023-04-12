"""Generic utilities."""
import numpy as np
import typing as ty
from math import ceil
import unicodedata
import re


__all__ = [
    "get_value",
    "rescale",
    "rescale_value",
    "chunks",
    "zip_chunks",
    "sequential_chunks",
    "get_min_max",
    "check_image_orientation",
    "slugify",
    "get_kws",
    "format_count",
    "format_size",
]


def get_kws(func: ty.Callable, **kwargs) -> ty.Dict:
    """Get kwargs."""
    import inspect

    kws = {}
    args = inspect.getfullargspec(func).args

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
    """Convert bytes to nicer format"""
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


def isnumber(value):
    """Quick and easy way to check if input is a number

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
    """Check whether the value order is correct (min -> max)

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
    if not isnumber(value_min) or not isnumber(value_max):
        return value_min, value_max

    if value_max < value_min:
        value_max, value_min = value_min, value_max
    return value_min, value_max


def get_value(new_value, current_value):
    """Get value"""
    if new_value is None:
        return current_value
    return new_value


def rescale(
    values: ty.Union[np.ndarray, ty.List], new_min: float, new_max: float, dtype=None
) -> np.ndarray:
    """Rescale values from one range to another

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
    new_values = ((values - old_min) / (old_max - old_min)) * (
        new_max - new_min
    ) + new_min
    return new_values.astype(dtype)


def rescale_value(
    value: float,
    old_min: float,
    old_max: float,
    new_min: float = 0.0,
    new_max: float = 1.0,
    clip: bool = True,
) -> float:
    """Rescale value to new range

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
    new_value = ((value - old_min) / (old_max - old_min)) * (
        new_max - new_min
    ) + new_min
    return new_value


def chunks(item_list, n_items: int = 0, n_tasks: int = 0):
    """Yield successive n-sized chunks from `item_list`."""
    if n_items == 0 and n_tasks == 0:
        raise ValueError("You must specified either 'n_items' or 'n_tasks'.")
    if n_tasks:
        n_items = ceil(len(item_list) / n_tasks)
    for i in range(0, len(item_list), n_items):
        yield item_list[i : i + n_items]


def zip_chunks(item_list, n_items: int, *items):
    """Yield successive n-sized chunks from `item_list`"""
    yield from zip(chunks(item_list, n_items), *[chunks(x, n_items) for x in items])


def sequential_chunks(item_list, n_items: int):
    """Create multiple lists of chunks that will be subsequent in nature. For instance if the `item_list` has
    the following values: [0, 1, 2, 3] and `n_items=2`, it will generate two lists of [0, 2] and [1, 3]. This can
    be helpful when doing tasks in chunks but want to see results of subsequent chunks."""
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
    """Get the minimum and maximum value of an array"""
    return [np.min(values), np.max(values)]


def check_image_orientation(array):
    """Transpose image if the primary size is larger than the secondary size in order to improve
    images

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
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^.=\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")
