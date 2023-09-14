"""Module with color functions."""
import random
import typing as ty
from ast import literal_eval

import numpy as np


def rgb_255_to_1(color: ty.Union[ty.Tuple, ty.List], decimals: int = 3):
    """Convert color that is in RGB (255-scale) to RGB (1-scale).

    Parameters
    ----------
    color : list or np.array
        color
    decimals : int, optional
        number of decimal spaces

    Returns
    -------
    rgbList : list
        color in 1-scale
    """
    # make sure color is an rgb format and not string format
    try:
        color = literal_eval(color)
    except Exception:
        color = color

    color = np.array(color)
    if color.shape[0] not in [3, 4]:
        raise ValueError("Color must have shape (3,) or (4,)")
    if color.max() > 255:
        raise ValueError("Color or transparency cannot be larger than 255")
    if color.min() < 0:
        raise ValueError("Color or transparency cannots be larger than 0")

    color = (color / 255).round(decimals)

    return list(color)


def hex_to_rgb_1(hex_str, decimals=3):
    """Convert hex color to rgb color in 1-scale."""
    hex_color = hex_str.lstrip("#")
    hlen = len(hex_color)
    rgb = tuple(int(hex_color[i : i + int(hlen / 3)], 16) for i in range(0, int(hlen), int(hlen / 3)))
    return [np.round(rgb[0] / 255.0, decimals), np.round(rgb[1] / 255.0, decimals), np.round(rgb[2] / 255.0, decimals)]


def hex_to_rgb_255(hex_str):
    """Convert hex color to rgb color in 255-scale."""
    hex_color = hex_str.lstrip("#")
    hlen = len(hex_color)
    rgb = [int(hex_color[i : i + int(hlen / 3)], 16) for i in range(0, int(hlen), int(hlen / 3))]

    return rgb


def get_random_hex_color():
    """Return random hex color."""
    return "#%06x" % random.randint(0, 0xFFFFFF)
