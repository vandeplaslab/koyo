"""Module with color functions."""

from __future__ import annotations

import colorsys
import random
from ast import literal_eval

import numpy as np


def mpl_to_rgba_255(color: str | tuple | list) -> tuple[int, int, int, int]:
    """Convert color from matplotlib to RGBA 255-scale."""
    from matplotlib.colors import to_rgba_array

    return tuple(to_rgba_array(color)[0].astype(np.uint8) * 255)


def rgb_1_to_hex(color: tuple | list):
    """Convert RGB to Hex."""
    color = list(color)[0:3]
    return "#" + "".join([f"{int(c * 255):02x}" for c in color])


def rgb_255_to_1(color: tuple | list, decimals: int = 3):
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


def get_random_hex_color() -> str:
    """Return random hex color."""
    return f"#{random.randint(0, 0xFFFFFF):06x}"


def get_next_color(n: int, other_colors: list[str] | None = None) -> str:
    """
    Get the next color based on the index, avoiding duplicates in `other_colors`.

    Parameters
    ----------
        n (int): The index of the desired color.
        other_colors (list[str] | None): A list of colors to avoid, default is None.

    Returns
    -------
        str: The next distinct color in lowercase hex format.
    """
    if other_colors is None:
        other_colors = []

    # Predefined list of colors
    colors = [
        "#ff0000",
        "#00ff00",
        "#0000ff",
        "#ffff00",
        "#ff00ff",
        "#00ffff",
        "#ff4500",
        "#ff69b4",
        "#1e90ff",
        "#32cd32",
        "#ffd700",
        "#40e0d0",
        "#ff6347",
        "#7b68ee",
        "#adff2f",
        "#87ceeb",
    ]

    # Extend colors if `n` exceeds predefined list
    if n >= len(colors):
        additional_count = n - len(colors) + 1
        hue_step = 1 / (additional_count + 1)
        for i in range(additional_count):
            h = (i + 1) * hue_step
            s, l = 0.7, 0.5
            r, g, b = colorsys.hls_to_rgb(h, l, s)
            colors.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")

    # Get the nth color and check for duplicates
    color = colors[n].lower()
    if color in (c.lower() for c in other_colors):
        return get_next_color(n + 1, other_colors=other_colors)
    return color


def generate_distinct_colors(starting_colors: list[str], n_colors: int) -> list[str]:
    """
    Generate a list of n_colors distinct colors starting from the given list.

    Parameters
    ----------
        starting_colors (List[str]): List of predefined colors in hex format.
        n_colors (int): The total number of distinct colors required.

    Returns
    -------
        List[str]: List of n_colors in hex format.
    """

    def hsl_to_hex(h, s, l):
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    # Convert predefined colors to HSL and collect them
    starting_hsl = []
    for hex_color in starting_colors:
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        starting_hsl.append((h, s, l))

    # Generate new colors if needed
    additional_colors_needed = max(0, n_colors - len(starting_colors))
    if additional_colors_needed > 0:
        step = 1 / (additional_colors_needed + 1)
        for i in range(additional_colors_needed):
            h = (i + 1) * step  # Spread hues evenly
            s, l = 0.7, 0.5  # Fixed saturation and lightness for distinctiveness
            starting_hsl.append((h, s, l))

    # Convert all HSL colors back to hex
    result_colors = starting_colors[:]
    result_colors += [hsl_to_hex(h, s, l) for h, s, l in starting_hsl[len(starting_colors) :]]
    return result_colors[:n_colors]
