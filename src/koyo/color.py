"""Module with color functions."""

from __future__ import annotations

import colorsys
import random
import typing as ty
import warnings
from ast import literal_eval

import numpy as np

if ty.TYPE_CHECKING:
    from matplotlib.colors import Colormap as MplColormap


# All parsable input color types that a user can provide
ColorType = ty.Union[list, tuple, np.ndarray, str]


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


def hex_to_rgb_1(hex_str, decimals: int = 3, with_alpha: bool = False):
    """Convert hex color to rgb color in 1-scale."""
    hex_color = hex_str.lstrip("#")
    hlen = len(hex_color)
    rgb = tuple(int(hex_color[i : i + int(hlen / 3)], 16) for i in range(0, int(hlen), int(hlen / 3)))
    rgb = [np.round(rgb[0] / 255.0, decimals), np.round(rgb[1] / 255.0, decimals), np.round(rgb[2] / 255.0, decimals)]
    if with_alpha:
        return [*rgb, 1.0]
    return rgb


def hex_to_rgb_255(hex_str):
    """Convert hex color to rgb color in 255-scale."""
    hex_color = hex_str.lstrip("#")
    hlen = len(hex_color)
    rgb = [int(hex_color[i : i + int(hlen / 3)], 16) for i in range(0, int(hlen), int(hlen / 3))]
    return rgb


def rgb_to_hex(colors, multiplier: int = 255) -> str:
    """Convert list/tuple of colors to hex."""
    return f"#{int(colors[0] * multiplier):02x}{int(colors[1] * multiplier):02x}{int(colors[2] * multiplier):02x}"


def hex_to_rgb(hex_str, decimals=3, alpha: float | None = None):
    """Convert hex color to numpy array."""
    hex_color = hex_str.lstrip("#")
    hex_len = len(hex_color)
    rgb = [int(hex_color[i : i + int(hex_len / 3)], 16) for i in range(0, int(hex_len), int(hex_len / 3))]
    if alpha is not None:
        if alpha == 1:
            warnings.warn(
                "The provided alpha value is equal to 1 - this function accepts values in 0-255 range.", stacklevel=2
            )
        rgb.append(alpha)
    return np.round(np.asarray(rgb) / 255, decimals)


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
        additional_count = n - len(colors) + 50
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


def colormap_to_hex(colormap: MplColormap) -> list[str]:
    """Convert mpl colormap to hex."""
    return [rgb_to_hex(colormap(i)) for i in range(colormap.N)]


def get_colors_from_colormap(colormap: str, n_colors: int, is_reversed: bool = False) -> list[str]:
    """Get list of colors from colormap."""
    import matplotlib.cm

    if is_reversed and not colormap.endswith("_r"):
        colormap += "_r"

    return colormap_to_hex(matplotlib.cm.get_cmap(colormap, n_colors))


def find_text_color(base_color, dark_color="black", light_color="white", coef_choice=0):
    """
    Takes a background color and returns the appropriate light or dark text color.
    Users can specify the dark and light text color, or accept the defaults of 'black' and 'white'
    base_color: The color of the background. This must be
        specified in RGBA with values between 0 and 1 (note, this is the default
        return value format of a call to base_color = cmap(number) to get the
        color corresponding to a desired number). Note, the value of `A` in RGBA
        is not considered in determining light/dark.
    dark_color: Any valid matplotlib color value.
        Function will return this value if the text should be colored dark
    light_color: Any valid matplotlib color value.
        Function will return this value if thet text should be colored light.
    coef_choice: slightly different approaches to calculating brightness. Currently two options in
        a list, user can enter 0 or 1 as list index. 0 is default.
    """
    # Coefficients:
    # option 0: http://www.nbdtech.com/Blog/archive/2008/04/27/Calculating-the-Perceived-Brightness-of-a-Color.aspx
    # option 1: http://stackoverflow.com/questions/596216/formula-to-determine-brightness-of-rgb-color
    coef_options = [
        np.array((0.241, 0.691, 0.068, 0)),
        np.array((0.299, 0.587, 0.114, 0)),
    ]

    coefs = coef_options[coef_choice]
    rgb = np.array(base_color) * 255
    brightness = np.sqrt(np.dot(coefs, rgb**2))

    # Threshold from option 0 link; determined by trial and error.
    # base is light
    if brightness > 130:
        return dark_color
    return light_color


def make_listed_colormap(colors: list[str], is_vispy: bool = False) -> MplColormap:
    """Make listed colormap."""
    from matplotlib.colors import LinearSegmentedColormap
    from vispy.color.colormap import Colormap

    if is_vispy:
        colors.insert(0, "#FFFFFF")

    colormap = LinearSegmentedColormap.from_list("colormap", colors, len(colors))
    if is_vispy:
        mpl_colors = colormap(np.linspace(0, 1, len(colors)))
        mpl_colors[0][-1] = 0  # first color is white with alpha=0
        return Colormap(mpl_colors)
    return colormap


def _check_color_dim(val):
    """Ensures input is Nx4.

    Parameters
    ----------
    val : np.ndarray
        A color array of possibly less than 4 columns

    Returns
    -------
    val : np.ndarray
        A four columns version of the input array. If the original array
        was a missing the fourth channel, it's added as 1.0 values.
    """
    val = np.atleast_2d(val)
    if val.shape[1] not in (3, 4):
        strval = str(val)
        if len(strval) > 100:
            strval = strval[:97] + "..."
        raise RuntimeError(f"Value must have second dimension of size 3 or 4. Got `{strval}`, shape={val.shape}")

    if val.shape[1] == 3:
        val = np.column_stack([val, np.float32(1.0)])
    return val


def rgbs_to_hex(rgbs: ty.Sequence) -> np.ndarray:
    """Convert RGB to hex quadruplet.

    Taken from vispy with slight modifications.

    Parameters
    ----------
    rgbs : Sequence
        A list-like container of colors in RGBA format with values
        between [0, 1]

    Returns
    -------
    arr : np.ndarray
        An array of the hex representation of the input colors

    """
    rgbs = _check_color_dim(rgbs)
    return np.array(
        [f"#{'%02x' * 4}" % tuple((255 * rgb).astype(np.uint8)) for rgb in rgbs],
        "|U9",
    )


def transform_color(color: ColorType) -> np.ndarray:
    """Transform color."""
    if isinstance(color, str):
        if color.startswith("#"):
            return np.atleast_2d(hex_to_rgb_1(color, with_alpha=True))
        if color.startswith("0x"):
            return np.atleast_2d(hex_to_rgb_1(color[2:], with_alpha=True))
        if color.startswith("rgb"):
            return np.atleast_2d(literal_eval(color))
        if color.startswith("rgba"):
            return np.atleast_2d(literal_eval(color))
        raise ValueError(f"Invalid color string: {color}")
    if isinstance(color, (list, tuple)):
        color = np.atleast_2d(color)
        if color.max() > 1:
            color = color / 255.0
        if color.shape[1] == 4:
            return color
        if color.shape[1] == 3:
            return np.concatenate((color, np.ones((color.shape[0], 1))), axis=1)
        raise ValueError(f"Invalid color list/tuple: {color}")
    if isinstance(color, np.ndarray):
        if color.ndim == 1:
            if color.shape[0] == 3:
                return np.concatenate((color, [1]))
            if color.shape[0] == 4:
                return color
        raise ValueError(f"Invalid color array: {color}")
