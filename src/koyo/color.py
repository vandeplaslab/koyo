"""Module with color conversion and generation utilities."""

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


def mpl_to_rgba_255(color: ty.Union[str, tuple, list]) -> tuple:
    """Convert a matplotlib color to an RGBA tuple in 255-scale.

    Parameters
    ----------
    color : str or tuple or list
        Any matplotlib-compatible color specification.

    Returns
    -------
    tuple
        A 4-element tuple of integers ``(R, G, B, A)`` each in ``[0, 255]``.
    """
    from matplotlib.colors import to_rgba_array

    return tuple((to_rgba_array(color)[0] * 255).astype(np.uint8))


def rgb_1_to_hex(color: ty.Union[tuple, list]) -> str:
    """Convert an RGB color in 1-scale to a hex string.

    Only the first three channels (R, G, B) are used; an alpha channel is
    ignored if present.

    Parameters
    ----------
    color : tuple or list
        RGB or RGBA color values in the range ``[0, 1]``.

    Returns
    -------
    str
        Hex color string of the form ``'#rrggbb'``.
    """
    rgb = list(color)[:3]
    return "#" + "".join(f"{int(c * 255):02x}" for c in rgb)


def rgb_255_to_1(color: ty.Union[tuple, list, str], decimals: int = 3) -> ty.List[float]:
    """Convert a color from 255-scale RGB to 1-scale RGB.

    Parameters
    ----------
    color : tuple, list, or str
        Color in 255-scale. If a string representation of a list or tuple is
        provided (e.g. ``'(255, 0, 0)'``), it is parsed automatically.
    decimals : int, optional
        Number of decimal places to round each channel to. Default is ``3``.

    Returns
    -------
    list of float
        Color channels normalized to ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``color`` does not have exactly 3 or 4 channels, if any value
        exceeds 255, or if any value is negative.
    """
    # Parse string representation if needed
    if isinstance(color, str):
        try:
            color = literal_eval(color)
        except Exception:
            pass

    color = np.array(color)
    if color.shape[0] not in (3, 4):
        raise ValueError("Color must have shape (3,) or (4,).")
    if color.max() > 255:
        raise ValueError("Color channel values cannot be larger than 255.")
    if color.min() < 0:
        raise ValueError("Color channel values cannot be smaller than 0.")

    return list((color / 255).round(decimals))


def hex_to_rgb_1(
    hex_str: str,
    decimals: int = 3,
    with_alpha: bool = False,
) -> ty.List[float]:
    """Convert a hex color string to RGB values in 1-scale.

    Parameters
    ----------
    hex_str : str
        Hex color string, with or without a leading ``'#'``.
    decimals : int, optional
        Number of decimal places to round each channel to. Default is ``3``.
    with_alpha : bool, optional
        If ``True``, append ``1.0`` as the alpha channel. Default is ``False``.

    Returns
    -------
    list of float
        RGB (or RGBA) channel values in ``[0, 1]``.
    """
    hex_color = hex_str.lstrip("#")
    channel_width = len(hex_color) // 3
    rgb = [
        round(int(hex_color[i : i + channel_width], 16) / 255.0, decimals)
        for i in range(0, len(hex_color), channel_width)
    ]
    if with_alpha:
        return [*rgb, 1.0]
    return rgb


def hex_to_rgb_255(hex_str: str) -> ty.List[int]:
    """Convert a hex color string to RGB values in 255-scale.

    Parameters
    ----------
    hex_str : str
        Hex color string, with or without a leading ``'#'``.

    Returns
    -------
    list of int
        RGB channel values, each in ``[0, 255]``.
    """
    hex_color = hex_str.lstrip("#")
    channel_width = len(hex_color) // 3
    return [int(hex_color[i : i + channel_width], 16) for i in range(0, len(hex_color), channel_width)]


def rgb_to_hex(colors: ty.Union[tuple, list], multiplier: int = 255) -> str:
    """Convert an RGB or RGBA color to a hex string.

    Parameters
    ----------
    colors : tuple or list
        Color values. Only the first three channels are used.
    multiplier : int, optional
        Scale factor applied to each channel before conversion. Use ``255``
        when channels are in ``[0, 1]``, or ``1`` when already in
        ``[0, 255]``. Default is ``255``.

    Returns
    -------
    str
        Hex color string of the form ``'#rrggbb'``.
    """
    return f"#{int(colors[0] * multiplier):02x}{int(colors[1] * multiplier):02x}{int(colors[2] * multiplier):02x}"


def hex_to_rgb(
    hex_str: str,
    decimals: int = 3,
    alpha: ty.Optional[float] = None,
) -> np.ndarray:
    """Convert a hex color string to a numpy array in 1-scale.

    Parameters
    ----------
    hex_str : str
        Hex color string, with or without a leading ``'#'``.
    decimals : int, optional
        Number of decimal places to round each channel to. Default is ``3``.
    alpha : float, optional
        If provided, append this value as the alpha channel. Expected range
        is ``[0, 255]``. A value of ``1`` triggers a warning because it is
        likely a mistake (1-scale vs 255-scale confusion).

    Returns
    -------
    np.ndarray
        1-D array of RGB (or RGBA) channel values in ``[0, 1]``.

    Warns
    -----
    UserWarning
        If ``alpha`` equals ``1``, since this function expects alpha in
        the ``[0, 255]`` range.
    """
    hex_color = hex_str.lstrip("#")
    hex_len = len(hex_color)
    rgb = [int(hex_color[i : i + hex_len // 3], 16) for i in range(0, hex_len, hex_len // 3)]
    if alpha is not None:
        if alpha == 1:
            warnings.warn(
                "The provided alpha value is equal to 1 — this function accepts values in the 0-255 range.",
                stacklevel=2,
            )
        rgb.append(alpha)
    return np.round(np.asarray(rgb) / 255, decimals)


def get_random_hex_color() -> str:
    """Return a random hex color string.

    Returns
    -------
    str
        Hex color string of the form ``'#rrggbb'``.
    """
    return f"#{random.randint(0, 0xFFFFFF):06x}"


def get_next_color(n: int, other_colors: ty.Optional[ty.List[str]] = None) -> str:
    """Return the nth distinct color, avoiding any colors in ``other_colors``.

    Selects from a predefined palette of 16 colors. If ``n`` exceeds the
    palette length, additional colors are generated by distributing hues
    evenly in HLS space.

    Parameters
    ----------
    n : int
        Index of the desired color in the palette.
    other_colors : list of str, optional
        Hex color strings to avoid. If the selected color matches any entry,
        the function recurses with ``n + 1``. Default is ``None``.

    Returns
    -------
    str
        Hex color string in lowercase.
    """
    if other_colors is None:
        other_colors = []

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

    if n >= len(colors):
        additional_count = n - len(colors) + 50
        hue_step = 1.0 / (additional_count + 1)
        saturation, lightness = 0.7, 0.5
        for i in range(additional_count):
            hue = (i + 1) * hue_step
            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            colors.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")

    color = colors[n].lower()
    if color in {c.lower() for c in other_colors}:
        return get_next_color(n + 1, other_colors=other_colors)
    return color


def generate_distinct_colors(starting_colors: ty.List[str], n_colors: int) -> ty.List[str]:
    """Generate a list of distinct hex colors, extending a predefined palette as needed.

    Converts the starting colors to HLS space, then fills any remaining
    slots by distributing new hues evenly across the color wheel at fixed
    saturation and lightness.

    Parameters
    ----------
    starting_colors : list of str
        Predefined colors in hex format used as the beginning of the palette.
    n_colors : int
        Total number of distinct colors required.

    Returns
    -------
    list of str
        Hex color strings of length ``n_colors``.
    """

    def hls_to_hex(hue: float, saturation: float, lightness: float) -> str:
        """Convert HLS values to a hex color string.

        Parameters
        ----------
        hue : float
            Hue in ``[0, 1]``.
        saturation : float
            Saturation in ``[0, 1]``.
        lightness : float
            Lightness in ``[0, 1]``.

        Returns
        -------
        str
            Hex color string of the form ``'#rrggbb'``.
        """
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    # Convert predefined colors to HLS
    starting_hls: ty.List[ty.Tuple[float, float, float]] = []
    for hex_color in starting_colors:
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
        starting_hls.append((hue, saturation, lightness))

    # Generate additional colors if the palette is too short
    additional_needed = max(0, n_colors - len(starting_colors))
    if additional_needed > 0:
        hue_step = 1.0 / (additional_needed + 1)
        saturation, lightness = 0.7, 0.5
        for i in range(additional_needed):
            hue = (i + 1) * hue_step
            starting_hls.append((hue, saturation, lightness))

    result_colors = starting_colors[:]
    result_colors += [
        hls_to_hex(hue, saturation, lightness)
        for hue, saturation, lightness in starting_hls[len(starting_colors) :]
    ]
    return result_colors[:n_colors]


def colormap_to_hex(colormap: MplColormap) -> ty.List[str]:
    """Convert a matplotlib colormap to a list of hex color strings.

    Parameters
    ----------
    colormap : MplColormap
        A matplotlib colormap with a defined number of colors (``colormap.N``).

    Returns
    -------
    list of str
        Hex color strings, one per color in the colormap.
    """
    return [rgb_to_hex(colormap(i)) for i in range(colormap.N)]


def get_colors_from_colormap(
    colormap: str,
    n_colors: int,
    is_reversed: bool = False,
) -> ty.List[str]:
    """Sample ``n_colors`` hex colors from a named matplotlib colormap.

    Parameters
    ----------
    colormap : str
        Name of a matplotlib colormap (e.g. ``'viridis'``, ``'plasma'``).
    n_colors : int
        Number of colors to sample from the colormap.
    is_reversed : bool, optional
        If ``True`` and the colormap name does not already end in ``'_r'``,
        the reversed variant is used. Default is ``False``.

    Returns
    -------
    list of str
        Hex color strings of length ``n_colors``.
    """
    import matplotlib.cm

    if is_reversed and not colormap.endswith("_r"):
        colormap += "_r"

    # NOTE: matplotlib.cm.get_cmap is deprecated in Matplotlib 3.7+;
    # migrate to matplotlib.colormaps[name] when the minimum version allows.
    return colormap_to_hex(matplotlib.cm.get_cmap(colormap, n_colors))


def find_text_color(
    base_color: ty.Union[tuple, list, np.ndarray],
    dark_color: ty.Any = "black",
    light_color: ty.Any = "white",
    coef_choice: int = 0,
) -> ty.Any:
    """Return the appropriate text color (dark or light) for a given background.

    Perceived brightness is computed from the RGB channels of ``base_color``
    using one of two standard coefficient sets. The alpha channel is ignored.

    Parameters
    ----------
    base_color : tuple, list, or np.ndarray
        Background color in RGBA format with values in ``[0, 1]``, as
        returned by ``colormap(value)``.
    dark_color : any matplotlib color, optional
        Color returned when the background is light. Default is ``'black'``.
    light_color : any matplotlib color, optional
        Color returned when the background is dark. Default is ``'white'``.
    coef_choice : int, optional
        Selects the brightness coefficient set:

        - ``0`` — coefficients from NBD Tech (default):
          ``(0.241, 0.691, 0.068)``
        - ``1`` — coefficients from the W3C formula:
          ``(0.299, 0.587, 0.114)``

    Returns
    -------
    any matplotlib color
        Either ``dark_color`` or ``light_color`` depending on the perceived
        brightness of ``base_color``.

    Notes
    -----
    References for the coefficient options:

    - Option 0: http://www.nbdtech.com/Blog/archive/2008/04/27/Calculating-the-Perceived-Brightness-of-a-Color.aspx
    - Option 1: http://stackoverflow.com/questions/596216/formula-to-determine-brightness-of-rgb-color

    The brightness threshold of 130 (on a 0-255 scale) was determined
    empirically for option 0.
    """
    coef_options = [
        np.array((0.241, 0.691, 0.068, 0)),
        np.array((0.299, 0.587, 0.114, 0)),
    ]

    coefs = coef_options[coef_choice]
    rgb_255 = np.array(base_color) * 255
    brightness = np.sqrt(np.dot(coefs, rgb_255**2))

    if brightness > 130:
        return dark_color
    return light_color


def make_listed_colormap(
    colors: ty.List[str],
    is_vispy: bool = False,
) -> MplColormap:
    """Build a listed colormap from a list of hex colors.

    When ``is_vispy`` is ``True``, a white color with zero alpha is prepended
    to the palette so that the first colormap entry is fully transparent,
    and a vispy ``Colormap`` is returned instead of a matplotlib one.

    Parameters
    ----------
    colors : list of str
        Hex color strings used to build the colormap.
    is_vispy : bool, optional
        If ``True``, return a ``vispy.color.colormap.Colormap`` with a
        transparent first entry. Default is ``False``.

    Returns
    -------
    MplColormap or vispy.color.colormap.Colormap
        A colormap built from the provided colors.
    """
    from matplotlib.colors import LinearSegmentedColormap
    from vispy.color.colormap import Colormap

    if is_vispy:
        colors.insert(0, "#FFFFFF")

    colormap = LinearSegmentedColormap.from_list("colormap", colors, len(colors))
    if is_vispy:
        mpl_colors = colormap(np.linspace(0, 1, len(colors)))
        mpl_colors[0][-1] = 0  # first entry: white, fully transparent
        return Colormap(mpl_colors)
    return colormap


def _check_color_dim(val: np.ndarray) -> np.ndarray:
    """Ensure a color array has shape ``(N, 4)``.

    Parameters
    ----------
    val : np.ndarray
        Color array of shape ``(N, 3)`` or ``(N, 4)``.

    Returns
    -------
    np.ndarray
        Color array of shape ``(N, 4)``. If the input had only 3 columns,
        a fourth column of ``1.0`` (fully opaque) is appended.

    Raises
    ------
    RuntimeError
        If ``val`` does not have 3 or 4 columns after calling
        ``np.atleast_2d``.
    """
    val = np.atleast_2d(val)
    if val.shape[1] not in (3, 4):
        display_val = str(val)
        if len(display_val) > 100:
            display_val = display_val[:97] + "..."
        raise RuntimeError(
            f"Value must have second dimension of size 3 or 4. Got `{display_val}`, shape={val.shape}"
        )

    if val.shape[1] == 3:
        val = np.column_stack([val, np.float32(1.0)])
    return val


def rgbs_to_hex(rgbs: ty.Sequence) -> np.ndarray:
    """Convert an array of RGBA colors in 1-scale to hex strings.

    Taken from vispy with slight modifications.

    Parameters
    ----------
    rgbs : Sequence
        List-like container of colors in RGBA format with values in
        ``[0, 1]``. Shape must be ``(N, 3)`` or ``(N, 4)``.

    Returns
    -------
    np.ndarray
        Array of hex strings of dtype ``|U9``, one per input color.
    """
    rgbs = _check_color_dim(rgbs)
    return np.array(
        [f"#{'%02x' * 4}" % tuple((255 * rgb).astype(np.uint8)) for rgb in rgbs],
        "|U9",
    )


def transform_color(color: ColorType) -> ty.Optional[np.ndarray]:
    """Normalize a color of any supported type to a 2-D RGBA array in 1-scale.

    Accepts hex strings (with ``'#'`` or ``'0x'`` prefix), ``'rgb(...)'`` and
    ``'rgba(...)'`` strings, lists, tuples, and numpy arrays.

    Parameters
    ----------
    color : ColorType
        Input color. Supported forms:

        - ``str``: ``'#rrggbb'``, ``'0xrrggbb'``, ``'rgb(...)'``,
          ``'rgba(...)'``
        - ``list`` or ``tuple``: 3 or 4 channels in either 1-scale or
          255-scale (auto-detected via ``max > 1``).
        - ``np.ndarray``: 1-D array of 3 or 4 channels in 1-scale.

    Returns
    -------
    np.ndarray or None
        Array of shape ``(1, 4)`` with RGBA values in ``[0, 1]``, or
        ``None`` if the input type is not recognized.

    Raises
    ------
    ValueError
        If the input is a recognized type but has an unsupported shape or
        format.
    """
    if isinstance(color, str):
        if color.startswith("#"):
            return np.atleast_2d(hex_to_rgb_1(color, with_alpha=True))
        if color.startswith("0x"):
            return np.atleast_2d(hex_to_rgb_1(color[2:], with_alpha=True))
        if color.startswith(("rgb", "rgba")):
            return np.atleast_2d(literal_eval(color))
        raise ValueError(f"Invalid color string: {color!r}")

    if isinstance(color, (list, tuple)):
        arr = np.atleast_2d(color)
        if arr.max() > 1:
            arr = arr / 255.0
        if arr.shape[1] == 4:
            return arr
        if arr.shape[1] == 3:
            return np.concatenate((arr, np.ones((arr.shape[0], 1))), axis=1)
        raise ValueError(f"Invalid color list/tuple: {color!r}")

    if isinstance(color, np.ndarray):
        if color.ndim == 1:
            if color.shape[0] == 3:
                return np.concatenate((color, [1]))
            if color.shape[0] == 4:
                return color
        raise ValueError(f"Invalid color array: {color!r}")

    return None
