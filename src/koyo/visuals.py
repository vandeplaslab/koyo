"""Visuals."""

from __future__ import annotations

import math
import typing as ty
import warnings
from contextlib import suppress

import matplotlib.pyplot as plt
import numba as nb
import numpy as np

from koyo.image import clip_hotspots
from koyo.utilities import is_above_version

if ty.TYPE_CHECKING:
    from matplotlib.collections import LineCollection
    from PIL import Image
    from matplotlib.colorbar import Colorbar
    from matplotlib.image import AxesImage

MATPLOTLIB_ABOVE_3_5_2 = is_above_version("matplotlib", "3.5.2")


def tight_layout(fig: plt.Figure) -> None:
    """Apply tight layout to the figure."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()


def close_mpl_figure(fig: plt.Figure) -> None:
    """Close figure without raising exception."""
    with suppress(Exception):
        plt.close(fig)


def close_pil_image(fig: Image) -> None:
    """Close PIL image without raising exceptions."""
    with suppress(Exception):
        fig.close()
        del fig


def set_tick_fmt(ax: plt.Axes, use_offset: bool = False, axis: str = "both") -> plt.Axes:
    """Set tick format to control whether scientific notation is shown."""
    ax.ticklabel_format(axis=axis, style="scientific" if use_offset else "plain", useOffset=use_offset)
    return ax


def despine(ax: plt.Axes, orientation: str) -> plt.Axes:
    """Remove spines from 1D plots."""
    plt.setp(ax.xaxis.get_majorticklines(), visible=False)
    plt.setp(ax.xaxis.get_minorticklines(), visible=False)
    plt.setp(ax.yaxis.get_majorticklines(), visible=False)
    plt.setp(ax.yaxis.get_minorticklines(), visible=False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if orientation in ["horizontal", "horz", "h"]:
        ax.spines["left"].set_visible(False)
    else:
        ax.spines["bottom"].set_visible(False)
    return ax


def fig_to_pil(fig: plt.Axes) -> Image:
    """Convert a Matplotlib figure to a PIL Image and return it."""
    import io

    import matplotlib.pyplot as plt
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return img


def pil_to_fig(image: Image) -> plt.Figure:
    """Convert a PIL Image to a Matplotlib figure and return it."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(image.width / 72, image.height / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(image)
    return fig


def clear_axes(i: int, axs):
    """Clear axes that were not used."""
    if i < len(axs) - 1:
        for _i in range(i + 1, len(axs)):
            axs[_i].axis("off")


def auto_clear_axes(axs):
    """Automatically clear axes. This checks whether there are any lines or collections."""
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    for ax in axs:
        if not ax.lines and not ax.collections:
            ax.axis("off")


@nb.njit()
def convert_to_vertical_line_input(x: np.ndarray, y: np.ndarray):
    """Convert two arrays of the same length to list of [x, 0, y, y] vertices used by the matplotlib LineCollection."""
    return [[(x[i], 0), (x[i], y[i])] for i in range(len(x))]


def vertices_to_collection(
    x: np.ndarray,
    y: np.ndarray,
    color: str = "b",
    line_width=3,
    line_style="-",
    alpha: float = 1.0,
) -> LineCollection:
    """Convert list of [x, 0, y, y] vertices to line collection consumed by matplotlib."""
    from matplotlib.collections import LineCollection

    xy_values = convert_to_vertical_line_input(x, y)
    line_col = LineCollection(
        xy_values,
        colors=[color] * len(x),
        linewidths=line_width,
        linestyles=line_style,
        alpha=alpha,
    )
    return line_col


def compute_divider(value: float) -> float:
    """Compute divider."""
    divider = 1000000000
    value = abs(value)
    if value == 0:
        return 1
    while value == value % divider:
        divider = divider / 1000
    return len(str(int(divider))) - len(str(int(divider)).rstrip("0"))


def convert_divider_to_str(value, exp_value):
    value = float(value)
    if exp_value in [0, 1, 2]:
        if abs(value) < 0.0001:
            return f"{value:.6G}"
        if abs(value) < 0.01:
            return f"{value:.4G}"
        if abs(value) <= 1:
            return f"{value:.2G}"
        elif abs(value) <= 1000:
            if value.is_integer():
                return f"{value:.0F}"
            return f"{value:.1F}"
    elif exp_value in [3, 4, 5]:
        return f"{value / 1000:.1f}k"
    elif exp_value in [6, 7, 8]:
        return f"{value / 1000000:.1f}M"
    elif exp_value in [9, 10, 11, 12]:
        return f"{value / 1000000000:.1f}B"


def y_tick_fmt(x, pos=None):
    """Y-tick formatter."""
    return convert_divider_to_str(x, compute_divider(x))


def get_intensity_formatter():
    """Simple intensity formatter."""
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(y_tick_fmt)


def set_intensity_formatter(ax):
    """Set intensity formatter on axes."""
    ax.yaxis.set_major_formatter(get_intensity_formatter())


def add_ax_colorbar(mappable):
    """Add colorbar to axis."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


def make_legend_handles(
    names: list[str],
    colors: list[str],
    kind: list[str] = "line",
    width: list[float] = 3,
    extra_kws: dict | None = None,
):
    """Create custom legend."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    extra_kws = extra_kws or {}
    if isinstance(colors, (str, np.ndarray)):
        colors = [colors] * len(names)
    if isinstance(kind, str):
        kind = [kind] * len(names)
    if isinstance(width, (float, int)):
        width = [width] * len(names)
    handles = []
    for name, color, kind_, width_ in zip(names, colors, kind, width):
        kws = extra_kws.get(name, {})
        if kind_ == "line":
            handles.append(Line2D([0], [0], color=color, lw=width_, label=name, **kws))
        elif kind_ == "patch":
            handles.append(Patch(facecolor=color, edgecolor=color, label=name, **kws))
        elif kind_ == "patch":
            handles.append(Patch(facecolor=color, edgecolor=color, label=name, **kws))
        else:
            raise ValueError(f"Unknown kind {kind_}")
    return handles


def add_legend(
    fig: plt.Figure,
    ax: plt.Axes,
    legend_palettes: dict[str, dict[str, str]],
    fontsize: float = 14,
    labelsize: float = 16,
    x_pad: float = 0.01,
    sort_legend: bool = True,
) -> None:
    """Add legend to the plot.

    Parameters
    ----------
    fig
        Figure to add legend to.
    ax
        Axes to add legend to.
    legend_palettes
        Dictionary of tag name to tag to color mapping. The mapping should reflect all labels and colors that should be
        added to the legend.
    fontsize
        Font size of the legend labels.
    labelsize
        Font size of the legend title.
    x_pad : float
        Padding between the legend and the plot.
    sort_legend : bool
        Whether to sort the legend by tag name.
    """
    from matplotlib.patches import Patch
    from natsort import natsorted

    def _make_legend(n_col=1, loc="best"):
        return n_col, ax.legend(
            handles=lh,
            loc=loc,
            title=tag,
            frameon=False,
            handlelength=1.2,
            handleheight=1.4,
            fontsize=fontsize,
            title_fontsize=labelsize,
            ncol=n_col,
        )

    n_palettes = len(legend_palettes) > 1
    rend = fig.canvas.get_renderer()
    x_offset = ax.get_tightbbox(rend).transformed(ax.transAxes.inverted()).xmax + x_pad
    x_widest, y_offset = 0, 1
    if ax.get_title():
        y_offset += ax.title.get_window_extent(rend).transformed(ax.transAxes.inverted()).height
    # calculate maximum height
    max_height = 1.0
    xaxis = ax.get_xaxis()
    ticks = xaxis.get_ticklabels()
    if ticks:
        # find tick with longest label
        index = np.argmax([len(tick.get_text()) for tick in xaxis.get_ticklabels()])
        # get tick
        tick = xaxis.get_ticklabels()[index]
        max_height += tick.get_window_extent(rend).transformed(ax.transAxes.inverted()).height

    for tag, tag_to_color in legend_palettes.items():
        tag_to_color_ = natsorted(tag_to_color.keys()) if sort_legend else list(tag_to_color.keys())
        lh = [Patch(facecolor=tag_to_color[tag], label=tag) for tag in tag_to_color_]
        n_col = 1
        while True:
            n_col, leg = _make_legend(n_col, loc=(x_offset, y_offset))
            bb = leg.get_window_extent(rend).transformed(ax.transAxes.inverted())
            if bb.height <= max_height:
                break
            n_col += 1
        # if bb.height > 1:
        #     n_col, leg = _make_legend(2, (x_offset, y_offset))
        #     bb = leg.get_window_extent(rend).transformed(ax.transAxes.inverted())
        y_offset -= bb.height
        x_widest = max(x_widest, bb.width)
        if y_offset < 0 and n_palettes:
            x_offset += x_widest + 0.02
            y_offset = 1 - bb.height
        _, leg = _make_legend(n_col, (x_offset, y_offset))
        if not MATPLOTLIB_ABOVE_3_5_2:
            ax.add_artist(leg)


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


def stitch_mosaic(filename: str, filelist, n_cols: int):
    """Merge images of the same size and shape."""
    import math

    from PIL import Image

    shape = []
    for _filename in filelist:
        with Image.open(_filename) as file:
            shape.append((file.height, file.width))
    shape = np.array(shape)
    height, width = np.max(shape, axis=0)

    n_rows = math.ceil(len(filelist) / n_cols)

    dst = Image.new("RGB", (width * n_cols, height * n_rows))
    k = 0  # image counter
    for i in range(n_rows):
        y = height * i
        for j in range(n_cols):
            try:
                with Image.open(filelist[k]) as im:
                    x = width * j
                    dst.paste(im, (x, y))
                k += 1
            except IndexError:
                break
    dst.save(filename)
    del dst


def make_image_plot(heatmap, outfname: str | None = None, close: bool = False):
    """Generate simple heatmap."""
    from koyo.utilities import calculate_quantile_without_zeros

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(heatmap, aspect="equal", vmax=calculate_quantile_without_zeros(heatmap, 0.995))
    ax.axis("off")

    if outfname is not None:
        fig.savefig(outfname, bbox_inches="tight", pad_inches=0.1, dpi=150)

    if close:
        plt.close(fig)


def get_row_col(count, n_rows: int) -> tuple[int, int]:
    """Get number of rows and columns."""
    n_cols = math.ceil(count / n_rows)
    return n_rows, n_cols


def add_label(
    ax,
    label: str,
    x: float = 0.9,
    y: float = 0.98,
    label_color="w",
    font_size=14,
    font_weight="normal",
    va: str = "top",
    ha: str = "left",
    bbox: dict[str, ty.Any] | None = None,
):
    """Add label to the image."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=font_size,
        fontweight=font_weight,
        verticalalignment=va,
        horizontalalignment=ha,
        color=label_color,
        bbox=bbox,
    )


def get_ticks_with_unit(min_val: float, max_val: float, unit: str | None = None) -> tuple[list[float], list[str]]:
    """Get ticks for specified min/max range."""
    if not unit:
        return [min_val, max_val], [y_tick_fmt(min_val), y_tick_fmt(max_val)]
    between = np.linspace(min_val, max_val, 3)
    return between.tolist(), [y_tick_fmt(min_val), unit, y_tick_fmt(max_val)]


def inset_colorbar(
    ax: plt.Axes,
    im,
    ticks: ty.Sequence[float] | None = None,
    ticklabels: ty.Sequence[str] | None = None,
    xpos: float = 0.03,
    ypos: float = 0.05,
    labelcolor: str = "white",
    edgecolor: str = "white",
    **kwargs: ty.Any,
):
    """Add colorbar to axes."""

    def _parse_perc(value):
        if "%" in value:
            return float(value.replace("%", "")) / 100
        return float(value)

    width = _parse_perc(kwargs.get("width", "30%"))
    height = _parse_perc(kwargs.get("height", "4%"))
    cax = ax.inset_axes([xpos, ypos, width, height])
    cax.tick_params(labelcolor=labelcolor, labelsize=16)
    cbar = plt.colorbar(im, cax=cax, orientation=kwargs.get("orientation", "horizontal"), pad=0.1, ticks=ticks)
    if ticklabels:
        cax.ax.set_xticklabels(ticklabels)
    cbar.outline.set_edgecolor(edgecolor)
    cbar.outline.set_linewidth(1)
    return ax, cax, cbar


def fix_style(style: str) -> str:
    """Fix style so that it is compatible with matplotlib > v.3.6.0."""
    from matplotlib.style import available

    if style == "seaborn-grid":
        style = "seaborn-ticks"
    if style == "default" and "default" not in available:
        style = "seaborn-ticks"
    if style.startswith("seaborn"):
        if "v0_8" not in style and "seaborn-v0_8" in available:
            style = style.replace("seaborn", "seaborn-v0_8")
    if style != "default":
        assert style in available, f"Style '{style}' not available. Available styles: {available}"
    return style


def _annotate_heatmap(g, ax, mesh):
    from seaborn.utils import relative_luminance

    mesh.update_scalarmappable()
    height, width = g.annot_data.shape
    xpos, ypos = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    for x, y, m, color, val in zip(xpos.flat, ypos.flat, mesh.get_array(), mesh.get_facecolors(), g.annot_data.flat):
        if m is not np.ma.masked:
            lum = relative_luminance(color)
            text_color = ".15" if lum > 0.408 else "w"
            annotation = f"{val:.2f}"
            if val > 0:
                annotation = "1" if annotation in "1.00" else annotation.replace("0.", ".")
            else:
                annotation = "-1" if annotation in "-1.00" else annotation.replace("0.", ".")
            text_kwargs = {"color": text_color, "ha": "center", "va": "center"}
            text_kwargs.update(g.annot_kws)
            ax.text(x, y, annotation, **text_kwargs)


def _plot_or_update_image(
    ax: plt.Axes,
    array: np.ndarray,
    colormap: str = "viridis",
    colorbar: bool = True,
    clip: bool = False,
    is_normalized: bool = False,
    title: str = "",
    as_title: bool = True,
    quantile: float = 0.995,
    img: AxesImage | None = None,
    cbar: Colorbar | None = None,
    min_val: float | None = None,
    max_val: float | None = None,
) -> tuple[AxesImage, Colorbar | None]:
    """Plot or update image."""
    if min_val is None:
        min_val = np.nanmin(array)
    if clip:
        array = clip_hotspots(array, quantile)
    if not max_val and is_normalized:
        max_val = np.nanquantile(array, quantile) if quantile else np.nanmax(array)
    elif not max_val:
        max_val = np.nanmax(array)

    ticks, tick_labels = get_ticks_with_unit(min_val, max_val)
    if img is None:
        img = ax.imshow(array, vmax=max_val, cmap=colormap)
        if colorbar:
            _, _, cbar = inset_colorbar(
                ax,
                img,
                ticks=ticks,
                tick_labels=tick_labels,
                width="76%",
                xpos=0.12,
                ypos=-0.05,
                labelcolor="white",
                edgecolor="white",
            )
    else:
        img.set_array(array)
        img.set_cmap(colormap)
        img.set_clim(vmin=min_val, vmax=max_val)
        if cbar:
            cbar.mappable.set_clim(min_val, max_val)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(tick_labels)
    if as_title:
        ax.set_title(title, color="white", fontsize=16)
    else:
        add_label(
            ax,
            title,
            x=0.01,
            y=0.95,
            color="k",
            size=16,
            bbox={"boxstyle": "square", "facecolor": "white", "alpha": 0.75, "lw": 0},
        )
    return img, cbar
