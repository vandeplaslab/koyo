"""Visuals."""
import numba as nb
import numpy as np
from matplotlib.collections import LineCollection
import typing as ty

def fig_to_pil(fig):
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


def y_tick_fmt(x, pos):
    """Y-tick formatter."""

    def _convert_divider_to_str(value, exp_value):
        value = float(value)
        if exp_value in [0, 1, 2]:
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

    return _convert_divider_to_str(x, compute_divider(x))


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


def add_legend(
    fig,
    ax,
    legend_palettes: ty.Dict[str, ty.Dict[str, str]],
    fontsize: float = 14,
    labelsize: float = 16,
    x_pad: float = 0.01,
):
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
    """
    from natsort import natsorted
    from matplotlib.patches import Patch

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
    for tag, tag_to_color in legend_palettes.items():
        lh = [Patch(facecolor=tag_to_color[tag], label=tag) for tag in natsorted(tag_to_color.keys())]
        n_col = 1
        while True:
            n_col, leg = _make_legend(n_col, loc=(x_offset, y_offset))
            bb = leg.get_window_extent(rend).transformed(ax.transAxes.inverted())
            if bb.height <= 1:
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
        ax.add_artist(leg)