"""Visuals."""
import numba as nb
import numpy as np
from matplotlib.collections import LineCollection


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
