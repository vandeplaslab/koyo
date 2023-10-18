"""Mosaic utilities."""
import io
import typing as ty
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def add_label(
    ax, label: str, x: float = 0.9, y: float = 0.98, color="w", size=14, weight="normal", bbox=None, va="top", ha="left"
):
    """Add label to the image."""
    return ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=size,
        fontweight=weight,
        verticalalignment=va,
        horizontalalignment=ha,
        color=color,
        bbox=bbox,
    )


def fig_to_bytes(
    fig,
    bbox_inches: ty.Optional[str] = "tight",
    pad_inches: float = 0.1,
    dpi: int = 100,
    close: bool = False,
    transparent: bool = True,
):
    """Convert matplotlib figure to bytes."""
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="jpg",
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
        transparent=transparent,
    )
    buf.seek(0)
    if close:
        plt.close(fig)
    return buf


def make_fig_title(title: str, width: int, n_cols: int, text_color: str = "white", font_size: int = 36):
    """Make title for a single image."""
    fig = plt.figure(figsize=(width * n_cols / 72, 1))
    fig.text(
        0.5,
        0.5,
        title,
        transform=fig.transFigure,
        size=font_size,
        ha="center",
        va="center",
        color=text_color,
    )
    return fig_to_bytes(fig, bbox_inches=None, pad_inches=0, dpi=72, close=True)


def _merge_mosaic(
    n_rows: int,
    n_cols: int,
    width: int,
    height: int,
    items: ty.Dict[str, io.BytesIO],
    title_buf: ty.Optional[io.BytesIO] = None,
    silent: bool = True,
    color=(0, 0, 0, 0),
):
    from PIL import Image

    filelist = list(items.values())
    if title_buf is not None:
        title = Image.open(title_buf)
        dst = Image.new("RGB", (width * n_cols, height * n_rows + title.height), color=color)
        dst.paste(title, (0, 0))
        title.close()
        del title_buf
        y_offset = title.height
    else:
        dst = Image.new("RGB", (width * n_cols, height * n_rows), color=color)
        y_offset = 0
    k = 0  # image counter
    with tqdm(desc="Merging images...", total=len(filelist), disable=silent) as pbar:
        for i in range(n_rows):  # iterate over rows
            y = height * i + y_offset
            for j in range(n_cols):  # iterate over columns
                # load image
                try:
                    with Image.open(filelist[k]) as im:
                        x = width * j
                        dst.paste(im, (x, y))
                    k += 1  # increment image counter
                    pbar.update(1)
                except IndexError:
                    break
    return dst


def _get_mosaic_dims_for_list(items: ty.Dict[str, io.BytesIO], n_cols: int = 0, check_size_of_all: bool = True):
    from PIL import Image

    widths, heights = [], []
    if check_size_of_all:
        for buf in items.values():
            with Image.open(buf) as im:
                widths.append(im.width)
                heights.append(im.height)
        widths = np.unique(widths)
        heights = np.unique(heights)
        width, height = np.max(widths), np.max(heights)
    else:
        buf = next(iter(items.values()))
        with Image.open(buf) as im:
            width, height = im.width, im.height
    return _get_mosaic_dims(len(items), width, height, n_cols=n_cols)


def _get_mosaic_dims(n: int, width: int, height: int, n_cols: int = 0):
    ratio = 100
    _width, _height = width, height
    if n_cols == 0:
        n_rows, n_cols = 1, 1
        desired_ratio = 16 / 9
        while ratio > desired_ratio:
            n_cols = ceil(n / n_rows)
            width = _width * n_cols
            height = _height * n_rows
            ratio = width / height
            n_rows += 1
    else:
        if n_cols > n:
            n_cols = n
        n_rows = ceil(n / n_cols)
    if n_rows > ceil(n / n_cols):
        n_rows -= 1
    return n_rows, n_cols, _width, _height
