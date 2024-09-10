"""Mosaic utilities."""

from __future__ import annotations

import io
import typing as ty
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

if ty.TYPE_CHECKING:
    from PIL import Image


def add_label(
    ax: plt.Axes,
    label: str,
    x: float = 0.9,
    y: float = 0.98,
    color: str = "w",
    size: int = 14,
    weight: str = "normal",
    bbox: ty.Any = None,
    va: str = "top",
    ha: str = "left",
) -> plt.Text:
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
    fig: plt.Figure,
    bbox_inches: str | None = "tight",
    pad_inches: float = 0.1,
    dpi: int = 100,
    close: bool = False,
    transparent: bool = True,
) -> io.BytesIO:
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


def make_fig_title(title: str, width: int, n_cols: int, text_color: str = "white", font_size: int = 36) -> io.BytesIO:
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


def merge_mosaic(
    items: dict[str, io.BytesIO],
    title_buf: io.BytesIO | None = None,
    title: str = "",
    n_cols: int | None = None,
    silent: bool = True,
    color: tuple[int, ...] = (0, 0, 0, 0),  # black
    allow_placeholder: bool = False,
    placeholder_color: tuple[int, ...] = (0, 0, 0, 255),  # black
) -> Image:
    """Merge images."""
    nr, nc, w, h = _get_mosaic_dims_for_list(items, n_cols=n_cols)
    if title:
        title_buf = make_fig_title(title, w, nc)
    return _merge_mosaic(nr, nc, w, h, items, title_buf, silent, color, allow_placeholder, placeholder_color)


def _merge_mosaic(
    n_rows: int,
    n_cols: int,
    width: int,
    height: int,
    items: dict[str, io.BytesIO],
    title_buf: io.BytesIO | None = None,
    silent: bool = True,
    color: tuple[int, ...] = (0, 0, 0, 0),
    allow_placeholder: bool = False,
    placeholder_color: tuple[int, ...] = (128, 0, 0, 255),
) -> Image:
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
                    filename = filelist[k]
                    if filename:
                        with Image.open(filename) as im:
                            x = width * j
                            dst.paste(im, (x, y))
                    elif filename is None and allow_placeholder:
                        x = width * j
                        dst.paste(Image.new("RGB", (width, height), color=placeholder_color), (x, y))
                    k += 1  # increment image counter
                    pbar.update(1)
                except IndexError:
                    break
    return dst


def _get_mosaic_dims_for_list(
    items: dict[str, io.BytesIO], n_cols: int | None = 0, check_size_of_all: bool = True
) -> tuple[int, int, int, int]:
    from PIL import Image

    if n_cols is None:
        n_cols = 0

    widths, heights = [], []
    if check_size_of_all:
        for buf in items.values():
            if buf is None:
                continue
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


def _get_mosaic_dims(n: int, width: int, height: int, n_cols: int = 0) -> tuple[int, int, int, int]:
    ratio = 100.0
    _width, _height = width, height
    if n_cols == 0:
        n_rows, n_cols = 1, 1
        desired_ratio = 16 / 9
        while ratio > desired_ratio:
            n_cols = ceil(n / n_rows)
            height = _height * n_rows
            ratio = (_width * n_cols) / height
            n_rows += 1
    else:
        if n_cols > n:
            n_cols = n
        n_rows = ceil(n / n_cols)
    if n_rows > ceil(n / n_cols):
        n_rows -= 1
    return n_rows, n_cols, _width, _height
