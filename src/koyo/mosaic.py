"""Mosaic utilities."""

from __future__ import annotations

import io
import typing as ty
from functools import lru_cache
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba_array
from tqdm import tqdm

from koyo.typing import PathLike
from koyo.utilities import rotate
from koyo.visuals import fix_style

if ty.TYPE_CHECKING:
    from PIL import Image


class RectanglePacker:
    """Rectangle packer."""

    def __init__(self, min_width: int = 0, x_pad: int = 0, y_pad: int = 0):
        self.free_rectangles = []  # List of free spaces (x, y, w, h)
        self.placed_rectangles = []  # List of placed rectangles (x, y, w, h)
        self.current_width = 0  # Current canvas width
        self.current_height = 0  # Current canvas height
        self.min_width = min_width
        self.x_pad = x_pad
        self.y_pad = y_pad

    def initialize_canvas(self, initial_width: int, initial_height: int):
        """Initialize the canvas with a starting size."""
        self.current_width = max(initial_width, self.min_width)
        self.current_height = initial_height
        self.free_rectangles = [(0, 0, self.current_width, self.current_height)]

    def place_rectangle(self, rect_width: int, rect_height: int) -> tuple[int, int]:
        """
        Place a rectangle of size (rect_width, rect_height) in the packing area.
        Returns the (x, y) position if placed, or raises an error if no space is available.
        """
        padded_width = rect_width + self.x_pad * 2
        padded_height = rect_height + self.y_pad * 2

        for i, (x, y, w, h) in enumerate(self.free_rectangles):
            if padded_width <= w and padded_height <= h:
                # Place the rectangle
                self.placed_rectangles.append((x + self.x_pad, y + self.y_pad, rect_width, rect_height))
                self.split_free_space(i, x, y, padded_width, padded_height)
                return x + self.x_pad, y + self.y_pad

        # If no space is found, expand the canvas
        self.expand_canvas(padded_width, padded_height)
        return self.place_rectangle(rect_width, rect_height)

    def expand_canvas(self, rect_width: int, rect_height: int):
        """Expand the canvas size to accommodate a rectangle that doesn't fit."""
        new_width = max(self.current_width, rect_width)
        new_height = self.current_height + rect_height
        self.free_rectangles.append((0, self.current_height, new_width, rect_height))
        self.current_width = new_width
        self.current_height = new_height

    def split_free_space(self, index: int, x: int, y: int, rect_width: int, rect_height: int):
        """Split the free space at index into smaller regions after placing a rectangle."""
        original_x, original_y, original_w, original_h = self.free_rectangles.pop(index)

        # Space to the right of the rectangle
        right_space = (x + rect_width, y, original_x + original_w - (x + rect_width), rect_height)
        if right_space[2] > 0 and right_space[3] > 0:
            self.free_rectangles.append(right_space)

        # Space below the rectangle
        below_space = (x, y + rect_height, original_w, original_y + original_h - (y + rect_height))
        if below_space[2] > 0 and below_space[3] > 0:
            self.free_rectangles.append(below_space)

        # Remaining space to the left and above (if applicable)
        left_space = (original_x, original_y, x - original_x, original_h)
        if left_space[2] > 0 and left_space[3] > 0:
            self.free_rectangles.append(left_space)

        above_space = (original_x, original_y, original_w, y - original_y)
        if above_space[2] > 0 and above_space[3] > 0:
            self.free_rectangles.append(above_space)

        # Sort free spaces by area (smaller spaces prioritized for tight packing)
        self.free_rectangles.sort(key=lambda r: r[2] * r[3])

    def get_packed_area(self) -> tuple[int, int]:
        """Returns the width and height of the smallest rectangle that fits all placed items."""
        max_width = max(x + w for x, y, w, h in self.placed_rectangles)
        max_height = max(y + h for x, y, w, h in self.placed_rectangles)
        return max(max_width, self.min_width), max_height


def get_positions(
    images: list[Image], min_width: int = 0, x_pad: int = 0, y_pad: int = 0
) -> tuple[tuple[int, int], list[tuple[int, int]], list[Image]]:
    """Get positions and canvas size."""
    packer = RectanglePacker(min_width=min_width, x_pad=x_pad, y_pad=y_pad)

    # Sort images by area (descending) for better packing
    images = sorted(images, key=lambda im: im.width * im.height, reverse=True)

    # Start with an initial canvas size
    initial_width = max(im.width for im in images) + x_pad * 2
    initial_height = sum(im.height for im in images) // len(images) + y_pad * 2
    packer.initialize_canvas(initial_width, initial_height)

    # Pack each image
    packed_positions = []
    for image in images:
        rect_width, rect_height = image.width, image.height
        position = packer.place_rectangle(rect_width, rect_height)
        packed_positions.append(position)

    # Get the final bounding box for the packed items
    packed_width, packed_height = packer.get_packed_area()
    return (packed_width, packed_height), packed_positions, images


def pack_images(images: list[Image], min_width: int = 0, x_pad: int = 0, y_pad: int = 0) -> Image:
    """Packs a list of images into a high-density rectangle packing with automatic canvas sizing."""
    packer = RectanglePacker(min_width=min_width, x_pad=x_pad, y_pad=y_pad)

    # Sort images by area (descending) for better packing
    images = sorted(images, key=lambda im: im.width * im.height, reverse=True)

    # Start with an initial canvas size
    initial_width = max(im.width for im in images) + x_pad * 2
    initial_height = sum(im.height for im in images) // len(images) + y_pad * 2
    packer.initialize_canvas(initial_width, initial_height)

    # Pack each image
    packed_positions = []
    for image in images:
        rect_width, rect_height = image.width, image.height
        position = packer.place_rectangle(rect_width, rect_height)
        packed_positions.append((image, position))

    # Get the final bounding box for the packed items
    packed_width, packed_height = packer.get_packed_area()

    # Create a canvas to draw the packed images
    canvas = Image.new("RGB", (packed_width, packed_height), (0, 0, 0))
    for image, (x, y) in packed_positions:
        canvas.paste(image, (x, y))
    return canvas


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
    fig: plt.Figure | Image,
    bbox_inches: str | None = "tight",
    pad_inches: float = 0.1,
    dpi: int = 100,
    close: bool = False,
    transparent: bool = True,
) -> io.BytesIO:
    """Convert matplotlib figure to bytes."""
    buf = io.BytesIO()
    if isinstance(fig, plt.Figure):
        fig.savefig(
            buf,
            format="jpg",
            dpi=dpi,
            facecolor=fig.get_facecolor(),
            edgecolor=fig.get_edgecolor(),
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            transparent=transparent,
        )
        if close:
            plt.close(fig)
    else:
        fig.save(buf, format="PNG")
        if close:
            fig.close()
    buf.seek(0)
    return buf


def make_fig_title(title: str, width: int, n_cols: int, text_color: str = "auto", font_size: int = 36) -> io.BytesIO:
    """Make title for a single image."""
    n_rows = max(1, title.count("\n"))
    fig = plt.figure(figsize=(width * n_cols / 72, n_rows))
    text_color = text_color if text_color != "auto" else plt.rcParams["text.color"]

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
    x_pad: int = 0,
    y_pad: int = 0,
    add_title_to_images: bool = False,
    text_color: str | tuple[int, int, int, int] = (255, 255, 255, 255),
    font_size: int = 36,
) -> Image:
    """Merge images."""
    nr, nc, w, h = _get_mosaic_dims_for_list(items, n_cols=n_cols, x_pad=x_pad, y_pad=y_pad)
    if title:
        title_buf = make_fig_title(title, w, nc, font_size=font_size)
    return _merge_mosaic(
        nr,
        nc,
        w,
        h,
        items,
        title_buf,
        silent,
        color,
        allow_placeholder,
        placeholder_color,
        x_pad=x_pad,
        y_pad=y_pad,
        add_title_to_images=add_title_to_images,
        text_color=text_color,
    )


def merge_mosaic_packed(
    items: dict[str, io.BytesIO],
    title_buf: io.BytesIO | None = None,
    title: str = "",
    min_width: int = 0,
    silent: bool = True,
    color: tuple[int, ...] = (0, 0, 0, 0),  # black
    allow_placeholder: bool = False,
    placeholder_color: tuple[int, ...] = (0, 0, 0, 255),  # black
    x_pad: int = 0,
    y_pad: int = 0,
):
    """Pack images tightly into a grid."""
    from PIL import Image

    images = [Image.open(buf) for buf in items.values()]
    (w, h), positions, images = get_positions(images, min_width=min_width, x_pad=x_pad, y_pad=y_pad)
    if title:
        title_buf = make_fig_title(title, w, 1)

    if title_buf is not None:
        title = Image.open(title_buf)
        dst = Image.new("RGB", (w, h + title.height), color=color)
        dst.paste(title, (0, 0))
        title.close()
        del title_buf
        y_offset = title.height
    else:
        dst = Image.new("RGB", (w, h), color=color)
        y_offset = 0
    with tqdm(desc="Merging images...", total=len(images), disable=silent) as pbar:
        for im, (x, y) in zip(images, positions):
            dst.paste(im, (x, y + y_offset))
            pbar.update(1)
    return dst


def merge_mosaic_from_dir(
    image_dir: PathLike,
    title: str = "",
    n_cols: int | None = None,
    x_pad: int = 0,
    y_pad: int = 0,
) -> Image:
    """Merge images from a directory."""
    items = {}
    for filename in image_dir.glob("*"):
        if filename.is_file():
            with open(filename, "rb") as f:
                items[filename.stem] = io.BytesIO(f.read())
    return merge_mosaic(items, title=title, n_cols=n_cols, x_pad=x_pad, y_pad=y_pad)


def merge_mosaic_with_columns(
    image_dir: PathLike | list[io.BytesIO],
    n_cols: int,
    x_pad: int = 0,
    y_pad: int = 0,
    placeholder_color: tuple[int, ...] = (0, 0, 0, 255),  # black
) -> Image:
    """Merge images from a directory into a grid with a fixed number of columns."""
    # Load images into a list
    from PIL import Image

    items = []
    if isinstance(image_dir, list):
        for buf in image_dir:
            with Image.open(buf) as im:
                items.append((im.copy(), im.width, im.height))
    else:
        for filename in image_dir.glob("*"):
            if filename.is_file():
                with open(filename, "rb") as f:
                    with Image.open(io.BytesIO(f.read())) as im:
                        items.append((im.copy(), im.width, im.height))

    if not items:
        raise ValueError("No images found in the specified directory.")

    # Calculate the number of rows
    num_images = len(items)
    n_rows = ceil(num_images / n_cols)

    # Determine the width of each column and height of each row
    column_widths = [0] * n_cols
    row_heights = [0] * n_rows

    # Assign images to the grid
    grid = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    for idx, (im, width, height) in enumerate(items):
        row = idx // n_cols
        col = idx % n_cols
        grid[row][col] = im
        column_widths[col] = max(column_widths[col], width)
        row_heights[row] = max(row_heights[row], height)

    # Calculate the total mosaic dimensions
    mosaic_width = sum(column_widths) + (n_cols - 1) * x_pad
    mosaic_height = sum(row_heights) + (n_rows - 1) * y_pad

    # Create the output mosaic
    mosaic = Image.new("RGB", (mosaic_width, mosaic_height), color=placeholder_color)

    # Place images in the mosaic
    y_offset = 0
    for row_idx, row in enumerate(grid):
        x_offset = 0
        for col_idx, im in enumerate(row):
            if im:
                mosaic.paste(im, (x_offset, y_offset))
            x_offset += column_widths[col_idx] + x_pad
        y_offset += row_heights[row_idx] + y_pad

    return mosaic


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
    placeholder_color: tuple[int, ...] = (0, 0, 0, 255),
    x_pad: int = 0,
    y_pad: int = 0,
    add_title_to_images: bool = False,
    text_color: str | tuple[int, int, int, int] = (255, 255, 255, 255),
    align_to: ty.Literal["top", "center", "bottom"] = "center",
) -> Image:
    from PIL import Image

    if isinstance(color, tuple):
        assert len(color) in [3, 4], f"Color must be a tuple of 3 or 4 integers, not {color}."
    color = (*tuple(color), 255) if len(color) == 3 else color
    if isinstance(placeholder_color, tuple):
        assert len(placeholder_color) in [3, 4], (
            f"Placeholder color must be a tuple of 3 or 4 integers, not {placeholder_color}."
        )
    placeholder_color = (*tuple(placeholder_color), 255) if len(placeholder_color) == 3 else placeholder_color

    names = list(items.keys())
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
            for j in range(n_cols):  # iterate over columns
                try:
                    filename = filelist[k]
                    im = None
                    if filename:
                        im = Image.open(filename)
                    elif filename is None and allow_placeholder:
                        im = Image.new("RGB", (width, height), color=placeholder_color)

                    try:
                        if align_to == "top":
                            y = height * i + y_offset + y_pad
                        elif align_to == "center":
                            y = height * i + y_offset + (height - im.height) // 2
                        elif align_to == "bottom":
                            y = height * i + y_offset + height - im.height - y_pad
                    except AttributeError:
                        y = height * i + y_offset + y_pad

                    if im:
                        # load image
                        x = width * j + x_pad
                        dst.paste(im, (x, y))
                        if add_title_to_images and filename:
                            _add_text_to_pil_image(dst, str(names[k]), (x, y), text_color)
                        im.close()
                    k += 1  # increment image counter
                    pbar.update(1)
                except IndexError:
                    break
    return dst


def _get_mosaic_dims_for_list(
    items: dict[str, io.BytesIO],
    n_cols: int | None = 0,
    check_size_of_all: bool = True,
    x_pad: int = 0,
    y_pad: int = 0,
) -> tuple[int, int, int, int]:
    from PIL import Image

    if n_cols is None:
        n_cols = 0

    x_pad = x_pad * 2
    y_pad = y_pad * 2

    widths, heights = [], []
    if check_size_of_all:
        for buf in items.values():
            if buf is None:
                continue
            with Image.open(buf) as im:
                widths.append(im.width + x_pad)
                heights.append(im.height + y_pad)
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
    while n_rows > ceil(n / n_cols):
        n_rows -= 1
    return n_rows, n_cols, _width, _height


def plot_mosaic(
    data: dict[str, np.ndarray],
    title: str = "",
    colormap: str | dict[str, str] = "viridis",
    colorbar: bool = True,
    dpi: int = 100,
    min_val: float | None = None,
    max_val: float | None = None,
    figsize: tuple[float, float] = (6, 6),
    n_cols: int | None = None,
    style: str = "dark_background",
    color: tuple[int, int, int, int] | None = None,
    placeholder_color: tuple[int, int, int, int] | None = None,
    highlight: str | None = None,
    auto_rotate: bool = False,
    **kwargs: ty.Any,
) -> Image:
    """Plot mosaic."""
    from koyo.visuals import _plot_or_update_image

    img, cbar = None, None
    figures = {}
    with plt.style.context(fix_style(style)):
        if color is None:
            color = _get_color_rgba255(plt.rcParams["axes.facecolor"])
        if placeholder_color is None:
            placeholder_color = _get_color_rgba255(plt.rcParams["axes.facecolor"])

        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")

        for key in data:
            img, cbar = _plot_or_update_image(
                ax,
                rotate(data[key], auto_rotate),
                min_val=min_val,
                max_val=max_val,
                img=img,
                cbar=cbar,
                colorbar=colorbar,
                colormap=colormap if isinstance(colormap, str) else colormap[key],
                title=key,
            )
            if highlight and key == highlight:
                # change ax title color
                ax.title.set_color("red")
            else:
                ax.title.set_color(plt.rcParams["text.color"])
            figures[key] = fig_to_bytes(fig, close=False, dpi=dpi)
        plt.close(fig)
        image = merge_mosaic(
            figures,
            title=title,
            n_cols=n_cols,
            placeholder_color=placeholder_color,
            color=color,
        )
    return image


def plot_mosaic_no_colorbar(
    data: dict[str, np.ndarray],
    colormap: str | dict[str, str] = "viridis",
    min_val: float | None = None,
    max_val: float | None = None,
    title: str = "",
    auto_rotate: bool = False,
    n_cols: int | None = None,
    style: str = "dark_background",
    color: tuple[int, int, int, int] | None = None,
    placeholder_color: tuple[int, int, int, int] | None = None,
    **kwargs: ty.Any,
) -> Image:
    """Plot mosaic without colorbar."""
    images = {}
    with plt.style.context(style):
        if color is None:
            color = _get_color_rgba255(plt.rcParams["axes.facecolor"])
        if placeholder_color is None:
            placeholder_color = _get_color_rgba255(plt.rcParams["axes.facecolor"])
        for key in data:
            images[key] = fig_to_bytes(
                convert_array_to_image(
                    rotate(data[key], auto_rotate),
                    title="",
                    min_val=min_val,
                    max_val=max_val,
                    colormap=colormap if isinstance(colormap, str) else colormap[key],
                )
            )
        image = merge_mosaic(
            images,
            title=title,
            x_pad=5,
            y_pad=5,
            add_title_to_images=True,
            n_cols=n_cols,
            placeholder_color=placeholder_color,
            color=color,
        )
    return image


def plot_mosaic_individual(
    data: dict[str, np.ndarray],
    title: str = "",
    colormap: str | dict[str, str] = "viridis",
    colorbar: bool = True,
    dpi: int = 100,
    min_val: float | None = None,
    max_val: float | None = None,
    figsize: tuple[float, float] = (6, 6),
    style: str = "dark_background",
    n_cols: int | None = None,
    border_color: dict[str, str] | None = None,
    title_color: dict[str, str] | None = None,
    color: tuple[int, int, int, int] | None = None,
    placeholder_color: tuple[int, int, int, int] | None = None,
    auto_rotate: bool = False,
    **kwargs: ty.Any,
) -> Image:
    """Plot mosaic."""
    from koyo.visuals import _plot_image

    border_color = {} if border_color is None else border_color
    title_color = {} if title_color is None else title_color

    figures = {}
    with plt.style.context(fix_style(style)):
        if color is None:
            color = _get_color_rgba255(plt.rcParams["axes.facecolor"])
        if placeholder_color is None:
            placeholder_color = _get_color_rgba255(plt.rcParams["axes.facecolor"])

        for key in data:
            fig, ax = _plot_image(
                rotate(data[key], auto_rotate),
                min_val=min_val,
                max_val=max_val,
                colorbar=colorbar,
                colormap=colormap if isinstance(colormap, str) else colormap[key],
                title=key,
                figsize=figsize,
                border_color=border_color.get(key, None),
                title_color=title_color.get(key, None),
            )
            ax.axis("off")
            figures[key] = fig_to_bytes(fig, close=True, dpi=dpi)
        image = merge_mosaic(
            figures,
            title=title,
            n_cols=n_cols,
            placeholder_color=placeholder_color,
            color=color,
        )
    return image


def plot_mosaic_line_individual(
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    marker: float | dict[str, float] | None = None,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
    dpi: int = 100,
    figsize: tuple[float, float] = (6, 6),
    style: str = "seaborn-ticks",
    n_cols: int | None = None,
    border_color: dict[str, str] | None = None,
    title_color: dict[str, str] | None = None,
    color: tuple[int, int, int, int] | None = None,
    placeholder_color: tuple[int, int, int, int] | None = None,
) -> Image:
    """Plot mosaic."""
    from koyo.visuals import _plot_line

    border_color = {} if border_color is None else border_color
    title_color = {} if title_color is None else title_color

    figures = {}
    with plt.style.context(fix_style(style)):
        if color is None:
            color = _get_color_rgba255(plt.rcParams["axes.facecolor"])
        if placeholder_color is None:
            placeholder_color = _get_color_rgba255(plt.rcParams["axes.facecolor"])
        for key in data:
            fig, ax = _plot_line(
                data[key][0],
                data[key][1],
                marker=marker[key] if isinstance(marker, dict) else marker,
                title=key,
                x_label=x_label,
                y_label=y_label,
                figsize=figsize,
                border_color=border_color.get(key, None),
                title_color=title_color.get(key, None),
            )
            figures[key] = fig_to_bytes(fig, close=True, dpi=dpi)
        image = merge_mosaic(
            figures,
            title=title,
            n_cols=n_cols,
            color=color,
            placeholder_color=placeholder_color,
        )
    return image


def _get_color_rgba255(color) -> tuple[int, int, int, int]:
    color = to_rgba_array(color) * 255
    color = color.astype(int)
    if color.ndim == 2:
        return tuple(color[0])
    return tuple(color)


@lru_cache(maxsize=10)
def find_font(name: str = "arial.ttf") -> str | None:
    """Find font."""
    # get list of fonts
    from matplotlib import font_manager

    fonts = font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
    for font in fonts:
        if name in font.lower():
            return font
    return None


def convert_array_to_image(
    array: np.ndarray,
    colormap: str = "viridis",
    title: str = "",
    min_val: float | None = None,
    max_val: float | None = None,
) -> Image:
    """Convert a 2D array to a PNG image using a specified colormap."""
    from matplotlib.colors import Normalize, to_hex
    from PIL import Image

    # Define the colormap
    cmap = plt.get_cmap(colormap)

    # Normalize the array excluding NaNs or zeros
    valid_mask = ~np.isnan(array) & (array != 0)
    min_val = min_val if min_val is not None else np.min(array[valid_mask])
    max_val = max_val if max_val is not None else np.max(array[valid_mask])
    norm = Normalize(vmin=min_val, vmax=max_val)

    # Create the normalized array
    normalized_array = np.zeros_like(array, dtype=float)  # Default to 0 for NaNs and zeros
    normalized_array[valid_mask] = norm(array[valid_mask])

    # Apply the colormap
    colored_image = cmap(normalized_array)

    # Set NaNs and zeros to black (0, 0, 0, 1 in RGBA)
    colored_image[~valid_mask] = (0, 0, 0, 1)

    # Convert to 8-bit unsigned integers
    uint8_image = (colored_image[:, :, :3] * 255).astype(np.uint8)  # Drop alpha channel for saving

    # Save the image
    image = Image.fromarray(uint8_image)
    if title:
        if colormap == "viridis":
            color = "white"
        else:
            color = to_hex(plt.rcParams["text.color"])
        _add_text_to_pil_image(image, title, (3, 3), color)
    return image


def _add_text_to_pil_image(
    image: Image, text: str, position: tuple[int, int], color: str | tuple[int, int, int, int]
) -> None:
    from PIL import ImageDraw, ImageFont

    font_path = find_font()
    if font_path:
        font = ImageFont.truetype(font_path, 14)
    else:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)
    draw.text(position, text, fill=color, font=font)  # (255, 255, 255, 255))


def export_array_as_bytes(array: np.ndarray, colormap: str = "viridis") -> io.BytesIO:
    """Export a 2D array as a PNG image in bytes format."""
    image = convert_array_to_image(array, colormap)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output
