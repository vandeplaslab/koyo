"""Tests for koyo.mosaic."""

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pytest
from koyo.mosaic import (
    RectanglePacker,
    _get_color_rgba255,
    _get_mosaic_dims,
    _get_mosaic_dims_for_list,
    convert_array_to_image,
    export_array_as_bytes,
    fig_to_bytes,
    get_positions,
    merge_mosaic_with_columns,
    pack_images,
)
from PIL import Image


def _image_bytes(color, size=(10, 8)):
    image = Image.new("RGB", size, color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_rectangle_packer_respects_min_width():
    packer = RectanglePacker(min_width=50)
    packer.initialize_canvas(10, 10)
    packer.place_rectangle(10, 5)
    width, height = packer.get_packed_area()
    assert width >= 50
    assert height == 5


def test_get_positions_returns_sorted_images():
    images = [Image.new("RGB", (5, 5), "red"), Image.new("RGB", (10, 8), "blue")]
    (width, height), positions, sorted_images = get_positions(images)
    assert width >= 10
    assert height >= 13
    assert len(positions) == 2
    assert all(len(position) == 2 for position in positions)
    assert [im.size for im in sorted_images] == [(10, 8), (5, 5)]


def test_pack_images_combines_images():
    images = [Image.new("RGB", (5, 5), "red"), Image.new("RGB", (10, 8), "blue")]
    result = pack_images(images)
    assert result.size[0] >= 10
    assert result.size[1] >= 13


def test_fig_to_bytes_supports_figure_and_image():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig_bytes = fig_to_bytes(fig, close=True)
    image_bytes = fig_to_bytes(Image.new("RGB", (5, 5), "red"))
    assert fig_bytes.getbuffer().nbytes > 0
    assert image_bytes.getbuffer().nbytes > 0


def test_merge_mosaic_with_columns_list_input():
    buffers = [_image_bytes("red", (4, 3)), _image_bytes("blue", (6, 2)), _image_bytes("green", (5, 4))]
    result = merge_mosaic_with_columns(buffers, n_cols=2, x_pad=1, y_pad=2)
    assert result.size == (12, 9)


def test_merge_mosaic_with_columns_rejects_empty_dir(tmp_path):
    with pytest.raises(ValueError, match="No images found"):
        merge_mosaic_with_columns(tmp_path, n_cols=2)


def test_get_mosaic_dims_for_list_uses_largest_image():
    items = {"a": _image_bytes("red", (4, 3)), "b": _image_bytes("blue", (6, 5))}
    assert _get_mosaic_dims_for_list(items, n_cols=2, x_pad=1, y_pad=2) == (1, 2, 8, 9)


def test_get_mosaic_dims_caps_columns():
    assert _get_mosaic_dims(2, 10, 5, n_cols=5) == (1, 2, 10, 5)


def test_convert_array_to_image_and_export_bytes():
    array = np.array([[0.0, 1.0], [np.nan, 2.0]])
    image = convert_array_to_image(array, title="X")
    payload = export_array_as_bytes(array)
    assert image.size == (2, 2)
    assert payload.getbuffer().nbytes > 0


def test_get_color_rgba255():
    assert _get_color_rgba255("black") == (0, 0, 0, 255)
