"""Test color functions"""
import warnings

import numpy as np
import pytest

from koyo.color import (
    find_text_color,
    generate_distinct_colors,
    get_next_color,
    get_random_hex_color,
    hex_to_rgb,
    hex_to_rgb_1,
    hex_to_rgb_255,
    rgb_1_to_hex,
    rgb_255_to_1,
    rgb_to_hex,
    rgbs_to_hex,
    transform_color,
)


def test_convert_rgb_255_to_1():
    expected_color = [1.0, 1.0, 1.0]

    returned_color = rgb_255_to_1([255, 255, 255])
    assert set(expected_color) == set(returned_color)

    returned_color = rgb_255_to_1("[255, 255, 255]")
    assert set(expected_color) == set(returned_color)

    with pytest.raises(ValueError):
        rgb_255_to_1([255, 255, 255, 255, 255])
    with pytest.raises(ValueError):
        rgb_255_to_1([255, 255, 255, -1])
    with pytest.raises(ValueError):
        rgb_255_to_1([255, 255, 255, 256])


@pytest.mark.parametrize("hex_color, rgb_expected", (["#000000", [0, 0, 0]], ["#FFFFFF", [1, 1, 1]]))
def test_convert_hex_to_rgb_1(hex_color, rgb_expected):
    rgb_result = hex_to_rgb_1(hex_color)
    assert rgb_result == rgb_expected


@pytest.mark.parametrize("hex_color, rgb_expected", (["#000000", [0, 0, 0]], ["#FFFFFF", [255, 255, 255]]))
def test_convert_hex_to_rgb_255(hex_color, rgb_expected):
    rgb_result = hex_to_rgb_255(hex_color)
    assert rgb_result == rgb_expected


def test_get_random_hex_color():
    color = get_random_hex_color()
    assert color.startswith("#")
    color_rgb = hex_to_rgb_255(color)
    assert all(value <= 255 for value in color_rgb)


# ---------------------------------------------------------------------------
# rgb_1_to_hex
# ---------------------------------------------------------------------------


def test_rgb_1_to_hex_black():
    assert rgb_1_to_hex([0.0, 0.0, 0.0]) == "#000000"


def test_rgb_1_to_hex_white():
    assert rgb_1_to_hex([1.0, 1.0, 1.0]) == "#ffffff"


def test_rgb_1_to_hex_ignores_alpha():
    assert rgb_1_to_hex([1.0, 0.0, 0.0, 0.5]) == "#ff0000"


# ---------------------------------------------------------------------------
# rgb_to_hex
# ---------------------------------------------------------------------------


def test_rgb_to_hex_known_values():
    assert rgb_to_hex([1.0, 0.0, 0.0]) == "#ff0000"
    assert rgb_to_hex([0.0, 1.0, 0.0]) == "#00ff00"
    assert rgb_to_hex([0.0, 0.0, 1.0]) == "#0000ff"


# ---------------------------------------------------------------------------
# hex_to_rgb
# ---------------------------------------------------------------------------


def test_hex_to_rgb_shape():
    result = hex_to_rgb("#ff0000")
    assert result.shape == (3,)
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(0.0)


def test_hex_to_rgb_with_alpha():
    result = hex_to_rgb("#ff0000", alpha=128)
    assert result.shape == (4,)


def test_hex_to_rgb_alpha_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        hex_to_rgb("#ff0000", alpha=1)
    assert len(w) == 1
    assert "0-255" in str(w[0].message)


# ---------------------------------------------------------------------------
# get_next_color
# ---------------------------------------------------------------------------


def test_get_next_color_returns_hex():
    color = get_next_color(0)
    assert color.startswith("#")
    assert len(color) == 7


def test_get_next_color_avoids_other_colors():
    color0 = get_next_color(0)
    color = get_next_color(0, other_colors=[color0])
    assert color != color0


def test_get_next_color_beyond_palette():
    color = get_next_color(20)
    assert color.startswith("#")


# ---------------------------------------------------------------------------
# generate_distinct_colors
# ---------------------------------------------------------------------------


def test_generate_distinct_colors_length():
    starting = ["#ff0000", "#00ff00"]
    result = generate_distinct_colors(starting, 5)
    assert len(result) == 5


def test_generate_distinct_colors_preserves_start():
    starting = ["#ff0000", "#00ff00"]
    result = generate_distinct_colors(starting, 4)
    assert result[0] == "#ff0000"
    assert result[1] == "#00ff00"


def test_generate_distinct_colors_fewer_than_starting():
    starting = ["#ff0000", "#00ff00", "#0000ff"]
    result = generate_distinct_colors(starting, 2)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# rgbs_to_hex
# ---------------------------------------------------------------------------


def test_rgbs_to_hex_shape():
    rgbs = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]])
    result = rgbs_to_hex(rgbs)
    assert result.shape == (2,)
    assert result.dtype == np.dtype("|U9")


def test_rgbs_to_hex_values():
    rgbs = np.array([[0.0, 0.0, 0.0, 1.0]])
    result = rgbs_to_hex(rgbs)
    assert result[0] == "#000000ff"


# ---------------------------------------------------------------------------
# transform_color
# ---------------------------------------------------------------------------


def test_transform_color_hex_string():
    result = transform_color("#ff0000")
    assert result.shape == (1, 4)
    assert result[0, 0] == pytest.approx(1.0)


def test_transform_color_0x_prefix():
    result = transform_color("0xff0000")
    assert result is not None
    assert result.shape == (1, 4)


def test_transform_color_list_255_scale():
    result = transform_color([255, 0, 0])
    assert result.shape == (1, 4)
    assert result[0, 0] == pytest.approx(1.0)


def test_transform_color_tuple_1_scale():
    result = transform_color((0.5, 0.5, 0.5))
    assert result.shape == (1, 4)


def test_transform_color_ndarray_rgb():
    arr = np.array([1.0, 0.0, 0.0])
    result = transform_color(arr)
    assert result.shape == (4,)
    assert result[3] == 1.0  # alpha appended


def test_transform_color_ndarray_rgba():
    arr = np.array([1.0, 0.0, 0.0, 0.5])
    result = transform_color(arr)
    assert result is arr  # returned as-is


def test_transform_color_invalid_string():
    with pytest.raises(ValueError):
        transform_color("notacolor")


# ---------------------------------------------------------------------------
# find_text_color
# ---------------------------------------------------------------------------


def test_find_text_color_dark_background():
    # Very dark background → should return light text (white)
    result = find_text_color([0.0, 0.0, 0.0, 1.0])
    assert result == "white"


def test_find_text_color_light_background():
    # Very light background → should return dark text (black)
    result = find_text_color([1.0, 1.0, 1.0, 1.0])
    assert result == "black"


def test_find_text_color_custom_colors():
    result = find_text_color([1.0, 1.0, 1.0, 1.0], dark_color="navy", light_color="ivory")
    assert result == "navy"


def test_find_text_color_coef_options():
    # Both coefficient options should return the same answer for pure black/white
    for coef in (0, 1):
        assert find_text_color([0.0, 0.0, 0.0, 1.0], coef_choice=coef) == "white"
        assert find_text_color([1.0, 1.0, 1.0, 1.0], coef_choice=coef) == "black"
