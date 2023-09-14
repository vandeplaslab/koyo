"""Test color functions"""
import pytest
from koyo.color import get_random_hex_color, hex_to_rgb_1, hex_to_rgb_255, rgb_255_to_1


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
