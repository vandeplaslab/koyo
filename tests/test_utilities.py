import pytest
import numpy as np
from koyo.utilities import format_size, get_min_max, check_image_orientation, rescale


@pytest.mark.parametrize(
    "values, new_min, new_max",
    (
        [np.arange(0, 1000), 0, 1],
        [np.arange(0, 100), 0, 1000],
        [np.exp([0, 1, 2]), 0, 1],
        [1 / np.exp([0, 1, 2]), 0, 1],
    ),
)
def test_rescale(values, new_min, new_max):
    results = rescale(values, new_min, new_max, dtype="float32")
    assert results.min() == new_min
    assert results.max() == new_max


@pytest.mark.parametrize(
    "values, expected",
    (
        [[0, 1, 2, 3, 4], [0, 4]],
        [np.arange(-4123, 1003), [-4123, 1002]],
        [(12, 324, 51, 45), [12, 324]],
    ),
)
def test_get_min_max(values, expected):
    result = get_min_max(values)
    assert len(result) == 2
    assert result[0] == expected[0]
    assert result[1] == expected[1]


@pytest.mark.parametrize("shape", [(2, 10), (10, 2)])
def test_check_image_orientation(shape):
    good_shape = (2, 10)
    zvals = check_image_orientation(np.zeros(shape))
    assert zvals.shape == good_shape


def test_format_size():
    assert "100" == format_size(100)
    assert "1.0K" == format_size(2**10)
    assert "1.0M" == format_size(2**20)
    assert "1.0G" == format_size(2**30)
    assert "1.0T" == format_size(2**40)
    assert "1.0P" == format_size(2**50)
