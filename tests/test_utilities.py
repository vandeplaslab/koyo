import numpy as np
import pytest
from numpy.testing import assert_equal

from koyo.utilities import (
    check_image_orientation,
    check_value_order,
    chunks,
    find_nearest_index,
    find_nearest_index_batch,
    find_nearest_index_single,
    find_nearest_value_single,
    format_size,
    get_kws,
    get_min_max,
    get_pixels_within_radius,
    is_between,
    is_installed,
    is_number,
    rescale,
    split_tooltip,
    view_as_blocks,
)


def test_is_installed():
    assert is_installed("numpy")
    assert is_installed("numpy.linalg")
    assert not is_installed("not_installed_package")


def test_split_tooltip():
    text = "This is a test of different maximum widths"
    # Test with narrow width
    narrow = split_tooltip(text, 10)
    assert all(len(line) <= 10 for line in narrow.split("\n"))


def test_get_pixels_within_radius():
    array = np.arange(25).reshape(5, 5)
    x, y = 0, 0
    res = get_pixels_within_radius(array, x, y, 0)
    assert res == 0
    res = get_pixels_within_radius(array, x, y, 1)
    np.testing.assert_array_equal(res, [0, 1, 5, 6])

    x, y = 2, 2
    res = get_pixels_within_radius(array, x, y, 1)
    np.testing.assert_array_equal(res, [6, 7, 8, 11, 12, 13, 16, 17, 18])

    x, y = 4, 4
    res = get_pixels_within_radius(array, x, y, 1)
    np.testing.assert_array_equal(res, [18, 19, 23, 24])

    res = get_pixels_within_radius(array, x, y, 10)
    assert len(res) == 25


class TestViewAsBlocks:
    @staticmethod
    def test_wrong_block_dimension():
        A = np.arange(10)
        with pytest.raises(ValueError):
            view_as_blocks(A, 2, 2)

    @staticmethod
    def test_2d_array():
        array_a = np.arange(4 * 4).reshape(4, 4)
        array_b, _ = view_as_blocks(array_a, 2, 2)
        assert_equal(array_b[0], np.array([[0, 1], [4, 5]]))
        assert len(array_b) == 4

    @staticmethod
    @pytest.mark.parametrize("n_rows, n_cols", ((2, 3), (3, 2), (3, 3)))
    def test_2d_array_with_pad(n_rows, n_cols):
        array_a = np.arange(4 * 4).reshape(4, 4)
        array_b, _ = view_as_blocks(array_a, n_rows, n_cols)
        assert len(array_b) == 4

    @staticmethod
    @pytest.mark.parametrize("n_rows, n_cols", ((2, 3), (3, 2), (3, 3)))
    def test_2d_array_wrong_dimensions(n_rows, n_cols):
        array = np.arange(4 * 4).reshape(4, 4)

        with pytest.raises(ValueError):
            view_as_blocks(array, n_rows, n_cols, False)


def test_chunks():
    values = [0, 1, 2, 3, 4, 5]
    for chunk in chunks(values, 2):
        assert len(chunk) == 2
    assert len(list(chunks(values, n_tasks=2))) == 2


def test_find_nearest_index_single():
    array = np.arange(10)
    assert find_nearest_index_single(array, 0) == 0
    assert find_nearest_index_single(array, 0.5) == 0

    assert find_nearest_index(array, 0) == 0
    assert find_nearest_index(array, 0.5) == 0

    assert find_nearest_index_single(array, 9.5) == find_nearest_index(array, 9.5)


def test_find_nearest_index_batch():
    array = np.arange(10)
    indices = find_nearest_index_batch(array, [0, 0.5, 9.5])
    np.all(indices == [0, 0, 9])


def test_find_nearest_value_single():
    array = np.arange(10)
    assert find_nearest_value_single(array, 0) == 0
    assert find_nearest_value_single(array, 0.5) == 0
    assert find_nearest_value_single(array, 10) == 9
    assert find_nearest_value_single(array, 100) == 9


def test_is_number():
    assert is_number(1)
    assert is_number(1.0)
    assert not is_number("1")
    assert not is_number([1])
    assert not is_number((1,))


def test_get_kws():
    def test_func(a=1, b=2):
        pass

    kws = get_kws(test_func, a=2, c=3)
    assert "c" not in kws


def test_check_value_order():
    v1, v2 = 10, 20
    r1, r2 = check_value_order(v1, v2)
    assert r1 == v1
    assert r2 == v2

    v1, v2 = 100, 50
    r1, r2 = check_value_order(v1, v2)
    assert r1 == v2
    assert r2 == v1


def test_is_between():
    assert is_between(0, 0, 1)
    assert is_between(0.5, 0, 1)
    assert is_between(1, 0, 1)
    assert not is_between(1.1, 0, 1)
    assert not is_between(-0.1, 0, 1)


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
    assert format_size(100) == "100"
    assert format_size(2**10) == "1.0K"
    assert format_size(2**20) == "1.0M"
    assert format_size(2**30) == "1.0G"
    assert format_size(2**40) == "1.0T"
    assert format_size(2**50) == "1.0P"


def test_find_nearest_index_batch_unsorted_with_sort():
    array = np.array([3.0, 1.0, 2.0, 10.0, 0.0])
    values = np.array([2.1, 0.2, 9.0, 3.4])
    result = find_nearest_index_batch(array, values, sort=True)
    expected = [np.argmin(np.abs(array - v)) for v in values]
    assert_equal(result, expected)


def test_filter_array_preserves_order():
    from koyo.utilities import filter_array

    array = np.arange(100_000, dtype=np.float64)
    result = filter_array(array, array, 10.0, 20_000.0)
    assert_equal(result, array[(array >= 10) & (array <= 20_000)])


def test_get_pixels_within_radius_non_square():
    grid = np.arange(1000, dtype=float).reshape(100, 10)
    result = get_pixels_within_radius(grid, x=5, y=50, radius=1)
    assert sorted(result) == sorted(grid[49:52, 4:7].ravel())


def test_rescale_constant_and_int_dtype():
    assert_equal(rescale(np.ones(3), 0, 255, dtype=np.uint8), [0, 0, 0])
    assert_equal(rescale([0.2, 0.4, 0.6], 0, 100, dtype=np.uint8), [0, 50, 100])


def test_get_distributed_list_int_one_is_single_frame():
    from koyo.utilities import get_distributed_list

    assert len(get_distributed_list(10, 1, as_index=True)) == 1
    assert len(get_distributed_list(10, 1.0, as_index=True)) == 10
    assert len(get_distributed_list(10, 0.3, as_index=True, trim=True)) == 3
    assert len(get_distributed_list(10, 0, as_index=True)) == 0


def test_split_array_numpy_and_dataframe():
    from koyo.utilities import split_array

    arr = np.arange(20).reshape(10, 2)
    assert [len(c) for c in split_array(arr, 4)] == [4, 4, 2]
    assert [c.shape[1] for c in split_array(arr.T, 4, axis=1)] == [4, 4, 2]
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(arr)
    assert [len(c) for c in split_array(df, 4)] == [4, 4, 2]


def test_find_nearest_index_batch_sort_matches_unsorted_input():
    array = np.array([5.0, 1.0, 3.0])
    assert_equal(find_nearest_index_batch(array, [2.9, 4.9, 0.0], sort=True), [2, 0, 1])
