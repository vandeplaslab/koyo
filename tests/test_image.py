import numpy as np
import pytest

from koyo.image import clip_hotspots, pearson_similarity


@pytest.mark.parametrize("quantile", (0.99, 0.75))
def test_clip_hotspots(quantile):
    img = np.random.default_rng().integers(0, 255, (50, 50))

    _img = clip_hotspots(img, quantile)
    np.testing.assert_array_equal(img.shape, _img.shape)


@pytest.mark.parametrize("size", ((3, 3), (5, 5)))
def test_pearson_similarity(size):
    img = np.random.default_rng().integers(0, 255, (50, 50))

    score = pearson_similarity(img, img, size)
    assert score == 1.0

    with pytest.raises(ValueError):
        pearson_similarity(img, img, (3, 3, 3))


def test_colocalization_matrix_matches_pairwise():
    pytest.importorskip("sklearn")
    from koyo.image import colocalization, colocalization_matrix

    rng = np.random.default_rng(0)
    images = rng.random((3, 20, 20))
    matrix = colocalization_matrix(images)
    for i in range(3):
        for j in range(3):
            assert matrix[i, j] == pytest.approx(colocalization(images[i], images[j]), abs=1e-6)


def test_clip_hotspots_does_not_modify_input():
    img = np.array([[1.0, np.inf], [3.0, np.nan]])
    original = img.copy()
    clip_hotspots(img)
    np.testing.assert_array_equal(img, original)


def test_reshape_array_batch_from_coordinates_matches_loop():
    from koyo.image import reshape_array_batch, reshape_array_batch_from_coordinates

    rng = np.random.default_rng(0)
    coordinates = np.array([[0, 0], [1, 2], [3, 1]])  # (x, y)
    array = rng.random((3, 4))
    result = reshape_array_batch_from_coordinates(array, (3, 4), coordinates)
    for i in range(4):
        expected = np.zeros((3, 4))
        expected[coordinates[:, 1], coordinates[:, 0]] = array[:, i]
        np.testing.assert_array_equal(result[i], expected)
    batch = reshape_array_batch(array, (3, 4), np.array([0, 5, 7]))
    assert batch.shape == (4, 3, 4)
    np.testing.assert_array_equal(batch.reshape(4, -1)[:, [0, 5, 7]], array.T)
