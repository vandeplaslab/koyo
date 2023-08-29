import numpy as np
import pytest
from koyo.image import clip_hotspots, pearson_similarity


@pytest.mark.parametrize("quantile", (0.99, 0.75))
def test_clip_hotspots(quantile):
    img = np.random.randint(0, 255, (50, 50))

    _img = clip_hotspots(img, quantile)
    np.testing.assert_array_equal(img.shape, _img.shape)


@pytest.mark.parametrize("size", ((3, 3), (5, 5)))
def test_pearson_similarity(size):
    img = np.random.randint(0, 255, (50, 50))

    score = pearson_similarity(img, img, size)
    assert score == 1.0

    with pytest.raises(ValueError):
        pearson_similarity(img, img, (3, 3, 3))
