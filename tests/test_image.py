import numpy as np
from koyo.image import clip_hotspots


def test_clip_hotspots():
    image = np.random.random((10, 10))

    image2 = clip_hotspots(image)
    assert image2.shape == image.shape
