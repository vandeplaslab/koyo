import numpy as np
from koyo.rand import get_random_seed, temporary_seed


def test_get_random_seed():
    rand = get_random_seed()
    assert rand > 0


def test_temporary_seed():
    with temporary_seed(0):
        values_first = np.random.rand(4)
    with temporary_seed(0):
        values_second = np.random.rand(4)
    assert np.all(values_first == values_second)
