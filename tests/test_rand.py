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


def test_temporary_seed_is_reproducible():
    np.random.seed(5)
    np.random.rand()
    with temporary_seed(0):
        a = np.random.rand(3)
    np.random.rand(10)
    with temporary_seed(0):
        b = np.random.rand(3)
    np.testing.assert_array_equal(a, b)


def test_temporary_seed_different_seeds_differ():
    with temporary_seed(0):
        a = np.random.rand()
    with temporary_seed(1):
        b = np.random.rand()
    assert a != b
