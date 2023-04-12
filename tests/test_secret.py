"""Test secret"""
import numpy as np
from koyo.secret import get_short_hash, hash_iterable


def test_get_short_hash():
    value = get_short_hash()
    assert isinstance(value, str)


def test_hash_iterable():
    assert (
        hash_iterable([0, 1, 2])
        == hash_iterable(np.arange(3))
        == hash_iterable([2, 1, 0])
        == hash_iterable((2, 1, 0))
    )
