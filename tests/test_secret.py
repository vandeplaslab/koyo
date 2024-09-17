"""Test secret"""

import numpy as np
import pytest

from koyo.secret import get_short_hash, get_unique_str, hash_iterable, hash_parameters


def test_get_unique_str():
    value = get_unique_str()
    assert isinstance(value, str)


@pytest.mark.parametrize("n", (6, 10))
def test_get_short_hash(n):
    value = get_short_hash(n)
    assert isinstance(value, str)
    assert len(value) == n


@pytest.mark.xfail
def test_hash_iterable():
    assert (
        hash_iterable([0, 1, 2]) == hash_iterable(np.arange(3)) == hash_iterable([2, 1, 0]) == hash_iterable((2, 1, 0))
    )


def test_hash_parameters():
    res = hash_parameters(a=1, b=2, c=3)
    assert isinstance(res, str)
    assert hash_parameters(a=[0, 1]) == hash_parameters(a=[1, 0]) == hash_parameters(a=(0, 1))
