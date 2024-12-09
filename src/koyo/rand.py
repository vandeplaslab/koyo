"""Random number utilities."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np


def get_random_seed() -> int:
    """Get random seed."""
    return np.random.randint(0, np.iinfo(np.int32).max - 1, 1)[0]


def get_random_state(n: int = 1) -> int | list[int]:
    """Retrieve random state(s)."""
    from random import randint

    state = [randint(0, 2**32 - 1) for _ in range(n)]
    return state if n > 1 else state[0]


@contextmanager
def temporary_seed(seed: int, skip_if_negative_one: bool = False):
    """Temporarily set numpy seed."""
    if skip_if_negative_one and seed == -1:
        yield
        return
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
