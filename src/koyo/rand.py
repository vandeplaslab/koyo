from contextlib import contextmanager

import numpy as np


def get_random_seed() -> int:
    """Get random seed."""
    return np.random.randint(0, np.iinfo(np.int32).max - 1, 1)[0]


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
