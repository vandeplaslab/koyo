from contextlib import contextmanager

import numpy as np


def get_random_seed() -> int:
    """Get random seed."""
    return np.random.randint(0, np.iinfo(np.int32).max - 1, 1)[0]


@contextmanager
def temporary_seed(seed: int):
    """Temporarily set numpy seed."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
