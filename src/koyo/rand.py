import numpy as np
from contextlib import contextmanager


__all__ = [
    "get_random_seed",
    "temporary_seed",
]


def get_random_seed():
    """Get random seed."""
    return np.random.randint(0, 100000, (1,))[0]


@contextmanager
def temporary_seed(seed: int):
    """Temporarily set numpy seed."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
