"""Various context managers."""
import gc
from contextlib import contextmanager


@contextmanager
def no_gc():
    """Context manager to disable garbage collection."""
    enable = gc.isenabled()
    try:
        gc.disable()
        yield
    finally:
        if enable:
            gc.collect(0)
            gc.enable()


disabled_gc = no_gc
