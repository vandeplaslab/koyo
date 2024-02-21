"""Various context managers."""
import gc
import os
import sys
from contextlib import contextmanager


@contextmanager
def nullout():
    """Context manager to suppress stdout and stderr."""
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout = save_stdout
        sys.stderr = save_stderr


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
