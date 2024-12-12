"""Various context managers."""

import gc
import os
import pathlib
import sys
from contextlib import contextmanager, redirect_stdout


@contextmanager
def fix_pathlib():
    """Fix pathlib on Windows."""
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup


@contextmanager
def hide_stdout():
    """Hide stdout."""
    with redirect_stdout(open(os.devnull, "w")):
        yield


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
def noop(*args, **kwargs):
    """Function."""
    yield


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
