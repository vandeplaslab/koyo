"""Various context managers."""

import gc
import io
import os
import pathlib
import sys
from contextlib import contextmanager, redirect_stdout


@contextmanager
def captured_output(stream_name):
    """Return a context manager used by captured_stdout/stdin/stderr
    that temporarily replaces the sys stream *stream_name* with a StringIO.
    """
    orig_stdout = getattr(sys, stream_name)
    setattr(sys, stream_name, io.StringIO())
    try:
        yield getattr(sys, stream_name)
    finally:
        setattr(sys, stream_name, orig_stdout)


def captured_stdout():
    r"""Capture the output of sys.stdout:

    with captured_stdout() as stdout:
        print("hello")
    self.assertEqual(stdout.getvalue(), "hello\n")
    """
    return captured_output("stdout")


def captured_stderr():
    r"""Capture the output of sys.stderr:

    with captured_stderr() as stderr:
        print("hello", file=sys.stderr)
    self.assertEqual(stderr.getvalue(), "hello\n")
    """
    return captured_output("stderr")


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
        sys.stdout.close()
        sys.stderr.close()
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
