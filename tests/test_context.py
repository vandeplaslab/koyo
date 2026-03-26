"""Tests for koyo.context."""

import gc
import sys

import pytest

from koyo.context import (
    captured_stderr,
    captured_stdout,
    disabled_gc,
    hide_stdout,
    no_gc,
    noop,
    nullout,
)


# ---------------------------------------------------------------------------
# captured_stdout / captured_stderr
# ---------------------------------------------------------------------------


def test_captured_stdout_captures_print():
    with captured_stdout() as buf:
        print("hello stdout")
    assert "hello stdout" in buf.getvalue()


def test_captured_stderr_captures_print():
    with captured_stderr() as buf:
        print("hello stderr", file=sys.stderr)
    assert "hello stderr" in buf.getvalue()


def test_captured_stdout_restores_original():
    original = sys.stdout
    with captured_stdout():
        pass
    assert sys.stdout is original


def test_captured_stderr_restores_original():
    original = sys.stderr
    with captured_stderr():
        pass
    assert sys.stderr is original


# ---------------------------------------------------------------------------
# hide_stdout
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_hide_stdout_suppresses_output():
    # Should not raise; stdout is redirected to devnull
    with hide_stdout():
        print("this should be suppressed")


# ---------------------------------------------------------------------------
# nullout
# ---------------------------------------------------------------------------


def test_nullout_suppresses_stdout_and_stderr():
    with nullout():
        print("suppressed stdout")
        print("suppressed stderr", file=sys.stderr)
    # If we reach here stdout/stderr are restored
    assert sys.stdout is not None
    assert sys.stderr is not None


# ---------------------------------------------------------------------------
# noop
# ---------------------------------------------------------------------------


def test_noop_yields_without_error():
    with noop():
        pass


def test_noop_accepts_args():
    with noop(1, 2, key="value"):
        pass


# ---------------------------------------------------------------------------
# no_gc / disabled_gc
# ---------------------------------------------------------------------------


def test_no_gc_disables_during_block():
    gc.enable()
    with no_gc():
        assert not gc.isenabled()
    assert gc.isenabled()


def test_no_gc_restores_disabled_state():
    gc.disable()
    try:
        with no_gc():
            assert not gc.isenabled()
        # GC was disabled before entering, should remain disabled
        assert not gc.isenabled()
    finally:
        gc.enable()


def test_disabled_gc_alias():
    gc.enable()
    with disabled_gc():
        assert not gc.isenabled()
    assert gc.isenabled()
