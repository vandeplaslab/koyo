"""Tests for koyo.logging (pure utility functions only)."""

import sys

import pytest

from koyo.logging import (
    get_loguru_config,
    get_loguru_level,
    get_stderr,
    get_stdout,
)

# ---------------------------------------------------------------------------
# get_loguru_level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "level_in, expected",
    [
        (0, "TRACE"),
        (5, "TRACE"),
        (10, "DEBUG"),
        (20, "INFO"),
        (25, "SUCCESS"),
        (30, "WARNING"),
        (40, "ERROR"),
        (50, "CRITICAL"),
        ("info", "INFO"),
        ("DEBUG", "DEBUG"),
        ("warning", "WARNING"),
    ],
)
def test_get_loguru_level(level_in, expected):
    assert get_loguru_level(level_in) == expected


def test_get_loguru_level_near_boundary():
    # 15 is between debug(10) and info(20); nearest is 10 → DEBUG
    result = get_loguru_level(15)
    assert result in ("DEBUG", "INFO")  # nearest-value pick


def test_get_loguru_level_above_max():
    # 60 > 50; nearest should be CRITICAL
    result = get_loguru_level(60)
    assert result == "CRITICAL"


# ---------------------------------------------------------------------------
# get_loguru_config
# ---------------------------------------------------------------------------


def test_get_loguru_config_no_color():
    level, _fmt, colorize, _enqueue = get_loguru_config(20, no_color=True)
    assert level == "INFO"
    assert colorize is False


def test_get_loguru_config_with_color():
    _level, _fmt, colorize, _enqueue = get_loguru_config(20, no_color=False)
    assert colorize is True


def test_get_loguru_config_level_string():
    level, _fmt, _colorize, _enqueue = get_loguru_config("warning", no_color=True)
    assert level == "WARNING"


def test_get_loguru_config_enqueue_default():
    _level, _fmt, _colorize, enqueue = get_loguru_config(10, no_color=True)
    # Default enqueue is True
    assert enqueue is True


# ---------------------------------------------------------------------------
# get_stdout / get_stderr
# ---------------------------------------------------------------------------


def test_get_stdout_returns_stream():
    stream = get_stdout()
    assert stream is not None
    assert hasattr(stream, "write")


def test_get_stderr_returns_stream():
    stream = get_stderr()
    assert stream is not None
    assert hasattr(stream, "write")


def test_get_stdout_is_sys_stdout():
    assert get_stdout() is sys.stdout


def test_get_stderr_is_sys_stderr():
    assert get_stderr() is sys.stderr


def test_timed_call_is_context_manager():
    from koyo.logging import timed_call

    messages = []
    with timed_call(messages.append, "done"):
        pass
    assert len(messages) == 1
    assert messages[0].startswith("done in ")


def test_get_loguru_env_parses_false(monkeypatch):
    from koyo.logging import get_loguru_env

    monkeypatch.setenv("LOGURU_COLORIZE", "False")
    monkeypatch.setenv("LOGURU_ENQUEUE", "False")
    _, _, enqueue, colorize = get_loguru_env()
    assert enqueue is False
    assert colorize is False


def test_set_logger_passes_enqueue_and_colorize_in_order(monkeypatch):
    from unittest.mock import MagicMock

    import koyo.logging as klog

    calls = {}
    monkeypatch.setattr(
        klog,
        "set_loguru_env",
        lambda fmt, level, enqueue, colorize: calls.update(
            enqueue=enqueue,
            colorize=colorize,
        ),
    )
    monkeypatch.setattr(klog, "set_loguru_log", lambda *args, **kwargs: None)
    klog.set_logger(verbosity=2, no_color=True, logger=MagicMock())
    assert calls == {"enqueue": True, "colorize": False}
