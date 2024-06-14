"""Logging utilities."""

from __future__ import annotations

import os
import sys
import typing as ty
from functools import partial

from koyo.utilities import find_nearest_value

if ty.TYPE_CHECKING:
    from loguru import Logger


LEVEL_FORMAT = "<level>{level: <8}</level>"
TIME_FORMAT = "{time:YYYY-MM-DD HH:mm:ss:SSS}"
LOG_FMT = "[LEVEL_FORMAT][TIME_FORMAT][{process}] {message}".replace("TIME_FORMAT", TIME_FORMAT).replace(
    "LEVEL_FORMAT", LEVEL_FORMAT
)
COLOR_LOG_FMT = (
    "<green>[LEVEL_FORMAT]</green>" "<cyan>[TIME_FORMAT]</cyan>" "<red>[{process}]</red>" " {message}".replace(
        "TIME_FORMAT", TIME_FORMAT
    ).replace("LEVEL_FORMAT", LEVEL_FORMAT)
)


def timed_call(log_func: ty.Callable, message: str, *args: ty.Any, **kwargs: ty.Any) -> ty.Any:
    """Time a function call."""
    from koyo.timer import MeasureTimer

    with MeasureTimer() as timer:
        yield
    log_func(message + f" in {timer()}", *args, **kwargs)


def get_logger() -> Logger:
    """Get logger with extra functions."""
    from loguru import logger

    logger.timed_trace = partial(timed_call, logger.trace)
    logger.timed_debug = partial(timed_call, logger.debug)
    logger.timed_info = partial(timed_call, logger.info)
    logger.timed_success = partial(timed_call, logger.success)
    logger.timed_warning = partial(timed_call, logger.warning)
    logger.timed_error = partial(timed_call, logger.error)
    logger.timed_critical = partial(timed_call, logger.critical)
    logger.timed_exception = partial(timed_call, logger.exception)
    return logger


def set_loguru_env(fmt: str, level: str, enqueue: bool, colorize: bool):
    """Set loguru environment variables."""
    os.environ["LOGURU_AUTOINIT"] = "0"
    os.environ["LOGURU_FORMAT"] = str(fmt)
    os.environ["LOGURU_LEVEL"] = str(level)
    os.environ["LOGURU_ENQEUE"] = str(enqueue)
    os.environ["LOGURU_COLORIZE"] = str(colorize)


def get_loguru_level(level: int | str) -> str:
    """Get loguru level."""
    if isinstance(level, str):
        return level.upper()
    levels = {
        0: "trace",
        5: "trace",
        10: "debug",
        20: "info",
        25: "success",
        30: "warning",
        40: "error",
        50: "critical",
    }
    if level not in levels:
        level = find_nearest_value(list(levels.keys()), level)
    level = levels[level]
    return level.upper()


def get_loguru_env() -> tuple[str, str, bool, bool]:
    """Get logoru environment variables."""
    colorize = bool(os.environ.get("LOGURU_COLORIZE", "True"))
    level = os.environ.get("LOGURU_LEVEL", "info")
    enqueue = bool(os.environ.get("LOGURU_ENQEUE", "True"))
    fmt = os.environ.get("LOGURU_FORMAT", LOG_FMT if not colorize else COLOR_LOG_FMT)
    return fmt, level, enqueue, colorize


def get_loguru_config(
    level: str | int | float,
    no_color: bool,
    enqueue: bool = True,
    fmt: str = LOG_FMT,
    color_fmt: str = COLOR_LOG_FMT,
) -> tuple[str, str, bool, bool]:
    """Return level."""
    level = get_loguru_level(level)
    colorize = not no_color
    fmt = fmt if no_color else color_fmt
    return level.upper(), fmt, colorize, enqueue


def get_stderr() -> ty.TextIO:
    """Get stderr."""
    if sys.stderr is None:
        sys.stderr = sys.stdout
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    return sys.stderr


def set_loguru_log(
    sink=sys.stderr,
    level: str | int = 20,
    no_color: bool = False,
    enqueue: bool = True,
    fmt: str | None = None,
    diagnose: bool = False,
    catch: bool = False,
    colorize: bool | None = None,
    remove: bool = True,
    logger: Logger = None,
):
    """Set loguru formatting."""
    if logger is None:
        from loguru import logger
    if logger is None:
        raise ValueError("Logger is None - cannot set loguru log.")
    if sink is None:
        sink = get_stderr()
    if sink is None:
        raise ValueError("Sink is None - cannot set loguru log.")

    # automatically get format
    fmt = fmt if fmt is not None else (LOG_FMT if no_color else COLOR_LOG_FMT)
    level = get_loguru_level(level)

    if remove:
        logger.remove(None)
    logger.add(sink, level=level, format=fmt, colorize=not no_color, enqueue=enqueue, diagnose=diagnose, catch=catch)
