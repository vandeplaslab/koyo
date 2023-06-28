"""Logging utilities."""
import os
import sys
import typing as ty

if ty.TYPE_CHECKING:
    from loguru import Logger


LEVEL_FORMAT = "<level>{level: <8}</level>"
TIME_FORMAT = "{time:YYYY-MM-DD HH:mm:ss:SSS}"
LOG_FMT = "[LEVEL_FORMAT][TIME_FORMAT][{process}] {message}".replace("TIME_FORMAT", TIME_FORMAT).replace(
    "LEVEL_FORMAT", LEVEL_FORMAT
)
COLOR_LOG_FMT = (
    "<green>[LEVEL_FORMAT]</green>"
    "<cyan>[TIME_FORMAT]</cyan>"
    "<red>[{process}]</red>"
    " {message}".replace("TIME_FORMAT", TIME_FORMAT).replace("LEVEL_FORMAT", LEVEL_FORMAT)
)


def set_loguru_env(fmt: str, level: str, enqueue: bool, colorize: bool):
    """Set loguru environment variables."""
    os.environ["LOGURU_AUTOINIT"] = "0"
    os.environ["LOGURU_FORMAT"] = str(fmt)
    os.environ["LOGURU_LEVEL"] = str(level)
    os.environ["LOGURU_ENQEUE"] = str(enqueue)
    os.environ["LOGURU_COLORIZE"] = str(colorize)


def get_loguru_level(level: ty.Union[int, str]):
    """Get loguru level."""
    if isinstance(level, str):
        return level.upper()
    level = {
        0: "trace",
        5: "trace",
        10: "debug",
        20: "info",
        25: "success",
        30: "warning",
        40: "error",
        50: "critical",
    }[level]
    return level.upper()


def get_loguru_env():
    """Get logoru environment variables."""
    colorize = bool(os.environ.get("LOGURU_COLORIZE", "True"))
    level = os.environ.get("LOGURU_LEVEL", "info")
    enqueue = bool(os.environ.get("LOGURU_ENQEUE", "True"))
    fmt = os.environ.get("LOGURU_FORMAT", LOG_FMT if not colorize else COLOR_LOG_FMT)
    return fmt, level, enqueue, colorize


def get_loguru_config(
    level: ty.Union[str, int], no_color: bool, enqueue: bool = True, fmt: str = LOG_FMT, color_fmt: str = COLOR_LOG_FMT
) -> ty.Tuple[str, str, bool, bool]:
    """Return level."""
    level = get_loguru_level(level)
    colorize = not no_color
    fmt = fmt if no_color else color_fmt
    return level.upper(), fmt, colorize, enqueue


def set_loguru_log(
    sink=sys.stderr,
    level: ty.Union[str, int] = 20,
    no_color: bool = False,
    enqueue: bool = True,
    fmt: ty.Optional[str] = None,
    diagnose: bool = False,
    catch: bool = False,
    colorize: ty.Optional[bool] = None,
    remove: bool = True,
    logger: "Logger" = None,
):
    """Set loguru formatting."""
    if logger is None:
        from loguru import logger

    # automatically get format
    fmt = fmt if fmt is not None else (LOG_FMT if no_color else COLOR_LOG_FMT)
    level = get_loguru_level(level)

    if remove:
        logger.remove(None)
    logger.add(sink, level=level, format=fmt, colorize=not no_color, enqueue=enqueue, diagnose=diagnose, catch=catch)
