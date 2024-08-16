"""System utilities."""
import os
import platform
import sys

IS_WIN = sys.platform == "win32"
IS_LINUX = sys.platform == "linux"
IS_MAC = sys.platform == "darwin"
IS_MAC_ARM = IS_MAC and platform.processor() == "arm"
IS_PYINSTALLER = getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def is_envvar(key: str, value: str) -> bool:
    """Check if an environment variable is set."""
    return key in os.environ and os.environ[key] == str(value)


def is_envvar_set(key: str) -> bool:
    """Check if an environment variable is set."""
    return key in os.environ and os.environ[key]
