"""System utilities."""

import os
import platform
import sys
from pathlib import Path

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


def get_cli_path(name: str, env_key: str = "") -> str:
    """Get path to imimspy executable.

    The path is determined in the following order:
    1. First, we check whether environment variable `{env_key}_{name.upper()}_PATH` is set.
    2. If not, we check whether we are running as a PyInstaller app.
    3. If not, we check whether we are running as a Python app.
    4. If not, we raise an error.
    """
    import os
    import sys

    from koyo.utilities import running_as_pyinstaller_app

    env_var = f"{env_key}_{name.upper()}_PATH"
    if os.environ.get(env_var, None):
        script_path = Path(os.environ[env_var])
        if script_path.exists():
            return str(script_path)

    base_path = Path(sys.executable).parent
    if running_as_pyinstaller_app():
        if IS_WIN:
            script_path = base_path / f"{name}.exe"
        elif IS_MAC or IS_LINUX:
            script_path = base_path / name
        else:
            raise NotImplementedError(f"Unsupported OS: {sys.platform}")
        if script_path.exists():
            return str(script_path)
    else:
        # on Windows, {name} lives under the `Scripts` directory
        if IS_WIN:
            script_path = base_path
            if script_path.name != "Scripts":
                script_path = base_path / "Scripts"
            if script_path.exists() and (script_path / f"{name}.exe").exists():
                return str(script_path / f"{name}.exe")
        elif IS_MAC or IS_LINUX:
            script_path = base_path / name
            if script_path.exists():
                return str(script_path)
        else:
            script_path = base_path / f"{name}.exe"
            if script_path.exists():
                return str(script_path)
    raise RuntimeError(f"Could not find '{name}' executable.")