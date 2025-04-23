"""System utilities."""

from __future__ import annotations

import contextlib
import inspect
import os
import platform
import subprocess
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


def set_freeze_support() -> None:
    """Set freeze support for multiprocessing."""
    import sys
    from multiprocessing import freeze_support, set_start_method

    freeze_support()
    if sys.platform == "darwin":
        set_start_method("spawn", True)


def get_cli_path(name: str, env_key: str = "", default: str = "") -> str:
    """Get path to imimspy executable.

    The path is determined in the following order:
    1. First, we check whether environment variable `{env_key}_{name.upper()}_PATH` is set.
    2. If not, we check whether we are running as a PyInstaller app.
    3. If not, we check whether we are running as a Python app.
    4. If not, we raise an error.
    """
    import os
    import sys
    from pathlib import Path

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
    if default:
        return default
    raise RuntimeError(f"Could not find '{name}' executable.")


def who_called_me() -> tuple[str, str, int]:
    """Get the file name, function name, and line number of the caller."""
    # Get the current frame
    current_frame = inspect.currentframe()
    # Get the caller's frame
    caller_frame = current_frame.f_back

    # Extract file name, line number, and function name
    file_name = caller_frame.f_code.co_filename
    line_number = caller_frame.f_lineno
    function_name = caller_frame.f_code.co_name
    print(f"Called from file: {file_name}, function: {function_name}, line: {line_number}")
    return file_name, function_name, line_number


def who_called_me_stack(n: int = 6) -> None:
    """Print the last 5 callers that led to this function being called."""
    stack = inspect.stack()
    # The stack list starts with the current frame at index 0,
    # so the callers start from index 1 onward.
    # We will retrieve up to 5 callers, or fewer if the stack isn't that deep.
    limit = min(n, len(stack))  # 1 (immediate caller) + up to 5 total callers
    for i in range(1, limit):
        frame_info = stack[i]
        file_name = frame_info.filename
        function_name = frame_info.function
        line_number = frame_info.lineno
        print(f"Caller #{i}: File '{file_name}', Function '{function_name}', Line {line_number}")


def _linux_sys_name() -> str:
    """
    Try to discover linux system name base on /etc/os-release file or lsb_release command output
    https://www.freedesktop.org/software/systemd/man/os-release.html.
    """
    os_release_path = "/etc/os-release"

    if os.path.exists(os_release_path):
        with open(os_release_path) as f_p:
            data = {}
            for line in f_p:
                field, value = line.split("=")
                data[field.strip()] = value.strip().strip('"')
        if "PRETTY_NAME" in data:
            return data["PRETTY_NAME"]
        if "NAME" in data:
            if "VERSION" in data:
                return f"{data['NAME']} {data['VERSION']}"
            if "VERSION_ID" in data:
                return f"{data['NAME']} {data['VERSION_ID']}"
            return f"{data['NAME']} (no version)"
    return _linux_sys_name_lsb_release()


def _linux_sys_name_lsb_release() -> str:
    """Try to discover linux system name base on lsb_release command output."""
    with contextlib.suppress(subprocess.CalledProcessError):
        res = subprocess.run(["lsb_release", "-d", "-r"], check=True, capture_output=True)
        text = res.stdout.decode()
        data = {}
        for line in text.split("\n"):
            key, val = line.split(":")
            data[key.strip()] = val.strip()
        version_str = data["Description"]
        if not version_str.endswith(data["Release"]):
            version_str += " " + data["Release"]
        return version_str
    return ""


def _sys_name() -> str:
    """Discover MacOS or Linux Human readable information. For Linux provide information about distribution."""
    with contextlib.suppress(Exception):
        if sys.platform == "linux":
            return _linux_sys_name()
        if sys.platform == "darwin":
            with contextlib.suppress(subprocess.CalledProcessError):
                res = subprocess.run(
                    ["sw_vers", "-productVersion"],
                    check=True,
                    capture_output=True,
                )
                return f"MacOS {res.stdout.decode().strip()}"
    return ""


def get_module_path(module: str, filename: str) -> str:
    """Get module path."""
    import importlib.resources

    if not filename.endswith(".py"):
        filename += ".py"

    path = str(importlib.resources.files(module).joinpath(filename))
    return path
