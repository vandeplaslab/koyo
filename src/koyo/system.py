"""System utilities."""

from __future__ import annotations

import contextlib
import inspect
import os
import platform
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from loguru import logger

IS_WIN = sys.platform == "win32"
IS_LINUX = sys.platform == "linux"
IS_MAC = sys.platform == "darwin"
IS_MAC_ARM = IS_MAC and platform.processor() == "arm"
IS_PYINSTALLER = getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


@lru_cache
def check_available_space(path: str, min_required) -> bool:
    """Check available disk space at a specified path.

    Parameters
    ----------
    path : str
        Filesystem path to check (the drive/mount point is derived from it).
    min_required : int
        Minimum number of free bytes required.

    Returns
    -------
    bool
        ``True`` if free space exceeds *min_required*, or if the drive cannot
        be determined (e.g. on a path without a drive letter).
    """
    from psutil import disk_usage

    path = Path(path).drive
    if path:
        hdd = disk_usage(path)
        return hdd.free > min_required
    return True


def is_envvar(key: str, value: str) -> bool:
    """Check whether an environment variable is set to a specific value.

    Parameters
    ----------
    key : str
        Name of the environment variable.
    value : str
        Expected value.

    Returns
    -------
    bool
        ``True`` if the variable exists and equals *value*.
    """
    return key in os.environ and os.environ[key] == str(value)


def is_envvar_set(key: str) -> bool:
    """Check whether an environment variable exists and has a non-empty value.

    Parameters
    ----------
    key : str
        Name of the environment variable.

    Returns
    -------
    bool
        ``True`` if the variable is present and its value is not empty.
    """
    return key in os.environ and os.environ[key]


def set_freeze_support() -> None:
    """Enable multiprocessing freeze support for PyInstaller bundles.

    Calls ``multiprocessing.freeze_support()`` and, on macOS, sets the
    start method to ``"spawn"`` to avoid fork-related crashes.
    """
    import sys
    from multiprocessing import freeze_support, set_start_method

    freeze_support()
    if sys.platform == "darwin":
        set_start_method("spawn", True)


def get_cli_path(name: str, env_key: str = "", default: str = "") -> str:
    """Get a path to a named executable.

    The path is determined in the following order:

    1. Check whether environment variable ``{env_key}_{name.upper()}_PATH`` is set.
    2. Check whether we are running as a PyInstaller bundle.
    3. Check whether the executable is on the Python environment's PATH.
    4. Raise ``RuntimeError`` if the executable cannot be found.
    """
    from koyo.system import running_as_pyinstaller_app

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
    raise RuntimeError(f"Could not find '{name}' executable.")  # noqa: TRY003


def who_called_me() -> tuple[str, str, int]:
    """Return the file, function, and line number of the immediate caller.

    Returns
    -------
    file_name : str
        Absolute path to the source file of the caller.
    function_name : str
        Name of the calling function.
    line_number : int
        Line number within *file_name* where this function was invoked.

    Notes
    -----
    Also prints the caller information to stdout as a side-effect.
    """
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
    """Print the call stack up to *n* frames above this function.

    Parameters
    ----------
    n : int, optional
        Maximum number of caller frames to display (default 6).
    """
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
    print("\n")


def _linux_sys_name() -> str:
    """Try to discover linux system name based on /etc/os-release file or lsb_release command output.

    Notes
    -----
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
    """Try to discover linux system name based on lsb_release command output."""
    with contextlib.suppress(subprocess.CalledProcessError):
        res = subprocess.run(["lsb_release", "-d", "-r"], check=True, capture_output=True)  # noqa: S603,S607
        text = res.stdout.decode()
        data = {}
        for line in text.split("\n"):
            if ":" not in line:
                continue
            key, val = line.split(":")
            data[key.strip()] = val.strip()
        version_str = data["Description"]
        if not version_str.endswith(data["Release"]):
            version_str += " " + data["Release"]
        return version_str
    return ""


def _sys_name() -> str:
    """Discover a MacOS or Linux Human readable information. For Linux provide information about distribution."""
    with contextlib.suppress(Exception):
        if sys.platform == "linux":
            return _linux_sys_name()
        if sys.platform == "darwin":
            with contextlib.suppress(subprocess.CalledProcessError):
                res = subprocess.run(  # S603
                    ["sw_vers", "-productVersion"],  # noqa: S607
                    check=True,
                    capture_output=True,
                )
                return f"MacOS {res.stdout.decode().strip()}"
    return ""


def get_module_path(module: str, filename: str) -> str:
    """Return the filesystem path to a file inside an installed package.

    Parameters
    ----------
    module : str
        Importable package name (e.g. ``"koyo"``).
    filename : str
        Filename within the package, with or without the ``.py`` extension.

    Returns
    -------
    str
        Absolute path to the requested file.
    """
    import importlib.resources

    if not filename.endswith(".py"):
        filename += ".py"

    return str(importlib.resources.files(module).joinpath(filename))


def running_as_pyinstaller_app() -> bool:
    """Return ``True`` if the process is running inside a PyInstaller bundle."""
    import sys

    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def reraise_exception_if_debug(exc, message: str = "Exception occurred", env_key: str = "DEV_MODE") -> None:
    """Re-raise *exc* when a debug env-var is set, otherwise log it.

    Parameters
    ----------
    exc : BaseException
        The exception to handle.
    message : str, optional
        Message passed to ``logger.exception`` when not re-raising.
    env_key : str, optional
        Environment variable name whose value ``"1"`` enables re-raise
        (default ``"DEV_MODE"``).
    """
    import os

    if os.environ.get(env_key, "0") == "1":
        raise exc
    logger.exception(message)


@lru_cache(maxsize=4)
def running_as_briefcase_app() -> bool:
    """Infer whether we are running as a briefcase bundle."""
    import importlib_metadata

    # https://github.com/beeware/briefcase/issues/412
    # https://github.com/beeware/briefcase/pull/425
    # note that a module may not have a __package__ attribute
    try:
        app_module = sys.modules["__main__"].__package__
    except AttributeError:
        return False
    try:
        metadata = importlib_metadata.metadata(app_module)
    except importlib_metadata.PackageNotFoundError:
        return False
    return "Briefcase-Version" in metadata


def is_installed(module: str) -> bool:
    """Check whether a Python module is available without importing it.

    Parameters
    ----------
    module : str
        Fully-qualified module name (e.g. ``"numpy.linalg"``).

    Returns
    -------
    bool
        ``True`` if the module can be found on ``sys.path``.
    """
    import importlib.util

    try:
        loader = importlib.util.find_spec(module)
    except ModuleNotFoundError:
        return False
    return loader is not None


def get_version(module: str) -> str:
    """Return the installed version of a package.

    Parameters
    ----------
    module : str
        Package name as known to ``importlib.metadata``.

    Returns
    -------
    str
        Version string, or ``"N/A"`` if the package is not installed.
    """
    import importlib.metadata

    try:
        installed_version = importlib.metadata.version(module)
    except importlib.metadata.PackageNotFoundError:
        return "N/A"
    return installed_version


def is_above_version(module: str, version: str) -> bool:
    """Check whether an installed package meets a minimum version requirement.

    Parameters
    ----------
    module : str
        Package name as known to ``importlib.metadata``.
    version : str
        Minimum required version string (e.g. ``"1.20.0"``).

    Returns
    -------
    bool
        ``True`` if the installed version is >= *version*, ``False`` if the
        package is not installed or the version is lower.
    """
    import importlib.metadata

    from packaging.version import Version

    try:
        installed_version = importlib.metadata.version(module)
    except importlib.metadata.PackageNotFoundError:
        logger.warning(f"Module {module} not found.")
        return False
    installed_version = Version(installed_version)
    version = Version(version)
    if installed_version is not None:
        return installed_version >= version
    logger.warning(f"Module {module} not found.")
    return False


def is_network_path(path: str | os.PathLike) -> bool:
    """Return True if *path* lives on a network / remote filesystem.

    Works on Windows, macOS, and Linux without third-party dependencies.
    """
    path = os.path.abspath(path)

    if sys.platform == "win32":
        return _is_network_windows(path)
    if sys.platform == "darwin":
        return _is_network_macos(path)
    return _is_network_linux(path)


def _is_network_windows(path: str) -> bool:
    import ctypes

    # UNC paths (\\server\share\...) are always network
    if path.startswith("\\\\"):
        return True

    # Get the drive letter and ask Windows what type it is
    drive = os.path.splitdrive(path)[0] + "\\"
    DRIVE_REMOTE = 4
    get_drive_type = ctypes.windll.kernel32.GetDriveTypeW
    return get_drive_type(drive) == DRIVE_REMOTE


def _is_network_macos(path: str) -> bool:
    import subprocess

    # Network fs types commonly seen on macOS
    NETWORK_FS_TYPES = {
        "nfs",
        "afpfs",
        "smbfs",
        "cifs",
        "webdav",
        "ftp",
        "ftpfs",
        "osxfuse",
        "macfuse",
    }

    try:
        result = subprocess.run(
            ["df", "-P", "-T", path],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # df -T output: "Filesystem Type ..."  — type is second column of data row
        lines = result.stdout.strip().splitlines()
        if len(lines) >= 2:
            fs_type = lines[1].split()[1].lower()
            return fs_type in NETWORK_FS_TYPES
    except (subprocess.TimeoutExpired, FileNotFoundError, IndexError):
        pass

    return False


def _is_network_linux(path: str) -> bool:
    NETWORK_FS_TYPES = {
        "nfs",
        "nfs4",
        "cifs",
        "smb",
        "smbfs",
        "afs",
        "ncpfs",
        "glusterfs",
        "cephfs",
        "sshfs",
        "fuse.sshfs",
        "davfs",
        "fuse.davfs2",
        "lustre",
        "gpfs",
        "pvfs2",
    }

    try:
        # /proc/mounts lists all mounted filesystems
        with open("/proc/mounts") as f:
            mounts = f.readlines()
    except OSError:
        return False

    # Find the most specific (longest) mount point that is a prefix of path
    best_match = ""
    best_fs_type = ""
    for line in mounts:
        parts = line.split()
        if len(parts) < 3:
            continue
        mount_point = parts[1]
        fs_type = parts[2].lower()
        if path.startswith(mount_point) and len(mount_point) > len(best_match):
            best_match = mount_point
            best_fs_type = fs_type

    return best_fs_type in NETWORK_FS_TYPES
