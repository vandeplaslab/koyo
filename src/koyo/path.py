"""Path utilities."""
import os
import shutil
from pathlib import Path

from loguru import logger

from koyo.typing import PathLike


def get_copy_path(path: PathLike) -> Path:
    """Get copy path."""
    path = Path(path)
    if not path.exists():
        return path
    i = 1
    while True:
        new_path = path.parent / f"{path.stem}_copy{i}{path.suffix}"
        if not new_path.exists():
            break
        i += 1
    return new_path


def empty_directory(path: str) -> None:
    """Recursively clear directory."""
    path = Path(path)
    if not path.exists():
        return
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                logger.trace(f"Deleted '{file_path}'")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                logger.trace(f"Deleted '{file_path}'")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    shutil.rmtree(path, ignore_errors=True)


def open_directory(path: PathLike) -> None:
    """Open directory."""
    import webbrowser

    if not os.path.isdir(path):
        path = os.path.dirname(path)
    webbrowser.open(str(path))


def open_directory_alt(path: PathLike) -> None:
    """Open directory."""
    import platform
    import subprocess

    path = str(path)
    if platform.system() == "Windows":
        if os.path.exists(path):
            subprocess.call(["explorer", "/select,", path])
        else:
            subprocess.call(["explorer", os.path.dirname(path)])
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def create_directory(*path: str) -> Path:
    """Create directory.

    Parameters
    ----------
    path : str
        directory path
    """
    path = Path(os.path.join(*path))
    if not path.exists():
        try:
            path.mkdir(parents=True)
        except OSError:
            logger.warning("Failed to create directory")
        finally:
            return path


def is_network_path(path: PathLike) -> bool:
    """Check if the path is a network path."""
    if os.name == "nt":
        return is_network_path_win(path)
    return is_network_path_unix(path)


def is_network_path_win(path: PathLike) -> bool:
    """Check if the path is a network path."""
    import os

    import win32file

    # Get the root path (e.g., "C:\\" from "C:\\path\\to\\file")
    root_path = os.path.splitdrive(path)[0] + "\\"

    # Get the drive type
    drive_type = win32file.GetDriveType(root_path)

    # Check if the drive type is network
    return bool(drive_type == win32file.DRIVE_REMOTE)


def get_mount_point(path: PathLike) -> str:
    """Finds the mount point for a given path."""
    path = os.path.abspath(path)
    while not os.path.ismount(path):
        path = os.path.dirname(path)
    return path


def is_network_path_unix(path: PathLike) -> bool:
    """Checks if the path is on a network-mounted filesystem."""
    import subprocess

    mount_point = get_mount_point(path)
    # On Linux, you might directly parse /proc/mounts for more detailed info
    # For simplicity, we use df
    df_output = subprocess.check_output(["df", mount_point], universal_newlines=True)
    filesystem_type = df_output.split("\n")[1].split()[0]

    # This is a simple heuristic, you may need to adjust the types based on your needs
    network_filesystems = ["nfs", "smbfs", "cifs", "fuse.sshfs"]

    return any(fs_type in filesystem_type for fs_type in network_filesystems)
