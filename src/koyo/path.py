"""Path utilities."""

from __future__ import annotations
import os
import shutil
import typing as ty
from pathlib import Path

from loguru import logger

from koyo.typing import PathLike
from koyo.system import IS_WIN

DriveMap = tuple[tuple[str, str], ...]


def copy_file(src_path: str, dst_path: str) -> None:
    """Copy a file from src_path to dst_path using zero-copy os.sendfile()."""
    # Try zero-copy if available
    if hasattr(os, "sendfile"):
        with open(src_path, "rb") as fsrc, open(dst_path, "wb") as fdst:
            size = os.fstat(fsrc.fileno()).st_size
            offset = 0
            while offset < size:
                # Send up to 64 MB per call to avoid platform-specific limits
                sent = os.sendfile(fdst.fileno(), fsrc.fileno(), offset, min(size - offset, 64 * 1024 * 1024))
                if sent == 0:
                    break
                offset += sent
        return
    # Fallback to shutil.copyfile if zero-copy is not available
    try:
        shutil.copyfile(src_path, dst_path)
    except FileNotFoundError:
        logger.error(f"Source file '{src_path}' does not exist.")


def mglob(path: Path, *patterns: str, recursive: bool = False) -> ty.Generator[Path, None, None]:
    """Glob multiple patterns."""
    if not path.exists():
        return
    for pattern in patterns:
        if recursive:
            yield from path.rglob(pattern)
        else:
            yield from path.glob(pattern)


def dir_iter(path: PathLike) -> ty.Generator[Path, None, None]:
    """Iterate over directory."""
    path = Path(path)
    for item in path.iterdir():
        if item.is_dir():
            yield item


def uri_to_path(uri: str) -> Path:
    """Convert URI to path."""
    try:
        from urllib.parse import unquote, urlparse
        from urllib.request import url2pathname
    except ImportError:
        # backwards compatability
        from urllib import unquote, url2pathname

        from urlparse import urlparse

    parsed = urlparse(uri)
    host = f"{os.path.sep}{os.path.sep}{parsed.netloc}{os.path.sep}"
    return Path(os.path.normpath(os.path.join(host, url2pathname(unquote(parsed.path)))))


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


def remove_file(file: PathLike) -> None:
    """Remove file."""
    file = Path(file)
    if file.exists():
        if file.is_file():
            try:
                file.unlink()
                logger.trace(f"Removed '{file}'")
            except Exception as e:
                logger.error(f"Failed to remove '{file}'. Reason: {e}")
        else:
            logger.error(f"Failed to remove '{file}'. Reason: Not a file - it's a directory.")


def empty_directory(path: PathLike) -> None:
    """Recursively clear directory."""
    path = Path(path)
    if not path.exists():
        return
    try:
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
                logger.warning(f"Failed to delete {file_path}. Reason: {e}")
    except Exception as e:
        logger.error(f"Failed to empty directory '{path}'. Reason: {e}")
    try:
        shutil.rmtree(path, ignore_errors=True)
        logger.trace(f"Deleted '{path}'")
    except Exception as e:
        logger.error(f"Failed to delete '{path}'. Reason: {e}")


remove_directory = empty_directory


def move_directory(src: PathLike, dest: PathLike) -> None:
    """Move directory."""
    src = Path(src)
    dest = Path(dest)

    if src == dest:
        return

    try:
        shutil.move(src, dest)
        logger.trace(f"Moved '{src}' to '{dest}'")
    except Exception as e:
        logger.error(f"Failed to move '{src}' to '{dest}'. Reason: {e}")


def open_directory(path: PathLike) -> None:
    """Open directory."""
    import webbrowser

    if not os.path.isdir(path):
        path = os.path.dirname(path)
    webbrowser.open(str(path))


def open_directory_alt(path: PathLike, *which: str) -> None:
    """Open directory."""
    import platform
    import subprocess

    if str(path).startswith("file://"):
        path = str(path)[7:]
        path = path.lstrip("/")
    path = Path(path)
    if which:
        path = path.joinpath(*which)

    if platform.system() == "Windows":
        if path.exists():
            if path.is_file():
                subprocess.call(["explorer", "/select,", str(path)])
            else:
                subprocess.call(["explorer", str(path)])
        else:
            subprocess.call(["explorer", os.path.dirname(path)])
    elif platform.system() == "Darwin":
        if path.is_file():
            subprocess.call(["open", "-R", str(path)])
        else:
            subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


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


def apply_drive_mapping(file: Path, drive_map: DriveMap = ()) -> str:
    """Apply drive mapping."""
    file_ = str(file)
    for map_from, map_to in drive_map:
        file_ = file_.replace(map_from, map_to)
    return file_


def create_link(
    target: Path,
    output_dir: Path,
    prefix: str = "",
    suffix: str | None = None,
    link_name: str | None = None,
    drive_map: DriveMap = (),
) -> Path:
    """Create Shortcuts on Windows."""
    if not IS_WIN:
        raise ValueError("create_link is only supported on Windows systems.")

    from pylnk3 import for_file

    target = Path(target)
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    if suffix is None:
        suffix = target.suffix
    elif suffix != "":
        suffix = ""
    if link_name is None:
        link_name = target.stem
    if prefix:
        link_name = f"{prefix}_{link_name}"
    link_name = f"{link_name}{suffix}.lnk"

    link_file = output_dir / link_name
    target_ = apply_drive_mapping(target, drive_map)
    if Path(target_).exists():
        logger.warning(f"Target '{target_}' already exists - it will be overwritten.")
    link_file_ = apply_drive_mapping(link_file, drive_map)
    for_file(target_, link_file_)
    return link_file


def resolve_links(base_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    """Resolve Shortcuts on Windows."""
    links = []
    if IS_WIN:
        from pylnk3 import parse

        paths = list(base_dir.glob("*.lnk"))
        paths_ = []
        for path in paths:
            try:
                path_ = parse(str(path)).path
            except Exception as e:
                logger.warning(f"Could not parse link '{path}': {e}")
                continue
            paths_.append(Path(path_))
        paths_ = [path for path in paths_ if path.exists()]
        links = [path for path in paths_ if path.suffix in extensions]
    return links
