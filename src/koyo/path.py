"""Path utilities."""

from __future__ import annotations

import os
import hashlib
import re
import shutil
import typing as ty
from pathlib import Path, PureWindowsPath

from loguru import logger

from koyo.system import IS_LINUX, IS_MAC, IS_WIN
from koyo.typing import PathLike

DriveMap = tuple[tuple[str, str], ...]

_WINDOWS_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1F]')
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


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
    # Fall back to a buffered copy to avoid shutil's platform-specific fast-copy
    # helpers, which may still try to use os.sendfile internally.
    try:
        with open(src_path, "rb") as fsrc, open(dst_path, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst)
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


def empty_directory(path: PathLike, remove_parent: bool = True) -> None:
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

    if remove_parent:
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
        # only keep 'which' if the parts are (str, Path)
        which = [part for part in which if isinstance(part, (str, Path))]
        if which:
            path = path.joinpath(*which)

    if platform.system() == "Windows":
        if path.exists():
            if path.is_file():
                subprocess.call(["explorer", "/select,", str(path)])
            else:
                subprocess.call(["explorer", str(path)])
        else:
            subprocess.call(["explorer", str(path.parent)])
    elif platform.system() == "Darwin":
        if path.is_file():
            subprocess.call(["open", "-R", str(path)])
        else:
            subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def open_directory_universal(path: str | os.PathLike[str]) -> None:
    """
    Open a directory in the system file manager on macOS, Windows, or Linux.

    Supports:
    - local paths
    - network-mounted drives / shares
    - paths with spaces

    Parameters
    ----------
    path:
        Directory path to open.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    RuntimeError
        If opening the directory fails.
    """
    import subprocess

    directory = Path(path).expanduser()
    if not directory.exists():
        raise FileNotFoundError(f"Path does not exist: {directory}")
    if not directory.is_dir():
        directory = directory.parent

    try:
        if IS_WIN:
            # Use os.startfile on Windows; handles spaces fine.
            os.startfile(str(directory))  # type: ignore[attr-defined]
        elif IS_MAC:
            # subprocess with a list avoids shell quoting issues.
            subprocess.run(["open", str(directory)], check=True)
        elif IS_LINUX:
            subprocess.run(["xdg-open", str(directory)], check=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to open directory: {directory}") from exc


def get_subdirectories(path: PathLike) -> list[Path]:
    """Get subdirectories."""
    path = Path(path)
    if not path.exists():
        return []
    return [item for item in path.iterdir() if item.is_dir()]


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
    return None


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
    # Normalize separators so configured mappings behave the same on every OS.
    file_ = Path(file).as_posix()
    for map_from, map_to in drive_map:
        file_ = file_.replace(Path(map_from).as_posix(), Path(map_to).as_posix())
    return file_


def create_link(
    target: Path,
    output_dir: Path,
    prefix: str = "",
    suffix: str | None = None,
    link_name: str | None = None,
    drive_map: DriveMap = (),
) -> Path:
    """Create link cross-platform."""
    if IS_WIN:
        return create_lnk(target, output_dir, prefix, suffix, link_name, drive_map)
    return create_symlink(target, output_dir, prefix, suffix, link_name, drive_map)


def create_lnk(
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
    if Path(link_file).exists():
        logger.warning(f"Link file '{link_file}' already exists - it will be overwritten.")
    link_file_ = apply_drive_mapping(link_file, drive_map)
    for_file(target_, link_file_)
    return link_file


def create_symlink(
    target: Path,
    output_dir: Path,
    prefix: str = "",
    suffix: str | None = None,
    link_name: str | None = None,
    drive_map: DriveMap = (),
) -> Path:
    """Create a symlink on macOS/Linux (POSIX)."""
    target = Path(target)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Match your Windows naming behavior
    if suffix is None:
        suffix = target.suffix
    elif suffix != "":
        # same logic as your original (note: this makes suffix always "", unless None)
        suffix = ""

    if link_name is None:
        link_name = target.stem

    if prefix:
        link_name = f"{prefix}_{link_name}"

    # On POSIX, no .lnk extension; keep suffix behavior if you want
    link_name = f"{link_name}{suffix}"
    link_file = output_dir / link_name

    target_ = Path(apply_drive_mapping(target, drive_map))
    link_file_ = Path(apply_drive_mapping(link_file, drive_map))

    # If link already exists, replace it (like your Windows "overwrite" behavior)
    if link_file_.exists() or link_file_.is_symlink():
        link_file_.unlink()

    # Prefer relative links when possible (more portable inside the same tree)
    try:
        rel_target = os.path.relpath(target_, start=link_file_.parent)
        link_file_.symlink_to(rel_target)
    except Exception:
        # Fallback to absolute
        link_file_.symlink_to(target_)
    return link_file


def create_link_txtlnk(
    target: Path,
    output_dir: Path,
    prefix: str = "",
    suffix: str | None = None,
    link_name: str | None = None,
    drive_map: DriveMap = (),
) -> Path:
    """Create a link cross-platform.

    Rather than creating a symbolic link or a shortcut, we will save the path to a text file with the same name as
    the target, where the content of the text file is the path to the target. This way, we can achieve a similar
    functionality across different operating systems without relying on OS-specific features.
    """
    if suffix is None:
        suffix = target.suffix
    elif suffix != "":
        # same logic as your original (note: this makes suffix always "", unless None)
        suffix = ""

    if link_name is None:
        link_name = target.stem

    if prefix:
        link_name = f"{prefix}_{link_name}"

    # On POSIX, no .lnk extension; keep suffix behavior if you want
    link_name = f"{link_name}{suffix}.txtlnk"
    link_file = output_dir / link_name

    target_ = Path(apply_drive_mapping(target, drive_map))
    link_file_ = Path(apply_drive_mapping(link_file, drive_map))

    # Write the target path to the link file
    link_file_.parent.mkdir(parents=True, exist_ok=True)
    link_file_.write_text(str(target_))
    logger.debug(f"Created link file '{link_file_}' pointing to '{target_}'")
    return link_file_


def resolve_links(base_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    """Resolve Shortcuts on Windows."""
    links = []
    if IS_WIN:
        try:
            from pylnk3 import parse
        except ImportError:
            logger.debug("pylnk3 is not installed; skipping .lnk resolution on Windows.")
        else:
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
            links.extend(path for path in paths_ if path.suffix in extensions)

    # Resolve .txtlnk files on every platform as a lightweight fallback.
    for link_file in base_dir.glob("*.txtlnk"):
        try:
            target_path = link_file.read_text().strip()
            target_path = Path(target_path)
            if target_path.exists() and target_path.suffix in extensions:
                links.append(target_path)
            else:
                logger.warning(
                    f"Target path '{target_path}' from link '{link_file}' does not exist or has unsupported"
                    f" extension.",
                )
        except Exception as e:
            logger.warning(f"Could not read link file '{link_file}': {e}")
    return links


class WindowsPathError(ValueError):
    """Raised when a valid Windows-safe path cannot be produced."""


def _is_reserved_windows_name(name: str) -> bool:
    """
    Check whether a single filename component is a reserved Windows device name.
    The check ignores the extension, e.g. 'CON.txt' is still invalid.
    """
    base = name.split(".", 1)[0].rstrip(" .").upper()
    return base in _WINDOWS_RESERVED_NAMES


def _is_valid_windows_component(name: str) -> bool:
    """
    Validate a single path component for Windows.
    """
    if not name:
        return False
    if name in {".", ".."}:
        return True
    if _WINDOWS_INVALID_CHARS.search(name):
        return False
    if name.endswith(" ") or name.endswith("."):
        return False
    if _is_reserved_windows_name(name):
        return False
    return True


def _is_valid_windows_path(path: str | Path, max_path: int = 260) -> bool:
    """
    Conservative Windows path validation.

    Notes:
    - Uses the traditional MAX_PATH default of 260 unless overridden.
    - Modern Windows can support longer paths, but that is not always enabled.
    """
    s = str(path)
    p = PureWindowsPath(s)

    if len(str(p)) > max_path:
        return False

    for part in p.parts:
        # Skip drive roots and path roots
        if part in {"\\", "/"}:
            continue
        if re.fullmatch(r"[A-Za-z]:\\?", part):
            continue
        if not _is_valid_windows_component(part):
            return False

    return True


def shorten_path_for_windows(
    path: str | Path,
    *,
    max_path: int = 260,
    max_component_length: int = 255,
    hash_len: int = 4,
    collision_check: bool = True,
) -> Path:
    """
    Shorten a path's filename until it is valid on Windows.

    Strategy
    --------
    1. If already valid, return as-is.
    2. Shorten the filename stem while preserving the extension.
    3. If the shortened candidate collides with an existing file, use a short hash.
    4. If no valid candidate can be produced, raise WindowsPathError.

    Parameters
    ----------
    path:
        Input path.
    max_path:
        Maximum allowed full path length. Defaults to 260 for broad Windows compatibility.
    max_component_length:
        Maximum length for a single filename component. Defaults to 255.
    hash_len:
        Length of the fallback hash suffix.
    collision_check:
        If True, check whether the candidate already exists on disk.

    Returns
    -------
    Path
        A Windows-safe path.

    Raises
    ------
    WindowsPathError
        If the path cannot be made valid.
    """
    p = Path(path)
    parent = p.parent
    original_name = p.name

    if not original_name:
        raise WindowsPathError("Path has no filename to shorten.")

    # If the parent path is already too long, there may be no room left
    parent_str = str(parent)
    separator_len = 1 if parent_str not in {"", "."} else 0

    # Preserve full suffix chain, e.g. ".tar.gz"
    suffix = "".join(p.suffixes)
    if suffix and len(suffix) >= len(original_name):
        stem = original_name
        suffix = ""
    else:
        stem = original_name[: len(original_name) - len(suffix)]

    if not _is_valid_windows_component(original_name):
        # If invalid for reasons other than length, shortening alone will not fix it.
        # We allow reserved-name repair by changing the stem during truncation,
        # but invalid characters should fail immediately.
        if _WINDOWS_INVALID_CHARS.search(original_name) or original_name.endswith((" ", ".")):
            raise WindowsPathError(
                f"Filename contains Windows-invalid characters or trailing space/dot: {original_name!r}"
            )

    # Room available for the filename component based on total path length
    remaining_for_name = max_path - len(parent_str) - separator_len
    allowed_name_len = min(max_component_length, remaining_for_name)

    if allowed_name_len <= 0:
        raise WindowsPathError(f"Parent path is too long to fit any filename within max_path={max_path}.")

    if len(suffix) >= allowed_name_len:
        raise WindowsPathError(f"Extension {suffix!r} leaves no room for a valid filename stem.")

    def make_candidate(candidate_stem: str) -> Path:
        return parent / f"{candidate_stem}{suffix}"

    def is_unique(candidate: Path) -> bool:
        return not collision_check or not candidate.exists()

    def is_valid(candidate: Path) -> bool:
        return _is_valid_windows_path(candidate, max_path=max_path)

    # Early return if already valid
    if is_valid(p):
        return p

    # First pass: plain truncation
    max_stem_len = allowed_name_len - len(suffix)
    for n in range(max_stem_len, 0, -1):
        candidate_stem = stem[:n].rstrip(" .")
        if not candidate_stem:
            continue

        candidate = make_candidate(candidate_stem)

        if is_valid(candidate):
            if is_unique(candidate) or candidate == p:
                return candidate
            break  # collision -> go to hash fallback

    # Second pass: hash fallback
    digest = hashlib.blake2b(original_name.encode("utf-8"), digest_size=16).hexdigest()[:hash_len]
    hash_suffix = f"_{digest}"
    max_stem_len_with_hash = max_stem_len - len(hash_suffix)

    if max_stem_len_with_hash <= 0:
        raise WindowsPathError("Not enough space to add a hash-based fallback filename.")

    for n in range(max_stem_len_with_hash, 0, -1):
        candidate_stem = stem[:n].rstrip(" .")
        if not candidate_stem:
            continue

        hashed_stem = f"{candidate_stem}{hash_suffix}"
        candidate = make_candidate(hashed_stem)

        if is_valid(candidate):
            if is_unique(candidate) or candidate == p:
                return candidate

    raise WindowsPathError(f"Could not produce a valid Windows-safe filename for path: {str(path)!r}")
