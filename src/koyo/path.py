"""Path utilities."""
import os
from pathlib import Path
from koyo.typing import PathLike


def open_directory(path: PathLike):
    """Open directory."""
    import webbrowser

    if not os.path.isdir(path):
        path = os.path.dirname(path)
    webbrowser.open(str(path))


def open_directory_alt(path: PathLike):
    """Open directory."""
    import platform
    import subprocess

    path = str(path)
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])

def create_directory(*path: str) -> Path:
    """Create directory

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