"""Path utilities."""
import os
import shutil
from pathlib import Path
from koyo.typing import PathLike



def empty_directory(path: str) -> None:
    """Recursively clear directory."""
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    shutil.rmtree(path, ignore_errors=True)


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