"""Path utilities."""
from koyo.typing import PathLike


def open_directory(path: PathLike):
    """Open directory."""
    import os
    import webbrowser

    if not os.path.isdir(path):
        path = os.path.dirname(path)
    webbrowser.open(str(path))


def open_directory_alt(path: PathLike):
    """Open directory."""
    import os
    import platform
    import subprocess

    path = str(path)
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])
