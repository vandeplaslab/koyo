"""Plotly functions."""

from __future__ import annotations

import typing as ty
from pathlib import Path

from koyo.typing import PathLike
from koyo.utilities import is_installed

if ty.TYPE_CHECKING:
    try:
        from plotly.graph_objects import Figure
    except ImportError:
        Figure = None


def write_html(fig: Figure, filename: PathLike) -> Path:
    """Write HTML to file."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(filename), include_plotlyjs="cdn")
    return filename


def write_png(fig: Figure, filename: PathLike) -> Path | None:
    """Write PNG."""
    filename = Path(filename).with_suffix(".png")
    if is_installed("kaleido"):
        filename.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(filename))
        return filename


def show_html(filename: PathLike) -> None:
    """Show HTML in browser."""
    import webbrowser

    filename = Path(filename)
    webbrowser.open(filename.as_uri())
