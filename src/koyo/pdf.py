"""Export PDF."""
from __future__ import annotations
import typing as ty

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import numpy as np


def export_title(pdf: PdfPages | None, title: str) -> None:
    """Export title."""
    if title and pdf is not None:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.clf()
        fig.text(0.5, 0.5, title, transform=fig.transFigure, size=24, ha="center")
        pdf.savefig()
        plt.close(fig)


def export_figure(
    pdf: PdfPages | None,
    filename: Path,
    fig: plt.Figure,
    face_color: str | np.ndarray | None = None,
    bbox_inches: str | None = "tight",
    dpi: int = 150,
    override: bool = False,
    close: bool = False,
    **kwargs: ty.Any,
) -> None:
    """Export figure to file."""
    face_color = face_color if face_color is not None else fig.get_facecolor()
    if pdf is not None:
        pdf.savefig(dpi=dpi, facecolor=face_color, bbox_inches=bbox_inches, **kwargs)
    else:
        if override or not filename.exists():
            fig.savefig(filename, dpi=dpi, facecolor=face_color, bbox_inches=bbox_inches, **kwargs)
    if close:
        plt.close(fig)


def export_pil_figure(
    pdf: PdfPages | None,
    filename: Path,
    image: Image,
    override: bool = False,
    dpi: int = 150,
    close: bool = False,
    **kwargs: ty.Any,
) -> None:
    """Export PIL image to PDF file (without closing it)."""
    if pdf is not None:
        fig = plt.figure(figsize=(image.width / 72, image.height / 72))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.imshow(image)
        pdf.savefig(fig, dpi=dpi, **kwargs)
        plt.close(fig)
    else:
        if override or not filename.exists():
            image.save(filename, dpi=(dpi, dpi), **kwargs)
    if close:
        image.close()
