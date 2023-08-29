"""Export PDF."""
from pathlib import Path

import matplotlib.pyplot as plt


def export_title(pdf, title: str):
    """Export title."""
    if title and pdf is not None:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.clf()
        fig.text(0.5, 0.5, title, transform=fig.transFigure, size=24, ha="center")
        pdf.savefig()
        plt.close(fig)


def export_figure(
    pdf,
    filename: Path,
    fig,
    face_color=None,
    bbox_inches="tight",
    dpi=150,
    override: bool = False,
    close: bool = False,
    **kwargs,
):
    """Export figure to file."""
    face_color = face_color if face_color is not None else fig.get_facecolor()
    if pdf is not None:
        pdf.savefig(dpi=dpi, facecolor=face_color, bbox_inches=bbox_inches, **kwargs)
    else:
        if override or not filename.exists():
            fig.savefig(filename, dpi=dpi, facecolor=face_color, bbox_inches=bbox_inches, **kwargs)
    if close:
        plt.close(fig)
