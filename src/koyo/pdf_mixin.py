"""PDF export mixin class."""

from __future__ import annotations

import typing as ty
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from PIL import Image

from koyo.typing import PathLike

if ty.TYPE_CHECKING:
    from matplotlib.backends.backend_pdf import PdfPages


class PDFMixin:
    """Mixin class to help export figures to PDF files."""

    _pdf = None
    as_pdf: bool = False

    # @property
    # def filename(self) -> str:
    #     """Filename that can be used for PDF generation."""
    #     raise NotImplementedError("Must implement method")

    @property
    def pdf_filename(self) -> Path:
        """Return the PDF filename."""
        raise NotImplementedError("Must implement method")
        # if self.filename is None:
        #     raise ValueError("PDF filename not set")
        # return Path(self.filename).with_suffix(".pdf")

    @property
    def pdf(self) -> PdfPages | None:
        """Return PDF figure."""
        if self._pdf is None and self.as_pdf:
            from matplotlib.backends.backend_pdf import PdfPages

            self._pdf = PdfPages(self.pdf_filename)
        return self._pdf

    @pdf.setter
    def pdf(self, value: PdfPages | None) -> None:
        """Set PDF figure."""
        if self._pdf is not None:
            self._pdf.close()
        self._pdf = value

    @staticmethod
    def _make_pdf(filename: PathLike) -> PdfPages:
        """Make PDF."""
        from matplotlib.backends.backend_pdf import PdfPages

        return PdfPages(filename)

    @contextmanager
    def _export_pdf_figures(self, filename: PathLike | None = None) -> ty.Generator[PdfPages | None, None, None]:
        """Export figures."""
        import matplotlib

        matplotlib.use("agg")

        pdf, reset = None, False
        if self.as_pdf:
            if filename:
                pdf = self._make_pdf(filename)
            else:
                pdf = self.pdf
                reset = True
        try:
            yield pdf
        except Exception as e:
            logger.error(f"Error exporting PDF: {e}")
        finally:
            if self.as_pdf:
                self._save_pdf(pdf, reset=reset)

    def _check_figure(self, filename: Path, override: bool = False) -> bool:
        """Check if figure exists.

        Will return True if figure should be created.
        """
        if self.as_pdf:
            return True
        return override or not filename.exists()

    def _add_title_to_pdf(self, title: str, pdf: PdfPages | None = None) -> None:
        pdf = pdf or self.pdf
        add_title_to_pdf(pdf, title)

    def _add_content_to_pdf(self, content: str, title: str = "", pdf: PdfPages | None = None) -> None:
        pdf = pdf or self.pdf
        add_content_to_pdf(pdf, content, title)

    def _save_pdf(self, pdf: PdfPages, reset: bool = False) -> None:
        """Save PPTX."""
        if pdf and hasattr(pdf, "close"):
            pdf.close()
        if pdf and hasattr(pdf, "filename"):
            Path(pdf.filename)
        logger.debug(f"Saved PDF to {self.pdf_filename}")
        if reset:
            self._pdf = None

    def _add_mpl_figure_to_pdf(
        self,
        filename: PathLike,
        fig: plt.Figure,
        face_color: str | np.ndarray | None = None,
        bbox_inches: str | None = "tight",
        dpi: int = 150,
        override: bool = False,
        if_empty: str = "warn",
        close: bool = False,
        pdf: PdfPages | None = None,
        **kwargs: ty.Any,
    ) -> None:
        """Export figure to file."""
        if fig is None:
            self._inform_on_empty(if_empty)
            return

        pdf = pdf or self.pdf
        add_mpl_figure_to_pdf(pdf, filename, fig, face_color, bbox_inches, dpi, override, close=close, **kwargs)

    def _add_pil_image_to_pdf(
        self,
        filename: PathLike,
        image: Image,
        override: bool = False,
        close: bool = False,
        pdf: PdfPages | None = None,
        **kwargs: ty.Any,
    ) -> None:
        """Export figure to file."""
        pdf = pdf or self.pdf
        add_pil_image_to_pdf(pdf, filename, image, override=override, close=close, **kwargs)

    @staticmethod
    def _inform_on_empty(if_empty: str = "warn") -> None:
        """Inform the user if the figure was empty."""
        if if_empty == "none":
            return
        elif if_empty == "warn":
            logger.warning("Figure was empty")
        elif if_empty == "raise":
            raise ValueError("Figure was empty")


def add_title_to_pdf(pdf: PdfPages | None, title: str) -> None:
    """Export title."""
    if title and pdf is not None:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.clf()
        fig.text(0.5, 0.5, title, transform=fig.transFigure, size=24, ha="center")
        pdf.savefig()
        plt.close(fig)


def add_content_to_pdf(pdf: PdfPages | None, content: str, title: str = "") -> None:
    """Export content."""
    if content and pdf is not None:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.clf()
        if title:
            fig.text(0.5, 0.9, title, transform=fig.transFigure, size=16, ha="center")
        fig.text(0.5, 0.5, content, transform=fig.transFigure, size=12, ha="center")


def add_mpl_figure_to_pdf(
    pdf: PdfPages | None,
    filename: PathLike,
    fig: plt.Figure,
    face_color: str | np.ndarray | None = None,
    bbox_inches: str | None = "tight",
    dpi: int = 150,
    override: bool = False,
    close: bool = False,
    format: str = "jpg",
    **kwargs: ty.Any,
) -> None:
    """Export figure to file."""
    face_color = face_color if face_color is not None else fig.get_facecolor()
    if pdf is not None:
        pdf.savefig(dpi=dpi, facecolor=face_color, bbox_inches=bbox_inches, **kwargs)
    else:
        if override or not Path(filename).exists():
            fig.savefig(filename, dpi=dpi, facecolor=face_color, bbox_inches=bbox_inches, format=format, **kwargs)
    if close:
        plt.close(fig)


def add_pil_image_to_pdf(
    pdf: PdfPages | None,
    filename: PathLike,
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
        ax.imshow(image, aspect="auto")
        pdf.savefig(fig, dpi=dpi, **kwargs)
        plt.close(fig)
    else:
        if override or not Path(filename).exists():
            image.save(filename, dpi=(dpi, dpi), **kwargs)
    if close:
        image.close()
