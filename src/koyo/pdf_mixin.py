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

    @property
    def filename(self) -> str:
        """Filename that can be used for PDF generation."""
        raise NotImplementedError("Must implement method")

    @property
    def pdf_filename(self) -> Path:
        """Return the PDF filename."""
        if self.filename is None:
            raise ValueError("PDF filename not set")
        return Path(self.filename).with_suffix(".pdf")

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
    def _export_figures(self, filename: PathLike | None = None) -> ty.Generator[PdfPages | None, None, None]:
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
        yield pdf
        if self.as_pdf and hasattr(pdf, "close"):
            pdf.close()  # type: ignore[union-attr]
            if reset:
                self._pdf = None

    def _export_title(self, title: str, pdf: PdfPages | None = None) -> None:
        from koyo.pdf import export_title

        pdf = pdf or self.pdf
        export_title(pdf, title)

    def _check_figure(self, filename: Path, override: bool = False) -> bool:
        """Check if figure exists.

        Will return True if figure should be created.
        """
        if self.as_pdf:
            return True
        return override or not filename.exists()

    def _export_figure(
        self,
        filename: Path,
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
        from koyo.pdf import export_figure

        if fig is None:
            self._inform_on_empty(if_empty)
            return

        pdf = pdf or self.pdf
        export_figure(pdf, filename, fig, face_color, bbox_inches, dpi, override, close=close, **kwargs)

    def _export_pil_figure(
        self,
        filename: Path,
        image: Image,
        override: bool = False,
        close: bool = False,
        pdf: PdfPages | None = None,
        **kwargs: ty.Any,
    ) -> None:
        """Export figure to file."""
        from koyo.pdf import export_pil_figure

        pdf = pdf or self.pdf
        export_pil_figure(pdf, filename, image, override, close=close, **kwargs)

    def _insert_title(self, text: str, pdf: PdfPages | None = None) -> None:
        """Insert title page to PDF document."""
        pdf = pdf or self.pdf
        if text and pdf:
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.clf()
            fig.text(0.5, 0.5, text, transform=fig.transFigure, size=24, ha="center")
            pdf.savefig()
            plt.close(fig)

    @staticmethod
    def _inform_on_empty(if_empty: str = "warn") -> None:
        """Inform the user if the figure was empty."""
        if if_empty == "none":
            return
        elif if_empty == "warn":
            logger.warning("Figure was empty")
        elif if_empty == "raise":
            raise ValueError("Figure was empty")
