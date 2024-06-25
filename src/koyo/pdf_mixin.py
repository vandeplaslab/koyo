"""PDF export mixin class."""
from __future__ import annotations

import typing as ty
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

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

    @contextmanager
    def _export_figures(self) -> ty.Generator[None, None, None]:
        """Export figures."""
        import matplotlib

        matplotlib.use("agg")

        yield
        if self.as_pdf and hasattr(self.pdf, "close"):
            self.pdf.close()  # type: ignore[union-attr]
            self._pdf = None

    def _export_title(self, title: str) -> None:
        from koyo.pdf import export_title

        export_title(self.pdf, title)

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
        bbox_inches: str = "tight",
        dpi: int = 150,
        override: bool = False,
        if_empty: str = "warn",
        **kwargs: ty.Any,
    ) -> None:
        """Export figure to file."""
        from koyo.pdf import export_figure

        if fig is None:
            self._inform_on_empty(if_empty)
            return

        export_figure(self.pdf, filename, fig, face_color, bbox_inches, dpi, override, **kwargs)

    @staticmethod
    def _inform_on_empty(if_empty: str = "warn") -> None:
        """Inform the user if the figure was empty."""
        if if_empty == "none":
            return
        elif if_empty == "warn":
            logger.warning("Figure was empty")
        elif if_empty == "raise":
            raise ValueError("Figure was empty")
