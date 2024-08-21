"""Mixin class for figure exports"""

from __future__ import annotations

from contextlib import contextmanager

import typing as ty
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from koyo.pdf_mixin import PDFMixin
from koyo.pptx_mixin import PPTXMixin
from koyo.typing import PathLike
from loguru import logger

if ty.TYPE_CHECKING:
    from PIL import Image
    from matplotlib.backends.backend_pdf import PdfPages

    try:
        from pptx.presentation import Presentation
    except ImportError:
        Presentation = None  # type: ignore[assignment,misc]


class FigureMixin(PDFMixin, PPTXMixin):
    """Mixin class for figure exports"""

    @property
    def as_pptx_or_pdf(self) -> bool:
        """Check whether export is enabled."""
        return self.as_pptx or self.as_pdf

    def _check_export(self, as_pptx: bool | None = None, as_pdf: bool | None = None) -> None:
        """Check whether pdf/pptx export is enabled."""
        if as_pptx is not None:
            self.as_pptx = as_pptx
        if as_pdf is not None:
            self.as_pdf = as_pdf
        if self.as_pptx and self.as_pdf:
            raise ValueError("Cannot export to both PDF and PPTX")
        # if self.as_pptx:
        #     assert self.pptx_filename is not None, "PPTX filename not set"
        # if self.as_pdf:
        #     assert self.pdf_filename is not None, "PDF filename not set"

    def _get_export_filename(self, filename: str) -> str:
        """Get export filename."""
        filename = str(filename).split(".")[0]
        if self.as_pptx:
            return f"{filename}.pptx"
        elif self.as_pdf:
            return f"{filename}.pdf"
        return filename

    @contextmanager
    def _export_figures(self, filename: PathLike | None = None) -> ty.Iterator[PptxPdfWrapper]:
        """Export figures."""
        if filename:
            logger.trace(f"Exporting figures to {filename}")
        if self.as_pptx:
            with self._export_pptx_figures(filename) as pptx:
                yield PptxPdfWrapper(pptx, as_pptx=self.as_pptx, as_pdf=self.as_pdf)
        elif self.as_pdf:
            with self._export_pdf_figures(filename) as pdf:
                yield PptxPdfWrapper(pdf, as_pptx=self.as_pptx, as_pdf=self.as_pdf)
        else:
            yield PptxPdfWrapper(None, as_pptx=self.as_pptx, as_pdf=self.as_pdf)

    def _add_title(self, title: str, pdf: PdfPages | None = None, pptx: Presentation | None = None) -> None:
        """Add title to pptx."""
        if self.as_pptx:
            self._add_title_to_pptx(title, pptx=pptx)
        elif self.as_pdf:
            self._add_title_to_pdf(title, pdf=pdf)

    def _save(self, pptx: Presentation | None = None, pdf: PdfPages | None = None) -> None:
        """Save pptx/pdf."""
        if self.as_pptx:
            self._save_pptx(pptx)
        elif self.as_pdf:
            self._save_pdf(pdf)

    def _add_or_export_mpl_figure(
        self,
        filename: Path,
        fig: ty.Any,
        face_color: str | np.ndarray | None = None,
        bbox_inches: str | None = "tight",
        dpi: int = 150,
        override: bool = False,
        pdf: PdfPages | None = None,
        pptx: Presentation | None = None,
        **kwargs: ty.Any,
    ) -> None:
        """Export figure to file."""
        if self.as_pdf:
            self._add_mpl_figure_to_pdf(filename, fig, pdf=pdf, **kwargs)
        elif self.as_pptx:
            self._add_mpl_figure_to_pptx(filename, fig, pptx=pptx, **kwargs)
        elif override or not filename.exists():
            fig.savefig(filename, dpi=dpi, facecolor=face_color, bbox_inches=bbox_inches, **kwargs)

    def _add_or_export_pil_image(
        self,
        filename: Path,
        image: Image,
        dpi: int = 150,
        fmt: str = "JPEG",
        override: bool = False,
        pdf: PdfPages | None = None,
        pptx: Presentation | None = None,
        **kwargs: ty.Any,
    ) -> None:
        """Export PIL image to file."""
        if self.as_pdf:
            self._add_pil_image_to_pdf(filename, image, pdf=pdf, **kwargs)
        elif self.as_pptx:
            self._add_pil_image_to_pptx(filename, image, pptx=pptx, **kwargs)
        elif override or not filename.exists():
            image.save(filename, dpi=(dpi, dpi), format=fmt, **kwargs)


class PptxPdfWrapper:
    """Wrapper class that handles both PPTX and PDF exports"""

    def __init__(
        self, ppt_or_pdf: PdfPages | Presentation | None = None, as_pptx: bool = False, as_pdf: bool = False
    ) -> None:
        self.ppt_or_pdf = ppt_or_pdf
        self.as_pptx = as_pptx
        self.as_pdf = as_pdf

    @property
    def as_pptx_or_pdf(self) -> bool:
        """Check whether export is enabled."""
        return self.as_pptx or self.as_pdf

    def add_title(self, title: str) -> None:
        """Add title slide or page."""
        if self.as_pdf:
            from koyo.pdf_mixin import add_title_to_pdf

            add_title_to_pdf(self.ppt_or_pdf, title)
        elif self.as_pptx:
            from koyo.pptx_mixin import add_title_to_pptx

            add_title_to_pptx(self.ppt_or_pdf, title)  # type: ignore[arg-type]

    def add_or_export_pil_image(
        self,
        filename: Path,
        image: Image,
        dpi: int = 150,
        fmt: str = "JPEG",
        override: bool = False,
        close: bool = False,
        **kwargs: ty.Any,
    ) -> None:
        """Add or export PIL image."""
        if self.as_pdf:
            from koyo.pdf_mixin import add_pil_image_to_pdf

            add_pil_image_to_pdf(self.ppt_or_pdf, filename, image, override=override, **kwargs)
        elif self.as_pptx:
            from koyo.pptx_mixin import add_pil_image_to_pptx

            add_pil_image_to_pptx(self.ppt_or_pdf, filename, image, override=override, **kwargs)
        elif override or not filename.exists():
            image.save(filename, dpi=(dpi, dpi), format=fmt, **kwargs)
            if close:
                image.close()

    def add_or_export_mpl_figure(
        self,
        filename: Path,
        fig: ty.Any,
        face_color: str | np.ndarray | None = None,
        bbox_inches: str | None = "tight",
        dpi: int = 150,
        override: bool = False,
        close: bool = False,
        **kwargs: ty.Any,
    ) -> None:
        """Add or export matplotlib figure."""
        if self.as_pdf:
            from koyo.pdf_mixin import add_mpl_figure_to_pdf

            add_mpl_figure_to_pdf(
                self.ppt_or_pdf,
                filename,
                fig,
                face_color=face_color,
                bbox_inches=bbox_inches,
                dpi=dpi,
                override=override,
                close=close,
                **kwargs,
            )
        elif self.as_pptx:
            from koyo.pptx_mixin import add_mpl_figure_to_pptx

            add_mpl_figure_to_pptx(
                self.ppt_or_pdf,
                filename,
                fig,
                face_color=face_color,
                bbox_inches=bbox_inches,
                dpi=dpi,
                override=override,
                close=close,
                **kwargs,
            )
        elif override or not filename.exists():
            fig.savefig(filename, dpi=dpi, facecolor=face_color, bbox_inches=bbox_inches, **kwargs)
            if close:
                plt.close(fig)
