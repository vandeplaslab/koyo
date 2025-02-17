"""Mixin class for figure exports."""

from __future__ import annotations

import typing as ty
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from tqdm import tqdm

from koyo.pdf_mixin import PDFMixin
from koyo.pptx_mixin import PPTXMixin
from koyo.system import IS_WIN
from koyo.typing import PathLike

if ty.TYPE_CHECKING:
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image

    try:
        from pptx.presentation import Presentation
    except ImportError:
        Presentation = None  # type: ignore[assignment,misc]


class FigureMixin(PDFMixin, PPTXMixin):
    """Mixin class for figure exports."""

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

    def get_pptx_or_pdf_filename(self, filename: PathLike) -> str:
        """Get export filename."""
        filename = str(filename)
        parts = filename.rsplit(".", maxsplit=1)
        # if there are multiple ., then let's combine it using . but exclude the last one
        if len(parts) > 1:
            filename = ".".join(parts[:-1])
        if self.as_pptx:
            return f"{filename}.pptx"
        elif self.as_pdf:
            return f"{filename}.pdf"
        return filename

    def has_slides(self, pptx_filename: Path) -> bool:
        """Check whether PPTX has slides."""
        from pptx import Presentation

        if pptx_filename.exists():
            pptx = Presentation(pptx_filename)
            has_slides = bool(pptx.slides)
            return has_slides
        return False

    def get_actual_output_filename(self, output_dir: PathLike, pptx_or_pdf_path: PathLike) -> Path:
        """Get output filename, depending on the export format."""
        if self.as_pptx_or_pdf:
            return Path(pptx_or_pdf_path)
        return Path(output_dir)

    def make_directory_if_not_exporting(self, directory: PathLike) -> Path:
        """Make directory if it's not being exported to PDF/PPTX."""
        directory = Path(directory)
        if not self.as_pptx_or_pdf:
            directory.mkdir(parents=True, exist_ok=True)
        return directory

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

    def _add_content(
        self, content: str, title: str, pdf: PdfPages | None = None, pptx: Presentation | None = None
    ) -> None:
        """Add title to pptx."""
        if self.as_pptx:
            self._add_content_to_pptx(content, title, pptx=pptx)
        elif self.as_pdf:
            self._add_content_to_pdf(content, title, pdf=pdf)

    def _save(self, pptx: Presentation | None = None, pdf: PdfPages | None = None) -> None:
        """Save pptx/pdf."""
        if self.as_pptx:
            self._save_pptx(pptx)
        elif self.as_pdf:
            self._save_pdf(pdf)

    def _add_or_export_mpl_figure(
        self,
        filename: Path,
        fig: plt.Figure | None,
        face_color: str | np.ndarray | None = None,
        bbox_inches: str | None = "tight",
        dpi: int = 150,
        override: bool = False,
        pdf: PdfPages | None = None,
        pptx: Presentation | None = None,
        close: bool = False,
        **kwargs: ty.Any,
    ) -> None:
        """Export figure to file."""
        if fig is None:
            logger.warning("Figure is None, skipping export")
            return

        if self.as_pdf:
            self._add_mpl_figure_to_pdf(filename, fig, pdf=pdf, **kwargs)
        elif self.as_pptx:
            self._add_mpl_figure_to_pptx(filename, fig, pptx=pptx, **kwargs)
        elif override or not filename.exists():
            fig.savefig(filename, dpi=dpi, facecolor=face_color, bbox_inches=bbox_inches, **kwargs)
            if close:
                plt.close(fig)

    def _add_or_export_pil_image(
        self,
        filename: Path,
        image: Image,
        dpi: int = 150,
        fmt: str = "JPEG",
        override: bool = False,
        pdf: PdfPages | None = None,
        pptx: Presentation | None = None,
        close: bool = False,
        **kwargs: ty.Any,
    ) -> None:
        """Export PIL image to file."""
        if self.as_pdf:
            self._add_pil_image_to_pdf(filename, image, pdf=pdf, **kwargs)
        elif self.as_pptx:
            self._add_pil_image_to_pptx(filename, image, pptx=pptx, **kwargs)
        elif override or not filename.exists():
            image.save(filename, dpi=(dpi, dpi), format=fmt, **kwargs)
            if close:
                image.close()


class FigureExporter(FigureMixin):
    """Wrapper around the mixin class to provide the interface without having to subclass it."""

    def __init__(self, filename: PathLike | None = None, as_pptx: bool = False, as_pdf: bool = False) -> None:
        self.filename = Path(filename) if filename else None
        self._check_export(as_pptx=as_pptx, as_pdf=as_pdf)

    @property
    def pdf_filename(self) -> Path:
        """Return the PDF filename."""
        if not self.filename:
            raise ValueError("Filename not set")
        return Path(self.get_pptx_or_pdf_filename(self.filename))

    @property
    def pptx_filename(self) -> Path:
        """Return the PPTX filename."""
        if not self.filename:
            raise ValueError("Filename not set")
        return Path(self.get_pptx_or_pdf_filename(self.filename))

    def export_existing(
        self,
        input_dir: PathLike,
        extensions: tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg"),
        filename: PathLike | None = None,
        clear: bool = False,
        silent: bool = False,
    ) -> None:
        """Export existing figures from within a directory (this includes any PNG, JPEG) files read by PIL."""
        from PIL import Image

        extensions = tuple(ext if ext.startswith("*") else f"*{ext}" for ext in extensions)
        assert any(ext.startswith("*") for ext in extensions), "Extensions must start with '*'"
        assert "*.pptx" not in extensions, "Cannot export PPTX files"
        assert "*.pdf" not in extensions, "Cannot export PDF files"

        n = 0
        with self._export_figures(filename) as pptx_or_pdf:
            for ext in extensions:
                for file in tqdm(
                    Path(input_dir).rglob(ext), desc=f"Exporting images ({ext})", disable=silent, leave=False
                ):
                    pptx_or_pdf.add_or_export_pil_image(file, Image.open(file), close=True)
                    n += 1
        # if clear is enabled, remove directory if it is empty
        if clear:
            for ext in extensions:
                for file in Path(input_dir).rglob(ext):
                    file.unlink()
            if not list(Path(input_dir).rglob("*")):
                Path(input_dir).rmdir()
            logger.trace(f"Removed empty directory '{input_dir}'")
        logger.info(f"Exported {n} images from '{input_dir}' to '{self.filename}'")


class PptxPdfWrapper:
    """Wrapper class that handles both PPTX and PDF exports."""

    def __init__(
        self, ppt_or_pdf: PdfPages | Presentation | None = None, as_pptx: bool = False, as_pdf: bool = False
    ) -> None:
        self.ppt_or_pdf = ppt_or_pdf
        self.as_pptx = as_pptx
        self.as_pdf = as_pdf

    def make_directory_if_not_exporting(self, directory: PathLike) -> Path:
        """Make directory if it's not being exported to PDF/PPTX."""
        directory = Path(directory)
        if not self.as_pptx_or_pdf:
            directory.mkdir(parents=True, exist_ok=True)
        return directory

    def figure_exists(self, filename: Path, override: bool = False) -> bool:
        """Check whether figure exists."""
        return (filename.exists() and not override) and not self.as_pptx_or_pdf

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

    def add_content(self, content: str, title: str = "") -> None:
        """Add title slide or page."""
        if self.as_pptx:
            from koyo.pptx_mixin import add_content_to_pptx

            add_content_to_pptx(self.ppt_or_pdf, content, title)  # type: ignore[arg-type]
        elif self.as_pdf:
            from koyo.pdf_mixin import add_content_to_pdf

            add_content_to_pdf(self.ppt_or_pdf, content, title)

    def add_or_export_pil_image(
        self,
        filename: PathLike,
        image: Image,
        dpi: int = 150,
        fmt: str = "JPEG",
        override: bool = False,
        close: bool = False,
        title: str = "",
        **kwargs: ty.Any,
    ) -> None:
        """Add or export PIL image."""
        if self.as_pdf:
            from koyo.pdf_mixin import add_pil_image_to_pdf

            add_pil_image_to_pdf(self.ppt_or_pdf, filename, image, override=override, **kwargs)
        elif self.as_pptx:
            from koyo.pptx_mixin import add_pil_image_to_pptx

            add_pil_image_to_pptx(self.ppt_or_pdf, filename, image, override=override, title=title, **kwargs)
        elif override or not Path(filename).exists():
            image.save(filename, dpi=(dpi, dpi), format=fmt, **kwargs)
            if close:
                image.close()

    def add_or_export_mpl_figure(
        self,
        filename: PathLike,
        fig: plt.Figure | None,
        face_color: str | np.ndarray | None = None,
        bbox_inches: str | None = "tight",
        dpi: int = 150,
        override: bool = False,
        close: bool = False,
        title: str = "",
        **kwargs: ty.Any,
    ) -> None:
        """Add or export matplotlib figure."""
        if fig is None:
            logger.warning("Figure is None, skipping export")
            return
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
                title=title,
                **kwargs,
            )
        elif override or not Path(filename).exists():
            face_color = face_color if face_color is not None else fig.get_facecolor()
            filename = _ensure_filename_is_not_too_long(filename)
            fig.savefig(filename, dpi=dpi, facecolor=face_color, bbox_inches=bbox_inches, **kwargs)
            if close:
                plt.close(fig)


def _ensure_filename_is_not_too_long(filenme: PathLike) -> Path:
    """Ensures on Windows that filename is not too long."""
    if not IS_WIN:
        return Path(filenme)
    filename = Path(filenme)
    n = len(str(filename))
    if n > 250:
        parent = filename.parent
        suffix = filename.suffix
        if n - len(str(parent)) > 250:
            raise ValueError("Filename is too long")
        max_length = 250 - len(str(parent)) - len(suffix)
        if max_length > len(filename.stem):
            max_length = len(filename.stem)
        filename_ = filename.stem[0:max_length] + suffix
        logger.trace(f"Filename is too long, truncating to {filename_} from {filename.name}")
        return parent / filename_
    return filename
