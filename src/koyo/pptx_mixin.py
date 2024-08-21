"""PowerPoint mixin class."""

from __future__ import annotations

import typing as ty
from contextlib import contextmanager
from pathlib import Path
import io
import numpy as np
from koyo.typing import PathLike
from koyo.utilities import is_installed
import matplotlib.pyplot as plt
from loguru import logger

HAS_PPTX = is_installed("pptx")

if ty.TYPE_CHECKING:
    from PIL import Image

    try:
        from pptx.presentation import Presentation
    except ImportError:
        Presentation = None  # type: ignore[assignment,misc]


class PPTXMixin:
    """Mixin class to help export figures to PPTX files."""

    _pptx = None
    as_pptx: bool = False

    @property
    def pptx_filename(self) -> Path:
        """Return the PDF filename."""
        raise NotImplementedError("Must implement method")

    @property
    def pptx(self) -> Presentation | None:
        """Return PDF figure."""
        if self._pptx is None and self.as_pptx:
            self._pptx = self._make_pptx(self.pptx_filename)
        return self._pptx

    @pptx.setter
    def pptx(self, value: Presentation | None) -> None:
        """Set PDF figure."""
        if self._pptx is not None:
            self._pptx.save(self.pptx_filename)
        self._pptx = value

    @staticmethod
    def _make_pptx(filename: PathLike) -> Presentation:
        """Make PDF."""
        try:
            from pptx import Presentation
        except ImportError:
            raise ImportError("pptx is not installed. Please install it using `pip install python-pptx`.")

        pptx = Presentation()
        pptx._filename = filename  # type: ignore[attr-defined]
        return pptx

    @contextmanager
    def _export_pptx_figures(self, filename: PathLike | None = None) -> ty.Generator[Presentation | None, None, None]:
        """Export figures."""
        import matplotlib

        matplotlib.use("agg")

        pptx, reset = None, False
        if self.as_pptx:
            if filename:
                pptx = self._make_pptx(filename)
            else:
                pptx = self.pptx
                reset = True
        yield pptx
        if self.as_pptx:
            self._save_pptx(pptx, filename, reset)  # type: ignore[arg-type]

    def _add_title_to_pptx(self, title: str, pptx: Presentation | None = None) -> None:
        """Add title page to the slide deck."""
        pptx = pptx or self.pptx
        add_title_to_pptx(pptx, title)

    def _add_mpl_figure_to_pptx(
        self,
        filename: Path,
        fig: plt.Figure,
        face_color: str | np.ndarray | None = None,
        bbox_inches: str | None = "tight",
        dpi: int = 150,
        override: bool = False,
        if_empty: str = "warn",
        close: bool = False,
        pptx: Presentation | None = None,
        **kwargs: ty.Any,
    ) -> None:
        """Export figure to file."""

        if fig is None:
            self._inform_on_empty(if_empty)
            return

        pptx = pptx or self.pptx
        add_mpl_figure_to_pptx(pptx, filename, fig, face_color, bbox_inches, dpi, override, close=close, **kwargs)

    def _add_pil_image_to_pptx(
        self,
        filename: Path,
        image: Image,
        dpi: int = 150,
        fmt: str = "JPEG",
        override: bool = False,
        close: bool = False,
        pptx: Presentation | None = None,
        **kwargs: ty.Any,
    ) -> None:
        """Add PIL image to pptx."""
        pptx = pptx or self.pptx
        add_pil_image_to_pptx(pptx, filename, image, dpi, fmt=fmt, override=override, close=close, **kwargs)

    def _save_pptx(self, pptx: Presentation, filename: PathLike | None = None, reset: bool = False) -> None:
        """Save PPTX."""
        if hasattr(pptx, "_filename"):
            filename = getattr(pptx, "_filename")
            pptx.save(filename)  # type: ignore[union-attr,arg-type]
        else:
            filename or self.pptx_filename
        pptx.save(filename)  # type: ignore[union-attr,arg-type]
        logger.trace(f"Saved PPTX to {filename}")
        if reset:
            self._pptx = None

    @staticmethod
    def _inform_on_empty(if_empty: str = "warn") -> None:
        """Inform the user if the figure was empty."""
        if if_empty == "none":
            return
        elif if_empty == "warn":
            logger.warning("Figure was empty")
        elif if_empty == "raise":
            raise ValueError("Figure was empty")


def add_title_to_pptx(pptx: Presentation, title: str) -> None:
    """Add title to the slide."""
    slide = pptx.slides.add_slide(pptx.slide_layouts[0])
    title_placeholder = slide.shapes.title
    title_placeholder.text = title


def add_mpl_figure_to_pptx(
    pptx: Presentation | None,
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
    if pptx is not None:
        with io.BytesIO() as image_stream:
            fig.savefig(image_stream, dpi=dpi, facecolor=face_color, bbox_inches=bbox_inches, **kwargs)
            slide = pptx.slides.add_slide(pptx.slide_layouts[6])
            slide.shapes.add_picture(image_stream, 0, 0)  # , width=pptx.slide_width, height=pptx.slide_height)
    else:
        if override or not filename.exists():
            fig.savefig(filename, dpi=dpi, facecolor=face_color, bbox_inches=bbox_inches, **kwargs)
    if close:
        plt.close(fig)


def add_pil_image_to_pptx(
    pptx: Presentation | None,
    filename: Path,
    image: Image,
    dpi: int = 150,
    fmt: str = "JPEG",
    override: bool = False,
    close: bool = False,
    **kwargs: ty.Any,
) -> None:
    """Export figure to file."""
    quality = kwargs.pop("quality", 95)
    if pptx is not None:
        with io.BytesIO() as image_stream:
            image.save(image_stream, fmt, quality=quality, dpi=(dpi, dpi), **kwargs)
            slide = pptx.slides.add_slide(pptx.slide_layouts[6])
            slide.shapes.add_picture(image_stream, 0, 0)  # , width=pptx.slide_width, height=pptx.slide_height)
    else:
        if override or not filename.exists():
            image.save(filename, dpi=(dpi, dpi), **kwargs)
    if close:
        image.close()
