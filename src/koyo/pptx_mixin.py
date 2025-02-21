"""PowerPoint mixin class."""

from __future__ import annotations

import io
import typing as ty
from contextlib import contextmanager
from enum import IntEnum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from koyo.typing import PathLike
from koyo.utilities import is_installed

HAS_PPTX = is_installed("pptx")

if ty.TYPE_CHECKING:
    from PIL import Image

    try:
        from pptx.presentation import Presentation
    except ImportError:
        Presentation = None  # type: ignore[assignment,misc]


class SlideLayout(IntEnum):
    """Slide layout options."""

    TITLE = 0
    TITLE_AND_CONTENT = 1
    SECTION_HEADER = 2
    TITLE_AND_TWO_CONTENT = 3
    TITLE_AND_TWO_CONTENT_WITH_HEADER = 4
    TITLE_AND_BLANK = 5
    BLANK = 6
    HEADER_AND_TWO_CONTENT = 7
    PICTURE_AND_CAPTION = 8


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
            self._pptx.save(self.pptx_filename)  # type: ignore[arg-type]
        self._pptx = value

    @staticmethod
    def _make_pptx(filename: PathLike) -> Presentation:
        """Make PDF."""
        try:
            from pptx import Presentation
            from pptx.util import Inches
        except ImportError:
            raise ImportError("pptx is not installed. Please install it using `pip install python-pptx`.") from None

        pptx = Presentation()
        pptx.slide_width = Inches(16)
        pptx.slide_height = Inches(9)
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
        try:
            yield pptx
        except Exception as e:
            logger.exception(f"Error exporting to PPTX: {e}")
        finally:
            if self.as_pptx:
                self._save_pptx(pptx, filename, reset)  # type: ignore[arg-type]

    def _add_title_to_pptx(self, title: str, pptx: Presentation | None = None) -> None:
        """Add title page to the slide deck."""
        pptx = pptx or self.pptx
        add_title_to_pptx(pptx, title)  # type: ignore[arg-type]

    def _add_content_to_pptx(self, content: str, title: str = "", pptx: Presentation | None = None) -> None:
        """Add title page to the slide deck."""
        pptx = pptx or self.pptx
        add_content_to_pptx(pptx, content, title)  # type: ignore[arg-type]

    def _add_mpl_figure_to_pptx(
        self,
        filename: PathLike,
        fig: plt.Figure,
        face_color: str | np.ndarray | None = None,
        bbox_inches: str | None = "tight",
        dpi: int = 150,
        override: bool = False,
        if_empty: str = "warn",
        close: bool = False,
        title: str = "",
        pptx: Presentation | None = None,
        **kwargs: ty.Any,
    ) -> None:
        """Export figure to file."""
        if fig is None:
            self._inform_on_empty(if_empty)
            return

        pptx = pptx or self.pptx
        add_mpl_figure_to_pptx(
            pptx, filename, fig, face_color, bbox_inches, dpi, override, close=close, title=title, **kwargs
        )

    def _add_pil_image_to_pptx(
        self,
        filename: PathLike,
        image: Image,
        dpi: int = 150,
        fmt: str = "JPEG",
        override: bool = False,
        close: bool = False,
        title: str = "",
        pptx: Presentation | None = None,
        **kwargs: ty.Any,
    ) -> None:
        """Add PIL image to pptx."""
        pptx = pptx or self.pptx
        add_pil_image_to_pptx(
            pptx, filename, image, dpi, fmt=fmt, override=override, close=close, title=title, **kwargs
        )

    def _save_pptx(self, pptx: Presentation, filename: PathLike | None = None, reset: bool = False) -> None:
        """Save PPTX."""
        if hasattr(pptx, "_filename"):
            filename = pptx._filename
        else:
            filename = filename or self.pptx_filename
        pptx.save(filename)  # type: ignore[arg-type]
        logger.trace(f"Saved PPTX to {filename} with {len(pptx.slides)} slides")
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
    from pptx.util import Cm

    slide = pptx.slides.add_slide(pptx.slide_layouts[SlideLayout.TITLE])
    count = title.count("\n") + 1
    height = slide.shapes.title.height * count
    slide.shapes.title.text = title
    # adjust positions
    slide.shapes.title.left = Cm(0)
    slide.shapes.title.width = pptx.slide_width
    slide.shapes.title.height = height


def add_content_to_pptx(pptx: Presentation, content: str, title: str = "") -> None:
    """Add title to the slide."""
    from pptx.util import Cm

    slide = pptx.slides.add_slide(pptx.slide_layouts[SlideLayout.TITLE_AND_CONTENT])
    count = title.count("\n") + 1
    height = slide.shapes.title.height * count
    slide.shapes.title.text = title
    slide.placeholders[1].text = content
    # adjust positions
    slide.shapes.title.left = Cm(0)
    slide.shapes.title.width = pptx.slide_width
    slide.shapes.title.height = height


def add_mpl_figure_to_pptx(
    pptx: Presentation | None,
    filename: PathLike,
    fig: plt.Figure,
    face_color: str | np.ndarray | None = None,
    bbox_inches: str | None = "tight",
    dpi: int = 150,
    override: bool = False,
    close: bool = False,
    title: str = "",
    format: str = "jpg",
    **kwargs: ty.Any,
) -> None:
    """Export figure to file."""
    face_color = face_color if face_color is not None else fig.get_facecolor()
    if pptx is not None:
        with io.BytesIO() as image_stream:
            fig.savefig(image_stream, dpi=dpi, facecolor=face_color, bbox_inches=bbox_inches, format=format, **kwargs)
            slide, left, top = _insert_slide(pptx, title=title)
            slide.shapes.add_picture(image_stream, left, top)  # , width=pptx.slide_width, height=pptx.slide_height)
    else:
        if override or not Path(filename).exists():
            fig.savefig(filename, dpi=dpi, facecolor=face_color, bbox_inches=bbox_inches, **kwargs)
    if close:
        plt.close(fig)


def add_pil_image_to_pptx(
    pptx: Presentation | None,
    filename: PathLike,
    image: Image,
    dpi: int = 150,
    fmt: str = "JPEG",
    override: bool = False,
    close: bool = False,
    title: str = "",
    **kwargs: ty.Any,
) -> None:
    """Export figure to file."""
    quality = kwargs.pop("quality", 95)
    if pptx is not None:
        if image.mode == "RGBA" and fmt.upper() in ["JPEG", "JPG"]:
            image = image.convert("RGB")
        with io.BytesIO() as image_stream:
            image.save(image_stream, fmt, quality=quality, dpi=(dpi, dpi), **kwargs)
            slide, left, top = _insert_slide(pptx, title=title)
            slide.shapes.add_picture(image_stream, left, top)  # , width=pptx.slide_width, height=pptx.slide_height)
    else:
        if override or not Path(filename).exists():
            image.save(filename, dpi=(dpi, dpi), **kwargs)
    if close:
        image.close()


def _insert_slide(pptx: Presentation, title: str = "") -> tuple[ty.Any, int, int]:
    left = top = 0
    template_index = SlideLayout.TITLE_AND_BLANK if title else SlideLayout.BLANK
    slide = pptx.slides.add_slide(pptx.slide_layouts[template_index])
    if title:
        from pptx.util import Cm, Pt

        count = title.count("\n") + 1

        slide.shapes.title.text_frame.paragraphs[0].font.size = Pt(20)
        slide.shapes.title.text = title
        height = slide.shapes.title.height // 2
        height *= count
        slide.shapes.title.top = Cm(0)
        slide.shapes.title.left = Cm(0)
        slide.shapes.title.width = pptx.slide_width
        slide.shapes.title.height = height
        top = slide.shapes.title.top + height
    return slide, left, top
