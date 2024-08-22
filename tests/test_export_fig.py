"""Test PDF utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from koyo.fig_mixin import FigureExporter, FigureMixin
from koyo.pptx_mixin import HAS_PPTX


class PptxPdf(FigureMixin):
    """Mixin class."""

    def __init__(self, filename: Path, as_pdf: bool = False, as_pptx: bool = False):
        self.as_pdf = as_pdf
        self.as_pptx = as_pptx
        self._filename = Path(self.get_pptx_or_pdf_filename(filename))

    @property
    def pdf_filename(self) -> Path:
        """Filename."""
        return self._filename

    @property
    def pptx_filename(self) -> Path:
        """Filename."""
        return self._filename


def test_init(tmp_path):
    tmp = Path(tmp_path)
    # init PDF
    pptx_or_pdf = PptxPdf(tmp / "test.pdf", as_pdf=True, as_pptx=True)
    with pytest.raises(ValueError):
        pptx_or_pdf._check_export()


def test_api_auto_pdf(tmp_path):
    tmp = Path(tmp_path)
    pptx_or_pdf = PptxPdf(tmp / "test", as_pdf=True, as_pptx=False)
    filename = pptx_or_pdf.get_pptx_or_pdf_filename("test")
    with pptx_or_pdf._export_figures() as obj:
        assert obj is not None
        plt.plot([1, 2, 3], [1, 2, 3])
        obj.add_or_export_mpl_figure(tmp / "test.png", plt.gcf(), title="test")
        assert not (tmp / "test.png").exists()
        array = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
        image = Image.fromarray(array)
        obj.add_or_export_pil_image(tmp / "test.png", image, title="test")
        assert not (tmp / "test.png").exists()
        obj.add_title("Title")
    assert (tmp / filename).exists()


@pytest.mark.skipif(not HAS_PPTX, reason="pptx not installed")
def test_api_auto_pptx(tmp_path):
    tmp = Path(tmp_path)
    pptx_or_pdf = PptxPdf(tmp / "test", as_pdf=False, as_pptx=True)
    filename = pptx_or_pdf.get_pptx_or_pdf_filename("test")
    with pptx_or_pdf._export_figures() as obj:
        assert obj is not None
        plt.plot([1, 2, 3], [1, 2, 3])
        obj.add_or_export_mpl_figure(tmp / "test.png", plt.gcf(), title="test")
        assert not (tmp / "test.png").exists()
        array = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
        image = Image.fromarray(array)
        obj.add_or_export_pil_image(tmp / "test.png", image, title="test")
        assert not (tmp / "test.png").exists()
        obj.add_title("Title")
    assert (tmp / filename).exists()


def test_api_pdf(tmp_path):
    tmp = Path(tmp_path)
    pptx_or_pdf = PptxPdf(tmp / "test.pdf", as_pdf=True, as_pptx=False)

    with pptx_or_pdf._export_figures() as obj:
        assert obj is not None
        plt.plot([1, 2, 3], [1, 2, 3])
        obj.add_or_export_mpl_figure(tmp / "test.png", plt.gcf())
        assert not (tmp / "test.png").exists()
        array = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
        image = Image.fromarray(array)
        obj.add_or_export_pil_image(tmp / "test.png", image)
        assert not (tmp / "test.png").exists()
        obj.add_title("Title")
    assert (tmp / "test.pdf").exists()


@pytest.mark.skipif(not HAS_PPTX, reason="pptx not installed")
def test_api_pptx(tmp_path):
    tmp = Path(tmp_path)
    pptx_or_pdf = PptxPdf(tmp / "test.pptx", as_pdf=False, as_pptx=True)

    with pptx_or_pdf._export_figures() as obj:
        assert obj is not None
        plt.plot([1, 2, 3], [1, 2, 3])
        obj.add_or_export_mpl_figure(tmp / "test.png", plt.gcf())
        assert not (tmp / "test.png").exists()
        array = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
        image = Image.fromarray(array)
        obj.add_or_export_pil_image(tmp / "test.png", image)
        assert not (tmp / "test.png").exists()
        obj.add_title("Title")
    assert (tmp / "test.pptx").exists()


def test_pdf(tmp_path):
    tmp = Path(tmp_path)
    # init PDF
    pptx_or_pdf = PptxPdf(tmp / "test.pdf", as_pdf=True)
    assert pptx_or_pdf.pdf_filename == tmp / "test.pdf"

    # create random page
    page = pptx_or_pdf._make_pdf(tmp / "test2.pdf")
    assert page is not None
    # export title
    pptx_or_pdf._add_title("Title", pdf=page)
    # export figure
    plt.plot([1, 2, 3], [1, 2, 3])
    pptx_or_pdf._add_or_export_mpl_figure(tmp / "test.png", plt.gcf(), pdf=page)
    assert not (tmp / "test.png").exists()
    # export PIL figure
    array = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
    image = Image.fromarray(array)
    pptx_or_pdf._add_or_export_pil_image(tmp / "test.png", image, pdf=page)
    assert not (tmp / "test.png").exists()
    pptx_or_pdf._add_title("Title", pdf=page)

    pptx_or_pdf._save(pdf=page)
    assert (tmp / "test2.pdf").exists()


@pytest.mark.skipif(not HAS_PPTX, reason="pptx not installed")
def test_pptx(tmp_path):
    tmp = Path(tmp_path)
    # init PDF
    pptx_or_pdf = PptxPdf(tmp / "test.pptx", as_pptx=True)
    assert pptx_or_pdf.pptx_filename == tmp / "test.pptx"

    # create random page
    page = pptx_or_pdf._make_pptx(tmp / "test2.pptx")
    assert page is not None
    # export title
    pptx_or_pdf._add_title("Title", pptx=page)
    # export figure
    plt.plot([1, 2, 3], [1, 2, 3])
    pptx_or_pdf._add_or_export_mpl_figure(tmp / "test.png", plt.gcf(), pptx=page)
    assert not (tmp / "test.png").exists()
    plt.figure(figsize=(10, 3))
    plt.plot([1, 2, 3], [1, 2, 3])
    pptx_or_pdf._add_or_export_mpl_figure(tmp / "test.png", plt.gcf(), pptx=page)
    assert not (tmp / "test.png").exists()
    # export PIL figure
    array = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
    image = Image.fromarray(array)
    pptx_or_pdf._add_or_export_pil_image(tmp / "test.png", image, pptx=page)
    assert not (tmp / "test.png").exists()
    pptx_or_pdf._add_title("Title", pptx=page)

    pptx_or_pdf._save(pptx=page)
    assert (tmp / "test2.pptx").exists()


def test_exporter_pdf(tmp_path):
    tmp = Path(tmp_path)
    # init PDF
    pptx_or_pdf = FigureExporter(tmp / "test.pdf", as_pdf=True)
    assert pptx_or_pdf.pdf_filename == tmp / "test.pdf"
    plt.plot([1, 2, 3], [1, 2, 3])
    plt.gcf().savefig(tmp / "test.png")
    plt.gcf().savefig(tmp / "test.jpg")
    plt.gcf().savefig(tmp / "test.jpeg")
    plt.close("all")
    pptx_or_pdf.export_existing(tmp)
    assert (tmp / "test.pdf").exists()

    pptx_or_pdf.export_existing(tmp, filename=tmp / "test2.pdf", clear=True)
    assert not (tmp / "test.png").exists()
    assert (tmp / "test2.pdf").exists()


@pytest.mark.skipif(not HAS_PPTX, reason="pptx not installed")
def test_exporter_pptx(tmp_path):
    tmp = Path(tmp_path)
    # init PDF
    pptx_or_pdf = FigureExporter(tmp / "test.pptx", as_pptx=True)
    assert pptx_or_pdf.pptx_filename == tmp / "test.pptx"
    plt.plot([1, 2, 3], [1, 2, 3])
    plt.gcf().savefig(tmp / "test.png")
    plt.gcf().savefig(tmp / "test.jpg")
    plt.gcf().savefig(tmp / "test.jpeg")
    plt.close("all")
    pptx_or_pdf.export_existing(tmp)
    assert (tmp / "test.pptx").exists()

    pptx_or_pdf.export_existing(tmp, filename=tmp / "test2.pptx", clear=True)
    assert not (tmp / "test.png").exists()
    assert (tmp / "test2.pptx").exists()
