"""Test PDF utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from koyo.pdf_mixin import PDFMixin


class PDF(PDFMixin):
    """Mixin class."""

    def __init__(self, filename: Path, as_pdf: bool):
        self._filename = filename
        self.as_pdf = as_pdf

    @property
    def pdf_filename(self) -> Path:
        """Filename."""
        return self._filename


def test_pdf(tmp_path):
    tmp = Path(tmp_path)
    # init PDF
    pdf = PDF(tmp / "test.pdf", as_pdf=True)

    assert pdf.pdf_filename == tmp / "test.pdf"
    array = np.random.default_rng().integers(0, 255, (1000, 1000), dtype=np.uint8)
    image = Image.fromarray(array)
    pdf._add_pil_image_to_pdf(tmp / "test.png", image)
    pdf._save_pdf(pdf.pdf)
    assert (tmp / "test.pdf").exists()

    # create random page
    page = pdf._make_pdf(tmp / "test2.pdf")
    assert page is not None
    # export title
    pdf._add_title_to_pdf("Title", page)
    # export figure
    plt.plot([1, 2, 3], [1, 2, 3])
    pdf._add_mpl_figure_to_pdf(tmp / "test.png", plt.gcf(), pdf=page)
    assert not (tmp / "test.png").exists()
    # export PIL figure
    array = np.random.default_rng().integers(0, 255, (1000, 1000), dtype=np.uint8)
    image = Image.fromarray(array)
    pdf._add_pil_image_to_pdf(tmp / "test.png", image, pdf=page)
    assert not (tmp / "test.png").exists()
    pdf._add_title_to_pdf("Title", page)
    pdf._save_pdf(page)
    assert (tmp / "test2.pdf").exists()

    with pdf._export_pdf_figures(tmp / "test3.pdf") as page:
        array = np.random.default_rng().integers(0, 255, (1000, 1000), dtype=np.uint8)
        image = Image.fromarray(array)
        pdf._add_pil_image_to_pdf(tmp / "test.png", image, pdf=page)
    assert (tmp / "test3.pdf").exists()


def test_add_content_to_pdf_adds_page(tmp_path):
    pdf = PDF(tmp_path / "test.pdf", as_pdf=True)
    page = pdf._make_pdf(tmp_path / "content.pdf")
    n_figs = len(plt.get_fignums())
    pdf._add_content_to_pdf("Some content", "Title", pdf=page)
    assert page.get_pagecount() == 1
    assert len(plt.get_fignums()) == n_figs  # figure was closed
    pdf._save_pdf(page)


def test_add_mpl_figure_to_pdf_saves_given_figure(tmp_path, monkeypatch):
    pdf = PDF(tmp_path / "test.pdf", as_pdf=True)
    page = pdf._make_pdf(tmp_path / "figs.pdf")
    fig1 = plt.figure()
    fig2 = plt.figure()  # becomes the "current" figure
    saved = []
    original = page.savefig
    monkeypatch.setattr(page, "savefig", lambda figure=None, **kw: (saved.append(figure), original(figure, **kw)))
    pdf._add_mpl_figure_to_pdf(tmp_path / "x.png", fig1, pdf=page)
    assert saved == [fig1]
    plt.close(fig1)
    plt.close(fig2)
    pdf._save_pdf(page)


def test_export_pdf_figures_propagates_errors(tmp_path):
    import pytest

    pdf = PDF(tmp_path / "test.pdf", as_pdf=True)
    with pytest.raises(KeyError), pdf._export_pdf_figures(tmp_path / "err.pdf"):
        raise KeyError("boom")
