"""Test PDF utilities."""

import numpy as np

from koyo.pdf_mixin import PDFMixin
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image


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
    array = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
    image = Image.fromarray(array)
    pdf._add_pil_image_to_pdf(tmp / "test.png", image, pdf=page)
    assert not (tmp / "test.png").exists()
    pdf._add_title_to_pdf("Title", page)

    pdf._save_pdf(page)
    assert (tmp / "test2.pdf").exists()
