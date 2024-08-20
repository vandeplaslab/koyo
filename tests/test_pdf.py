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
        self.as_pdf = False

    @property
    def filename(self) -> Path:
        """Filename."""
        return self._filename


def test_pdf(tmp_path):
    tmp = Path(tmp_path)
    # init PDF
    pdf = PDF(tmp / "test.pdf", as_pdf=True)
    assert pdf.filename is not None
    assert pdf.pdf_filename == tmp / "test.pdf"

    # create random page
    page = pdf._make_pdf(tmp / "test2.pdf")
    assert page is not None
    # export title
    pdf._export_title("Title", page)
    # export figure
    plt.plot([1, 2, 3], [1, 2, 3])
    pdf._export_figure(tmp / "test.png", plt.gcf(), pdf=page)
    assert not (tmp / "test.png").exists()
    # export PIL figure
    array = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
    image = Image.fromarray(array)
    pdf._export_pil_figure(tmp / "test.png", image, pdf=page)
    assert not (tmp / "test.png").exists()
    pdf._export_title("Title", page)

    page.close()
    assert (tmp / "test2.pdf").exists()

    # pdf.pdf.close()
    # assert (tmp / "test.pdf").exists()
