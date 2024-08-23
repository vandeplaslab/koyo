"""Test pptx_mixin."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from koyo.pptx_mixin import HAS_PPTX, PPTXMixin


class PPTX(PPTXMixin):
    """Mixin class."""

    def __init__(self, filename: Path, as_pptx: bool):
        self._filename = filename
        self.as_pptx = as_pptx

    @property
    def pptx_filename(self) -> Path:
        """Filename."""
        return self._filename


@pytest.mark.skipif(not HAS_PPTX, reason="pptx not installed")
def test_pptx(tmp_path):
    tmp = Path(tmp_path)
    # init PDF
    pptx = PPTX(tmp / "test.pptx", as_pptx=True)

    assert pptx.pptx_filename == tmp / "test.pptx"
    array = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
    image = Image.fromarray(array)
    pptx._add_pil_image_to_pptx(tmp / "test.png", image, title="Title for plot")
    pptx._save_pptx(pptx.pptx)
    assert (tmp / "test.pptx").exists()

    # create random page
    page = pptx._make_pptx(tmp / "test2.pptx")
    assert page is not None
    # export figure
    plt.plot([1, 2, 3], [1, 2, 3])
    pptx._add_mpl_figure_to_pptx(tmp / "test.png", plt.gcf(), pptx=page)
    assert not (tmp / "test.png").exists()
    plt.plot([1, 2, 3], [1, 2, 3])
    pptx._add_mpl_figure_to_pptx(tmp / "test.png", plt.gcf(), pptx=page, title="Title for plot")
    assert not (tmp / "test.png").exists()
    # export PIL figure
    array = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
    image = Image.fromarray(array)
    pptx._add_pil_image_to_pptx(tmp / "test.png", image, pptx=page, title="Title for plot")
    assert not (tmp / "test.png").exists()
    pptx._add_title_to_pptx("Title", page)
    pptx._add_content_to_pptx("Other text\nOther text\nOther text", "Title", page)
    pptx._save_pptx(page)
    assert (tmp / "test2.pptx").exists()

    with pptx._export_pptx_figures(tmp / "test3.pptx") as page:
        array = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
        image = Image.fromarray(array)
        pptx._add_pil_image_to_pptx(tmp / "test.png", image, pptx=page, title="Title for plot")
    assert (tmp / "test3.pptx").exists()
