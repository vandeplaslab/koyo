"""Test pptx_mixin."""

import numpy as np
import pytest

from koyo.pptx_mixin import PPTXMixin, HAS_PPTX
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image


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

    # create random page
    page = pptx._make_pptx(tmp / "test2.pptx")
    assert page is not None
    # export title
    # pdf._add_title_to_pdf("Title", page)
    # export figure
    plt.plot([1, 2, 3], [1, 2, 3])
    pptx._add_mpl_figure_to_pptx(tmp / "test.png", plt.gcf(), pptx=page)
    assert not (tmp / "test.png").exists()
    # export PIL figure
    array = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
    image = Image.fromarray(array)
    pptx._add_pil_image_to_pptx(tmp / "test.png", image, pptx=page)
    assert not (tmp / "test.png").exists()
    pptx._add_title_to_pptx("Title", page)

    pptx._save_pptx(page)
    assert (tmp / "test2.pptx").exists()
