"""Image processing functions."""
import numpy as np


def clip_hotspots(img: np.ndarray, quantile: float = 0.99) -> np.ndarray:
    """Remove hotspots from ion image.

    Parameters
    ----------
    img : np.ndarray
        Image array.
    quantile : float
        Quantile at which to clip intensities.

    Returns
    -------
    img : np.ndarray
        Clipped array.

    """
    mask = np.isnan(img)
    img = np.nan_to_num(img)
    min_visible = np.max(img) / 256
    if min_visible > 0:
        hotspot_threshold = np.quantile(img[img > min_visible], quantile)
        img = np.clip(img, None, hotspot_threshold)
    img[mask] = np.nan
    return img
