"""Image processing functions."""
import typing as ty

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


def reshape_array(array: np.ndarray, image_shape: ty.Tuple[int, int], pixel_index: np.ndarray, fill_value: float = 0):
    """
    Reshape 1D data to 2D heatmap.

    Parameters
    ----------
    array: np.array / list
        1D array of values to be reshaped
    image_shape: tuple
        final shape of the image
    pixel_index: np.array
        array containing positions where pixels should be placed, considering missing values -
        e.g. not acquired pixels
    fill_value : float, optional
        if value is provided, it will be used to fill-in the values

    Returns
    -------
    im_array: np.array
        reshaped heatmap of shape `image_shape`
    """
    if isinstance(array, np.matrix):
        array = np.asarray(array).flatten()
    array = np.asarray(array)
    dtype = np.float32 if isinstance(fill_value, float) else array.dtype

    image_n_pixels = np.prod(image_shape)
    im_array = np.full(image_n_pixels, dtype=dtype, fill_value=fill_value)
    im_array[pixel_index] = array
    im_array = np.reshape(im_array, image_shape)
    return im_array


def reshape_array_from_coordinates(
    array: np.ndarray, image_shape: ty.Tuple[int, int], coordinates: np.ndarray, fill_value: float = 0
):
    """Reshape array based on xy coordinates."""
    dtype = np.float32 if np.isnan(fill_value) else array.dtype
    im = np.full(image_shape, fill_value=fill_value, dtype=dtype)
    im[coordinates[:, 1] - 1, coordinates[:, 0] - 1] = array
    return im


def reshape_array_batch(
    array: np.ndarray, image_shape: ty.Tuple[int, int], pixel_index: np.ndarray, fill_value: float = 0
):
    """Reshape many images into a data cube."""
    array = np.asarray(array)
    if array.ndim == 1:
        return reshape_array(array, image_shape, pixel_index, fill_value)
    count = array.shape[1]
    dtype = np.float32 if isinstance(fill_value, float) else array.dtype

    im_array = np.full((count, np.prod(image_shape)), dtype=dtype, fill_value=fill_value)
    for i in range(count):
        im_array[i, pixel_index] = array[:, i]
    # reshape data
    im_array = np.reshape(im_array, (count, *image_shape))
    return im_array


def reshape_array_batch_from_coordinates(
    array: np.ndarray, image_shape: ty.Tuple[int, int], coordinates: np.ndarray, fill_value: int = 0
):
    """Batch reshape image."""
    if array.ndim != 2:
        raise ValueError("Expected 2-D array.")
    n = array.shape[1]
    dtype = np.float32 if np.isnan(fill_value) else array.dtype
    im = np.full((n, *image_shape), fill_value=fill_value, dtype=dtype)
    for i in range(n):
        im[i, coordinates[:, 1] - 1, coordinates[:, 0] - 1] = array[:, i]
    return im
