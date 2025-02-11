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
    img = np.nan_to_num(img, posinf=0, neginf=0)
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


def unshape_array(image: np.ndarray, pixel_index: np.ndarray) -> np.ndarray:
    """Retrieve original vector of intensities from an image."""
    if image.ndim == 3:  # channels, height, width
        return image.reshape(image.shape[0], -1)[:, pixel_index].T
    return image.reshape(-1)[pixel_index]


def reshape_array_from_coordinates(
    array: np.ndarray,
    image_shape: ty.Tuple[int, int],
    coordinates: np.ndarray,
    fill_value: float = 0,
    offset: int = 0,
):
    """Reshape array based on xy coordinates.

    This function assumes that the coordinates are 1-indexed.
    """
    dtype = np.float32 if np.isnan(fill_value) else array.dtype
    im = np.full(image_shape, fill_value=fill_value, dtype=dtype)
    try:
        im[coordinates[:, 1] - offset, coordinates[:, 0] - offset] = array
    except IndexError:
        im[coordinates[:, 0] - offset, coordinates[:, 1] - offset] = array
    return im


def flatten_array_from_coordinates(array: np.ndarray, coordinates: np.ndarray, offset: int = 0) -> np.ndarray:
    """Flatten array based on xy coordinates."""
    if array.ndim == 2:
        try:
            return array[coordinates[:, 1] - offset, coordinates[:, 0] - offset]
        except IndexError:
            return array[coordinates[:, 0] - offset, coordinates[:, 1] - offset]
    else:
        try:
            res = array[:, coordinates[:, 1] - offset, coordinates[:, 0] - offset]
        except IndexError:
            res = array[:, coordinates[:, 0] - offset, coordinates[:, 1] - offset]
        # need to swap axes
        res = np.swapaxes(res, 0, 1)
        return res


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
    array: np.ndarray,
    image_shape: ty.Tuple[int, int],
    coordinates: np.ndarray,
    fill_value: int = 0,
    offset: int = 0,
):
    """Batch reshape image.

    This function assumes that the coordinates are 1-indexed.
    """
    if array.ndim != 2:
        raise ValueError("Expected 2-D array.")
    n_features = array.shape[1]
    dtype = np.float32 if np.isnan(fill_value) else array.dtype
    im = np.full((n_features, *image_shape), fill_value=fill_value, dtype=dtype)
    try:
        for i in range(n_features):
            im[i, coordinates[:, 1] - offset, coordinates[:, 0] - offset] = array[:, i]
    except IndexError:
        for i in range(n_features):
            im[i, coordinates[:, 0] - offset, coordinates[:, 1] - offset] = array[:, i]
    return im


def get_coordinates_from_index(index: np.ndarray, shape: ty.Tuple[int, int]) -> np.ndarray:
    """Convert frame index to xy coordinates."""
    index = np.asarray(index)
    # generate image shape
    index_im = reshape_array(index, shape, index)
    if shape[0] != index_im.shape[0]:
        raise ValueError("Image dimension 0 does not match that of the dataset")
    if shape[1] != index_im.shape[1]:
        raise ValueError("Image dimension 1 does not match that of the dataset")

    y_coordinates, x_coordinates = np.indices(index_im.shape)
    yx_coordinates = np.c_[np.ravel(y_coordinates), np.ravel(x_coordinates)][index]
    return yx_coordinates


def colocalization(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Calculate degree of colocalization between two ion images.

    This implementation is nearly identical to that of METASPACE.

    Citation: Ovchinnikova et al. (2020) ColocML. https://doi.org/10.1093/bioinformatics/btaa085

    Parameters
    ----------
    img_a: np.ndarray
        first image
    img_b : np.ndarray
        second image

    Returns
    -------
    similarity : float
        similarity score between two images
    """
    from scipy.ndimage import median_filter
    from sklearn.metrics.pairwise import cosine_similarity

    img_a = np.nan_to_num(img_a)
    img_b = np.nan_to_num(img_b)
    h, w = img_a.shape

    def _preprocess(img):
        img = img.copy().reshape((h, w))
        img[img < np.quantile(img, 0.5)] = 0
        return median_filter(img, (3, 3)).reshape([1, h * w])

    return cosine_similarity(_preprocess(img_a), _preprocess(img_b))[0, 0]


def colocalization_matrix(images: np.ndarray, size: ty.Tuple[int, ...] = (3, 3)) -> np.ndarray:
    """
    Calculates level of colocalization between all pairs of images in a list of ion images.
    If many checks are needed, it is usually faster to generate the entire matrix than to do
    many separate calls to "colocalization".

    Citation: Ovchinnikova et al. (2020) ColocML. https://doi.org/10.1093/bioinformatics/btaa085

    Parameters
    ----------
    images : List[np.ndarray]
        list of ion images
    size : Tuple[int, ...]
        size of the median filter.

    Returns
    -------
    similarity_matrix : np.ndarray
        colocalization similarity matrix
    """
    from scipy.ndimage import median_filter
    from sklearn.metrics.pairwise import pairwise_kernels

    assert images.ndim == 3, "Expected 2D array."  # flat images

    count = images.shape[0]
    if count == 0:
        similarity_matrix = np.ones((0, 0))
    elif count == 1:
        similarity_matrix = np.ones((1, 1))
    else:
        h, w = images[0].shape
        flat_images = np.vstack([i.flatten() for i in images])
        flat_images[flat_images < np.quantile(flat_images, 0.5, axis=1, keepdims=True)] = 0
        filtered_images = median_filter(images.reshape((count, h, w)), (1, *size)).reshape((count, h * w))
        similarity_matrix = pairwise_kernels(filtered_images, metric="cosine")
    return similarity_matrix


def pearson_similarity(img_a: np.ndarray, img_b: np.ndarray, size: ty.Tuple[int, int] = (3, 3)) -> float:
    """Calculate degree of similarity between two images using Pearson correlation.

    Parameters
    ----------
    img_a : np.ndarray
        First image array.
    img_b : np.ndarray
        Second image array.
    size : tuple
        Size of the median filter.

    Returns
    -------
    score : float
        Result of linear regression after each image had median filter applied to it.

    """
    from scipy.ndimage import median_filter
    from scipy.stats import linregress

    if len(size) != 2:
        raise ValueError("Median filter expected 2-element tuple.")

    img_a = median_filter(img_a, size)
    img_b = median_filter(img_b, size)
    mask = (img_a > 0) & (img_b > 0)
    return linregress(img_a[mask], img_b[mask]).rvalue
