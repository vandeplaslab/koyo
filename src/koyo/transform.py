"""Affine transformation functions."""

from __future__ import annotations

import contextlib

import numpy as np

from loguru import logger


def transform_xy_coordinates(
    xy: np.ndarray, *, yx_affine: np.ndarray | None = None, xy_affine: np.ndarray | None = None
) -> np.ndarray:
    """Transform xy coordinates using either yx or xy affine matrix."""
    if xy_affine is None and yx_affine is None:
        raise ValueError("Either xy_affine or yx_affine should be provided.")
    if xy_affine is not None and yx_affine is not None:
        raise ValueError("Only one of xy_affine or yx_affine should be provided.")
    xy = np.hstack([xy, np.ones((xy.shape[0], 1))])
    if yx_affine is not None:
        xy = np.dot(xy, yx_affine.T)
    if xy_affine is not None:
        xy = np.dot(xy, xy_affine)
    xy = xy[:, :2]
    return xy


def invert_affine(fwd_affine: np.ndarray) -> np.ndarray:
    """
    Invert a 3x3 homogeneous affine matrix.

    Parameters
    ----------
    fwd_affine : (3, 3) ndarray

    Returns
    -------
    inv_matrix : (3, 3) ndarray
    """
    if fwd_affine.shape != (3, 3):
        raise ValueError(f"Expected (3, 3) matrix, got {fwd_affine.shape}")
    return np.linalg.inv(fwd_affine)


def warp_points(yx: np.ndarray, fwd_affine: np.ndarray) -> np.ndarray:
    """
    Apply a 3x3 homogeneous affine matrix to yx pixel coordinates (forward mapping).

    Parameters
    ----------
    yx : (N, 2) ndarray of float — [[y0, x0], [y1, x1], ...]
    fwd_affine : (3, 3) homogeneous affine matrix in yx pixel space.

    Returns
    -------
    coords_out : (N, 2) ndarray of float
    """
    if fwd_affine.shape != (3, 3):
        raise ValueError(f"Expected (3, 3) matrix, got {fwd_affine.shape}")
    if yx.ndim != 2 or yx.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) coords, got {yx.shape}")

    ones = np.ones((len(yx), 1), dtype=np.float32)
    coords_h = np.hstack([yx, ones], dtype=np.float32)  # (N, 3)
    result_h = (fwd_affine @ coords_h.T).T  # (N, 3)
    return result_h[:, :2] / result_h[:, 2:3]  # de-homogenise


def warp_channel(
    image: np.ndarray, affine_inv: np.ndarray, output_shape: tuple[int, int], order: int = 1, silent: bool = False, use_cv2: bool | None = None
) -> np.ndarray:
    """Warp image."""
    import cv2
    from scipy.ndimage import affine_transform

    assert image.ndim == 2, "Only 2D images are supported for warping in this function."
    use_cv2 = max(max(image.shape), max(output_shape)) < 32767 if use_cv2 is None else use_cv2
    if not silent:
        logger.trace(f"Using {'cv2' if use_cv2 else 'scipy'} for warping.")
    if use_cv2:
        image = _convert_array_to_numpy(image)
        image = ensure_opencv_dtype(image)
        warped = cv2.warpAffine(
            image.T,
            invert_affine(affine_inv)[:2, :],
            output_shape,
            flags=cv2.INTER_NEAREST if order == 0 else cv2.INTER_LINEAR,
        ).T
    else:
        warped = affine_transform(image, affine_inv, order=order, output_shape=output_shape)
    return warped


def warp_channels(
    image: np.ndarray, affine_inv: np.ndarray, output_shape: tuple[int, int], order: int = 1, silent: bool = False, use_cv2: bool | None = None
) -> np.ndarray:
    """Warp multi-channel image."""
    n_channels = image.shape[0]
    warped = np.zeros((n_channels, *output_shape), dtype=image.dtype)
    for c in range(n_channels):
        warped[c] = warp_channel(image[c], affine_inv, output_shape, order=order, silent=silent, use_cv2=use_cv2)
    return warped


def warp_coordinates_from_image(image: np.ndarray, affine_inv: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Warp coordinates from an image."""
    tmp = warp_channel(image, affine_inv, output_shape=output_shape)
    y, x = np.where(tmp > 0)
    return y, x


def get_shape_from_affine(shape: tuple[int, int], fwd_affine: np.ndarray) -> tuple[int, int]:
    """
    Compute the bounding box shape of an image after a forward affine transform.

    Parameters
    ----------
    shape : (H, W)
    fwd_affine : (3, 3) forward affine matrix in yx pixel space

    Returns
    -------
    (H_out, W_out) : tight bounding box around the four transformed corners
    """
    h, w = shape
    # Four corners in yx, going round the image
    corners = np.array(
        [[0, 0], [0, w - 1], [h - 1, 0], [h - 1, w - 1]],
        dtype=float,
    )

    transformed = warp_points(corners, fwd_affine)  # (4, 2)

    y_min, x_min = transformed.min(axis=0)
    y_max, x_max = transformed.max(axis=0)

    h_out = int(np.ceil(y_max - y_min)) + 1
    w_out = int(np.ceil(x_max - x_min)) + 1
    return h_out, w_out


def get_shape_from_affine_shifted(shape: tuple[int, int], fwd_affine: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    """
    Returns the output shape and an adjusted matrix that shifts all content
    into non-negative coordinates (no clipping).

    Returns
    -------
    output_shape : (H_out, W_out)
    adjusted_matrix : (3, 3) matrix with translation baked in
    """
    h, w = shape
    corners = np.array(
        [
            [0, 0],
            [0, w - 1],
            [h - 1, 0],
            [h - 1, w - 1],
        ],
        dtype=float,
    )

    transformed = warp_points(corners, fwd_affine)

    # Shift to move content to (0, 0) if needed
    y_min, x_min = transformed.min(axis=0)
    shift = np.array(
        [
            [1, 0, -min(y_min, 0)],
            [0, 1, -min(x_min, 0)],
            [0, 0, 1],
        ]
    )
    adjusted = shift @ fwd_affine

    # Recompute corners with adjusted matrix
    transformed_adj = warp_points(corners, adjusted)
    y_max, x_max = transformed_adj.max(axis=0)

    h_out = int(np.ceil(y_max)) + 1
    w_out = int(np.ceil(x_max)) + 1
    return (h_out, w_out), adjusted


def arrange_warped(warped: list[np.ndarray], is_rgb: bool) -> np.ndarray:
    """Arrange warped images into a single array."""
    # stack image
    warped = np.dstack(warped)
    # ensure that RGB remains RGB but AF remain AF
    if warped.ndim == 3 and not is_rgb:
        # if warped.ndim == 3 and np.argmin(warped.shape) == 2 and not is_rgb:
        warped = np.moveaxis(warped, 2, 0)
    return warped


def _convert_array_to_numpy(array: np.ndarray) -> np.ndarray:
    """Convert array to numpy if it's a dask array."""
    with contextlib.suppress(ImportError):
        import dask.array as da

        if isinstance(array, da.Array):
            return array.compute()
    return array


def ensure_opencv_dtype(arr: np.ndarray) -> np.ndarray:
    """Convert arrays to OpenCV-compatible dtypes."""
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8)
    if arr.dtype in (np.uint8, np.uint16, np.float32, np.float64, np.int16, np.int32):
        return arr
    return arr.astype(np.float32)