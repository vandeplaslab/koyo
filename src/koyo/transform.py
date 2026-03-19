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

    ones = np.ones((len(yx), 1), dtype=yx.dtype)
    coords_h = np.hstack([yx, ones])  # (N, 3)
    result_h = (fwd_affine @ coords_h.T).T  # (N, 3)
    return result_h[:, :2] / result_h[:, 2:3]  # de-homogenise


def warp_channel(
    image: np.ndarray, affine_inv: np.ndarray, output_shape: tuple[int, int], order: int = 1, silent: bool = False
) -> np.ndarray:
    """Warp image."""
    import cv2
    from scipy.ndimage import affine_transform

    assert image.ndim == 2, "Only 2D images are supported for warping in this function."
    use_cv2 = max(max(image.shape), max(output_shape)) < 32767
    if not silent:
        logger.trace(f"Using {'cv2' if use_cv2 else 'scipy'} for warping.")
    if use_cv2:
        image = _convert_array_to_numpy(image)
        warped = cv2.warpAffine(
            image.T,
            invert_affine(affine_inv)[:2, :],
            output_shape,
            flags=cv2.INTER_NEAREST if order == 0 else cv2.INTER_LINEAR,
        ).T
    else:
        warped = affine_transform(image, affine_inv, order=order, output_shape=output_shape)
    return warped


def _convert_array_to_numpy(array: np.ndarray) -> np.ndarray:
    """Convert array to numpy if it's a dask array."""
    with contextlib.suppress(ImportError):
        import dask.array as da

        if isinstance(array, da.Array):
            return array.compute()
    return array
