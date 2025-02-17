"""Affine transformation functions."""

from __future__ import annotations

import numpy as np


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
