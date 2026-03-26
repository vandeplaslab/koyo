"""Tests for koyo.transform."""

import numpy as np
import pytest

from koyo.transform import (
    ensure_opencv_dtype,
    get_shape_from_affine,
    invert_affine,
    transform_xy_coordinates,
    warp_points,
)


# ---------------------------------------------------------------------------
# transform_xy_coordinates
# ---------------------------------------------------------------------------


def test_transform_xy_coordinates_identity_yx():
    xy = np.array([[1.0, 2.0], [3.0, 4.0]])
    identity = np.eye(3)
    result = transform_xy_coordinates(xy, yx_affine=identity)
    np.testing.assert_allclose(result, xy)


def test_transform_xy_coordinates_identity_xy():
    xy = np.array([[1.0, 2.0], [3.0, 4.0]])
    identity = np.eye(3)
    result = transform_xy_coordinates(xy, xy_affine=identity)
    np.testing.assert_allclose(result, xy)


def test_transform_xy_coordinates_translation():
    xy = np.array([[0.0, 0.0], [1.0, 1.0]])
    # xy_affine uses row-vector convention: result = xy_h @ matrix
    # Translation matrix in row-vector form: [[1,0,0],[0,1,0],[tx,ty,1]]
    affine = np.array([[1, 0, 0], [0, 1, 0], [10, 20, 1]], dtype=float)
    result = transform_xy_coordinates(xy, xy_affine=affine)
    expected = np.array([[10.0, 20.0], [11.0, 21.0]])
    np.testing.assert_allclose(result, expected)


def test_transform_xy_coordinates_error_no_affine():
    xy = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="Either"):
        transform_xy_coordinates(xy)


def test_transform_xy_coordinates_error_both_affines():
    xy = np.array([[1.0, 2.0]])
    identity = np.eye(3)
    with pytest.raises(ValueError, match="Only one"):
        transform_xy_coordinates(xy, xy_affine=identity, yx_affine=identity)


# ---------------------------------------------------------------------------
# invert_affine
# ---------------------------------------------------------------------------


def test_invert_affine_identity():
    identity = np.eye(3)
    result = invert_affine(identity)
    np.testing.assert_allclose(result, identity)


def test_invert_affine_round_trip():
    affine = np.array([[2, 0, 5], [0, 3, -1], [0, 0, 1]], dtype=float)
    inv = invert_affine(affine)
    product = affine @ inv
    np.testing.assert_allclose(product, np.eye(3), atol=1e-10)


def test_invert_affine_wrong_shape():
    with pytest.raises(ValueError, match="Expected"):
        invert_affine(np.eye(4))


# ---------------------------------------------------------------------------
# warp_points
# ---------------------------------------------------------------------------


def test_warp_points_identity():
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = warp_points(pts, np.eye(3))
    np.testing.assert_allclose(result, pts, atol=1e-5)


def test_warp_points_translation():
    pts = np.array([[0.0, 0.0], [1.0, 1.0]])
    affine = np.array([[1, 0, 5], [0, 1, 3], [0, 0, 1]], dtype=float)
    result = warp_points(pts, affine)
    expected = np.array([[5.0, 3.0], [6.0, 4.0]])
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_warp_points_wrong_affine_shape():
    pts = np.array([[0.0, 0.0]])
    with pytest.raises(ValueError, match="Expected \\(3, 3\\)"):
        warp_points(pts, np.eye(4))


def test_warp_points_wrong_coords_shape():
    affine = np.eye(3)
    with pytest.raises(ValueError, match="Expected \\(N, 2\\)"):
        warp_points(np.array([0.0, 1.0, 2.0]), affine)


# ---------------------------------------------------------------------------
# get_shape_from_affine
# ---------------------------------------------------------------------------


def test_get_shape_from_affine_identity():
    shape = (100, 200)
    out_shape = get_shape_from_affine(shape, np.eye(3))
    assert out_shape == shape


def test_get_shape_from_affine_scale():
    shape = (10, 20)
    scale = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=float)
    h_out, w_out = get_shape_from_affine(shape, scale)
    # Scaled output should be approximately double
    assert h_out >= shape[0]
    assert w_out >= shape[1]


# ---------------------------------------------------------------------------
# ensure_opencv_dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype_in, dtype_out",
    [
        (np.bool_, np.uint8),
        (np.uint8, np.uint8),
        (np.uint16, np.uint16),
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.int16, np.int16),
        (np.int32, np.int32),
        (np.int64, np.float32),  # unsupported → float32
    ],
)
def test_ensure_opencv_dtype(dtype_in, dtype_out):
    arr = np.zeros((4, 4), dtype=dtype_in)
    result = ensure_opencv_dtype(arr)
    assert result.dtype == dtype_out
