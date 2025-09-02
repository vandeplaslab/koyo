"""Test zarr functionality."""

import numpy as np
import pytest

from koyo.utilities import is_installed
from koyo.zarr import get_chunk_shape_along_axis, load_array_from_zip, save_array_to_zip


@pytest.mark.skipif(not is_installed("zarr"), reason="zarr not installed")
def test_get_chunk_shape_along_axis():
    """Test chunk shape retrieval."""
    array_1d = np.zeros(10)
    assert get_chunk_shape_along_axis(array_1d, 0) == (10,)

    array_2d = np.zeros((10, 20))
    assert get_chunk_shape_along_axis(array_2d, 0) == (10, 1)
    assert get_chunk_shape_along_axis(array_2d, 1) == (1, 20)


@pytest.mark.skipif(not is_installed("zarr"), reason="zarr not installed")
def test_save_load_array_to_zip(tmp_path):
    """Test saving and loading a 2D array to/from a zip file."""
    array = np.random.rand(100, 200)
    zip_path = tmp_path / "test_array.zip"

    # Save the array
    save_array_to_zip(array, str(zip_path), chunk_size=(50, 50))
    assert zip_path.exists(), "Zip file was not created"

    # Load the array back
    loaded_array = load_array_from_zip(str(zip_path))

    # Check if the loaded array matches the original
    np.testing.assert_array_equal(array, loaded_array)


@pytest.mark.skipif(not is_installed("zarr"), reason="zarr not installed")
def test_load_array_from_zip_nonexistent():
    """Test loading from a non-existent zip file."""
    with pytest.raises(FileNotFoundError):
        load_array_from_zip("non_existent.zip")
