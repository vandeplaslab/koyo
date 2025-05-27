"""zarr.py."""

import numpy as np
import zarr
from numcodecs import Blosc
from zarr.storage import ZipStore


def get_chunk_shape_along_axis(array: np.ndarray, axis: int = 0) -> tuple[int, int]:
    """Get chunk size."""
    if array.ndim == 1:
        return (array.shape[0],)
    elif array.ndim == 2:
        if axis == 0:
            return array.shape[0], 1
        elif axis == 1:
            return 1, array.shape[1]
    return 256, 256


def save_array_to_zip(array: np.ndarray, zip_path: str, chunk_size: tuple[int, int] = (256, 256)):
    """
    Save a 2D NumPy array to a .zip file using zarr + Blosc (zstd) compression.

    Parameters
    ----------
    - array: 2D NumPy array to store
    - zip_path: Path to output .zip file
    - chunk_size: Chunk shape for compression
    """
    if array.ndim != 2:
        raise ValueError("Only 2D arrays supported")

    # compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
    store = ZipStore(zip_path, mode="w")
    zarr.save_array(store, array, chunks=chunk_size, compressor=compressor)
    store.close()


def load_array_from_zip(zip_path: str) -> np.ndarray:
    """
    Load a 2D NumPy array from a compressed .zip file stored with Zarr.

    Parameters
    ----------
    - zip_path: Path to the .zip file

    Returns
    -------
    - The decompressed NumPy array
    """
    store = ZipStore(zip_path, mode="r")
    array = zarr.open(store, mode="r")
    result = array[:]
    store.close()
    return result
