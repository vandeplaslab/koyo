"""zarr.py."""

from __future__ import annotations

import numpy as np

from koyo.system import is_installed

HAS_ZARR = is_installed("zarr")

if HAS_ZARR:
    import zarr
    from zarr.storage import ZipStore

    try:
        from zarr.codecs import BloscCodec, BloscShuffle
    except ImportError:  # Zarr 2
        from numcodecs import Blosc

        BloscCodec = BloscShuffle = None
    else:
        Blosc = None
else:
    zarr = Blosc = BloscCodec = BloscShuffle = ZipStore = None


def get_chunk_shape_along_axis(array: np.ndarray, axis: int = 0) -> tuple[int, int]:
    """Get chunk size."""
    if array.ndim == 1:
        return (array.shape[0],)
    if array.ndim == 2:
        if axis == 0:
            return array.shape[0], 1
        if axis == 1:
            return 1, array.shape[1]
    return 256, 256


def save_array_to_zip(array: np.ndarray, zip_path: str, chunk_size: tuple[int, int] = (256, 256)) -> None:
    """Save a 2D NumPy array to a ZIP file using Zarr and Blosc compression.

    Parameters
    ----------
    array : np.ndarray
        Two-dimensional array to store.
    zip_path : str
        Path to the output ZIP file.
    chunk_size : tuple[int, int]
        Chunk shape used for compression.
    """
    if not HAS_ZARR:
        raise ImportError("zarr is not installed. Please install it to use this function.")
    if array.ndim != 2:
        raise ValueError("Only 2D arrays supported")

    with ZipStore(zip_path, mode="w") as store:
        if BloscCodec is not None:
            compressor = BloscCodec(cname="lz4", clevel=5, shuffle=BloscShuffle.bitshuffle)
            output = zarr.create_array(
                store=store,
                shape=array.shape,
                dtype=array.dtype,
                chunks=chunk_size,
                compressors=compressor,
            )
            output[:] = array
        else:
            compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
            zarr.save_array(store, array, chunks=chunk_size, compressor=compressor)


def load_array_from_zip(zip_path: str) -> np.ndarray:
    """Load a NumPy array from a compressed Zarr ZIP file.

    Parameters
    ----------
    zip_path : str
        Path to the ZIP file.

    Returns
    -------
    np.ndarray
        The decompressed NumPy array.
    """
    if not HAS_ZARR:
        raise ImportError("zarr is not installed. Please install it to use this function.")

    zarr.errors.ArrayNotFoundError
    store = ZipStore(zip_path, mode="r")
    array = zarr.open_array(store, mode="r")
    try:
        return array[:]
    finally:
        store.close()
