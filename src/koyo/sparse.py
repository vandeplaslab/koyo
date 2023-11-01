import numpy as np
import scipy.sparse as sparse


def get_index_dtype(arrays=(), maxval=None, check_contents=False):
    """
    Based on input (integer) arrays `a`, determine a suitable index data
    type that can hold the data in the arrays.

    Parameters
    ----------
    arrays : tuple of array_like
        Input arrays whose types/contents to check
    maxval : float, optional
        Maximum value needed
    check_contents : bool, optional
        Whether to check the values in the arrays and not just their types.
        Default: False (check only the types)

    Returns
    -------
    dtype : dtype
        Suitable index data type (int32 or int64)

    """
    int32min = np.iinfo(np.int32).min
    int32max = np.iinfo(np.int32).max

    dtype = np.intc
    if maxval is not None:
        if maxval > int32max:
            dtype = np.int64

    if isinstance(arrays, np.ndarray):
        arrays = (arrays,)

    for arr in arrays:
        arr = np.asarray(arr)
        if not np.can_cast(arr.dtype, np.int32):
            if check_contents:
                if arr.size == 0:
                    # a bigger type not needed
                    continue
                elif np.issubdtype(arr.dtype, np.integer):
                    maxval = arr.max()
                    minval = arr.min()
                    if minval >= int32min and maxval <= int32max:
                        # a bigger type not needed
                        continue

            dtype = np.int64
            break

    return dtype


def asindices(x):
    try:
        x = np.asarray(x)

        # Check index contents to avoid creating 64bit arrays needlessly
        idx_dtype = get_index_dtype((x,), check_contents=True)
        if idx_dtype != x.dtype:
            x = x.astype(idx_dtype)
    except Exception:
        raise IndexError("invalid index")
    else:
        return x


def check_bounds(indices, N):
    if indices.size == 0:
        return (0, 0)

    max_indx = indices.max()
    if max_indx >= N:
        raise IndexError("index (%d) out of range" % max_indx)

    min_indx = indices.min()
    if min_indx < -N:
        raise IndexError("index (%d) out of range" % (N + min_indx))

    return min_indx, max_indx


def check_ellipsis(index):
    """Process indices with Ellipsis. Returns modified index."""
    if index is Ellipsis:
        return (slice(None), slice(None))
    elif isinstance(index, tuple):
        # Find first ellipsis
        for j, v in enumerate(index):
            if v is Ellipsis:
                first_ellipsis = j
                break
        else:
            first_ellipsis = None

        # Expand the first one
        if first_ellipsis is not None:
            # Shortcuts
            if len(index) == 1:
                return (slice(None), slice(None))
            elif len(index) == 2:
                if first_ellipsis == 0:
                    if index[1] is Ellipsis:
                        return (slice(None), slice(None))
                    else:
                        return (slice(None), index[1])
                else:
                    return (index[0], slice(None))

            # General case
            tail = ()
            for v in index[first_ellipsis + 1 :]:
                if v is not Ellipsis:
                    tail = (*tail, v)
            nd = first_ellipsis + len(tail)
            nslice = max(0, 2 - nd)
            return index[:first_ellipsis] + (slice(None),) * nslice + tail
    return index


def unpack_index(index):
    index = check_ellipsis(index)

    if isinstance(index, tuple):
        if len(index) == 2:
            row, col = index
        elif len(index) == 1:
            row, col = index[0], slice(None)
    else:
        row, col = index, slice(None)

    return row, col


def extractor(indices, N, fmt="csr"):
    """
    row = np.arange(500)
    P = extractor(row, shape[0]).
    """
    indices = asindices(indices).copy()

    min_indx, __ = check_bounds(indices, N)

    if min_indx < 0:
        indices[indices < 0] += N

    indptr = np.arange(len(indices) + 1, dtype=indices.dtype)
    data = np.ones(len(indices), dtype=np.int32)
    shape = (len(indices), N)
    return sparse.csr_matrix((data, indices, indptr), shape=shape, dtype=np.int32, copy=False).asformat(fmt)


def get_array_statistics(array) -> str:
    """Calculate array statistics."""
    from koyo.utilities import calculate_array_size

    statistics: str = calculate_array_sparsity(array)
    size: str = calculate_array_size(array)
    dense_size: str = estimate_dense_size(array)

    return statistics + " | " + size + " | [As dense: " + dense_size + "]"


def estimate_dense_size(array) -> str:
    """Calculate approximate size of the array if it was dense."""
    from koyo.utilities import format_size

    total_val = float(np.prod(array.shape))
    n_bytes = total_val * array.dtype.itemsize
    return format_size(n_bytes)


def calculate_array_sparsity(array):
    """Calculate the sparsity and density of an array."""
    if hasattr(array, "nnz"):
        non_zero = array.nnz
        fmt = array.format
    else:
        non_zero = np.count_nonzero(array)
        fmt = "dense"
    total_val = np.prod(array.shape)
    sparsity = (total_val - non_zero) / total_val
    density = non_zero / total_val
    return f"Sparsity: {sparsity * 100:.4f}% | Density {density * 100:.4f}% | NNZ: {non_zero} | FMT: {fmt}"
