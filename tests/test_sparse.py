import numpy as np
import pytest
from koyo.sparse import get_array_statistics
from scipy import sparse


@pytest.mark.parametrize(
    "array",
    (
        np.zeros((10, 10)),
        np.ones((10, 10)),
        sparse.csr_matrix((3, 4), dtype=np.int8),
        sparse.csc_matrix((3, 4), dtype=np.int8),
    ),
)
def test_get_array_statistics(array):
    result = get_array_statistics(array)
    assert isinstance(result, str)
    assert "Sparsity" in result
