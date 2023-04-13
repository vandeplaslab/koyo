import numpy as np
import pytest
from koyo.numpy import nanquantile_along_axis, quantile_along_axis


@pytest.mark.parametrize("axis", (0, 1))
@pytest.mark.parametrize("array", (np.random.rand(100, 10), np.random.rand(10, 100)))
def test_nanquantile_along_axis(array, axis):
    result = nanquantile_along_axis(array, 0.25, 0.75, axis)
    assert result.shape == (2, array.shape[1 - axis])
    q1, q2 = np.nanquantile(array, (0.25, 0.75), axis=axis)
    assert np.allclose(result[0], q1)
    assert np.allclose(result[1], q2)


@pytest.mark.parametrize("axis", (0, 1))
@pytest.mark.parametrize("array", (np.random.rand(100, 10), np.random.rand(10, 100)))
def test_quantile_along_axis(array, axis):
    result = quantile_along_axis(array, 0.25, 0.75, axis)
    assert result.shape == (2, array.shape[1 - axis])
    q1, q2 = np.quantile(array, (0.25, 0.75), axis=axis)
    assert np.allclose(result[0], q1)
    assert np.allclose(result[1], q2)
