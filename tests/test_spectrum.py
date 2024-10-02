from koyo.spectrum import ppm_error, ppm_diff, get_ppm_axis
import numpy as np


def test_ppm_error():
    ppm = ppm_error(100, 100)
    assert ppm == 0


def test_ppm_diff():
    diff = ppm_diff([100, 100.0001])
    assert diff.size == 1
    np.testing.assert_almost_equal(diff[0], 1, 1)

    diff = ppm_diff([1000, 1000.01])
    assert diff.size == 1
    np.testing.assert_almost_equal(diff[0], 10, 1)


def test_ppm_axis():
    # get new axis
    mz = get_ppm_axis(100, 200, 1)
    # get ppm spacing between each bin
    diff = ppm_diff(mz)
    # make sure that ppm spacing is within 1 dp
    np.testing.assert_almost_equal(diff, 1, 1)

    mz = get_ppm_axis(100, 200, 5)
    # get ppm spacing between each bin
    diff = ppm_diff(mz)
    # make sure that ppm spacing is within 1 dp
    np.testing.assert_almost_equal(diff, 5, 1)
