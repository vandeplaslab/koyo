from koyo.spectrum import ppm_error


def test_ppm_error():
    ppm = ppm_error(100, 100)
    assert ppm == 0
