import pytest

import dpnp

import numpy

@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32, numpy.complex64, numpy.complex128],
                         ids=['float64', 'float32', 'int64', 'int32', 'complex64', 'complex128'])
def test_fft(type):
    # 1 dim array
    data = numpy.arange(100, dtype=type)
    # TODO:
    # doesn't work correct with `complex64` (not supported)
    # dpnp_data = dpnp.arange(100, dtype=type)
    dpnp_data = dpnp.array(data)

    np_res = numpy.fft.fft(data)
    dpnp_res = dpnp.fft.fft(dpnp_data)

    numpy.testing.assert_array_equal(dpnp_res, np_res)
    assert dpnp_res.dtype == np_res.dtype
