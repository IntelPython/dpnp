import pytest

import dpnp

import numpy


@pytest.mark.parametrize("dtype", ['complex128', 'complex64', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize("shape", [(100,), (10, 10), (2, 5, 10), (2, 5, 2, 5)], ids=['1dim', '2dim', '3dim', '4dim'])
def test_fft(dtype, shape):
    # 1 dim array
    data = numpy.arange(100, dtype=numpy.dtype(dtype)).reshape(shape)
    # TODO:
    # doesn't work correct with `complex64` (not supported)
    # dpnp_data = dpnp.arange(100, dtype=dpnp.dtype(type))
    dpnp_data = dpnp.array(data)

    np_res = numpy.fft.fft(data)
    dpnp_res = dpnp.asnumpy(dpnp.fft.fft(dpnp_data))

    numpy.testing.assert_allclose(dpnp_res, np_res, rtol=1e-4, atol=1e-7)
    assert dpnp_res.dtype == np_res.dtype
