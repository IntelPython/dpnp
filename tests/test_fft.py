import pytest

import dpnp

import numpy


@pytest.mark.parametrize("type", ['complex128', 'complex64', 'float32', 'float64', 'int32', 'int64'])
def test_fft(type):
    # 1 dim array
    data = numpy.arange(100, dtype=numpy.dtype(type))
    # TODO:
    # doesn't work correct with `complex64` (not supported)
    # dpnp_data = dpnp.arange(100, dtype=dpnp.dtype(type))
    dpnp_data = dpnp.array(data)

    np_res = numpy.fft.fft(data)
    dpnp_res = dpnp.asnumpy(dpnp.fft.fft(dpnp_data))

    assert dpnp_res.dtype == np_res.dtype
    numpy.testing.assert_allclose(dpnp_res, np_res, rtol=1e-4, atol=1e-7)


# TODO:
# will be removed
@pytest.mark.parametrize("type", ['complex128', 'complex64', 'float32', 'float64'])
def test_fft1(type):
    # 1 dim array
    data = numpy.arange(100, dtype=numpy.dtype(type))
    # TODO:
    # doesn't work correct with `complex64` (not supported)
    # dpnp_data = dpnp.arange(100, dtype=dpnp.dtype(type))
    dpnp_data = dpnp.array(data)

    np_res = numpy.fft.fft(data)
    dpnp_res = dpnp.asnumpy(dpnp.fft.fft(dpnp_data))

    assert dpnp_res.dtype == np_res.dtype
    numpy.testing.assert_allclose(dpnp_res[:30], np_res[:30], rtol=1e-4, atol=1e-7)


# @pytest.mark.parametrize("type", ['complex128', 'complex64', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize("type", ['complex128', 'complex64', 'float32', 'float64'])
def test_ifft(type):
    # 1 dim array
    data = numpy.arange(100, dtype=numpy.dtype(type))
    # TODO:
    # doesn't work correct with `complex64` (not supported)
    # dpnp_data = dpnp.arange(100, dtype=dpnp.dtype(type))
    dpnp_data = dpnp.array(data)

    np_res = numpy.fft.ifft(data)
    dpnp_res = dpnp.asnumpy(dpnp.fft.ifft(dpnp_data))

    assert dpnp_res.dtype == np_res.dtype
    numpy.testing.assert_allclose(dpnp_res, np_res, rtol=1e-4, atol=1e-7)


# TODO:
# will be removed
@pytest.mark.parametrize("type", ['float32', 'float64'])
def test_ifft1(type):
    # 1 dim array
    data = numpy.arange(100, dtype=numpy.dtype(type))
    # TODO:
    # doesn't work correct with `complex64` (not supported)
    # dpnp_data = dpnp.arange(100, dtype=dpnp.dtype(type))
    dpnp_data = dpnp.array(data)

    np_res = numpy.fft.ifft(data)
    dpnp_res = dpnp.asnumpy(dpnp.fft.ifft(dpnp_data))

    assert dpnp_res.dtype == np_res.dtype
    numpy.testing.assert_allclose(dpnp_res[:30], np_res[:30], rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize("type", ['complex128', 'complex64'])
def test_irfft(type):
    # 1 dim array
    data = numpy.arange(100, dtype=numpy.dtype(type))
    # TODO:
    # doesn't work correct with `complex64` (not supported)
    # dpnp_data = dpnp.arange(100, dtype=dpnp.dtype(type))
    dpnp_data = dpnp.array(data)

    np_res = numpy.fft.irfft(data)
    dpnp_res = dpnp.asnumpy(dpnp.fft.irfft(dpnp_data))

    assert dpnp_res.dtype == np_res.dtype
    assert dpnp_res.size == np_res.size
    numpy.testing.assert_allclose(dpnp_res, np_res, rtol=1e-4, atol=1e-7)


# TODO:
# redesign tests