import numpy
import pytest

import dpnp

from .helper import assert_dtype_allclose, has_support_aspect64

pytestmark = pytest.mark.skipif(
    not has_support_aspect64(), reason="Aborted on Iris Xe: SAT-6028"
)


@pytest.mark.parametrize(
    "type", ["complex128", "complex64", "float32", "float64", "int32", "int64"]
)
@pytest.mark.parametrize("norm", [None, "forward", "ortho"])
def test_fft(type, norm):
    # 1 dim array
    data = numpy.arange(100, dtype=numpy.dtype(type))
    dpnp_data = dpnp.array(data)

    np_res = numpy.fft.fft(data, norm=norm)
    dpnp_res = dpnp.fft.fft(dpnp_data, norm=norm)

    assert_dtype_allclose(dpnp_res, np_res)


@pytest.mark.parametrize(
    "type", ["complex128", "complex64", "float32", "float64", "int32", "int64"]
)
@pytest.mark.parametrize("shape", [(8, 8), (4, 16), (4, 4, 4), (2, 4, 4, 2)])
@pytest.mark.parametrize("norm", [None, "forward", "ortho"])
def test_fft_ndim(type, shape, norm):
    np_data = numpy.arange(64, dtype=numpy.dtype(type)).reshape(shape)
    dpnp_data = dpnp.arange(64, dtype=numpy.dtype(type)).reshape(shape)

    np_res = numpy.fft.fft(np_data, norm=norm)
    dpnp_res = dpnp.fft.fft(dpnp_data, norm=norm)

    assert_dtype_allclose(dpnp_res, np_res)


@pytest.mark.parametrize(
    "type", ["complex128", "complex64", "float32", "float64", "int32", "int64"]
)
@pytest.mark.parametrize(
    "shape", [(64,), (8, 8), (4, 16), (4, 4, 4), (2, 4, 4, 2)]
)
@pytest.mark.parametrize("norm", [None, "forward", "ortho"])
def test_fft_ifft(type, shape, norm):
    np_data = numpy.arange(64, dtype=numpy.dtype(type)).reshape(shape)
    dpnp_data = dpnp.arange(64, dtype=numpy.dtype(type)).reshape(shape)

    np_res = numpy.fft.ifft(np_data, norm=norm)
    dpnp_res = dpnp.fft.ifft(dpnp_data, norm=norm)

    assert_dtype_allclose(dpnp_res, np_res)


@pytest.mark.parametrize("type", ["float32", "float64", "int32", "int64"])
@pytest.mark.parametrize(
    "shape", [(64,), (8, 8), (4, 16), (4, 4, 4), (2, 4, 4, 2)]
)
def test_fft_rfft(type, shape):
    np_data = numpy.arange(64, dtype=numpy.dtype(type)).reshape(shape)
    dpnp_data = dpnp.arange(64, dtype=numpy.dtype(type)).reshape(shape)

    np_res = numpy.fft.rfft(np_data)
    dpnp_res = dpnp.fft.rfft(dpnp_data)

    assert_dtype_allclose(dpnp_res, np_res)
