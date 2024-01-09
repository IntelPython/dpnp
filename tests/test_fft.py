import numpy
import pytest

import dpnp

from .helper import assert_dtype_allclose, get_all_dtypes


@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
@pytest.mark.parametrize("norm", [None, "forward", "ortho"])
def test_fft(dtype, norm):
    # 1 dim array
    data = numpy.arange(100, dtype=dtype)
    dpnp_data = dpnp.array(data)

    np_res = numpy.fft.fft(data, norm=norm)
    dpnp_res = dpnp.fft.fft(dpnp_data, norm=norm)

    assert_dtype_allclose(dpnp_res, np_res, check_only_type_kind=True)


@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
@pytest.mark.parametrize("shape", [(8, 8), (4, 16), (4, 4, 4), (2, 4, 4, 2)])
@pytest.mark.parametrize("norm", [None, "forward", "ortho"])
def test_fft_ndim(dtype, shape, norm):
    np_data = numpy.arange(64, dtype=dtype).reshape(shape)
    dpnp_data = dpnp.arange(64, dtype=dtype).reshape(shape)

    np_res = numpy.fft.fft(np_data, norm=norm)
    dpnp_res = dpnp.fft.fft(dpnp_data, norm=norm)

    assert_dtype_allclose(dpnp_res, np_res, check_only_type_kind=True)


@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
@pytest.mark.parametrize(
    "shape", [(64,), (8, 8), (4, 16), (4, 4, 4), (2, 4, 4, 2)]
)
@pytest.mark.parametrize("norm", [None, "forward", "ortho"])
def test_fft_ifft(dtype, shape, norm):
    np_data = numpy.arange(64, dtype=dtype).reshape(shape)
    dpnp_data = dpnp.arange(64, dtype=dtype).reshape(shape)

    np_res = numpy.fft.ifft(np_data, norm=norm)
    dpnp_res = dpnp.fft.ifft(dpnp_data, norm=norm)

    assert_dtype_allclose(dpnp_res, np_res, check_only_type_kind=True)


@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True))
@pytest.mark.parametrize(
    "shape", [(64,), (8, 8), (4, 16), (4, 4, 4), (2, 4, 4, 2)]
)
def test_fft_rfft(dtype, shape):
    np_data = numpy.arange(64, dtype=dtype).reshape(shape)
    dpnp_data = dpnp.arange(64, dtype=dtype).reshape(shape)

    np_res = numpy.fft.rfft(np_data)
    dpnp_res = dpnp.fft.rfft(dpnp_data)

    assert_dtype_allclose(dpnp_res, np_res, check_only_type_kind=True)


@pytest.mark.parametrize(
    "func_name",
    [
        "fft",
        "ifft",
        "rfft",
    ],
)
def test_fft_invalid_dtype(func_name):
    a = dpnp.array([True, False, True])
    dpnp_func = getattr(dpnp.fft, func_name)
    with pytest.raises(TypeError):
        dpnp_func(a)
