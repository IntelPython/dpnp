import numpy
import pytest
from numpy.testing import assert_equal

import dpnp

from .helper import (
    assert_dtype_allclose,
    get_all_dtypes,
    get_complex_dtypes,
    get_float_complex_dtypes,
)


@pytest.mark.parametrize("func", ["abs", "absolute"])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
def test_abs(func, dtype):
    a = numpy.array([1, 0, 2, -3, -1, 2, 21, -9]).astype(dtype=dtype)
    ia = dpnp.array(a)

    result = getattr(dpnp, func)(ia)
    expected = getattr(numpy, func)(a)
    assert_dtype_allclose(result, expected)

    # out keyword
    dp_out = dpnp.empty(expected.shape, dtype=expected.dtype)
    result = getattr(dpnp, func)(ia, out=dp_out)
    assert result is dp_out
    assert_dtype_allclose(result, expected)


@pytest.mark.parametrize("stride", [-4, -2, -1, 1, 2, 4])
@pytest.mark.parametrize("dtype", get_complex_dtypes())
def test_abs_complex(stride, dtype):
    np_arr = numpy.array(
        [
            complex(numpy.nan, numpy.nan),
            complex(numpy.nan, numpy.inf),
            complex(numpy.inf, numpy.nan),
            complex(numpy.inf, numpy.inf),
            complex(0.0, numpy.inf),
            complex(numpy.inf, 0.0),
            complex(0.0, 0.0),
            complex(0.0, numpy.nan),
            complex(numpy.nan, 0.0),
        ],
        dtype=dtype,
    )
    dpnp_arr = dpnp.array(np_arr)
    assert_equal(numpy.abs(np_arr[::stride]), dpnp.abs(dpnp_arr[::stride]))


@pytest.mark.parametrize(
    "arraysize", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 18, 19]
)
@pytest.mark.parametrize("stride", [-4, -3, -2, -1, 1, 2, 3, 4])
@pytest.mark.parametrize("astype", get_complex_dtypes())
def test_abs_complex_avx(arraysize, stride, astype):
    np_arr = numpy.ones(arraysize, dtype=astype)
    dpnp_arr = dpnp.array(np_arr)
    assert_equal(numpy.abs(np_arr[::stride]), dpnp.abs(dpnp_arr[::stride]))


@pytest.mark.parametrize("dtype", get_float_complex_dtypes())
def test_abs_values(dtype):
    np_arr = numpy.array(
        [numpy.nan, -numpy.nan, numpy.inf, -numpy.inf, 0.0, -0.0, -1.0, 1.0],
        dtype=dtype,
    )
    dpnp_arr = dpnp.array(np_arr)
    assert_equal(numpy.abs(np_arr), dpnp.abs(dpnp_arr))
