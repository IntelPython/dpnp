import math
import pytest
from .helper import get_all_dtypes

import dpnp

import numpy
from numpy.testing import (
    assert_allclose
)


def _getattr(ex, str_):
    attrs = str_.split(".")
    res = ex
    for attr in attrs:
        res = getattr(res, attr)
    return res


@pytest.mark.parametrize("func_name",
                         ['abs', ])
@pytest.mark.parametrize("dtype", get_all_dtypes())
def test_strides(func_name, dtype):
    shape = (4, 4)
    a = numpy.ones(shape[0] * shape[1], dtype=dtype).reshape(shape)
    a_strides = a[0::2, 0::2]
    dpa = dpnp.array(a)
    dpa_strides = dpa[0::2, 0::2]

    dpnp_func = _getattr(dpnp, func_name)
    result = dpnp_func(dpa_strides)

    numpy_func = _getattr(numpy, func_name)
    expected = numpy_func(a_strides)

    assert_allclose(expected, result)


@pytest.mark.usefixtures("suppress_divide_invalid_numpy_warnings")
@pytest.mark.parametrize("func_name",
                         ["arccos", "arccosh", "arcsin", "arcsinh", "arctan", "arctanh", "cbrt", "ceil", "copy", "cos",
                          "cosh", "conjugate", "degrees", "ediff1d", "exp", "exp2", "expm1", "fabs", "floor", "log",
                          "log10", "log1p", "log2", "negative", "radians", "sign", "sin", "sinh", "sqrt", "square",
                          "tanh", "trunc"])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True))
@pytest.mark.parametrize("shape",
                         [(10,)],
                         ids=["(10,)"])
def test_strides_1arg(func_name, dtype, shape):
    a = numpy.arange(numpy.prod(shape), dtype=dtype).reshape(shape)
    b = a[::2]

    dpa = dpnp.reshape(dpnp.arange(numpy.prod(shape), dtype=dtype), shape)
    dpb = dpa[::2]

    dpnp_func = _getattr(dpnp, func_name)
    result = dpnp_func(dpb)

    numpy_func = _getattr(numpy, func_name)
    expected = numpy_func(b)

    assert_allclose(result, expected)


@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True))
@pytest.mark.parametrize("shape",
                         [(10,)],
                         ids=["(10,)"])
def test_strides_erf(dtype, shape):
    a = numpy.arange(numpy.prod(shape), dtype=dtype).reshape(shape)
    b = a[::2]

    dpa = dpnp.reshape(dpnp.arange(numpy.prod(shape), dtype=dtype), shape)
    dpb = dpa[::2]

    result = dpnp.erf(dpb)

    expected = numpy.empty_like(b)
    for idx, val in enumerate(b):
        expected[idx] = math.erf(val)

    assert_allclose(result, expected)


@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True))
@pytest.mark.parametrize("shape",
                         [(10,)],
                         ids=["(10,)"])
def test_strides_reciprocal(dtype, shape):
    start, stop = 1, numpy.prod(shape) + 1

    a = numpy.arange(start, stop, dtype=dtype).reshape(shape)
    b = a[::2]

    dpa = dpnp.reshape(dpnp.arange(start, stop, dtype=dtype), shape)
    dpb = dpa[::2]

    result = dpnp.reciprocal(dpb)
    expected = numpy.reciprocal(b)

    assert_allclose(result, expected, rtol=1e-06)


@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True))
@pytest.mark.parametrize("shape",
                         [(10,)],
                         ids=["(10,)"])
def test_strides_tan(dtype, shape):
    a = numpy.arange(numpy.prod(shape), dtype=dtype).reshape(shape)
    b = a[::2]

    dpa = dpnp.reshape(dpnp.arange(numpy.prod(shape), dtype=dtype), shape)
    dpb = dpa[::2]

    result = dpnp.tan(dpb)
    expected = numpy.tan(b)

    assert_allclose(result, expected, rtol=1e-06)


@pytest.mark.parametrize("func_name",
                         ["add", "arctan2", "hypot", "maximum", "minimum", "multiply", "power", "subtract"])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True))
@pytest.mark.parametrize("shape",
                         [(3, 3)],
                         ids=["(3, 3)"])
def test_strides_2args(func_name, dtype, shape):
    a = numpy.arange(numpy.prod(shape), dtype=dtype).reshape(shape)
    b = a.T

    dpa = dpnp.reshape(dpnp.arange(numpy.prod(shape), dtype=dtype), shape)
    dpb = dpa.T

    dpnp_func = _getattr(dpnp, func_name)
    result = dpnp_func(dpa, dpb)

    numpy_func = _getattr(numpy, func_name)
    expected = numpy_func(a, b)

    assert_allclose(result, expected)


@pytest.mark.parametrize("func_name",
                         ["bitwise_and", "bitwise_or", "bitwise_xor", "left_shift", "right_shift"])
@pytest.mark.parametrize("dtype",
                         [numpy.int64, numpy.int32],
                         ids=["int64", "int32"])
@pytest.mark.parametrize("shape",
                         [(3, 3)],
                         ids=["(3, 3)"])
def test_strides_bitwise(func_name, dtype, shape):
    a = numpy.arange(numpy.prod(shape), dtype=dtype).reshape(shape)
    b = a.T

    dpa = dpnp.reshape(dpnp.arange(numpy.prod(shape), dtype=dtype), shape)
    dpb = dpa.T

    dpnp_func = _getattr(dpnp, func_name)
    result = dpnp_func(dpa, dpb)

    numpy_func = _getattr(numpy, func_name)
    expected = numpy_func(a, b)

    assert_allclose(result, expected)


@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True))
@pytest.mark.parametrize("shape",
                         [(3, 3)],
                         ids=["(3, 3)"])
def test_strides_copysign(dtype, shape):
    a = numpy.arange(numpy.prod(shape), dtype=dtype).reshape(shape)
    b = -a.T

    dpa = dpnp.reshape(dpnp.arange(numpy.prod(shape), dtype=dtype), shape)
    dpb = dpnp.negative(dpa.T)

    result = dpnp.copysign(dpa, dpb)
    expected = numpy.copysign(a, b)

    assert_allclose(result, expected)


@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True))
@pytest.mark.parametrize("shape",
                         [(3, 3)],
                         ids=["(3, 3)"])
def test_strides_fmod(dtype, shape):
    a = numpy.arange(numpy.prod(shape), dtype=dtype).reshape(shape)
    b = a.T + 1

    dpa = dpnp.reshape(dpnp.arange(numpy.prod(shape), dtype=dtype), shape)
    dpb = dpa.T + 1

    result = dpnp.fmod(dpa, dpb)
    expected = numpy.fmod(a, b)

    assert_allclose(result, expected)


@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True))
@pytest.mark.parametrize("shape",
                         [(3, 3)],
                         ids=["(3, 3)"])
def test_strides_true_divide(dtype, shape):
    a = numpy.arange(numpy.prod(shape), dtype=dtype).reshape(shape)
    b = a.T + 1

    dpa = dpnp.reshape(dpnp.arange(numpy.prod(shape), dtype=dtype), shape)
    dpb = dpa.T + 1

    result = dpnp.true_divide(dpa, dpb)
    expected = numpy.true_divide(a, b)

    assert_allclose(result, expected)


@pytest.mark.parametrize("func_name",
                         ["add", "multiply", "power"])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True))
def test_strided_out_2args(func_name, dtype):
    np_out = numpy.ones((5, 3, 2))[::3]
    np_a = numpy.arange(numpy.prod(np_out.shape), dtype=dtype).reshape(np_out.shape)
    np_b = numpy.full(np_out.shape, fill_value=0.7, dtype=dtype)

    dp_out = dpnp.ones((5, 3, 2))[::3]
    dp_a = dpnp.array(np_a)
    dp_b = dpnp.array(np_b)

    np_res = _getattr(numpy, func_name)(np_a, np_b, out=np_out)
    dp_res = _getattr(dpnp, func_name)(dp_a, dp_b, out=dp_out)

    assert_allclose(dp_res.asnumpy(), np_res)
    assert_allclose(dp_out.asnumpy(), np_out)


@pytest.mark.parametrize("func_name",
                         ["add", "multiply", "power"])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True))
def test_strided_in_out_2args(func_name, dtype):
    sh = (3, 4, 2)
    prod = numpy.prod(sh)

    np_out = numpy.ones(sh, dtype=dtype)[::2]
    np_a = numpy.arange(prod, dtype=dtype).reshape(sh)[::2]
    np_b = numpy.full(sh, fill_value=0.7, dtype=dtype)[::2].T

    dp_out = dpnp.ones(sh, dtype=dtype)[::2]
    dp_a = dpnp.arange(prod, dtype=dtype).reshape(sh)[::2]
    dp_b = dpnp.full(sh, fill_value=0.7, dtype=dtype)[::2].T

    np_res = _getattr(numpy, func_name)(np_a, np_b, out=np_out)
    dp_res = _getattr(dpnp, func_name)(dp_a, dp_b, out=dp_out)

    assert_allclose(dp_res.asnumpy(), np_res, rtol=1e-06)
    assert_allclose(dp_out.asnumpy(), np_out, rtol=1e-06)


@pytest.mark.parametrize("func_name",
                         ["add", "multiply", "power"])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True))
def test_strided_in_out_2args_diff_out_dtype(func_name, dtype):
    sh = (3, 3, 2)
    prod = numpy.prod(sh)

    np_out = numpy.ones(sh, dtype=numpy.complex64)[::2]
    np_a = numpy.arange(prod, dtype=dtype).reshape(sh)[::2].T
    np_b = numpy.full(sh, fill_value=0.7, dtype=dtype)[::2]

    dp_out = dpnp.ones(sh, dtype=dpnp.complex64)[::2]
    dp_a = dpnp.arange(prod, dtype=dtype).reshape(sh)[::2].T
    dp_b = dpnp.full(sh, fill_value=0.7, dtype=dtype)[::2]

    np_res = _getattr(numpy, func_name)(np_a, np_b, out=np_out)
    dp_res = _getattr(dpnp, func_name)(dp_a, dp_b, out=dp_out)

    assert_allclose(dp_res.asnumpy(), np_res, rtol=1e-06)
    assert_allclose(dp_out.asnumpy(), np_out, rtol=1e-06)


@pytest.mark.parametrize("func_name",
                         ["add", "multiply", "power"])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True, no_none=True))
def test_strided_in_2args_overlap(func_name, dtype):
    size = 5

    np_a = numpy.arange(2 * size, dtype=dtype)
    dp_a = dpnp.arange(2 * size, dtype=dtype)

    np_res = _getattr(numpy, func_name)(np_a[size::], np_a[::2], out=np_a[:size:])
    dp_res = _getattr(dpnp, func_name)(dp_a[size::], dp_a[::2], out=dp_a[:size:])

    assert_allclose(dp_res.asnumpy(), np_res, rtol=1e-06)
    assert_allclose(dp_a.asnumpy(), np_a, rtol=1e-06)


@pytest.mark.parametrize("func_name",
                         ["add", "multiply", "power"])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True, no_none=True))
def test_strided_in_out_2args_overlap(func_name, dtype):
    sh = (4, 3, 2)
    prod = numpy.prod(sh)

    np_a = numpy.arange(prod, dtype=dtype).reshape(sh)
    np_b = numpy.full(np_a[::2].shape, fill_value=0.7, dtype=dtype)

    dp_a = dpnp.arange(prod, dtype=dtype).reshape(sh)
    dp_b = dpnp.full(dp_a[::2].shape, fill_value=0.7, dtype=dtype)

    np_res = _getattr(numpy, func_name)(np_a[::2], np_b, out=np_a[1::2])
    dp_res = _getattr(dpnp, func_name)(dp_a[::2], dp_b, out=dp_a[1::2])

    assert_allclose(dp_res.asnumpy(), np_res, rtol=1e-06)
    assert_allclose(dp_a.asnumpy(), np_a, rtol=1e-06)
