import math

import numpy
import pytest
from numpy.testing import assert_array_equal

import dpnp

from .helper import (
    assert_dtype_allclose,
    generate_random_numpy_array,
    get_all_dtypes,
    get_complex_dtypes,
    get_float_complex_dtypes,
    get_integer_dtypes,
    get_integer_float_dtypes,
    numpy_version,
)


@pytest.mark.usefixtures("suppress_divide_invalid_numpy_warnings")
@pytest.mark.parametrize(
    "func",
    [
        "abs",
        "arccos",
        "arccosh",
        "arcsin",
        "arcsinh",
        "arctan",
        "arctanh",
        "argsort",
        "conjugate",
        "copy",
        "cos",
        "cosh",
        "conjugate",
        "ediff1d",
        "exp",
        "exp2",
        "expm1",
        "imag",
        "log",
        "log10",
        "log1p",
        "log2",
        "max",
        "min",
        "mean",
        "median",
        "negative",
        "positive",
        "real",
        "sign",
        "sin",
        "sinh",
        "sort",
        "sqrt",
        "square",
        "std",
        "tan",
        "tanh",
        "var",
    ],
)
@pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True, no_bool=True))
@pytest.mark.parametrize("stride", [2, -1, -3])
def test_1arg_support_complex(func, dtype, stride):
    x = generate_random_numpy_array(10, dtype=dtype)
    a, ia = x[::stride], dpnp.array(x)[::stride]

    if numpy_version() < "2.0.0" and func == "sign":
        pytest.skip("numpy.sign definition is different for complex numbers.")
    # dpnp default is stable
    kwargs = {"kind": "stable"} if func == "argsort" else {}
    result = getattr(dpnp, func)(ia)
    expected = getattr(numpy, func)(a, **kwargs)
    assert_dtype_allclose(result, expected, factor=24)


@pytest.mark.parametrize(
    "func",
    [
        "cbrt",
        "ceil",
        "degrees",
        "fabs",
        "floor",
        "radians",
        "trunc",
        "unwrap",
    ],
)
@pytest.mark.parametrize("dtype", get_integer_float_dtypes())
@pytest.mark.parametrize("stride", [2, -1, -3])
def test_1arg(func, dtype, stride):
    x = generate_random_numpy_array(10, dtype=dtype)
    a, ia = x[::stride], dpnp.array(x)[::stride]

    result = getattr(dpnp, func)(ia)
    expected = getattr(numpy, func)(a)

    # numpy.ceil, numpy.floor, numpy.trunc always return float dtype for
    # NumPy < 2.1.0 while for NumPy >= 2.1.0, output has the dtype of input
    # (dpnp follows this behavior)
    if numpy_version() < "2.1.0":
        check_type = False
    else:
        check_type = True
    assert_dtype_allclose(result, expected, check_type=check_type)


@pytest.mark.usefixtures("suppress_divide_invalid_numpy_warnings")
@pytest.mark.parametrize("dtype", get_integer_float_dtypes())
@pytest.mark.parametrize("stride", [2, -1, -3])
def test_rsqrt(dtype, stride):
    x = generate_random_numpy_array(10, dtype=dtype)
    a, ia = x[::stride], dpnp.array(x)[::stride]

    result = dpnp.rsqrt(ia)
    expected = 1 / numpy.sqrt(a)
    assert_dtype_allclose(result, expected)


@pytest.mark.parametrize("dtype", get_integer_float_dtypes())
@pytest.mark.parametrize("stride", [2, -1, -3])
def test_logsumexp(dtype, stride):
    x = generate_random_numpy_array(10, dtype=dtype)
    a, ia = x[::stride], dpnp.array(x)[::stride]

    result = dpnp.logsumexp(ia)
    expected = numpy.logaddexp.reduce(a)
    # for int8, uint8, NumPy returns float16 but dpnp returns float64
    # for int16, uint16, NumPy returns float32 but dpnp returns float64
    flag = dtype in [dpnp.int8, dpnp.uint8, dpnp.int16, dpnp.uint16]
    assert_dtype_allclose(result, expected, check_only_type_kind=flag)


@pytest.mark.parametrize("dtype", get_integer_float_dtypes())
@pytest.mark.parametrize("stride", [2, -1, -3])
def test_cumlogsumexp(dtype, stride):
    x = generate_random_numpy_array(10, dtype=dtype)
    a, ia = x[::stride], dpnp.array(x)[::stride]

    result = dpnp.cumlogsumexp(ia)
    expected = numpy.logaddexp.accumulate(a)
    # for int8, uint8, NumPy returns float16 but dpnp returns float64
    # for int16, uint16, NumPy returns float32 but dpnp returns float64
    flag = dtype in [dpnp.int8, dpnp.uint8, dpnp.int16, dpnp.uint16]
    assert_dtype_allclose(result, expected, check_only_type_kind=flag)


@pytest.mark.parametrize("dtype", get_integer_float_dtypes())
@pytest.mark.parametrize("stride", [2, -1, -3])
def test_reduce_hypot(dtype, stride):
    x = generate_random_numpy_array(10, dtype=dtype)
    a, ia = x[::stride], dpnp.array(x)[::stride]

    result = dpnp.reduce_hypot(ia)
    expected = numpy.hypot.reduce(a)
    # for int8, uint8, NumPy returns float16 but dpnp returns float64
    # for int16, uint16, NumPy returns float32 but dpnp returns float64
    flag = dtype in [dpnp.int8, dpnp.uint8, dpnp.int16, dpnp.uint16]
    assert_dtype_allclose(result, expected, check_only_type_kind=flag)


@pytest.mark.parametrize(
    "dtype",
    get_integer_float_dtypes(
        no_unsigned=True, xfail_dtypes=[dpnp.int8, dpnp.int16]
    ),
)
def test_erf(dtype):
    a = dpnp.linspace(-1, 1, num=10, dtype=dtype)
    b = a[::2]
    result = dpnp.erf(b)

    expected = numpy.empty_like(b.asnumpy())
    for idx, val in enumerate(b):
        expected[idx] = math.erf(val)

    assert_dtype_allclose(result, expected)


@pytest.mark.parametrize("dtype", get_float_complex_dtypes())
@pytest.mark.parametrize("stride", [2, -1, -3])
def test_reciprocal(dtype, stride):
    x = generate_random_numpy_array(10, dtype=dtype)
    a, ia = x[::stride], dpnp.array(x)[::stride]

    result = dpnp.reciprocal(ia)
    expected = numpy.reciprocal(a)
    assert_dtype_allclose(result, expected)


@pytest.mark.parametrize("dtype", get_complex_dtypes())
@pytest.mark.parametrize("stride", [2, -1, -3])
def test_angle(dtype, stride):
    x = generate_random_numpy_array(10, dtype=dtype)
    a, ia = x[::stride], dpnp.array(x)[::stride]

    result = dpnp.angle(ia)
    expected = numpy.angle(a)
    assert_dtype_allclose(result, expected)


@pytest.mark.parametrize(
    "func",
    [
        "add",
        "arctan2",
        "divide",
        "fmax",
        "fmin",
        "hypot",
        "logaddexp",
        "logaddexp2",
        "maximum",
        "minimum",
        "multiply",
        "power",
        "subtract",
    ],
)
@pytest.mark.parametrize("dtype", get_integer_float_dtypes())
@pytest.mark.usefixtures("suppress_divide_invalid_numpy_warnings")
def test_2args(func, dtype):
    # Integers to negative integer powers are not allowed
    dt_list = [dpnp.int8, dpnp.int16, dpnp.int32, dpnp.int64]
    low = 0 if func == "power" and dtype in dt_list else -10
    a = generate_random_numpy_array((3, 3), dtype=dtype, low=low)
    ia = dpnp.array(a)
    b, ib = a.T, ia.T

    result = getattr(dpnp, func)(ia, ib)
    expected = getattr(numpy, func)(a, b)
    assert_dtype_allclose(result, expected)


@pytest.mark.parametrize(
    "func",
    ["bitwise_and", "bitwise_or", "bitwise_xor", "left_shift", "right_shift"],
)
@pytest.mark.parametrize("dtype", get_integer_dtypes())
def test_bitwise(func, dtype):
    # negative values for shift is not supported
    low = 0 if func in ["left_shift", "right_shift"] else -10
    a = generate_random_numpy_array((3, 3), dtype=dtype, low=low)
    ia = dpnp.array(a)
    b, ib = a.T, ia.T

    result = getattr(dpnp, func)(ia, ib)
    expected = getattr(numpy, func)(a, b)
    assert_array_equal(result, expected)


@pytest.mark.parametrize("dtype", get_integer_float_dtypes())
def test_copysign(dtype):
    a = generate_random_numpy_array((3, 3), dtype=dtype)
    ia = dpnp.array(a)
    b, ib = -a.T, dpnp.negative(ia.T)

    result = dpnp.copysign(ia, ib)
    expected = numpy.copysign(a, b)
    assert_dtype_allclose(result, expected)


@pytest.mark.parametrize("func", ["fmod", "true_divide", "remainder"])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True, no_complex=True))
def test_division(func, dtype):
    a = generate_random_numpy_array((3, 3), dtype=dtype)
    ia = dpnp.array(a)
    b, ib = a.T + 1, ia.T + 1

    result = getattr(dpnp, func)(ia, ib)
    expected = getattr(numpy, func)(a, b)
    assert_dtype_allclose(result, expected)


@pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
@pytest.mark.parametrize("func", ["add", "multiply", "power", "subtract"])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True, no_bool=True))
def test_2args_out(func, dtype):
    shape = (5, 3, 2)
    out = numpy.empty(shape, dtype=dtype)[::3]
    iout = dpnp.empty(shape, dtype=dtype)[::3]

    a = generate_random_numpy_array(out.shape, dtype=dtype)
    b = numpy.full(out.shape, fill_value=0.7, dtype=dtype)
    ia, ib = dpnp.array(a), dpnp.array(b)

    expected = getattr(numpy, func)(a, b, out=out)
    result = getattr(dpnp, func)(ia, ib, out=iout)

    assert result is iout
    assert_dtype_allclose(result, expected)


@pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
@pytest.mark.parametrize("func", ["add", "multiply", "power", "subtract"])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True, no_bool=True))
def test_2args_in_out(func, dtype):
    sh = (3, 4, 2)
    out = numpy.empty(sh, dtype=dtype)[::2]
    iout = dpnp.empty(sh, dtype=dtype)[::2]

    a = generate_random_numpy_array(sh, dtype=dtype)
    b = numpy.full(sh, fill_value=0.7, dtype=dtype)
    ia, ib = dpnp.array(a), dpnp.array(b)

    a, b = a[::2], b[::2].T
    ia, ib = ia[::2], ib[::2].T

    expected = getattr(numpy, func)(a, b, out=out)
    result = getattr(dpnp, func)(ia, ib, out=iout)
    assert result is iout
    assert_dtype_allclose(result, expected)


@pytest.mark.parametrize("func", ["add", "multiply", "power", "subtract"])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True, no_bool=True))
@pytest.mark.skip("dpctl doesn't support type mismatch of out array")
def test_2args_in_out_diff_out_dtype(func, dtype):
    sh = (3, 3, 2)
    out = numpy.ones(sh, dtype=numpy.complex64)[::2]
    iout = dpnp.ones(sh, dtype=dpnp.complex64)[::2]

    a = generate_random_numpy_array(sh, dtype=dtype)
    b = numpy.full(sh, fill_value=0.7, dtype=dtype)
    ia, ib = dpnp.array(a), dpnp.array(b)

    a, b = a[::2].T, b[::2]
    ia, ib = ia[::2].T, ib[::2]

    expected = getattr(numpy, func)(a, b, out=out)
    result = getattr(dpnp, func)(ia, ib, out=iout)

    assert result is iout
    assert_dtype_allclose(result, expected)


@pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
@pytest.mark.parametrize("func", ["add", "multiply", "power", "subtract"])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True, no_bool=True))
def test_2args_in_overlap(func, dtype):
    size = 5
    # Integers to negative integer powers are not allowed
    dt_list = [dpnp.int8, dpnp.int16, dpnp.int32, dpnp.int64]
    low = 0 if func == "power" and dtype in dt_list else -10
    a = generate_random_numpy_array(2 * size, dtype=dtype, low=low)
    ia = dpnp.array(a)

    expected = getattr(numpy, func)(a[size::], a[::2], out=a[:size:])
    result = getattr(dpnp, func)(ia[size::], ia[::2], out=ia[:size:])

    assert_dtype_allclose(result, expected)
    assert_dtype_allclose(ia, a)


@pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
@pytest.mark.parametrize("func", ["add", "multiply", "power", "subtract"])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True, no_bool=True))
def test_2args_in_out_overlap(func, dtype):
    a = generate_random_numpy_array((4, 3, 2), dtype=dtype)
    b = numpy.full(a[::2].shape, fill_value=0.7, dtype=dtype)
    ia, ib = dpnp.array(a), dpnp.array(b)

    expected = getattr(numpy, func)(a[::2], b, out=a[1::2])
    result = getattr(dpnp, func)(ia[::2], ib, out=ia[1::2])

    assert_dtype_allclose(result, expected)
    assert_dtype_allclose(ia, a)
