import numpy
import pytest
from numpy.testing import (
    assert_allclose,
)

import dpnp

from .helper import (
    assert_dtype_allclose,
    get_all_dtypes,
    get_float_dtypes,
    has_support_aspect16,
    has_support_aspect64,
)

# full list of umaths
umaths = [i for i in dir(numpy) if isinstance(getattr(numpy, i), numpy.ufunc)]

types = {
    "d": numpy.float64,
    "f": numpy.float32,
    "l": numpy.int64,
    "i": numpy.int32,
}

supported_types = "fli"
if has_support_aspect64():
    supported_types += "d"


def check_types(args_str):
    for s in args_str:
        if s not in supported_types:
            return False
    return True


def shaped_arange(shape, xp=numpy, dtype=numpy.float32):
    size = 1
    for i in shape:
        size = size * i
    array_data = numpy.arange(1, size + 1, 1).tolist()
    return xp.reshape(xp.array(array_data, dtype=dtype), shape)


def get_args(args_str, sh, xp=numpy):
    args = []
    for s in args_str:
        args.append(shaped_arange(shape=sh, xp=xp, dtype=types[s]))
    return tuple(args)


test_cases = []
for umath in umaths:
    np_umath = getattr(numpy, umath)

    for type_ in np_umath.types:
        args_str = type_[: type_.find("->")]
        if check_types(args_str):
            val_ = (umath, args_str)
            if val_ not in test_cases:
                test_cases.append(val_)


def get_id(val):
    return val.__str__()


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.usefixtures("suppress_divide_invalid_numpy_warnings")
@pytest.mark.parametrize("test_cases", test_cases, ids=get_id)
def test_umaths(test_cases):
    umath, args_str = test_cases
    if umath == "matmul":
        sh = (4, 4)
    elif umath == "power":
        sh = (2, 3)
    else:
        sh = (3, 4)

    args = get_args(args_str, sh, xp=numpy)
    iargs = get_args(args_str, sh, xp=dpnp)

    # original
    expected = getattr(numpy, umath)(*args)

    # DPNP
    result = getattr(dpnp, umath)(*iargs)

    assert_allclose(result, expected, rtol=1e-6)


def _get_numpy_arrays(func_name, dtype, range):
    """
    Return a sample array and an output array.

    Create an appropriate array specified by `dtype` and `range` which is used as
    an input for a function specified by `func_name` to obtain the output.
    """
    low = range[0]
    high = range[1]
    size = range[2]
    if dtype == numpy.bool_:
        np_array = numpy.arange(2, dtype=dtype)
        result = getattr(numpy, func_name)(np_array)
    elif dpnp.issubdtype(dtype, dpnp.complexfloating):
        a = numpy.random.uniform(low=low, high=high, size=size)
        b = numpy.random.uniform(low=low, high=high, size=size)
        np_array = numpy.array(a + 1j * b, dtype=dtype)
        result = getattr(numpy, func_name)(np_array)
    else:
        a = numpy.random.uniform(low=low, high=high, size=size)
        np_array = numpy.array(a, dtype=dtype)
        result = getattr(numpy, func_name)(np_array)

    return np_array, result


def _get_output_data_type(dtype):
    """Return a data type specified by input `dtype` and device capabilities."""
    if dpnp.issubdtype(dtype, dpnp.bool):
        out_dtype = dpnp.float16 if has_support_aspect16() else dpnp.float32
    elif dpnp.issubdtype(dtype, dpnp.complexfloating):
        out_dtype = dpnp.complex64
        if has_support_aspect64() and dtype != dpnp.complex64:
            out_dtype = dpnp.complex128
    else:
        out_dtype = dpnp.float32
        if has_support_aspect64() and dtype != dpnp.float32:
            out_dtype = dpnp.float64

    return out_dtype


class TestSin:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_sin(self, dtype):
        np_array, expected = _get_numpy_arrays("sin", dtype, [-5, 5, 10])

        dp_array = dpnp.array(np_array)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.sin(dp_array, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype",
        [numpy.float32, numpy.int64, numpy.int32],
        ids=["numpy.float32", "numpy.int64", "numpy.int32"],
    )
    def test_invalid_dtype(self, dtype):
        dp_array = dpnp.arange(10, dtype=dpnp.complex64)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.sin(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array = dpnp.arange(10)
        dp_out = dpnp.empty(shape, dtype=dp_array.dtype)

        with pytest.raises(ValueError):
            dpnp.sin(dp_array, out=dp_out)


class TestSinh:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_sinh(self, dtype):
        np_array, expected = _get_numpy_arrays("sinh", dtype, [-5, 5, 10])

        dp_array = dpnp.array(np_array)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.sinh(dp_array, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype",
        [numpy.float32, numpy.int64, numpy.int32],
        ids=["numpy.float32", "numpy.int64", "numpy.int32"],
    )
    def test_invalid_dtype(self, dtype):
        dp_array = dpnp.arange(10, dtype=dpnp.complex64)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.sinh(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array = dpnp.arange(10)
        dp_out = dpnp.empty(shape, dtype=dp_array.dtype)

        with pytest.raises(ValueError):
            dpnp.sinh(dp_array, out=dp_out)


class TestCos:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_cos(self, dtype):
        np_array, expected = _get_numpy_arrays("cos", dtype, [-5, 5, 10])

        dp_array = dpnp.array(np_array)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.cos(dp_array, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype",
        [numpy.float32, numpy.int64, numpy.int32],
        ids=["numpy.float32", "numpy.int64", "numpy.int32"],
    )
    def test_invalid_dtype(self, dtype):
        dp_array = dpnp.arange(10, dtype=dpnp.complex64)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.cos(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array = dpnp.arange(10)
        dp_out = dpnp.empty(shape, dtype=dp_array.dtype)

        with pytest.raises(ValueError):
            dpnp.cos(dp_array, out=dp_out)


class TestCosh:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_cosh(self, dtype):
        np_array, expected = _get_numpy_arrays("cosh", dtype, [-5, 5, 10])

        dp_array = dpnp.array(np_array)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.cosh(dp_array, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype",
        [numpy.float32, numpy.int64, numpy.int32],
        ids=["numpy.float32", "numpy.int64", "numpy.int32"],
    )
    def test_invalid_dtype(self, dtype):
        dp_array = dpnp.arange(10, dtype=dpnp.complex64)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.cosh(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array = dpnp.arange(10)
        dp_out = dpnp.empty(shape, dtype=dp_array.dtype)

        with pytest.raises(ValueError):
            dpnp.cosh(dp_array, out=dp_out)


class TestLog:
    @pytest.mark.usefixtures("suppress_divide_numpy_warnings")
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_log(self, dtype):
        np_array, expected = _get_numpy_arrays("log", dtype, [0.1, 10, 10])

        dp_array = dpnp.array(np_array)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.log(dp_array, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype",
        [numpy.float32, numpy.int64, numpy.int32],
        ids=["numpy.float32", "numpy.int64", "numpy.int32"],
    )
    def test_invalid_dtype(self, dtype):
        dp_array = dpnp.arange(10, dtype=dpnp.complex64)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.log(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array = dpnp.arange(10)
        dp_out = dpnp.empty(shape, dtype=dp_array.dtype)

        with pytest.raises(ValueError):
            dpnp.log(dp_array, out=dp_out)


class TestExp:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_exp(self, dtype):
        np_array, expected = _get_numpy_arrays("exp", dtype, [-3, 8, 10])

        dp_array = dpnp.array(np_array)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.exp(dp_array, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.exp(dp_array, out=dp_out)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape, dtype):
        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(shape, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.exp(dp_array, out=dp_out)


class TestExp2:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_exp2(self, dtype):
        np_array, expected = _get_numpy_arrays("exp2", dtype, [-3, 8, 10])

        dp_array = dpnp.array(np_array)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.exp2(dp_array, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.exp2(dp_array, out=dp_out)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape, dtype):
        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(shape, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.exp2(dp_array, out=dp_out)


class TestCbrt:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    def test_cbrt(self, dtype):
        np_array, expected = _get_numpy_arrays("cbrt", dtype, [-5, 5, 10])

        dp_array = dpnp.array(np_array)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.cbrt(dp_array, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.cbrt(dp_array, out=dp_out)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape, dtype):
        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(shape, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.cbrt(dp_array, out=dp_out)


class TestRsqrt:
    @pytest.mark.usefixtures("suppress_divide_numpy_warnings")
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    def test_rsqrt(self, dtype):
        np_array, expected = _get_numpy_arrays("sqrt", dtype, [0, 10, 10])
        expected = numpy.reciprocal(expected)

        dp_array = dpnp.array(np_array)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.rsqrt(dp_array, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.rsqrt(dp_array, out=dp_out)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape, dtype):
        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(shape, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.rsqrt(dp_array, out=dp_out)


class TestArccos:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_arccos(self, dtype):
        np_array, expected = _get_numpy_arrays("arccos", dtype, [-1, 1, 10])

        dp_array = dpnp.array(np_array)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.arccos(dp_array, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.arccos(dp_array, out=dp_out)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape, dtype):
        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(shape, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.arccos(dp_array, out=dp_out)


class TestArccosh:
    @pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_arccosh(self, dtype):
        np_array, expected = _get_numpy_arrays("arccosh", dtype, [1, 10, 10])

        dp_array = dpnp.array(np_array)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.arccosh(dp_array, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.arccosh(dp_array, out=dp_out)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape, dtype):
        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(shape, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.arccosh(dp_array, out=dp_out)


class TestArcsin:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_arcsin(self, dtype):
        np_array, expected = _get_numpy_arrays("arcsin", dtype, [-1, 1, 10])

        dp_array = dpnp.array(np_array)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.arcsin(dp_array, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.arcsin(dp_array, out=dp_out)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape, dtype):
        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(shape, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.arcsin(dp_array, out=dp_out)


class TestArcsinh:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_arcsinh(self, dtype):
        np_array, expected = _get_numpy_arrays("arcsinh", dtype, [-5, 5, 10])

        dp_array = dpnp.array(np_array)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.arcsinh(dp_array, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.arcsinh(dp_array, out=dp_out)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape, dtype):
        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(shape, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.arcsinh(dp_array, out=dp_out)


class TestArctan:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_arctan(self, dtype):
        np_array, expected = _get_numpy_arrays("arctan", dtype, [-5, 5, 10])

        dp_array = dpnp.array(np_array)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.arctan(dp_array, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.arctan(dp_array, out=dp_out)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape, dtype):
        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(shape, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.arctan(dp_array, out=dp_out)


class TestArctanh:
    @pytest.mark.usefixtures("suppress_divide_numpy_warnings")
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_arctanh(self, dtype):
        np_array, expected = _get_numpy_arrays("arctanh", dtype, [-1, 1, 10])

        dp_array = dpnp.array(np_array)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.arctanh(dp_array, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.arctanh(dp_array, out=dp_out)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape, dtype):
        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(shape, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.arctanh(dp_array, out=dp_out)


class TestTan:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_tan(self, dtype):
        np_array, expected = _get_numpy_arrays("tan", dtype, [-5, 5, 10])

        dp_array = dpnp.array(np_array)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.tan(dp_array, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.tan(dp_array, out=dp_out)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape, dtype):
        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(shape, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.tan(dp_array, out=dp_out)


class TestArctan2:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    def test_arctan2(self, dtype):
        np_array1, _ = _get_numpy_arrays("array", dtype, [-5, 5, 10])
        np_array2, _ = _get_numpy_arrays("array", dtype, [-5, 5, 10])
        expected = numpy.arctan2(np_array1, np_array2)

        dp_array1 = dpnp.array(np_array1)
        dp_array2 = dpnp.array(np_array2)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.arctan2(dp_array1, dp_array2, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape, dtype):
        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(shape, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.arctan2(dp_array, dp_array, out=dp_out)


class TestCopySign:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    def test_copysign(self, dtype):
        np_array1, _ = _get_numpy_arrays("array", dtype, [1, 10, 10])
        np_array2, _ = _get_numpy_arrays("array", dtype, [-10, -1, 10])
        expected = numpy.copysign(np_array1, np_array2)

        dp_array1 = dpnp.array(np_array1)
        dp_array2 = dpnp.array(np_array2)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.copysign(dp_array1, dp_array2, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape, dtype):
        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(shape, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.copysign(dp_array, dp_array, out=dp_out)


class TestSqrt:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_sqrt(self, dtype):
        np_array, expected = _get_numpy_arrays("sqrt", dtype, [0, 10, 10])

        dp_array = dpnp.array(np_array)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.sqrt(dp_array, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype", [numpy.int64, numpy.int32], ids=["numpy.int64", "numpy.int32"]
    )
    def test_invalid_dtype(self, dtype):
        dp_array = dpnp.arange(10, dtype=dpnp.float32)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.sqrt(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array = dpnp.arange(10, dtype=dpnp.float32)
        dp_out = dpnp.empty(shape, dtype=dpnp.float32)

        with pytest.raises(ValueError):
            dpnp.sqrt(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["4", "()", "[]", "(3, 7)", "[2, 4]"],
    )
    def test_invalid_out(self, out):
        a = dpnp.arange(10)

        numpy.testing.assert_raises(TypeError, dpnp.sqrt, a, out)
        numpy.testing.assert_raises(TypeError, numpy.sqrt, a.asnumpy(), out)


class TestSquare:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_square(self, dtype):
        np_array, expected = _get_numpy_arrays("square", dtype, [-5, 5, 10])

        dp_array = dpnp.array(np_array)
        out_dtype = numpy.int8 if dtype == numpy.bool_ else dtype
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.square(dp_array, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    def test_invalid_dtype(self, dtype):
        dp_array = dpnp.ones(10, dtype=dpnp.bool)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.square(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array = dpnp.arange(10, dtype=dpnp.float32)
        dp_out = dpnp.empty(shape, dtype=dpnp.float32)

        with pytest.raises(ValueError):
            dpnp.square(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["4", "()", "[]", "(3, 7)", "[2, 4]"],
    )
    def test_invalid_out(self, out):
        a = dpnp.arange(10)

        numpy.testing.assert_raises(TypeError, dpnp.square, a, out)
        numpy.testing.assert_raises(TypeError, numpy.square, a.asnumpy(), out)
