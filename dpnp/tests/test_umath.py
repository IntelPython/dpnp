import numpy
import pytest
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_raises,
)

import dpnp

from .helper import (
    assert_dtype_allclose,
    get_abs_array,
    get_all_dtypes,
    get_float_complex_dtypes,
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


# implement missing umaths and to remove the list
new_umaths_numpy_20 = [
    "bitwise_count",  # SAT-7323
]


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.usefixtures("suppress_divide_invalid_numpy_warnings")
@pytest.mark.parametrize("test_cases", test_cases, ids=get_id)
def test_umaths(test_cases):
    umath, args_str = test_cases
    if umath in new_umaths_numpy_20:
        pytest.skip("new umaths from numpy 2.0 are not supported yet")

    if umath in ["matmul", "matvec", "vecmat"]:
        sh = (4, 4)
    elif umath in ["power", "pow"]:
        sh = (2, 3)
    else:
        sh = (3, 4)

    args = get_args(args_str, sh, xp=numpy)
    iargs = get_args(args_str, sh, xp=dpnp)

    if umath == "reciprocal":
        if args[0].dtype in [numpy.int32, numpy.int64]:
            pytest.skip(
                "For integer input array, numpy.reciprocal returns zero."
            )
    elif umath == "ldexp":
        if (
            numpy.lib.NumpyVersion(numpy.__version__) < "2.0.0"
            and args[1].dtype == numpy.int64
            and numpy.dtype("l") != numpy.int64
        ):
            pytest.skip("numpy.ldexp doesn't have a loop for the input types")

    # original
    expected = getattr(numpy, umath)(*args)

    # DPNP
    result = getattr(dpnp, umath)(*iargs)

    assert_allclose(result, expected, rtol=1e-6)


def _get_numpy_arrays_1in_1out(func_name, dtype, range):
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


def _get_numpy_arrays_2in_1out(func_name, dtype, range):
    """
    Return two sample arrays and an output array.

    Create two appropriate arrays specified by `dtype` and `range` which are
    used as inputs for a function specified by `func_name` to obtain the output.
    """
    low = range[0]
    high = range[1]
    size = range[2]
    if dtype == numpy.bool_:
        np_array1 = numpy.arange(2, dtype=dtype)
        np_array2 = numpy.arange(2, dtype=dtype)
        result = getattr(numpy, func_name)(np_array1, np_array2)
    elif dpnp.issubdtype(dtype, dpnp.complexfloating):
        a = numpy.random.uniform(low=low, high=high, size=size)
        b = numpy.random.uniform(low=low, high=high, size=size)
        np_array1 = numpy.array(a + 1j * b, dtype=dtype)
        a = numpy.random.uniform(low=low, high=high, size=size)
        b = numpy.random.uniform(low=low, high=high, size=size)
        np_array2 = numpy.array(a + 1j * b, dtype=dtype)
        result = getattr(numpy, func_name)(np_array1, np_array2)
    else:
        a = numpy.random.uniform(low=low, high=high, size=size)
        np_array1 = numpy.array(a, dtype=dtype)
        a = numpy.random.uniform(low=low, high=high, size=size)
        np_array2 = numpy.array(a, dtype=dtype)
        result = getattr(numpy, func_name)(np_array1, np_array2)

    return np_array1, np_array2, result


def _get_output_data_type(dtype):
    """Return a data type specified by input `dtype` and device capabilities."""
    dtype_float16 = any(
        dpnp.issubdtype(dtype, t) for t in (dpnp.bool, dpnp.int8, dpnp.uint8)
    )
    dtype_float32 = any(
        dpnp.issubdtype(dtype, t) for t in (dpnp.int16, dpnp.uint16)
    )
    if dtype_float16:
        out_dtype = dpnp.float16 if has_support_aspect16() else dpnp.float32
    elif dtype_float32:
        out_dtype = dpnp.float32
    elif dpnp.issubdtype(dtype, dpnp.complexfloating):
        out_dtype = dpnp.complex64
        if has_support_aspect64() and dtype != dpnp.complex64:
            out_dtype = dpnp.complex128
    else:
        out_dtype = dpnp.float32
        if has_support_aspect64() and dtype != dpnp.float32:
            out_dtype = dpnp.float64

    return out_dtype


class TestArctan2:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    def test_arctan2(self, dtype):
        np_array1, np_array2, expected = _get_numpy_arrays_2in_1out(
            "arctan2", dtype, [0, 10, 10]
        )

        dp_array1 = dpnp.array(np_array1)
        dp_array2 = dpnp.array(np_array2)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.arctan2(dp_array1, dp_array2, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.arctan2(dp_array, dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array = dpnp.arange(10)
        dp_out = dpnp.empty(shape)
        with pytest.raises(ValueError):
            dpnp.arctan2(dp_array, dp_array, out=dp_out)

    def test_alias(self):
        x = dpnp.array([-1, +1, +1, -1])
        y = dpnp.array([-1, -1, +1, +1])

        res1 = dpnp.arctan2(y, x)
        res2 = dpnp.atan2(y, x)
        assert_array_equal(res1, res2)


class TestCbrt:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    def test_cbrt(self, dtype):
        np_array, expected = _get_numpy_arrays_1in_1out(
            "cbrt", dtype, [-5, 5, 10]
        )

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

        with pytest.raises(ValueError):
            dpnp.cbrt(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array = dpnp.arange(10)
        dp_out = dpnp.empty(shape)

        with pytest.raises(ValueError):
            dpnp.cbrt(dp_array, out=dp_out)


class TestCopySign:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    def test_copysign(self, dtype):
        np_array1, np_array2, expected = _get_numpy_arrays_2in_1out(
            "copysign", dtype, [0, 10, 10]
        )

        dp_array1 = dpnp.array(np_array1)
        dp_array2 = dpnp.array(np_array2)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.copysign(dp_array1, dp_array2, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)
        with pytest.raises(ValueError):
            dpnp.copysign(dp_array, dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array = dpnp.arange(10)
        dp_out = dpnp.empty(shape)
        with pytest.raises(ValueError):
            dpnp.copysign(dp_array, dp_array, out=dp_out)


class TestDegrees:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_complex=True)
    )
    def test_basic(self, dtype):
        a = get_abs_array([numpy.pi, -0.5 * numpy.pi], dtype)
        ia = dpnp.array(a)

        result = dpnp.degrees(ia)
        expected = numpy.degrees(a)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_nan_infs(self, dtype):
        a = numpy.array([numpy.nan, -numpy.inf, numpy.inf], dtype=dtype)
        ia = dpnp.array(a)

        result = dpnp.degrees(ia)
        expected = numpy.degrees(a)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_large_values(self, dtype):
        a = numpy.arange(0, 10**5, 70, dtype=dtype) * numpy.pi
        ia = dpnp.array(a)

        result = dpnp.degrees(ia)
        expected = numpy.degrees(a)
        assert_dtype_allclose(result, expected)


class TestFloatPower:
    @pytest.mark.parametrize("dt1", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("dt2", get_all_dtypes(no_none=True))
    def test_type_conversion(self, dt1, dt2):
        a = numpy.array([0, 1, 2, 3, 4, 5], dtype=dt1)
        b = numpy.array([1.0, 2.0, 3.0, 3.0, 2.0, 1.0], dtype=dt2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.float_power(ia, ib)
        expected = numpy.float_power(a, b)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
    @pytest.mark.parametrize(
        "dt", get_all_dtypes(no_none=True, no_unsigned=True)
    )
    def test_negative_base_value(self, dt):
        a = numpy.array([-1, -4], dtype=dt)
        ia = dpnp.array(a)

        result = dpnp.float_power(ia, 1.5)
        expected = numpy.float_power(a, 1.5)
        assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "dt", get_all_dtypes(no_none=True, no_unsigned=True)
    )
    def test_negative_base_value_complex_dtype(self, dt):
        a = numpy.array([-1, -4], dtype=dt)
        ia = dpnp.array(a)

        dt = dpnp.complex128 if has_support_aspect64() else dpnp.complex64
        result = dpnp.float_power(ia, 1.5, dtype=dt)

        # numpy.float_power does not have a loop for complex64
        expected = numpy.float_power(a, 1.5, dtype=numpy.complex128)
        assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "exp_val", [2, 0, -3.2, numpy.nan, -numpy.inf, numpy.inf]
    )
    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_nan_infs_base(self, exp_val, dtype):
        a = numpy.array([numpy.nan, -numpy.inf, numpy.inf], dtype=dtype)
        ia = dpnp.array(a)

        result = dpnp.float_power(ia, exp_val)
        expected = numpy.float_power(a, exp_val)
        assert_allclose(result, expected)


class TestLogAddExp:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    def test_logaddexp(self, dtype):
        np_array1, np_array2, expected = _get_numpy_arrays_2in_1out(
            "logaddexp", dtype, [0, 10, 10]
        )

        dp_array1 = dpnp.array(np_array1)
        dp_array2 = dpnp.array(np_array2)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.logaddexp(dp_array1, dp_array2, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)
        with pytest.raises(ValueError):
            dpnp.logaddexp(dp_array, dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array = dpnp.arange(10)
        dp_out = dpnp.empty(shape)
        with pytest.raises(ValueError):
            dpnp.logaddexp(dp_array, dp_array, out=dp_out)

    @pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
    @pytest.mark.parametrize(
        "val1, val2",
        [
            pytest.param(numpy.nan, numpy.inf),
            pytest.param(numpy.inf, numpy.nan),
            pytest.param(numpy.nan, 0),
            pytest.param(0, numpy.nan),
            pytest.param(numpy.nan, numpy.nan),
        ],
    )
    def test_nan(self, val1, val2):
        a = numpy.array(val1)
        b = numpy.array(val2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.logaddexp(ia, ib)
        expected = numpy.logaddexp(a, b)
        assert_equal(result, expected)


class TestLogAddExp2:
    @pytest.mark.parametrize(
        "dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    def test_values(self, dt):
        a = numpy.log2(numpy.array([1, 2, 3, 4, 5], dtype=dt))
        b = numpy.log2(numpy.array([5, 4, 3, 2, 1], dtype=dt))
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.logaddexp2(ia, ib)
        expected = numpy.logaddexp2(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dt",
        [numpy.bool_, numpy.int32, numpy.int64, numpy.float32, numpy.float64],
    )
    def test_range(self, dt):
        a = numpy.array([1000000, -1000000, 1000200, -1000200], dtype=dt)
        b = numpy.array([1000200, -1000200, 1000000, -1000000], dtype=dt)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.logaddexp2(ia, ib)
        expected = numpy.logaddexp2(a, b)
        assert_almost_equal(result, expected)

    @pytest.mark.parametrize("dt", get_float_dtypes())
    def test_inf(self, dt):
        inf = numpy.inf
        a = numpy.array([inf, -inf, inf, -inf, inf, 1, -inf, 1], dtype=dt)
        b = numpy.array([inf, inf, -inf, -inf, 1, inf, 1, -inf], dtype=dt)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.logaddexp2(ia, ib)
        expected = numpy.logaddexp2(a, b)
        assert_equal(result, expected)

    @pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
    @pytest.mark.parametrize(
        "val1, val2",
        [
            pytest.param(numpy.nan, numpy.inf),
            pytest.param(numpy.inf, numpy.nan),
            pytest.param(numpy.nan, 0),
            pytest.param(0, numpy.nan),
            pytest.param(numpy.nan, numpy.nan),
        ],
    )
    def test_nan(self, val1, val2):
        a = numpy.array(val1)
        b = numpy.array(val2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.logaddexp2(ia, ib)
        expected = numpy.logaddexp2(a, b)
        assert_equal(result, expected)


class TestRadians:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_complex=True)
    )
    def test_basic(self, dtype):
        a = get_abs_array([120.0, -90.0], dtype)
        ia = dpnp.array(a)

        result = dpnp.radians(ia)
        expected = numpy.radians(a)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_nan_infs(self, dtype):
        a = numpy.array([numpy.nan, -numpy.inf, numpy.inf], dtype=dtype)
        ia = dpnp.array(a)

        result = dpnp.radians(ia)
        expected = numpy.radians(a)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_large_values(self, dtype):
        a = numpy.arange(0, 10**5, 70, dtype=dtype)
        ia = dpnp.array(a)

        result = dpnp.radians(ia)
        expected = numpy.radians(a)
        assert_dtype_allclose(result, expected)


class TestReciprocal:
    @pytest.mark.usefixtures("suppress_divide_numpy_warnings")
    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_reciprocal(self, dtype):
        np_array, expected = _get_numpy_arrays_1in_1out(
            "reciprocal", dtype, [-5, 5, 10]
        )

        dp_array = dpnp.array(np_array)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.reciprocal(dp_array, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes()[:-1])
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_float_complex_dtypes()[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.reciprocal(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array = dpnp.arange(10)
        dp_out = dpnp.empty(shape)

        with pytest.raises(ValueError):
            dpnp.reciprocal(dp_array, out=dp_out)


class TestRsqrt:
    @pytest.mark.usefixtures("suppress_divide_numpy_warnings")
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    def test_rsqrt(self, dtype):
        np_array, expected = _get_numpy_arrays_1in_1out(
            "sqrt", dtype, [0, 10, 10]
        )
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

        with pytest.raises(ValueError):
            dpnp.rsqrt(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array = dpnp.arange(10)
        dp_out = dpnp.empty(shape)
        with pytest.raises(ValueError):
            dpnp.rsqrt(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["4", "()", "[]", "(3, 7)", "[2, 4]"],
    )
    def test_invalid_out(self, out):
        a = dpnp.arange(10)
        assert_raises(TypeError, dpnp.rsqrt, a, out)


class TestSquare:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_square(self, dtype):
        np_array, expected = _get_numpy_arrays_1in_1out(
            "square", dtype, [-5, 5, 10]
        )

        dp_array = dpnp.array(np_array)
        out_dtype = numpy.int8 if dtype == numpy.bool_ else dtype
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.square(dp_array, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True)[:-1])
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.square(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array = dpnp.arange(10)
        dp_out = dpnp.empty(shape)
        with pytest.raises(ValueError):
            dpnp.square(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["4", "()", "[]", "(3, 7)", "[2, 4]"],
    )
    def test_invalid_out(self, out):
        a = dpnp.arange(10)

        assert_raises(TypeError, dpnp.square, a, out)
        assert_raises(TypeError, numpy.square, a.asnumpy(), out)


class TestUmath:
    @pytest.fixture(
        params=[
            {"func_name": "arccos", "input_values": [-1, 1, 10]},
            {"func_name": "arccosh", "input_values": [1, 10, 10]},
            {"func_name": "arcsin", "input_values": [-1, 1, 10]},
            {"func_name": "arcsinh", "input_values": [-5, 5, 10]},
            {"func_name": "arctan", "input_values": [-5, 5, 10]},
            {"func_name": "arctanh", "input_values": [-1, 1, 10]},
            {"func_name": "cos", "input_values": [-5, 5, 10]},
            {"func_name": "cosh", "input_values": [-5, 5, 10]},
            {"func_name": "exp", "input_values": [-3, 8, 10]},
            {"func_name": "exp2", "input_values": [-5, 5, 10]},
            {"func_name": "expm1", "input_values": [-5, 5, 10]},
            {"func_name": "log", "input_values": [0, 10, 10]},
            {"func_name": "log10", "input_values": [0, 10, 10]},
            {"func_name": "log2", "input_values": [0, 10, 10]},
            {"func_name": "log1p", "input_values": [0, 10, 10]},
            {"func_name": "sin", "input_values": [-5, 5, 10]},
            {"func_name": "sinh", "input_values": [-5, 5, 10]},
            {"func_name": "sqrt", "input_values": [0, 10, 10]},
            {"func_name": "tan", "input_values": [-1.5, 1.5, 10]},
            {"func_name": "tanh", "input_values": [-5, 5, 10]},
        ],
        ids=[
            "arccos",
            "arccosh",
            "arcsin",
            "arcsinh",
            "arctan",
            "arctanh",
            "cos",
            "cosh",
            "exp",
            "exp2",
            "expm1",
            "log",
            "log10",
            "log2",
            "log1p",
            "sin",
            "sinh",
            "sqrt",
            "tan",
            "tanh",
        ],
    )
    def func_params(self, request):
        return request.param

    @pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
    @pytest.mark.usefixtures("suppress_divide_invalid_numpy_warnings")
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_out(self, func_params, dtype):
        func_name = func_params["func_name"]
        input_values = func_params["input_values"]
        np_array, expected = _get_numpy_arrays_1in_1out(
            func_name, dtype, input_values
        )

        dp_array = dpnp.array(np_array)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = getattr(dpnp, func_name)(dp_array, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True)[:-1])
    def test_invalid_dtype(self, func_params, dtype):
        func_name = func_params["func_name"]
        dpnp_dtype = get_all_dtypes(no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(ValueError):
            getattr(dpnp, func_name)(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, func_params, shape):
        func_name = func_params["func_name"]
        dp_array = dpnp.arange(10)
        dp_out = dpnp.empty(shape)
        with pytest.raises(ValueError):
            getattr(dpnp, func_name)(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["4", "()", "[]", "(3, 7)", "[2, 4]"],
    )
    def test_invalid_out(self, func_params, out):
        func_name = func_params["func_name"]
        a = dpnp.arange(10)
        assert_raises(TypeError, getattr(dpnp, func_name), a, out)
        assert_raises(TypeError, getattr(numpy, func_name), a.asnumpy(), out)


def test_trigonometric_hyperbolic_aliases():
    a = dpnp.array([-0.5, 0, 0.5])

    assert_array_equal(dpnp.arcsin(a), dpnp.asin(a))
    assert_array_equal(dpnp.arccos(a), dpnp.acos(a))
    assert_array_equal(dpnp.arctan(a), dpnp.atan(a))
    assert_array_equal(dpnp.arctanh(a), dpnp.atanh(a))
    assert_array_equal(dpnp.arcsinh(a), dpnp.asinh(a))

    a = dpnp.array([1, 1.5, 2])
    assert_array_equal(dpnp.arccosh(a), dpnp.acosh(a))
