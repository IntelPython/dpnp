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
import dpnp.backend.extensions.vm._vm_impl as vmi
from dpnp.dpnp_utils import map_dtype_to_device

from .helper import (
    assert_dtype_allclose,
    generate_random_numpy_array,
    get_abs_array,
    get_all_dtypes,
    get_float_complex_dtypes,
    get_float_dtypes,
    has_support_aspect16,
    has_support_aspect64,
    is_cuda_device,
    is_gpu_device,
)

# full list of umaths
umaths = [i for i in dir(numpy) if isinstance(getattr(numpy, i), numpy.ufunc)]

supported_types = "?bBhHiIlLkK"
if has_support_aspect16():
    supported_types += "e"
supported_types += "fF"
if has_support_aspect64():
    supported_types += "dD"


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
        args.append(shaped_arange(shape=sh, xp=xp, dtype=numpy.dtype(s)))
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


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
@pytest.mark.usefixtures("suppress_divide_invalid_numpy_warnings")
@pytest.mark.parametrize("test_cases", test_cases, ids=get_id)
def test_umaths(test_cases):
    umath, args_str = test_cases

    if umath in ["matmul", "matvec", "vecmat"]:
        sh = (4, 4)
    elif umath in ["power", "pow"]:
        sh = (2, 3)
    else:
        sh = (3, 4)

    args = get_args(args_str, sh, xp=numpy)
    iargs = get_args(args_str, sh, xp=dpnp)

    if umath == "reciprocal":
        if numpy.issubdtype(args[0].dtype, numpy.integer):
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
    elif (
        umath == "floor_divide"
        and args[0].dtype in [dpnp.float16, dpnp.float32]
        and is_gpu_device()
    ):
        pytest.skip("dpctl-1652")
    elif (
        umath == "tan"
        and dpnp.issubdtype(args[0].dtype, dpnp.complexfloating)
        and not (vmi._is_available() and has_support_aspect64())
    ):
        pytest.skip("dpctl-2031")
    elif umath in ["divmod", "frexp"]:
        pytest.skip("Not implemented umath")
    elif umath == "modf":
        if args[0].dtype == dpnp.float16:
            pytest.skip("dpnp.modf is not supported with dpnp.float16")
        elif is_cuda_device():
            pytest.skip("dpnp.modf is not supported on CUDA device")

    expected = getattr(numpy, umath)(*args)
    result = getattr(dpnp, umath)(*iargs)
    for x, y in zip(result, expected):
        assert_dtype_allclose(x, y)


class TestArctan2:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_complex=True)
    )
    def test_arctan2(self, dtype):
        a = generate_random_numpy_array(10, dtype, low=0)
        b = generate_random_numpy_array(10, dtype, low=0)
        expected = numpy.arctan2(a, b)

        ia, ib = dpnp.array(a), dpnp.array(b)
        dt_out = map_dtype_to_device(expected.dtype, ia.sycl_device)
        iout = dpnp.empty(expected.shape, dtype=dt_out)

        result = dpnp.arctan2(ia, ib, out=iout)
        assert result is iout
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dt_out", get_all_dtypes(no_none=True, no_complex=True)[:-1]
    )
    def test_invalid_dtype(self, dt_out):
        dt_in = get_all_dtypes(no_none=True, no_complex=True)[-1]
        a = dpnp.arange(10, dtype=dt_in)
        iout = dpnp.empty(10, dtype=dt_out)

        with pytest.raises(ValueError):
            dpnp.arctan2(a, a, out=iout)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15,)", "(2, 2)"]
    )
    def test_invalid_shape(self, shape):
        a = dpnp.arange(10)
        iout = dpnp.empty(shape)
        with pytest.raises(ValueError):
            dpnp.arctan2(a, a, out=iout)

    def test_alias(self):
        x = dpnp.array([-1, +1, +1, -1])
        y = dpnp.array([-1, -1, +1, +1])

        res1 = dpnp.arctan2(y, x)
        res2 = dpnp.atan2(y, x)
        assert_array_equal(res1, res2)


class TestCopySign:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_complex=True)
    )
    def test_copysign(self, dtype):
        a = generate_random_numpy_array(10, dtype, low=0)
        b = generate_random_numpy_array(10, dtype, low=0)
        expected = numpy.copysign(a, b)

        ia, ib = dpnp.array(a), dpnp.array(b)
        dt_out = map_dtype_to_device(expected.dtype, ia.sycl_device)
        iout = dpnp.empty(expected.shape, dtype=dt_out)
        result = dpnp.copysign(ia, ib, out=iout)

        assert result is iout
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dt_out", get_all_dtypes(no_none=True, no_complex=True)[:-1]
    )
    def test_invalid_dtype(self, dt_out):
        dt_in = get_all_dtypes(no_none=True, no_complex=True)[-1]
        a = dpnp.arange(10, dtype=dt_in)
        iout = dpnp.empty(10, dtype=dt_out)
        with pytest.raises(ValueError):
            dpnp.copysign(a, a, out=iout)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15,)", "(2, 2)"]
    )
    def test_invalid_shape(self, shape):
        a = dpnp.arange(10)
        iout = dpnp.empty(shape)
        with pytest.raises(ValueError):
            dpnp.copysign(a, a, out=iout)


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
        # numpy and dpnp promote the result differently
        assert_allclose(result, expected, strict=False)

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
        # numpy and dpnp promote the result differently
        assert_allclose(result, expected, strict=False)


class TestLogAddExp:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_complex=True)
    )
    def test_logaddexp(self, dtype):
        a = generate_random_numpy_array(10, dtype, low=0)
        b = generate_random_numpy_array(10, dtype, low=0)
        expected = numpy.logaddexp(a, b)

        ia, ib = dpnp.array(a), dpnp.array(b)
        dt_out = map_dtype_to_device(expected.dtype, ia.sycl_device)
        iout = dpnp.empty(expected.shape, dtype=dt_out)
        result = dpnp.logaddexp(ia, ib, out=iout)

        assert result is iout
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dt_out", get_all_dtypes(no_none=True, no_complex=True)[:-1]
    )
    def test_invalid_dtype(self, dt_out):
        dt_in = get_all_dtypes(no_none=True, no_complex=True)[-1]
        a = dpnp.arange(10, dtype=dt_in)
        iout = dpnp.empty(10, dtype=dt_out)
        with pytest.raises(ValueError):
            dpnp.logaddexp(a, a, out=iout)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15,)", "(2, 2)"]
    )
    def test_invalid_shape(self, shape):
        a = dpnp.arange(10)
        iout = dpnp.empty(shape)
        with pytest.raises(ValueError):
            dpnp.logaddexp(a, a, out=iout)

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
        "dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    def test_range(self, dt):
        a = get_abs_array([1000000, -1000000, 1000200, -1000200], dtype=dt)
        b = get_abs_array([1000200, -1000200, 1000000, -1000000], dtype=dt)
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
        a = generate_random_numpy_array(10, dtype)
        expected = numpy.reciprocal(a)

        ia = dpnp.array(a)
        dt_out = map_dtype_to_device(expected.dtype, ia.sycl_device)
        iout = dpnp.empty(expected.shape, dtype=dt_out)
        result = dpnp.reciprocal(ia, out=iout)

        assert result is iout
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dt_out", get_float_complex_dtypes()[:-1])
    def test_invalid_dtype(self, dt_out):
        dt_in = get_float_complex_dtypes()[-1]
        a = dpnp.arange(10, dtype=dt_in)
        iout = dpnp.empty(10, dtype=dt_out)
        assert_raises(ValueError, dpnp.reciprocal, a, out=iout)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15,)", "(2, 2)"]
    )
    def test_invalid_shape(self, shape):
        a = dpnp.arange(10)
        iout = dpnp.empty(shape)
        assert_raises(ValueError, dpnp.reciprocal, a, out=iout)


class TestRsqrtCbrt:
    @pytest.fixture(
        params=[
            {"func": "cbrt", "values": [-10, 10]},
            {"func": "rsqrt", "values": [0, 10]},
        ],
        ids=["cbrt", "rsqrt"],
    )
    def func_params(self, request):
        return request.param

    @pytest.mark.usefixtures("suppress_divide_numpy_warnings")
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_complex=True)
    )
    def test_basic(self, func_params, dtype):
        func = func_params["func"]
        values = func_params["values"]
        a = generate_random_numpy_array(
            10, dtype, low=values[0], high=values[1]
        )
        if func == "rsqrt":
            expected = numpy.reciprocal(numpy.sqrt(a))
        else:
            expected = getattr(numpy, func)(a)

        ia = dpnp.array(a)
        dt_out = map_dtype_to_device(expected.dtype, ia.sycl_device)
        iout = dpnp.empty(expected.shape, dtype=dt_out)
        result = getattr(dpnp, func)(ia, out=iout)
        assert result is iout
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dt_out", get_all_dtypes(no_none=True, no_complex=True)[:-1]
    )
    def test_invalid_dtype(self, func_params, dt_out):
        func = func_params["func"]
        dt_in = get_all_dtypes(no_none=True, no_complex=True)[-1]
        a = dpnp.arange(10, dtype=dt_in)
        iout = dpnp.empty(10, dtype=dt_out)
        assert_raises(ValueError, getattr(dpnp, func), a, out=iout)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15,)", "(2, 2)"]
    )
    def test_invalid_shape(self, func_params, shape):
        func = func_params["func"]
        a = dpnp.arange(10)
        iout = dpnp.empty(shape)
        assert_raises(ValueError, getattr(dpnp, func), a, out=iout)

    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["4", "()", "[]", "(3, 7)", "[2, 4]"],
    )
    def test_invalid_out(self, func_params, out):
        func = func_params["func"]
        a = dpnp.arange(10)
        assert_raises(TypeError, getattr(dpnp, func), a, out)


class TestUmath:
    @pytest.fixture(
        params=[
            {"func": "arccos", "values": [-1, 1]},
            {"func": "arccosh", "values": [1, 10]},
            {"func": "arcsin", "values": [-1, 1]},
            {"func": "arcsinh", "values": [-5, 5]},
            {"func": "arctan", "values": [-5, 5]},
            {"func": "arctanh", "values": [-1, 1]},
            {"func": "cos", "values": [-5, 5]},
            {"func": "cosh", "values": [-5, 5]},
            {"func": "exp", "values": [-3, 8]},
            {"func": "exp2", "values": [-5, 5]},
            {"func": "expm1", "values": [-5, 5]},
            {"func": "log", "values": [0, 10]},
            {"func": "log10", "values": [0, 10]},
            {"func": "log2", "values": [0, 10]},
            {"func": "log1p", "values": [0, 10]},
            {"func": "sin", "values": [-5, 5]},
            {"func": "sinh", "values": [-5, 5]},
            {"func": "sqrt", "values": [0, 10]},
            {"func": "square", "values": [-10, 10]},
            {"func": "tan", "values": [-1.5, 1.5]},
            {"func": "tanh", "values": [-5, 5]},
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
            "square",
            "tan",
            "tanh",
        ],
    )
    def func_params(self, request):
        return request.param

    @pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
    @pytest.mark.usefixtures("suppress_divide_invalid_numpy_warnings")
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_basic(self, func_params, dtype):
        func = func_params["func"]
        values = func_params["values"]
        a = generate_random_numpy_array(
            10, dtype, low=values[0], high=values[1]
        )
        expected = getattr(numpy, func)(a)

        ia = dpnp.array(a)
        dt_out = map_dtype_to_device(expected.dtype, ia.sycl_device)
        iout = dpnp.empty(expected.shape, dtype=dt_out)
        result = getattr(dpnp, func)(ia, out=iout)

        assert result is iout
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dt_out", get_all_dtypes(no_none=True)[:-1])
    def test_invalid_dtype(self, func_params, dt_out):
        func = func_params["func"]
        dt_in = get_all_dtypes(no_none=True)[-1]
        a = dpnp.arange(10, dtype=dt_in)
        iout = dpnp.empty(10, dtype=dt_out)
        assert_raises(ValueError, getattr(dpnp, func), a, out=iout)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15,)", "(2, 2)"]
    )
    def test_invalid_shape(self, xp, func_params, shape):
        func = func_params["func"]
        a = xp.arange(10)
        iout = xp.empty(shape)
        assert_raises(ValueError, getattr(xp, func), a, out=iout)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["scalar", "empty_tuple", "empty_list", "tuple", "list"],
    )
    def test_invalid_out(self, func_params, xp, out):
        func = func_params["func"]
        a = xp.arange(10)
        assert_raises(TypeError, getattr(xp, func), a, out)


def test_trigonometric_hyperbolic_aliases():
    a = dpnp.array([-0.5, 0, 0.5])

    assert_array_equal(dpnp.arcsin(a), dpnp.asin(a))
    assert_array_equal(dpnp.arccos(a), dpnp.acos(a))
    assert_array_equal(dpnp.arctan(a), dpnp.atan(a))
    assert_array_equal(dpnp.arctanh(a), dpnp.atanh(a))
    assert_array_equal(dpnp.arcsinh(a), dpnp.asinh(a))

    a = dpnp.array([1, 1.5, 2])
    assert_array_equal(dpnp.arccosh(a), dpnp.acosh(a))
