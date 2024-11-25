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
    get_all_dtypes,
    get_complex_dtypes,
    get_float_complex_dtypes,
    get_float_dtypes,
    get_integer_dtypes,
    has_support_aspect16,
)
from .test_umath import (
    _get_numpy_arrays_2in_1out,
    _get_output_data_type,
)

"""
The scope includes tests with only functions which are instances of
`DPNPUnaryFunc` class.

"""


class TestAdd:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_add(self, dtype):
        np_array1, np_array2, expected = _get_numpy_arrays_2in_1out(
            "add", dtype, [-5, 5, 10]
        )

        dp_array1 = dpnp.array(np_array1)
        dp_array2 = dpnp.array(np_array2)
        dp_out = dpnp.empty(expected.shape, dtype=dtype)
        result = dpnp.add(dp_array1, dp_array2, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_out_overlap(self, dtype):
        size = 1 if dtype == dpnp.bool else 15
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        dpnp.add(dp_a[size::], dp_a[::2], out=dp_a[:size:])

        np_a = numpy.arange(2 * size, dtype=dtype)
        numpy.add(np_a[size::], np_a[::2], out=np_a[:size:])

        assert_dtype_allclose(dp_a, np_a)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_inplace_strides(self, dtype):
        size = 21
        np_a = numpy.arange(size, dtype=dtype)
        np_a[::3] += 4

        dp_a = dpnp.arange(size, dtype=dtype)
        dp_a[::3] += 4

        assert_dtype_allclose(dp_a, np_a)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array1 = dpnp.arange(10)
        dp_array2 = dpnp.arange(10)
        dp_out = dpnp.empty(shape)

        with pytest.raises(ValueError):
            dpnp.add(dp_array1, dp_array2, out=dp_out)

    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["4", "()", "[]", "(3, 7)", "[2, 4]"],
    )
    def test_invalid_out(self, out):
        a = dpnp.arange(10)

        assert_raises(TypeError, dpnp.add, a, 2, out)
        assert_raises(TypeError, numpy.add, a.asnumpy(), 2, out)


class TestBoundFuncs:
    @pytest.fixture(
        params=[
            {"func_name": "fmax", "input_values": [-5, 5, 10]},
            {"func_name": "fmin", "input_values": [-5, 5, 10]},
            {"func_name": "maximum", "input_values": [-5, 5, 10]},
            {"func_name": "minimum", "input_values": [-5, 5, 10]},
        ],
        ids=[
            "fmax",
            "fmin",
            "maximum",
            "minimum",
        ],
    )
    def func_params(self, request):
        return request.param

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_out(self, func_params, dtype):
        func_name = func_params["func_name"]
        input_values = func_params["input_values"]
        np_array1, np_array2, expected = _get_numpy_arrays_2in_1out(
            func_name, dtype, input_values
        )

        dp_array1 = dpnp.array(np_array1)
        dp_array2 = dpnp.array(np_array2)
        dp_out = dpnp.empty(expected.shape, dtype=dtype)
        result = getattr(dpnp, func_name)(dp_array1, dp_array2, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_out_overlap(self, func_params, dtype):
        func_name = func_params["func_name"]
        size = 15
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        getattr(dpnp, func_name)(dp_a[size::], dp_a[::2], out=dp_a[:size:])

        np_a = numpy.arange(2 * size, dtype=dtype)
        getattr(numpy, func_name)(np_a[size::], np_a[::2], out=np_a[:size:])

        assert_dtype_allclose(dp_a, np_a)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, func_params, shape):
        func_name = func_params["func_name"]
        dp_array1 = dpnp.arange(10)
        dp_array2 = dpnp.arange(10)
        dp_out = dpnp.empty(shape)

        with pytest.raises(ValueError):
            getattr(dpnp, func_name)(dp_array1, dp_array2, out=dp_out)

    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["4", "()", "[]", "(3, 7)", "[2, 4]"],
    )
    def test_invalid_out(self, func_params, out):
        func_name = func_params["func_name"]
        a = dpnp.arange(10)

        assert_raises(TypeError, getattr(dpnp, func_name), a, 2, out)
        assert_raises(TypeError, getattr(numpy, func_name), a.asnumpy(), 2, out)


class TestDivide:
    @pytest.mark.usefixtures("suppress_divide_invalid_numpy_warnings")
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_divide(self, dtype):
        np_array1, np_array2, expected = _get_numpy_arrays_2in_1out(
            "divide", dtype, [-5, 5, 10]
        )

        dp_array1 = dpnp.array(np_array1)
        dp_array2 = dpnp.array(np_array2)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.divide(dp_array1, dp_array2, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_divide_invalid_numpy_warnings")
    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_out_overlap(self, dtype):
        size = 15
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        dpnp.divide(dp_a[size::], dp_a[::2], out=dp_a[:size:])

        np_a = numpy.arange(2 * size, dtype=dtype)
        numpy.divide(np_a[size::], np_a[::2], out=np_a[:size:])

        assert_dtype_allclose(dp_a, np_a)

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_inplace_strides(self, dtype):
        size = 21
        np_a = numpy.arange(size, dtype=dtype)
        np_a[::3] /= 4

        dp_a = dpnp.arange(size, dtype=dtype)
        dp_a[::3] /= 4

        assert_allclose(dp_a, np_a)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array1 = dpnp.arange(10)
        dp_array2 = dpnp.arange(5, 15)
        dp_out = dpnp.empty(shape)

        with pytest.raises(ValueError):
            dpnp.divide(dp_array1, dp_array2, out=dp_out)

    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["4", "()", "[]", "(3, 7)", "[2, 4]"],
    )
    def test_invalid_out(self, out):
        a = dpnp.arange(10)

        assert_raises(TypeError, dpnp.divide, a, 2, out)
        assert_raises(TypeError, numpy.divide, a.asnumpy(), 2, out)


class TestFloorDivide:
    @pytest.mark.usefixtures("suppress_divide_numpy_warnings")
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_floor_divide(self, dtype):
        np_array1, np_array2, expected = _get_numpy_arrays_2in_1out(
            "floor_divide", dtype, [-5, 5, 10]
        )

        dp_array1 = dpnp.array(np_array1)
        dp_array2 = dpnp.array(np_array2)
        dp_out = dpnp.empty(expected.shape, dtype=dtype)
        result = dpnp.floor_divide(dp_array1, dp_array2, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_divide_numpy_warnings")
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_out_overlap(self, dtype):
        size = 15
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        dpnp.floor_divide(dp_a[size::], dp_a[::2], out=dp_a[:size:])

        # original
        np_a = numpy.arange(2 * size, dtype=dtype)
        numpy.floor_divide(np_a[size::], np_a[::2], out=np_a[:size:])

        assert_dtype_allclose(dp_a, np_a)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True, no_complex=True)
    )
    def test_inplace_strides(self, dtype):
        size = 21

        np_a = numpy.arange(size, dtype=dtype)
        np_a[::3] //= 4

        dp_a = dpnp.arange(size, dtype=dtype)
        dp_a[::3] //= 4

        assert_dtype_allclose(dp_a, np_a)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array1 = dpnp.arange(10)
        dp_array2 = dpnp.arange(5, 15)
        dp_out = dpnp.empty(shape)

        with pytest.raises(ValueError):
            dpnp.floor_divide(dp_array1, dp_array2, out=dp_out)

    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["4", "()", "[]", "(3, 7)", "[2, 4]"],
    )
    def test_invalid_out(self, out):
        a = dpnp.arange(10)
        assert_raises(TypeError, dpnp.floor_divide, a, 2, out)
        assert_raises(TypeError, numpy.floor_divide, a.asnumpy(), 2, out)


class TestFmaxFmin:
    @pytest.mark.skipif(not has_support_aspect16(), reason="no fp16 support")
    @pytest.mark.parametrize("func", ["fmax", "fmin"])
    def test_half(self, func):
        a = numpy.array([0, 1, 2, 4, 2], dtype=numpy.float16)
        b = numpy.array([-2, 5, 1, 4, 3], dtype=numpy.float16)
        c = numpy.array([0, -1, -numpy.inf, numpy.nan, 6], dtype=numpy.float16)
        ia, ib, ic = dpnp.array(a), dpnp.array(b), dpnp.array(c)

        result = getattr(dpnp, func)(ia, ib)
        expected = getattr(numpy, func)(a, b)
        assert_equal(result, expected)

        result = getattr(dpnp, func)(ib, ic)
        expected = getattr(numpy, func)(b, c)
        assert_equal(result, expected)

    @pytest.mark.parametrize("func", ["fmax", "fmin"])
    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_float_nans(self, func, dtype):
        a = numpy.array([0, numpy.nan, numpy.nan], dtype=dtype)
        b = numpy.array([numpy.nan, 0, numpy.nan], dtype=dtype)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = getattr(dpnp, func)(ia, ib)
        expected = getattr(numpy, func)(a, b)
        assert_equal(result, expected)

    @pytest.mark.parametrize("func", ["fmax", "fmin"])
    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize(
        "nan_val",
        [
            complex(numpy.nan, 0),
            complex(0, numpy.nan),
            complex(numpy.nan, numpy.nan),
        ],
        ids=["nan+0j", "nanj", "nan+nanj"],
    )
    def test_complex_nans(self, func, dtype, nan_val):
        a = numpy.array([0, nan_val, nan_val], dtype=dtype)
        b = numpy.array([nan_val, 0, nan_val], dtype=dtype)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = getattr(dpnp, func)(ia, ib)
        expected = getattr(numpy, func)(a, b)
        assert_equal(result, expected)

    @pytest.mark.parametrize("func", ["fmax", "fmin"])
    @pytest.mark.parametrize("dtype", get_float_dtypes(no_float16=False))
    def test_precision(self, func, dtype):
        dtmin = numpy.finfo(dtype).min
        dtmax = numpy.finfo(dtype).max
        d1 = dtype(0.1)
        d1_next = numpy.nextafter(d1, numpy.inf)

        test_cases = [
            # v1     v2
            (dtmin, -numpy.inf),
            (dtmax, -numpy.inf),
            (d1, d1_next),
            (dtmax, numpy.nan),
        ]

        for v1, v2 in test_cases:
            a = numpy.array([v1])
            b = numpy.array([v2])
            ia, ib = dpnp.array(a), dpnp.array(b)

            result = getattr(dpnp, func)(ia, ib)
            expected = getattr(numpy, func)(a, b)
            assert_allclose(result, expected)


class TestHeavside:
    @pytest.mark.parametrize("val", [0.5, 1.0])
    @pytest.mark.parametrize("dt", get_float_dtypes())
    def test_basic(self, val, dt):
        a = numpy.array(
            [[-30.0, -0.1, 0.0, 0.2], [7.5, numpy.nan, numpy.inf, -numpy.inf]],
            dtype=dt,
        )
        ia = dpnp.array(a)

        result = dpnp.heaviside(ia, val)
        expected = numpy.heaviside(a, val)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "a_dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize(
        "b_dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    def test_both_input_as_arrays(self, a_dt, b_dt):
        a = numpy.array([-1.5, 0, 2.0], dtype=a_dt)
        b = numpy.array([-0, 0.5, 1.0], dtype=b_dt)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.heaviside(ia, ib)
        expected = numpy.heaviside(a, b)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    @pytest.mark.parametrize("dt", get_complex_dtypes())
    def test_complex_dtype(self, xp, dt):
        a = xp.array([-1.5, 0, 2.0], dtype=dt)
        assert_raises((TypeError, ValueError), xp.heaviside, a, 0.5)


class TestLdexp:
    @pytest.mark.parametrize("mant_dt", get_float_dtypes())
    @pytest.mark.parametrize("exp_dt", get_integer_dtypes())
    def test_basic(self, mant_dt, exp_dt):
        if (
            numpy.lib.NumpyVersion(numpy.__version__) < "2.0.0"
            and exp_dt == numpy.int64
            and numpy.dtype("l") != numpy.int64
        ):
            pytest.skip("numpy.ldexp doesn't have a loop for the input types")

        mant = numpy.array(2.0, dtype=mant_dt)
        exp = numpy.array(3, dtype=exp_dt)
        imant, iexp = dpnp.array(mant), dpnp.array(exp)

        result = dpnp.ldexp(imant, iexp)
        expected = numpy.ldexp(mant, exp)
        assert_almost_equal(result, expected)

    def test_float_scalar(self):
        a = numpy.array(3)
        ia = dpnp.array(a)

        result = dpnp.ldexp(2.0, ia)
        expected = numpy.ldexp(2.0, a)
        assert_almost_equal(result, expected)

    @pytest.mark.parametrize("max_min", ["max", "min"])
    def test_overflow(self, max_min):
        exp_val = getattr(numpy.iinfo(numpy.dtype("l")), max_min)

        result = dpnp.ldexp(dpnp.array(2.0), exp_val)
        with numpy.errstate(over="ignore"):
            # we can't use here numpy.array(2.0), because NumPy 2.0 will cast
            # `exp_val` to int32 dtype then and `OverflowError` will be raised
            expected = numpy.ldexp(2.0, exp_val)
        assert_equal(result, expected)

    @pytest.mark.parametrize("val", [numpy.nan, numpy.inf, -numpy.inf])
    def test_nan_int_mant(self, val):
        mant = numpy.array(val)
        imant = dpnp.array(mant)

        result = dpnp.ldexp(imant, 5)
        expected = numpy.ldexp(mant, 5)
        assert_equal(result, expected)

    def test_zero_exp(self):
        exp = numpy.array(0)
        iexp = dpnp.array(exp)

        result = dpnp.ldexp(-2.5, iexp)
        expected = numpy.ldexp(-2.5, exp)
        assert_equal(result, expected)

    @pytest.mark.parametrize("stride", [-4, -2, -1, 1, 2, 4])
    @pytest.mark.parametrize("dt", get_float_dtypes())
    def test_strides(self, stride, dt):
        mant = numpy.array(
            [0.125, 0.25, 0.5, 1.0, 1.0, 2.0, 4.0, 8.0], dtype=dt
        )
        exp = numpy.array([3, 2, 1, 0, 0, -1, -2, -3], dtype="i")
        out = numpy.zeros(8, dtype=dt)
        imant, iexp, iout = dpnp.array(mant), dpnp.array(exp), dpnp.array(out)

        result = dpnp.ldexp(imant[::stride], iexp[::stride], out=iout[::stride])
        expected = numpy.ldexp(mant[::stride], exp[::stride], out=out[::stride])
        assert_equal(result, expected)

    def test_bool_exp(self):
        result = dpnp.ldexp(3.7, dpnp.array(True))
        expected = numpy.ldexp(3.7, numpy.array(True))
        assert_almost_equal(result, expected)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    def test_uint64_exp(self, xp):
        x = xp.array(4, dtype=numpy.uint64)
        assert_raises((ValueError, TypeError), xp.ldexp, 7.3, x)


class TestMultiply:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_multiply(self, dtype):
        np_array1, np_array2, expected = _get_numpy_arrays_2in_1out(
            "multiply", dtype, [0, 10, 10]
        )

        dp_array1 = dpnp.array(np_array1)
        dp_array2 = dpnp.array(np_array2)
        dp_out = dpnp.empty(expected.shape, dtype=dtype)
        result = dpnp.multiply(dp_array1, dp_array2, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_out_overlap(self, dtype):
        size = 1 if dtype == dpnp.bool else 15
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        dpnp.multiply(dp_a[size::], dp_a[::2], out=dp_a[:size:])

        np_a = numpy.arange(2 * size, dtype=dtype)
        numpy.multiply(np_a[size::], np_a[::2], out=np_a[:size:])

        assert_dtype_allclose(dp_a, np_a)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_inplace_strides(self, dtype):
        size = 21
        np_a = numpy.arange(size, dtype=dtype)
        np_a[::3] *= 4

        dp_a = dpnp.arange(size, dtype=dtype)
        dp_a[::3] *= 4

        assert_dtype_allclose(dp_a, np_a)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array1 = dpnp.arange(10)
        dp_array2 = dpnp.arange(10)
        dp_out = dpnp.empty(shape)

        with pytest.raises(ValueError):
            dpnp.multiply(dp_array1, dp_array2, out=dp_out)

    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["4", "()", "[]", "(3, 7)", "[2, 4]"],
    )
    def test_invalid_out(self, out):
        a = dpnp.arange(10)

        assert_raises(TypeError, dpnp.multiply, a, 2, out)
        assert_raises(TypeError, numpy.multiply, a.asnumpy(), 2, out)


class TestNextafter:
    @pytest.mark.parametrize("dt", get_float_dtypes())
    @pytest.mark.parametrize(
        "val1, val2",
        [
            pytest.param(1, 2),
            pytest.param(1, 0),
            pytest.param(1, 1),
        ],
    )
    def test_float(self, val1, val2, dt):
        v1 = numpy.array(val1, dtype=dt)
        v2 = numpy.array(val2, dtype=dt)
        iv1, iv2 = dpnp.array(v1), dpnp.array(v2)

        result = dpnp.nextafter(iv1, iv2)
        expected = numpy.nextafter(v1, v2)
        assert_equal(result, expected)

    @pytest.mark.parametrize("dt", get_float_dtypes())
    def test_float_nan(self, dt):
        a = numpy.array(1, dtype=dt)
        ia = dpnp.array(a)

        result = dpnp.nextafter(ia, dpnp.nan)
        expected = numpy.nextafter(a, numpy.nan)
        assert_equal(result, expected)

        result = dpnp.nextafter(dpnp.nan, ia)
        expected = numpy.nextafter(numpy.nan, a)
        assert_equal(result, expected)

    @pytest.mark.parametrize("val", [0x7C00, 0x8000], ids=["val1", "val2"])
    def test_f16_strides(self, val):
        a = numpy.arange(val, dtype=numpy.uint16).astype(numpy.float16)
        hinf = numpy.array((numpy.inf,), dtype=numpy.float16)
        ia, ihinf = dpnp.array(a), dpnp.array(hinf)

        result = dpnp.nextafter(ia[:-1], ihinf)
        expected = numpy.nextafter(a[:-1], hinf)
        assert_equal(result, expected)

        result = dpnp.nextafter(ia[0], -ihinf)
        expected = numpy.nextafter(a[0], -hinf)
        assert_equal(result, expected)

        result = dpnp.nextafter(ia[1:], -ihinf)
        expected = numpy.nextafter(a[1:], -hinf)
        assert_equal(result, expected)

    @pytest.mark.parametrize("val", [0x7C00, 0x8000], ids=["val1", "val2"])
    def test_f16_array_inf(self, val):
        a = numpy.arange(val, dtype=numpy.uint16).astype(numpy.float16)
        hinf = numpy.array((numpy.inf,), dtype=numpy.float16)
        ia, ihinf = dpnp.array(a), dpnp.array(hinf)

        result = dpnp.nextafter(ihinf, ia)
        expected = numpy.nextafter(hinf, a)
        assert_equal(result, expected)

        result = dpnp.nextafter(-ihinf, ia)
        expected = numpy.nextafter(-hinf, a)
        assert_equal(result, expected)

    @pytest.mark.parametrize(
        "sign1, sign2",
        [
            pytest.param(1, 1),
            pytest.param(1, -1),
            pytest.param(-1, 1),
            pytest.param(-1, -1),
        ],
    )
    def test_f16_inf(self, sign1, sign2):
        hinf1 = numpy.array((sign1 * numpy.inf,), dtype=numpy.float16)
        hinf2 = numpy.array((sign2 * numpy.inf,), dtype=numpy.float16)
        ihinf1, ihinf2 = dpnp.array(hinf1), dpnp.array(hinf2)

        result = dpnp.nextafter(ihinf1, ihinf2)
        expected = numpy.nextafter(hinf1, hinf2)
        assert_equal(result, expected)

    @pytest.mark.parametrize("val", [0x7C00, 0x8000], ids=["val1", "val2"])
    def test_f16_array_nan(self, val):
        a = numpy.arange(val, dtype=numpy.uint16).astype(numpy.float16)
        nan = numpy.array((numpy.nan,), dtype=numpy.float16)
        ia, inan = dpnp.array(a), dpnp.array(nan)

        result = dpnp.nextafter(ia, inan)
        expected = numpy.nextafter(a, nan)
        assert_equal(result, expected)

        result = dpnp.nextafter(inan, ia)
        expected = numpy.nextafter(nan, a)
        assert_equal(result, expected)

    @pytest.mark.parametrize(
        "val1, val2",
        [
            pytest.param(numpy.nan, numpy.nan),
            pytest.param(numpy.inf, numpy.nan),
            pytest.param(numpy.nan, numpy.inf),
        ],
    )
    def test_f16_inf_nan(self, val1, val2):
        v1 = numpy.array((val1,), dtype=numpy.float16)
        v2 = numpy.array((val2,), dtype=numpy.float16)
        iv1, iv2 = dpnp.array(v1), dpnp.array(v2)

        result = dpnp.nextafter(iv1, iv2)
        expected = numpy.nextafter(v1, v2)
        assert_equal(result, expected)

    @pytest.mark.parametrize(
        "val, scalar",
        [
            pytest.param(65504, -numpy.inf),
            pytest.param(-65504, numpy.inf),
            pytest.param(numpy.inf, 0),
            pytest.param(-numpy.inf, 0),
            pytest.param(0, numpy.nan),
            pytest.param(numpy.nan, 0),
        ],
    )
    def test_f16_corner_values_with_scalar(self, val, scalar):
        a = numpy.array(val, dtype=numpy.float16)
        ia = dpnp.array(a)
        scalar = numpy.float16(scalar)

        result = dpnp.nextafter(ia, scalar)
        expected = numpy.nextafter(a, scalar)
        assert_equal(result, expected)


class TestPower:
    @pytest.mark.parametrize("val_type", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("data_type", get_all_dtypes())
    @pytest.mark.parametrize("val", [1.5, 1, 5], ids=["1.5", "1", "5"])
    @pytest.mark.parametrize(
        "array",
        [
            [[0, 0], [0, 0]],
            [[1, 2], [1, 2]],
            [[1, 2], [3, 4]],
            [[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]],
            [
                [[[1, 2], [3, 4]], [[1, 2], [2, 1]]],
                [[[1, 3], [3, 1]], [[0, 1], [1, 3]]],
            ],
        ],
        ids=[
            "[[0, 0], [0, 0]]",
            "[[1, 2], [1, 2]]",
            "[[1, 2], [3, 4]]",
            "[[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]",
            "[[[[1, 2], [3, 4]], [[1, 2], [2, 1]]], [[[1, 3], [3, 1]], [[0, 1], [1, 3]]]]",
        ],
    )
    def test_basic(self, array, val, data_type, val_type):
        np_a = numpy.array(array, dtype=data_type)
        dpnp_a = dpnp.array(array, dtype=data_type)
        val_ = val_type(val)

        result = dpnp.power(dpnp_a, val_)
        expected = numpy.power(np_a, val_)
        assert_allclose(expected, result, rtol=1e-6)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_power(self, dtype):
        numpy.random.seed(42)
        np_array1, np_array2, expected = _get_numpy_arrays_2in_1out(
            "power", dtype, [0, 10, 10]
        )

        dp_array1 = dpnp.array(np_array1)
        dp_array2 = dpnp.array(np_array2)
        out_dtype = numpy.int8 if dtype == numpy.bool_ else dtype
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.power(dp_array1, dp_array2, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_out_overlap(self, dtype):
        size = 1 if dtype == dpnp.bool else 10
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        dpnp.power(dp_a[size::], dp_a[::2], out=dp_a[:size:]),

        np_a = numpy.arange(2 * size, dtype=dtype)
        numpy.power(np_a[size::], np_a[::2], out=np_a[:size:])

        assert_dtype_allclose(dp_a, np_a)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    def test_inplace_strided_out(self, dtype):
        size = 5
        np_a = numpy.arange(2 * size, dtype=dtype)
        np_a[::3] **= 3

        dp_a = dpnp.arange(2 * size, dtype=dtype)
        dp_a[::3] **= 3

        assert_dtype_allclose(dp_a, np_a)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array1 = dpnp.arange(10)
        dp_array2 = dpnp.arange(10)
        dp_out = dpnp.empty(shape)

        with pytest.raises(ValueError):
            dpnp.power(dp_array1, dp_array2, out=dp_out)

    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["4", "()", "[]", "(3, 7)", "[2, 4]"],
    )
    def test_invalid_out(self, out):
        a = dpnp.arange(10)

        assert_raises(TypeError, dpnp.power, a, 2, out)
        assert_raises(TypeError, numpy.power, a.asnumpy(), 2, out)

    @pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
    def test_complex_values(self):
        np_arr = numpy.array([0j, 1 + 1j, 0 + 2j, 1 + 2j, numpy.nan, numpy.inf])
        dp_arr = dpnp.array(np_arr)
        func = lambda x: x**2

        assert_dtype_allclose(func(dp_arr), func(np_arr))

    @pytest.mark.parametrize("val", [0, 1], ids=["0", "1"])
    @pytest.mark.parametrize("dtype", [dpnp.int32, dpnp.int64])
    def test_integer_power_of_0_or_1(self, val, dtype):
        np_arr = numpy.arange(10, dtype=dtype)
        dp_arr = dpnp.array(np_arr)
        func = lambda x: val**x

        assert_equal(func(np_arr), func(dp_arr))

    @pytest.mark.parametrize("dtype", [dpnp.int32, dpnp.int64])
    def test_integer_to_negative_power(self, dtype):
        a = dpnp.arange(2, 10, dtype=dtype)
        b = dpnp.full(8, -2, dtype=dtype)
        zeros = dpnp.zeros(8, dtype=dtype)
        ones = dpnp.ones(8, dtype=dtype)

        assert_array_equal(ones ** (-2), zeros)
        assert_equal(
            a ** (-3), zeros
        )  # positive integer to negative integer power
        assert_equal(
            b ** (-4), zeros
        )  # negative integer to negative integer power

    def test_float_to_inf(self):
        a = numpy.array(
            [1, 1, 2, 2, -2, -2, numpy.inf, -numpy.inf], dtype=numpy.float32
        )
        b = numpy.array(
            [
                numpy.inf,
                -numpy.inf,
                numpy.inf,
                -numpy.inf,
                numpy.inf,
                -numpy.inf,
                numpy.inf,
                -numpy.inf,
            ],
            dtype=numpy.float32,
        )
        numpy_res = a**b
        dpnp_res = dpnp.array(a) ** dpnp.array(b)

        assert_allclose(numpy_res, dpnp_res.asnumpy())

    @pytest.mark.parametrize("shape", [(), (3, 2)], ids=["()", "(3, 2)"])
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_power_scalar(self, shape, dtype):
        np_a = numpy.ones(shape, dtype=dtype)
        dpnp_a = dpnp.ones(shape, dtype=dtype)

        result = 4.2**dpnp_a**-1.3
        expected = 4.2**np_a**-1.3
        assert_allclose(result, expected, rtol=1e-6)

        result **= dpnp_a
        expected **= np_a
        assert_allclose(result, expected, rtol=1e-6)

    def test_alias(self):
        a = dpnp.arange(10)
        res1 = dpnp.power(a, 3)
        res2 = dpnp.pow(a, 3)

        assert_array_equal(res1, res2)


class TestRationalFunctions:
    @pytest.mark.parametrize("func", ["gcd", "lcm"])
    @pytest.mark.parametrize("dt1", get_integer_dtypes())
    @pytest.mark.parametrize("dt2", get_integer_dtypes())
    def test_basic(self, func, dt1, dt2):
        a = numpy.array([12, 120], dtype=dt1)
        b = numpy.array([20, 120], dtype=dt2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        expected = getattr(numpy, func)(a, b)
        result = getattr(dpnp, func)(ia, ib)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("func", ["gcd", "lcm"])
    @pytest.mark.parametrize("dt", get_integer_dtypes())
    def test_broadcasting(self, func, dt):
        a = numpy.arange(6, dtype=dt)
        ia = dpnp.array(a)
        b = 20

        expected = getattr(numpy, func)(a, b)
        result = getattr(dpnp, func)(ia, b)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dt", [numpy.int32, numpy.int64])
    def test_gcd_overflow(self, dt):
        a = dt(numpy.iinfo(dt).min)  # negative power of two
        ia = dpnp.array(a)
        q = -(a // 4)

        # verify that we don't overflow when taking abs(x)
        # not relevant for lcm, where the result is unrepresentable anyway
        expected = numpy.gcd(a, q)
        result = dpnp.gcd(ia, q)
        assert_array_equal(result, expected)

    def test_lcm_overflow(self):
        big = numpy.int32(numpy.iinfo(numpy.int32).max // 11)
        a, b = 2 * big, 5 * big
        ia, ib = dpnp.array(a), dpnp.array(b)

        # verify that we don't overflow when a*b does overflow
        expected = numpy.lcm(a, b)
        result = dpnp.lcm(ia, ib)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("func", ["gcd", "lcm"])
    @pytest.mark.parametrize("xp", [dpnp, numpy])
    def test_inf_and_nan(self, func, xp):
        inf = xp.array([xp.inf])
        assert_raises((TypeError, ValueError), getattr(xp, func), inf, 1)
        assert_raises((TypeError, ValueError), getattr(xp, func), 1, inf)
        assert_raises((TypeError, ValueError), getattr(xp, func), xp.nan, inf)
        assert_raises(
            (TypeError, ValueError), getattr(xp, func), 4, float(xp.inf)
        )
