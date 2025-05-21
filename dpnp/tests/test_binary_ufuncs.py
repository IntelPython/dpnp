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
from dpnp.dpnp_utils import map_dtype_to_device

from .helper import (
    assert_dtype_allclose,
    generate_random_numpy_array,
    get_abs_array,
    get_all_dtypes,
    get_complex_dtypes,
    get_float_complex_dtypes,
    get_float_dtypes,
    get_integer_dtypes,
    get_integer_float_dtypes,
    has_support_aspect16,
    numpy_version,
)

"""
The scope includes tests with only functions which are instances of
`DPNPUnaryFunc` class.

"""


class TestAdd:
    ALL_DTYPES = get_all_dtypes(no_none=True)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_add(self, dtype):
        a = generate_random_numpy_array(10, dtype)
        b = generate_random_numpy_array(10, dtype)
        expected = numpy.add(a, b)

        ia, ib = dpnp.array(a), dpnp.array(b)
        iout = dpnp.empty(expected.shape, dtype=dtype)
        result = dpnp.add(ia, ib, out=iout)

        assert result is iout
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_out_overlap(self, dtype):
        size = 1 if dtype == dpnp.bool else 15
        a = numpy.arange(2 * size, dtype=dtype)
        ia = dpnp.array(a)

        dpnp.add(ia[size::], ia[::2], out=ia[:size:])
        numpy.add(a[size::], a[::2], out=a[:size:])

        assert_dtype_allclose(ia, a)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    def test_inplace_strides(self, dtype):
        size = 21
        a = numpy.arange(size, dtype=dtype)
        a[::3] += 4

        ia = dpnp.arange(size, dtype=dtype)
        ia[::3] += 4

        assert_dtype_allclose(ia, a)

    @pytest.mark.parametrize("dtype1", ALL_DTYPES)
    @pytest.mark.parametrize("dtype2", ALL_DTYPES)
    def test_inplace_dtype(self, dtype1, dtype2):
        a = get_abs_array([[-7, 6, -3, 2, -1], [0, -3, 4, 5, -6]], dtype=dtype1)
        b = get_abs_array([5, -2, 0, 1, 0], dtype=dtype2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        if numpy.issubdtype(dtype1, numpy.signedinteger) and numpy.issubdtype(
            dtype2, numpy.uint64
        ):
            # For this special case, NumPy raises an error but dpnp works
            b = b.astype(numpy.int64)
            a += b
            ia += ib
            assert_dtype_allclose(ia, a)
        elif numpy.can_cast(dtype2, dtype1, casting="same_kind"):
            a += b
            ia += ib
            assert_dtype_allclose(ia, a)
        else:
            with pytest.raises(TypeError):
                a += b

            with pytest.raises(ValueError):
                ia += ib

    @pytest.mark.parametrize("dtype1", ALL_DTYPES)
    @pytest.mark.parametrize("dtype2", ALL_DTYPES)
    def test_inplace_dtype_explicit(self, dtype1, dtype2):
        a = get_abs_array([[-7, 6, -3, 2, -1], [0, -3, 4, 5, -6]], dtype=dtype1)
        b = get_abs_array([5, -2, 0, 1, 0], dtype=dtype2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        if numpy.issubdtype(dtype1, numpy.signedinteger) and numpy.issubdtype(
            dtype2, numpy.uint64
        ):
            # For this special case, NumPy raises an error but dpnp works
            result = dpnp.add(ia, ib, out=ia)
            expected = numpy.add(a, b.astype(numpy.int64), out=a)
            assert_dtype_allclose(result, expected)
        elif numpy.can_cast(dtype2, dtype1, casting="same_kind"):
            result = dpnp.add(ia, ib, out=ia)
            expected = numpy.add(a, b, out=a)
            assert_dtype_allclose(result, expected)
        else:
            assert_raises(TypeError, numpy.add, a, b, out=a)
            assert_raises(ValueError, dpnp.add, ia, ib, out=ia)

    @pytest.mark.parametrize("shape", [(0,), (15,), (2, 2)])
    def test_invalid_shape(self, shape):
        a, b = dpnp.arange(10), dpnp.arange(10)
        out = dpnp.empty(shape)

        with pytest.raises(ValueError):
            dpnp.add(a, b, out=out)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["scalar", "empty_tuple", "empty_list", "tuple", "list"],
    )
    def test_invalid_out(self, xp, out):
        a = xp.arange(10)
        assert_raises(TypeError, xp.add, a, 2, out)


@pytest.mark.parametrize("func", ["fmax", "fmin", "maximum", "minimum"])
class TestBoundFuncs:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_basic(self, func, dtype):
        a = generate_random_numpy_array(10, dtype)
        b = generate_random_numpy_array(10, dtype)
        expected = getattr(numpy, func)(a, b)

        ia, ib = dpnp.array(a), dpnp.array(b)
        iout = dpnp.empty(expected.shape, dtype=dtype)
        result = getattr(dpnp, func)(ia, ib, out=iout)

        assert result is iout
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    def test_out_overlap(self, func, dtype):
        size = 15
        a = numpy.arange(2 * size, dtype=dtype)
        ia = dpnp.array(a)

        getattr(dpnp, func)(ia[size::], ia[::2], out=ia[:size:])
        getattr(numpy, func)(a[size::], a[::2], out=a[:size:])

        assert_dtype_allclose(ia, a)

    @pytest.mark.parametrize("shape", [(0,), (15,), (2, 2)])
    def test_invalid_shape(self, func, shape):
        a, b = dpnp.arange(10), dpnp.arange(10)
        out = dpnp.empty(shape)

        with pytest.raises(ValueError):
            getattr(dpnp, func)(a, b, out=out)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["scalar", "empty_tuple", "empty_list", "tuple", "list"],
    )
    def test_invalid_out(self, func, xp, out):
        a = xp.arange(10)
        assert_raises(TypeError, getattr(xp, func), a, 2, out)


class TestDivide:
    @pytest.mark.usefixtures("suppress_divide_invalid_numpy_warnings")
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_divide(self, dtype):
        a = generate_random_numpy_array(10, dtype)
        b = generate_random_numpy_array(10, dtype)
        expected = numpy.divide(a, b)

        ia, ib = dpnp.array(a), dpnp.array(b)
        out_dtype = map_dtype_to_device(expected.dtype, ia.sycl_device)
        iout = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.divide(ia, ib, out=iout)

        assert result is iout
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_divide_invalid_numpy_warnings")
    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_out_overlap(self, dtype):
        size = 15
        a = numpy.arange(2 * size, dtype=dtype)
        ia = dpnp.array(a)

        dpnp.divide(ia[size::], ia[::2], out=ia[:size:])
        numpy.divide(a[size::], a[::2], out=a[:size:])

        assert_dtype_allclose(ia, a)

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_inplace_strides(self, dtype):
        size = 21
        a = numpy.arange(size, dtype=dtype)
        a[::3] /= 4

        ia = dpnp.arange(size, dtype=dtype)
        ia[::3] /= 4

        assert_allclose(ia, a)

    @pytest.mark.parametrize("dtype1", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("dtype2", get_float_complex_dtypes())
    def test_inplace_dtype(self, dtype1, dtype2):
        a = get_abs_array([[-7, 6, -3, 2, -1], [0, -3, 4, 5, -6]], dtype=dtype1)
        b = get_abs_array([5, -2, -10, 1, 10], dtype=dtype2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        if numpy.can_cast(dtype2, dtype1, casting="same_kind"):
            a /= b
            ia /= ib
            assert_dtype_allclose(ia, a)
        else:
            with pytest.raises(TypeError):
                a /= b

            with pytest.raises(ValueError):
                ia /= ib

    @pytest.mark.parametrize("dtype1", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("dtype2", get_float_complex_dtypes())
    def test_inplace_dtype_explicit(self, dtype1, dtype2):
        a = get_abs_array([[-7, 6, -3, 2, -1], [0, -3, 4, 5, -6]], dtype=dtype1)
        b = get_abs_array([5, -2, -10, 1, 10], dtype=dtype2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        if numpy.can_cast(dtype2, dtype1, casting="same_kind"):
            result = dpnp.divide(ia, ib, out=ia)
            expected = numpy.divide(a, b, out=a)
            assert_dtype_allclose(result, expected)
        else:
            assert_raises(TypeError, numpy.divide, a, b, out=a)
            assert_raises(ValueError, dpnp.divide, ia, ib, out=ia)

    @pytest.mark.parametrize("shape", [(0,), (15,), (2, 2)])
    def test_invalid_shape(self, shape):
        a, b = dpnp.arange(10), dpnp.arange(10)
        out = dpnp.empty(shape)

        with pytest.raises(ValueError):
            dpnp.divide(a, b, out=out)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["scalar", "empty_tuple", "empty_list", "tuple", "list"],
    )
    def test_invalid_out(self, xp, out):
        a = xp.arange(10)
        assert_raises(TypeError, xp.divide, a, 2, out)


@pytest.mark.parametrize("func", ["floor_divide", "remainder"])
class TestFloorDivideRemainder:
    ALL_DTYPES = get_integer_float_dtypes()

    def do_inplace_op(self, base, other, func):
        if func == "floor_divide":
            base //= other
        else:
            base %= other

    @pytest.mark.usefixtures("suppress_divide_numpy_warnings")
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_basic(self, func, dtype):
        a = generate_random_numpy_array(10, dtype)
        b = generate_random_numpy_array(10, dtype, low=-5, high=5, seed_value=8)
        expected = getattr(numpy, func)(a, b)

        ia, ib = dpnp.array(a), dpnp.array(b)
        iout = dpnp.empty(expected.shape, dtype=dtype)
        result = getattr(dpnp, func)(ia, ib, out=iout)

        assert result is iout
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_out_overlap(self, func, dtype):
        size = 15
        a = numpy.arange(1, 2 * size + 1, dtype=dtype)
        ia = dpnp.array(a)

        getattr(dpnp, func)(ia[size::], ia[::2], out=ia[:size:])
        getattr(numpy, func)(a[size::], a[::2], out=a[:size:])

        assert_dtype_allclose(ia, a)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_inplace_strides(self, func, dtype):
        size = 21

        a = numpy.arange(size, dtype=dtype)
        self.do_inplace_op(a[::3], 4, func)

        ia = dpnp.arange(size, dtype=dtype)
        self.do_inplace_op(ia[::3], 4, func)

        assert_dtype_allclose(ia, a)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_inplace_scalar(self, func, dtype):

        a = numpy.array(10, dtype=dtype)
        self.do_inplace_op(10, a, func)

        ia = dpnp.array(10, dtype=dtype)
        self.do_inplace_op(10, ia, func)

        assert_dtype_allclose(ia, a)

    @pytest.mark.parametrize("dtype1", [dpnp.bool] + ALL_DTYPES)
    @pytest.mark.parametrize("dtype2", get_float_dtypes())
    def test_inplace_dtype(self, func, dtype1, dtype2):
        a = get_abs_array([[-7, 6, -3, 2, -1], [0, -3, 4, 5, -6]], dtype=dtype1)
        b = get_abs_array([5, -2, -10, 1, 10], dtype=dtype2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        if numpy.can_cast(dtype2, dtype1, casting="same_kind"):
            self.do_inplace_op(a, b, func)
            self.do_inplace_op(ia, ib, func)
            assert_dtype_allclose(ia, a)
        else:
            with pytest.raises(TypeError):
                self.do_inplace_op(a, b, func)

            with pytest.raises(ValueError):
                self.do_inplace_op(ia, ib, func)

    @pytest.mark.parametrize("shape", [(0,), (15,), (2, 2)])
    def test_invalid_shape(self, func, shape):
        a, b = dpnp.arange(10), dpnp.arange(10)
        out = dpnp.empty(shape)

        with pytest.raises(ValueError):
            getattr(dpnp, func)(a, b, out=out)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["scalar", "empty_tuple", "empty_list", "tuple", "list"],
    )
    def test_invalid_out(self, func, xp, out):
        a = xp.arange(10)
        assert_raises(TypeError, getattr(xp, func), a, 2, out)


@pytest.mark.parametrize("func", ["fmax", "fmin"])
class TestFmaxFmin:
    @pytest.mark.skipif(not has_support_aspect16(), reason="no fp16 support")
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

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_float_nans(self, func, dtype):
        a = numpy.array([0, numpy.nan, numpy.nan], dtype=dtype)
        b = numpy.array([numpy.nan, 0, numpy.nan], dtype=dtype)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = getattr(dpnp, func)(ia, ib)
        expected = getattr(numpy, func)(a, b)
        assert_equal(result, expected)

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
        a = get_abs_array([-1.5, 0, 2.0], a_dt)
        b = get_abs_array([-0, 0.5, 1.0], b_dt)
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
            numpy_version() < "2.0.0"
            and exp_dt == numpy.int64
            and numpy.dtype("l") != numpy.int64
        ):
            pytest.skip("numpy.ldexp doesn't have a loop for the input types")

        mant = numpy.array(2.0, dtype=mant_dt)
        exp = numpy.array(3, dtype=exp_dt)
        imant, iexp = dpnp.array(mant), dpnp.array(exp)

        if dpnp.issubdtype(exp_dt, dpnp.uint64):
            assert_raises(ValueError, dpnp.ldexp, imant, iexp)
            assert_raises(TypeError, numpy.ldexp, mant, exp)
        elif numpy_version() < "2.0.0" and dpnp.issubdtype(exp_dt, dpnp.uint32):
            # For this special case, NumPy < "2.0.0" raises an error on Windows
            result = dpnp.ldexp(imant, iexp)
            expected = numpy.ldexp(mant, exp.astype(numpy.int32))
            assert_almost_equal(result, expected)
        else:
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
    ALL_DTYPES = get_all_dtypes(no_none=True)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_multiply(self, dtype):
        a = generate_random_numpy_array(10, dtype)
        b = generate_random_numpy_array(10, dtype)
        expected = numpy.multiply(a, b)

        ia, ib = dpnp.array(a), dpnp.array(b)
        iout = dpnp.empty(expected.shape, dtype=dtype)
        result = dpnp.multiply(ia, ib, out=iout)

        assert result is iout
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_out_overlap(self, dtype):
        size = 1 if dtype == dpnp.bool else 15
        a = numpy.arange(2 * size, dtype=dtype)
        ia = dpnp.array(a)

        dpnp.multiply(ia[size::], ia[::2], out=ia[:size:])
        numpy.multiply(a[size::], a[::2], out=a[:size:])

        assert_dtype_allclose(ia, a)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    def test_inplace_strides(self, dtype):
        size = 21
        a = numpy.arange(size, dtype=dtype)
        a[::3] *= 4

        ia = dpnp.arange(size, dtype=dtype)
        ia[::3] *= 4

        assert_dtype_allclose(ia, a)

    @pytest.mark.parametrize("dtype1", ALL_DTYPES)
    @pytest.mark.parametrize("dtype2", ALL_DTYPES)
    def test_inplace_dtype(self, dtype1, dtype2):
        a = get_abs_array([[-7, 6, -3, 2, -1], [0, -3, 4, 5, -6]], dtype=dtype1)
        b = get_abs_array([5, -2, 0, 1, 0], dtype=dtype2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        if numpy.issubdtype(dtype1, numpy.signedinteger) and numpy.issubdtype(
            dtype2, numpy.uint64
        ):
            # For this special case, NumPy raises an error but dpnp works
            b = b.astype(numpy.int64)
            a *= b
            ia *= ib
            assert_dtype_allclose(ia, a)
        elif numpy.can_cast(dtype2, dtype1, casting="same_kind"):
            a *= b
            ia *= ib
            assert_dtype_allclose(ia, a)
        else:
            with pytest.raises(TypeError):
                a *= b

            with pytest.raises(ValueError):
                ia *= ib

    @pytest.mark.parametrize("shape", [(0,), (15,), (2, 2)])
    def test_invalid_shape(self, shape):
        a, b = dpnp.arange(10), dpnp.arange(10)
        out = dpnp.empty(shape)

        with pytest.raises(ValueError):
            dpnp.multiply(a, b, out=out)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["scalar", "empty_tuple", "empty_list", "tuple", "list"],
    )
    def test_invalid_out(self, xp, out):
        a = xp.arange(10)
        assert_raises(TypeError, xp.multiply, a, 2, out)


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

    @pytest.mark.skipif(not has_support_aspect16(), reason="no fp16 support")
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

    @pytest.mark.skipif(not has_support_aspect16(), reason="no fp16 support")
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

    @pytest.mark.skipif(not has_support_aspect16(), reason="no fp16 support")
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

    @pytest.mark.skipif(not has_support_aspect16(), reason="no fp16 support")
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

    @pytest.mark.skipif(not has_support_aspect16(), reason="no fp16 support")
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

    @pytest.mark.skipif(not has_support_aspect16(), reason="no fp16 support")
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
    ALL_DTYPES = get_all_dtypes(no_none=True)

    @pytest.mark.parametrize("val_type", ALL_DTYPES)
    @pytest.mark.parametrize("data_type", ALL_DTYPES)
    @pytest.mark.parametrize("val", [1.5, 1, 3])
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
        a = numpy.array(array, dtype=data_type)
        ia = dpnp.array(array, dtype=data_type)
        val_ = val_type(val)

        result = dpnp.power(ia, val_)
        expected = numpy.power(a, val_)
        assert_allclose(result, expected, rtol=1e-6)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_power(self, dtype):
        a = generate_random_numpy_array(10, dtype, low=0)
        b = generate_random_numpy_array(10, dtype, low=0)
        expected = numpy.power(a, b)

        ia, ib = dpnp.array(a), dpnp.array(b)
        out_dtype = numpy.int8 if dtype == numpy.bool_ else dtype
        iout = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.power(ia, ib, out=iout)

        assert result is iout
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    def test_out_overlap(self, dtype):
        size = 1 if dtype == dpnp.bool else 10
        a = numpy.arange(2 * size, dtype=dtype)
        ia = dpnp.array(a)

        dpnp.power(ia[size::], ia[::2], out=ia[:size:]),
        numpy.power(a[size::], a[::2], out=a[:size:])

        assert_dtype_allclose(ia, a)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    def test_inplace_strided_out(self, dtype):
        size = 5
        a = numpy.arange(2 * size, dtype=dtype)
        a[::3] **= 3

        ia = dpnp.arange(2 * size, dtype=dtype)
        ia[::3] **= 3

        assert_dtype_allclose(ia, a)

    @pytest.mark.parametrize("dtype1", ALL_DTYPES)
    @pytest.mark.parametrize("dtype2", ALL_DTYPES)
    def test_inplace_dtype(self, dtype1, dtype2):
        a = get_abs_array([[-7, 6, -3, 2, -1], [0, -3, 4, 5, -6]], dtype=dtype1)
        b = get_abs_array([5, 2, 0, 1, 3], dtype=dtype2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        if numpy.issubdtype(dtype1, numpy.signedinteger) and numpy.issubdtype(
            dtype2, numpy.uint64
        ):
            # For this special case, NumPy raises an error but dpnp works
            b = b.astype(numpy.int64)
            a **= b
            ia **= ib
            assert_dtype_allclose(ia, a)
        elif numpy.can_cast(dtype2, dtype1, casting="same_kind") and not (
            dtype1 == dtype2 == dpnp.bool
        ):
            a **= b
            ia **= ib
            assert_dtype_allclose(ia, a)
        else:
            with pytest.raises(TypeError):
                a **= b

            with pytest.raises(ValueError):
                ia **= ib

    @pytest.mark.parametrize("shape", [(0,), (15,), (2, 2)])
    def test_invalid_shape(self, shape):
        a, b = dpnp.arange(10), dpnp.arange(10)
        out = dpnp.empty(shape)

        with pytest.raises(ValueError):
            dpnp.power(a, b, out=out)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["scalar", "empty_tuple", "empty_list", "tuple", "list"],
    )
    def test_invalid_out(self, xp, out):
        a = xp.arange(10)
        assert_raises(TypeError, xp.power, a, 2, out)

    @pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
    def test_complex_values(self):
        a = numpy.array([0j, 1 + 1j, 0 + 2j, 1 + 2j, numpy.nan, numpy.inf])
        ia = dpnp.array(a)
        func = lambda x: x**2

        assert_dtype_allclose(func(ia), func(a))

    @pytest.mark.parametrize("val", [0, 1], ids=["0", "1"])
    @pytest.mark.parametrize("dtype", get_integer_dtypes())
    def test_integer_power_of_0_or_1(self, val, dtype):
        a = numpy.arange(10, dtype=dtype)
        ia = dpnp.array(a)
        func = lambda x: val**x

        assert_equal(func(ia), func(a))

    @pytest.mark.parametrize("dtype", get_integer_dtypes(no_unsigned=True))
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

        expected = a**b
        result = dpnp.array(a) ** dpnp.array(b)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("shape", [(), (3, 2)], ids=["()", "(3, 2)"])
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_power_scalar(self, shape, dtype):
        a = numpy.ones(shape, dtype=dtype)
        ia = dpnp.ones(shape, dtype=dtype)

        result = 4.2**ia**-1.3
        expected = 4.2**a**-1.3
        assert_allclose(result, expected, rtol=1e-6)

        result **= ia
        expected **= a
        assert_allclose(result, expected, rtol=1e-6)

    def test_alias(self):
        a = dpnp.arange(10)
        res1 = dpnp.power(a, 3)
        res2 = dpnp.pow(a, 3)

        assert_array_equal(res1, res2)


class TestRationalFunctions:
    @pytest.mark.parametrize("func", ["gcd", "lcm"])
    @pytest.mark.parametrize("dt1", get_integer_dtypes(no_unsigned=True))
    @pytest.mark.parametrize("dt2", get_integer_dtypes(no_unsigned=True))
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


class TestSubtract:
    ALL_DTYPES = get_all_dtypes(no_none=True, no_bool=True)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_add(self, dtype):
        a = generate_random_numpy_array(10, dtype)
        b = generate_random_numpy_array(10, dtype)
        expected = numpy.subtract(a, b)

        ia, ib = dpnp.array(a), dpnp.array(b)
        iout = dpnp.empty(expected.shape, dtype=dtype)
        result = dpnp.subtract(ia, ib, out=iout)

        assert result is iout
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_out_overlap(self, dtype):
        size = 1 if dtype == dpnp.bool else 15
        a = numpy.arange(2 * size, dtype=dtype)
        ia = dpnp.array(a)

        dpnp.subtract(ia[size::], ia[::2], out=ia[:size:])
        numpy.subtract(a[size::], a[::2], out=a[:size:])

        assert_dtype_allclose(ia, a)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_inplace_strides(self, dtype):
        size = 21
        a = numpy.arange(size, dtype=dtype)
        a[::3] -= 4

        ia = dpnp.arange(size, dtype=dtype)
        ia[::3] -= 4

        assert_dtype_allclose(ia, a)

    @pytest.mark.parametrize("dtype1", ALL_DTYPES)
    @pytest.mark.parametrize("dtype2", ALL_DTYPES)
    def test_inplace_dtype(self, dtype1, dtype2):
        a = get_abs_array([[-7, 6, -3, 2, -1], [0, -3, 4, 5, -6]], dtype=dtype1)
        b = get_abs_array([5, -2, 0, 1, 0], dtype=dtype2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        if numpy.issubdtype(dtype1, numpy.signedinteger) and numpy.issubdtype(
            dtype2, numpy.uint64
        ):
            # For this special case, NumPy raises an error but dpnp works
            b = b.astype(numpy.int64)
            a -= b
            ia -= ib
            assert_dtype_allclose(ia, a)
        elif numpy.can_cast(dtype2, dtype1, casting="same_kind"):
            a -= b
            ia -= ib
            assert_dtype_allclose(ia, a)
        else:
            with pytest.raises(TypeError):
                a -= b

            with pytest.raises(ValueError):
                ia -= ib

    @pytest.mark.parametrize("shape", [(0,), (15,), (2, 2)])
    def test_invalid_shape(self, shape):
        a, b = dpnp.arange(10), dpnp.arange(10)
        out = dpnp.empty(shape)

        with pytest.raises(ValueError):
            dpnp.subtract(a, b, out=out)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["scalar", "empty_tuple", "empty_list", "tuple", "list"],
    )
    def test_invalid_out(self, xp, out):
        a = xp.arange(10)
        assert_raises(TypeError, xp.subtract, a, 2, out)
