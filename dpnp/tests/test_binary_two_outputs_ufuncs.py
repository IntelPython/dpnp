import itertools

import numpy
import pytest
from numpy.testing import (
    assert_array_equal,
)

import dpnp

from .helper import (
    generate_random_numpy_array,
    get_all_dtypes,
    get_complex_dtypes,
    get_float_dtypes,
    get_integer_dtypes,
)

"""
The scope includes tests with only functions which are instances of
`DPNPBinaryTwoOutputsFunc` class.

"""


@pytest.mark.parametrize("func", ["divmod"])
class TestBinaryTwoOutputs:
    ALL_DTYPES = get_all_dtypes(no_none=True)
    ALL_DTYPES_NO_COMPLEX = get_all_dtypes(
        no_none=True, no_float16=False, no_complex=True
    )
    ALL_FLOAT_DTYPES = get_float_dtypes(no_float16=False)

    def _signs(self, dtype):
        if numpy.issubdtype(dtype, numpy.unsignedinteger):
            return (+1,)
        else:
            return (+1, -1)

    @pytest.mark.usefixtures("suppress_divide_numpy_warnings")
    @pytest.mark.parametrize("dt", ALL_DTYPES_NO_COMPLEX)
    def test_basic(self, func, dt):
        a = generate_random_numpy_array((2, 5), dtype=dt)
        b = generate_random_numpy_array((2, 5), dtype=dt)
        ia, ib = dpnp.array(a), dpnp.array(b)

        res1, res2 = getattr(dpnp, func)(ia, ib)
        exp1, exp2 = getattr(numpy, func)(a, b)
        assert_array_equal(res1, exp1)
        assert_array_equal(res2, exp2)

    @pytest.mark.parametrize("dt1", ALL_DTYPES_NO_COMPLEX)
    @pytest.mark.parametrize("dt2", ALL_DTYPES_NO_COMPLEX)
    def test_signs(self, func, dt1, dt2):
        for sign1, sign2 in itertools.product(
            self._signs(dt1), self._signs(dt2)
        ):
            a = numpy.array(sign1 * 71, dtype=dt1)
            b = numpy.array(sign2 * 19, dtype=dt2)
            ia, ib = dpnp.array(a), dpnp.array(b)

            res1, res2 = getattr(dpnp, func)(ia, ib)
            exp1, exp2 = getattr(numpy, func)(a, b)
            assert_array_equal(res1, exp1)
            assert_array_equal(res2, exp2)

    @pytest.mark.parametrize("dt", ALL_FLOAT_DTYPES)
    def test_float_exact(self, func, dt):
        # test that float results are exact for small integers
        nlst = list(range(-127, 0))
        plst = list(range(1, 128))
        dividend = nlst + [0] + plst
        divisor = nlst + plst
        arg = list(itertools.product(dividend, divisor))

        a, b = numpy.array(arg, dtype=dt).T
        ia, ib = dpnp.array(a), dpnp.array(b)

        res1, res2 = getattr(dpnp, func)(ia, ib)
        exp1, exp2 = getattr(numpy, func)(a, b)
        assert_array_equal(res1, exp1)
        assert_array_equal(res2, exp2)

    @pytest.mark.parametrize("dt1", get_float_dtypes())
    @pytest.mark.parametrize("dt2", get_float_dtypes())
    @pytest.mark.parametrize(
        "sign1, sign2", [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]
    )
    def test_float_roundoff(self, func, dt1, dt2, sign1, sign2):
        a = numpy.array(sign1 * 78 * 6e-8, dtype=dt1)
        b = numpy.array(sign2 * 6e-8, dtype=dt2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        res1, res2 = getattr(dpnp, func)(ia, ib)
        exp1, exp2 = getattr(numpy, func)(a, b)
        assert_array_equal(res1, exp1)
        assert_array_equal(res2, exp2)

    @pytest.mark.usefixtures("suppress_divide_invalid_numpy_warnings")
    @pytest.mark.parametrize("dt", ALL_FLOAT_DTYPES)
    @pytest.mark.parametrize(
        "val1", [0.0, 1.0, numpy.inf, -numpy.inf, numpy.nan]
    )
    @pytest.mark.parametrize(
        "val2", [0.0, 1.0, numpy.inf, -numpy.inf, numpy.nan]
    )
    def test_special_float_values(self, func, dt, val1, val2):
        a = numpy.array(val1, dtype=dt)
        b = numpy.array(val2, dtype=dt)
        ia, ib = dpnp.array(a), dpnp.array(b)

        res1, res2 = getattr(dpnp, func)(ia, ib)
        exp1, exp2 = getattr(numpy, func)(a, b)
        assert_array_equal(res1, exp1)
        assert_array_equal(res2, exp2)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize("dt", ALL_FLOAT_DTYPES)
    def test_float_overflow(self, func, dt):
        a = numpy.finfo(dt).tiny
        a = numpy.array(a, dtype=dt)
        ia = dpnp.array(a, dtype=dt)

        res1, res2 = getattr(dpnp, func)(4, ia)
        exp1, exp2 = getattr(numpy, func)(4, a)
        assert_array_equal(res1, exp1)
        assert_array_equal(res2, exp2)

    @pytest.mark.parametrize("dt", ALL_FLOAT_DTYPES)
    def test_out(self, func, dt):
        a = numpy.array(5.7, dtype=dt)
        ia = dpnp.array(a)

        out1 = numpy.empty((), dtype=dt)
        out2 = numpy.empty((), dtype=dt)
        iout1, iout2 = dpnp.array(out1), dpnp.array(out2)

        res1, res2 = getattr(dpnp, func)(ia, 2, iout1)
        exp1, exp2 = getattr(numpy, func)(a, 2, out1)
        assert_array_equal(res1, exp1)
        assert_array_equal(res2, exp2)
        assert res1 is iout1

        res1, res2 = getattr(dpnp, func)(ia, 2, None, iout2)
        exp1, exp2 = getattr(numpy, func)(a, 2, None, out2)
        assert_array_equal(res1, exp1)
        assert_array_equal(res2, exp2)
        assert res2 is iout2

        res1, res2 = getattr(dpnp, func)(ia, 2, iout1, iout2)
        exp1, exp2 = getattr(numpy, func)(a, 2, out1, out2)
        assert_array_equal(res1, exp1)
        assert_array_equal(res2, exp2)
        assert res1 is iout1
        assert res2 is iout2

    @pytest.mark.parametrize("dt1", ALL_DTYPES_NO_COMPLEX)
    @pytest.mark.parametrize("dt2", ALL_DTYPES_NO_COMPLEX)
    @pytest.mark.parametrize("out1_dt", ALL_DTYPES)
    @pytest.mark.parametrize("out2_dt", ALL_DTYPES)
    def test_2out_all_dtypes(self, func, dt1, dt2, out1_dt, out2_dt):
        a = numpy.ones((3, 1), dtype=dt1)
        b = numpy.ones((3, 4), dtype=dt2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        out1 = numpy.zeros_like(b, dtype=out1_dt)
        out2 = numpy.zeros_like(b, dtype=out2_dt)
        iout1, iout2 = dpnp.array(out1), dpnp.array(out2)

        try:
            res1, res2 = getattr(dpnp, func)(ia, ib, out=(iout1, iout2))
        except TypeError:
            # expect numpy to fail with the same reason
            with pytest.raises(TypeError):
                _ = getattr(numpy, func)(a, b, out=(out1, out2))
        else:
            exp1, exp2 = getattr(numpy, func)(a, b, out=(out1, out2))
            assert_array_equal(res1, exp1)
            assert_array_equal(res2, exp2)
            assert res1 is iout1
            assert res2 is iout2

    @pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
    @pytest.mark.parametrize("stride", [-4, -2, -1, 1, 2, 4])
    @pytest.mark.parametrize("dt", ALL_FLOAT_DTYPES)
    def test_strides_out(self, func, stride, dt):
        a = numpy.array(
            [numpy.nan, numpy.nan, numpy.inf, -numpy.inf, 0.0, -0.0, 1.0, -1.0],
            dtype=dt,
        )
        ia = dpnp.array(a)

        out1 = numpy.ones_like(a, dtype=dt)
        out2 = 2 * numpy.ones_like(a, dtype=dt)
        iout_mant, iout_exp = dpnp.array(out1), dpnp.array(out2)

        res1, res2 = getattr(dpnp, func)(
            ia[::stride], 2, out=(iout_mant[::stride], iout_exp[::stride])
        )
        exp1, exp2 = getattr(numpy, func)(
            a[::stride], 2, out=(out1[::stride], out2[::stride])
        )
        assert_array_equal(res1, exp1)
        assert_array_equal(res2, exp2)

        assert_array_equal(iout_mant, out1)
        assert_array_equal(iout_exp, out2)

    @pytest.mark.parametrize("dt", ALL_FLOAT_DTYPES)
    def test_out1_overlap(self, func, dt):
        size = 15
        a = numpy.ones(2 * size, dtype=dt)
        ia = dpnp.array(a)

        # out1 overlaps memory of input array
        _ = getattr(dpnp, func)(ia[size::], 1, ia[::2])
        _ = getattr(numpy, func)(a[size::], 1, a[::2])
        assert_array_equal(ia, a)

    @pytest.mark.parametrize("dt", ALL_FLOAT_DTYPES)
    def test_empty(self, func, dt):
        a = numpy.empty(0, dtype=dt)
        ia = dpnp.array(a)

        res1, res2 = getattr(dpnp, func)(ia, ia)
        exp1, exp2 = getattr(numpy, func)(a, a)
        assert_array_equal(res1, exp1, strict=True)
        assert_array_equal(res2, exp2, strict=True)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize("dt", get_complex_dtypes())
    def test_complex_dtype(self, func, xp, dt):
        a = xp.array(
            [0.9 + 1j, -0.1 + 1j, 0.9 + 0.5 * 1j, 0.9 + 2.0 * 1j], dtype=dt
        )
        with pytest.raises((TypeError, ValueError)):
            _ = getattr(xp, func)(a, 7)


class TestDivmod:
    @pytest.mark.usefixtures("suppress_divide_numpy_warnings")
    @pytest.mark.parametrize("dt", get_integer_dtypes())
    def test_int_zero(self, dt):
        a = numpy.array(0, dtype=dt)
        ia = dpnp.array(a)

        res1, res2 = dpnp.divmod(ia, 0)
        exp1, exp2 = numpy.divmod(a, 0)
        assert_array_equal(res1, exp1)
        assert_array_equal(res2, exp2)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize("dt", get_integer_dtypes(no_unsigned=True))
    def test_min_int(self, dt):
        a = numpy.array(numpy.iinfo(dt).min, dtype=dt)
        ia = dpnp.array(a)

        res1, res2 = dpnp.divmod(ia, -1)
        exp1, exp2 = numpy.divmod(a, -1)
        assert_array_equal(res1, exp1)
        assert_array_equal(res2, exp2)

    @pytest.mark.parametrize("dt", get_integer_dtypes(no_unsigned=True))
    def test_special_int(self, dt):
        # a and b have different sign and mod != 0
        a, b = numpy.array(-1, dtype=dt), numpy.array(3, dtype=dt)
        ia, ib = dpnp.array(a), dpnp.array(b)

        res1, res2 = dpnp.divmod(ia, ib)
        exp1, exp2 = numpy.divmod(a, b)
        assert_array_equal(res1, exp1)
        assert_array_equal(res2, exp2)
