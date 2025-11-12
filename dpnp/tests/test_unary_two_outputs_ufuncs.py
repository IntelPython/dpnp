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
    is_win_platform,
)

"""
The scope includes tests with only functions which are instances of
`DPNPUnaryTwoOutputsFunc` class.

"""


@pytest.mark.parametrize("func", ["frexp", "modf"])
class TestUnaryTwoOutputs:
    ALL_DTYPES = get_all_dtypes(no_none=True)
    ALL_DTYPES_NO_COMPLEX = get_all_dtypes(
        no_none=True, no_float16=False, no_complex=True
    )
    ALL_FLOAT_DTYPES = get_float_dtypes(no_float16=False)

    def _get_out_dtypes(self, func, dt):
        if func == "frexp":
            return (dt, numpy.int32)
        return (dt, dt)

    @pytest.mark.parametrize("dt", ALL_DTYPES_NO_COMPLEX)
    def test_basic(self, func, dt):
        a = generate_random_numpy_array((2, 5), dtype=dt)
        ia = dpnp.array(a)

        res1, res2 = getattr(dpnp, func)(ia)
        exp1, exp2 = getattr(numpy, func)(a)
        assert_array_equal(res1, exp1)
        assert_array_equal(res2, exp2)

    @pytest.mark.parametrize("dt", ALL_FLOAT_DTYPES)
    def test_out(self, func, dt):
        a = numpy.array(5.7, dtype=dt)
        ia = dpnp.array(a)

        dt1, dt2 = self._get_out_dtypes(func, dt)
        out1 = numpy.empty((), dtype=dt1)
        out2 = numpy.empty((), dtype=dt2)
        iout1, iout2 = dpnp.array(out1), dpnp.array(out2)

        res1, res2 = getattr(dpnp, func)(ia, iout1)
        exp1, exp2 = getattr(numpy, func)(a, out1)
        assert_array_equal(res1, exp1)
        assert_array_equal(res2, exp2)
        assert res1 is iout1

        res1, res2 = getattr(dpnp, func)(ia, None, iout2)
        exp1, exp2 = getattr(numpy, func)(a, None, out2)
        assert_array_equal(res1, exp1)
        assert_array_equal(res2, exp2)
        assert res2 is iout2

        res1, res2 = getattr(dpnp, func)(ia, iout1, iout2)
        exp1, exp2 = getattr(numpy, func)(a, out1, out2)
        assert_array_equal(res1, exp1)
        assert_array_equal(res2, exp2)
        assert res1 is iout1
        assert res2 is iout2

    @pytest.mark.parametrize("dt", ALL_DTYPES_NO_COMPLEX)
    @pytest.mark.parametrize("out1_dt", ALL_DTYPES)
    @pytest.mark.parametrize("out2_dt", ALL_DTYPES)
    def test_out_all_dtypes(self, func, dt, out1_dt, out2_dt):
        a = numpy.ones(9, dtype=dt)
        ia = dpnp.array(a)

        out1 = numpy.zeros(9, dtype=out1_dt)
        out2 = numpy.zeros(9, dtype=out2_dt)
        iout1, iout2 = dpnp.array(out1), dpnp.array(out2)

        try:
            res1, res2 = getattr(dpnp, func)(ia, out=(iout1, iout2))
        except TypeError:
            # expect numpy to fail with the same reason
            with pytest.raises(TypeError):
                _ = getattr(numpy, func)(a, out=(out1, out2))
        else:
            exp1, exp2 = getattr(numpy, func)(a, out=(out1, out2))
            assert_array_equal(res1, exp1)
            assert_array_equal(res2, exp2)
            assert res1 is iout1
            assert res2 is iout2

    @pytest.mark.parametrize("stride", [-4, -2, -1, 1, 2, 4])
    @pytest.mark.parametrize("dt", ALL_FLOAT_DTYPES)
    def test_strides_out(self, func, stride, dt):
        if func == "frexp" and is_win_platform():
            pytest.skip(
                "numpy.frexp gives different answers for NAN/INF on Windows and Linux"
            )

        a = numpy.array(
            [numpy.nan, numpy.nan, numpy.inf, -numpy.inf, 0.0, -0.0, 1.0, -1.0],
            dtype=dt,
        )
        ia = dpnp.array(a)

        dt1, dt2 = self._get_out_dtypes(func, dt)
        out_mant = numpy.ones_like(a, dtype=dt1)
        out_exp = 2 * numpy.ones_like(a, dtype=dt2)
        iout_mant, iout_exp = dpnp.array(out_mant), dpnp.array(out_exp)

        res1, res2 = getattr(dpnp, func)(
            ia[::stride], out=(iout_mant[::stride], iout_exp[::stride])
        )
        exp1, exp2 = getattr(numpy, func)(
            a[::stride], out=(out_mant[::stride], out_exp[::stride])
        )
        assert_array_equal(res1, exp1)
        assert_array_equal(res2, exp2)

        assert_array_equal(iout_mant, out_mant)
        assert_array_equal(iout_exp, out_exp)

    @pytest.mark.parametrize("dt", ALL_FLOAT_DTYPES)
    def test_out1_overlap(self, func, dt):
        size = 15
        a = numpy.ones(2 * size, dtype=dt)
        ia = dpnp.array(a)

        # out1 overlaps memory of input array
        _ = getattr(dpnp, func)(ia[size::], ia[::2])
        _ = getattr(numpy, func)(a[size::], a[::2])
        assert_array_equal(ia, a)

    @pytest.mark.parametrize("dt", ALL_FLOAT_DTYPES)
    def test_empty(self, func, dt):
        a = numpy.empty(0, dtype=dt)
        ia = dpnp.array(a)

        res1, res2 = getattr(dpnp, func)(ia)
        exp1, exp2 = getattr(numpy, func)(a)
        assert_array_equal(res1, exp1, strict=True)
        assert_array_equal(res2, exp2, strict=True)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize("dt", get_complex_dtypes())
    def test_complex_dtype(self, func, xp, dt):
        a = xp.array([-2, 5, 1, 4, 3], dtype=dt)
        with pytest.raises((TypeError, ValueError)):
            _ = getattr(xp, func)(a)
