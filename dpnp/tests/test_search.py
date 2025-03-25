import dpctl.tensor as dpt
import numpy
import pytest
from numpy.testing import assert_array_equal, assert_equal, assert_raises

import dpnp

from .helper import (
    generate_random_numpy_array,
    get_all_dtypes,
)


@pytest.mark.parametrize("func", ["argmax", "argmin"])
class TestArgmaxArgmin:
    @pytest.mark.parametrize("axis", [None, 0, 1, -1, 2, -2])
    @pytest.mark.parametrize("keepdims", [False, True])
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_basic(self, func, axis, keepdims, dtype):
        a = generate_random_numpy_array((4, 4, 6, 8), dtype=dtype)
        ia = dpnp.array(a)

        expected = getattr(numpy, func)(a, axis=axis, keepdims=keepdims)
        result = getattr(dpnp, func)(ia, axis=axis, keepdims=keepdims)
        assert_array_equal(result, expected)

    def test_out(self, func):
        a = generate_random_numpy_array((2, 2, 3), dtype=numpy.float32)
        ia = dpnp.array(a)

        # out is dpnp_array
        expected = getattr(numpy, func)(a, axis=0)
        dpnp_out = dpnp.empty(expected.shape, dtype=expected.dtype)
        result = getattr(dpnp, func)(ia, axis=0, out=dpnp_out)
        assert dpnp_out is result
        assert_array_equal(result, expected)

        # out is usm_ndarray
        dpt_out = dpt.empty(expected.shape, dtype=expected.dtype)
        result = getattr(dpnp, func)(ia, axis=0, out=dpt_out)
        assert dpt_out is result.get_array()
        assert_array_equal(result, expected)

        # out is a numpy array -> TypeError
        result = numpy.empty_like(expected)
        with pytest.raises(TypeError):
            getattr(dpnp, func)(ia, axis=0, out=result)

        # out shape is incorrect -> ValueError
        result = dpnp.array(numpy.zeros((2, 2)), dtype=dpnp.intp)
        with pytest.raises(ValueError):
            getattr(dpnp, func)(ia, axis=0, out=result)

    @pytest.mark.parametrize("in_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("out_dt", get_all_dtypes(no_none=True))
    def test_out_dtype(self, func, in_dt, out_dt):
        a = generate_random_numpy_array((2, 2, 3), dtype=in_dt)
        out = numpy.zeros_like(a, shape=(2, 3), dtype=out_dt)
        ia, iout = dpnp.array(a), dpnp.array(out)

        if numpy.can_cast(out.dtype, numpy.intp, casting="safe"):
            result = getattr(dpnp, func)(ia, out=iout, axis=1)
            expected = getattr(numpy, func)(a, out=out, axis=1)
            assert_array_equal(result, expected)
            assert result is iout
        else:
            assert_raises(TypeError, getattr(numpy, func), a, out=out, axis=1)
            assert_raises(TypeError, getattr(dpnp, func), ia, out=iout, axis=1)

    @pytest.mark.parametrize("axis", [None, 0, 1, -1])
    @pytest.mark.parametrize("keepdims", [False, True])
    def test_ndarray(self, func, axis, keepdims):
        a = generate_random_numpy_array((4, 6, 8), dtype=numpy.float32)
        ia = dpnp.array(a)

        expected = getattr(a, func)(axis=axis, keepdims=keepdims)
        result = getattr(ia, func)(axis=axis, keepdims=keepdims)
        assert_array_equal(result, expected)


class TestArgwhere:
    @pytest.mark.parametrize("dt", get_all_dtypes(no_none=True))
    def test_basic(self, dt):
        a = numpy.array([4, 0, 2, 1, 3], dtype=dt)
        ia = dpnp.array(a)

        result = dpnp.argwhere(ia)
        expected = numpy.argwhere(a)
        assert_equal(result, expected)

    @pytest.mark.parametrize("ndim", [0, 1, 2])
    def test_ndim(self, ndim):
        # get an nd array with multiple elements in every dimension
        a = numpy.empty((2,) * ndim)

        # none
        a[...] = False
        ia = dpnp.array(a)

        result = dpnp.argwhere(ia)
        expected = numpy.argwhere(a)
        assert_equal(result, expected)

        # only one
        a[...] = False
        a.flat[0] = True
        ia = dpnp.array(a)

        result = dpnp.argwhere(ia)
        expected = numpy.argwhere(a)
        assert_equal(result, expected)

        # all but one
        a[...] = True
        a.flat[0] = False
        ia = dpnp.array(a)

        result = dpnp.argwhere(ia)
        expected = numpy.argwhere(a)
        assert_equal(result, expected)

        # all
        a[...] = True
        ia = dpnp.array(a)

        result = dpnp.argwhere(ia)
        expected = numpy.argwhere(a)
        assert_equal(result, expected)

    def test_2d(self):
        a = numpy.arange(6).reshape((2, 3))
        ia = dpnp.array(a)

        result = dpnp.argwhere(ia > 1)
        expected = numpy.argwhere(a > 1)
        assert_array_equal(result, expected)


class TestWhere:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_basic(self, dtype):
        a = numpy.ones(53, dtype=bool)
        ia = dpnp.array(a)

        expected = numpy.where(a, dtype(0), dtype(1))
        result = dpnp.where(ia, dtype(0), dtype(1))
        assert_array_equal(result, expected)

        expected = numpy.where(~a, dtype(0), dtype(1))
        result = dpnp.where(~ia, dtype(0), dtype(1))
        assert_array_equal(result, expected)

        d = numpy.ones_like(a).astype(dtype)
        e = numpy.zeros_like(d)
        a[7] = False

        ia[7] = False
        id = dpnp.array(d)
        ie = dpnp.array(e)

        expected = numpy.where(a, e, e)
        result = dpnp.where(ia, ie, ie)
        assert_array_equal(result, expected)

        expected = numpy.where(a, d, e)
        result = dpnp.where(ia, id, ie)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize(
        "slice_a, slice_d, slice_e",
        [
            pytest.param(
                slice(None, None, None),
                slice(None, None, None),
                slice(0, 1, None),
            ),
            pytest.param(
                slice(None, None, None),
                slice(0, 1, None),
                slice(None, None, None),
            ),
            pytest.param(
                slice(None, None, 2), slice(None, None, 2), slice(None, None, 2)
            ),
            pytest.param(
                slice(1, None, 2), slice(1, None, 2), slice(1, None, 2)
            ),
            pytest.param(
                slice(None, None, 3), slice(None, None, 3), slice(None, None, 3)
            ),
            pytest.param(
                slice(1, None, 3), slice(1, None, 3), slice(1, None, 3)
            ),
            pytest.param(
                slice(None, None, -2),
                slice(None, None, -2),
                slice(None, None, -2),
            ),
            pytest.param(
                slice(None, None, -3),
                slice(None, None, -3),
                slice(None, None, -3),
            ),
            pytest.param(
                slice(1, None, -3), slice(1, None, -3), slice(1, None, -3)
            ),
        ],
    )
    def test_strided(self, dtype, slice_a, slice_d, slice_e):
        a = numpy.ones(53, dtype=bool)
        a[7] = False
        d = numpy.ones_like(a).astype(dtype)
        e = numpy.zeros_like(d)

        ia = dpnp.array(a)
        id = dpnp.array(d)
        ie = dpnp.array(e)

        expected = numpy.where(a[slice_a], d[slice_d], e[slice_e])
        result = dpnp.where(ia[slice_a], id[slice_d], ie[slice_e])
        assert_array_equal(result, expected)

    def test_zero_sized(self):
        a = numpy.array([], dtype=bool).reshape(0, 3)
        b = numpy.array([], dtype=numpy.float32).reshape(0, 3)

        ia = dpnp.array(a)
        ib = dpnp.array(b)

        expected = numpy.where(a, 0, b)
        result = dpnp.where(ia, 0, ib)
        assert_array_equal(result, expected)

    def test_ndim(self):
        a = numpy.zeros((2, 25))
        b = numpy.ones((2, 25))
        c = numpy.array([True, False])

        ia = dpnp.array(a)
        ib = dpnp.array(b)
        ic = dpnp.array(c)

        expected = numpy.where(c[:, numpy.newaxis], a, b)
        result = dpnp.where(ic[:, dpnp.newaxis], ia, ib)
        assert_array_equal(result, expected)

        expected = numpy.where(c, a.T, b.T)
        result = dpnp.where(ic, ia.T, ib.T)
        assert_array_equal(result, expected)

    def test_dtype_mix(self):
        a = numpy.uint32(1)
        b = numpy.array(
            [5.0, 0.0, 3.0, 2.0, -1.0, -4.0, 0.0, -10.0, 10.0, 1.0, 0.0, 3.0],
            dtype=numpy.float32,
        )
        c = numpy.array(
            [
                False,
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
            ]
        )

        ia = dpnp.array(a)
        ib = dpnp.array(b)
        ic = dpnp.array(c)

        expected = numpy.where(c, a, b)
        result = dpnp.where(ic, ia, ib)
        assert_array_equal(result, expected)

        b = b.astype(numpy.int64)
        ib = dpnp.array(b)

        expected = numpy.where(c, a, b)
        result = dpnp.where(ic, ia, ib)
        assert_array_equal(result, expected)

        # non bool mask
        c = c.astype(int)
        c[c != 0] = 34242324
        ic = dpnp.array(c)

        expected = numpy.where(c, a, b)
        result = dpnp.where(ic, ia, ib)
        assert_array_equal(result, expected)

        # invert
        tmpmask = c != 0
        c[c == 0] = 41247212
        c[tmpmask] = 0
        ic = dpnp.array(c)

        expected = numpy.where(c, a, b)
        result = dpnp.where(ic, ia, ib)
        assert_array_equal(result, expected)

    def test_error(self):
        c = dpnp.array([True, True])
        a = dpnp.ones((4, 5))
        b = dpnp.ones((5, 5))
        assert_raises(ValueError, dpnp.where, c, a, a)
        assert_raises(ValueError, dpnp.where, c[0], a, b)

    def test_empty_result(self):
        a = numpy.zeros((1, 1))
        ia = dpnp.array(a)

        expected = numpy.vstack(numpy.where(a == 99.0))
        result = dpnp.vstack(dpnp.where(ia == 99.0))
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("x_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("y_dt", get_all_dtypes(no_none=True))
    def test_out(self, x_dt, y_dt):
        cond = numpy.random.rand(50).reshape((2, 25)).astype(dtype="?")
        x = numpy.random.rand(50).reshape((2, 25)).astype(dtype=x_dt)
        y = numpy.random.rand(50).reshape((2, 25)).astype(dtype=y_dt)
        out = numpy.zeros_like(cond, dtype=numpy.result_type(x, y))

        icond = dpnp.array(cond)
        ix = dpnp.array(x)
        iy = dpnp.array(y)
        iout = dpnp.array(out)

        result = dpnp.where(icond, ix, iy, out=iout)
        expected = numpy.where(cond, x, y)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("order", ["C", "F", "A", "K", None])
    def test_order(self, order):
        cond = numpy.array([True, False, True])
        x = numpy.array([2, 5, 3], order="F")
        y = numpy.array([-2, -5, -3], order="C")

        icond = dpnp.array(cond)
        ix = dpnp.array(x)
        iy = dpnp.array(y)

        result = dpnp.where(icond, ix, iy, order=order)
        expected = numpy.where(cond, x, y)
        assert_array_equal(result, expected)

        if order == "F":
            assert result.flags.f_contiguous
        else:
            assert result.flags.c_contiguous
