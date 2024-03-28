import dpctl.tensor as dpt
import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal, assert_raises

import dpnp

from .helper import assert_dtype_allclose, get_all_dtypes


@pytest.mark.parametrize("func", ["argmax", "argmin", "nanargmin", "nanargmax"])
@pytest.mark.parametrize("axis", [None, 0, 1, -1, 2, -2])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
def test_argmax_argmin(func, axis, keepdims, dtype):
    a = numpy.arange(768, dtype=dtype).reshape((4, 4, 6, 8))
    if func in ["nanargmin", "nanargmax"] and dpnp.issubdtype(
        a.dtype, dpnp.inexact
    ):
        a[2:3, 2, 3:4, 4] = numpy.nan
    ia = dpnp.array(a)

    np_res = getattr(numpy, func)(a, axis=axis, keepdims=keepdims)
    dpnp_res = getattr(dpnp, func)(ia, axis=axis, keepdims=keepdims)
    assert_dtype_allclose(dpnp_res, np_res)


@pytest.mark.parametrize("func", ["argmax", "argmin"])
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_argmax_argmin_bool(func, axis, keepdims):
    a = numpy.arange(2, dtype=numpy.bool_)
    a = numpy.tile(a, (2, 2))
    ia = dpnp.array(a)

    np_res = getattr(numpy, func)(a, axis=axis, keepdims=keepdims)
    dpnp_res = getattr(dpnp, func)(ia, axis=axis, keepdims=keepdims)
    assert_dtype_allclose(dpnp_res, np_res)


@pytest.mark.parametrize("func", ["argmax", "argmin", "nanargmin", "nanargmax"])
def test_argmax_argmin_out(func):
    a = numpy.arange(12, dtype=numpy.float32).reshape((2, 2, 3))
    if func in ["nanargmin", "nanargmax"]:
        a[1, 0, 2] = numpy.nan
    ia = dpnp.array(a)

    # out is dpnp_array
    np_res = getattr(numpy, func)(a, axis=0)
    dpnp_out = dpnp.empty(np_res.shape, dtype=np_res.dtype)
    dpnp_res = getattr(dpnp, func)(ia, axis=0, out=dpnp_out)
    assert dpnp_out is dpnp_res
    assert_allclose(dpnp_res, np_res)

    # out is usm_ndarray
    dpt_out = dpt.empty(np_res.shape, dtype=np_res.dtype)
    dpnp_res = getattr(dpnp, func)(ia, axis=0, out=dpt_out)
    assert dpt_out is dpnp_res.get_array()
    assert_allclose(dpnp_res, np_res)

    # out is a numpy array -> TypeError
    dpnp_res = numpy.empty_like(np_res)
    with pytest.raises(TypeError):
        getattr(dpnp, func)(ia, axis=0, out=dpnp_res)

    # out shape is incorrect -> ValueError
    dpnp_res = dpnp.array(numpy.empty((2, 2)))
    with pytest.raises(ValueError):
        getattr(dpnp, func)(ia, axis=0, out=dpnp_res)


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_ndarray_argmax_argmin(axis, keepdims):
    a = numpy.arange(192, dtype=numpy.float32).reshape((4, 6, 8))
    ia = dpnp.array(a)

    np_res = a.argmax(axis=axis, keepdims=keepdims)
    dpnp_res = ia.argmax(axis=axis, keepdims=keepdims)
    assert_dtype_allclose(dpnp_res, np_res)

    np_res = a.argmin(axis=axis, keepdims=keepdims)
    dpnp_res = ia.argmin(axis=axis, keepdims=keepdims)
    assert_dtype_allclose(dpnp_res, np_res)


@pytest.mark.parametrize("func", ["nanargmin", "nanargmax"])
def test_nanargmax_nanargmin_error(func):
    ia = dpnp.arange(12, dtype=dpnp.float32).reshape((2, 2, 3))
    ia[:, :, 2] = dpnp.nan

    # All-NaN slice encountered -> ValueError
    with pytest.raises(ValueError):
        getattr(dpnp, func)(ia, axis=0)


class TestWhere:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_basic(self, dtype):
        a = numpy.ones(53, dtype=bool)
        ia = dpnp.array(a)

        np_res = numpy.where(a, dtype(0), dtype(1))
        dpnp_res = dpnp.where(ia, dtype(0), dtype(1))
        assert_array_equal(np_res, dpnp_res)

        np_res = numpy.where(~a, dtype(0), dtype(1))
        dpnp_res = dpnp.where(~ia, dtype(0), dtype(1))
        assert_array_equal(np_res, dpnp_res)

        d = numpy.ones_like(a).astype(dtype)
        e = numpy.zeros_like(d)
        a[7] = False

        ia[7] = False
        id = dpnp.array(d)
        ie = dpnp.array(e)

        np_res = numpy.where(a, e, e)
        dpnp_res = dpnp.where(ia, ie, ie)
        assert_array_equal(np_res, dpnp_res)

        np_res = numpy.where(a, d, e)
        dpnp_res = dpnp.where(ia, id, ie)
        assert_array_equal(np_res, dpnp_res)

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

        np_res = numpy.where(a[slice_a], d[slice_d], e[slice_e])
        dpnp_res = dpnp.where(ia[slice_a], id[slice_d], ie[slice_e])
        assert_array_equal(np_res, dpnp_res)

    def test_zero_sized(self):
        a = numpy.array([], dtype=bool).reshape(0, 3)
        b = numpy.array([], dtype=numpy.float32).reshape(0, 3)

        ia = dpnp.array(a)
        ib = dpnp.array(b)

        np_res = numpy.where(a, 0, b)
        dpnp_res = dpnp.where(ia, 0, ib)
        assert_array_equal(np_res, dpnp_res)

    def test_ndim(self):
        a = numpy.zeros((2, 25))
        b = numpy.ones((2, 25))
        c = numpy.array([True, False])

        ia = dpnp.array(a)
        ib = dpnp.array(b)
        ic = dpnp.array(c)

        np_res = numpy.where(c[:, numpy.newaxis], a, b)
        dpnp_res = dpnp.where(ic[:, dpnp.newaxis], ia, ib)
        assert_array_equal(np_res, dpnp_res)

        np_res = numpy.where(c, a.T, b.T)
        dpnp_res = numpy.where(ic, ia.T, ib.T)
        assert_array_equal(np_res, dpnp_res)

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

        np_res = numpy.where(c, a, b)
        dpnp_res = dpnp.where(ic, ia, ib)
        assert_array_equal(np_res, dpnp_res)

        b = b.astype(numpy.int64)
        ib = dpnp.array(b)

        np_res = numpy.where(c, a, b)
        dpnp_res = dpnp.where(ic, ia, ib)
        assert_array_equal(np_res, dpnp_res)

        # non bool mask
        c = c.astype(int)
        c[c != 0] = 34242324
        ic = dpnp.array(c)

        np_res = numpy.where(c, a, b)
        dpnp_res = dpnp.where(ic, ia, ib)
        assert_array_equal(np_res, dpnp_res)

        # invert
        tmpmask = c != 0
        c[c == 0] = 41247212
        c[tmpmask] = 0
        ic = dpnp.array(c)

        np_res = numpy.where(c, a, b)
        dpnp_res = dpnp.where(ic, ia, ib)
        assert_array_equal(np_res, dpnp_res)

    def test_error(self):
        c = dpnp.array([True, True])
        a = dpnp.ones((4, 5))
        b = dpnp.ones((5, 5))
        assert_raises(ValueError, dpnp.where, c, a, a)
        assert_raises(ValueError, dpnp.where, c[0], a, b)

    def test_empty_result(self):
        a = numpy.zeros((1, 1))
        ia = dpnp.array(a)

        np_res = numpy.vstack(numpy.where(a == 99.0))
        dpnp_res = dpnp.vstack(dpnp.where(ia == 99.0))
        assert_array_equal(np_res, dpnp_res)
