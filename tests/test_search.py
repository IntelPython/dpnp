import dpctl.tensor as dpt
import numpy
import pytest
from numpy.testing import assert_allclose

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
