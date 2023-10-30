import dpctl.tensor as dpt
import numpy
import pytest
from numpy.testing import assert_allclose

import dpnp

from .helper import get_all_dtypes


@pytest.mark.parametrize("func", ["argmax", "argmin"])
@pytest.mark.parametrize("axis", [None, 0, 1, -1, 2, -2])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
def test_argmax_argmin(func, axis, keepdims, dtype):
    a = numpy.arange(768, dtype=dtype).reshape((4, 4, 6, 8))
    ia = dpnp.array(a)

    np_res = getattr(numpy, func)(a, axis=axis, keepdims=keepdims)
    dpnp_res = getattr(dpnp, func)(ia, axis=axis, keepdims=keepdims)

    assert dpnp_res.shape == np_res.shape
    assert_allclose(dpnp_res, np_res)


@pytest.mark.parametrize("func", ["argmax", "argmin"])
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_argmax_argmin_bool(func, axis, keepdims):
    a = numpy.arange(2, dtype=dpnp.bool)
    a = numpy.tile(a, (2, 2))
    ia = dpnp.array(a)

    np_res = getattr(numpy, func)(a, axis=axis, keepdims=keepdims)
    dpnp_res = getattr(dpnp, func)(ia, axis=axis, keepdims=keepdims)

    assert dpnp_res.shape == np_res.shape
    assert_allclose(dpnp_res, np_res)


@pytest.mark.parametrize("func", ["argmax", "argmin"])
def test_argmax_argmin_out(func):
    a = numpy.arange(6).reshape((2, 3))
    ia = dpnp.array(a)

    np_res = getattr(numpy, func)(a, axis=0)
    dpnp_res = dpnp.array(numpy.empty_like(np_res))
    getattr(dpnp, func)(ia, axis=0, out=dpnp_res)
    assert_allclose(dpnp_res, np_res)

    dpnp_res = dpt.asarray(numpy.empty_like(np_res))
    getattr(dpnp, func)(ia, axis=0, out=dpnp_res)
    assert_allclose(dpnp_res, np_res)

    dpnp_res = numpy.empty_like(np_res)
    with pytest.raises(TypeError):
        getattr(dpnp, func)(ia, axis=0, out=dpnp_res)

    dpnp_res = dpnp.array(numpy.empty((2, 3)))
    with pytest.raises(ValueError):
        getattr(dpnp, func)(ia, axis=0, out=dpnp_res)
