import numpy
import pytest
from numpy.testing import assert_allclose

import dpnp

from .helper import get_all_dtypes


@pytest.mark.parametrize("axis", [None, 0, 1, -1, 2, -2])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
def test_argmax_argmin(axis, keepdims, dtype):
    a = numpy.arange(768, dtype=dtype).reshape((4, 4, 6, 8))
    ia = dpnp.array(a)

    np_res = numpy.argmax(a, axis=axis, keepdims=keepdims)
    dpnp_res = dpnp.argmax(ia, axis=axis, keepdims=keepdims)

    assert dpnp_res.shape == np_res.shape
    assert_allclose(dpnp_res, np_res)

    np_res = numpy.argmin(a, axis=axis, keepdims=keepdims)
    dpnp_res = dpnp.argmin(ia, axis=axis, keepdims=keepdims)

    assert dpnp_res.shape == np_res.shape
    assert_allclose(dpnp_res, np_res)


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_argmax_argmin_bool(axis, keepdims):
    a = numpy.arange(2, dtype=dpnp.bool)
    a = numpy.tile(a, (2, 2))
    ia = dpnp.array(a)

    np_res = numpy.argmax(a, axis=axis, keepdims=keepdims)
    dpnp_res = dpnp.argmax(ia, axis=axis, keepdims=keepdims)

    assert dpnp_res.shape == np_res.shape
    assert_allclose(dpnp_res, np_res)

    np_res = numpy.argmin(a, axis=axis, keepdims=keepdims)
    dpnp_res = dpnp.argmin(ia, axis=axis, keepdims=keepdims)

    assert dpnp_res.shape == np_res.shape
    assert_allclose(dpnp_res, np_res)


@pytest.mark.parametrize("axis", [None, 0, 1, -1, 2, -2])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
def test_argmax_argmin_out(axis, keepdims, dtype):
    a = numpy.arange(768, dtype=dtype).reshape((4, 4, 6, 8))
    ia = dpnp.array(a)

    np_res = numpy.argmax(a, axis=axis, keepdims=keepdims)
    dpnp_res = dpnp.array(numpy.empty_like(np_res))
    dpnp.argmax(ia, axis=axis, keepdims=keepdims, out=dpnp_res)

    assert dpnp_res.shape == np_res.shape
    assert_allclose(dpnp_res, np_res)

    np_res = numpy.argmin(a, axis=axis, keepdims=keepdims)
    dpnp_res = dpnp.array(numpy.empty_like(np_res))
    dpnp.argmin(ia, axis=axis, keepdims=keepdims, out=dpnp_res)

    assert dpnp_res.shape == np_res.shape
    assert_allclose(dpnp_res, np_res)
