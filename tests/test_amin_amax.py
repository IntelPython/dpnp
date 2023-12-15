import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import dpnp

from .helper import get_all_dtypes


@pytest.mark.parametrize("func", ["amax", "amin"])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("dtype", get_all_dtypes())
def test_amax_amin(func, keepdims, dtype):
    a = numpy.array(
        [
            [[-2.0, 3.0], [9.1, 0.2]],
            [[-2.0, 5.0], [-2, -1.2]],
            [[1.0, -2.0], [5.0, -1.1]],
        ],
        dtype=dtype,
    )
    ia = dpnp.array(a)

    for axis in range(len(a)):
        result = getattr(dpnp, func)(ia, axis=axis, keepdims=keepdims)
        expected = getattr(numpy, func)(a, axis=axis, keepdims=keepdims)
        assert_allclose(expected, result)


def _get_min_max_input(type, shape):
    size = 1
    for i in range(len(shape)):
        size *= shape[i]

    a = numpy.arange(size, dtype=type)
    a[int(size / 2)] = size * size
    a[int(size / 3)] = -(size * size)

    return a.reshape(shape)


@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
@pytest.mark.parametrize(
    "shape", [(4,), (2, 3), (4, 5, 6)], ids=["(4,)", "(2,3)", "(4,5,6)"]
)
def test_amax_diff_shape(dtype, shape):
    a = _get_min_max_input(dtype, shape)

    ia = dpnp.array(a)

    np_res = numpy.amax(a)
    dpnp_res = dpnp.amax(ia)
    assert_array_equal(dpnp_res, np_res)

    np_res = a.max()
    dpnp_res = ia.max()
    numpy.testing.assert_array_equal(dpnp_res, np_res)


@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
@pytest.mark.parametrize(
    "shape", [(4,), (2, 3), (4, 5, 6)], ids=["(4,)", "(2,3)", "(4,5,6)"]
)
def test_amin_diff_shape(dtype, shape):
    a = _get_min_max_input(dtype, shape)

    ia = dpnp.array(a)

    np_res = numpy.amin(a)
    dpnp_res = dpnp.amin(ia)
    assert_array_equal(dpnp_res, np_res)

    np_res = a.min()
    dpnp_res = ia.min()
    assert_array_equal(dpnp_res, np_res)
