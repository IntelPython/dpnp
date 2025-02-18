import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import dpnp

from .helper import get_abs_array, get_all_dtypes


@pytest.mark.parametrize("func", ["amax", "amin"])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
def test_amax_amin(func, keepdims, dtype):
    a = [
        [[-2.0, 3.0], [9.1, 0.2]],
        [[-2.0, 5.0], [-2, -1.2]],
        [[1.0, -2.0], [5.0, -1.1]],
    ]
    a = get_abs_array(a, dtype)
    ia = dpnp.array(a)

    for axis in range(len(a)):
        result = getattr(dpnp, func)(ia, axis=axis, keepdims=keepdims)
        expected = getattr(numpy, func)(a, axis=axis, keepdims=keepdims)
        assert_allclose(result, expected)


def _get_min_max_input(dtype, shape):
    size = numpy.prod(shape)
    a = numpy.arange(size, dtype=dtype)
    a[int(size / 2)] = size + 5
    if numpy.issubdtype(dtype, numpy.unsignedinteger):
        a[int(size / 3)] = size
    else:
        a[int(size / 3)] = -(size + 5)

    return a.reshape(shape)


@pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True, no_bool=True))
@pytest.mark.parametrize(
    "shape", [(4,), (2, 3), (4, 5, 6)], ids=["1D", "2D", "3D"]
)
def test_amax_diff_shape(dtype, shape):
    a = _get_min_max_input(dtype, shape)
    ia = dpnp.array(a)

    expected = numpy.amax(a)
    result = dpnp.amax(ia)
    assert_array_equal(result, expected)

    expected = a.max()
    result = ia.max()
    assert_array_equal(result, expected)


@pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True, no_bool=True))
@pytest.mark.parametrize(
    "shape", [(4,), (2, 3), (4, 5, 6)], ids=["1D", "2D", "3D"]
)
def test_amin_diff_shape(dtype, shape):
    a = _get_min_max_input(dtype, shape)
    ia = dpnp.array(a)

    expected = numpy.amin(a)
    result = dpnp.amin(ia)
    assert_array_equal(result, expected)

    expected = a.min()
    result = ia.min()
    assert_array_equal(result, expected)
