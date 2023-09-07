import numpy
import pytest
from numpy.testing import (
    assert_array_equal,
)

import dpnp
from tests.helper import (
    assert_dtype_allclose,
    get_float_dtypes,
    has_support_aspect64,
)


# Note: dpnp.sum() always upcast integers to (u)int64 and float32 to
# float64 for dtype=None. `numpy.sum()` does that too for integers, but not for
# float32, so we need to special-case it for these tests
@pytest.mark.parametrize("dtype", get_float_dtypes())
def test_sum_float(dtype):
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
        result = dpnp.sum(ia, axis=axis)
        expected = numpy.sum(a, axis=axis)
        assert_dtype_allclose(result, expected)


def test_sum_int():
    a = numpy.array([1, 0, 2, -3, -1, 2, 21, -9])
    ia = dpnp.array(a)

    result = dpnp.sum(ia)
    expected = numpy.sum(a)
    assert_array_equal(expected, result)


def test_sum_axis():
    a = numpy.array(
        [
            [[-2.0, 3.0], [9.1, 0.2]],
            [[-2.0, 5.0], [-2, -1.2]],
            [[1.0, -2.0], [5.0, -1.1]],
        ],
        dtype="f4",
    )
    ia = dpnp.array(a)

    result = dpnp.sum(ia, axis=1)
    if has_support_aspect64():
        expected = numpy.sum(a, axis=1, dtype=numpy.float64)
    else:
        expected = numpy.sum(a, axis=1)
    assert_array_equal(expected, result)
