import numpy
import pytest
from numpy.testing import (
    assert_array_equal,
)

import dpnp
from tests.helper import (
    assert_dtype_allclose,
    get_all_dtypes,
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

    # Flag for type check in special cases
    # Use only type kinds checks when dpnp handles float32 arrays
    # as `dpnp.sum()` and `numpy.sum()` return different dtypes
    check_type_kind = dtype == dpnp.float32
    for axis in range(len(a)):
        result = dpnp.sum(ia, axis=axis)
        expected = numpy.sum(a, axis=axis)
        assert_dtype_allclose(
            result, expected, check_only_type_kind=check_type_kind
        )


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


@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
@pytest.mark.parametrize("axis", [0, 1, (0, 1)])
def test_sum_out(dtype, axis):
    a = dpnp.arange(2 * 4, dtype=dtype).reshape(2, 4)
    a_np = dpnp.asnumpy(a)

    expected = numpy.sum(a_np, axis=axis)
    res = dpnp.empty(expected.shape, dtype=dtype)
    a.sum(axis=axis, out=res)
    assert_array_equal(expected, res.asnumpy())


def test_sum_NotImplemented():
    ia = dpnp.arange(5)
    with pytest.raises(NotImplementedError):
        dpnp.sum(ia, where=False)

    with pytest.raises(NotImplementedError):
        dpnp.sum(ia, initial=1)
