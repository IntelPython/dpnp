import dpnp
from .helper import skip_or_check_if_dtype_not_supported

import numpy


dtype = numpy.float64 if skip_or_check_if_dtype_not_supported(
    numpy.float64, check_dtype=True
    ) else numpy.float32


def test_sum_float():
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
        numpy.testing.assert_array_equal(expected, result)


def test_sum_int():
    a = numpy.array([1, 0, 2, -3, -1, 2, 21, -9])
    ia = dpnp.array(a)

    result = dpnp.sum(ia)
    expected = numpy.sum(a)
    numpy.testing.assert_array_equal(expected, result)


def test_sum_axis():
    a = numpy.array(
        [
            [[-2.0, 3.0], [9.1, 0.2]],
            [[-2.0, 5.0], [-2, -1.2]],
            [[1.0, -2.0], [5.0, -1.1]],
        ],
        dtype=dtype,
    )
    ia = dpnp.array(a)

    result = dpnp.sum(ia, axis=1)
    expected = numpy.sum(a, axis=1)
    numpy.testing.assert_array_equal(expected, result)
