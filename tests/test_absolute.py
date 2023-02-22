import pytest
from .helper import get_all_dtypes

import dpnp as inp

import numpy


@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True))
def test_abs(dtype):
    a = numpy.array([1, 0, 2, -3, -1, 2, 21, -9], dtype=dtype)
    ia = inp.array(a)

    result = inp.abs(ia)
    expected = numpy.abs(a)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize(
    "dtype", get_all_dtypes(no_bool=True, no_none=True, no_complex=True)
)
def test_absolute(dtype):
    a = numpy.array([[-2.0, 3.0, 9.1], [-2.0, 5.0, -2], [1.0, -2.0, 5.0]], dtype=dtype)
    ia = inp.array(a)

    result = inp.absolute(ia)
    expected = numpy.absolute(a)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize(
    "dtype", get_all_dtypes(no_bool=True, no_none=True, no_complex=True)
)
def test_absolute_float_3d(dtype):
    a = numpy.array(
        [
            [[-2.0, 3.0], [9.1, 0.2]],
            [[-2.0, 5.0], [-2, -1.2]],
            [[1.0, -2.0], [5.0, -1.1]],
        ],
        dtype=dtype,
    )
    ia = inp.array(a)

    result = inp.absolute(ia)
    expected = numpy.absolute(a)
    numpy.testing.assert_array_equal(expected, result)
