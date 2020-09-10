import pytest

import dpnp as inp

import numpy


@pytest.mark.parametrize("type",
                         [numpy.int64],
                         ids=['int64'])
def test_abs_int(type):
    a = numpy.array([1, 0, 2, -3, -1, 2, 21, -9])
    ia = inp.array(a)

    result = inp.abs(ia)
    expected = numpy.abs(a)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("type",
                         [numpy.int64],
                         ids=['int64'])
def test_absolute_int(type):
    a = numpy.array([1, 0, 2, -3, -1, 2, 21, -9])
    ia = inp.array(a)

    result = inp.absolute(ia)
    expected = numpy.absolute(a)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("type",
                         [numpy.float64],
                         ids=['float64'])
def test_absolute_float(type):
    a = numpy.array([[-2., 3., 9.1], [-2., 5.0, -2], [1.0, -2., 5.0]])
    ia = inp.array(a)

    result = inp.absolute(ia)
    expected = numpy.absolute(a)
    numpy.testing.assert_array_equal(expected, result)
