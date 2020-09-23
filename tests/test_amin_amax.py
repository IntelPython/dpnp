import pytest

import dpnp as inp

import numpy


@pytest.mark.parametrize("type",
                         [numpy.float64],
                         ids=['float64'])
def test_amax_float64(type):
    a = numpy.array([[[-2., 3.], [9.1, 0.2]], [[-2., 5.0], [-2, -1.2]], [[1.0, -2.], [5.0, -1.1]]])
    ia = inp.array(a)

    for axis in range(len(a)):
        result = inp.amax(ia, axis=axis)
        expected = numpy.amax(a, axis=axis)
        numpy.testing.assert_array_equal(expected, result)



@pytest.mark.parametrize("type",
                         [numpy.int64],
                         ids=['int64'])
def test_amax_int(type):
    a = numpy.array([1, 0, 2, -3, -1, 2, 21, -9])
    ia = inp.array(a)

    result = inp.amax(ia)
    expected = numpy.amax(a)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("type",
                         [numpy.float64],
                         ids=['float64'])
def test_amin_float64(type):
    a = numpy.array([[[-2., 3.], [9.1, 0.2]], [[-2., 5.0], [-2, -1.2]], [[1.0, -2.], [5.0, -1.1]]])
    ia = inp.array(a)

    for axis in range(len(a)):
        result = inp.amin(ia, axis=axis)
        expected = numpy.amin(a, axis=axis)
        numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("type",
                         [numpy.int64],
                         ids=['int64'])
def test_amin_int(type):
    a = numpy.array([1, 0, 2, -3, -1, 2, 21, -9])
    ia = inp.array(a)

    result = inp.amin(ia)
    expected = numpy.amin(a)
    numpy.testing.assert_array_equal(expected, result)
