import pytest

import dpnp as inp

import numpy


@pytest.mark.parametrize("type",
                         [numpy.float64],
                         ids=['float64'])
def test_absolute1(type):
    n = 10**5
    a = numpy.random.random(n)*2-1
    ia = inp.array(a)

    result = inp.absolute(ia)
    expected = numpy.absolute(a)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("type",
                         [numpy.float64],
                         ids=['float64'])
def test_absolute2(type):
    n = 10**2
    m = 10**3
    a = numpy.random.random((n, m)) * 2 - 1
    ia = inp.array(a)

    result = inp.absolute(ia)
    expected = numpy.absolute(a)
    numpy.testing.assert_array_equal(expected, result)
