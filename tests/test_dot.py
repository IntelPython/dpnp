import pytest

import dpnp as inp

import numpy


@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
def test_dot_ones(type):
    n = 10**5
    a = numpy.ones(n, dtype=type)
    b = numpy.ones(n, dtype=type)
    ia = inp.array(a)
    ib = inp.array(b)

    result = inp.dot(ia, ib)
    expected = numpy.dot(a, b)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
def test_dot_arange(type):
    n = 10**2
    m = 10**3
    a = numpy.hstack((numpy.arange(n, dtype=type),) * m)
    b = numpy.flipud(a)
    ia = inp.array(a)
    ib = inp.array(b)

    result = inp.dot(ia, ib)
    expected = numpy.dot(a, b)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
def test_multi_dot(type):
    n = 16
    a = inp.arange(n, dtype=type).reshape((4, 4))
    b = inp.arange(n, dtype=type).reshape((4, 4))
    c = inp.arange(n, dtype=type).reshape((4, 4))
    d = inp.arange(n, dtype=type).reshape((4, 4))

    a1 = numpy.arange(n, dtype=type).reshape((4, 4))
    b1 = numpy.arange(n, dtype=type).reshape((4, 4))
    c1 = numpy.arange(n, dtype=type).reshape((4, 4))
    d1 = numpy.arange(n, dtype=type).reshape((4, 4))

    result = inp.linalg.multi_dot([a, b, c, d])
    expected = numpy.linalg.multi_dot([a1, b1, c1, d1])
    numpy.testing.assert_array_equal(expected, result)
