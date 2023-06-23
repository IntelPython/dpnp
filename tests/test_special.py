import math

import numpy

import dpnp


def test_erf():
    a = numpy.linspace(2.0, 3.0, num=10)
    ia = dpnp.linspace(2.0, 3.0, num=10)

    numpy.testing.assert_allclose(a, ia)

    expected = numpy.empty_like(a)
    for idx, val in enumerate(a):
        expected[idx] = math.erf(val)

    result = dpnp.erf(ia)

    numpy.testing.assert_array_equal(result, expected)


def test_erf_fallback():
    a = numpy.linspace(2.0, 3.0, num=10)
    dpa = dpnp.linspace(2.0, 3.0, num=10)

    expected = numpy.empty_like(a)
    for idx, val in enumerate(a):
        expected[idx] = math.erf(val)

    result = dpnp.erf(dpa)

    numpy.testing.assert_array_equal(result, expected)
