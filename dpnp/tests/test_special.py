import math

import numpy
from numpy.testing import assert_allclose

import dpnp


def test_erf():
    a = numpy.linspace(2.0, 3.0, num=10)
    ia = dpnp.array(a)

    expected = numpy.empty_like(a)
    for idx, val in enumerate(a):
        expected[idx] = math.erf(val)

    result = dpnp.erf(ia)

    assert_allclose(result, expected)


def test_erf_fallback():
    a = numpy.linspace(2.0, 3.0, num=10)
    dpa = dpnp.linspace(2.0, 3.0, num=10)

    expected = numpy.empty_like(a)
    for idx, val in enumerate(a):
        expected[idx] = math.erf(val)

    result = dpnp.erf(dpa)

    assert_allclose(result, expected)
