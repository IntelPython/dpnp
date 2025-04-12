import math

import numpy

import dpnp

from .helper import assert_dtype_allclose


def test_erf():
    a = numpy.linspace(2.0, 3.0, num=10)
    ia = dpnp.array(a)

    expected = numpy.empty_like(a)
    for idx, val in enumerate(a):
        expected[idx] = math.erf(val)

    result = dpnp.erf(ia)

    assert_dtype_allclose(result, expected)


def test_erf_fallback():
    a = numpy.linspace(2.0, 3.0, num=10)
    dpa = dpnp.linspace(2.0, 3.0, num=10)

    expected = numpy.empty_like(a)
    for idx, val in enumerate(a):
        expected[idx] = math.erf(val)

    result = dpnp.erf(dpa)

    assert_dtype_allclose(result, expected)
