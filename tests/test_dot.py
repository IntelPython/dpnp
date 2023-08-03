import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import dpnp as inp

from .helper import assert_dtype_allclose, get_all_dtypes


@pytest.mark.parametrize("type", get_all_dtypes(no_bool=True, no_complex=True))
def test_dot_ones(type):
    n = 10**5
    a = numpy.ones(n, dtype=type)
    b = numpy.ones(n, dtype=type)
    ia = inp.array(a)
    ib = inp.array(b)

    result = inp.dot(ia, ib)
    expected = numpy.dot(a, b)
    assert_array_equal(expected, result)


@pytest.mark.parametrize("type", get_all_dtypes(no_bool=True, no_complex=True))
def test_dot_arange(type):
    n = 10**2
    m = 10**3
    a = numpy.hstack((numpy.arange(n, dtype=type),) * m)
    b = numpy.flipud(a)
    ia = inp.array(a)
    ib = inp.array(b)

    result = inp.dot(ia, ib)
    expected = numpy.dot(a, b)
    assert_dtype_allclose(result, expected)


@pytest.mark.parametrize("type", get_all_dtypes(no_bool=True, no_complex=True))
def test_multi_dot(type):
    n = 16
    a = inp.reshape(inp.arange(n, dtype=type), (4, 4))
    b = inp.reshape(inp.arange(n, dtype=type), (4, 4))
    c = inp.reshape(inp.arange(n, dtype=type), (4, 4))
    d = inp.reshape(inp.arange(n, dtype=type), (4, 4))

    a1 = numpy.arange(n, dtype=type).reshape((4, 4))
    b1 = numpy.arange(n, dtype=type).reshape((4, 4))
    c1 = numpy.arange(n, dtype=type).reshape((4, 4))
    d1 = numpy.arange(n, dtype=type).reshape((4, 4))

    result = inp.linalg.multi_dot([a, b, c, d])
    expected = numpy.linalg.multi_dot([a1, b1, c1, d1])
    assert_array_equal(expected, result)
