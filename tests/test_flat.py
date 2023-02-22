import pytest

import dpnp as inp

import numpy


@pytest.mark.parametrize("dtype", [numpy.int64], ids=["int64"])
def test_flat(dtype):
    a = numpy.array([1, 0, 2, -3, -1, 2, 21, -9], dtype=dtype)
    ia = inp.array(a)

    result = ia.flat[0]
    expected = a.flat[0]
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("dtype", [numpy.int64], ids=["int64"])
def test_flat2(dtype):
    a = numpy.arange(1, 7, dtype=dtype).reshape(2, 3)
    ia = inp.array(a)

    result = ia.flat[3]
    expected = a.flat[3]
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("dtype", [numpy.int64], ids=["int64"])
def test_flat3(dtype):
    a = numpy.arange(1, 7, dtype=dtype).reshape(2, 3).T
    ia = inp.array(a)

    result = ia.flat[3]
    expected = a.flat[3]
    numpy.testing.assert_array_equal(expected, result)
