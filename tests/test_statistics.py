import pytest

import dpnp

import numpy


@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
@pytest.mark.parametrize("size",
                         [2, 4, 8, 16, 3, 9, 27, 81])
def test_median(type, size):
    a = numpy.arange(size, dtype=type)
    ia = dpnp.array(a)

    np_res = numpy.median(a)
    dpnp_res = dpnp.median(ia)

    numpy.testing.assert_allclose(dpnp_res, np_res)


@pytest.mark.parametrize("axis",
                         [0, 1, -1, 2, -2, (1, 2), (0, -2)])
def test_max(axis):
    a = numpy.arange(768, dtype=numpy.float64).reshape((4, 4, 6, 8))
    ia = dpnp.array(a)

    np_res = numpy.max(a, axis=axis)
    dpnp_res = dpnp.max(ia, axis=axis)

    numpy.testing.assert_allclose(dpnp_res, np_res)
