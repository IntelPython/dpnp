import pytest

import dpnp

import numpy


@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32, numpy.bool_],
                         ids=['float64', 'float32', 'int64', 'int32', 'bool'])
@pytest.mark.parametrize("shape",
                         [(0,), (4,), (2, 3), (2, 2, 2)],
                         ids=['(0,)', '(4,)', '(2,3)', '(2,2,2)'])
def test_all(type, shape):
    size = 1
    for i in range(len(shape)):
        size *= shape[i]

    for i in range(2 ** size):
        t = i

        a = numpy.empty(size, dtype=type)

        for j in range(size):
            a[j] = 0 if t % 2 == 0 else j + 1
            t = t >> 1

        a = a.reshape(shape)

        ia = dpnp.array(a)

        np_res = numpy.all(a)
        dpnp_res = dpnp.all(ia)
        numpy.testing.assert_allclose(dpnp_res, np_res)

        np_res = a.all()
        dpnp_res = ia.all()
        numpy.testing.assert_allclose(dpnp_res, np_res)


@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
def test_allclose(type):

    a = numpy.random.rand(10)
    b = a + numpy.random.rand(10) * 1e-8

    dpnp_a = dpnp.array(a, dtype=type)
    dpnp_b = dpnp.array(b, dtype=type)

    np_res = numpy.allclose(a, b)
    dpnp_res = dpnp.allclose(dpnp_a, dpnp_b)
    numpy.testing.assert_allclose(dpnp_res, np_res)

    a[0] = numpy.inf

    dpnp_a = dpnp.array(a)

    np_res = numpy.allclose(a, b)
    dpnp_res = dpnp.allclose(dpnp_a, dpnp_b)
    numpy.testing.assert_allclose(dpnp_res, np_res)


@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32, numpy.bool_],
                         ids=['float64', 'float32', 'int64', 'int32', 'bool'])
@pytest.mark.parametrize("shape",
                         [(0,), (4,), (2, 3), (2, 2, 2)],
                         ids=['(0,)', '(4,)', '(2,3)', '(2,2,2)'])
def test_any(type, shape):
    size = 1
    for i in range(len(shape)):
        size *= shape[i]

    for i in range(2 ** size):
        t = i

        a = numpy.empty(size, dtype=type)

        for j in range(size):
            a[j] = 0 if t % 2 == 0 else j + 1
            t = t >> 1

        a = a.reshape(shape)

        ia = dpnp.array(a)

        np_res = numpy.any(a)
        dpnp_res = dpnp.any(ia)
        numpy.testing.assert_allclose(dpnp_res, np_res)

        np_res = a.any()
        dpnp_res = ia.any()
        numpy.testing.assert_allclose(dpnp_res, np_res)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
def test_greater():
    a = numpy.array([1, 2, 3, 4, 5, 6, 7, 8])
    ia = dpnp.array(a)
    for i in range(len(a) + 1):
        np_res = (a > i)
        dpnp_res = (ia > i)
        numpy.testing.assert_equal(dpnp_res, np_res)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
def test_greater_equal():
    a = numpy.array([1, 2, 3, 4, 5, 6, 7, 8])
    ia = dpnp.array(a)
    for i in range(len(a) + 1):
        np_res = (a >= i)
        dpnp_res = (ia >= i)
        numpy.testing.assert_equal(dpnp_res, np_res)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
def test_less():
    a = numpy.array([1, 2, 3, 4, 5, 6, 7, 8])
    ia = dpnp.array(a)
    for i in range(len(a) + 1):
        np_res = (a < i)
        dpnp_res = (ia < i)
        numpy.testing.assert_equal(dpnp_res, np_res)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
def test_less_equal():
    a = numpy.array([1, 2, 3, 4, 5, 6, 7, 8])
    ia = dpnp.array(a)
    for i in range(len(a) + 1):
        np_res = (a <= i)
        dpnp_res = (ia <= i)
        numpy.testing.assert_equal(dpnp_res, np_res)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
def test_not_equal():
    a = numpy.array([1, 2, 3, 4, 5, 6, 7, 8])
    ia = dpnp.array(a)
    for i in range(len(a)):
        np_res = (a != i)
        dpnp_res = (ia != i)
        numpy.testing.assert_equal(dpnp_res, np_res)
