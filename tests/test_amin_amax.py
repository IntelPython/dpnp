import pytest

import dpnp

import numpy


@pytest.mark.parametrize('type',
                         [numpy.float64],
                         ids=['float64'])
def test_amax_float64(type):
    a = numpy.array([[[-2., 3.], [9.1, 0.2]], [[-2., 5.0], [-2, -1.2]], [[1.0, -2.], [5.0, -1.1]]])
    ia = dpnp.array(a)

    for axis in range(len(a)):
        result = dpnp.amax(ia, axis=axis)
        expected = numpy.amax(a, axis=axis)
        numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize('type',
                         [numpy.int64],
                         ids=['int64'])
def test_amax_int(type):
    a = numpy.array([1, 0, 2, -3, -1, 2, 21, -9])
    ia = dpnp.array(a)

    result = dpnp.amax(ia)
    expected = numpy.amax(a)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize('type',
                         [numpy.float64],
                         ids=['float64'])
def test_amin_float64(type):
    a = numpy.array([[[-2., 3.], [9.1, 0.2]], [[-2., 5.0], [-2, -1.2]], [[1.0, -2.], [5.0, -1.1]]])
    ia = dpnp.array(a)

    for axis in range(len(a)):
        result = dpnp.amin(ia, axis=axis)
        expected = numpy.amin(a, axis=axis)
        numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize('type',
                         [numpy.int64],
                         ids=['int64'])
def test_amin_int(type):
    a = numpy.array([1, 0, 2, -3, -1, 2, 21, -9])
    ia = dpnp.array(a)

    result = dpnp.amin(ia)
    expected = numpy.amin(a)
    numpy.testing.assert_array_equal(expected, result)


def _get_min_max_input(type, shape):
    size = 1
    for i in range(len(shape)):
        size *= shape[i]

    a = numpy.arange(size, dtype=type)
    a[int(size / 2)] = size * size
    a[int(size / 3)] = -(size * size)

    return a.reshape(shape)


@pytest.mark.usefixtures('allow_fall_back_on_numpy')
@pytest.mark.parametrize('type',
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
@pytest.mark.parametrize('shape',
                         [(4,), (2, 3), (4, 5, 6)],
                         ids=['(4,)', '(2,3)', '(4,5,6)'])
def test_amax(type, shape):
    a = _get_min_max_input(type, shape)

    ia = dpnp.array(a)

    np_res = numpy.amax(a)
    dpnp_res = dpnp.amax(ia)
    numpy.testing.assert_array_equal(dpnp_res, np_res)

    np_res = a.max()
    dpnp_res = ia.max()
    numpy.testing.assert_array_equal(dpnp_res, np_res)


@pytest.mark.usefixtures('allow_fall_back_on_numpy')
@pytest.mark.parametrize('type',
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
@pytest.mark.parametrize('shape',
                         [(4,), (2, 3), (4, 5, 6)],
                         ids=['(4,)', '(2,3)', '(4,5,6)'])
def test_amin(type, shape):
    a = _get_min_max_input(type, shape)

    ia = dpnp.array(a)

    np_res = numpy.amin(a)
    dpnp_res = dpnp.amin(ia)
    numpy.testing.assert_array_equal(dpnp_res, np_res)

    np_res = a.min()
    dpnp_res = ia.min()
    numpy.testing.assert_array_equal(dpnp_res, np_res)
