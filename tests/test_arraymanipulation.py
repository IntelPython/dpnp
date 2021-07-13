import pytest

import dpnp
import numpy


@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
@pytest.mark.parametrize("input",
                         [[1, 2, 3], [1., 2., 3.], dpnp.array([1, 2, 3]), dpnp.array([1., 2., 3.])],
                         ids=['intlist', 'floatlist', 'intarray', 'floatarray'])
def test_asfarray(type, input):
    np_res = numpy.asfarray(input, type)
    dpnp_res = dpnp.asfarray(input, type)

    numpy.testing.assert_array_equal(dpnp_res, np_res)


class TestHstack:
    def test_non_iterable(self):
        numpy.testing.assert_raises(TypeError, dpnp.hstack, 1)

    def test_empty_input(self):
        numpy.testing.assert_raises(ValueError, dpnp.hstack, ())

    def test_0D_array(self):
        b = dpnp.array(2)
        a = dpnp.array(1)
        res = dpnp.hstack([a, b])
        desired = dpnp.array([1, 2])
        numpy.testing.assert_array_equal(res, desired)

    def test_1D_array(self):
        a = dpnp.array([1])
        b = dpnp.array([2])
        res = dpnp.hstack([a, b])
        desired = dpnp.array([1, 2])
        numpy.testing.assert_array_equal(res, desired)

    def test_2D_array(self):
        a = dpnp.array([[1], [2]])
        b = dpnp.array([[1], [2]])
        res = dpnp.hstack([a, b])
        desired = dpnp.array([[1, 1], [2, 2]])
        numpy.testing.assert_array_equal(res, desired)

    def test_generator(self):
        with numpy.testing.assert_warns(FutureWarning):
            dpnp.hstack((numpy.arange(3) for _ in range(2)))
        with numpy.testing.assert_warns(FutureWarning):
            dpnp.hstack(map(lambda x: x, numpy.ones((3, 2))))


class TestVstack:
    def test_non_iterable(self):
        numpy.testing.assert_raises(TypeError, vstack, 1)

    def test_empty_input(self):
        numpy.testing.assert_raises(ValueError, vstack, ())

    def test_0D_array(self):
        a = dpnp.array(1)
        b = dpnp.array(2)
        res = dpnp.vstack([a, b])
        desired = dpnp.array([[1], [2]])
        numpy.testing.assert_array_equal(res, desired)

    def test_1D_array(self):
        a = dpnp.array([1])
        b = dpnp.array([2])
        res = dpnp.vstack([a, b])
        desired = dpnp.array([[1], [2]])
        numpy.testing.assert_array_equal(res, desired)

    def test_2D_array(self):
        a = dpnp.array([[1], [2]])
        b = dpnp.array([[1], [2]])
        res = dpnp.vstack([a, b])
        desired = dpnp.array([[1], [2], [1], [2]])
        numpy.testing.assert_array_equal(res, desired)

    def test_2D_array2(self):
        a = dpnp.array([1, 2])
        b = dpnp.array([1, 2])
        res = dpnp.vstack([a, b])
        desired = dpnp.array([[1, 2], [1, 2]])
        numpy.testing.assert_array_equal(res, desired)

    def test_generator(self):
        with numpy.testing.assert_warns(FutureWarning):
            dpnp.vstack((numpy.arange(3) for _ in range(2)))
