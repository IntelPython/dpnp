import pytest

import dpnp as inp

import numpy

import unittest


@pytest.mark.parametrize("array",
                         [[[0, 0], [0, 0]],
                          [[1, 2], [1, 2]],
                          [[1, 2], [3, 4]],
                          [[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]],
                          [[[[1, 2], [3, 4]], [[1, 2], [2, 1]]], [[[1, 3], [3, 1]], [[0, 1], [1, 3]]]]
                          ],
                         ids=['[[0, 0], [0, 0]]',
                              '[[1, 2], [1, 2]]',
                              '[[1, 2], [3, 4]]',
                              '[[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]',
                              '[[[[1, 2], [3, 4]], [[1, 2], [2, 1]]], [[[1, 3], [3, 1]], [[0, 1], [1, 3]]]]'
                              ])
def test_diff(array):
    a = numpy.array(array)
    ia = inp.array(a)
    result = inp.linalg.det(ia)
    expected = numpy.linalg.det(a)
    numpy.testing.assert_allclose(expected, result)


class Test_nancumsum(unittest.TestCase):

    def test_nancumsum_dim1_int(self):
        a = numpy.array([1, 2, 3, 4, 5])
        ia = dpnp.array(a)

        result = dpnp.nancumsum(ia)
        expected = numpy.nancumsum(a)
        numpy.testing.assert_array_equal(expected, result)


    def test_nancumsum_dim1_nan(self):
        a = numpy.array([1, 2, numpy.nan, 4, 5])
        ia = dpnp.array(a)

        result = dpnp.nancumsum(ia)
        expected = numpy.nancumsum(a)
        numpy.testing.assert_array_equal(expected, result)


    def test_nancumsum_dim2_nan(self):
        a = numpy.array([[1, 2, numpy.nan], [3, -4, -5]])
        ia = dpnp.array(a)

        result = dpnp.nancumsum(ia)
        expected = numpy.nancumsum(a)
        numpy.testing.assert_array_equal(expected, result)


class Test_nancumprod(unittest.TestCase):

    def test_nancumprod_dim1_int():
        a = numpy.array([1, 2, 3, 4, 5])
        ia = dpnp.array(a)

        result = dpnp.nancumprod(ia)
        expected = numpy.nancumprod(a)
        numpy.testing.assert_array_equal(expected, result)


    def test_nancumprod_dim1_nan():
        a = numpy.array([1, 2, numpy.nan, 4, 5])
        ia = dpnp.array(a)

        result = dpnp.nancumprod(ia)
        expected = numpy.nancumprod(a)
        numpy.testing.assert_array_equal(expected, result)


    def test_nancumprod_dim2_nan():
        a = numpy.array([[1, 2, numpy.nan], [3, -4, -5]])
        ia = dpnp.array(a)

        result = dpnp.nancumsum(ia)
        expected = numpy.nancumsum(a)
        numpy.testing.assert_array_equal(expected, result)


if __name__ == '__main__':
    unittest.main()
