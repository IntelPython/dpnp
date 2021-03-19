import pytest

import dpnp

import numpy


@pytest.mark.parametrize("kth",
                         [0, 1],
                         ids=['0', '1'])
@pytest.mark.parametrize("dtype",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
@pytest.mark.parametrize("array",
                         [[3, 4, 2, 1],
                          [[1, 0], [3, 0]],
                          [[3, 2], [1, 6]],
                          [[4, 2, 3], [3, 4, 1]],
                          [[[1, -3], [3, 0]], [[5, 2], [0, 1]], [[1, 0], [0, 1]]],
                          [[[[8, 2], [3, 0]], [[5, 2], [0, 1]]], [[[1, 3], [3, 1]], [[5, 2], [0, 1]]]]],
                         ids=['[3, 4, 2, 1]',
                              '[[1, 0], [3, 0]]',
                              '[[3, 2], [1, 6]]',
                              '[[4, 2, 3], [3, 4, 1]]',
                              '[[[1, -3], [3, 0]], [[5, 2], [0, 1]], [[1, 0], [0, 1]]]',
                              '[[[[8, 2], [3, 0]], [[5, 2], [0, 1]]], [[[1, 3], [3, 1]], [[5, 2], [0, 1]]]]'])
def test_partition(array, dtype, kth):
    a = numpy.array(array, dtype)
    ia = dpnp.array(array, dtype)
    expected = numpy.partition(a, kth)
    result = dpnp.partition(ia, kth)
    numpy.testing.assert_array_equal(expected, result)
