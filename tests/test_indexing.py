import pytest

import dpnp

import numpy


@pytest.mark.parametrize("m",
                         [None, 0, 1, 2, 3, 4],
                         ids=['None', '0', '1', '2', '3', '4'])
@pytest.mark.parametrize("k",
                         [0, 1, 2, 3, 4, 5],
                         ids=['0', '1', '2', '3', '4', '5'])
@pytest.mark.parametrize("n",
                         [1, 2, 3, 4, 5, 6],
                         ids=['1', '2', '3', '4', '5', '6'])
def test_tril_indices(n, k, m):
    result = dpnp.tril_indices(n, k, m)
    expected = numpy.tril_indices(n, k, m)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("k",
                         [0, 1, 2, 3, 4, 5],
                         ids=['0', '1', '2', '3', '4', '5'])
@pytest.mark.parametrize("array",
                         [[[0, 0], [0, 0]],
                          [[1, 2], [1, 2]],
                          [[1, 2], [3, 4]],],
                         ids=['[[0, 0], [0, 0]]',
                              '[[1, 2], [1, 2]]',
                              '[[1, 2], [3, 4]]'])
def test_tril_indices_from(array, k):
    a = numpy.array(array)
    ia = dpnp.array(a)
    result = dpnp.tril_indices_from(ia, k)
    expected = numpy.tril_indices_from(a, k)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("m",
                         [None, 0, 1, 2, 3, 4],
                         ids=['None', '0', '1', '2', '3', '4'])
@pytest.mark.parametrize("k",
                         [0, 1, 2, 3, 4, 5],
                         ids=['0', '1', '2', '3', '4', '5'])
@pytest.mark.parametrize("n",
                         [1, 2, 3, 4, 5, 6],
                         ids=['1', '2', '3', '4', '5', '6'])
def test_triu_indices(n, k, m):
    result = dpnp.triu_indices(n, k, m)
    expected = numpy.triu_indices(n, k, m)
    numpy.testing.assert_array_equal(expected, result)
