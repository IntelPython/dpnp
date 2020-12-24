import pytest

import dpnp

import numpy


@pytest.mark.parametrize("offset",
                         [0, 1],
                         ids=['0', '1'])
@pytest.mark.parametrize("array",
                         [[[0, 0], [0, 0]],
                          [[1, 2], [1, 2]],
                          [[1, 2], [3, 4]],
                          [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                          [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
                          [[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]],
                          [[[[1, 2], [3, 4]], [[1, 2], [2, 1]]], [[[1, 3], [3, 1]], [[0, 1], [1, 3]]]],
                          [[[[1, 2, 3], [3, 4, 5]], [[1, 2, 3], [2, 1, 0]]], [[[1, 3, 5], [3, 1, 0]], [[0, 1, 2], [1, 3, 4]]]],
                          [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]]]],
                         ids=['[[0, 0], [0, 0]]',
                              '[[1, 2], [1, 2]]',
                              '[[1, 2], [3, 4]]',
                              '[[0, 1, 2], [3, 4, 5], [6, 7, 8]]',
                              '[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]',
                              '[[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]',
                              '[[[[1, 2], [3, 4]], [[1, 2], [2, 1]]], [[[1, 3], [3, 1]], [[0, 1], [1, 3]]]]',
                              '[[[[1, 2, 3], [3, 4, 5]], [[1, 2, 3], [2, 1, 0]]], [[[1, 3, 5], [3, 1, 0]], [[0, 1, 2], [1, 3, 4]]]]',
                              '[[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]]]'])
def test_diagonal(array, offset):
    a = numpy.array(array)
    ia = dpnp.array(a)
    expected = numpy.diagonal(a, offset)
    result = dpnp.diagonal(ia, offset)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("v",
                         [0, 1, 2, 3, 4],
                         ids=['0', '1', '2', '3', '4'])
@pytest.mark.parametrize("ind",
                         [0, 1, 2, 3],
                         ids=['0', '1', '2', '3'])
@pytest.mark.parametrize("array",
                         [[[0, 0], [0, 0]],
                          [[1, 2], [1, 2]],
                          [[1, 2], [3, 4]],
                          [[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]],
                          [[[[1, 2], [3, 4]], [[1, 2], [2, 1]]], [[[1, 3], [3, 1]], [[0, 1], [1, 3]]]]],
                         ids=['[[0, 0], [0, 0]]',
                              '[[1, 2], [1, 2]]',
                              '[[1, 2], [3, 4]]',
                              '[[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]',
                              '[[[[1, 2], [3, 4]], [[1, 2], [2, 1]]], [[[1, 3], [3, 1]], [[0, 1], [1, 3]]]]'])
def test_put(array, ind, v):
    a = numpy.array(array)
    ia = dpnp.array(a)
    numpy.put(a, ind, v)
    dpnp.put(ia, ind, v)
    numpy.testing.assert_array_equal(a, ia)


@pytest.mark.parametrize("v",
                         [[10, 20], [30, 40]],
                         ids=['[10, 20]', '[30, 40]'])
@pytest.mark.parametrize("ind",
                         [[0, 1], [2, 3]],
                         ids=['[0, 1]', '[2, 3]'])
@pytest.mark.parametrize("array",
                         [[[0, 0], [0, 0]],
                          [[1, 2], [1, 2]],
                          [[1, 2], [3, 4]],
                          [[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]],
                          [[[[1, 2], [3, 4]], [[1, 2], [2, 1]]], [[[1, 3], [3, 1]], [[0, 1], [1, 3]]]]],
                         ids=['[[0, 0], [0, 0]]',
                              '[[1, 2], [1, 2]]',
                              '[[1, 2], [3, 4]]',
                              '[[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]',
                              '[[[[1, 2], [3, 4]], [[1, 2], [2, 1]]], [[[1, 3], [3, 1]], [[0, 1], [1, 3]]]]'])
def test_put2(array, ind, v):
    a = numpy.array(array)
    ia = dpnp.array(a)
    numpy.put(a, ind, v)
    dpnp.put(ia, ind, v)
    numpy.testing.assert_array_equal(a, ia)


def test_put3():
    a = numpy.arange(5)
    ia = dpnp.array(a)
    dpnp.put(ia, [0, 2], [-44, -55])
    numpy.put(a, [0, 2], [-44, -55])
    numpy.testing.assert_array_equal(a, ia)


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
                          [[1, 2], [3, 4]], ],
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


@pytest.mark.parametrize("k",
                         [0, 1, 2, 3, 4, 5],
                         ids=['0', '1', '2', '3', '4', '5'])
@pytest.mark.parametrize("array",
                         [[[0, 0], [0, 0]],
                          [[1, 2], [1, 2]],
                          [[1, 2], [3, 4]], ],
                         ids=['[[0, 0], [0, 0]]',
                              '[[1, 2], [1, 2]]',
                              '[[1, 2], [3, 4]]'])
def test_triu_indices_from(array, k):
    a = numpy.array(array)
    ia = dpnp.array(a)
    result = dpnp.triu_indices_from(ia, k)
    expected = numpy.triu_indices_from(a, k)
    numpy.testing.assert_array_equal(expected, result)
