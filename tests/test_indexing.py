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
