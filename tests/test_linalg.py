import pytest

import dpnp as inp

import numpy


def vvsort(val, vec, size):
    for i in range(size):
        imax = i
        for j in range(i + 1, size):
            if numpy.abs(val[imax]) < numpy.abs(val[j]):
                imax = j

        temp = val[i]
        val[i] = val[imax]
        val[imax] = temp

        for k in range(size):
            temp = vec[k, i]
            vec[k, i] = vec[k, imax]
            vec[k, imax] = temp


@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
@pytest.mark.parametrize("size",
                         [2, 4, 8, 16, 300])
def test_eig_arange(type, size):
    a = numpy.arange(size * size, dtype=type).reshape((size, size))
    symm = numpy.tril(a) + numpy.tril(a, -1).T + numpy.diag(numpy.full((size,), size * size, dtype=type))
    isymm = inp.array(symm)

    dpnp_val, dpnp_vec = inp.linalg.eig(isymm)
    np_val, np_vec = numpy.linalg.eig(symm)

    # DPNP sort val/vec by abs value
    vvsort(dpnp_val, dpnp_vec, size)

    # NP sort val/vec by abs value
    vvsort(np_val, np_vec, size)

    # NP change sign of vectors
    for i in range(np_vec.shape[1]):
        if np_vec[0, i] * dpnp_vec[0, i] < 0:
            np_vec[:, i] = -np_vec[:, i]

    numpy.testing.assert_allclose(dpnp_val, np_val, rtol=1e-05, atol=1e-05)
    numpy.testing.assert_allclose(dpnp_vec, np_vec, rtol=1e-05, atol=1e-05)


def test_matrix_rank():
    arrays = [
              [0, 0],
              # [0, 1],
              [1, 2],
              [[0, 0], [0, 0]],
              # [[1, 2], [1, 2]],
              # [[1, 2], [3, 4]],
              ]
    tols = [None]
    for array in arrays:
        for tol in tols:
            a = numpy.array(array)
            ia = inp.array(a)
            result = inp.linalg.matrix_rank(ia, tol=tol)
            expected = numpy.linalg.matrix_rank(a, tol=tol)
            numpy.testing.assert_array_equal(expected, result)
