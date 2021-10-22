import pytest

import dpnp as inp

import numpy


def vvsort(val, vec, size, xp):
    for i in range(size):
        imax = i
        for j in range(i + 1, size):
            unravel_imax = numpy.unravel_index(imax, val.shape)
            unravel_j = numpy.unravel_index(j, val.shape)
            if xp.abs(val[unravel_imax]) < xp.abs(val[unravel_j]):
                imax = j

        unravel_i = numpy.unravel_index(i, val.shape)
        unravel_imax = numpy.unravel_index(imax, val.shape)

        temp = xp.empty(tuple(), dtype=vec.dtype)
        temp[()] = val[unravel_i]  # make a copy
        val[unravel_i] = val[unravel_imax]
        val[unravel_imax] = temp

        for k in range(size):
            temp = xp.empty(tuple(), dtype=val.dtype)
            temp[()] = vec[k, i]  # make a copy
            vec[k, i] = vec[k, imax]
            vec[k, imax] = temp


@pytest.mark.parametrize("array",
                         [[[[1, -2], [2, 5]]],
                          [[[1., -2.], [2., 5.]]],
                          [[[1., -2.], [2., 5.]], [[1., -2.], [2., 5.]]]],
                         ids=['[[[1, -2], [2, 5]]]',
                              '[[[1., -2.], [2., 5.]]]',
                              '[[[1., -2.], [2., 5.]], [[1., -2.], [2., 5.]]]'])
def test_cholesky(array):
    a = numpy.array(array)
    ia = inp.array(a)
    result = inp.linalg.cholesky(ia)
    expected = numpy.linalg.cholesky(a)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("arr",
                         [[[1, 0, -1], [0, 1, 0], [1, 0, 1]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
                         ids=['[[1, 0, -1], [0, 1, 0], [1, 0, 1]]', '[[1, 2, 3], [4, 5, 6], [7, 8, 9]]'])
@pytest.mark.parametrize("p",
                         [None, 1, -1, 2, -2, numpy.inf, -numpy.inf, 'fro'],
                         ids=['None', '1', '-1', '2', '-2', 'numpy.inf', '-numpy.inf', '"fro"'])
def test_cond(arr, p):
    a = numpy.array(arr)
    ia = inp.array(a)
    result = inp.linalg.cond(ia, p)
    expected = numpy.linalg.cond(a, p)
    numpy.testing.assert_array_equal(expected, result)


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
def test_det(array):
    a = numpy.array(array)
    ia = inp.array(a)
    result = inp.linalg.det(ia)
    expected = numpy.linalg.det(a)
    numpy.testing.assert_allclose(expected, result)


@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
@pytest.mark.parametrize("size",
                         [2, 4, 8, 16, 300])
def test_eig_arange(type, size):
    a = numpy.arange(size * size, dtype=type).reshape((size, size))
    symm_orig = numpy.tril(a) + numpy.tril(a, -1).T + numpy.diag(numpy.full((size,), size * size, dtype=type))
    symm = symm_orig
    dpnp_symm_orig = inp.array(symm)
    dpnp_symm = dpnp_symm_orig

    dpnp_val, dpnp_vec = inp.linalg.eig(dpnp_symm)
    np_val, np_vec = numpy.linalg.eig(symm)

    # DPNP sort val/vec by abs value
    vvsort(dpnp_val, dpnp_vec, size, inp)

    # NP sort val/vec by abs value
    vvsort(np_val, np_vec, size, numpy)

    # NP change sign of vectors
    for i in range(np_vec.shape[1]):
        if np_vec[0, i] * dpnp_vec[0, i] < 0:
            np_vec[:, i] = -np_vec[:, i]

    numpy.testing.assert_array_equal(symm_orig, symm)
    numpy.testing.assert_array_equal(dpnp_symm_orig, dpnp_symm)

    assert (dpnp_val.dtype == np_val.dtype)
    assert (dpnp_vec.dtype == np_vec.dtype)
    assert (dpnp_val.shape == np_val.shape)
    assert (dpnp_vec.shape == np_vec.shape)

    numpy.testing.assert_allclose(dpnp_val, np_val, rtol=1e-05, atol=1e-05)
    numpy.testing.assert_allclose(dpnp_vec, np_vec, rtol=1e-05, atol=1e-05)


def test_eigvals():
    arrays = [
        [[0, 0], [0, 0]],
        [[1, 2], [1, 2]],
        [[1, 2], [3, 4]]
    ]
    for array in arrays:
        a = numpy.array(array)
        ia = inp.array(a)
        result = inp.linalg.eigvals(ia)
        expected = numpy.linalg.eigvals(a)
        numpy.testing.assert_allclose(expected, result, atol=0.5)


@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
@pytest.mark.parametrize("array",
                         [[[1., 2.], [3., 4.]], [[0, 1, 2], [3, 2, -1], [4, -2, 3]]],
                         ids=['[[1., 2.], [3., 4.]]', '[[0, 1, 2], [3, 2, -1], [4, -2, 3]]'])
def test_inv(type, array):
    a = numpy.array(array, dtype=type)
    ia = inp.array(a)
    result = inp.linalg.inv(ia)
    expected = numpy.linalg.inv(a)
    numpy.testing.assert_allclose(expected, result)


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


@pytest.mark.parametrize("array",
                         [[7], [1, 2], [1, 0]],
                         ids=['[7]', '[1, 2]', '[1, 0]'])
@pytest.mark.parametrize("ord",
                         [None, -numpy.Inf, -2, -1, 0, 1, 2, 3, numpy.Inf],
                         ids=['None', '-numpy.Inf', '-2', '-1', '0', '1', '2', '3', 'numpy.Inf'])
@pytest.mark.parametrize("axis",
                         [0, None],
                         ids=['0', 'None'])
def test_norm1(array, ord, axis):
    a = numpy.array(array)
    ia = inp.array(a)
    result = inp.linalg.norm(ia, ord=ord, axis=axis)
    expected = numpy.linalg.norm(a, ord=ord, axis=axis)
    numpy.testing.assert_allclose(expected, result)


@pytest.mark.parametrize("array",
                         [[[1, 0]], [[1, 2]], [[1, 0], [3, 0]], [[1, 2], [3, 4]]],
                         ids=['[[1, 0]]', '[[1, 2]]', '[[1, 0], [3, 0]]', '[[1, 2], [3, 4]]'])
@pytest.mark.parametrize("ord",
                         [None, -numpy.Inf, -2, -1, 1, 2, numpy.Inf, 'fro', 'nuc'],
                         ids=['None', '-numpy.Inf', '-2', '-1', '1', '2', 'numpy.Inf', '"fro"', '"nuc"'])
@pytest.mark.parametrize("axis",
                         [(0, 1), None],
                         ids=['(0, 1)', 'None'])
def test_norm2(array, ord, axis):
    a = numpy.array(array)
    ia = inp.array(a)
    result = inp.linalg.norm(ia, ord=ord, axis=axis)
    expected = numpy.linalg.norm(a, ord=ord, axis=axis)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("array",
                         [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[1, 0], [3, 0]], [[5, 0], [7, 0]]]],
                         ids=['[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]', '[[[1, 0], [3, 0]], [[5, 0], [7, 0]]]'])
@pytest.mark.parametrize("ord",
                         [None, -numpy.Inf, -2, -1, 1, 2, numpy.Inf],
                         ids=['None', '-numpy.Inf', '-2', '-1', '1', '2', 'numpy.Inf'])
@pytest.mark.parametrize("axis",
                         [0, 1, 2, (0, 1), (0, 2), (1, 2)],
                         ids=['0', '1', '2', '(0, 1)', '(0, 2)', '(1, 2)'])
def test_norm3(array, ord, axis):
    a = numpy.array(array)
    ia = inp.array(a)
    result = inp.linalg.norm(ia, ord=ord, axis=axis)
    expected = numpy.linalg.norm(a, ord=ord, axis=axis)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
@pytest.mark.parametrize("shape",
                         [(2, 2), (3, 4), (5, 3), (16, 16)],
                         ids=['(2,2)', '(3,4)', '(5,3)', '(16,16)'])
def test_qr(type, shape):
    a = numpy.arange(shape[0] * shape[1], dtype=type).reshape(shape)
    ia = inp.array(a)

    np_q, np_r = numpy.linalg.qr(a, "complete")
    dpnp_q, dpnp_r = inp.linalg.qr(ia, "complete")

    assert (dpnp_q.dtype == np_q.dtype)
    assert (dpnp_r.dtype == np_r.dtype)
    assert (dpnp_q.shape == np_q.shape)
    assert (dpnp_r.shape == np_r.shape)

    if type == numpy.float32:
        tol = 1e-02
    else:
        tol = 1e-11

    # check decomposition
    numpy.testing.assert_allclose(ia, numpy.dot(inp.asnumpy(dpnp_q), inp.asnumpy(dpnp_r)), rtol=tol, atol=tol)

    # NP change sign for comparison
    ncols = min(a.shape[0], a.shape[1])
    for i in range(ncols):
        j = numpy.where(numpy.abs(np_q[:, i]) > tol)[0][0]
        if np_q[j, i] * dpnp_q[j, i] < 0:
            np_q[:, i] = -np_q[:, i]
            np_r[i, :] = -np_r[i, :]

        if numpy.any(numpy.abs(np_r[i, :]) > tol):
            numpy.testing.assert_allclose(inp.asnumpy(dpnp_q)[:, i], np_q[:, i], rtol=tol, atol=tol)

    numpy.testing.assert_allclose(dpnp_r, np_r, rtol=tol, atol=tol)


@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
@pytest.mark.parametrize("shape",
                         [(2, 2), (3, 4), (5, 3), (16, 16)],
                         ids=['(2,2)', '(3,4)', '(5,3)', '(16,16)'])
def test_svd(type, shape):
    a = numpy.arange(shape[0] * shape[1], dtype=type).reshape(shape)
    ia = inp.array(a)

    np_u, np_s, np_vt = numpy.linalg.svd(a)
    dpnp_u, dpnp_s, dpnp_vt = inp.linalg.svd(ia)

    assert (dpnp_u.dtype == np_u.dtype)
    assert (dpnp_s.dtype == np_s.dtype)
    assert (dpnp_vt.dtype == np_vt.dtype)
    assert (dpnp_u.shape == np_u.shape)
    assert (dpnp_s.shape == np_s.shape)
    assert (dpnp_vt.shape == np_vt.shape)

    if type == numpy.float32:
        tol = 1e-03
    else:
        tol = 1e-12

    # check decomposition
    dpnp_diag_s = inp.zeros(shape, dtype=dpnp_s.dtype)
    for i in range(dpnp_s.size):
        dpnp_diag_s[i, i] = dpnp_s[i]

    # check decomposition
    numpy.testing.assert_allclose(ia, inp.dot(dpnp_u, inp.dot(dpnp_diag_s, dpnp_vt)), rtol=tol, atol=tol)

    # compare singular values
    # numpy.testing.assert_allclose(dpnp_s, np_s, rtol=tol, atol=tol)

    # change sign of vectors
    for i in range(min(shape[0], shape[1])):
        if np_u[0, i] * dpnp_u[0, i] < 0:
            np_u[:, i] = -np_u[:, i]
            np_vt[i, :] = -np_vt[i, :]

    # compare vectors for non-zero values
    for i in range(numpy.count_nonzero(np_s > tol)):
        numpy.testing.assert_allclose(inp.asnumpy(dpnp_u)[:, i], np_u[:, i], rtol=tol, atol=tol)
        numpy.testing.assert_allclose(inp.asnumpy(dpnp_vt)[i, :], np_vt[i, :], rtol=tol, atol=tol)
