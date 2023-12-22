import dpctl
import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal, assert_raises

import dpnp as inp

from .helper import (
    assert_dtype_allclose,
    get_all_dtypes,
    get_complex_dtypes,
    has_support_aspect64,
    is_cpu_device,
)


def vvsort(val, vec, size, xp):
    val_kwargs = {}
    if hasattr(val, "sycl_queue"):
        val_kwargs["sycl_queue"] = getattr(val, "sycl_queue", None)

    vec_kwargs = {}
    if hasattr(vec, "sycl_queue"):
        vec_kwargs["sycl_queue"] = getattr(vec, "sycl_queue", None)

    for i in range(size):
        imax = i
        for j in range(i + 1, size):
            unravel_imax = numpy.unravel_index(imax, val.shape)
            unravel_j = numpy.unravel_index(j, val.shape)
            if xp.abs(val[unravel_imax]) < xp.abs(val[unravel_j]):
                imax = j

        unravel_i = numpy.unravel_index(i, val.shape)
        unravel_imax = numpy.unravel_index(imax, val.shape)

        # swap elements in val array
        temp = xp.array(val[unravel_i], dtype=val.dtype, **val_kwargs)
        val[unravel_i] = val[unravel_imax]
        val[unravel_imax] = temp

        # swap corresponding columns in vec matrix
        temp = xp.array(vec[:, i], dtype=vec.dtype, **vec_kwargs)
        vec[:, i] = vec[:, imax]
        vec[:, imax] = temp


@pytest.mark.parametrize(
    "array",
    [
        [[[1, -2], [2, 5]]],
        [[[1.0, -2.0], [2.0, 5.0]]],
        [[[1.0, -2.0], [2.0, 5.0]], [[1.0, -2.0], [2.0, 5.0]]],
    ],
    ids=[
        "[[[1, -2], [2, 5]]]",
        "[[[1., -2.], [2., 5.]]]",
        "[[[1., -2.], [2., 5.]], [[1., -2.], [2., 5.]]]",
    ],
)
def test_cholesky(array):
    a = numpy.array(array)
    ia = inp.array(a)
    result = inp.linalg.cholesky(ia)
    expected = numpy.linalg.cholesky(a)
    assert_array_equal(expected, result)


@pytest.mark.parametrize(
    "shape",
    [
        (0, 0),
        (3, 0, 0),
    ],
    ids=[
        "(0, 0)",
        "(3, 0, 0)",
    ],
)
def test_cholesky_0D(shape):
    a = numpy.empty(shape)
    ia = inp.array(a)
    result = inp.linalg.cholesky(ia)
    expected = numpy.linalg.cholesky(a)
    assert_array_equal(expected, result)


@pytest.mark.parametrize(
    "arr",
    [[[1, 0, -1], [0, 1, 0], [1, 0, 1]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
    ids=[
        "[[1, 0, -1], [0, 1, 0], [1, 0, 1]]",
        "[[1, 2, 3], [4, 5, 6], [7, 8, 9]]",
    ],
)
@pytest.mark.parametrize(
    "p",
    [None, 1, -1, 2, -2, numpy.inf, -numpy.inf, "fro"],
    ids=["None", "1", "-1", "2", "-2", "numpy.inf", "-numpy.inf", '"fro"'],
)
def test_cond(arr, p):
    a = numpy.array(arr)
    ia = inp.array(a)
    result = inp.linalg.cond(ia, p)
    expected = numpy.linalg.cond(a, p)
    assert_array_equal(expected, result)


@pytest.mark.parametrize(
    "array",
    [
        [[0, 0], [0, 0]],
        [[1, 2], [1, 2]],
        [[1, 2], [3, 4]],
        [[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]],
        [
            [[[1, 2], [3, 4]], [[1, 2], [2, 1]]],
            [[[1, 3], [3, 1]], [[0, 1], [1, 3]]],
        ],
    ],
    ids=[
        "[[0, 0], [0, 0]]",
        "[[1, 2], [1, 2]]",
        "[[1, 2], [3, 4]]",
        "[[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]",
        "[[[[1, 2], [3, 4]], [[1, 2], [2, 1]]], [[[1, 3], [3, 1]], [[0, 1], [1, 3]]]]",
    ],
)
def test_det(array):
    a = numpy.array(array)
    ia = inp.array(a)
    result = inp.linalg.det(ia)
    expected = numpy.linalg.det(a)
    assert_allclose(expected, result)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
def test_det_empty():
    a = numpy.empty((0, 0, 2, 2), dtype=numpy.float32)
    ia = inp.array(a)

    np_det = numpy.linalg.det(a)
    dpnp_det = inp.linalg.det(ia)

    assert dpnp_det.dtype == np_det.dtype
    assert dpnp_det.shape == np_det.shape

    assert_allclose(np_det, dpnp_det)


@pytest.mark.parametrize("type", get_all_dtypes(no_bool=True, no_complex=True))
@pytest.mark.parametrize("size", [2, 4, 8, 16, 300])
def test_eig_arange(type, size):
    a = numpy.arange(size * size, dtype=type).reshape((size, size))
    symm_orig = (
        numpy.tril(a)
        + numpy.tril(a, -1).T
        + numpy.diag(numpy.full((size,), size * size, dtype=type))
    )
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

    assert_array_equal(symm_orig, symm)
    assert_array_equal(dpnp_symm_orig, dpnp_symm)

    if has_support_aspect64():
        assert dpnp_val.dtype == np_val.dtype
        assert dpnp_vec.dtype == np_vec.dtype
    assert dpnp_val.shape == np_val.shape
    assert dpnp_vec.shape == np_vec.shape
    assert dpnp_val.usm_type == dpnp_symm.usm_type
    assert dpnp_vec.usm_type == dpnp_symm.usm_type

    assert_allclose(dpnp_val, np_val, rtol=1e-05, atol=1e-05)
    assert_allclose(dpnp_vec, np_vec, rtol=1e-05, atol=1e-05)


@pytest.mark.parametrize("type", get_all_dtypes(no_bool=True, no_none=True))
@pytest.mark.parametrize("size", [2, 4, 8])
def test_eigh_arange(type, size):
    if dpctl.get_current_device_type() != dpctl.device_type.gpu:
        pytest.skip(
            "eig function doesn't work on CPU: https://github.com/IntelPython/dpnp/issues/1005"
        )
    a = numpy.arange(size * size, dtype=type).reshape((size, size))
    symm_orig = (
        numpy.tril(a)
        + numpy.tril(a, -1).T
        + numpy.diag(numpy.full((size,), size * size, dtype=type))
    )
    symm = symm_orig
    dpnp_symm_orig = inp.array(symm)
    dpnp_symm = dpnp_symm_orig

    dpnp_val, dpnp_vec = inp.linalg.eigh(dpnp_symm)
    np_val, np_vec = numpy.linalg.eigh(symm)

    # DPNP sort val/vec by abs value
    vvsort(dpnp_val, dpnp_vec, size, inp)

    # NP sort val/vec by abs value
    vvsort(np_val, np_vec, size, numpy)

    # NP change sign of vectors
    for i in range(np_vec.shape[1]):
        if (np_vec[0, i] * dpnp_vec[0, i]).asnumpy() < 0:
            np_vec[:, i] = -np_vec[:, i]

    assert_array_equal(symm_orig, symm)
    assert_array_equal(dpnp_symm_orig, dpnp_symm)

    assert dpnp_val.shape == np_val.shape
    assert dpnp_vec.shape == np_vec.shape
    assert dpnp_val.usm_type == dpnp_symm.usm_type
    assert dpnp_vec.usm_type == dpnp_symm.usm_type

    assert_allclose(dpnp_val, np_val, rtol=1e-05, atol=1e-04)
    assert_allclose(dpnp_vec, np_vec, rtol=1e-05, atol=1e-04)


@pytest.mark.parametrize("type", get_all_dtypes(no_bool=True, no_complex=True))
def test_eigvals(type):
    if dpctl.get_current_device_type() != dpctl.device_type.gpu:
        pytest.skip(
            "eigvals function doesn't work on CPU: https://github.com/IntelPython/dpnp/issues/1005"
        )
    arrays = [[[0, 0], [0, 0]], [[1, 2], [1, 2]], [[1, 2], [3, 4]]]
    for array in arrays:
        a = numpy.array(array, dtype=type)
        ia = inp.array(a)
        result = inp.linalg.eigvals(ia)
        expected = numpy.linalg.eigvals(a)
        assert_allclose(expected, result, atol=0.5)


@pytest.mark.parametrize("type", get_all_dtypes(no_bool=True, no_complex=True))
@pytest.mark.parametrize(
    "array",
    [[[1.0, 2.0], [3.0, 4.0]], [[0, 1, 2], [3, 2, -1], [4, -2, 3]]],
    ids=["[[1., 2.], [3., 4.]]", "[[0, 1, 2], [3, 2, -1], [4, -2, 3]]"],
)
def test_inv(type, array):
    a = numpy.array(array, dtype=type)
    ia = inp.array(a)
    result = inp.linalg.inv(ia)
    expected = numpy.linalg.inv(a)
    assert_allclose(expected, result, rtol=1e-06)


@pytest.mark.parametrize(
    "type", get_all_dtypes(no_bool=True, no_complex=True, no_none=True)
)
@pytest.mark.parametrize(
    "array",
    [
        [0, 0],
        [0, 1],
        [1, 2],
        [[0, 0], [0, 0]],
        [[1, 2], [1, 2]],
        [[1, 2], [3, 4]],
    ],
    ids=[
        "[0, 0]",
        "[0, 1]",
        "[1, 2]",
        "[[0, 0], [0, 0]]",
        "[[1, 2], [1, 2]]",
        "[[1, 2], [3, 4]]",
    ],
)
@pytest.mark.parametrize("tol", [None], ids=["None"])
def test_matrix_rank(type, tol, array):
    a = numpy.array(array, dtype=type)
    ia = inp.array(a)

    result = inp.linalg.matrix_rank(ia, tol=tol)
    expected = numpy.linalg.matrix_rank(a, tol=tol)

    assert_allclose(expected, result)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.usefixtures("suppress_divide_numpy_warnings")
@pytest.mark.parametrize(
    "array", [[7], [1, 2], [1, 0]], ids=["[7]", "[1, 2]", "[1, 0]"]
)
@pytest.mark.parametrize(
    "ord",
    [None, -inp.Inf, -2, -1, 0, 1, 2, 3, inp.Inf],
    ids=["None", "-dpnp.Inf", "-2", "-1", "0", "1", "2", "3", "dpnp.Inf"],
)
@pytest.mark.parametrize("axis", [0, None], ids=["0", "None"])
def test_norm1(array, ord, axis):
    a = numpy.array(array)
    ia = inp.array(a)
    result = inp.linalg.norm(ia, ord=ord, axis=axis)
    expected = numpy.linalg.norm(a, ord=ord, axis=axis)
    assert_allclose(expected, result)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize(
    "array",
    [[[1, 0]], [[1, 2]], [[1, 0], [3, 0]], [[1, 2], [3, 4]]],
    ids=["[[1, 0]]", "[[1, 2]]", "[[1, 0], [3, 0]]", "[[1, 2], [3, 4]]"],
)
@pytest.mark.parametrize(
    "ord",
    [None, -inp.Inf, -2, -1, 1, 2, inp.Inf, "fro", "nuc"],
    ids=[
        "None",
        "-dpnp.Inf",
        "-2",
        "-1",
        "1",
        "2",
        "dpnp.Inf",
        '"fro"',
        '"nuc"',
    ],
)
@pytest.mark.parametrize("axis", [(0, 1), None], ids=["(0, 1)", "None"])
def test_norm2(array, ord, axis):
    a = numpy.array(array)
    ia = inp.array(a)
    result = inp.linalg.norm(ia, ord=ord, axis=axis)
    expected = numpy.linalg.norm(a, ord=ord, axis=axis)
    assert_allclose(expected, result)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize(
    "array",
    [
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        [[[1, 0], [3, 0]], [[5, 0], [7, 0]]],
    ],
    ids=[
        "[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]",
        "[[[1, 0], [3, 0]], [[5, 0], [7, 0]]]",
    ],
)
@pytest.mark.parametrize(
    "ord",
    [None, -inp.Inf, -2, -1, 1, 2, inp.Inf],
    ids=["None", "-dpnp.Inf", "-2", "-1", "1", "2", "dpnp.Inf"],
)
@pytest.mark.parametrize(
    "axis",
    [0, 1, 2, (0, 1), (0, 2), (1, 2)],
    ids=["0", "1", "2", "(0, 1)", "(0, 2)", "(1, 2)"],
)
def test_norm3(array, ord, axis):
    a = numpy.array(array)
    ia = inp.array(a)
    result = inp.linalg.norm(ia, ord=ord, axis=axis)
    expected = numpy.linalg.norm(a, ord=ord, axis=axis)
    assert_allclose(expected, result)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("type", get_all_dtypes(no_bool=True, no_complex=True))
@pytest.mark.parametrize(
    "shape",
    [(2, 2), (3, 4), (5, 3), (16, 16), (0, 0), (0, 2), (2, 0)],
    ids=["(2,2)", "(3,4)", "(5,3)", "(16,16)", "(0,0)", "(0,2)", "(2,0)"],
)
@pytest.mark.parametrize(
    "mode", ["complete", "reduced"], ids=["complete", "reduced"]
)
def test_qr(type, shape, mode):
    a = numpy.arange(shape[0] * shape[1], dtype=type).reshape(shape)
    ia = inp.array(a)

    np_q, np_r = numpy.linalg.qr(a, mode)
    dpnp_q, dpnp_r = inp.linalg.qr(ia, mode)

    support_aspect64 = has_support_aspect64()

    if support_aspect64:
        assert dpnp_q.dtype == np_q.dtype
        assert dpnp_r.dtype == np_r.dtype
    assert dpnp_q.shape == np_q.shape
    assert dpnp_r.shape == np_r.shape

    tol = 1e-6
    if type == inp.float32:
        tol = 1e-02
    elif not support_aspect64 and type in (inp.int32, inp.int64, None):
        tol = 1e-02

    # check decomposition
    assert_allclose(
        ia,
        inp.dot(dpnp_q, dpnp_r),
        rtol=tol,
        atol=tol,
    )

    # NP change sign for comparison
    ncols = min(a.shape[0], a.shape[1])
    for i in range(ncols):
        j = numpy.where(numpy.abs(np_q[:, i]) > tol)[0][0]
        if np_q[j, i] * dpnp_q[j, i] < 0:
            np_q[:, i] = -np_q[:, i]
            np_r[i, :] = -np_r[i, :]

        if numpy.any(numpy.abs(np_r[i, :]) > tol):
            assert_allclose(
                inp.asnumpy(dpnp_q)[:, i], np_q[:, i], rtol=tol, atol=tol
            )

    assert_allclose(dpnp_r, np_r, rtol=tol, atol=tol)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
def test_qr_not_2D():
    a = numpy.arange(12, dtype=numpy.float32).reshape((3, 2, 2))
    ia = inp.array(a)

    np_q, np_r = numpy.linalg.qr(a)
    dpnp_q, dpnp_r = inp.linalg.qr(ia)

    assert dpnp_q.dtype == np_q.dtype
    assert dpnp_r.dtype == np_r.dtype
    assert dpnp_q.shape == np_q.shape
    assert dpnp_r.shape == np_r.shape

    assert_allclose(ia, inp.matmul(dpnp_q, dpnp_r))

    a = numpy.empty((0, 3, 2), dtype=numpy.float32)
    ia = inp.array(a)

    np_q, np_r = numpy.linalg.qr(a)
    dpnp_q, dpnp_r = inp.linalg.qr(ia)

    assert dpnp_q.dtype == np_q.dtype
    assert dpnp_r.dtype == np_r.dtype
    assert dpnp_q.shape == np_q.shape
    assert dpnp_r.shape == np_r.shape

    assert_allclose(ia, inp.matmul(dpnp_q, dpnp_r))


class TestSolve:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_solve(self, dtype):
        a_np = numpy.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        a_dp = inp.array(a_np)

        expected = numpy.linalg.solve(a_np, a_np)
        result = inp.linalg.solve(a_dp, a_dp)

        assert_allclose(expected, result, rtol=1e-06)

    @pytest.mark.parametrize("a_dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize("b_dtype", get_all_dtypes(no_bool=True))
    def test_solve_diff_type(self, a_dtype, b_dtype):
        a_np = numpy.array([[1, 2], [3, -5]], dtype=a_dtype)
        b_np = numpy.array([4, 1], dtype=b_dtype)

        a_dp = inp.array(a_np)
        b_dp = inp.array(b_np)

        expected = numpy.linalg.solve(a_np, b_np)
        result = inp.linalg.solve(a_dp, b_dp)

        assert_dtype_allclose(result, expected)

    def test_solve_strides(self):
        a_np = numpy.array(
            [
                [2, 3, 1, 4, 5],
                [5, 6, 7, 8, 9],
                [9, 7, 7, 2, 3],
                [1, 4, 5, 1, 8],
                [8, 9, 8, 5, 3],
            ]
        )
        b_np = numpy.array([5, 8, 9, 2, 1])

        a_dp = inp.array(a_np)
        b_dp = inp.array(b_np)

        # positive strides
        expected = numpy.linalg.solve(a_np[::2, ::2], b_np[::2])
        result = inp.linalg.solve(a_dp[::2, ::2], b_dp[::2])
        assert_allclose(expected, result, rtol=1e-05)

        # negative strides
        expected = numpy.linalg.solve(a_np[::-2, ::-2], b_np[::-2])
        result = inp.linalg.solve(a_dp[::-2, ::-2], b_dp[::-2])
        assert_allclose(expected, result, rtol=1e-05)

    # TODO: remove skipif when MKLD-16626 is resolved
    @pytest.mark.skipif(is_cpu_device(), reason="MKLD-16626")
    @pytest.mark.parametrize(
        "matrix, vector",
        [
            ([[1, 2], [2, 4]], [1, 2]),
            ([[0, 0], [0, 0]], [0, 0]),
            ([[1, 1], [1, 1]], [2, 2]),
            ([[2, 4], [1, 2]], [3, 1.5]),
            ([[1, 2], [0, 0]], [3, 0]),
            ([[1, 0], [2, 0]], [3, 4]),
        ],
        ids=[
            "Linearly dependent rows",
            "Zero matrix",
            "Identical rows",
            "Linearly dependent columns",
            "Zero row",
            "Zero column",
        ],
    )
    def test_solve_singular_matrix(self, matrix, vector):
        a_np = numpy.array(matrix, dtype="float32")
        b_np = numpy.array(vector, dtype="float32")

        a_dp = inp.array(a_np)
        b_dp = inp.array(b_np)

        assert_raises(numpy.linalg.LinAlgError, numpy.linalg.solve, a_np, b_np)
        assert_raises(inp.linalg.LinAlgError, inp.linalg.solve, a_dp, b_dp)

    def test_solve_errors(self):
        a_dp = inp.array([[1, 0.5], [0.5, 1]], dtype="float32")
        b_dp = inp.array(a_dp, dtype="float32")

        # diffetent queue
        a_queue = dpctl.SyclQueue()
        b_queue = dpctl.SyclQueue()
        a_dp_q = inp.array(a_dp, sycl_queue=a_queue)
        b_dp_q = inp.array(b_dp, sycl_queue=b_queue)
        assert_raises(ValueError, inp.linalg.solve, a_dp_q, b_dp_q)

        # unsupported type
        a_np = inp.asnumpy(a_dp)
        b_np = inp.asnumpy(b_dp)
        assert_raises(TypeError, inp.linalg.solve, a_np, b_dp)
        assert_raises(TypeError, inp.linalg.solve, a_dp, b_np)

        # a.ndim < 2
        a_dp_ndim_1 = a_dp.flatten()
        assert_raises(
            inp.linalg.LinAlgError, inp.linalg.solve, a_dp_ndim_1, b_dp
        )


class TestSvd:
    def get_tol(self, dtype):
        tol = 1e-06
        if dtype in (inp.float32, inp.complex64):
            tol = 1e-04
        elif not has_support_aspect64() and dtype in (
            inp.int32,
            inp.int64,
            None,
        ):
            tol = 1e-05
        return tol

    def check_types_shapes(
        self, dp_u, dp_s, dp_vt, np_u, np_s, np_vt, compute_vt=True
    ):
        if has_support_aspect64():
            if compute_vt:
                assert dp_u.dtype == np_u.dtype
                assert dp_vt.dtype == np_vt.dtype
            assert dp_s.dtype == np_s.dtype
        else:
            if compute_vt:
                assert dp_u.dtype.kind == np_u.dtype.kind
                assert dp_vt.dtype.kind == np_vt.dtype.kind
            assert dp_s.dtype.kind == np_s.dtype.kind

        if compute_vt:
            assert dp_u.shape == np_u.shape
            assert dp_vt.shape == np_vt.shape
        assert dp_s.shape == np_s.shape

    # Checks the accuracy of singular value decomposition (SVD).
    # Compares the reconstructed matrix from the decomposed components
    # with the original matrix.
    # Additionally checks for equality of singular values
    # between dpnp and numpy decompositions
    def check_decomposition(
        self, dp_a, dp_u, dp_s, dp_vt, np_u, np_s, np_vt, compute_vt, tol
    ):
        if compute_vt:
            dpnp_diag_s = inp.zeros_like(dp_a, dtype=dp_s.dtype)
            for i in range(min(dp_a.shape[-2], dp_a.shape[-1])):
                dpnp_diag_s[..., i, i] = dp_s[..., i]
            # TODO: remove it when dpnp.dot is updated
            # dpnp.dot does not support complex type
            if inp.issubdtype(dp_a.dtype, inp.complexfloating):
                reconstructed = numpy.dot(
                    inp.asnumpy(dp_u),
                    numpy.dot(inp.asnumpy(dpnp_diag_s), inp.asnumpy(dp_vt)),
                )
            else:
                reconstructed = inp.dot(dp_u, inp.dot(dpnp_diag_s, dp_vt))
            assert_allclose(dp_a, reconstructed, rtol=tol, atol=tol)

        assert_allclose(dp_s, np_s, rtol=tol, atol=1e-03)

        if compute_vt:
            for i in range(min(dp_a.shape[-2], dp_a.shape[-1])):
                if np_u[..., 0, i] * dp_u[..., 0, i] < 0:
                    np_u[..., :, i] = -np_u[..., :, i]
                    np_vt[..., i, :] = -np_vt[..., i, :]
            for i in range(numpy.count_nonzero(np_s > tol)):
                assert_allclose(
                    inp.asnumpy(dp_u[..., :, i]),
                    np_u[..., :, i],
                    rtol=tol,
                    atol=tol,
                )
                assert_allclose(
                    inp.asnumpy(dp_vt[..., i, :]),
                    np_vt[..., i, :],
                    rtol=tol,
                    atol=tol,
                )

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "shape",
        [(2, 2), (3, 4), (5, 3), (16, 16)],
        ids=["(2,2)", "(3,4)", "(5,3)", "(16,16)"],
    )
    def test_svd(self, dtype, shape):
        a = numpy.arange(shape[0] * shape[1], dtype=dtype).reshape(shape)
        dp_a = inp.array(a)

        np_u, np_s, np_vt = numpy.linalg.svd(a)
        dp_u, dp_s, dp_vt = inp.linalg.svd(dp_a)

        self.check_types_shapes(dp_u, dp_s, dp_vt, np_u, np_s, np_vt)
        tol = self.get_tol(dtype)
        self.check_decomposition(
            dp_a, dp_u, dp_s, dp_vt, np_u, np_s, np_vt, True, tol
        )

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize("compute_vt", [True, False], ids=["True", "False"])
    @pytest.mark.parametrize(
        "shape",
        [(2, 2), (16, 16)],
        ids=["(2,2)", "(16, 16)"],
    )
    def test_svd_hermitian(self, dtype, compute_vt, shape):
        a = numpy.random.randn(*shape) + 1j * numpy.random.randn(*shape)
        a = numpy.conj(a.T) @ a

        a = a.astype(dtype)
        dp_a = inp.array(a)

        if compute_vt:
            np_u, np_s, np_vt = numpy.linalg.svd(
                a, compute_uv=compute_vt, hermitian=True
            )
            dp_u, dp_s, dp_vt = inp.linalg.svd(
                dp_a, compute_uv=compute_vt, hermitian=True
            )
        else:
            np_s = numpy.linalg.svd(a, compute_uv=compute_vt, hermitian=True)
            dp_s = inp.linalg.svd(dp_a, compute_uv=compute_vt, hermitian=True)
            np_u = np_vt = dp_u = dp_vt = None

        self.check_types_shapes(
            dp_u, dp_s, dp_vt, np_u, np_s, np_vt, compute_vt
        )
        tol = self.get_tol(dtype)
        self.check_decomposition(
            dp_a, dp_u, dp_s, dp_vt, np_u, np_s, np_vt, compute_vt, tol
        )

    def test_svd_errors(self):
        a_dp = inp.array([[1, 2], [3, 4]], dtype="float32")

        # unsupported type
        a_np = inp.asnumpy(a_dp)
        assert_raises(TypeError, inp.linalg.svd, a_np)

        # a.ndim < 2
        a_dp_ndim_1 = a_dp.flatten()
        assert_raises(inp.linalg.LinAlgError, inp.linalg.svd, a_dp_ndim_1)
