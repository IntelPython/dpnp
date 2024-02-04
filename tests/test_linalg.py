import dpctl
import numpy
import pytest
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    assert_raises,
)

import dpnp as inp
from tests.third_party.cupy import testing

from .helper import (
    assert_dtype_allclose,
    get_all_dtypes,
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


class TestCholesky:
    @pytest.mark.parametrize(
        "array",
        [
            [[1, 2], [2, 5]],
            [[[5, 2], [2, 6]], [[7, 3], [3, 8]], [[3, 1], [1, 4]]],
            [
                [[[5, 2], [2, 5]], [[6, 3], [3, 6]]],
                [[[7, 2], [2, 7]], [[8, 3], [3, 8]]],
            ],
        ],
        ids=[
            "2D_array",
            "3D_array",
            "4D_array",
        ],
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_cholesky(self, array, dtype):
        a = numpy.array(array, dtype=dtype)
        ia = inp.array(a)
        result = inp.linalg.cholesky(ia)
        expected = numpy.linalg.cholesky(a)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "array",
        [
            [[1, 2], [2, 5]],
            [[[5, 2], [2, 6]], [[7, 3], [3, 8]], [[3, 1], [1, 4]]],
            [
                [[[5, 2], [2, 5]], [[6, 3], [3, 6]]],
                [[[7, 2], [2, 7]], [[8, 3], [3, 8]]],
            ],
        ],
        ids=[
            "2D_array",
            "3D_array",
            "4D_array",
        ],
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_cholesky_upper(self, array, dtype):
        ia = inp.array(array, dtype=dtype)
        result = inp.linalg.cholesky(ia, upper=True)

        if ia.ndim > 2:
            n = ia.shape[-1]
            ia_reshaped = ia.reshape(-1, n, n)
            res_reshaped = result.reshape(-1, n, n)
            batch_size = ia_reshaped.shape[0]
            for idx in range(batch_size):
                # Reconstruct the matrix using the Cholesky decomposition result
                if inp.issubdtype(dtype, inp.complexfloating):
                    reconstructed = (
                        res_reshaped[idx].T.conj() @ res_reshaped[idx]
                    )
                else:
                    reconstructed = res_reshaped[idx].T @ res_reshaped[idx]
                assert_dtype_allclose(
                    reconstructed, ia_reshaped[idx], check_type=False
                )
        else:
            # Reconstruct the matrix using the Cholesky decomposition result
            if inp.issubdtype(dtype, inp.complexfloating):
                reconstructed = result.T.conj() @ result
            else:
                reconstructed = result.T @ result
            assert_dtype_allclose(reconstructed, ia, check_type=False)

    # upper parameter support will be added in numpy 2.0 version
    @testing.with_requires("numpy>=2.0")
    @pytest.mark.parametrize(
        "array",
        [
            [[1, 2], [2, 5]],
            [[[5, 2], [2, 6]], [[7, 3], [3, 8]], [[3, 1], [1, 4]]],
            [
                [[[5, 2], [2, 5]], [[6, 3], [3, 6]]],
                [[[7, 2], [2, 7]], [[8, 3], [3, 8]]],
            ],
        ],
        ids=[
            "2D_array",
            "3D_array",
            "4D_array",
        ],
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_cholesky_upper_numpy(self, array, dtype):
        a = numpy.array(array, dtype=dtype)
        ia = inp.array(a)
        result = inp.linalg.cholesky(ia, upper=True)
        expected = numpy.linalg.cholesky(a, upper=True)
        assert_dtype_allclose(result, expected)

    def test_cholesky_strides(self):
        a_np = numpy.array(
            [
                [5, 2, 0, 0, 1],
                [2, 6, 0, 0, 2],
                [0, 0, 7, 0, 0],
                [0, 0, 0, 4, 0],
                [1, 2, 0, 0, 5],
            ]
        )

        a_dp = inp.array(a_np)

        # positive strides
        expected = numpy.linalg.cholesky(a_np[::2, ::2])
        result = inp.linalg.cholesky(a_dp[::2, ::2])
        assert_allclose(expected, result, rtol=1e-3, atol=1e-4)

        # negative strides
        expected = numpy.linalg.cholesky(a_np[::-2, ::-2])
        result = inp.linalg.cholesky(a_dp[::-2, ::-2])
        assert_allclose(expected, result, rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize(
        "shape",
        [
            (0, 0),
            (3, 0, 0),
            (0, 2, 2),
        ],
        ids=[
            "(0, 0)",
            "(3, 0, 0)",
            "(0, 2, 2)",
        ],
    )
    def test_cholesky_empty(self, shape):
        a = numpy.empty(shape)
        ia = inp.array(a)
        result = inp.linalg.cholesky(ia)
        expected = numpy.linalg.cholesky(a)
        assert_array_equal(expected, result)

    def test_cholesky_errors(self):
        a_dp = inp.array([[1, 2], [2, 5]], dtype="float32")

        # unsupported type
        a_np = inp.asnumpy(a_dp)
        assert_raises(TypeError, inp.linalg.cholesky, a_np)

        # a.ndim < 2
        a_dp_ndim_1 = a_dp.flatten()
        assert_raises(inp.linalg.LinAlgError, inp.linalg.cholesky, a_dp_ndim_1)

        # a is not square
        a_dp = inp.ones((2, 3))
        assert_raises(inp.linalg.LinAlgError, inp.linalg.cholesky, a_dp)


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


class TestDet:
    # TODO: Remove the use of fixture for test_det
    # when dpnp.prod() will support complex dtypes on Gen9
    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize(
        "array",
        [
            [[1, 2], [3, 4]],
            [[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]],
            [
                [[[1, 2], [3, 4]], [[1, 2], [2, 1]]],
                [[[1, 3], [3, 1]], [[0, 1], [1, 3]]],
            ],
        ],
        ids=[
            "2D_array",
            "3D_array",
            "4D_array",
        ],
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_det(self, array, dtype):
        a = numpy.array(array, dtype=dtype)
        ia = inp.array(a)
        result = inp.linalg.det(ia)
        expected = numpy.linalg.det(a)
        assert_allclose(expected, result)

    def test_det_strides(self):
        a_np = numpy.array(
            [
                [2, 3, 1, 4, 5],
                [5, 6, 7, 8, 9],
                [9, 7, 7, 2, 3],
                [1, 4, 5, 1, 8],
                [8, 9, 8, 5, 3],
            ]
        )

        a_dp = inp.array(a_np)

        # positive strides
        expected = numpy.linalg.det(a_np[::2, ::2])
        result = inp.linalg.det(a_dp[::2, ::2])
        assert_allclose(expected, result, rtol=1e-3, atol=1e-4)

        # negative strides
        expected = numpy.linalg.det(a_np[::-2, ::-2])
        result = inp.linalg.det(a_dp[::-2, ::-2])
        assert_allclose(expected, result, rtol=1e-3, atol=1e-4)

    def test_det_empty(self):
        a = numpy.empty((0, 0, 2, 2), dtype=numpy.float32)
        ia = inp.array(a)

        np_det = numpy.linalg.det(a)
        dpnp_det = inp.linalg.det(ia)

        assert dpnp_det.dtype == np_det.dtype
        assert dpnp_det.shape == np_det.shape

        assert_allclose(np_det, dpnp_det)

    @pytest.mark.parametrize(
        "matrix",
        [
            [[1, 2], [2, 4]],
            [[0, 0], [0, 0]],
            [[1, 1], [1, 1]],
            [[2, 4], [1, 2]],
            [[1, 2], [0, 0]],
            [[1, 0], [2, 0]],
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
    def test_det_singular_matrix(self, matrix):
        a_np = numpy.array(matrix, dtype="float32")
        a_dp = inp.array(a_np)

        expected = numpy.linalg.det(a_np)
        result = inp.linalg.det(a_dp)

        assert_allclose(expected, result, rtol=1e-3, atol=1e-4)

    # TODO: remove skipif when MKLD-16626 is resolved
    # _getrf_batch does not raise an error with singular matrices.
    # Skip running on cpu because dpnp uses _getrf_batch only on cpu.
    @pytest.mark.skipif(is_cpu_device(), reason="MKLD-16626")
    def test_det_singular_matrix_3D(self):
        a_np = numpy.array(
            [[[1, 2], [3, 4]], [[1, 2], [1, 2]], [[1, 3], [3, 1]]]
        )
        a_dp = inp.array(a_np)

        expected = numpy.linalg.det(a_np)
        result = inp.linalg.det(a_dp)

        assert_allclose(expected, result, rtol=1e-3, atol=1e-4)

    def test_det_errors(self):
        a_dp = inp.array([[1, 2], [3, 5]], dtype="float32")

        # unsupported type
        a_np = inp.asnumpy(a_dp)
        assert_raises(TypeError, inp.linalg.det, a_np)


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


class TestInv:
    @pytest.mark.parametrize(
        "array",
        [
            [[1, 2], [3, 4]],
            [[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]],
            [
                [[[1, 2], [3, 4]], [[1, 2], [2, 1]]],
                [[[1, 3], [3, 1]], [[0, 1], [1, 3]]],
            ],
        ],
        ids=[
            "2D_array",
            "3D_array",
            "4D_array",
        ],
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_inv(self, array, dtype):
        a = numpy.array(array, dtype=dtype)
        ia = inp.array(a)
        result = inp.linalg.inv(ia)
        expected = numpy.linalg.inv(a)
        assert_dtype_allclose(result, expected)

    def test_inv_strides(self):
        a_np = numpy.array(
            [
                [2, 3, 1, 4, 5],
                [5, 6, 7, 8, 9],
                [9, 7, 7, 2, 3],
                [1, 4, 5, 1, 8],
                [8, 9, 8, 5, 3],
            ]
        )

        a_dp = inp.array(a_np)

        # positive strides
        expected = numpy.linalg.inv(a_np[::2, ::2])
        result = inp.linalg.inv(a_dp[::2, ::2])
        assert_allclose(expected, result, rtol=1e-3, atol=1e-4)

        # negative strides
        expected = numpy.linalg.inv(a_np[::-2, ::-2])
        result = inp.linalg.inv(a_dp[::-2, ::-2])
        assert_allclose(expected, result, rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize(
        "shape",
        [
            (0, 0),
            (3, 0, 0),
            (0, 2, 2),
        ],
        ids=[
            "(0, 0)",
            "(3, 0, 0)",
            "(0, 2, 2)",
        ],
    )
    def test_inv_empty(self, shape):
        a = numpy.empty(shape)
        ia = inp.array(a)
        result = inp.linalg.inv(ia)
        expected = numpy.linalg.inv(a)
        assert_dtype_allclose(result, expected)

    # TODO: remove skipif when MKLD-16626 is resolved
    @pytest.mark.skipif(is_cpu_device(), reason="MKLD-16626")
    @pytest.mark.parametrize(
        "matrix",
        [
            [[1, 2], [2, 4]],
            [[0, 0], [0, 0]],
            [[1, 1], [1, 1]],
            [[2, 4], [1, 2]],
            [[1, 2], [0, 0]],
            [[1, 0], [2, 0]],
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
    def test_inv_singular_matrix(self, matrix):
        a_np = numpy.array(matrix, dtype="float32")
        a_dp = inp.array(a_np)

        assert_raises(numpy.linalg.LinAlgError, numpy.linalg.inv, a_np)
        assert_raises(inp.linalg.LinAlgError, inp.linalg.inv, a_dp)

    # TODO: remove skipif when MKLD-16626 is resolved
    # _getrf_batch does not raise an error with singular matrices.
    @pytest.mark.skip("MKLD-16626")
    def test_inv_singular_matrix_3D(self):
        a_np = numpy.array(
            [[[1, 2], [3, 4]], [[1, 2], [1, 2]], [[1, 3], [3, 1]]]
        )
        a_dp = inp.array(a_np)

        assert_raises(numpy.linalg.LinAlgError, numpy.linalg.inv, a_np)
        assert_raises(inp.linalg.LinAlgError, inp.linalg.inv, a_dp)

    def test_inv_errors(self):
        a_dp = inp.array([[1, 2], [2, 5]], dtype="float32")

        # unsupported type
        a_np = inp.asnumpy(a_dp)
        assert_raises(TypeError, inp.linalg.inv, a_np)

        # a.ndim < 2
        a_dp_ndim_1 = a_dp.flatten()
        assert_raises(inp.linalg.LinAlgError, inp.linalg.inv, a_dp_ndim_1)

        # a is not square
        a_dp = inp.ones((2, 3))
        assert_raises(inp.linalg.LinAlgError, inp.linalg.inv, a_dp)


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


class TestQr:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "shape",
        [(2, 2), (3, 4), (5, 3), (16, 16), (2, 2, 2), (2, 4, 2), (2, 2, 4)],
        ids=[
            "(2, 2)",
            "(3, 4)",
            "(5, 3)",
            "(16, 16)",
            "(2, 2, 2)",
            "(2, 4, 2)",
            "(2, 2, 4)",
        ],
    )
    @pytest.mark.parametrize(
        "mode",
        ["r", "raw", "complete", "reduced"],
        ids=["r", "raw", "complete", "reduced"],
    )
    def test_qr(self, dtype, shape, mode):
        a = numpy.random.rand(*shape).astype(dtype)
        ia = inp.array(a)

        if mode == "r":
            np_r = numpy.linalg.qr(a, mode)
            dpnp_r = inp.linalg.qr(ia, mode)
        else:
            np_q, np_r = numpy.linalg.qr(a, mode)
            dpnp_q, dpnp_r = inp.linalg.qr(ia, mode)

            # check decomposition
            if mode in ("complete", "reduced"):
                # _orgqr doesn`t support complex type
                if not inp.issubdtype(dtype, inp.complexfloating):
                    if a.ndim == 2:
                        assert_almost_equal(
                            inp.dot(dpnp_q, dpnp_r),
                            a,
                            decimal=5,
                        )
                    else:
                        batch_size = a.shape[0]
                        for i in range(batch_size):
                            assert_almost_equal(
                                inp.dot(dpnp_q[i], dpnp_r[i]),
                                a[i],
                                decimal=5,
                            )
            else:  # mode=="raw"
                # _orgqr doesn`t support complex type
                if not inp.issubdtype(dtype, inp.complexfloating):
                    assert_dtype_allclose(dpnp_q, np_q)
                    assert_dtype_allclose(dpnp_r, np_r)
        if mode in ("raw", "r"):
            assert_dtype_allclose(dpnp_r, np_r)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "shape",
        [(0, 0), (0, 2), (2, 0), (2, 0, 3), (2, 3, 0), (0, 2, 3)],
        ids=[
            "(0, 0)",
            "(0, 2)",
            "(2 ,0)",
            "(2, 0, 3)",
            "(2, 3, 0)",
            "(0, 2, 3)",
        ],
    )
    @pytest.mark.parametrize(
        "mode",
        ["r", "raw", "complete", "reduced"],
        ids=["r", "raw", "complete", "reduced"],
    )
    def test_qr_empty(self, dtype, shape, mode):
        a = numpy.empty(shape, dtype=dtype)
        ia = inp.array(a)

        if mode == "r":
            np_r = numpy.linalg.qr(a, mode)
            dpnp_r = inp.linalg.qr(ia, mode)
        else:
            np_q, np_r = numpy.linalg.qr(a, mode)
            dpnp_q, dpnp_r = inp.linalg.qr(ia, mode)

            assert_dtype_allclose(dpnp_q, np_q)

        assert_dtype_allclose(dpnp_r, np_r)

    @pytest.mark.parametrize(
        "mode",
        ["r", "raw", "complete", "reduced"],
        ids=["r", "raw", "complete", "reduced"],
    )
    def test_qr_strides(self, mode):
        a = numpy.random.rand(5, 5)
        ia = inp.array(a)

        # positive strides
        if mode == "r":
            np_r = numpy.linalg.qr(a[::2, ::2], mode)
            dpnp_r = inp.linalg.qr(ia[::2, ::2], mode)
        else:
            np_q, np_r = numpy.linalg.qr(a[::2, ::2], mode)
            dpnp_q, dpnp_r = inp.linalg.qr(ia[::2, ::2], mode)

            assert_dtype_allclose(dpnp_q, np_q)

        assert_dtype_allclose(dpnp_r, np_r)

        # negative strides
        if mode == "r":
            np_r = numpy.linalg.qr(a[::-2, ::-2], mode)
            dpnp_r = inp.linalg.qr(ia[::-2, ::-2], mode)
        else:
            np_q, np_r = numpy.linalg.qr(a[::-2, ::-2], mode)
            dpnp_q, dpnp_r = inp.linalg.qr(ia[::-2, ::-2], mode)

            assert_dtype_allclose(dpnp_q, np_q)

        assert_dtype_allclose(dpnp_r, np_r)

    def test_qr_errors(self):
        a_dp = inp.array([[1, 2], [3, 5]], dtype="float32")

        # unsupported type
        a_np = inp.asnumpy(a_dp)
        assert_raises(TypeError, inp.linalg.qr, a_np)

        # a.ndim < 2
        a_dp_ndim_1 = a_dp.flatten()
        assert_raises(inp.linalg.LinAlgError, inp.linalg.qr, a_dp_ndim_1)

        # invalid mode
        assert_raises(ValueError, inp.linalg.qr, a_dp, "c")


@pytest.mark.parametrize("type", get_all_dtypes(no_bool=True, no_complex=True))
@pytest.mark.parametrize(
    "shape",
    [(2, 2), (3, 4), (5, 3), (16, 16)],
    ids=["(2,2)", "(3,4)", "(5,3)", "(16,16)"],
)
def test_svd(type, shape):
    a = numpy.arange(shape[0] * shape[1], dtype=type).reshape(shape)
    ia = inp.array(a)

    np_u, np_s, np_vt = numpy.linalg.svd(a)
    dpnp_u, dpnp_s, dpnp_vt = inp.linalg.svd(ia)

    support_aspect64 = has_support_aspect64()

    if support_aspect64:
        assert dpnp_u.dtype == np_u.dtype
        assert dpnp_s.dtype == np_s.dtype
        assert dpnp_vt.dtype == np_vt.dtype
    assert dpnp_u.shape == np_u.shape
    assert dpnp_s.shape == np_s.shape
    assert dpnp_vt.shape == np_vt.shape

    tol = 1e-12
    if type == inp.float32:
        tol = 1e-03
    elif not support_aspect64 and type in (inp.int32, inp.int64, None):
        tol = 1e-03

    # check decomposition
    dpnp_diag_s = inp.zeros(shape, dtype=dpnp_s.dtype)
    for i in range(dpnp_s.size):
        dpnp_diag_s[i, i] = dpnp_s[i]

    # check decomposition
    assert_allclose(
        ia, inp.dot(dpnp_u, inp.dot(dpnp_diag_s, dpnp_vt)), rtol=tol, atol=tol
    )

    # compare singular values
    # assert_allclose(dpnp_s, np_s, rtol=tol, atol=tol)

    # change sign of vectors
    for i in range(min(shape[0], shape[1])):
        if np_u[0, i] * dpnp_u[0, i] < 0:
            np_u[:, i] = -np_u[:, i]
            np_vt[i, :] = -np_vt[i, :]

    # compare vectors for non-zero values
    for i in range(numpy.count_nonzero(np_s > tol)):
        assert_allclose(
            inp.asnumpy(dpnp_u)[:, i], np_u[:, i], rtol=tol, atol=tol
        )
        assert_allclose(
            inp.asnumpy(dpnp_vt)[i, :], np_vt[i, :], rtol=tol, atol=tol
        )


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


class TestSlogdet:
    # TODO: Remove the use of fixture for test_slogdet_2d and test_slogdet_3d
    # when dpnp.prod() will support complex dtypes on Gen9
    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_slogdet_2d(self, dtype):
        a_np = numpy.array([[1, 2], [3, 4]], dtype=dtype)
        a_dp = inp.array(a_np)

        sign_expected, logdet_expected = numpy.linalg.slogdet(a_np)
        sign_result, logdet_result = inp.linalg.slogdet(a_dp)

        assert_allclose(sign_expected, sign_result)
        assert_allclose(logdet_expected, logdet_result, rtol=1e-3, atol=1e-4)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_slogdet_3d(self, dtype):
        a_np = numpy.array(
            [
                [[1, 2], [3, 4]],
                [[1, 2], [2, 1]],
                [[1, 3], [3, 1]],
            ],
            dtype=dtype,
        )
        a_dp = inp.array(a_np)

        sign_expected, logdet_expected = numpy.linalg.slogdet(a_np)
        sign_result, logdet_result = inp.linalg.slogdet(a_dp)

        assert_allclose(sign_expected, sign_result)
        assert_allclose(logdet_expected, logdet_result, rtol=1e-3, atol=1e-4)

    def test_slogdet_strides(self):
        a_np = numpy.array(
            [
                [2, 3, 1, 4, 5],
                [5, 6, 7, 8, 9],
                [9, 7, 7, 2, 3],
                [1, 4, 5, 1, 8],
                [8, 9, 8, 5, 3],
            ]
        )

        a_dp = inp.array(a_np)

        # positive strides
        sign_expected, logdet_expected = numpy.linalg.slogdet(a_np[::2, ::2])
        sign_result, logdet_result = inp.linalg.slogdet(a_dp[::2, ::2])
        assert_allclose(sign_expected, sign_result)
        assert_allclose(logdet_expected, logdet_result, rtol=1e-3, atol=1e-4)

        # negative strides
        sign_expected, logdet_expected = numpy.linalg.slogdet(a_np[::-2, ::-2])
        sign_result, logdet_result = inp.linalg.slogdet(a_dp[::-2, ::-2])
        assert_allclose(sign_expected, sign_result)
        assert_allclose(logdet_expected, logdet_result, rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize(
        "matrix",
        [
            [[1, 2], [2, 4]],
            [[0, 0], [0, 0]],
            [[1, 1], [1, 1]],
            [[2, 4], [1, 2]],
            [[1, 2], [0, 0]],
            [[1, 0], [2, 0]],
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
    def test_slogdet_singular_matrix(self, matrix):
        a_np = numpy.array(matrix, dtype="float32")
        a_dp = inp.array(a_np)

        sign_expected, logdet_expected = numpy.linalg.slogdet(a_np)
        sign_result, logdet_result = inp.linalg.slogdet(a_dp)

        assert_allclose(sign_expected, sign_result)
        assert_allclose(logdet_expected, logdet_result, rtol=1e-3, atol=1e-4)

    # TODO: remove skipif when MKLD-16626 is resolved
    # _getrf_batch does not raise an error with singular matrices.
    # Skip running on cpu because dpnp uses _getrf_batch only on cpu.
    @pytest.mark.skipif(is_cpu_device(), reason="MKLD-16626")
    def test_slogdet_singular_matrix_3D(self):
        a_np = numpy.array(
            [[[1, 2], [3, 4]], [[1, 2], [1, 2]], [[1, 3], [3, 1]]]
        )
        a_dp = inp.array(a_np)

        sign_expected, logdet_expected = numpy.linalg.slogdet(a_np)
        sign_result, logdet_result = inp.linalg.slogdet(a_dp)

        assert_allclose(sign_expected, sign_result)
        assert_allclose(logdet_expected, logdet_result, rtol=1e-3, atol=1e-4)

    def test_slogdet_errors(self):
        a_dp = inp.array([[1, 2], [3, 5]], dtype="float32")

        # unsupported type
        a_np = inp.asnumpy(a_dp)
        assert_raises(TypeError, inp.linalg.slogdet, a_np)
