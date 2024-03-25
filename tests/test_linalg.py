import dpctl
import numpy
import pytest
from dpctl.utils import ExecutionPlacementError
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
    generate_random_numpy_array,
    get_all_dtypes,
    get_float_complex_dtypes,
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


class TestEigenvalue:
    @pytest.mark.parametrize(
        "func",
        [
            "eigh",
            "eigvalsh",
        ],
    )
    @pytest.mark.parametrize(
        "shape",
        [(2, 2), (2, 3, 3), (2, 2, 3, 3)],
        ids=["(2,2)", "(2,3,3)", "(2,2,3,3)"],
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "order",
        [
            "C",
            "F",
        ],
    )
    def test_eigenvalues(self, func, shape, dtype, order):
        a = generate_random_numpy_array(
            shape, dtype, hermitian=True, seed_value=81
        )
        a_order = numpy.array(a, order=order)
        a_dp = inp.array(a, order=order)

        if func == "eigh":
            w, v = numpy.linalg.eigh(a_order)
            w_dp, v_dp = inp.linalg.eigh(a_dp)

            assert_dtype_allclose(v_dp, v)

        else:  # eighvalsh
            w = numpy.linalg.eigvalsh(a_order)
            w_dp = inp.linalg.eigvalsh(a_dp)

        assert_dtype_allclose(w_dp, w)

    @pytest.mark.parametrize(
        "func",
        [
            "eigh",
            "eigvalsh",
        ],
    )
    def test_eigenvalue_errors(self, func):
        a_dp = inp.array([[1, 3], [3, 2]], dtype="float32")

        # unsupported type
        a_np = inp.asnumpy(a_dp)
        dpnp_func = getattr(inp.linalg, func)
        assert_raises(TypeError, dpnp_func, a_np)

        # a.ndim < 2
        a_dp_ndim_1 = a_dp.flatten()
        assert_raises(inp.linalg.LinAlgError, dpnp_func, a_dp_ndim_1)

        # a is not square
        a_dp_not_scquare = inp.ones((2, 3))
        assert_raises(inp.linalg.LinAlgError, dpnp_func, a_dp_not_scquare)

        # invalid UPLO
        assert_raises(ValueError, dpnp_func, a_dp, UPLO="N")


@pytest.mark.parametrize("type", get_all_dtypes(no_bool=True, no_complex=True))
def test_eigvals(type):
    if dpctl.SyclDevice().device_type != dpctl.device_type.gpu:
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


class TestMatrixPower:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize(
        "data, power",
        [
            (
                numpy.block(
                    [
                        [numpy.eye(2), numpy.zeros((2, 2))],
                        [numpy.zeros((2, 2)), numpy.eye(2) * 2],
                    ]
                ),
                3,
            ),  # Block-diagonal matrix
            (numpy.eye(3, k=1) + numpy.eye(3), 3),  # Non-diagonal matrix
            (
                numpy.eye(3, k=1) + numpy.eye(3),
                -3,
            ),  # Inverse of non-diagonal matrix
        ],
    )
    def test_matrix_power(self, data, power, dtype):
        a = data.astype(dtype)
        a_dp = inp.array(a)

        result = inp.linalg.matrix_power(a_dp, power)
        expected = numpy.linalg.matrix_power(a, power)

        assert_dtype_allclose(result, expected)

    def test_matrix_power_errors(self):
        a_dp = inp.eye(4, dtype="float32")

        # unsupported type `a`
        a_np = inp.asnumpy(a_dp)
        assert_raises(TypeError, inp.linalg.matrix_power, a_np, 2)

        # unsupported type `power`
        assert_raises(TypeError, inp.linalg.matrix_power, a_dp, 1.5)
        assert_raises(TypeError, inp.linalg.matrix_power, a_dp, [2])

        # not invertible
        # TODO: remove it when mkl>=2024.0 is released (MKLD-16626)
        if not is_cpu_device():
            noninv = inp.array([[1, 0], [0, 0]])
            assert_raises(
                inp.linalg.LinAlgError, inp.linalg.matrix_power, noninv, -1
            )


class TestMatrixRank:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize(
        "data",
        [
            numpy.eye(4),
            numpy.diag([1, 1, 1, 0]),
            numpy.zeros((4, 4)),
            numpy.array([1, 0, 0, 0]),
            numpy.zeros((4,)),
            numpy.array(1),
        ],
    )
    def test_matrix_rank(self, data, dtype):
        a = data.astype(dtype)
        a_dp = inp.array(a)

        np_rank = numpy.linalg.matrix_rank(a)
        dp_rank = inp.linalg.matrix_rank(a_dp)
        assert np_rank == dp_rank

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize(
        "data",
        [
            numpy.eye(4),
            numpy.ones((4, 4)),
            numpy.zeros((4, 4)),
            numpy.diag([1, 1, 1, 0]),
        ],
    )
    def test_matrix_rank_hermitian(self, data, dtype):
        a = data.astype(dtype)
        a_dp = inp.array(a)

        np_rank = numpy.linalg.matrix_rank(a, hermitian=True)
        dp_rank = inp.linalg.matrix_rank(a_dp, hermitian=True)
        assert np_rank == dp_rank

    @pytest.mark.parametrize(
        "high_tol, low_tol",
        [
            (0.99e-6, 1.01e-6),
            (numpy.array(0.99e-6), numpy.array(1.01e-6)),
            (numpy.array([0.99e-6]), numpy.array([1.01e-6])),
        ],
        ids=[
            "float",
            "0-D array",
            "1-D array",
        ],
    )
    def test_matrix_rank_tolerance(self, high_tol, low_tol):
        a = numpy.eye(4)
        a[-1, -1] = 1e-6
        a_dp = inp.array(a)

        if isinstance(high_tol, numpy.ndarray):
            dp_high_tol = inp.array(
                high_tol, usm_type=a_dp.usm_type, sycl_queue=a_dp.sycl_queue
            )
            dp_low_tol = inp.array(
                low_tol, usm_type=a_dp.usm_type, sycl_queue=a_dp.sycl_queue
            )
        else:
            dp_high_tol = high_tol
            dp_low_tol = low_tol

        np_rank_high_tol = numpy.linalg.matrix_rank(
            a, hermitian=True, tol=high_tol
        )
        dp_rank_high_tol = inp.linalg.matrix_rank(
            a_dp, hermitian=True, tol=dp_high_tol
        )
        assert np_rank_high_tol == dp_rank_high_tol

        np_rank_low_tol = numpy.linalg.matrix_rank(
            a, hermitian=True, tol=low_tol
        )
        dp_rank_low_tol = inp.linalg.matrix_rank(
            a_dp, hermitian=True, tol=dp_low_tol
        )
        assert np_rank_low_tol == dp_rank_low_tol

    def test_matrix_rank_errors(self):
        a_dp = inp.array([[1, 2], [3, 4]], dtype="float32")

        # unsupported type `a`
        a_np = inp.asnumpy(a_dp)
        assert_raises(TypeError, inp.linalg.matrix_rank, a_np)

        # unsupported type `tol`
        tol = numpy.array(0.5, dtype="float32")
        assert_raises(TypeError, inp.linalg.matrix_rank, a_dp, tol)
        assert_raises(TypeError, inp.linalg.matrix_rank, a_dp, [0.5])

        # diffetent queue
        a_queue = dpctl.SyclQueue()
        tol_queue = dpctl.SyclQueue()
        a_dp_q = inp.array(a_dp, sycl_queue=a_queue)
        tol_dp_q = inp.array([0.5], dtype="float32", sycl_queue=tol_queue)
        assert_raises(
            ExecutionPlacementError,
            inp.linalg.matrix_rank,
            a_dp_q,
            tol_dp_q,
        )


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
    # TODO: New packages that fix issue CMPLRLLVM-53771 are only available in internal CI.
    # Skip the tests on cpu until these packages are available for the external CI.
    # Specifically dpcpp_linux-64>=2024.1.0
    @pytest.mark.skipif(is_cpu_device(), reason="CMPLRLLVM-53771")
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
        # Set seed_value=81 to prevent
        # random generation of the input singular matrix
        a = generate_random_numpy_array(shape, dtype, seed_value=81)
        ia = inp.array(a)

        if mode == "r":
            np_r = numpy.linalg.qr(a, mode)
            dpnp_r = inp.linalg.qr(ia, mode)
        else:
            np_q, np_r = numpy.linalg.qr(a, mode)
            dpnp_q, dpnp_r = inp.linalg.qr(ia, mode)

            # check decomposition
            if mode in ("complete", "reduced"):
                if a.ndim == 2:
                    assert_almost_equal(
                        inp.dot(dpnp_q, dpnp_r),
                        a,
                        decimal=5,
                    )
                else:  # a.ndim > 2
                    assert_almost_equal(
                        inp.matmul(dpnp_q, dpnp_r),
                        a,
                        decimal=5,
                    )
            else:  # mode=="raw"
                assert_dtype_allclose(dpnp_q, np_q)

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

    @pytest.mark.skipif(is_cpu_device(), reason="CMPLRLLVM-53771")
    @pytest.mark.parametrize(
        "mode",
        ["r", "raw", "complete", "reduced"],
        ids=["r", "raw", "complete", "reduced"],
    )
    def test_qr_strides(self, mode):
        a = generate_random_numpy_array((5, 5))
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
            tol = 1e-04
        self._tol = tol

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
        self, dp_a, dp_u, dp_s, dp_vt, np_u, np_s, np_vt, compute_vt
    ):
        tol = self._tol
        if compute_vt:
            dpnp_diag_s = inp.zeros_like(dp_a, dtype=dp_s.dtype)
            for i in range(min(dp_a.shape[-2], dp_a.shape[-1])):
                dpnp_diag_s[..., i, i] = dp_s[..., i]
                reconstructed = inp.dot(dp_u, inp.dot(dpnp_diag_s, dp_vt))
            # TODO: use assert dpnp.allclose() inside check_decomposition()
            # when it will support complex dtypes
            assert_allclose(dp_a, reconstructed, rtol=tol, atol=1e-4)

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
        self.get_tol(dtype)
        self.check_decomposition(
            dp_a, dp_u, dp_s, dp_vt, np_u, np_s, np_vt, True
        )

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    @pytest.mark.parametrize("compute_vt", [True, False], ids=["True", "False"])
    @pytest.mark.parametrize(
        "shape",
        [(2, 2), (16, 16)],
        ids=["(2, 2)", "(16, 16)"],
    )
    def test_svd_hermitian(self, dtype, compute_vt, shape):
        # Set seed_value=81 to prevent
        # random generation of the input singular matrix
        a = generate_random_numpy_array(
            shape, dtype, hermitian=True, seed_value=81
        )
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

        self.get_tol(dtype)

        self.check_decomposition(
            dp_a, dp_u, dp_s, dp_vt, np_u, np_s, np_vt, compute_vt
        )

    def test_svd_errors(self):
        a_dp = inp.array([[1, 2], [3, 4]], dtype="float32")

        # unsupported type
        a_np = inp.asnumpy(a_dp)
        assert_raises(TypeError, inp.linalg.svd, a_np)

        # a.ndim < 2
        a_dp_ndim_1 = a_dp.flatten()
        assert_raises(inp.linalg.LinAlgError, inp.linalg.svd, a_dp_ndim_1)


class TestPinv:
    def get_tol(self, dtype):
        tol = 1e-06
        if dtype in (inp.float32, inp.complex64):
            tol = 1e-04
        elif not has_support_aspect64() and dtype in (
            inp.int32,
            inp.int64,
            None,
        ):
            tol = 1e-04
        self._tol = tol

    def check_types_shapes(self, dp_B, np_B):
        if has_support_aspect64():
            assert dp_B.dtype == np_B.dtype
        else:
            assert dp_B.dtype.kind == np_B.dtype.kind

        assert dp_B.shape == np_B.shape

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
    def test_pinv(self, dtype, shape):
        # Set seed_value=81 to prevent
        # random generation of the input singular matrix
        a = generate_random_numpy_array(shape, dtype, seed_value=81)
        a_dp = inp.array(a)

        B = numpy.linalg.pinv(a)
        B_dp = inp.linalg.pinv(a_dp)

        self.check_types_shapes(B_dp, B)
        self.get_tol(dtype)
        tol = self._tol
        assert_allclose(B_dp, B, rtol=tol, atol=tol)

        if a.ndim == 2:
            reconstructed = inp.dot(a_dp, inp.dot(B_dp, a_dp))
        else:  # a.ndim > 2
            reconstructed = inp.matmul(a_dp, inp.matmul(B_dp, a_dp))

        assert_allclose(reconstructed, a_dp, rtol=tol, atol=tol)

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    @pytest.mark.parametrize(
        "shape",
        [(2, 2), (16, 16)],
        ids=["(2, 2)", "(16, 16)"],
    )
    def test_pinv_hermitian(self, dtype, shape):
        # Set seed_value=81 to prevent
        # random generation of the input singular matrix
        a = generate_random_numpy_array(
            shape, dtype, hermitian=True, seed_value=81
        )
        a_dp = inp.array(a)

        B = numpy.linalg.pinv(a, hermitian=True)
        B_dp = inp.linalg.pinv(a_dp, hermitian=True)

        self.check_types_shapes(B_dp, B)
        self.get_tol(dtype)
        tol = self._tol

        reconstructed = inp.dot(inp.dot(a_dp, B_dp), a_dp)
        assert_allclose(reconstructed, a_dp, rtol=tol, atol=tol)

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
    def test_pinv_empty(self, dtype, shape):
        a = numpy.empty(shape, dtype=dtype)
        a_dp = inp.array(a)

        B = numpy.linalg.pinv(a)
        B_dp = inp.linalg.pinv(a_dp)

        assert_dtype_allclose(B_dp, B)

    def test_pinv_strides(self):
        a = generate_random_numpy_array((5, 5))
        a_dp = inp.array(a)

        self.get_tol(a_dp.dtype)
        tol = self._tol

        # positive strides
        B = numpy.linalg.pinv(a[::2, ::2])
        B_dp = inp.linalg.pinv(a_dp[::2, ::2])
        assert_allclose(B_dp, B, rtol=tol, atol=tol)

        # negative strides
        B = numpy.linalg.pinv(a[::-2, ::-2])
        B_dp = inp.linalg.pinv(a_dp[::-2, ::-2])
        assert_allclose(B_dp, B, rtol=tol, atol=tol)

    def test_pinv_errors(self):
        a_dp = inp.array([[1, 2], [3, 4]], dtype="float32")

        # unsupported type `a`
        a_np = inp.asnumpy(a_dp)
        assert_raises(TypeError, inp.linalg.pinv, a_np)

        # unsupported type `rcond`
        rcond = numpy.array(0.5, dtype="float32")
        assert_raises(TypeError, inp.linalg.pinv, a_dp, rcond)
        assert_raises(TypeError, inp.linalg.pinv, a_dp, [0.5])

        # non-broadcastable `rcond`
        rcond_dp = inp.array([0.5], dtype="float32")
        assert_raises(ValueError, inp.linalg.pinv, a_dp, rcond_dp)

        # a.ndim < 2
        a_dp_ndim_1 = a_dp.flatten()
        assert_raises(inp.linalg.LinAlgError, inp.linalg.pinv, a_dp_ndim_1)

        # diffetent queue
        a_queue = dpctl.SyclQueue()
        rcond_queue = dpctl.SyclQueue()
        a_dp_q = inp.array(a_dp, sycl_queue=a_queue)
        rcond_dp_q = inp.array([0.5], dtype="float32", sycl_queue=rcond_queue)
        assert_raises(ValueError, inp.linalg.pinv, a_dp_q, rcond_dp_q)


class TestTensorinv:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize(
        "shape, ind",
        [
            ((4, 6, 8, 3), 2),
            ((24, 8, 3), 1),
        ],
        ids=[
            "(4, 6, 8, 3)",
            "(24, 8, 3)",
        ],
    )
    def test_tensorinv(self, dtype, shape, ind):
        a = numpy.eye(24, dtype=dtype).reshape(shape)
        a_dp = inp.array(a)

        ainv = numpy.linalg.tensorinv(a, ind=ind)
        ainv_dp = inp.linalg.tensorinv(a_dp, ind=ind)

        assert ainv.shape == ainv_dp.shape
        assert_dtype_allclose(ainv_dp, ainv)

    def test_test_tensorinv_errors(self):
        a_dp = inp.eye(24, dtype="float32").reshape(4, 6, 8, 3)

        # unsupported type `a`
        a_np = inp.asnumpy(a_dp)
        assert_raises(TypeError, inp.linalg.pinv, a_np)

        # unsupported type `ind`
        assert_raises(TypeError, inp.linalg.tensorinv, a_dp, 2.0)
        assert_raises(TypeError, inp.linalg.tensorinv, a_dp, [2.0])
        assert_raises(ValueError, inp.linalg.tensorinv, a_dp, -1)

        # non-square
        assert_raises(inp.linalg.LinAlgError, inp.linalg.tensorinv, a_dp, 1)


class TestTensorsolve:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize(
        "axes",
        [None, (1,), (2,)],
        ids=[
            "None",
            "(1,)",
            "(2,)",
        ],
    )
    def test_tensorsolve_axes(self, dtype, axes):
        a = numpy.eye(12).reshape(12, 3, 4).astype(dtype)
        b = numpy.ones(a.shape[0], dtype=dtype)

        a_dp = inp.array(a)
        b_dp = inp.array(b)

        res_np = numpy.linalg.tensorsolve(a, b, axes=axes)
        res_dp = inp.linalg.tensorsolve(a_dp, b_dp, axes=axes)

        assert res_np.shape == res_dp.shape
        assert_dtype_allclose(res_dp, res_np)

    def test_tensorsolve_errors(self):
        a_dp = inp.eye(24, dtype="float32").reshape(4, 6, 8, 3)
        b_dp = inp.ones(a_dp.shape[:2], dtype="float32")

        # unsupported type `a` and `b`
        a_np = inp.asnumpy(a_dp)
        b_np = inp.asnumpy(b_dp)
        assert_raises(TypeError, inp.linalg.tensorsolve, a_np, b_dp)
        assert_raises(TypeError, inp.linalg.tensorsolve, a_dp, b_np)

        # unsupported type `axes`
        assert_raises(TypeError, inp.linalg.tensorsolve, a_dp, 2.0)
        assert_raises(TypeError, inp.linalg.tensorsolve, a_dp, -2)

        # incorrect axes
        assert_raises(
            inp.linalg.LinAlgError, inp.linalg.tensorsolve, a_dp, b_dp, (1,)
        )
