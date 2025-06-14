import dpctl
import dpctl.tensor as dpt
import numpy
import pytest
from dpctl.tensor._numpy_helper import AxisError
from dpctl.utils import ExecutionPlacementError
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
    assert_equal,
    assert_raises,
    assert_raises_regex,
    suppress_warnings,
)

import dpnp

from .helper import (
    assert_dtype_allclose,
    generate_random_numpy_array,
    get_all_dtypes,
    get_float_complex_dtypes,
    get_integer_float_dtypes,
    has_support_aspect64,
    is_cpu_device,
    numpy_version,
    requires_intel_mkl_version,
)
from .third_party.cupy import testing


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


# check linear algebra functions from dpnp.linalg
# with multidimensional usm_ndarray as input
@pytest.mark.parametrize(
    "func, gen_kwargs, func_kwargs",
    [
        pytest.param("cholesky", {"hermitian": True}, {}),
        pytest.param("cond", {}, {}),
        pytest.param("det", {}, {}),
        pytest.param("eig", {}, {}),
        pytest.param("eigh", {"hermitian": True}, {}),
        pytest.param("eigvals", {}, {}),
        pytest.param("eigvalsh", {"hermitian": True}, {}),
        pytest.param("inv", {}, {}),
        pytest.param("matrix_power", {}, {"n": 4}),
        pytest.param("matrix_rank", {}, {}),
        pytest.param("norm", {}, {}),
        pytest.param("pinv", {}, {}),
        pytest.param("qr", {}, {}),
        pytest.param("slogdet", {}, {}),
        pytest.param("solve", {}, {}),
        pytest.param("svd", {}, {}),
        pytest.param("tensorinv", {}, {"ind": 1}),
        pytest.param("tensorsolve", {}, {}),
    ],
)
def test_usm_ndarray_linalg_batch(func, gen_kwargs, func_kwargs):
    shape = (
        (2, 2, 3, 3) if func not in ["tensorinv", "tensorsolve"] else (4, 2, 2)
    )

    if func == "tensorsolve":
        shape_b = (4,)
        dpt_args = [
            dpt.asarray(
                generate_random_numpy_array(shape, seed_value=81, **gen_kwargs)
            ),
            dpt.asarray(
                generate_random_numpy_array(
                    shape_b, seed_value=81, **gen_kwargs
                )
            ),
        ]
    elif func in ["lstsq", "solve"]:
        dpt_args = [
            dpt.asarray(
                generate_random_numpy_array(shape, seed_value=81, **gen_kwargs)
            )
            for _ in range(2)
        ]
    else:
        dpt_args = [
            dpt.asarray(generate_random_numpy_array(shape, **gen_kwargs))
        ]

    result = getattr(dpnp.linalg, func)(*dpt_args, **func_kwargs)

    if isinstance(result, tuple):
        for res in result:
            assert isinstance(res, dpnp.ndarray)
    else:
        assert isinstance(result, dpnp.ndarray)


# check linear algebra functions from dpnp
# with multidimensional usm_ndarray as input
@pytest.mark.parametrize(
    "func", ["dot", "inner", "kron", "matmul", "outer", "tensordot", "vdot"]
)
def test_usm_ndarray_linearalgebra_batch(func):
    shape = (2, 2, 2, 2)

    dpt_args = [
        dpt.asarray(generate_random_numpy_array(shape, seed_value=81))
        for _ in range(2)
    ]

    result = getattr(dpnp, func)(*dpt_args)

    assert isinstance(result, dpnp.ndarray)


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
        ids=["2D", "3D", "4D"],
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_cholesky(self, array, dtype):
        a = numpy.array(array, dtype=dtype)
        ia = dpnp.array(a)
        result = dpnp.linalg.cholesky(ia)
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
        ids=["2D", "3D", "4D"],
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_cholesky_upper(self, array, dtype):
        ia = dpnp.array(array, dtype=dtype)
        result = dpnp.linalg.cholesky(ia, upper=True)

        if ia.ndim > 2:
            n = ia.shape[-1]
            ia_reshaped = ia.reshape(-1, n, n)
            res_reshaped = result.reshape(-1, n, n)
            batch_size = ia_reshaped.shape[0]
            for idx in range(batch_size):
                # Reconstruct the matrix using the Cholesky decomposition result
                if dpnp.issubdtype(dtype, dpnp.complexfloating):
                    reconstructed = (
                        res_reshaped[idx].T.conj() @ res_reshaped[idx]
                    )
                else:
                    reconstructed = res_reshaped[idx].T @ res_reshaped[idx]
                assert dpnp.allclose(reconstructed, ia_reshaped[idx])
        else:
            # Reconstruct the matrix using the Cholesky decomposition result
            if dpnp.issubdtype(dtype, dpnp.complexfloating):
                reconstructed = result.T.conj() @ result
            else:
                reconstructed = result.T @ result
            assert dpnp.allclose(reconstructed, ia)

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
        ids=["2D", "3D", "4D"],
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_cholesky_upper_numpy(self, array, dtype):
        a = numpy.array(array, dtype=dtype)
        ia = dpnp.array(a)
        result = dpnp.linalg.cholesky(ia, upper=True)
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

        a_dp = dpnp.array(a_np)

        # positive strides
        expected = numpy.linalg.cholesky(a_np[::2, ::2])
        result = dpnp.linalg.cholesky(a_dp[::2, ::2])
        assert_allclose(result, expected)

        # negative strides
        expected = numpy.linalg.cholesky(a_np[::-2, ::-2])
        result = dpnp.linalg.cholesky(a_dp[::-2, ::-2])
        assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "shape",
        [(0, 0), (3, 0, 0), (0, 2, 2)],
        ids=["(0, 0)", "(3, 0, 0)", "(0, 2, 2)"],
    )
    def test_cholesky_empty(self, shape):
        a = numpy.empty(shape)
        ia = dpnp.array(a)
        result = dpnp.linalg.cholesky(ia)
        expected = numpy.linalg.cholesky(a)
        assert_array_equal(result, expected)

    def test_cholesky_errors(self):
        a_dp = dpnp.array([[1, 2], [2, 5]], dtype="float32")

        # unsupported type
        a_np = dpnp.asnumpy(a_dp)
        assert_raises(TypeError, dpnp.linalg.cholesky, a_np)

        # a.ndim < 2
        a_dp_ndim_1 = a_dp.flatten()
        assert_raises(
            dpnp.linalg.LinAlgError, dpnp.linalg.cholesky, a_dp_ndim_1
        )

        # a is not square
        a_dp = dpnp.ones((2, 3))
        assert_raises(dpnp.linalg.LinAlgError, dpnp.linalg.cholesky, a_dp)


class TestCond:
    def setup_method(self):
        numpy.random.seed(70)

    @pytest.mark.parametrize(
        "shape", [(0, 4, 4), (4, 0, 3, 3)], ids=["(0, 5, 3)", "(4, 0, 2, 3)"]
    )
    @pytest.mark.parametrize(
        "p", [None, -dpnp.inf, -2, -1, 1, 2, dpnp.inf, "fro"]
    )
    def test_empty(self, shape, p):
        a = numpy.empty(shape)
        ia = dpnp.array(a)

        result = dpnp.linalg.cond(ia, p=p)
        expected = numpy.linalg.cond(a, p=p)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    @pytest.mark.parametrize(
        "shape", [(4, 4), (2, 4, 3, 3)], ids=["(4, 4)", "(2, 4, 3, 3)"]
    )
    @pytest.mark.parametrize(
        "p", [None, -dpnp.inf, -2, -1, 1, 2, dpnp.inf, "fro"]
    )
    def test_basic(self, dtype, shape, p):
        a = generate_random_numpy_array(shape, dtype)
        ia = dpnp.array(a)

        result = dpnp.linalg.cond(ia, p=p)
        expected = numpy.linalg.cond(a, p=p)
        assert_dtype_allclose(result, expected, factor=16)

    @pytest.mark.parametrize(
        "p", [None, -dpnp.inf, -2, -1, 1, 2, dpnp.inf, "fro"]
    )
    def test_bool(self, p):
        a = numpy.array([[True, True], [True, False]])
        ia = dpnp.array(a)

        result = dpnp.linalg.cond(ia, p=p)
        expected = numpy.linalg.cond(a, p=p)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "p", [None, -dpnp.inf, -2, -1, 1, 2, dpnp.inf, "fro"]
    )
    def test_nan_to_inf(self, p):
        a = numpy.zeros((2, 2))
        ia = dpnp.array(a)

        # NumPy does not raise LinAlgError on singular matrices.
        # It returns `inf`, `0`, or large/small finite values
        # depending on the norm and the matrix content.
        # DPNP raises LinAlgError for 1, -1, inf, -inf, and 'fro'
        # due to use of gesv in the 2D case.
        # For [None, 2, -2], DPNP does not raise.
        if p in [None, 2, -2]:
            result = dpnp.linalg.cond(ia, p=p)
            expected = numpy.linalg.cond(a, p=p)
            assert_dtype_allclose(result, expected)
        else:
            assert_raises(dpnp.linalg.LinAlgError, dpnp.linalg.cond, ia, p=p)

    @pytest.mark.parametrize(
        "p", [None, -dpnp.inf, -2, -1, 1, 2, dpnp.inf, "fro"]
    )
    @pytest.mark.parametrize(
        "stride",
        [(-2, -3, 2, -2), (-2, 4, -4, -4), (2, 3, 4, 4), (-1, 3, 3, -3)],
        ids=[
            "(-2, -3, 2, -2)",
            "(-2, 4, -4, -4)",
            "(2, 3, 4, 4)",
            "(-1, 3, 3, -3)",
        ],
    )
    def test_strided(self, p, stride):
        A = numpy.random.rand(6, 8, 10, 10)
        B = dpnp.asarray(A)
        slices = tuple(slice(None, None, stride[i]) for i in range(A.ndim))
        a = A[slices]
        b = B[slices]

        result = dpnp.linalg.cond(b, p=p)
        expected = numpy.linalg.cond(a, p=p)
        assert_dtype_allclose(result, expected, factor=24)

    def test_error(self):
        # cond is not defined on empty arrays
        ia = dpnp.empty((2, 0))
        with pytest.raises(ValueError):
            dpnp.linalg.cond(ia, p=1)


class TestDet:
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
        ids=["2D", "3D", "4D"],
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_det(self, array, dtype):
        a = numpy.array(array, dtype=dtype)
        ia = dpnp.array(a)
        result = dpnp.linalg.det(ia)
        expected = numpy.linalg.det(a)
        assert_allclose(result, expected)

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

        a_dp = dpnp.array(a_np)

        # positive strides
        expected = numpy.linalg.det(a_np[::2, ::2])
        result = dpnp.linalg.det(a_dp[::2, ::2])
        assert_allclose(result, expected, rtol=1e-5)

        # negative strides
        expected = numpy.linalg.det(a_np[::-2, ::-2])
        result = dpnp.linalg.det(a_dp[::-2, ::-2])
        assert_allclose(result, expected)

    def test_det_empty(self):
        a = numpy.empty((0, 0, 2, 2), dtype=numpy.float32)
        ia = dpnp.array(a)

        expected = numpy.linalg.det(a)
        result = dpnp.linalg.det(ia)
        assert_allclose(result, expected)

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
        a_dp = dpnp.array(a_np)

        expected = numpy.linalg.det(a_np)
        result = dpnp.linalg.det(a_dp)

        assert_allclose(result, expected)

    def test_det_singular_matrix_3D(self):
        a_np = numpy.array(
            [[[1, 2], [3, 4]], [[1, 2], [1, 2]], [[1, 3], [3, 1]]]
        )
        a_dp = dpnp.array(a_np)

        expected = numpy.linalg.det(a_np)
        result = dpnp.linalg.det(a_dp)

        assert_allclose(result, expected)

    def test_det_errors(self):
        a_dp = dpnp.array([[1, 2], [3, 5]], dtype="float32")

        # unsupported type
        a_np = dpnp.asnumpy(a_dp)
        assert_raises(TypeError, dpnp.linalg.det, a_np)


class TestEigenvalue:
    # Eigenvalue decomposition of a matrix or a batch of matrices
    # by checking if the eigen equation A*v=w*v holds for given eigenvalues(w)
    # and eigenvectors(v).
    def assert_eigen_decomposition(self, a, w, v, rtol=1e-5, atol=1e-5):
        a_ndim = a.ndim
        if a_ndim == 2:
            assert_allclose(a @ v, v @ dpnp.diag(w), rtol=rtol, atol=atol)
        else:  # a_ndim > 2
            if a_ndim > 3:
                a = a.reshape(-1, *a.shape[-2:])
                w = w.reshape(-1, w.shape[-1])
                v = v.reshape(-1, *v.shape[-2:])
            for i in range(a.shape[0]):
                assert_allclose(
                    a[i].dot(v[i]), w[i] * v[i], rtol=rtol, atol=atol
                )

    @pytest.mark.parametrize("func", ["eig", "eigvals", "eigh", "eigvalsh"])
    @pytest.mark.parametrize(
        "shape",
        [(2, 2), (2, 3, 3), (2, 2, 3, 3)],
        ids=["(2, 2)", "(2, 3, 3)", "(2, 2, 3, 3)"],
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_eigenvalues(self, func, shape, dtype, order):
        # Set a `hermitian` flag for generate_random_numpy_array() to
        # get a symmetric array for eigh() and eigvalsh() or
        # non-symmetric for eig() and eigvals()
        is_hermitian = func in ("eigh, eigvalsh")
        a = generate_random_numpy_array(
            shape, dtype, order, hermitian=is_hermitian, low=-4, high=4
        )
        a_dp = dpnp.array(a)

        # NumPy with OneMKL and with rocSOLVER sorts in ascending order,
        # so w's should be directly comparable.
        # However, both OneMKL and rocSOLVER pick a different convention for
        # constructing eigenvectors, so v's are not directly comparable and
        # we verify them through the eigen equation A*v=w*v.
        if func in ("eig", "eigh"):
            w, _ = getattr(numpy.linalg, func)(a)
            result = getattr(dpnp.linalg, func)(a_dp)
            w_dp, v_dp = result.eigenvalues, result.eigenvectors

            self.assert_eigen_decomposition(a_dp, w_dp, v_dp)

        else:  # eighvals or eigvalsh
            w = getattr(numpy.linalg, func)(a)
            w_dp = getattr(dpnp.linalg, func)(a_dp)

        assert_dtype_allclose(w_dp, w, factor=24)

    # eigh() and eigvalsh() are tested in cupy tests
    @pytest.mark.parametrize("func", ["eig", "eigvals"])
    @pytest.mark.parametrize(
        "shape",
        [(0, 0), (2, 0, 0), (0, 3, 3)],
        ids=["(0, 0)", "(2, 0, 0)", "(0, 3, 3)"],
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_eigenvalue_empty(self, func, shape, dtype):
        a_np = numpy.empty(shape, dtype=dtype)
        a_dp = dpnp.array(a_np)

        if func == "eig":
            w, v = getattr(numpy.linalg, func)(a_np)
            result = getattr(dpnp.linalg, func)(a_dp)
            w_dp, v_dp = result.eigenvalues, result.eigenvectors

            assert_dtype_allclose(v_dp, v)

        else:  # eigvals
            w = getattr(numpy.linalg, func)(a_np)
            w_dp = getattr(dpnp.linalg, func)(a_dp)

        assert_dtype_allclose(w_dp, w)

    @pytest.mark.parametrize("func", ["eig", "eigvals", "eigh", "eigvalsh"])
    def test_eigenvalue_errors(self, func):
        a_dp = dpnp.array([[1, 3], [3, 2]], dtype="float32")

        # unsupported type
        a_np = dpnp.asnumpy(a_dp)
        dpnp_func = getattr(dpnp.linalg, func)
        assert_raises(TypeError, dpnp_func, a_np)

        # a.ndim < 2
        a_dp_ndim_1 = a_dp.flatten()
        assert_raises(dpnp.linalg.LinAlgError, dpnp_func, a_dp_ndim_1)

        # a is not square
        a_dp_not_scquare = dpnp.ones((2, 3))
        assert_raises(dpnp.linalg.LinAlgError, dpnp_func, a_dp_not_scquare)

        # invalid UPLO
        if func in ("eigh", "eigvalsh"):
            assert_raises(ValueError, dpnp_func, a_dp, UPLO="N")


class TestEinsum:
    def test_einsum_trivial_cases(self):
        a = dpnp.arange(25).reshape(5, 5)
        b = dpnp.arange(5)
        a_np = a.asnumpy()
        b_np = b.asnumpy()

        # one input, no optimization is needed
        result = dpnp.einsum("i", b, optimize="greedy")
        expected = numpy.einsum("i", b_np, optimize="greedy")
        assert_dtype_allclose(result, expected)

        # two inputs, no optimization is needed
        result = dpnp.einsum("ij,jk", a, a, optimize="greedy")
        expected = numpy.einsum("ij,jk", a_np, a_np, optimize="greedy")
        assert_dtype_allclose(result, expected)

        # no optimization in optimal mode
        result = dpnp.einsum("ij,jk", a, a, optimize=["optimal", 1])
        expected = numpy.einsum("ij,jk", a_np, a_np, optimize=["optimal", 1])
        assert_dtype_allclose(result, expected)

        # naive cost equal or smaller than optimized cost
        result = dpnp.einsum("i,i,i", b, b, b, optimize="greedy")
        expected = numpy.einsum("i,i,i", b_np, b_np, b_np, optimize="greedy")
        assert_dtype_allclose(result, expected)

    def test_einsum_out(self):
        a = dpnp.ones((5, 5))
        out = dpnp.empty((5,))
        result = dpnp.einsum("ii->i", a, out=out)
        assert result is out
        expected = numpy.einsum("ii->i", a.asnumpy())
        assert_dtype_allclose(result, expected)

    def test_einsum_error1(self):
        a = dpnp.ones((5, 5))
        out = dpnp.empty((5,), sycl_queue=dpctl.SyclQueue())
        # inconsistent sycl_queue
        assert_raises(ExecutionPlacementError, dpnp.einsum, "ii->i", a, out=out)

        # unknown value for optimize keyword
        assert_raises(TypeError, dpnp.einsum, "ii->i", a, optimize="blah")

        # repeated scripts in output
        assert_raises(ValueError, dpnp.einsum, "ij,jk->ii", a, a)

        a = dpnp.ones((5, 4))
        # different size for same label 5 != 4
        assert_raises(ValueError, dpnp.einsum, "ii", a)

        a = dpnp.arange(25).reshape(5, 5)
        # subscript is not within the valid range [0, 52)
        assert_raises(ValueError, dpnp.einsum, a, [53, 53])

    @pytest.mark.parametrize("do_opt", [True, False])
    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_einsum_error2(self, do_opt, xp):
        a = xp.asarray(0)
        b = xp.asarray([0])
        c = xp.asarray([0, 0])
        d = xp.asarray([[0, 0], [0, 0]])

        # Need enough arguments
        assert_raises(ValueError, xp.einsum, optimize=do_opt)
        assert_raises(ValueError, xp.einsum, "", optimize=do_opt)

        # subscripts must be a string
        assert_raises(TypeError, xp.einsum, a, 0, optimize=do_opt)

        # this call returns a segfault
        assert_raises(TypeError, xp.einsum, *(None,) * 63, optimize=do_opt)

        # number of operands must match count in subscripts string
        assert_raises(ValueError, xp.einsum, "", a, 0, optimize=do_opt)
        assert_raises(ValueError, xp.einsum, ",", 0, b, b, optimize=do_opt)
        assert_raises(ValueError, xp.einsum, ",", b, optimize=do_opt)

        # can't have more subscripts than dimensions in the operand
        assert_raises(ValueError, xp.einsum, "i", a, optimize=do_opt)
        assert_raises(ValueError, xp.einsum, "ij", c, optimize=do_opt)
        assert_raises(ValueError, xp.einsum, "...i", a, optimize=do_opt)
        assert_raises(ValueError, xp.einsum, "i...j", c, optimize=do_opt)
        assert_raises(ValueError, xp.einsum, "i...", a, optimize=do_opt)
        assert_raises(ValueError, xp.einsum, "ij...", c, optimize=do_opt)

        # invalid ellipsis
        assert_raises(ValueError, xp.einsum, "i..", c, optimize=do_opt)
        assert_raises(ValueError, xp.einsum, ".i...", c, optimize=do_opt)
        assert_raises(ValueError, xp.einsum, "j->..j", c, optimize=do_opt)
        assert_raises(ValueError, xp.einsum, "j->.j...", c, optimize=do_opt)

        # invalid subscript character
        assert_raises(ValueError, xp.einsum, "i%...", c, optimize=do_opt)
        assert_raises(ValueError, xp.einsum, "...j$", c, optimize=do_opt)
        assert_raises(ValueError, xp.einsum, "i->&", c, optimize=do_opt)

        # output subscripts must appear in input
        assert_raises(ValueError, xp.einsum, "i->ij", c, optimize=do_opt)

        # output subscripts may only be specified once
        assert_raises(ValueError, xp.einsum, "ij->jij", d, optimize=do_opt)

        # dimensions must match when being collapsed
        a = xp.arange(6).reshape(2, 3)
        assert_raises(ValueError, xp.einsum, "ii", a, optimize=do_opt)
        assert_raises(ValueError, xp.einsum, "ii->i", a, optimize=do_opt)

        with assert_raises_regex(ValueError, "'b'"):
            # 'c' erroneously appeared in the error message
            a = xp.ones((3, 3, 4, 5, 6))
            b = xp.ones((3, 4, 5))
            xp.einsum("aabcb,abc", a, b)

    @pytest.mark.parametrize("do_opt", [True, False])
    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_einsum_specific_errors(self, do_opt, xp):
        a = xp.asarray(0)
        # out parameter must be an array
        assert_raises(TypeError, xp.einsum, "", a, out="test", optimize=do_opt)

        # order parameter must be a valid order
        assert_raises(ValueError, xp.einsum, "", a, order="W", optimize=do_opt)

        # other keyword arguments are rejected
        assert_raises(TypeError, xp.einsum, "", a, bad_arg=0, optimize=do_opt)

        # broadcasting to new dimensions must be enabled explicitly
        a = xp.arange(6).reshape(2, 3)
        b = xp.asarray([[0, 1], [0, 1]])
        assert_raises(ValueError, xp.einsum, "i", a, optimize=do_opt)
        out = xp.arange(4).reshape(2, 2)
        assert_raises(
            ValueError, xp.einsum, "i->i", b, out=out, optimize=do_opt
        )

        # Check order kwarg, asanyarray allows 1d to pass through
        a = xp.arange(6).reshape(-1, 1)
        assert_raises(
            ValueError, xp.einsum, "i->i", a, optimize=do_opt, order="d"
        )

    def check_einsum_sums(self, dtype, do_opt=False):
        dtype = numpy.dtype(dtype)
        # Check various sums.  Does many sizes to exercise unrolled loops.

        # sum(a, axis=-1)
        for n in range(1, 17):
            a = numpy.arange(n, dtype=dtype)
            a_dp = dpnp.array(a)
            expected = numpy.einsum("i->", a, optimize=do_opt)
            assert_dtype_allclose(
                dpnp.einsum("i->", a_dp, optimize=do_opt), expected
            )
            assert_dtype_allclose(
                dpnp.einsum(a_dp, [0], [], optimize=do_opt), expected
            )

        for n in range(1, 17):
            a = numpy.arange(2 * 3 * n, dtype=dtype).reshape(2, 3, n)
            a_dp = dpnp.array(a)
            expected = numpy.einsum("...i->...", a, optimize=do_opt)
            result = dpnp.einsum("...i->...", a_dp, optimize=do_opt)
            assert_dtype_allclose(result, expected)

            result = dpnp.einsum(
                a_dp, [Ellipsis, 0], [Ellipsis], optimize=do_opt
            )
            assert_dtype_allclose(result, expected)

        # sum(a, axis=0)
        for n in range(1, 17):
            a = numpy.arange(2 * n, dtype=dtype).reshape(2, n)
            a_dp = dpnp.array(a)
            expected = numpy.einsum("i...->...", a, optimize=do_opt)
            result = dpnp.einsum("i...->...", a_dp, optimize=do_opt)
            assert_dtype_allclose(result, expected)

            result = dpnp.einsum(
                a_dp, [0, Ellipsis], [Ellipsis], optimize=do_opt
            )
            assert_dtype_allclose(result, expected)

        for n in range(1, 17):
            a = numpy.arange(2 * 3 * n, dtype=dtype).reshape(2, 3, n)
            a_dp = dpnp.array(a)
            expected = numpy.einsum("i...->...", a, optimize=do_opt)
            result = dpnp.einsum("i...->...", a_dp, optimize=do_opt)
            assert_dtype_allclose(result, expected)

            result = dpnp.einsum(
                a_dp, [0, Ellipsis], [Ellipsis], optimize=do_opt
            )
            assert_dtype_allclose(result, expected)

        # trace(a)
        for n in range(1, 17):
            a = numpy.arange(n * n, dtype=dtype).reshape(n, n)
            a_dp = dpnp.array(a)
            expected = numpy.einsum("ii", a, optimize=do_opt)
            result = dpnp.einsum("ii", a_dp, optimize=do_opt)
            assert_dtype_allclose(result, expected)

            result = dpnp.einsum(a_dp, [0, 0], optimize=do_opt)
            assert_dtype_allclose(result, expected)

            # should accept dpnp array in subscript list
            dp_array = dpnp.asarray([0, 0])
            assert_dtype_allclose(
                dpnp.einsum(a_dp, dp_array, optimize=do_opt), expected
            )
            assert_dtype_allclose(
                dpnp.einsum(a_dp, list(dp_array), optimize=do_opt), expected
            )

        # multiply(a, b)
        for n in range(1, 17):
            a = numpy.arange(3 * n, dtype=dtype).reshape(3, n)
            b = numpy.arange(2 * 3 * n, dtype=dtype).reshape(2, 3, n)
            a_dp = dpnp.array(a)
            b_dp = dpnp.array(b)
            expected = numpy.einsum("..., ...", a, b, optimize=do_opt)
            result = dpnp.einsum("..., ...", a_dp, b_dp, optimize=do_opt)
            assert_dtype_allclose(result, expected)

            result = dpnp.einsum(
                a_dp, [Ellipsis], b_dp, [Ellipsis], optimize=do_opt
            )
            assert_dtype_allclose(result, expected)

        # inner(a,b)
        for n in range(1, 17):
            a = numpy.arange(2 * 3 * n, dtype=dtype).reshape(2, 3, n)
            b = numpy.arange(n, dtype=dtype)
            a_dp = dpnp.array(a)
            b_dp = dpnp.array(b)
            expected = numpy.einsum("...i, ...i", a, b, optimize=do_opt)
            result = dpnp.einsum("...i, ...i", a_dp, b_dp, optimize=do_opt)
            assert_dtype_allclose(result, expected)

            result = dpnp.einsum(
                a_dp, [Ellipsis, 0], b_dp, [Ellipsis, 0], optimize=do_opt
            )
            assert_dtype_allclose(result, expected)

        for n in range(1, 11):
            a = numpy.arange(n * 3 * 2, dtype=dtype).reshape(n, 3, 2)
            b = numpy.arange(n, dtype=dtype)
            a_dp = dpnp.array(a)
            b_dp = dpnp.array(b)
            expected = numpy.einsum("i..., i...", a, b, optimize=do_opt)
            result = dpnp.einsum("i..., i...", a_dp, b_dp, optimize=do_opt)
            assert_dtype_allclose(result, expected)

            result = dpnp.einsum(
                a_dp, [0, Ellipsis], b_dp, [0, Ellipsis], optimize=do_opt
            )
            assert_dtype_allclose(result, expected)

        # outer(a,b)
        for n in range(1, 17):
            a = numpy.arange(3, dtype=dtype) + 1
            b = numpy.arange(n, dtype=dtype) + 1
            a_dp = dpnp.array(a)
            b_dp = dpnp.array(b)
            expected = numpy.einsum("i,j", a, b, optimize=do_opt)
            assert_dtype_allclose(
                dpnp.einsum("i,j", a_dp, b_dp, optimize=do_opt), expected
            )
            assert_dtype_allclose(
                dpnp.einsum(a_dp, [0], b_dp, [1], optimize=do_opt), expected
            )

        # Suppress the complex warnings for the 'as f8' tests
        with suppress_warnings() as sup:
            sup.filter(numpy.exceptions.ComplexWarning)

            # matvec(a,b) / a.dot(b) where a is matrix, b is vector
            for n in range(1, 17):
                a = numpy.arange(4 * n, dtype=dtype).reshape(4, n)
                b = numpy.arange(n, dtype=dtype)
                a_dp = dpnp.array(a)
                b_dp = dpnp.array(b)
                expected = numpy.einsum("ij, j", a, b, optimize=do_opt)
                result = dpnp.einsum("ij, j", a_dp, b_dp, optimize=do_opt)
                assert_dtype_allclose(result, expected)

                result = dpnp.einsum(a_dp, [0, 1], b_dp, [1], optimize=do_opt)
                assert_dtype_allclose(result, expected)

                c = dpnp.arange(4, dtype=a_dp.dtype)
                args = ["ij, j", a_dp, b_dp]
                result = dpnp.einsum(
                    *args, out=c, dtype="f4", casting="unsafe", optimize=do_opt
                )
                assert result is c
                assert_dtype_allclose(result, expected)

                c[...] = 0
                args = [a_dp, [0, 1], b_dp, [1]]
                result = dpnp.einsum(
                    *args, out=c, dtype="f4", casting="unsafe", optimize=do_opt
                )
                assert result is c
                assert_dtype_allclose(result, expected)

            for n in range(1, 17):
                a = numpy.arange(4 * n, dtype=dtype).reshape(4, n)
                b = numpy.arange(n, dtype=dtype)
                a_dp = dpnp.array(a)
                b_dp = dpnp.array(b)
                expected = numpy.einsum("ji,j", a.T, b.T, optimize=do_opt)
                result = dpnp.einsum("ji,j", a_dp.T, b_dp.T, optimize=do_opt)
                assert_dtype_allclose(result, expected)

                result = dpnp.einsum(
                    a_dp.T, [1, 0], b_dp.T, [1], optimize=do_opt
                )
                assert_dtype_allclose(result, expected)

                c = dpnp.arange(4, dtype=a_dp.dtype)
                args = ["ji,j", a_dp.T, b_dp.T]
                result = dpnp.einsum(
                    *args, out=c, dtype="f4", casting="unsafe", optimize=do_opt
                )
                assert result is c
                assert_dtype_allclose(result, expected)

                c[...] = 0
                args = [a_dp.T, [1, 0], b_dp.T, [1]]
                result = dpnp.einsum(
                    *args, out=c, dtype="f4", casting="unsafe", optimize=do_opt
                )
                assert result is c
                assert_dtype_allclose(result, expected)

            # matmat(a,b) / a.dot(b) where a is matrix, b is matrix
            for n in range(1, 17):
                a = numpy.arange(4 * n, dtype=dtype).reshape(4, n)
                b = numpy.arange(n * 6, dtype=dtype).reshape(n, 6)
                a_dp = dpnp.array(a)
                b_dp = dpnp.array(b)
                expected = numpy.einsum("ij, jk", a, b, optimize=do_opt)
                result = dpnp.einsum("ij, jk", a_dp, b_dp, optimize=do_opt)
                assert_dtype_allclose(result, expected)

                result = dpnp.einsum(
                    a_dp, [0, 1], b_dp, [1, 2], optimize=do_opt
                )
                assert_dtype_allclose(result, expected)

            for n in range(1, 17):
                a = numpy.arange(4 * n, dtype=dtype).reshape(4, n)
                b = numpy.arange(n * 6, dtype=dtype).reshape(n, 6)
                c = numpy.arange(24, dtype=dtype).reshape(4, 6)
                a_dp = dpnp.array(a)
                b_dp = dpnp.array(b)
                d = dpnp.array(c)
                args = ["ij, jk", a, b]
                expected = numpy.einsum(
                    *args, out=c, dtype="f4", casting="unsafe", optimize=do_opt
                )
                args = ["ij, jk", a_dp, b_dp]
                result = dpnp.einsum(
                    *args, out=d, dtype="f4", casting="unsafe", optimize=do_opt
                )
                assert result is d
                assert_dtype_allclose(result, expected)

                d[...] = 0
                args = [a_dp, [0, 1], b_dp, [1, 2]]
                result = dpnp.einsum(
                    *args, out=d, dtype="f4", casting="unsafe", optimize=do_opt
                )
                assert result is d
                assert_dtype_allclose(result, expected)

            # matrix triple product
            a = numpy.arange(12, dtype=dtype).reshape(3, 4)
            b = numpy.arange(20, dtype=dtype).reshape(4, 5)
            c = numpy.arange(30, dtype=dtype).reshape(5, 6)
            a_dp = dpnp.array(a)
            b_dp = dpnp.array(b)
            c_dp = dpnp.array(c)
            # equivalent of a.dot(b).dot(c)
            # if optimize is True, NumPy does not respect the given dtype
            args = ["ij,jk,kl", a, b, c]
            expected = numpy.einsum(
                *args, dtype="f4", casting="unsafe", optimize=False
            )
            args = ["ij,jk,kl", a_dp, b_dp, c_dp]
            result = dpnp.einsum(
                *args, dtype="f4", casting="unsafe", optimize=do_opt
            )
            assert_dtype_allclose(result, expected)

            args = a_dp, [0, 1], b_dp, [1, 2], c_dp, [2, 3]
            result = dpnp.einsum(
                *args, dtype="f4", casting="unsafe", optimize=do_opt
            )
            assert_dtype_allclose(result, expected)

            d = numpy.arange(18, dtype=dtype).reshape(3, 6)
            d_dp = dpnp.array(d)
            args = ["ij,jk,kl", a, b, c]
            expected = numpy.einsum(
                *args, out=d, dtype="f4", casting="unsafe", optimize=do_opt
            )
            args = ["ij,jk,kl", a_dp, b_dp, c_dp]
            result = dpnp.einsum(
                *args, out=d_dp, dtype="f4", casting="unsafe", optimize=do_opt
            )
            assert result is d_dp
            assert_dtype_allclose(result, expected)

            d_dp[...] = 0
            args = [a_dp, [0, 1], b_dp, [1, 2], c_dp, [2, 3]]
            result = dpnp.einsum(
                *args, out=d_dp, dtype="f4", casting="unsafe", optimize=do_opt
            )
            assert result is d_dp
            assert_dtype_allclose(result, expected)

            # tensordot(a, b)
            a = numpy.arange(60, dtype=dtype).reshape(3, 4, 5)
            b = numpy.arange(24, dtype=dtype).reshape(4, 3, 2)
            a_dp = dpnp.array(a)
            b_dp = dpnp.array(b)
            # equivalent of numpy.tensordot(a, b, axes=([1, 0], [0, 1]))
            expected = numpy.einsum("ijk, jil -> kl", a, b, optimize=do_opt)
            result = dpnp.einsum("ijk, jil -> kl", a_dp, b_dp, optimize=do_opt)
            assert_dtype_allclose(result, expected)

            result = dpnp.einsum(
                a_dp, [0, 1, 2], b_dp, [1, 0, 3], optimize=do_opt
            )
            assert_dtype_allclose(result, expected)

            c = dpnp.arange(10, dtype=a_dp.dtype).reshape(5, 2)
            args = ["ijk, jil -> kl", a_dp, b_dp]
            result = dpnp.einsum(
                *args, out=c, dtype="f4", casting="unsafe", optimize=do_opt
            )
            assert result is c
            assert_dtype_allclose(result, expected)

            c[...] = 0
            args = [a_dp, [0, 1, 2], b_dp, [1, 0, 3]]
            result = dpnp.einsum(
                *args, out=c, dtype="f4", casting="unsafe", optimize=do_opt
            )
            assert result is c
            assert_dtype_allclose(result, expected)

        # logical_and(logical_and(a!=0, b!=0), c!=0)
        neg_val = -2 if dtype.kind != "u" else numpy.iinfo(dtype).max - 1
        a = numpy.array([1, 3, neg_val, 0, 12, 13, 0, 1], dtype=dtype)
        b = numpy.array([0, 3.5, 0, neg_val, 0, 1, 3, 12], dtype=dtype)
        c = numpy.array([True, True, False, True, True, False, True, True])
        a_dp = dpnp.array(a)
        b_dp = dpnp.array(b)
        c_dp = dpnp.array(c)
        expected = numpy.einsum(
            "i,i,i->i", a, b, c, dtype="?", casting="unsafe", optimize=do_opt
        )
        args = ["i,i,i->i", a_dp, b_dp, c_dp]
        result = dpnp.einsum(
            *args, dtype="?", casting="unsafe", optimize=do_opt
        )
        assert_dtype_allclose(result, expected)

        args = [a_dp, [0], b_dp, [0], c_dp, [0], [0]]
        result = dpnp.einsum(
            *args, dtype="?", casting="unsafe", optimize=do_opt
        )
        assert_dtype_allclose(result, expected)

        # NumPy >= 2.0 follows NEP-50 to determine the output dtype when one
        # of the inputs is a scalar while NumPy < 2.0 does not
        if numpy.lib.NumpyVersion(numpy.__version__) < "2.0.0":
            check_type = False
        else:
            check_type = True
        a = numpy.arange(9, dtype=dtype)
        a_dp = dpnp.array(a)
        expected = numpy.einsum(",i->", 3, a)
        assert_dtype_allclose(
            dpnp.einsum(",i->", 3, a_dp), expected, check_type=check_type
        )
        assert_dtype_allclose(
            dpnp.einsum(3, [], a_dp, [0], []), expected, check_type=check_type
        )

        expected = numpy.einsum("i,->", a, 3)
        assert_dtype_allclose(
            dpnp.einsum("i,->", a_dp, 3), expected, check_type=check_type
        )
        assert_dtype_allclose(
            dpnp.einsum(a_dp, [0], 3, [], []), expected, check_type=check_type
        )

        # Various stride0, contiguous, and SSE aligned variants
        for n in range(1, 25):
            a = numpy.arange(n, dtype=dtype)
            a_dp = dpnp.array(a)
            assert_dtype_allclose(
                dpnp.einsum("...,...", a_dp, a_dp, optimize=do_opt),
                numpy.einsum("...,...", a, a, optimize=do_opt),
            )
            assert_dtype_allclose(
                dpnp.einsum("i,i", a_dp, a_dp, optimize=do_opt),
                numpy.einsum("i,i", a, a, optimize=do_opt),
            )
            assert_dtype_allclose(
                dpnp.einsum("i,->i", a_dp, 2, optimize=do_opt),
                numpy.einsum("i,->i", a, 2, optimize=do_opt),
                check_type=check_type,
            )
            assert_dtype_allclose(
                dpnp.einsum(",i->i", 2, a_dp, optimize=do_opt),
                numpy.einsum(",i->i", 2, a, optimize=do_opt),
                check_type=check_type,
            )
            assert_dtype_allclose(
                dpnp.einsum("i,->", a_dp, 2, optimize=do_opt),
                numpy.einsum("i,->", a, 2, optimize=do_opt),
                check_type=check_type,
            )
            assert_dtype_allclose(
                dpnp.einsum(",i->", 2, a_dp, optimize=do_opt),
                numpy.einsum(",i->", 2, a, optimize=do_opt),
                check_type=check_type,
            )

            assert_dtype_allclose(
                dpnp.einsum("...,...", a_dp[1:], a_dp[:-1], optimize=do_opt),
                numpy.einsum("...,...", a[1:], a[:-1], optimize=do_opt),
            )
            assert_dtype_allclose(
                dpnp.einsum("i,i", a_dp[1:], a_dp[:-1], optimize=do_opt),
                numpy.einsum("i,i", a[1:], a[:-1], optimize=do_opt),
            )
            assert_dtype_allclose(
                dpnp.einsum("i,->i", a_dp[1:], 2, optimize=do_opt),
                numpy.einsum("i,->i", a[1:], 2, optimize=do_opt),
                check_type=check_type,
            )
            assert_dtype_allclose(
                dpnp.einsum(",i->i", 2, a_dp[1:], optimize=do_opt),
                numpy.einsum(",i->i", 2, a[1:], optimize=do_opt),
                check_type=check_type,
            )
            assert_dtype_allclose(
                dpnp.einsum("i,->", a_dp[1:], 2, optimize=do_opt),
                numpy.einsum("i,->", a[1:], 2, optimize=do_opt),
                check_type=check_type,
            )
            assert_dtype_allclose(
                dpnp.einsum(",i->", 2, a_dp[1:], optimize=do_opt),
                numpy.einsum(",i->", 2, a[1:], optimize=do_opt),
                check_type=check_type,
            )

        # special case
        a = numpy.arange(2) + 1
        b = numpy.arange(4).reshape(2, 2) + 3
        c = numpy.arange(4).reshape(2, 2) + 7
        a_dp = dpnp.array(a)
        b_dp = dpnp.array(b)
        c_dp = dpnp.array(c)
        assert_dtype_allclose(
            dpnp.einsum("z,mz,zm->", a_dp, b_dp, c_dp, optimize=do_opt),
            numpy.einsum("z,mz,zm->", a, b, c, optimize=do_opt),
        )

        # singleton dimensions broadcast
        a = numpy.ones((10, 2))
        b = numpy.ones((1, 2))
        a_dp = dpnp.array(a)
        b_dp = dpnp.array(b)
        assert_dtype_allclose(
            dpnp.einsum("ij,ij->j", a_dp, b_dp, optimize=do_opt),
            numpy.einsum("ij,ij->j", a, b, optimize=do_opt),
        )

        # a blas-compatible contraction broadcasting case
        a = numpy.array([2.0, 3.0])
        b = numpy.array([4.0])
        a_dp = dpnp.array(a)
        b_dp = dpnp.array(b)
        assert_dtype_allclose(
            dpnp.einsum("i, i", a_dp, b_dp, optimize=do_opt),
            numpy.einsum("i, i", a, b, optimize=do_opt),
        )

        # all-ones array
        a = numpy.ones((1, 5)) / 2
        b = numpy.ones((5, 5)) / 2
        a_dp = dpnp.array(a)
        b_dp = dpnp.array(b)
        assert_dtype_allclose(
            dpnp.einsum("...ij,...jk->...ik", a_dp, a_dp, optimize=do_opt),
            numpy.einsum("...ij,...jk->...ik", a, a, optimize=do_opt),
        )
        assert_dtype_allclose(
            dpnp.einsum("...ij,...jk->...ik", a_dp, b_dp, optimize=do_opt),
            numpy.einsum("...ij,...jk->...ik", a, b, optimize=do_opt),
        )

        # special case
        a = numpy.eye(2, dtype=dtype)
        b = numpy.ones(2, dtype=dtype)
        a_dp = dpnp.array(a)
        b_dp = dpnp.array(b)
        assert_dtype_allclose(  # contig_contig_outstride0_two
            dpnp.einsum("ji,i->", a_dp, b_dp, optimize=do_opt),
            numpy.einsum("ji,i->", a, b, optimize=do_opt),
        )
        assert_dtype_allclose(  # stride0_contig_outstride0_two
            dpnp.einsum("i,ij->", b_dp, a_dp, optimize=do_opt),
            numpy.einsum("i,ij->", b, a, optimize=do_opt),
        )
        assert_dtype_allclose(  # contig_stride0_outstride0_two
            dpnp.einsum("ij,i->", a_dp, b_dp, optimize=do_opt),
            numpy.einsum("ij,i->", a, b, optimize=do_opt),
        )

    def test_einsum_sums_int32(self):
        self.check_einsum_sums("i4")
        self.check_einsum_sums("i4", True)

    def test_einsum_sums_uint32(self):
        self.check_einsum_sums("u4")
        self.check_einsum_sums("u4", True)

    def test_einsum_sums_int64(self):
        self.check_einsum_sums("i8")

    def test_einsum_sums_uint64(self):
        self.check_einsum_sums("u8")

    def test_einsum_sums_float32(self):
        self.check_einsum_sums("f4")

    def test_einsum_sums_float64(self):
        self.check_einsum_sums("f8")
        self.check_einsum_sums("f8", True)

    def test_einsum_sums_cfloat64(self):
        self.check_einsum_sums("c8")
        self.check_einsum_sums("c8", True)

    def test_einsum_sums_cfloat128(self):
        self.check_einsum_sums("c16")

    def test_einsum_misc(self):
        for opt in [True, False]:
            a = numpy.ones((1, 2))
            b = numpy.ones((2, 2, 1))
            a_dp = dpnp.array(a)
            b_dp = dpnp.array(b)
            expected = numpy.einsum("ij...,j...->i...", a, b, optimize=opt)
            result = dpnp.einsum("ij...,j...->i...", a_dp, b_dp, optimize=opt)
            assert_dtype_allclose(result, expected)

            a = numpy.array([1, 2, 3])
            b = numpy.array([2, 3, 4])
            a_dp = dpnp.array(a)
            b_dp = dpnp.array(b)
            expected = numpy.einsum("...i,...i", a, b, optimize=True)
            result = dpnp.einsum("...i,...i", a_dp, b_dp, optimize=True)
            assert_dtype_allclose(result, expected)

            a = numpy.ones((5, 12, 4, 2, 3), numpy.int64)
            b = numpy.ones((5, 12, 11), numpy.int64)
            a_dp = dpnp.array(a)
            b_dp = dpnp.array(b)
            expected = numpy.einsum("ijklm,ijn,ijn->", a, b, b, optimize=opt)
            result1 = dpnp.einsum(
                "ijklm,ijn,ijn->", a_dp, b_dp, b_dp, optimize=opt
            )
            assert_dtype_allclose(result1, expected)
            result2 = dpnp.einsum("ijklm,ijn->", a_dp, b_dp, optimize=opt)
            assert_dtype_allclose(result2, expected)

            a = numpy.arange(1, 3)
            b = numpy.arange(1, 5).reshape(2, 2)
            c = numpy.arange(1, 9).reshape(4, 2)
            a_dp = dpnp.array(a)
            b_dp = dpnp.array(b)
            c_dp = dpnp.array(c)
            expected = numpy.einsum("x,yx,zx->xzy", a, b, c, optimize=opt)
            result = dpnp.einsum("x,yx,zx->xzy", a_dp, b_dp, c_dp, optimize=opt)
            assert_dtype_allclose(result, expected)

        # Ensure explicitly setting out=None does not cause an error
        a = numpy.array([1])
        b = numpy.array([2])
        a_dp = dpnp.array(a)
        b_dp = dpnp.array(b)
        expected = numpy.einsum("i,j", a, b, out=None)
        result = dpnp.einsum("i,j", a_dp, b_dp, out=None)
        assert_dtype_allclose(result, expected)

    def test_subscript_range(self):
        # make sure that all letters of Latin alphabet (both uppercase & lowercase) can be used
        # when creating a subscript from arrays
        a = dpnp.ones((2, 3))
        b = dpnp.ones((3, 4))
        dpnp.einsum(a, [0, 20], b, [20, 2], [0, 2])
        dpnp.einsum(a, [0, 27], b, [27, 2], [0, 2])
        dpnp.einsum(a, [0, 51], b, [51, 2], [0, 2])
        assert_raises(ValueError, dpnp.einsum, a, [0, 52], b, [52, 2], [0, 2])
        assert_raises(ValueError, dpnp.einsum, a, [-1, 5], b, [5, 2], [-1, 2])

    def test_einsum_broadcast(self):
        a = numpy.arange(2 * 3 * 4).reshape(2, 3, 4)
        b = numpy.arange(3)
        a_dp = dpnp.array(a)
        b_dp = dpnp.array(b)
        expected = numpy.einsum("ijk,j->ijk", a, b, optimize=False)
        result = dpnp.einsum("ijk,j->ijk", a_dp, b_dp, optimize=False)
        assert_dtype_allclose(result, expected)
        for opt in [True, False]:
            assert_dtype_allclose(
                dpnp.einsum("ij...,j...->ij...", a_dp, b_dp, optimize=opt),
                expected,
            )
            assert_dtype_allclose(
                dpnp.einsum("ij...,...j->ij...", a_dp, b_dp, optimize=opt),
                expected,
            )
            assert_dtype_allclose(
                dpnp.einsum("ij...,j->ij...", a_dp, b_dp, optimize=opt),
                expected,
            )

        a = numpy.arange(12).reshape((4, 3))
        b = numpy.arange(6).reshape((3, 2))
        a_dp = dpnp.array(a)
        b_dp = dpnp.array(b)
        expected = numpy.einsum("ik,kj->ij", a, b, optimize=False)
        result = dpnp.einsum("ik,kj->ij", a_dp, b_dp, optimize=False)
        assert_dtype_allclose(result, expected)
        for opt in [True, False]:
            assert_dtype_allclose(
                dpnp.einsum("ik...,k...->i...", a_dp, b_dp, optimize=opt),
                expected,
            )
            assert_dtype_allclose(
                dpnp.einsum("ik...,...kj->i...j", a_dp, b_dp, optimize=opt),
                expected,
            )
            assert_dtype_allclose(
                dpnp.einsum("...k,kj", a_dp, b_dp, optimize=opt), expected
            )
            assert_dtype_allclose(
                dpnp.einsum("ik,k...->i...", a_dp, b_dp, optimize=opt), expected
            )

        dims = [2, 3, 4, 5]
        a = numpy.arange(numpy.prod(dims)).reshape(dims)
        v = numpy.arange(4)
        a_dp = dpnp.array(a)
        v_dp = dpnp.array(v)
        expected = numpy.einsum("ijkl,k->ijl", a, v, optimize=False)
        result = dpnp.einsum("ijkl,k->ijl", a_dp, v_dp, optimize=False)
        assert_dtype_allclose(result, expected)
        for opt in [True, False]:
            assert_dtype_allclose(
                dpnp.einsum("ijkl,k", a_dp, v_dp, optimize=opt), expected
            )
            assert_dtype_allclose(
                dpnp.einsum("...kl,k", a_dp, v_dp, optimize=opt), expected
            )
            assert_dtype_allclose(
                dpnp.einsum("...kl,k...", a_dp, v_dp, optimize=opt), expected
            )

        J, K, M = 8, 8, 6
        a = numpy.arange(J * K * M).reshape(1, 1, 1, J, K, M)
        b = numpy.arange(J * K * M * 3).reshape(J, K, M, 3)
        a_dp = dpnp.array(a)
        b_dp = dpnp.array(b)
        expected = numpy.einsum("...lmn,...lmno->...o", a, b, optimize=False)
        result = dpnp.einsum("...lmn,...lmno->...o", a_dp, b_dp, optimize=False)
        assert_dtype_allclose(result, expected)
        for opt in [True, False]:
            assert_dtype_allclose(
                dpnp.einsum("...lmn,lmno->...o", a_dp, b_dp, optimize=opt),
                expected,
            )

    def test_einsum_stride(self):
        a = numpy.arange(2 * 3).reshape(2, 3).astype(numpy.float32)
        b = numpy.arange(2 * 3 * 2731).reshape(2, 3, 2731).astype(numpy.int16)
        a_dp = dpnp.array(a)
        b_dp = dpnp.array(b)
        expected = numpy.einsum("cl, cpx->lpx", a, b)
        result = dpnp.einsum("cl, cpx->lpx", a_dp, b_dp)
        assert_dtype_allclose(result, expected)

        a = numpy.arange(3 * 3).reshape(3, 3).astype(numpy.float64)
        b = numpy.arange(3 * 3 * 64 * 64)
        b = b.reshape(3, 3, 64, 64).astype(numpy.float32)
        a_dp = dpnp.array(a)
        b_dp = dpnp.array(b)
        expected = numpy.einsum("cl, cpxy->lpxy", a, b)
        result = dpnp.einsum("cl, cpxy->lpxy", a_dp, b_dp)
        assert_dtype_allclose(result, expected)

    def test_einsum_collapsing(self):
        x = numpy.random.normal(0, 1, (5, 5, 5, 5))
        y = numpy.zeros((5, 5))
        expected = numpy.einsum("aabb->ab", x, out=y)
        x_dp = dpnp.array(x)
        y_dp = dpnp.array(y)
        result = dpnp.einsum("aabb->ab", x_dp, out=y_dp)
        assert result is y_dp
        assert_dtype_allclose(result, expected)

    def test_einsum_tensor(self):
        tensor = numpy.random.random_sample((10, 10, 10, 10))
        tensor_dp = dpnp.array(tensor)
        expected = numpy.einsum("ijij->", tensor)
        result = dpnp.einsum("ijij->", tensor_dp)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_integer_float_dtypes())
    def test_different_paths(self, dtype):
        # Simple test, designed to exercise most specialized code paths,
        # note the +0.5 for floats.  This makes sure we use a float value
        # where the results must be exact.
        a = (numpy.arange(7) + 0.5).astype(dtype)
        s = numpy.array(2, dtype=dtype)

        a_dp = dpnp.asarray(a)
        s_dp = dpnp.asarray(s)

        # contig -> scalar:
        expected = numpy.einsum("i->", a)
        result = dpnp.einsum("i->", a_dp)
        assert_dtype_allclose(result, expected)

        # contig, contig -> contig:
        expected = numpy.einsum("i,i->i", a, a)
        result = dpnp.einsum("i,i->i", a_dp, a_dp)
        assert_dtype_allclose(result, expected)

        # noncontig, noncontig -> contig:
        expected = numpy.einsum("i,i->i", a.repeat(2)[::2], a.repeat(2)[::2])
        result = dpnp.einsum("i,i->i", a_dp.repeat(2)[::2], a_dp.repeat(2)[::2])
        assert_dtype_allclose(result, expected)

        # contig + contig -> scalar
        expected = numpy.einsum("i,i->", a, a)
        result = dpnp.einsum("i,i->", a_dp, a_dp)
        assert_dtype_allclose(result, expected)

        # contig + scalar -> contig (with out)
        out_dp = dpnp.ones(7, dtype=dtype)
        expected = numpy.einsum("i,->i", a, s)
        result = dpnp.einsum("i,->i", a_dp, s_dp, out=out_dp)
        assert result is out_dp
        assert_dtype_allclose(result, expected)

        # scalar + contig -> contig (with out)
        expected = numpy.einsum(",i->i", s, a)
        result = dpnp.einsum(",i->i", s_dp, a_dp)
        assert_dtype_allclose(result, expected)

        # scalar + contig -> scalar
        # Use einsum to compare to not have difference due to sum round-offs:
        result1 = dpnp.einsum(",i->", s_dp, a_dp)
        result2 = dpnp.einsum("i->", s_dp * a_dp)
        assert dpnp.allclose(result1, result2)

        # contig + scalar -> scalar
        # Use einsum to compare to not have difference due to sum round-offs:
        result3 = dpnp.einsum("i,->", a_dp, s_dp)
        assert dpnp.allclose(result2, result3)

        # contig + contig + contig -> scalar
        a = numpy.array([0.5, 0.5, 0.25, 4.5, 3.0], dtype=dtype)
        a_dp = dpnp.array(a)
        expected = numpy.einsum("i,i,i->", a, a, a)
        result = dpnp.einsum("i,i,i->", a_dp, a_dp, a_dp)
        assert_dtype_allclose(result, expected)

        # four arrays:
        expected = numpy.einsum("i,i,i,i->", a, a, a, a)
        result = dpnp.einsum("i,i,i,i->", a_dp, a_dp, a_dp, a_dp)
        assert_dtype_allclose(result, expected)

    def test_small_boolean_arrays(self):
        # Use array of True embedded in False.
        a = numpy.zeros((16, 1, 1), dtype=dpnp.bool)[:2]
        a[...] = True
        a_dp = dpnp.array(a)
        out_dp = dpnp.zeros((16, 1, 1), dtype=dpnp.bool)[:2]
        expected = numpy.einsum("...ij,...jk->...ik", a, a)
        result = dpnp.einsum("...ij,...jk->...ik", a_dp, a_dp, out=out_dp)
        assert result is out_dp
        assert_dtype_allclose(result, expected)

    def test_out_is_res(self):
        a = numpy.arange(9).reshape(3, 3)
        a_dp = dpnp.array(a)
        expected = numpy.einsum("...ij,...jk->...ik", a, a)
        result = dpnp.einsum("...ij,...jk->...ik", a_dp, a_dp, out=a_dp)
        assert result is a_dp
        assert_dtype_allclose(result, expected)

    def optimize_compare(self, subscripts, operands=None):
        # Setup for optimize einsum
        chars = "abcdefghij"
        sizes = numpy.array([2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3])
        global_size_dict = dict(zip(chars, sizes))

        # Tests all paths of the optimization function against
        # conventional einsum
        if operands is None:
            args = [subscripts]
            terms = subscripts.split("->")[0].split(",")
            for term in terms:
                dims = [global_size_dict[x] for x in term]
                args.append(numpy.random.rand(*dims))
        else:
            args = [subscripts] + operands

        dpnp_args = [args[0]]
        for arr in args[1:]:
            dpnp_args.append(dpnp.asarray(arr))

        expected = numpy.einsum(*args)
        # no optimization
        result = dpnp.einsum(*dpnp_args, optimize=False)
        assert_dtype_allclose(result, expected, factor=16)

        result = dpnp.einsum(*dpnp_args, optimize="greedy")
        assert_dtype_allclose(result, expected, factor=16)

        result = dpnp.einsum(*dpnp_args, optimize="optimal")
        assert_dtype_allclose(result, expected, factor=16)

    def test_hadamard_like_products(self):
        # Hadamard outer products
        self.optimize_compare("a,ab,abc->abc")
        self.optimize_compare("a,b,ab->ab")

    def test_index_transformations(self):
        # Simple index transformation cases
        self.optimize_compare("ea,fb,gc,hd,abcd->efgh")
        self.optimize_compare("ea,fb,abcd,gc,hd->efgh")
        self.optimize_compare("abcd,ea,fb,gc,hd->efgh")

    def test_complex(self):
        # Long test cases
        self.optimize_compare("acdf,jbje,gihb,hfac,gfac,gifabc,hfac")
        self.optimize_compare("acdf,jbje,gihb,hfac,gfac,gifabc,hfac")
        self.optimize_compare("cd,bdhe,aidb,hgca,gc,hgibcd,hgac")
        self.optimize_compare("abhe,hidj,jgba,hiab,gab")
        self.optimize_compare("bde,cdh,agdb,hica,ibd,hgicd,hiac")
        self.optimize_compare("chd,bde,agbc,hiad,hgc,hgi,hiad")
        self.optimize_compare("chd,bde,agbc,hiad,bdi,cgh,agdb")
        self.optimize_compare("bdhe,acad,hiab,agac,hibd")

    def test_collapse(self):
        # Inner products
        self.optimize_compare("ab,ab,c->")
        self.optimize_compare("ab,ab,c->c")
        self.optimize_compare("ab,ab,cd,cd->")
        self.optimize_compare("ab,ab,cd,cd->ac")
        self.optimize_compare("ab,ab,cd,cd->cd")
        self.optimize_compare("ab,ab,cd,cd,ef,ef->")

    def test_expand(self):
        # Outer products
        self.optimize_compare("ab,cd,ef->abcdef")
        self.optimize_compare("ab,cd,ef->acdf")
        self.optimize_compare("ab,cd,de->abcde")
        self.optimize_compare("ab,cd,de->be")
        self.optimize_compare("ab,bcd,cd->abcd")
        self.optimize_compare("ab,bcd,cd->abd")

    def test_edge_cases(self):
        # Difficult edge cases for optimization
        self.optimize_compare("eb,cb,fb->cef")
        self.optimize_compare("dd,fb,be,cdb->cef")
        self.optimize_compare("bca,cdb,dbf,afc->")
        self.optimize_compare("dcc,fce,ea,dbf->ab")
        self.optimize_compare("fdf,cdd,ccd,afe->ae")
        self.optimize_compare("abcd,ad")
        self.optimize_compare("ed,fcd,ff,bcf->be")
        self.optimize_compare("baa,dcf,af,cde->be")
        self.optimize_compare("bd,db,eac->ace")
        self.optimize_compare("fff,fae,bef,def->abd")
        self.optimize_compare("efc,dbc,acf,fd->abe")
        self.optimize_compare("ja,ac,da->jcd")

    def test_inner_product(self):
        # Inner products
        self.optimize_compare("ab,ab")
        self.optimize_compare("ac,ca")
        self.optimize_compare("abc,abc")
        self.optimize_compare("abc,bac")
        self.optimize_compare("abc,cba")

    def test_random_cases(self):
        # Randomly built test cases
        self.optimize_compare("aab,fa,df,ecc->bde")
        self.optimize_compare("ecb,fef,bad,ed->ac")
        self.optimize_compare("bcf,bbb,fbf,fc->")
        self.optimize_compare("bb,ff,be->e")
        self.optimize_compare("bcb,bb,fc,fff->")
        self.optimize_compare("fbb,dfd,fc,fc->")
        self.optimize_compare("afd,ea,cc,dc->ef")
        self.optimize_compare("adb,bc,fa,cfc->d")
        self.optimize_compare("bbd,bda,fc,db->acf")
        self.optimize_compare("dba,ead,cad->bce")
        self.optimize_compare("aef,fbc,dca->bde")

    def test_combined_views_mapping(self):
        a = dpnp.arange(9).reshape(1, 1, 3, 1, 3)
        expected = numpy.einsum("bbcdc->d", a.asnumpy())
        result = dpnp.einsum("bbcdc->d", a)
        assert_dtype_allclose(result, expected)

    def test_broadcasting_dot_cases(self):
        a = numpy.random.rand(1, 5, 4)
        b = numpy.random.rand(4, 6)
        c = numpy.random.rand(5, 6)
        d = numpy.random.rand(10)

        self.optimize_compare("ijk,kl,jl", operands=[a, b, c])
        self.optimize_compare("ijk,kl,jl,i->i", operands=[a, b, c, d])

        e = numpy.random.rand(1, 1, 5, 4)
        f = numpy.random.rand(7, 7)
        self.optimize_compare("abjk,kl,jl", operands=[e, b, c])
        self.optimize_compare("abjk,kl,jl,ab->ab", operands=[e, b, c, f])

        g = numpy.arange(64).reshape(2, 4, 8)
        self.optimize_compare("obk,ijk->ioj", operands=[g, g])

    def test_output_order(self):
        # Ensure output order is respected for optimize cases, the below
        # contraction should yield a reshaped tensor view
        a = dpnp.ones((2, 3, 5), order="F")
        b = dpnp.ones((4, 3), order="F")

        for opt in [True, False]:
            tmp = dpnp.einsum("...ft,mf->...mt", a, b, order="a", optimize=opt)
            assert tmp.flags.f_contiguous

            tmp = dpnp.einsum("...ft,mf->...mt", a, b, order="f", optimize=opt)
            assert tmp.flags.f_contiguous

            tmp = dpnp.einsum("...ft,mf->...mt", a, b, order="c", optimize=opt)
            assert tmp.flags.c_contiguous

            tmp = dpnp.einsum("...ft,mf->...mt", a, b, order="k", optimize=opt)
            assert tmp.flags.c_contiguous is False
            assert tmp.flags.f_contiguous is False

            tmp = dpnp.einsum("...ft,mf->...mt", a, b, order=None, optimize=opt)
            assert tmp.flags.c_contiguous is False
            assert tmp.flags.f_contiguous is False

            tmp = dpnp.einsum("...ft,mf->...mt", a, b, optimize=opt)
            assert tmp.flags.c_contiguous is False
            assert tmp.flags.f_contiguous is False

        c = dpnp.ones((4, 3), order="C")
        for opt in [True, False]:
            tmp = dpnp.einsum("...ft,mf->...mt", a, c, order="a", optimize=opt)
            assert tmp.flags.c_contiguous

        d = dpnp.ones((2, 3, 5), order="C")
        for opt in [True, False]:
            tmp = dpnp.einsum("...ft,mf->...mt", d, c, order="a", optimize=opt)
            assert tmp.flags.c_contiguous

    def test_einsum_path(self):
        # Test einsum path for covergae
        a = numpy.random.rand(1, 2, 3, 4)
        b = numpy.random.rand(4, 3, 2, 1)
        a_dp = dpnp.array(a)
        b_dp = dpnp.array(b)
        expected = numpy.einsum_path("ijkl,dcba->dcba", a, b)
        result = dpnp.einsum_path("ijkl,dcba->dcba", a_dp, b_dp)
        assert expected[0] == result[0]
        assert expected[1] == result[1]


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
        ids=["2D", "3D", "4D"],
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_inv(self, array, dtype):
        a = numpy.array(array, dtype=dtype)
        ia = dpnp.array(a)
        result = dpnp.linalg.inv(ia)
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

        a_dp = dpnp.array(a_np)

        # positive strides
        expected = numpy.linalg.inv(a_np[::2, ::2])
        result = dpnp.linalg.inv(a_dp[::2, ::2])
        assert_allclose(result, expected, rtol=1e-6)

        # negative strides
        expected = numpy.linalg.inv(a_np[::-2, ::-2])
        result = dpnp.linalg.inv(a_dp[::-2, ::-2])
        assert_allclose(result, expected, rtol=1e-6)

    @pytest.mark.parametrize(
        "shape",
        [(0, 0), (3, 0, 0), (0, 2, 2)],
        ids=["(0, 0)", "(3, 0, 0)", "(0, 2, 2)"],
    )
    def test_inv_empty(self, shape):
        a = numpy.empty(shape)
        ia = dpnp.array(a)
        result = dpnp.linalg.inv(ia)
        expected = numpy.linalg.inv(a)
        assert_dtype_allclose(result, expected)

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
        a_dp = dpnp.array(a_np)

        assert_raises(numpy.linalg.LinAlgError, numpy.linalg.inv, a_np)
        assert_raises(dpnp.linalg.LinAlgError, dpnp.linalg.inv, a_dp)

    # TODO: remove skipif when Intel MKL 2025.2 is released
    @pytest.mark.skipif(
        not requires_intel_mkl_version("2025.2"), reason="mkl<2025.2"
    )
    def test_inv_singular_matrix_3D(self):
        a_np = numpy.array(
            [[[1, 2], [3, 4]], [[1, 2], [1, 2]], [[1, 3], [3, 1]]]
        )
        a_dp = dpnp.array(a_np)

        assert_raises(numpy.linalg.LinAlgError, numpy.linalg.inv, a_np)
        assert_raises(dpnp.linalg.LinAlgError, dpnp.linalg.inv, a_dp)

    def test_inv_errors(self):
        a_dp = dpnp.array([[1, 2], [2, 5]], dtype="float32")

        # unsupported type
        a_np = dpnp.asnumpy(a_dp)
        assert_raises(TypeError, dpnp.linalg.inv, a_np)

        # a.ndim < 2
        a_dp_ndim_1 = a_dp.flatten()
        assert_raises(dpnp.linalg.LinAlgError, dpnp.linalg.inv, a_dp_ndim_1)

        # a is not square
        a_dp = dpnp.ones((2, 3))
        assert_raises(dpnp.linalg.LinAlgError, dpnp.linalg.inv, a_dp)


class TestLstsq:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize(
        "a_shape, b_shape",
        [
            ((2, 2), (2, 2)),
            ((2, 2), (2, 3)),
            ((2, 2), (2,)),
            ((4, 2), (4, 2)),
            ((4, 2), (4, 6)),
            ((4, 2), (4,)),
            ((2, 4), (2, 4)),
            ((2, 4), (2, 6)),
            ((2, 4), (2,)),
        ],
    )
    def test_lstsq(self, a_shape, b_shape, dtype):
        a_np = numpy.random.rand(*a_shape).astype(dtype)
        b_np = numpy.random.rand(*b_shape).astype(dtype)

        a_dp = dpnp.array(a_np)
        b_dp = dpnp.array(b_np)

        result = dpnp.linalg.lstsq(a_dp, b_dp)
        # if rcond is not set, FutureWarning is given.
        # By default Numpy uses None for calculations
        expected = numpy.linalg.lstsq(a_np, b_np, rcond=None)

        for param_dp, param_np in zip(result, expected):
            assert_dtype_allclose(param_dp, param_np)

    @pytest.mark.parametrize("a_dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("b_dtype", get_all_dtypes())
    def test_lstsq_diff_type(self, a_dtype, b_dtype):
        a_np = generate_random_numpy_array((2, 2), a_dtype)
        b_np = generate_random_numpy_array(2, b_dtype)
        a_dp = dpnp.array(a_np)
        b_dp = dpnp.array(b_np)

        # if rcond is not set, FutureWarning is given.
        # By default Numpy uses None for calculations
        expected = numpy.linalg.lstsq(a_np, b_np, rcond=None)
        result = dpnp.linalg.lstsq(a_dp, b_dp)

        for param_dp, param_np in zip(result, expected):
            assert_dtype_allclose(param_dp, param_np)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize(
        ["m", "n", "nrhs"],
        [(0, 4, 1), (0, 4, 2), (4, 0, 1), (4, 0, 2), (4, 2, 0), (0, 0, 0)],
    )
    def test_lstsq_empty(self, m, n, nrhs, dtype):
        a_np = numpy.arange(m * n).reshape(m, n).astype(dtype)
        b_np = numpy.ones((m, nrhs)).astype(dtype)

        a_dp = dpnp.array(a_np)
        b_dp = dpnp.array(b_np)

        result = dpnp.linalg.lstsq(a_dp, b_dp)
        # if rcond is not set, FutureWarning is given.
        # By default Numpy uses None for calculations
        expected = numpy.linalg.lstsq(a_np, b_np, rcond=None)

        for param_dp, param_np in zip(result, expected):
            assert_dtype_allclose(param_dp, param_np)

    def test_lstsq_errors(self):
        a_dp = dpnp.array([[1, 0.5], [0.5, 1]], dtype="float32")
        b_dp = dpnp.array(a_dp, dtype="float32")

        # diffetent queue
        a_queue = dpctl.SyclQueue()
        b_queue = dpctl.SyclQueue()
        a_dp_q = dpnp.array(a_dp, sycl_queue=a_queue)
        b_dp_q = dpnp.array(b_dp, sycl_queue=b_queue)
        assert_raises(ValueError, dpnp.linalg.lstsq, a_dp_q, b_dp_q)

        # unsupported type `a` and `b`
        a_np = dpnp.asnumpy(a_dp)
        b_np = dpnp.asnumpy(b_dp)
        assert_raises(TypeError, dpnp.linalg.lstsq, a_np, b_dp)
        assert_raises(TypeError, dpnp.linalg.lstsq, a_dp, b_np)

        # unsupported type `rcond`
        assert_raises(TypeError, dpnp.linalg.lstsq, a_dp, b_dp, [-1])


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
        a_dp = dpnp.array(a)

        result = dpnp.linalg.matrix_power(a_dp, power)
        expected = numpy.linalg.matrix_power(a, power)

        assert_dtype_allclose(result, expected)

    def test_matrix_power_errors(self):
        a_dp = dpnp.eye(4, dtype="float32")

        # unsupported type `a`
        a_np = dpnp.asnumpy(a_dp)
        assert_raises(TypeError, dpnp.linalg.matrix_power, a_np, 2)

        # unsupported type `power`
        assert_raises(TypeError, dpnp.linalg.matrix_power, a_dp, 1.5)
        assert_raises(TypeError, dpnp.linalg.matrix_power, a_dp, [2])

        # not invertible
        noninv = dpnp.array([[1, 0], [0, 0]])
        assert_raises(
            dpnp.linalg.LinAlgError, dpnp.linalg.matrix_power, noninv, -1
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
        a_dp = dpnp.array(a)

        np_rank = numpy.linalg.matrix_rank(a)
        dp_rank = dpnp.linalg.matrix_rank(a_dp)
        assert dp_rank.asnumpy() == np_rank

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
        a_dp = dpnp.array(a)

        np_rank = numpy.linalg.matrix_rank(a, hermitian=True)
        dp_rank = dpnp.linalg.matrix_rank(a_dp, hermitian=True)
        assert dp_rank.asnumpy() == np_rank

    @pytest.mark.parametrize(
        "high_tol, low_tol",
        [
            (0.99e-6, 1.01e-6),
            (numpy.array(0.99e-6), numpy.array(1.01e-6)),
            (numpy.array([0.99e-6]), numpy.array([1.01e-6])),
        ],
        ids=["float", "0-D array", "1-D array"],
    )
    def test_matrix_rank_tolerance(self, high_tol, low_tol):
        a = numpy.eye(4)
        a[-1, -1] = 1e-6
        a_dp = dpnp.array(a)

        if isinstance(high_tol, numpy.ndarray):
            dp_high_tol = dpnp.array(
                high_tol, usm_type=a_dp.usm_type, sycl_queue=a_dp.sycl_queue
            )
            dp_low_tol = dpnp.array(
                low_tol, usm_type=a_dp.usm_type, sycl_queue=a_dp.sycl_queue
            )
        else:
            dp_high_tol = high_tol
            dp_low_tol = low_tol

        np_rank_high_tol = numpy.linalg.matrix_rank(
            a, hermitian=True, tol=high_tol
        )
        dp_rank_high_tol = dpnp.linalg.matrix_rank(
            a_dp, hermitian=True, tol=dp_high_tol
        )
        assert dp_rank_high_tol.asnumpy() == np_rank_high_tol

        np_rank_low_tol = numpy.linalg.matrix_rank(
            a, hermitian=True, tol=low_tol
        )
        dp_rank_low_tol = dpnp.linalg.matrix_rank(
            a_dp, hermitian=True, tol=dp_low_tol
        )
        assert dp_rank_low_tol.asnumpy() == np_rank_low_tol

    # rtol kwarg was added in numpy 2.0
    @testing.with_requires("numpy>=2.0")
    @pytest.mark.parametrize(
        "tol",
        [0.99e-6, numpy.array(1.01e-6), numpy.ones(4) * [0.99e-6]],
        ids=["float", "0-D array", "1-D array"],
    )
    def test_matrix_rank_tol(self, tol):
        a = numpy.zeros((4, 3, 2))
        a_dp = dpnp.array(a)

        if isinstance(tol, numpy.ndarray):
            dp_tol = dpnp.array(
                tol, usm_type=a_dp.usm_type, sycl_queue=a_dp.sycl_queue
            )
        else:
            dp_tol = tol

        expected = numpy.linalg.matrix_rank(a, rtol=tol)
        result = dpnp.linalg.matrix_rank(a_dp, rtol=dp_tol)
        assert_dtype_allclose(result, expected)

        expected = numpy.linalg.matrix_rank(a, tol=tol)
        result = dpnp.linalg.matrix_rank(a_dp, tol=dp_tol)
        assert_dtype_allclose(result, expected)

    def test_matrix_rank_errors(self):
        a_dp = dpnp.array([[1, 2], [3, 4]], dtype="float32")

        # unsupported type `a`
        a_np = dpnp.asnumpy(a_dp)
        assert_raises(TypeError, dpnp.linalg.matrix_rank, a_np)

        # unsupported type `tol`
        tol = numpy.array(0.5, dtype="float32")
        assert_raises(TypeError, dpnp.linalg.matrix_rank, a_dp, tol)
        assert_raises(TypeError, dpnp.linalg.matrix_rank, a_dp, [0.5])

        # diffetent queue
        a_queue = dpctl.SyclQueue()
        tol_queue = dpctl.SyclQueue()
        a_dp_q = dpnp.array(a_dp, sycl_queue=a_queue)
        tol_dp_q = dpnp.array([0.5], dtype="float32", sycl_queue=tol_queue)
        assert_raises(
            ExecutionPlacementError,
            dpnp.linalg.matrix_rank,
            a_dp_q,
            tol_dp_q,
        )

        # both tol and rtol are given
        assert_raises(
            ValueError, dpnp.linalg.matrix_rank, a_dp, tol=1e-06, rtol=1e-04
        )


# numpy.linalg.matrix_transpose() is available since numpy >= 2.0
@testing.with_requires("numpy>=2.0")
# dpnp.linalg.matrix_transpose() calls dpnp.matrix_transpose()
# 1 test to increase code coverage
def test_matrix_transpose():
    a = numpy.arange(6).reshape((2, 3))
    a_dp = dpnp.array(a)

    expected = numpy.linalg.matrix_transpose(a)
    result = dpnp.linalg.matrix_transpose(a_dp)

    assert_allclose(result, expected)

    with assert_raises_regex(
        ValueError, "array must be at least 2-dimensional"
    ):
        dpnp.linalg.matrix_transpose(a_dp[:, 0])


class TestNorm:
    @pytest.mark.usefixtures("suppress_divide_numpy_warnings")
    @pytest.mark.parametrize(
        "shape", [(0,), (5, 0), (2, 0, 3)], ids=["(0,)", "(5, 0)", "(2, 0, 3)"]
    )
    @pytest.mark.parametrize("ord", [None, -2, -1, 0, 1, 2, 3])
    @pytest.mark.parametrize("axis", [0, None])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_empty(self, shape, ord, axis, keepdims):
        a = numpy.empty(shape)
        ia = dpnp.array(a)
        kwarg = {"ord": ord, "axis": axis, "keepdims": keepdims}

        if axis is None and a.ndim > 1 and ord in [0, 3]:
            # Invalid norm order for matrices (a.ndim == 2) or
            # Improper number of dimensions to norm (a.ndim>2)
            assert_raises(ValueError, dpnp.linalg.norm, ia, **kwarg)
            assert_raises(ValueError, numpy.linalg.norm, a, **kwarg)
        elif axis is None and a.ndim > 2 and ord is not None:
            # Improper number of dimensions to norm
            assert_raises(ValueError, dpnp.linalg.norm, ia, **kwarg)
            assert_raises(ValueError, numpy.linalg.norm, a, **kwarg)
        elif (
            axis is None
            and ord is not None
            and a.ndim != 1
            and a.shape[-1] == 0
        ):
            if ord in [-2, -1, 0, 3]:
                # reduction cannot be performed over zero-size axes
                assert_raises(ValueError, dpnp.linalg.norm, ia, **kwarg)
                assert_raises(ValueError, numpy.linalg.norm, a, **kwarg)
            else:
                if numpy_version() >= "2.3.0":
                    result = dpnp.linalg.norm(ia, **kwarg)
                    expected = numpy.linalg.norm(a, **kwarg)
                    assert_dtype_allclose(result, expected)
                else:
                    assert_equal(
                        dpnp.linalg.norm(ia, **kwarg), 0.0, strict=False
                    )
        else:
            result = dpnp.linalg.norm(ia, **kwarg)
            expected = numpy.linalg.norm(a, **kwarg)
            assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "ord", [None, -dpnp.inf, -2, -1, 0, 1, 2, 3, dpnp.inf]
    )
    @pytest.mark.parametrize("axis", [0, None])
    def test_0D(self, ord, axis):
        a = numpy.array(2)
        ia = dpnp.array(a)
        if axis is None and ord is not None:
            # Improper number of dimensions to norm
            assert_raises(ValueError, dpnp.linalg.norm, ia, ord=ord, axis=axis)
            assert_raises(ValueError, numpy.linalg.norm, a, ord=ord, axis=axis)
        elif axis is not None:
            assert_raises(IndexError, dpnp.linalg.norm, ia, ord=ord, axis=axis)
            assert_raises(AxisError, numpy.linalg.norm, a, ord=ord, axis=axis)
        else:
            result = dpnp.linalg.norm(ia, ord=ord, axis=axis)
            expected = numpy.linalg.norm(a, ord=ord, axis=axis)
            assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_divide_numpy_warnings")
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize(
        "ord", [None, -dpnp.inf, -2, -1, 0, 1, 2, 3.5, dpnp.inf]
    )
    @pytest.mark.parametrize("axis", [0, None])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_1D(self, dtype, ord, axis, keepdims):
        a = generate_random_numpy_array(10, dtype)
        ia = dpnp.array(a)

        result = dpnp.linalg.norm(ia, ord=ord, axis=axis, keepdims=keepdims)
        expected = numpy.linalg.norm(a, ord=ord, axis=axis, keepdims=keepdims)
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_divide_numpy_warnings")
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize(
        "ord", [None, -dpnp.inf, -2, -1, 1, 2, 3, dpnp.inf, "fro", "nuc"]
    )
    @pytest.mark.parametrize(
        "axis", [0, 1, (1, 0), None], ids=["0", "1", "(1, 0)", "None"]
    )
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_2D(self, dtype, ord, axis, keepdims):
        a = generate_random_numpy_array((3, 5), dtype)
        ia = dpnp.array(a)
        kwarg = {"ord": ord, "axis": axis, "keepdims": keepdims}

        if (axis in [-1, 0, 1] and ord in ["nuc", "fro"]) or (
            (isinstance(axis, tuple) or axis is None) and ord == 3
        ):
            # Invalid norm order for vectors
            assert_raises(ValueError, dpnp.linalg.norm, ia, **kwarg)
            assert_raises(ValueError, numpy.linalg.norm, a, **kwarg)
        else:
            result = dpnp.linalg.norm(ia, **kwarg)
            expected = numpy.linalg.norm(a, **kwarg)
            assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_divide_numpy_warnings")
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize(
        "ord", [None, -dpnp.inf, -2, -1, 1, 2, 3, dpnp.inf, "fro", "nuc"]
    )
    @pytest.mark.parametrize(
        "axis",
        [-1, 0, 1, (0, 1), (-1, -2), None],
        ids=["-1", "0", "1", "(0, 1)", "(-1, -2)", "None"],
    )
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_ND(self, dtype, ord, axis, keepdims):
        a = generate_random_numpy_array((2, 3, 4, 5), dtype)
        ia = dpnp.array(a)
        kwarg = {"ord": ord, "axis": axis, "keepdims": keepdims}

        if (axis in [-1, 0, 1] and ord in ["nuc", "fro"]) or (
            isinstance(axis, tuple) and ord == 3
        ):
            # Invalid norm order for vectors
            assert_raises(ValueError, dpnp.linalg.norm, ia, **kwarg)
            assert_raises(ValueError, numpy.linalg.norm, a, **kwarg)
        elif axis is None and ord is not None:
            # Improper number of dimensions to norm
            assert_raises(ValueError, dpnp.linalg.norm, ia, **kwarg)
            assert_raises(ValueError, numpy.linalg.norm, a, **kwarg)
        else:
            result = dpnp.linalg.norm(ia, **kwarg)
            expected = numpy.linalg.norm(a, **kwarg)
            assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_divide_numpy_warnings")
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize(
        "ord", [None, -dpnp.inf, -2, -1, 1, 2, 3, dpnp.inf, "fro", "nuc"]
    )
    @pytest.mark.parametrize(
        "axis",
        [-1, 0, 1, (0, 1), (-2, -1), None],
        ids=["-1", "0", "1", "(0, 1)", "(-2, -1)", "None"],
    )
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_usm_ndarray(self, dtype, ord, axis, keepdims):
        a = generate_random_numpy_array((2, 3, 4, 5), dtype)
        ia = dpt.asarray(a)
        kwarg = {"ord": ord, "axis": axis, "keepdims": keepdims}

        if (axis in [-1, 0, 1] and ord in ["nuc", "fro"]) or (
            isinstance(axis, tuple) and ord == 3
        ):
            # Invalid norm order for vectors
            assert_raises(ValueError, dpnp.linalg.norm, ia, **kwarg)
            assert_raises(ValueError, numpy.linalg.norm, a, **kwarg)
        elif axis is None and ord is not None:
            # Improper number of dimensions to norm
            assert_raises(ValueError, dpnp.linalg.norm, ia, **kwarg)
            assert_raises(ValueError, numpy.linalg.norm, a, **kwarg)
        else:
            result = dpnp.linalg.norm(ia, **kwarg)
            expected = numpy.linalg.norm(a, **kwarg)
            assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("stride", [3, -1, -5])
    def test_strided_1D(self, stride):
        a = numpy.arange(25)
        ia = dpnp.array(a)

        result = dpnp.linalg.norm(ia[::stride])
        expected = numpy.linalg.norm(a[::stride])
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "axis", [-1, 0, (0, 1), None], ids=["-1", "0", "(0, 1)", "None"]
    )
    @pytest.mark.parametrize(
        "stride",
        [(-2, -4), (2, 4), (-3, 5), (3, -1)],
        ids=["(-2, -4)", "(2, 4)", "(-3, 5)", "(3, -1)"],
    )
    def test_strided_2D(self, axis, stride):
        A = numpy.random.rand(20, 30)
        B = dpnp.asarray(A)
        slices = tuple(slice(None, None, stride[i]) for i in range(A.ndim))
        a, b = A[slices], B[slices]

        result = dpnp.linalg.norm(b, axis=axis)
        expected = numpy.linalg.norm(a, axis=axis)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "axis",
        [-1, 0, 1, 2, (0, 1), (-2, -1)],
        ids=["-1", "0", "1", "2", "(0, 1)", "(-1, -2)"],
    )
    @pytest.mark.parametrize(
        "stride",
        [(-2, -3, -1, -4), (-2, 4, -3, 5), (2, 3, 1, 4)],
        ids=["(-2, -3, -1, -4)", "(-2, 4, -3, 5)", "(2, 3, 1, 4)"],
    )
    def test_strided_ND(self, axis, stride):
        A = numpy.random.rand(12, 16, 20, 24)
        B = dpnp.asarray(A)
        slices = tuple(slice(None, None, stride[i]) for i in range(A.ndim))
        a, b = A[slices], B[slices]

        result = dpnp.linalg.norm(b, axis=axis)
        expected = numpy.linalg.norm(a, axis=axis)
        assert_dtype_allclose(result, expected)

    @testing.with_requires("numpy>=2.0")
    @pytest.mark.parametrize(
        "ord",
        [None, -dpnp.inf, -2, -1, 1, 2, dpnp.inf, "fro", "nuc"],
    )
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_matrix_norm(self, ord, keepdims):
        a = generate_random_numpy_array((3, 5))
        ia = dpnp.array(a)

        result = dpnp.linalg.matrix_norm(ia, ord=ord, keepdims=keepdims)
        expected = numpy.linalg.matrix_norm(a, ord=ord, keepdims=keepdims)
        assert_dtype_allclose(result, expected)

    @testing.with_requires("numpy>=2.3")
    @pytest.mark.parametrize("dtype", [dpnp.float32, dpnp.int32])
    @pytest.mark.parametrize(
        "shape, axis", [[(2, 0), None], [(2, 0), (0, 1)], [(0, 2), (0, 1)]]
    )
    @pytest.mark.parametrize("ord", [None, "fro", "nuc", 1, 2, dpnp.inf])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_matrix_norm_empty(self, dtype, shape, axis, ord, keepdims):
        a = numpy.zeros(shape, dtype=dtype)
        ia = dpnp.array(a)
        result = dpnp.linalg.norm(ia, axis=axis, ord=ord, keepdims=keepdims)
        expected = numpy.linalg.norm(a, axis=axis, ord=ord, keepdims=keepdims)
        assert_dtype_allclose(result, expected)

    @testing.with_requires("numpy>=2.3")
    @pytest.mark.parametrize("dtype", [dpnp.float32, dpnp.int32])
    @pytest.mark.parametrize("axis", [None, 0])
    @pytest.mark.parametrize("ord", [None, 1, 2, dpnp.inf])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_vector_norm_empty(self, dtype, axis, ord, keepdims):
        a = numpy.zeros(0, dtype=dtype)
        ia = dpnp.array(a)
        result = dpnp.linalg.vector_norm(
            ia, axis=axis, ord=ord, keepdims=keepdims
        )
        expected = numpy.linalg.vector_norm(
            a, axis=axis, ord=ord, keepdims=keepdims
        )
        assert_dtype_allclose(result, expected)
        if keepdims:
            # norm and vector_norm have different paths in dpnp when keepdims=True,
            # to cover both of them test with norm as well
            result = dpnp.linalg.norm(ia, axis=axis, ord=ord, keepdims=keepdims)
            assert_dtype_allclose(result, expected)

    @testing.with_requires("numpy>=2.0")
    @pytest.mark.parametrize(
        "ord", [None, -dpnp.inf, -2, -1, 0, 1, 2, 3.5, dpnp.inf]
    )
    def test_vector_norm_0D(self, ord):
        a = numpy.array(2)
        ia = dpnp.array(a)

        result = dpnp.linalg.vector_norm(ia, ord=ord)
        expected = numpy.linalg.vector_norm(a, ord=ord)
        assert_dtype_allclose(result, expected)

    @testing.with_requires("numpy>=2.0")
    @pytest.mark.parametrize(
        "ord", [None, -dpnp.inf, -2, -1, 0, 1, 2, 3.5, dpnp.inf]
    )
    @pytest.mark.parametrize("axis", [0, None])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_vector_norm_1D(self, ord, axis, keepdims):
        a = generate_random_numpy_array(10)
        ia = dpnp.array(a)
        kwarg = {"ord": ord, "axis": axis, "keepdims": keepdims}

        result = dpnp.linalg.vector_norm(ia, **kwarg)
        expected = numpy.linalg.vector_norm(a, **kwarg)
        assert_dtype_allclose(result, expected)

    @testing.with_requires("numpy>=2.0")
    @pytest.mark.usefixtures("suppress_divide_numpy_warnings")
    @pytest.mark.parametrize(
        "ord", [None, -dpnp.inf, -2, -1, 1, 2, 3.5, dpnp.inf]
    )
    @pytest.mark.parametrize(
        "axis",
        [-1, 0, (0, 1), (-1, -2), (0, 1, -2, -1), None],
        ids=["-1", "0", "(0, 1)", "(-1, -2)", "(0, 1, -2, -1)", "None"],
    )
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_vector_norm_ND(self, ord, axis, keepdims):
        a = numpy.arange(120).reshape(2, 3, 4, 5)
        ia = dpnp.array(a)
        kwarg = {"ord": ord, "axis": axis, "keepdims": keepdims}

        result = dpnp.linalg.vector_norm(ia, **kwarg)
        expected = numpy.linalg.vector_norm(a, **kwarg)
        assert_dtype_allclose(result, expected)

    def test_error(self):
        a = numpy.arange(120).reshape(2, 3, 4, 5)
        ia = dpnp.array(a)

        # Duplicate axes given
        assert_raises(ValueError, dpnp.linalg.norm, ia, axis=(2, 2))
        assert_raises(ValueError, numpy.linalg.norm, a, axis=(2, 2))

        #'axis' must be None, an integer or a tuple of integers
        assert_raises(TypeError, dpnp.linalg.norm, ia, axis=[2])
        assert_raises(TypeError, numpy.linalg.norm, a, axis=[2])

        # Invalid norm order for vectors
        assert_raises(ValueError, dpnp.linalg.norm, ia, axis=1, ord=[3])


class TestQr:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "shape",
        [
            (2, 1),
            (2, 2),
            (3, 4),
            (5, 3),
            (16, 16),
            (3, 3, 1),
            (2, 2, 2),
            (2, 4, 2),
            (2, 2, 4),
        ],
        ids=[
            "(2, 1)",
            "(2, 2)",
            "(3, 4)",
            "(5, 3)",
            "(16, 16)",
            "(3, 3, 1)",
            "(2, 2, 2)",
            "(2, 4, 2)",
            "(2, 2, 4)",
        ],
    )
    @pytest.mark.parametrize("mode", ["r", "raw", "complete", "reduced"])
    def test_qr(self, dtype, shape, mode):
        a = generate_random_numpy_array(shape, dtype, seed_value=81)
        ia = dpnp.array(a)

        if mode == "r":
            np_r = numpy.linalg.qr(a, mode)
            dpnp_r = dpnp.linalg.qr(ia, mode)
        else:
            np_q, np_r = numpy.linalg.qr(a, mode)

            # check decomposition
            if mode in ("complete", "reduced"):
                result = dpnp.linalg.qr(ia, mode)
                dpnp_q, dpnp_r = result.Q, result.R
                assert dpnp.allclose(
                    dpnp.matmul(dpnp_q, dpnp_r), ia, atol=1e-05
                )
            else:  # mode=="raw"
                dpnp_q, dpnp_r = dpnp.linalg.qr(ia, mode)
                assert_dtype_allclose(dpnp_q, np_q, factor=24)

        if mode in ("raw", "r"):
            assert_dtype_allclose(dpnp_r, np_r, factor=24)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "shape",
        [(32, 32), (8, 16, 16)],
        ids=["(32, 32)", "(8, 16, 16)"],
    )
    @pytest.mark.parametrize("mode", ["r", "raw", "complete", "reduced"])
    def test_qr_large(self, dtype, shape, mode):
        a = generate_random_numpy_array(shape, dtype, seed_value=81)
        ia = dpnp.array(a)

        if mode == "r":
            np_r = numpy.linalg.qr(a, mode)
            dpnp_r = dpnp.linalg.qr(ia, mode)
        else:
            np_q, np_r = numpy.linalg.qr(a, mode)

            # check decomposition
            if mode in ("complete", "reduced"):
                result = dpnp.linalg.qr(ia, mode)
                dpnp_q, dpnp_r = result.Q, result.R
                assert dpnp.allclose(dpnp.matmul(dpnp_q, dpnp_r), ia, atol=1e-5)
            else:  # mode=="raw"
                dpnp_q, dpnp_r = dpnp.linalg.qr(ia, mode)
                assert_allclose(dpnp_q, np_q, atol=1e-4)
        if mode in ("raw", "r"):
            assert_allclose(dpnp_r, np_r, atol=1e-4)

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
    @pytest.mark.parametrize("mode", ["r", "raw", "complete", "reduced"])
    def test_qr_empty(self, dtype, shape, mode):
        a = numpy.empty(shape, dtype=dtype)
        ia = dpnp.array(a)

        if mode == "r":
            np_r = numpy.linalg.qr(a, mode)
            dpnp_r = dpnp.linalg.qr(ia, mode)
        else:
            np_q, np_r = numpy.linalg.qr(a, mode)

            if mode in ("complete", "reduced"):
                result = dpnp.linalg.qr(ia, mode)
                dpnp_q, dpnp_r = result.Q, result.R
            else:
                dpnp_q, dpnp_r = dpnp.linalg.qr(ia, mode)

            assert_dtype_allclose(dpnp_q, np_q)

        assert_dtype_allclose(dpnp_r, np_r)

    @pytest.mark.parametrize("mode", ["r", "raw", "complete", "reduced"])
    def test_qr_strides(self, mode):
        a = generate_random_numpy_array((5, 5))
        ia = dpnp.array(a)

        # positive strides
        if mode == "r":
            np_r = numpy.linalg.qr(a[::2, ::2], mode)
            dpnp_r = dpnp.linalg.qr(ia[::2, ::2], mode)
        else:
            np_q, np_r = numpy.linalg.qr(a[::2, ::2], mode)

            if mode in ("complete", "reduced"):
                result = dpnp.linalg.qr(ia[::2, ::2], mode)
                dpnp_q, dpnp_r = result.Q, result.R
            else:
                dpnp_q, dpnp_r = dpnp.linalg.qr(ia[::2, ::2], mode)

            assert_dtype_allclose(dpnp_q, np_q)

        assert_dtype_allclose(dpnp_r, np_r)

        # negative strides
        if mode == "r":
            np_r = numpy.linalg.qr(a[::-2, ::-2], mode)
            dpnp_r = dpnp.linalg.qr(ia[::-2, ::-2], mode)
        else:
            np_q, np_r = numpy.linalg.qr(a[::-2, ::-2], mode)

            if mode in ("complete", "reduced"):
                result = dpnp.linalg.qr(ia[::-2, ::-2], mode)
                dpnp_q, dpnp_r = result.Q, result.R
            else:
                dpnp_q, dpnp_r = dpnp.linalg.qr(ia[::-2, ::-2], mode)

            assert_dtype_allclose(dpnp_q, np_q)

        assert_dtype_allclose(dpnp_r, np_r)

    def test_qr_errors(self):
        a_dp = dpnp.array([[1, 2], [3, 5]], dtype="float32")

        # unsupported type
        a_np = dpnp.asnumpy(a_dp)
        assert_raises(TypeError, dpnp.linalg.qr, a_np)

        # a.ndim < 2
        a_dp_ndim_1 = a_dp.flatten()
        assert_raises(dpnp.linalg.LinAlgError, dpnp.linalg.qr, a_dp_ndim_1)

        # invalid mode
        assert_raises(ValueError, dpnp.linalg.qr, a_dp, "c")


class TestSolve:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_solve(self, dtype):
        a_np = numpy.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        a_dp = dpnp.array(a_np)

        expected = numpy.linalg.solve(a_np, a_np)
        result = dpnp.linalg.solve(a_dp, a_dp)

        assert_allclose(result, expected)

    @testing.with_requires("numpy>=2.0")
    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    @pytest.mark.parametrize(
        "a_shape, b_shape",
        [
            ((4, 4), (2, 2, 4, 3)),
            ((2, 5, 5), (1, 5, 3)),
            ((2, 4, 4), (2, 2, 4, 2)),
            ((3, 2, 2), (3, 1, 2, 1)),
            ((2, 2, 2, 2, 2), (2,)),
            ((2, 2, 2, 2, 2), (2, 3)),
        ],
    )
    def test_solve_broadcast(self, a_shape, b_shape, dtype):
        a_np = generate_random_numpy_array(a_shape, dtype)
        b_np = generate_random_numpy_array(b_shape, dtype)

        a_dp = dpnp.array(a_np)
        b_dp = dpnp.array(b_np)

        expected = numpy.linalg.solve(a_np, b_np)
        result = dpnp.linalg.solve(a_dp, b_dp)

        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_solve_nrhs_greater_n(self, dtype):
        # Test checking the case when nrhs > n for
        # for a.shape = (n x n) and b.shape = (n x nrhs).
        a_np = numpy.array([[1, 2], [3, 5]], dtype=dtype)
        b_np = numpy.array([[1, 1, 1], [2, 2, 2]], dtype=dtype)

        a_dp = dpnp.array(a_np)
        b_dp = dpnp.array(b_np)

        expected = numpy.linalg.solve(a_np, b_np)
        result = dpnp.linalg.solve(a_dp, b_dp)

        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("a_dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize("b_dtype", get_all_dtypes(no_bool=True))
    def test_solve_diff_type(self, a_dtype, b_dtype):
        a_np = generate_random_numpy_array((2, 2), a_dtype)
        b_np = generate_random_numpy_array(2, b_dtype)
        a_dp = dpnp.array(a_np)
        b_dp = dpnp.array(b_np)

        expected = numpy.linalg.solve(a_np, b_np)
        result = dpnp.linalg.solve(a_dp, b_dp)

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

        a_dp = dpnp.array(a_np)
        b_dp = dpnp.array(b_np)

        # positive strides
        expected = numpy.linalg.solve(a_np[::2, ::2], b_np[::2])
        result = dpnp.linalg.solve(a_dp[::2, ::2], b_dp[::2])
        assert_allclose(result, expected, rtol=1e-6)

        # negative strides
        expected = numpy.linalg.solve(a_np[::-2, ::-2], b_np[::-2])
        result = dpnp.linalg.solve(a_dp[::-2, ::-2], b_dp[::-2])
        assert_allclose(result, expected)

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

        a_dp = dpnp.array(a_np)
        b_dp = dpnp.array(b_np)

        assert_raises(numpy.linalg.LinAlgError, numpy.linalg.solve, a_np, b_np)
        assert_raises(dpnp.linalg.LinAlgError, dpnp.linalg.solve, a_dp, b_dp)

    def test_solve_errors(self):
        a_dp = dpnp.array([[1, 0.5], [0.5, 1]], dtype="float32")
        b_dp = dpnp.array(a_dp, dtype="float32")

        # diffetent queue
        a_queue = dpctl.SyclQueue()
        b_queue = dpctl.SyclQueue()
        a_dp_q = dpnp.array(a_dp, sycl_queue=a_queue)
        b_dp_q = dpnp.array(b_dp, sycl_queue=b_queue)
        assert_raises(ValueError, dpnp.linalg.solve, a_dp_q, b_dp_q)

        # unsupported type
        a_np = dpnp.asnumpy(a_dp)
        b_np = dpnp.asnumpy(b_dp)
        assert_raises(TypeError, dpnp.linalg.solve, a_np, b_dp)
        assert_raises(TypeError, dpnp.linalg.solve, a_dp, b_np)

        # a.ndim < 2
        a_dp_ndim_1 = a_dp.flatten()
        assert_raises(
            dpnp.linalg.LinAlgError, dpnp.linalg.solve, a_dp_ndim_1, b_dp
        )

        # b.ndim == 0
        b_dp_ndim_0 = dpnp.array(2)
        assert_raises(ValueError, dpnp.linalg.solve, a_dp, b_dp_ndim_0)


class TestSlogdet:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_slogdet_2d(self, dtype):
        a_np = numpy.array([[1, 2], [3, 4]], dtype=dtype)
        a_dp = dpnp.array(a_np)

        sign_expected, logdet_expected = numpy.linalg.slogdet(a_np)
        result = dpnp.linalg.slogdet(a_dp)
        sign_result, logdet_result = result.sign, result.logabsdet

        assert_allclose(sign_result, sign_expected)
        assert_allclose(logdet_result, logdet_expected)

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
        a_dp = dpnp.array(a_np)

        sign_expected, logdet_expected = numpy.linalg.slogdet(a_np)
        result = dpnp.linalg.slogdet(a_dp)
        sign_result, logdet_result = result.sign, result.logabsdet

        assert_allclose(sign_result, sign_expected)
        assert_allclose(logdet_result, logdet_expected)

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

        a_dp = dpnp.array(a_np)

        # positive strides
        sign_expected, logdet_expected = numpy.linalg.slogdet(a_np[::2, ::2])
        result = dpnp.linalg.slogdet(a_dp[::2, ::2])
        sign_result, logdet_result = result.sign, result.logabsdet
        assert_allclose(sign_result, sign_expected)
        assert_allclose(logdet_result, logdet_expected)

        # negative strides
        sign_expected, logdet_expected = numpy.linalg.slogdet(a_np[::-2, ::-2])
        result = dpnp.linalg.slogdet(a_dp[::-2, ::-2])
        sign_result, logdet_result = result.sign, result.logabsdet
        assert_allclose(sign_result, sign_expected)
        assert_allclose(logdet_result, logdet_expected)

    # TODO: remove skipif when Intel MKL 2025.2 is released
    # Skip running on CPU because dpnp uses _getrf_batch only on CPU
    # for dpnp.linalg.det/slogdet.
    @pytest.mark.skipif(
        is_cpu_device() and not requires_intel_mkl_version("2025.2"),
        reason="mkl<2025.2",
    )
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
        a_dp = dpnp.array(a_np)

        sign_expected, logdet_expected = numpy.linalg.slogdet(a_np)
        result = dpnp.linalg.slogdet(a_dp)
        sign_result, logdet_result = result.sign, result.logabsdet

        assert_allclose(sign_result, sign_expected)
        assert_allclose(logdet_result, logdet_expected)

    # TODO: remove skipif when Intel MKL 2025.2 is released
    # Skip running on CPU because dpnp uses _getrf_batch only on CPU
    # for dpnp.linalg.det/slogdet.
    @pytest.mark.skipif(
        is_cpu_device() and not requires_intel_mkl_version("2025.2"),
        reason="mkl<2025.2",
    )
    def test_slogdet_singular_matrix_3D(self):
        a_np = numpy.array(
            [[[1, 2], [3, 4]], [[1, 2], [1, 2]], [[1, 3], [3, 1]]]
        )
        a_dp = dpnp.array(a_np)

        sign_expected, logdet_expected = numpy.linalg.slogdet(a_np)
        result = dpnp.linalg.slogdet(a_dp)
        sign_result, logdet_result = result.sign, result.logabsdet

        assert_allclose(sign_result, sign_expected)
        assert_allclose(logdet_result, logdet_expected)

    def test_slogdet_errors(self):
        a_dp = dpnp.array([[1, 2], [3, 5]], dtype="float32")

        # unsupported type
        a_np = dpnp.asnumpy(a_dp)
        assert_raises(TypeError, dpnp.linalg.slogdet, a_np)


class TestSvd:
    def get_tol(self, dtype):
        tol = 1e-06
        if dtype in (dpnp.float32, dpnp.complex64):
            tol = 1e-03
        elif not has_support_aspect64() and (
            dtype is None or dpnp.issubdtype(dtype, dpnp.integer)
        ):
            tol = 1e-03
        self._tol = tol

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
            dpnp_diag_s = dpnp.zeros_like(dp_a, dtype=dp_s.dtype)
            for i in range(min(dp_a.shape[-2], dp_a.shape[-1])):
                dpnp_diag_s[..., i, i] = dp_s[..., i]
                reconstructed = dpnp.dot(dp_u, dpnp.dot(dpnp_diag_s, dp_vt))

            assert dpnp.allclose(dp_a, reconstructed, rtol=tol, atol=1e-4)

        assert_allclose(dp_s, np_s, rtol=tol, atol=1e-03)

        if compute_vt:
            for i in range(min(dp_a.shape[-2], dp_a.shape[-1])):
                if np_u[..., 0, i] * dpnp.asnumpy(dp_u[..., 0, i]) < 0:
                    np_u[..., :, i] = -np_u[..., :, i]
                    np_vt[..., i, :] = -np_vt[..., i, :]
            for i in range(numpy.count_nonzero(np_s > tol)):
                assert_allclose(
                    dp_u[..., :, i],
                    np_u[..., :, i],
                    rtol=tol,
                    atol=tol,
                )
                assert_allclose(
                    dp_vt[..., i, :],
                    np_vt[..., i, :],
                    rtol=tol,
                    atol=tol,
                )

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "shape",
        [(2, 2), (3, 4), (5, 3), (16, 16)],
        ids=["(2, 2)", "(3, 4)", "(5, 3)", "(16, 16)"],
    )
    def test_svd(self, dtype, shape):
        a = numpy.arange(shape[0] * shape[1], dtype=dtype).reshape(shape)
        dp_a = dpnp.array(a)

        np_u, np_s, np_vh = numpy.linalg.svd(a)
        result = dpnp.linalg.svd(dp_a)
        dp_u, dp_s, dp_vh = result.U, result.S, result.Vh

        self.get_tol(dtype)
        self.check_decomposition(
            dp_a, dp_u, dp_s, dp_vh, np_u, np_s, np_vh, True
        )

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    @pytest.mark.parametrize("compute_vt", [True, False])
    @pytest.mark.parametrize(
        "shape", [(2, 2), (16, 16)], ids=["(2, 2)", "(16, 16)"]
    )
    def test_svd_hermitian(self, dtype, compute_vt, shape):
        a = generate_random_numpy_array(shape, dtype, hermitian=True)
        dp_a = dpnp.array(a)

        if compute_vt:
            np_u, np_s, np_vh = numpy.linalg.svd(
                a, compute_uv=compute_vt, hermitian=True
            )
            result = dpnp.linalg.svd(
                dp_a, compute_uv=compute_vt, hermitian=True
            )
            dp_u, dp_s, dp_vh = result.U, result.S, result.Vh
        else:
            np_s = numpy.linalg.svd(a, compute_uv=compute_vt, hermitian=True)
            dp_s = dpnp.linalg.svd(dp_a, compute_uv=compute_vt, hermitian=True)
            np_u = np_vh = dp_u = dp_vh = None

        self.get_tol(dtype)

        self.check_decomposition(
            dp_a, dp_u, dp_s, dp_vh, np_u, np_s, np_vh, compute_vt
        )

    def test_svd_errors(self):
        a_dp = dpnp.array([[1, 2], [3, 4]], dtype="float32")

        # unsupported type
        a_np = dpnp.asnumpy(a_dp)
        assert_raises(TypeError, dpnp.linalg.svd, a_np)

        # a.ndim < 2
        a_dp_ndim_1 = a_dp.flatten()
        assert_raises(dpnp.linalg.LinAlgError, dpnp.linalg.svd, a_dp_ndim_1)


# numpy.linalg.svdvals() is available since numpy >= 2.0
@testing.with_requires("numpy>=2.0")
class TestSvdvals:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "shape",
        [(3, 5), (4, 2), (2, 3, 3), (3, 5, 2)],
        ids=["(3, 5)", "(4, 2)", "(2, 3, 3)", "(3, 5, 2)"],
    )
    def test_svdvals(self, dtype, shape):
        a = numpy.arange(numpy.prod(shape), dtype=dtype).reshape(shape)
        dp_a = dpnp.array(a)

        expected = numpy.linalg.svdvals(a)
        result = dpnp.linalg.svdvals(dp_a)

        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "shape",
        [(0, 0), (1, 0, 0), (0, 2, 2)],
        ids=["(0, 0)", "(1, 0, 0)", "(0, 2, 2)"],
    )
    def test_svdvals_empty(self, shape):
        a = generate_random_numpy_array(shape, dpnp.default_float_type())
        dp_a = dpnp.array(a)

        expected = numpy.linalg.svdvals(a)
        result = dpnp.linalg.svdvals(dp_a)

        assert_dtype_allclose(result, expected)

    def test_svdvals_errors(self):
        a_dp = dpnp.array([[1, 2], [3, 4]], dtype="float32")

        # unsupported type
        a_np = dpnp.asnumpy(a_dp)
        assert_raises(TypeError, dpnp.linalg.svdvals, a_np)

        # a.ndim < 2
        a_dp_ndim_1 = a_dp.flatten()
        assert_raises(dpnp.linalg.LinAlgError, dpnp.linalg.svdvals, a_dp_ndim_1)


class TestPinv:
    def get_tol(self, dtype):
        tol = 1e-06
        if dtype in (dpnp.float32, dpnp.complex64):
            tol = 1e-03
        elif not has_support_aspect64() and (
            dtype is None or dpnp.issubdtype(dtype, dpnp.integer)
        ):
            tol = 1e-03
        self._tol = tol

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
        a = generate_random_numpy_array(shape, dtype, seed_value=81)
        a_dp = dpnp.array(a)

        B = numpy.linalg.pinv(a)
        B_dp = dpnp.linalg.pinv(a_dp)

        self.get_tol(dtype)
        tol = self._tol
        assert_allclose(B_dp, B, rtol=tol, atol=tol)

        if a.ndim == 2:
            reconstructed = dpnp.dot(a_dp, dpnp.dot(B_dp, a_dp))
        else:  # a.ndim > 2
            reconstructed = dpnp.matmul(a_dp, dpnp.matmul(B_dp, a_dp))

        assert dpnp.allclose(reconstructed, a_dp, rtol=tol, atol=tol)

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    @pytest.mark.parametrize(
        "shape", [(2, 2), (16, 16)], ids=["(2, 2)", "(16, 16)"]
    )
    def test_pinv_hermitian(self, dtype, shape):
        a = generate_random_numpy_array(
            shape, dtype, hermitian=True, seed_value=64, low=-5, high=5
        )
        a_dp = dpnp.array(a)

        B = numpy.linalg.pinv(a, hermitian=True)
        B_dp = dpnp.linalg.pinv(a_dp, hermitian=True)

        self.get_tol(dtype)
        tol = self._tol

        reconstructed = dpnp.dot(dpnp.dot(a_dp, B_dp), a_dp)
        assert dpnp.allclose(reconstructed, a_dp, rtol=tol, atol=tol)

    # rtol kwarg was added in numpy 2.0
    @testing.with_requires("numpy>=2.0")
    def test_pinv_rtol(self):
        a = numpy.ones((2, 2))
        a_dp = dpnp.array(a)

        expected = numpy.linalg.pinv(a, rtol=1e-15)
        result = dpnp.linalg.pinv(a_dp, rtol=1e-15)
        assert_dtype_allclose(result, expected)

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
        a_dp = dpnp.array(a)

        B = numpy.linalg.pinv(a)
        B_dp = dpnp.linalg.pinv(a_dp)

        assert_dtype_allclose(B_dp, B)

    def test_pinv_strides(self):
        a = generate_random_numpy_array((5, 5))
        a_dp = dpnp.array(a)

        # positive strides
        B = numpy.linalg.pinv(a[::2, ::2])
        B_dp = dpnp.linalg.pinv(a_dp[::2, ::2])
        assert_allclose(B_dp, B, rtol=1e-6, atol=1e-6)

        # negative strides
        B = numpy.linalg.pinv(a[::-2, ::-2])
        B_dp = dpnp.linalg.pinv(a_dp[::-2, ::-2])
        assert_allclose(B_dp, B, rtol=1e-6, atol=1e-6)

    def test_pinv_errors(self):
        a_dp = dpnp.array([[1, 2], [3, 4]], dtype="float32")

        # unsupported type `a`
        a_np = dpnp.asnumpy(a_dp)
        assert_raises(TypeError, dpnp.linalg.pinv, a_np)

        # unsupported type `rcond`
        rcond = numpy.array(0.5, dtype="float32")
        assert_raises(TypeError, dpnp.linalg.pinv, a_dp, rcond)
        assert_raises(TypeError, dpnp.linalg.pinv, a_dp, [0.5])

        # non-broadcastable `rcond`
        rcond_dp = dpnp.array([0.5], dtype="float32")
        assert_raises(ValueError, dpnp.linalg.pinv, a_dp, rcond_dp)

        # a.ndim < 2
        a_dp_ndim_1 = a_dp.flatten()
        assert_raises(dpnp.linalg.LinAlgError, dpnp.linalg.pinv, a_dp_ndim_1)

        # diffetent queue
        a_queue = dpctl.SyclQueue()
        rcond_queue = dpctl.SyclQueue()
        a_dp_q = dpnp.array(a_dp, sycl_queue=a_queue)
        rcond_dp_q = dpnp.array([0.5], dtype="float32", sycl_queue=rcond_queue)
        assert_raises(ValueError, dpnp.linalg.pinv, a_dp_q, rcond_dp_q)

        # both rcond and rtol are given
        assert_raises(
            ValueError, dpnp.linalg.pinv, a_dp, rcond=1e-06, rtol=1e-04
        )


class TestTensorinv:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize(
        "shape, ind",
        [((4, 6, 8, 3), 2), ((24, 8, 3), 1)],
        ids=["(4, 6, 8, 3)", "(24, 8, 3)"],
    )
    def test_tensorinv(self, dtype, shape, ind):
        a = numpy.eye(24, dtype=dtype).reshape(shape)
        a_dp = dpnp.array(a)

        ainv = numpy.linalg.tensorinv(a, ind=ind)
        ainv_dp = dpnp.linalg.tensorinv(a_dp, ind=ind)

        assert ainv.shape == ainv_dp.shape
        assert_dtype_allclose(ainv_dp, ainv)

    def test_test_tensorinv_errors(self):
        a_dp = dpnp.eye(24, dtype="float32").reshape(4, 6, 8, 3)

        # unsupported type `a`
        a_np = dpnp.asnumpy(a_dp)
        assert_raises(TypeError, dpnp.linalg.pinv, a_np)

        # unsupported type `ind`
        assert_raises(TypeError, dpnp.linalg.tensorinv, a_dp, 2.0)
        assert_raises(TypeError, dpnp.linalg.tensorinv, a_dp, [2.0])
        assert_raises(ValueError, dpnp.linalg.tensorinv, a_dp, -1)

        # non-square
        assert_raises(dpnp.linalg.LinAlgError, dpnp.linalg.tensorinv, a_dp, 1)


class TestTensorsolve:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize(
        "axes", [None, (1,), (2,)], ids=["None", "(1,)", "(2,)"]
    )
    def test_tensorsolve_axes(self, dtype, axes):
        a = numpy.eye(12).reshape(12, 3, 4).astype(dtype)
        b = numpy.ones(a.shape[0], dtype=dtype)

        a_dp = dpnp.array(a)
        b_dp = dpnp.array(b)

        res_np = numpy.linalg.tensorsolve(a, b, axes=axes)
        res_dp = dpnp.linalg.tensorsolve(a_dp, b_dp, axes=axes)

        assert res_np.shape == res_dp.shape
        assert_dtype_allclose(res_dp, res_np)

    def test_tensorsolve_errors(self):
        a_dp = dpnp.eye(24, dtype="float32").reshape(4, 6, 8, 3)
        b_dp = dpnp.ones(a_dp.shape[:2], dtype="float32")

        # unsupported type `a` and `b`
        a_np = dpnp.asnumpy(a_dp)
        b_np = dpnp.asnumpy(b_dp)
        assert_raises(TypeError, dpnp.linalg.tensorsolve, a_np, b_dp)
        assert_raises(TypeError, dpnp.linalg.tensorsolve, a_dp, b_np)

        # unsupported type `axes`
        assert_raises(TypeError, dpnp.linalg.tensorsolve, a_dp, 2.0)
        assert_raises(TypeError, dpnp.linalg.tensorsolve, a_dp, -2)

        # incorrect axes
        assert_raises(
            dpnp.linalg.LinAlgError, dpnp.linalg.tensorsolve, a_dp, b_dp, (1,)
        )
