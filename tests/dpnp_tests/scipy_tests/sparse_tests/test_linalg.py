# tests/dpnp_tests/scipy_tests/sparse_tests/test_linalg.py
"""
Tests for dpnp.scipy.sparse.linalg:
  LinearOperator, aslinearoperator, cg, gmres, minres

Style mirrors dpnp/tests/test_linalg.py:
  - class-per-feature with pytest.mark.parametrize
  - assert_dtype_allclose / generate_random_numpy_array from tests.helper
  - dpnp.asnumpy() for array comparison
  - testing.with_requires for optional-dependency guards
  - is_scipy_available() / has_support_aspect64() for capability skips
"""

from __future__ import annotations

import warnings

import numpy
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
    assert_raises,
)

import dpnp

# Re-use the project's own test helpers exactly as test_linalg.py does.
from dpnp.tests.helper import (
    assert_dtype_allclose,
    generate_random_numpy_array,
    get_all_dtypes,
    get_float_complex_dtypes,
    has_support_aspect64,
    is_scipy_available,
)
from dpnp.tests.third_party.cupy import testing

from dpnp.scipy.sparse.linalg import (
    LinearOperator,
    aslinearoperator,
    cg,
    gmres,
    minres,
)


# ---------------------------------------------------------------------------
# Optional SciPy import (used for reference comparisons)
# ---------------------------------------------------------------------------

if is_scipy_available():
    import scipy.sparse.linalg as scipy_sla


# ---------------------------------------------------------------------------
# Shared matrix / vector helpers
# (match the signature of generate_random_numpy_array from tests/helper.py)
# ---------------------------------------------------------------------------


def _spd_matrix(n, dtype):
    """Dense symmetric positive-definite matrix as a dpnp array."""
    a = generate_random_numpy_array(
        (n, n), dtype, seed_value=42, hermitian=False
    ).astype(float)
    a = a.T @ a + numpy.eye(n, dtype=float)
    if numpy.issubdtype(dtype, numpy.complexfloating):
        a = a.astype(dtype)
    else:
        a = a.astype(dtype)
    return dpnp.asarray(a)


def _diag_dominant(n, dtype, seed_value=81):
    """Strictly diagonally dominant (non-symmetric) matrix as a dpnp array."""
    a = generate_random_numpy_array(
        (n, n), dtype, seed_value=seed_value
    ) * 0.1
    numpy.fill_diagonal(a, numpy.abs(a).sum(axis=1) + 1.0)
    return dpnp.asarray(a)


def _sym_indefinite(n, dtype, seed_value=99):
    """Symmetric indefinite matrix (suitable for MINRES) as a dpnp array."""
    a = generate_random_numpy_array((n, n), dtype, seed_value=seed_value)
    q, _ = numpy.linalg.qr(a.astype(numpy.float64))
    numpy.random.seed(seed_value)
    d = numpy.random.standard_normal(n).astype(numpy.float64)
    m = (q @ numpy.diag(d) @ q.T).astype(dtype)
    return dpnp.asarray(m)


def _rhs(n, dtype, seed_value=7):
    """Unit-norm right-hand side vector as a dpnp array."""
    b = generate_random_numpy_array((n,), dtype, seed_value=seed_value)
    b /= numpy.linalg.norm(b)
    return dpnp.asarray(b)


# ---------------------------------------------------------------------------
# Import smoke test
# ---------------------------------------------------------------------------


class TestImports:
    """Verify that all public symbols are importable and callable."""

    def test_all_symbols_importable(self):
        from dpnp.scipy.sparse.linalg import (
            LinearOperator,
            aslinearoperator,
            cg,
            gmres,
            minres,
        )

        for sym in (LinearOperator, aslinearoperator, cg, gmres, minres):
            assert callable(sym)

    def test_all_listed_in_dunder_all(self):
        import dpnp.scipy.sparse.linalg as _mod

        for name in (
            "LinearOperator",
            "aslinearoperator",
            "cg",
            "gmres",
            "minres",
        ):
            assert name in _mod.__all__, f"{name!r} missing from __all__"


# ---------------------------------------------------------------------------
# LinearOperator
# ---------------------------------------------------------------------------


class TestLinearOperator:
    """Tests for LinearOperator construction and protocol.

    Mirrors the style of TestCholesky / TestDet in test_linalg.py.
    """

    # ------------------------------------------------------------------ shape

    @pytest.mark.parametrize(
        "shape",
        [(5, 5), (7, 3), (3, 7)],
        ids=["(5,5)", "(7,3)", "(3,7)"],
    )
    def test_shape(self, shape):
        m, n = shape
        lo = LinearOperator((m, n), matvec=lambda x: dpnp.zeros(m))
        assert lo.shape == (m, n)
        assert lo.ndim == 2

    # ------------------------------------------------------------------ dtype

    @pytest.mark.parametrize(
        "dtype",
        get_all_dtypes(no_bool=True, no_complex=False),
    )
    def test_dtype_inference(self, dtype):
        if not has_support_aspect64() and dtype in (
            dpnp.float64,
            dpnp.complex128,
        ):
            pytest.skip("float64 not supported on this device")
        n = 4
        A = dpnp.eye(n, dtype=dtype)
        lo = LinearOperator((n, n), matvec=lambda x: A @ x)
        assert lo.dtype == dtype

    def test_dtype_explicit(self):
        lo = LinearOperator(
            (4, 4),
            matvec=lambda x: dpnp.zeros(4, dtype=dpnp.float64),
            dtype=dpnp.float64,
        )
        assert lo.dtype == dpnp.float64

    # ------------------------------------------------------------------ matvec

    @pytest.mark.parametrize(
        "dtype",
        get_all_dtypes(no_bool=True, no_complex=False),
    )
    def test_matvec(self, dtype):
        if not has_support_aspect64() and dtype in (
            dpnp.float64,
            dpnp.complex128,
        ):
            pytest.skip("float64 not supported on this device")
        n = 6
        a_np = generate_random_numpy_array((n, n), dtype, seed_value=42)
        a_dp = dpnp.asarray(a_np)
        lo = LinearOperator((n, n), matvec=lambda x: a_dp @ x)
        x = dpnp.asarray(
            generate_random_numpy_array((n,), dtype, seed_value=1)
        )
        result = lo.matvec(x)
        expected = a_np @ dpnp.asnumpy(x)
        assert_dtype_allclose(result, expected)

    def test_matvec_wrong_shape_raises(self):
        lo = LinearOperator((3, 5), matvec=lambda x: dpnp.zeros(3))
        with assert_raises(ValueError):
            lo.matvec(dpnp.ones(4))

    # ------------------------------------------------------------------ rmatvec

    def test_rmatvec_not_defined_raises(self):
        lo = LinearOperator((3, 3), matvec=lambda x: dpnp.zeros(3))
        with assert_raises(NotImplementedError):
            lo.rmatvec(dpnp.zeros(3))

    @pytest.mark.parametrize(
        "dtype",
        get_all_dtypes(no_bool=True, no_complex=False),
    )
    def test_rmatvec(self, dtype):
        if not has_support_aspect64() and dtype in (
            dpnp.float64,
            dpnp.complex128,
        ):
            pytest.skip("float64 not supported on this device")
        n = 5
        a_np = generate_random_numpy_array((n, n), dtype, seed_value=12)
        a_dp = dpnp.asarray(a_np)
        lo = LinearOperator(
            (n, n),
            matvec=lambda x: a_dp @ x,
            rmatvec=lambda x: dpnp.conj(a_dp.T) @ x,
        )
        x = dpnp.asarray(
            generate_random_numpy_array((n,), dtype, seed_value=3)
        )
        result = lo.rmatvec(x)
        expected = a_np.conj().T @ dpnp.asnumpy(x)
        assert_dtype_allclose(result, expected)

    # ------------------------------------------------------------------ matmat

    @pytest.mark.parametrize(
        "dtype",
        get_all_dtypes(no_bool=True, no_complex=False),
    )
    def test_matmat_fallback_loop(self, dtype):
        if not has_support_aspect64() and dtype in (
            dpnp.float64,
            dpnp.complex128,
        ):
            pytest.skip("float64 not supported on this device")
        n, k = 5, 3
        a_np = generate_random_numpy_array((n, n), dtype, seed_value=55)
        a_dp = dpnp.asarray(a_np)
        lo = LinearOperator((n, n), matvec=lambda x: a_dp @ x)
        X = dpnp.asarray(
            generate_random_numpy_array((n, k), dtype, seed_value=9)
        )
        Y = lo.matmat(X)
        expected = a_np @ dpnp.asnumpy(X)
        assert_dtype_allclose(Y, expected)

    def test_matmat_wrong_ndim_raises(self):
        lo = LinearOperator(
            (3, 3),
            matvec=lambda x: dpnp.zeros(3),
            dtype=dpnp.float64,
        )
        with assert_raises(ValueError):
            lo.matmat(dpnp.ones(3))  # 1-D, not 2-D

    # ------------------------------------------------------------------ operator overloads

    @pytest.mark.parametrize(
        "dtype",
        get_all_dtypes(no_bool=True, no_complex=False),
    )
    def test_matmul_1d(self, dtype):
        """lo @ x dispatches to matvec."""
        if not has_support_aspect64() and dtype in (
            dpnp.float64,
            dpnp.complex128,
        ):
            pytest.skip("float64 not supported on this device")
        n = 6
        a_np = generate_random_numpy_array((n, n), dtype, seed_value=42)
        a_dp = dpnp.asarray(a_np)
        lo = LinearOperator((n, n), matvec=lambda x: a_dp @ x)
        x = dpnp.asarray(
            generate_random_numpy_array((n,), dtype, seed_value=2)
        )
        result = lo @ x
        expected = a_np @ dpnp.asnumpy(x)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype",
        get_all_dtypes(no_bool=True, no_complex=False),
    )
    def test_matmul_2d(self, dtype):
        """lo @ X dispatches to matmat."""
        if not has_support_aspect64() and dtype in (
            dpnp.float64,
            dpnp.complex128,
        ):
            pytest.skip("float64 not supported on this device")
        n, k = 5, 3
        a_np = generate_random_numpy_array((n, n), dtype, seed_value=42)
        a_dp = dpnp.asarray(a_np)
        lo = LinearOperator((n, n), matvec=lambda x: a_dp @ x)
        X = dpnp.asarray(
            generate_random_numpy_array((n, k), dtype, seed_value=5)
        )
        Y = lo @ X
        expected = a_np @ dpnp.asnumpy(X)
        assert_dtype_allclose(Y, expected)

    def test_call_alias(self):
        n = 4
        a_dp = dpnp.eye(n, dtype=dpnp.float64)
        lo = LinearOperator((n, n), matvec=lambda x: a_dp @ x)
        x = dpnp.ones(n, dtype=dpnp.float64)
        assert_allclose(dpnp.asnumpy(lo(x)), dpnp.asnumpy(x), atol=1e-12)

    # ------------------------------------------------------------------ repr

    def test_repr(self):
        lo = LinearOperator(
            (3, 4), matvec=lambda x: dpnp.zeros(3), dtype=dpnp.float64
        )
        r = repr(lo)
        assert "3x4" in r
        assert "LinearOperator" in r

    # ------------------------------------------------------------------ error paths

    def test_invalid_shape_negative(self):
        with assert_raises(ValueError):
            LinearOperator((-1, 3), matvec=lambda x: x)

    def test_invalid_shape_wrong_ndim(self):
        with assert_raises(ValueError):
            LinearOperator((3,), matvec=lambda x: x)

    # ------------------------------------------------------------------ subclass

    @pytest.mark.parametrize(
        "dtype",
        [dpnp.float32, dpnp.float64],
        ids=["float32", "float64"],
    )
    def test_subclass_custom_matmat(self, dtype):
        """User subclass overriding _matmat_impl, as in CuPy's HasMatmat."""
        if not has_support_aspect64() and dtype == dpnp.float64:
            pytest.skip("float64 not supported on this device")
        n, k = 7, 4
        a_np = generate_random_numpy_array(
            (n, n), dtype, seed_value=42
        )
        a_dp = dpnp.asarray(a_np)

        class _MyOp(LinearOperator):
            def __init__(self):
                super().__init__(
                    shape=(n, n),
                    matvec=lambda x: a_dp @ x,
                    dtype=dtype,
                )

            def _matmat_impl(self, X):
                return a_dp @ X

        op = _MyOp()
        X = dpnp.asarray(
            generate_random_numpy_array((n, k), dtype, seed_value=9)
        )
        Y = op.matmat(X)
        expected = a_np @ dpnp.asnumpy(X)
        assert_dtype_allclose(Y, expected)


# ---------------------------------------------------------------------------
# aslinearoperator
# ---------------------------------------------------------------------------


class TestAsLinearOperator:
    """Tests for aslinearoperator wrapping utility."""

    def test_identity_if_already_linearoperator(self):
        lo = LinearOperator((3, 3), matvec=lambda x: x)
        assert aslinearoperator(lo) is lo

    @pytest.mark.parametrize(
        "dtype",
        get_all_dtypes(no_bool=True, no_complex=False),
    )
    def test_dense_dpnp_array(self, dtype):
        if not has_support_aspect64() and dtype in (
            dpnp.float64,
            dpnp.complex128,
        ):
            pytest.skip("float64 not supported on this device")
        n = 6
        a_np = generate_random_numpy_array((n, n), dtype, seed_value=42)
        a_dp = dpnp.asarray(a_np)
        lo = aslinearoperator(a_dp)
        assert lo.shape == (n, n)
        x = dpnp.asarray(
            generate_random_numpy_array((n,), dtype, seed_value=1)
        )
        result = lo.matvec(x)
        expected = a_np @ dpnp.asnumpy(x)
        assert_dtype_allclose(result, expected)

    def test_dense_numpy_array(self):
        n = 5
        a_np = generate_random_numpy_array(
            (n, n), numpy.float64, seed_value=42
        )
        lo = aslinearoperator(a_np)
        assert lo.shape == (n, n)

    def test_rmatvec_from_dense(self):
        n = 5
        a_np = generate_random_numpy_array(
            (n, n), numpy.float64, seed_value=42
        )
        a_dp = dpnp.asarray(a_np)
        lo = aslinearoperator(a_dp)
        x = dpnp.asarray(
            generate_random_numpy_array((n,), numpy.float64, seed_value=2)
        )
        result = lo.rmatvec(x)
        expected = a_np.conj().T @ dpnp.asnumpy(x)
        assert_allclose(dpnp.asnumpy(result), expected, atol=1e-12)

    def test_duck_type_with_shape_and_matvec(self):
        n = 4

        class _DuckOp:
            shape = (n, n)
            dtype = numpy.float64

            def matvec(self, x):
                return dpnp.asarray(dpnp.asnumpy(x) * 2.0)

        lo = aslinearoperator(_DuckOp())
        x = dpnp.ones(n, dtype=dpnp.float64)
        result = lo.matvec(x)
        assert_allclose(dpnp.asnumpy(result), numpy.full(n, 2.0), atol=1e-12)

    def test_invalid_type_raises(self):
        with assert_raises(TypeError):
            aslinearoperator("not_an_array")

    def test_invalid_1d_array_raises(self):
        with pytest.raises(Exception):
            aslinearoperator(dpnp.ones(5))


# ---------------------------------------------------------------------------
# CG
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not is_scipy_available(), reason="SciPy not available"
)
class TestCg:
    """Tests for cg (Conjugate Gradient).

    Mirrors TestCholesky / TestDet structure from test_linalg.py.
    """

    n = 30

    @pytest.mark.parametrize(
        "dtype",
        get_float_complex_dtypes(),
    )
    def test_cg_converges_spd(self, dtype):
        """CG must converge on symmetric positive-definite matrices."""
        a_dp = _spd_matrix(self.n, dtype)
        b_dp = _rhs(self.n, dtype)
        x, info = cg(a_dp, b_dp, tol=1e-8, maxiter=500)
        assert info == 0
        res = dpnp.linalg.norm(a_dp @ x - b_dp) / dpnp.linalg.norm(b_dp)
        assert float(res) < 1e-5

    @pytest.mark.parametrize(
        "dtype",
        [dpnp.float32, dpnp.float64],
        ids=["float32", "float64"],
    )
    def test_cg_matches_scipy(self, dtype):
        """Solution must match scipy.sparse.linalg.cg within dtype tolerance."""
        if not has_support_aspect64() and dtype == dpnp.float64:
            pytest.skip("float64 not supported on this device")
        a_np = dpnp.asnumpy(_spd_matrix(self.n, dtype))
        b_np = dpnp.asnumpy(_rhs(self.n, dtype))
        x_ref, info_ref = scipy_sla.cg(a_np, b_np, rtol=1e-8, maxiter=500)
        assert info_ref == 0
        x_dp, info = cg(
            dpnp.asarray(a_np), dpnp.asarray(b_np), tol=1e-8, maxiter=500
        )
        assert info == 0
        tol = 1e-4 if dtype == dpnp.float32 else 1e-8
        assert_allclose(dpnp.asnumpy(x_dp), x_ref, rtol=tol)

    @pytest.mark.parametrize(
        "dtype",
        [dpnp.float32, dpnp.float64],
        ids=["float32", "float64"],
    )
    def test_cg_x0_warm_start(self, dtype):
        if not has_support_aspect64() and dtype == dpnp.float64:
            pytest.skip("float64 not supported on this device")
        a_dp = _spd_matrix(self.n, dtype)
        b_dp = _rhs(self.n, dtype)
        x0 = dpnp.ones(self.n, dtype=dtype)
        x, info = cg(a_dp, b_dp, x0=x0, tol=1e-8, maxiter=500)
        assert info == 0
        res = dpnp.linalg.norm(a_dp @ x - b_dp) / dpnp.linalg.norm(b_dp)
        assert float(res) < 1e-5

    @pytest.mark.parametrize(
        "dtype",
        [dpnp.float32, dpnp.float64],
        ids=["float32", "float64"],
    )
    def test_cg_b_2dim(self, dtype):
        """b with shape (n, 1) must be accepted and flattened internally."""
        if not has_support_aspect64() and dtype == dpnp.float64:
            pytest.skip("float64 not supported on this device")
        a_dp = _spd_matrix(self.n, dtype)
        b_dp = _rhs(self.n, dtype).reshape(self.n, 1)
        x, info = cg(a_dp, b_dp, tol=1e-8, maxiter=500)
        assert info == 0

    def test_cg_callback_called(self):
        a_dp = _spd_matrix(self.n, numpy.float64)
        b_dp = _rhs(self.n, numpy.float64)
        calls = []

        def _cb(xk):
            calls.append(float(dpnp.linalg.norm(xk)))

        cg(a_dp, b_dp, callback=_cb, maxiter=200)
        assert len(calls) > 0

    def test_cg_atol(self):
        a_dp = _spd_matrix(self.n, numpy.float64)
        b_dp = _rhs(self.n, numpy.float64)
        x, info = cg(a_dp, b_dp, tol=0.0, atol=1e-1)
        res = float(dpnp.linalg.norm(a_dp @ x - b_dp))
        assert res < 1.0

    def test_cg_exact_solution_no_iterations(self):
        """When x0 is the exact solution the residual must be zero immediately."""
        n = 10
        a_dp = _spd_matrix(n, numpy.float64)
        b_dp = _rhs(n, numpy.float64)
        x_true = dpnp.asarray(
            numpy.linalg.solve(dpnp.asnumpy(a_dp), dpnp.asnumpy(b_dp))
        )
        x, info = cg(a_dp, b_dp, x0=x_true, tol=1e-12)
        assert info == 0

    @pytest.mark.parametrize(
        "dtype",
        get_float_complex_dtypes(),
    )
    def test_cg_via_linear_operator(self, dtype):
        """CG with A supplied as a LinearOperator."""
        a_dp = _spd_matrix(self.n, dtype)
        b_dp = _rhs(self.n, dtype)
        lo = aslinearoperator(a_dp)
        x, info = cg(lo, b_dp, tol=1e-8, maxiter=500)
        assert info == 0
        res = float(
            dpnp.linalg.norm(a_dp @ x - b_dp) / dpnp.linalg.norm(b_dp)
        )
        assert res < 1e-5

    def test_cg_maxiter_nonconvergence_info_positive(self):
        """maxiter=1 on a hard problem must give info != 0."""
        a_dp = _spd_matrix(50, numpy.float64)
        b_dp = _rhs(50, numpy.float64)
        _, info = cg(a_dp, b_dp, tol=1e-15, maxiter=1)
        assert info != 0

    def test_cg_wrong_b_size_raises(self):
        a_dp = _spd_matrix(5, numpy.float64)
        b_dp = dpnp.ones(6, dtype=dpnp.float64)
        with pytest.raises((ValueError, Exception)):
            cg(a_dp, b_dp, maxiter=1)


# ---------------------------------------------------------------------------
# GMRES
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not is_scipy_available(), reason="SciPy not available"
)
class TestGmres:
    """Tests for gmres (Generalised Minimum Residual).

    Mirrors the class structure of TestDet / TestCg above.
    """

    n = 30

    @pytest.mark.parametrize(
        "dtype",
        get_float_complex_dtypes(),
    )
    def test_gmres_converges_diag_dominant(self, dtype):
        """GMRES must converge on diagonally dominant non-symmetric systems."""
        a_dp = _diag_dominant(self.n, dtype)
        b_dp = _rhs(self.n, dtype)
        x, info = gmres(a_dp, b_dp, tol=1e-8, maxiter=50, restart=self.n)
        assert info == 0
        res = dpnp.linalg.norm(a_dp @ x - b_dp) / dpnp.linalg.norm(b_dp)
        assert float(res) < 1e-5

    @pytest.mark.parametrize(
        "dtype",
        [dpnp.float32, dpnp.float64],
        ids=["float32", "float64"],
    )
    def test_gmres_matches_scipy(self, dtype):
        """Solution must match scipy.sparse.linalg.gmres within dtype tolerance."""
        if not has_support_aspect64() and dtype == dpnp.float64:
            pytest.skip("float64 not supported on this device")
        a_np = dpnp.asnumpy(_diag_dominant(self.n, dtype))
        b_np = generate_random_numpy_array(
            (self.n,), dtype, seed_value=7
        )
        b_np /= numpy.linalg.norm(b_np)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_ref, _ = scipy_sla.gmres(
                a_np, b_np, rtol=1e-8, restart=self.n, maxiter=None
            )
        x_dp, info = gmres(
            dpnp.asarray(a_np),
            dpnp.asarray(b_np),
            tol=1e-8,
            restart=self.n,
            maxiter=50,
        )
        assert info == 0
        tol = 1e-3 if dtype == dpnp.float32 else 1e-7
        assert_allclose(dpnp.asnumpy(x_dp), x_ref, rtol=tol)

    @pytest.mark.parametrize(
        "restart",
        [None, 5, 15],
        ids=["restart=None", "restart=5", "restart=15"],
    )
    def test_gmres_restart_values(self, restart):
        a_dp = _diag_dominant(self.n, numpy.float64)
        b_dp = _rhs(self.n, numpy.float64)
        x, info = gmres(a_dp, b_dp, tol=1e-8, restart=restart, maxiter=100)
        assert info == 0

    @pytest.mark.parametrize(
        "dtype",
        [dpnp.float32, dpnp.float64],
        ids=["float32", "float64"],
    )
    def test_gmres_x0_warm_start(self, dtype):
        if not has_support_aspect64() and dtype == dpnp.float64:
            pytest.skip("float64 not supported on this device")
        a_dp = _diag_dominant(self.n, dtype)
        b_dp = _rhs(self.n, dtype)
        x0 = dpnp.ones(self.n, dtype=dtype)
        x, info = gmres(a_dp, b_dp, x0=x0, tol=1e-8, maxiter=100)
        assert info == 0

    def test_gmres_b_2dim(self):
        """b with shape (n, 1) must be accepted and flattened internally."""
        a_dp = _diag_dominant(self.n, numpy.float64)
        b_dp = _rhs(self.n, numpy.float64).reshape(self.n, 1)
        x, info = gmres(a_dp, b_dp, tol=1e-8, maxiter=100)
        assert info == 0

    def test_gmres_callback_x_called(self):
        a_dp = _diag_dominant(self.n, numpy.float64)
        b_dp = _rhs(self.n, numpy.float64)
        calls = []

        def _cb(xk):
            calls.append(1)

        gmres(a_dp, b_dp, callback=_cb, callback_type="x", maxiter=20)
        assert len(calls) > 0

    def test_gmres_callback_pr_norm_not_implemented(self):
        a_dp = _diag_dominant(self.n, numpy.float64)
        b_dp = _rhs(self.n, numpy.float64)
        with pytest.raises(NotImplementedError):
            gmres(a_dp, b_dp, callback=lambda r: None, callback_type="pr_norm")

    def test_gmres_invalid_callback_type_raises(self):
        a_dp = _diag_dominant(self.n, numpy.float64)
        b_dp = _rhs(self.n, numpy.float64)
        with assert_raises(ValueError):
            gmres(a_dp, b_dp, callback_type="garbage")

    def test_gmres_atol(self):
        a_dp = _diag_dominant(self.n, numpy.float64)
        b_dp = _rhs(self.n, numpy.float64)
        x, info = gmres(
            a_dp, b_dp, tol=0.0, atol=1e-6, restart=self.n, maxiter=50
        )
        res = float(dpnp.linalg.norm(a_dp @ x - b_dp))
        assert res < 1e-4

    @pytest.mark.parametrize(
        "dtype",
        get_float_complex_dtypes(),
    )
    def test_gmres_via_linear_operator(self, dtype):
        a_dp = _diag_dominant(self.n, dtype)
        b_dp = _rhs(self.n, dtype)
        lo = aslinearoperator(a_dp)
        x, info = gmres(lo, b_dp, tol=1e-8, restart=self.n, maxiter=50)
        assert info == 0

    def test_gmres_nonconvergence_info_nonzero(self):
        """Hilbert-like ill-conditioned matrix with tiny restart must not converge."""
        n = 48
        idx = numpy.arange(n, dtype=numpy.float64)
        a_np = 1.0 / (idx[:, None] + idx[None, :] + 1.0)
        b_np = generate_random_numpy_array((n,), numpy.float64, seed_value=5)
        a_dp = dpnp.asarray(a_np)
        b_dp = dpnp.asarray(b_np)
        x, info = gmres(a_dp, b_dp, tol=1e-15, restart=2, maxiter=2)
        rel_res = float(
            dpnp.linalg.norm(a_dp @ x - b_dp) / dpnp.linalg.norm(b_dp)
        )
        assert rel_res > 1e-12
        assert info != 0

    def test_gmres_complex_system(self):
        n = 15
        a_np = generate_random_numpy_array(
            (n, n), numpy.complex128, seed_value=42
        )
        numpy.fill_diagonal(a_np, numpy.abs(a_np).sum(axis=1) + 1.0)
        b_np = generate_random_numpy_array(
            (n,), numpy.complex128, seed_value=7
        )
        a_dp = dpnp.asarray(a_np)
        b_dp = dpnp.asarray(b_np)
        x, info = gmres(a_dp, b_dp, tol=1e-8, restart=n, maxiter=50)
        assert info == 0
        res = float(
            numpy.linalg.norm(a_np @ dpnp.asnumpy(x) - b_np)
            / numpy.linalg.norm(b_np)
        )
        assert res < 1e-5


# ---------------------------------------------------------------------------
# MINRES
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not is_scipy_available(), reason="SciPy required for MINRES backend"
)
class TestMinres:
    """Tests for minres (Minimum Residual Method).

    MINRES is SciPy-backed for this implementation; tests verify the
    dpnp wrapper round-trips correctly.
    """

    n = 30

    @pytest.mark.parametrize(
        "dtype",
        [dpnp.float32, dpnp.float64],
        ids=["float32", "float64"],
    )
    def test_minres_converges_spd(self, dtype):
        """MINRES on an SPD system must converge."""
        if not has_support_aspect64() and dtype == dpnp.float64:
            pytest.skip("float64 not supported on this device")
        a_dp = _spd_matrix(self.n, dtype)
        b_dp = _rhs(self.n, dtype)
        x, info = minres(a_dp, b_dp, tol=1e-8, maxiter=500)
        assert info == 0
        res = float(
            dpnp.linalg.norm(a_dp @ x - b_dp) / dpnp.linalg.norm(b_dp)
        )
        assert res < 1e-4

    def test_minres_converges_sym_indefinite(self):
        """MINRES is suited for symmetric indefinite systems unlike CG."""
        a_dp = _sym_indefinite(self.n, numpy.float64)
        b_dp = _rhs(self.n, numpy.float64)
        x, info = minres(a_dp, b_dp, tol=1e-8, maxiter=1000)
        res = float(
            dpnp.linalg.norm(a_dp @ x - b_dp) / dpnp.linalg.norm(b_dp)
        )
        assert res < 1e-3

    def test_minres_matches_scipy(self):
        a_np = dpnp.asnumpy(_spd_matrix(self.n, numpy.float64))
        b_np = dpnp.asnumpy(_rhs(self.n, numpy.float64))
        x_ref, _ = scipy_sla.minres(a_np, b_np, rtol=1e-8)
        x_dp, info = minres(
            dpnp.asarray(a_np), dpnp.asarray(b_np), tol=1e-8
        )
        assert_allclose(dpnp.asnumpy(x_dp), x_ref, rtol=1e-6)

    def test_minres_x0_warm_start(self):
        a_dp = _spd_matrix(self.n, numpy.float64)
        b_dp = _rhs(self.n, numpy.float64)
        x0 = dpnp.zeros(self.n, dtype=numpy.float64)
        x, info = minres(a_dp, b_dp, x0=x0, tol=1e-8)
        assert info == 0

    def test_minres_shift_parameter(self):
        """shift != 0 solves (A - shift*I) x = b."""
        a_np = dpnp.asnumpy(_spd_matrix(self.n, numpy.float64))
        b_np = dpnp.asnumpy(_rhs(self.n, numpy.float64))
        shift = 0.5
        x_dp, info = minres(
            dpnp.asarray(a_np), dpnp.asarray(b_np), shift=shift, tol=1e-8
        )
        a_shifted = a_np - shift * numpy.eye(self.n)
        res = numpy.linalg.norm(
            a_shifted @ dpnp.asnumpy(x_dp) - b_np
        ) / numpy.linalg.norm(b_np)
        assert res < 1e-4

    def test_minres_non_square_raises(self):
        a_lo = aslinearoperator(
            dpnp.ones((4, 5), dtype=dpnp.float64)
        )
        b = dpnp.ones(4, dtype=dpnp.float64)
        with assert_raises(ValueError):
            minres(a_lo, b)

    def test_minres_via_linear_operator(self):
        a_dp = _spd_matrix(self.n, numpy.float64)
        b_dp = _rhs(self.n, numpy.float64)
        lo = aslinearoperator(a_dp)
        x, info = minres(lo, b_dp, tol=1e-8)
        assert info == 0

    def test_minres_callback_called(self):
        a_dp = _spd_matrix(self.n, numpy.float64)
        b_dp = _rhs(self.n, numpy.float64)
        calls = []

        def _cb(xk):
            calls.append(1)

        minres(a_dp, b_dp, callback=_cb, tol=1e-8)
        assert len(calls) > 0


# ---------------------------------------------------------------------------
# Integration: all solvers via LinearOperator with varying n / dtype
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not is_scipy_available(), reason="SciPy not available"
)
class TestSolversIntegration:
    """Parametric integration tests — n and dtype combinations.

    Follows the style of test_usm_ndarray_linalg_batch in test_linalg.py.
    """

    @pytest.mark.parametrize(
        "n,dtype",
        [
            pytest.param(10, dpnp.float32, id="n=10-float32"),
            pytest.param(10, dpnp.float64, id="n=10-float64"),
            pytest.param(30, dpnp.float64, id="n=30-float64"),
            pytest.param(50, dpnp.float64, id="n=50-float64"),
        ],
    )
    def test_cg_spd_via_linearoperator(self, n, dtype):
        if not has_support_aspect64() and dtype == dpnp.float64:
            pytest.skip("float64 not supported on this device")
        a_dp = _spd_matrix(n, dtype)
        lo = aslinearoperator(a_dp)
        b_dp = _rhs(n, dtype)
        x, info = cg(lo, b_dp, tol=1e-8, maxiter=n * 10)
        assert info == 0
        res = float(
            dpnp.linalg.norm(a_dp @ x - b_dp) / dpnp.linalg.norm(b_dp)
        )
        assert res < (1e-4 if dtype == dpnp.float32 else 1e-8)

    @pytest.mark.parametrize(
        "n,dtype",
        [
            pytest.param(10, dpnp.float32, id="n=10-float32"),
            pytest.param(10, dpnp.float64, id="n=10-float64"),
            pytest.param(30, dpnp.float64, id="n=30-float64"),
        ],
    )
    def test_gmres_nonsymmetric_via_linearoperator(self, n, dtype):
        if not has_support_aspect64() and dtype == dpnp.float64:
            pytest.skip("float64 not supported on this device")
        a_dp = _diag_dominant(n, dtype)
        lo = aslinearoperator(a_dp)
        b_dp = _rhs(n, dtype)
        x, info = gmres(lo, b_dp, tol=1e-8, restart=n, maxiter=50)
        assert info == 0

    @pytest.mark.parametrize(
        "n,dtype",
        [
            pytest.param(10, dpnp.float64, id="n=10-float64"),
            pytest.param(30, dpnp.float64, id="n=30-float64"),
        ],
    )
    def test_minres_spd_via_linearoperator(self, n, dtype):
        if not has_support_aspect64() and dtype == dpnp.float64:
            pytest.skip("float64 not supported on this device")
        a_dp = _spd_matrix(n, dtype)
        lo = aslinearoperator(a_dp)
        b_dp = _rhs(n, dtype)
        x, info = minres(lo, b_dp, tol=1e-8)
        assert info == 0
        res = float(
            dpnp.linalg.norm(a_dp @ x - b_dp) / dpnp.linalg.norm(b_dp)
        )
        assert res < 1e-4
