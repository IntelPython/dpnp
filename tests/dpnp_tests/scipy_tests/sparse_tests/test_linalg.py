# tests/dpnp_tests/scipy_tests/sparse_tests/test_linalg.py
"""
Comprehensive tests for dpnp.scipy.sparse.linalg:
  LinearOperator, aslinearoperator, cg, gmres, minres

Modeled after CuPy's cupyx_tests/scipy_tests/sparse_tests/test_linalg.py,
adapted for the dpnp testing environment (no cupy.testing harness).

Requirements:
    pytest >= 7.0
    numpy
    scipy
    dpnp
"""

from __future__ import annotations

import warnings
import numpy
import pytest

try:
    import scipy.sparse
    import scipy.sparse.linalg as scipy_sla
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

import dpnp
from dpnp.scipy.sparse.linalg import (
    LinearOperator,
    aslinearoperator,
    cg,
    gmres,
    minres,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_RNG = numpy.random.default_rng(42)


def _spd_matrix(n, dtype):
    """Return a dense symmetric positive-definite dpnp array."""
    a = _RNG.standard_normal((n, n)).astype(dtype)
    a = a.T @ a + numpy.eye(n, dtype=dtype)
    return dpnp.asarray(a)


def _diag_dominant(n, dtype, rng=None):
    """Return a strictly diagonally dominant (non-symmetric) dpnp array."""
    rng = rng or _RNG
    a = rng.standard_normal((n, n)).astype(dtype)
    a = a * 0.1
    numpy.fill_diagonal(a, numpy.abs(a).sum(axis=1) + 1.0)
    return dpnp.asarray(a)


def _sym_indefinite(n, dtype):
    """Return a symmetric indefinite dpnp array (for MINRES)."""
    q, _ = numpy.linalg.qr(_RNG.standard_normal((n, n)).astype(dtype))
    d = _RNG.standard_normal(n).astype(dtype)
    return dpnp.asarray(q @ numpy.diag(d) @ q.T)


def _rhs(n, dtype):
    b = _RNG.standard_normal(n).astype(dtype)
    b /= numpy.linalg.norm(b)
    return dpnp.asarray(b)


def _ref_solve(A_np, b_np):
    return numpy.linalg.solve(A_np, b_np)


# ---------------------------------------------------------------------------
# ─── LinearOperator ──────────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class TestLinearOperatorBasic:
    """Basic constructor, properties, and protocol tests."""

    @pytest.mark.parametrize("m,n", [(5, 5), (7, 3), (3, 7)])
    def test_shape(self, m, n):
        lo = LinearOperator((m, n), matvec=lambda x: dpnp.zeros(m))
        assert lo.shape == (m, n)
        assert lo.ndim == 2

    def test_dtype_inference(self):
        A = dpnp.eye(4, dtype=dpnp.float32)
        lo = LinearOperator((4, 4), matvec=lambda x: A @ x)
        assert lo.dtype == dpnp.float32

    def test_dtype_explicit(self):
        lo = LinearOperator(
            (4, 4), matvec=lambda x: dpnp.zeros(4, dtype=dpnp.float64),
            dtype=dpnp.float64)
        assert lo.dtype == dpnp.float64

    def test_matvec_shape_check(self):
        lo = LinearOperator((3, 5), matvec=lambda x: dpnp.zeros(3))
        x_bad = dpnp.ones(4)
        with pytest.raises(ValueError):
            lo.matvec(x_bad)

    def test_matmat_fallback_loop(self):
        n = 4
        A_np = numpy.eye(n, dtype=numpy.float64)
        A_dp = dpnp.asarray(A_np)
        lo = LinearOperator((n, n), matvec=lambda x: A_dp @ x)
        X = dpnp.asarray(_RNG.standard_normal((n, 3)))
        Y = lo.matmat(X)
        numpy.testing.assert_allclose(
            dpnp.asnumpy(Y), dpnp.asnumpy(X), atol=1e-12)

    def test_rmatvec_raises_if_not_defined(self):
        lo = LinearOperator((3, 3), matvec=lambda x: dpnp.zeros(3))
        with pytest.raises(NotImplementedError):
            lo.rmatvec(dpnp.zeros(3))

    def test_rmatvec_defined(self):
        n = 5
        A_np = _RNG.standard_normal((n, n))
        A_dp = dpnp.asarray(A_np)
        lo = LinearOperator(
            (n, n),
            matvec=lambda x: A_dp @ x,
            rmatvec=lambda x: dpnp.conj(A_dp.T) @ x,
        )
        x = dpnp.asarray(_RNG.standard_normal(n))
        y_dpnp = dpnp.asnumpy(lo.rmatvec(x))
        y_ref = A_np.conj().T @ dpnp.asnumpy(x)
        numpy.testing.assert_allclose(y_dpnp, y_ref, atol=1e-12)

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64,
                                        numpy.complex64, numpy.complex128])
    def test_matmul_operator(self, dtype):
        n = 6
        A_np = _RNG.standard_normal((n, n)).astype(dtype)
        A_dp = dpnp.asarray(A_np)
        lo = LinearOperator((n, n), matvec=lambda x: A_dp @ x)
        x = dpnp.asarray(_RNG.standard_normal(n).astype(dtype))
        result = lo @ x
        expected = A_np @ dpnp.asnumpy(x)
        numpy.testing.assert_allclose(
            dpnp.asnumpy(result), expected,
            rtol=1e-5 if dtype in (numpy.float32, numpy.complex64) else 1e-12)

    def test_matmul_2d(self):
        n, k = 5, 3
        A_np = _RNG.standard_normal((n, n))
        A_dp = dpnp.asarray(A_np)
        lo = LinearOperator((n, n), matvec=lambda x: A_dp @ x)
        X = dpnp.asarray(_RNG.standard_normal((n, k)))
        Y = lo @ X
        expected = A_np @ dpnp.asnumpy(X)
        numpy.testing.assert_allclose(dpnp.asnumpy(Y), expected, atol=1e-12)

    def test_call_alias(self):
        n = 4
        A_dp = dpnp.eye(n, dtype=dpnp.float64)
        lo = LinearOperator((n, n), matvec=lambda x: A_dp @ x)
        x = dpnp.ones(n)
        numpy.testing.assert_allclose(
            dpnp.asnumpy(lo(x)), dpnp.asnumpy(x), atol=1e-12)

    def test_repr(self):
        lo = LinearOperator((3, 4), matvec=lambda x: dpnp.zeros(3),
                            dtype=dpnp.float64)
        r = repr(lo)
        assert "3x4" in r
        assert "LinearOperator" in r

    def test_invalid_shape_negative(self):
        with pytest.raises(ValueError):
            LinearOperator((-1, 3), matvec=lambda x: x)

    def test_invalid_shape_wrong_ndim(self):
        with pytest.raises(ValueError):
            LinearOperator((3,), matvec=lambda x: x)


class TestLinearOperatorSubclass:
    """Test user-defined subclasses with _matvec / _matmat overrides,
    mirroring CuPy's HasMatvec / HasMatmat pattern."""

    def _build_A(self, n, dtype):
        A_np = _RNG.standard_normal((n, n)).astype(dtype)
        return A_np, dpnp.asarray(A_np)

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    def test_subclass_matvec(self, dtype):
        n = 8
        A_np, A_dp = self._build_A(n, dtype)

        class MyOp(LinearOperator):
            def __init__(self):
                super().__init__(
                    shape=(n, n),
                    matvec=lambda x: A_dp @ x,
                    dtype=dpnp.float64,
                )

        op = MyOp()
        x = dpnp.asarray(_RNG.standard_normal(n).astype(dtype))
        result = op.matvec(x)
        expected = A_np @ dpnp.asnumpy(x)
        numpy.testing.assert_allclose(
            dpnp.asnumpy(result), expected,
            rtol=1e-5 if dtype == numpy.float32 else 1e-12)

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    def test_subclass_matmat(self, dtype):
        n, k = 7, 4
        A_np, A_dp = self._build_A(n, dtype)

        class MyOp(LinearOperator):
            def __init__(self):
                super().__init__(
                    shape=(n, n),
                    matvec=lambda x: A_dp @ x,
                    dtype=dpnp.float64,
                )
            def _matmat_impl(self, X):
                return A_dp @ X

        op = MyOp()
        X = dpnp.asarray(_RNG.standard_normal((n, k)).astype(dtype))
        Y = op.matmat(X)
        expected = A_np @ dpnp.asnumpy(X)
        numpy.testing.assert_allclose(
            dpnp.asnumpy(Y), expected,
            rtol=1e-5 if dtype == numpy.float32 else 1e-12)


# ---------------------------------------------------------------------------
# ─── aslinearoperator ────────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class TestAsLinearOperator:

    def test_identity_on_linearoperator(self):
        lo = LinearOperator((3, 3), matvec=lambda x: x)
        assert aslinearoperator(lo) is lo

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64,
                                        numpy.complex64, numpy.complex128])
    def test_dense_dpnp_array(self, dtype):
        n = 6
        A_np = _RNG.standard_normal((n, n)).astype(dtype)
        A_dp = dpnp.asarray(A_np)
        lo = aslinearoperator(A_dp)
        assert lo.shape == (n, n)
        x = dpnp.asarray(_RNG.standard_normal(n).astype(dtype))
        y = lo.matvec(x)
        expected = A_np @ dpnp.asnumpy(x)
        numpy.testing.assert_allclose(
            dpnp.asnumpy(y), expected,
            rtol=1e-5 if dtype in (numpy.float32, numpy.complex64) else 1e-12)

    def test_dense_numpy_array(self):
        n = 5
        A_np = _RNG.standard_normal((n, n))
        lo = aslinearoperator(A_np)
        assert lo.shape == (n, n)

    def test_rmatvec_from_dense(self):
        n = 5
        A_np = _RNG.standard_normal((n, n))
        A_dp = dpnp.asarray(A_np)
        lo = aslinearoperator(A_dp)
        x = dpnp.asarray(_RNG.standard_normal(n))
        y = lo.rmatvec(x)
        expected = A_np.conj().T @ dpnp.asnumpy(x)
        numpy.testing.assert_allclose(dpnp.asnumpy(y), expected, atol=1e-12)

    def test_duck_type_with_shape_and_matvec(self):
        n = 4

        class DuckOp:
            shape = (n, n)
            dtype = numpy.float64
            def matvec(self, x):
                return dpnp.asarray(dpnp.asnumpy(x) * 2.0)

        lo = aslinearoperator(DuckOp())
        x = dpnp.ones(n)
        y = lo.matvec(x)
        numpy.testing.assert_allclose(dpnp.asnumpy(y), numpy.ones(n) * 2.0)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            aslinearoperator("not_an_array")

    def test_invalid_1d_array_raises(self):
        with pytest.raises(Exception):
            aslinearoperator(dpnp.ones(5))


# ---------------------------------------------------------------------------
# ─── CG ──────────────────────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy required")
class TestCg:
    """Tests mirroring CuPy's TestCg class."""

    n = 30

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64,
                                        numpy.complex64, numpy.complex128])
    def test_converges_spd(self, dtype):
        A = _spd_matrix(self.n, dtype)
        b = _rhs(self.n, dtype)
        x, info = cg(A, b, tol=1e-8, maxiter=500)
        assert info == 0
        res = dpnp.linalg.norm(A @ x - b) / dpnp.linalg.norm(b)
        assert float(res) < 1e-5

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    def test_matches_scipy_reference(self, dtype):
        A_np = dpnp.asnumpy(_spd_matrix(self.n, dtype))
        b_np = dpnp.asnumpy(_rhs(self.n, dtype))
        x_ref, info_ref = scipy_sla.cg(A_np, b_np, rtol=1e-8, maxiter=500)
        assert info_ref == 0
        x_dp, info = cg(dpnp.asarray(A_np), dpnp.asarray(b_np),
                        tol=1e-8, maxiter=500)
        assert info == 0
        numpy.testing.assert_allclose(
            dpnp.asnumpy(x_dp), x_ref,
            rtol=1e-4 if dtype == numpy.float32 else 1e-8)

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    def test_x0_warm_start(self, dtype):
        A = _spd_matrix(self.n, dtype)
        b = _rhs(self.n, dtype)
        x0 = dpnp.ones(self.n, dtype=dtype)
        x, info = cg(A, b, x0=x0, tol=1e-8, maxiter=500)
        assert info == 0
        res = dpnp.linalg.norm(A @ x - b) / dpnp.linalg.norm(b)
        assert float(res) < 1e-5

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    def test_b_2dim(self, dtype):
        """b with shape (n, 1) should be accepted and flattened."""
        A = _spd_matrix(self.n, dtype)
        b = _rhs(self.n, dtype).reshape(self.n, 1)
        x, info = cg(A, b, tol=1e-8, maxiter=500)
        assert info == 0

    def test_callback_is_called(self):
        A = _spd_matrix(self.n, numpy.float64)
        b = _rhs(self.n, numpy.float64)
        calls = []
        def cb(xk):
            calls.append(float(dpnp.linalg.norm(xk)))
        cg(A, b, callback=cb, maxiter=200)
        assert len(calls) > 0

    @pytest.mark.parametrize("dtype", [numpy.float64])
    def test_atol(self, dtype):
        A = _spd_matrix(self.n, dtype)
        b = _rhs(self.n, dtype)
        x, info = cg(A, b, tol=0.0, atol=1e-1)
        res = float(dpnp.linalg.norm(A @ x - b))
        assert res < 1.0

    def test_exact_solution_zero_iter(self):
        """If x0 is already the solution, residual is zero and CG returns info=0."""
        n = 10
        A = _spd_matrix(n, numpy.float64)
        b = _rhs(n, numpy.float64)
        x_true = dpnp.asarray(
            numpy.linalg.solve(dpnp.asnumpy(A), dpnp.asnumpy(b)))
        x, info = cg(A, b, x0=x_true, tol=1e-12)
        assert info == 0

    @pytest.mark.parametrize("dtype", [numpy.float64])
    def test_via_linear_operator(self, dtype):
        A_np = dpnp.asnumpy(_spd_matrix(self.n, dtype))
        A_dp = dpnp.asarray(A_np)
        b = dpnp.asarray(_RNG.standard_normal(self.n))
        lo = aslinearoperator(A_dp)
        x, info = cg(lo, b, tol=1e-8, maxiter=500)
        assert info == 0
        res = float(dpnp.linalg.norm(
            dpnp.asarray(A_np) @ x - b)) / float(dpnp.linalg.norm(b))
        assert res < 1e-5

    def test_invalid_non_square(self):
        A = dpnp.ones((5, 6), dtype=dpnp.float64)
        b = dpnp.ones(5)
        with pytest.raises(Exception):
            cg(A, b)

    def test_invalid_b_wrong_size(self):
        A = _spd_matrix(5, numpy.float64)
        b = dpnp.ones(6)
        with pytest.raises((ValueError, Exception)):
            cg(A, b, maxiter=1)

    def test_maxiter_nonconvergence_info(self):
        """Setting maxiter=1 on a hard problem should return info > 0."""
        A = _spd_matrix(50, numpy.float64)
        b = _rhs(50, numpy.float64)
        x, info = cg(A, b, tol=1e-15, maxiter=1)
        assert info != 0


# ---------------------------------------------------------------------------
# ─── GMRES ───────────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy required")
class TestGmres:
    """Tests mirroring CuPy's TestGmres class."""

    n = 30

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64,
                                        numpy.complex64, numpy.complex128])
    def test_converges_diag_dominant(self, dtype):
        A = _diag_dominant(self.n, dtype)
        b = _rhs(self.n, dtype)
        x, info = gmres(A, b, tol=1e-8, maxiter=50, restart=30)
        assert info == 0
        res = dpnp.linalg.norm(A @ x - b) / dpnp.linalg.norm(b)
        assert float(res) < 1e-5

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    def test_matches_scipy_reference(self, dtype):
        A_np = dpnp.asnumpy(_diag_dominant(self.n, dtype))
        b_np = _RNG.standard_normal(self.n).astype(dtype)
        b_np /= numpy.linalg.norm(b_np)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_ref, info_ref = scipy_sla.gmres(
                A_np, b_np, rtol=1e-8, restart=self.n, maxiter=None)
        x_dp, info = gmres(
            dpnp.asarray(A_np), dpnp.asarray(b_np),
            tol=1e-8, restart=self.n, maxiter=50)
        assert info == 0
        numpy.testing.assert_allclose(
            dpnp.asnumpy(x_dp), x_ref,
            rtol=1e-3 if dtype == numpy.float32 else 1e-7)

    @pytest.mark.parametrize("restart", [None, 5, 15])
    def test_restart_values(self, restart):
        A = _diag_dominant(self.n, numpy.float64)
        b = _rhs(self.n, numpy.float64)
        x, info = gmres(A, b, tol=1e-8, restart=restart, maxiter=100)
        assert info == 0

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    def test_x0_warm_start(self, dtype):
        A = _diag_dominant(self.n, dtype)
        b = _rhs(self.n, dtype)
        x0 = dpnp.ones(self.n, dtype=dtype)
        x, info = gmres(A, b, x0=x0, tol=1e-8, maxiter=100)
        assert info == 0

    def test_b_2dim(self):
        """b with shape (n, 1) should be accepted."""
        A = _diag_dominant(self.n, numpy.float64)
        b = _rhs(self.n, numpy.float64).reshape(self.n, 1)
        x, info = gmres(A, b, tol=1e-8, maxiter=100)
        assert info == 0

    def test_callback_x_called(self):
        A = _diag_dominant(self.n, numpy.float64)
        b = _rhs(self.n, numpy.float64)
        calls = []
        def cb(xk):
            calls.append(1)
        gmres(A, b, callback=cb, callback_type='x', maxiter=20)
        assert len(calls) > 0

    def test_callback_pr_norm_not_implemented(self):
        A = _diag_dominant(self.n, numpy.float64)
        b = _rhs(self.n, numpy.float64)
        with pytest.raises(NotImplementedError):
            gmres(A, b, callback=lambda r: None, callback_type='pr_norm')

    def test_invalid_callback_type(self):
        A = _diag_dominant(self.n, numpy.float64)
        b = _rhs(self.n, numpy.float64)
        with pytest.raises(ValueError):
            gmres(A, b, callback_type='garbage')

    @pytest.mark.parametrize("dtype", [numpy.float64])
    def test_via_linear_operator(self, dtype):
        A_np = dpnp.asnumpy(_diag_dominant(self.n, dtype))
        A_dp = dpnp.asarray(A_np)
        b = dpnp.asarray(_RNG.standard_normal(self.n))
        lo = aslinearoperator(A_dp)
        x, info = gmres(lo, b, tol=1e-8, restart=self.n, maxiter=50)
        assert info == 0

    def test_nonconvergence_info_nonzero(self):
        """restart=2, maxiter=2 on a size-48 Hilbert-like matrix must not converge."""
        n = 48
        idx = numpy.arange(n, dtype=numpy.float64)
        A_np = 1.0 / (idx[:, None] + idx[None, :] + 1.0)
        b_np = _RNG.standard_normal(n)
        A_dp = dpnp.asarray(A_np)
        b_dp = dpnp.asarray(b_np)
        x, info = gmres(A_dp, b_dp, tol=1e-15, restart=2, maxiter=2)
        rel_res = float(dpnp.linalg.norm(A_dp @ x - b_dp) /
                        dpnp.linalg.norm(b_dp))
        assert rel_res > 1e-12
        assert info != 0

    def test_complex_system(self):
        n = 15
        A_np = (_RNG.standard_normal((n, n)) +
                1j * _RNG.standard_normal((n, n))).astype(numpy.complex128)
        numpy.fill_diagonal(A_np, numpy.abs(A_np).sum(axis=1) + 1.0)
        b_np = (_RNG.standard_normal(n) +
                1j * _RNG.standard_normal(n)).astype(numpy.complex128)
        A_dp = dpnp.asarray(A_np)
        b_dp = dpnp.asarray(b_np)
        x, info = gmres(A_dp, b_dp, tol=1e-8, restart=n, maxiter=50)
        assert info == 0
        res = float(numpy.linalg.norm(A_np @ dpnp.asnumpy(x) - b_np) /
                    numpy.linalg.norm(b_np))
        assert res < 1e-5

    def test_atol_parameter(self):
        A = _diag_dominant(self.n, numpy.float64)
        b = _rhs(self.n, numpy.float64)
        x, info = gmres(A, b, tol=0.0, atol=1e-6, restart=self.n, maxiter=50)
        res = float(dpnp.linalg.norm(A @ x - b))
        assert res < 1e-4


# ---------------------------------------------------------------------------
# ─── MINRES ────────────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy required for MINRES")
class TestMinres:
    """Tests for MINRES (SciPy-backed implementation)."""

    n = 30

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    def test_converges_spd(self, dtype):
        """MINRES on SPD system should converge."""
        A = _spd_matrix(self.n, dtype)
        b = _rhs(self.n, dtype)
        x, info = minres(A, b, tol=1e-8, maxiter=500)
        assert info == 0
        res = float(dpnp.linalg.norm(A @ x - b) / dpnp.linalg.norm(b))
        assert res < 1e-4

    @pytest.mark.parametrize("dtype", [numpy.float64])
    def test_converges_sym_indefinite(self, dtype):
        """MINRES distinguishes itself on symmetric-indefinite systems."""
        A = _sym_indefinite(self.n, dtype)
        b = _rhs(self.n, dtype)
        x, info = minres(A, b, tol=1e-8, maxiter=1000)
        res = float(dpnp.linalg.norm(A @ x - b) / dpnp.linalg.norm(b))
        assert res < 1e-3

    @pytest.mark.parametrize("dtype", [numpy.float64])
    def test_matches_scipy_reference(self, dtype):
        A_np = dpnp.asnumpy(_spd_matrix(self.n, dtype))
        b_np = dpnp.asnumpy(_rhs(self.n, dtype))
        x_ref, _ = scipy_sla.minres(A_np, b_np, rtol=1e-8)
        x_dp, info = minres(
            dpnp.asarray(A_np), dpnp.asarray(b_np), tol=1e-8)
        numpy.testing.assert_allclose(
            dpnp.asnumpy(x_dp), x_ref, rtol=1e-6)

    def test_x0_warm_start(self):
        A = _spd_matrix(self.n, numpy.float64)
        b = _rhs(self.n, numpy.float64)
        x0 = dpnp.zeros(self.n, dtype=numpy.float64)
        x, info = minres(A, b, x0=x0, tol=1e-8)
        assert info == 0

    def test_shift_parameter(self):
        """shift != 0: solves (A - shift*I) x = b."""
        A_np = dpnp.asnumpy(_spd_matrix(self.n, numpy.float64))
        b_np = dpnp.asnumpy(_rhs(self.n, numpy.float64))
        shift = 0.5
        A_dp = dpnp.asarray(A_np)
        b_dp = dpnp.asarray(b_np)
        x, info = minres(A_dp, b_dp, shift=shift, tol=1e-8)
        A_shifted = A_np - shift * numpy.eye(self.n)
        res = numpy.linalg.norm(A_shifted @ dpnp.asnumpy(x) - b_np)
        assert res / numpy.linalg.norm(b_np) < 1e-4

    def test_non_square_raises(self):
        A = aslinearoperator(dpnp.ones((4, 5), dtype=dpnp.float64))
        b = dpnp.ones(4)
        with pytest.raises(ValueError):
            minres(A, b)

    def test_via_linear_operator(self):
        A_np = dpnp.asnumpy(_spd_matrix(self.n, numpy.float64))
        A_dp = dpnp.asarray(A_np)
        b = dpnp.asarray(_RNG.standard_normal(self.n))
        lo = aslinearoperator(A_dp)
        x, info = minres(lo, b, tol=1e-8)
        assert info == 0

    @pytest.mark.parametrize("dtype", [numpy.float64])
    def test_callback_is_called(self, dtype):
        A = _spd_matrix(self.n, dtype)
        b = _rhs(self.n, dtype)
        calls = []
        def cb(xk):
            calls.append(1)
        minres(A, b, callback=cb, tol=1e-8)
        assert len(calls) > 0


# ---------------------------------------------------------------------------
# ─── Integration: all solvers via LinearOperator ─────────────────────────────────────────
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy required")
class TestSolversViaLinearOperator:
    """Parametric integration tests with varying n and dtype."""

    @pytest.mark.parametrize("n,dtype", [
        (10, numpy.float32), (10, numpy.float64),
        (30, numpy.float64), (50, numpy.float64),
    ])
    def test_cg_spd_lo(self, n, dtype):
        A_dp = _spd_matrix(n, dtype)
        lo = aslinearoperator(A_dp)
        b = _rhs(n, dtype)
        x, info = cg(lo, b, tol=1e-8, maxiter=n * 10)
        assert info == 0
        res = float(dpnp.linalg.norm(A_dp @ x - b) / dpnp.linalg.norm(b))
        atol = 1e-4 if dtype == numpy.float32 else 1e-8
        assert res < atol

    @pytest.mark.parametrize("n,dtype", [
        (10, numpy.float32), (10, numpy.float64),
        (30, numpy.float64),
    ])
    def test_gmres_nonsymmetric_lo(self, n, dtype):
        A_dp = _diag_dominant(n, dtype)
        lo = aslinearoperator(A_dp)
        b = _rhs(n, dtype)
        x, info = gmres(lo, b, tol=1e-8, restart=n, maxiter=50)
        assert info == 0


# ---------------------------------------------------------------------------
# ─── Import smoke tests ───────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class TestImports:
    def test_all_symbols_importable(self):
        from dpnp.scipy.sparse.linalg import (
            LinearOperator, aslinearoperator, cg, gmres, minres)
        assert callable(LinearOperator)
        assert callable(aslinearoperator)
        assert callable(cg)
        assert callable(gmres)
        assert callable(minres)

    def test_all_listed_in_dunder_all(self):
        import dpnp.scipy.sparse.linalg as mod
        for name in ("LinearOperator", "aslinearoperator", "cg", "gmres", "minres"):
            assert name in mod.__all__, f"{name!r} missing from __all__"
