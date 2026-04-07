# Copyright (c) 2025, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Intel Corporation nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

"""Tests for dpnp.scipy.sparse.linalg: LinearOperator, cg, gmres, minres.

The test structure and helper usage mirror dpnp/tests/test_linalg.py so that
the suite fits naturally into the existing CI infrastructure.

Note: dpnp.ndarray deliberately blocks implicit numpy conversion (raises
TypeError in __array__) to prevent silent dtype=object arrays.  All
assertions that need a host-side NumPy array must call `arr.asnumpy()`
explicitly instead of `numpy.asarray(arr)`.
"""

import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal, assert_raises

import dpnp
from dpnp.scipy.sparse.linalg import (
    LinearOperator,
    aslinearoperator,
    cg,
    gmres,
    minres,
)

from .helper import (
    assert_dtype_allclose,
    generate_random_numpy_array,
    get_float_complex_dtypes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(x):
    """Convert a dpnp array (or plain numpy array) to numpy safely."""
    if isinstance(x, dpnp.ndarray):
        return x.asnumpy()
    return numpy.asarray(x)


def _make_spd(n, dtype, rng):
    """Return a symmetric positive-definite matrix of size n."""
    A = rng.standard_normal((n, n)).astype(dtype)
    return A.T @ A + n * numpy.eye(n, dtype=dtype)


def _make_sym_indef(n, dtype, rng):
    """Return a symmetric (possibly indefinite) matrix of size n."""
    Q, _ = numpy.linalg.qr(rng.standard_normal((n, n)).astype(dtype))
    D = numpy.diag(rng.standard_normal(n).astype(dtype))
    return Q @ D @ Q.T


def _make_nonsym(n, dtype, rng):
    """Return a diagonally dominant (non-symmetric) matrix of size n."""
    A = rng.standard_normal((n, n)).astype(dtype)
    A += n * numpy.eye(n, dtype=dtype)
    return A


def _rel_residual(A_np, x_dp, b_np):
    """Relative residual ||Ax - b|| / ||b||."""
    x_np = _to_numpy(x_dp)
    r = A_np @ x_np - b_np
    b_nrm = numpy.linalg.norm(b_np)
    return numpy.linalg.norm(r) / (b_nrm if b_nrm > 0 else 1.0)


# ---------------------------------------------------------------------------
# TestLinearOperator
# ---------------------------------------------------------------------------

class TestLinearOperator:
    """Tests for the LinearOperator class and aslinearoperator helper."""

    # --- basic construction ---

    def test_basic_construction_shape_dtype(self):
        n = 8
        A_np = numpy.eye(n, dtype=numpy.float64)
        A_dp = dpnp.asarray(A_np)

        op = LinearOperator((n, n), matvec=lambda x: A_dp @ x)
        assert op.shape == (n, n)
        assert op.ndim == 2

    def test_dtype_inferred_from_matvec(self):
        n = 6
        A_dp = dpnp.eye(n, dtype=numpy.float32)
        op = LinearOperator((n, n), matvec=lambda x: A_dp @ x)
        assert op.dtype == numpy.float32

    def test_dtype_explicit_override(self):
        n = 4
        A_dp = dpnp.eye(n)
        op = LinearOperator((n, n), matvec=lambda x: A_dp @ x, dtype=numpy.float32)
        assert op.dtype == numpy.float32

    @pytest.mark.parametrize("n", [1, 5, 20])
    def test_matvec_identity(self, n):
        A_dp = dpnp.eye(n, dtype=numpy.float64)
        op = LinearOperator((n, n), matvec=lambda x: A_dp @ x)
        x_dp = dpnp.arange(n, dtype=numpy.float64)
        y_dp = op.matvec(x_dp)
        assert_allclose(_to_numpy(y_dp), _to_numpy(x_dp), rtol=1e-12)

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    def test_matvec_dense(self, dtype):
        rng = numpy.random.default_rng(0)
        n = 10
        A_np = _make_spd(n, dtype, rng)
        A_dp = dpnp.asarray(A_np)
        x_np = rng.standard_normal(n).astype(dtype)
        x_dp = dpnp.asarray(x_np)

        op = LinearOperator((n, n), matvec=lambda x: A_dp @ x, dtype=dtype)
        y_dp = op.matvec(x_dp)
        y_ref = A_np @ x_np
        assert_allclose(_to_numpy(y_dp), y_ref, rtol=1e-5)

    # --- rmatvec ---

    def test_rmatvec_defined(self):
        rng = numpy.random.default_rng(1)
        n = 8
        A_np = rng.standard_normal((n, n)).astype(numpy.float64)
        A_dp = dpnp.asarray(A_np)
        x_np = rng.standard_normal(n)
        x_dp = dpnp.asarray(x_np)

        op = LinearOperator(
            (n, n),
            matvec=lambda x: A_dp @ x,
            rmatvec=lambda x: A_dp.T @ x,
        )
        y_dp = op.rmatvec(x_dp)
        y_ref = A_np.T @ x_np
        assert_allclose(_to_numpy(y_dp), y_ref, rtol=1e-12)

    def test_rmatvec_not_defined_raises(self):
        n = 4
        A_dp = dpnp.eye(n)
        op = LinearOperator((n, n), matvec=lambda x: A_dp @ x)
        x_dp = dpnp.ones(n)
        with pytest.raises(NotImplementedError):
            op.rmatvec(x_dp)

    # --- matmat ---

    def test_matmat_fallback_loop(self):
        rng = numpy.random.default_rng(2)
        n, k = 6, 4
        A_np = rng.standard_normal((n, n)).astype(numpy.float64)
        A_dp = dpnp.asarray(A_np)
        X_np = rng.standard_normal((n, k)).astype(numpy.float64)
        X_dp = dpnp.asarray(X_np)

        op = LinearOperator((n, n), matvec=lambda x: A_dp @ x)
        Y_dp = op.matmat(X_dp)
        Y_ref = A_np @ X_np
        assert_allclose(_to_numpy(Y_dp), Y_ref, rtol=1e-10)

    def test_matmat_explicit(self):
        rng = numpy.random.default_rng(3)
        n, k = 5, 3
        A_np = rng.standard_normal((n, n)).astype(numpy.float64)
        A_dp = dpnp.asarray(A_np)
        X_np = rng.standard_normal((n, k)).astype(numpy.float64)
        X_dp = dpnp.asarray(X_np)

        op = LinearOperator(
            (n, n),
            matvec=lambda x: A_dp @ x,
            matmat=lambda X: A_dp @ X,
        )
        Y_dp = op.matmat(X_dp)
        assert_allclose(_to_numpy(Y_dp), A_np @ X_np, rtol=1e-10)

    # --- __matmul__ / __call__ ---

    def test_matmul_1d(self):
        n = 5
        A_dp = dpnp.eye(n, dtype=numpy.float64) * 2.0
        op = LinearOperator((n, n), matvec=lambda x: A_dp @ x)
        x_dp = dpnp.ones(n)
        y_dp = op @ x_dp
        assert_allclose(_to_numpy(y_dp), numpy.full(n, 2.0))

    def test_matmul_2d(self):
        n, k = 4, 3
        A_dp = dpnp.eye(n, dtype=numpy.float64)
        X_dp = dpnp.ones((n, k))
        op = LinearOperator((n, n), matvec=lambda x: A_dp @ x)
        Y_dp = op @ X_dp
        assert_allclose(_to_numpy(Y_dp), numpy.ones((n, k)))

    def test_call_delegates_to_matmul(self):
        n = 4
        A_dp = dpnp.eye(n, dtype=numpy.float64)
        op = LinearOperator((n, n), matvec=lambda x: A_dp @ x)
        x_dp = dpnp.ones(n)
        assert_allclose(_to_numpy(op(x_dp)), _to_numpy(op @ x_dp))

    # --- operator algebra ---

    def test_adjoint_property_H(self):
        rng = numpy.random.default_rng(4)
        n = 6
        A_np = rng.standard_normal((n, n)).astype(numpy.float64)
        A_dp = dpnp.asarray(A_np)
        op = LinearOperator(
            (n, n),
            matvec=lambda x: A_dp @ x,
            rmatvec=lambda x: A_dp.T @ x,
        )
        x_dp = dpnp.asarray(rng.standard_normal(n))
        y_H = op.H.matvec(x_dp)
        y_ref = A_np.T @ _to_numpy(x_dp)
        assert_allclose(_to_numpy(y_H), y_ref, rtol=1e-12)

    def test_transpose_property_T(self):
        rng = numpy.random.default_rng(5)
        n = 6
        A_np = rng.standard_normal((n, n)).astype(numpy.float64)
        A_dp = dpnp.asarray(A_np)
        op = LinearOperator(
            (n, n),
            matvec=lambda x: A_dp @ x,
            rmatvec=lambda x: A_dp.T @ x,
        )
        x_dp = dpnp.asarray(rng.standard_normal(n))
        y_T = op.T.matvec(x_dp)
        # For real A, T == H
        y_ref = A_np.T @ _to_numpy(x_dp)
        assert_allclose(_to_numpy(y_T), y_ref, rtol=1e-12)

    def test_add_two_operators(self):
        n = 5
        A_dp = dpnp.eye(n, dtype=numpy.float64)
        B_dp = dpnp.eye(n, dtype=numpy.float64) * 2.0
        opA = LinearOperator((n, n), matvec=lambda x: A_dp @ x)
        opB = LinearOperator((n, n), matvec=lambda x: B_dp @ x)
        opC = opA + opB
        x_dp = dpnp.ones(n)
        y_dp = opC.matvec(x_dp)
        assert_allclose(_to_numpy(y_dp), numpy.full(n, 3.0))

    def test_scalar_multiply(self):
        n = 4
        A_dp = dpnp.eye(n, dtype=numpy.float64)
        op = LinearOperator((n, n), matvec=lambda x: A_dp @ x)
        op3 = op * 3.0
        x_dp = dpnp.ones(n)
        y_dp = op3.matvec(x_dp)
        assert_allclose(_to_numpy(y_dp), numpy.full(n, 3.0))

    def test_product_operator(self):
        n = 5
        A_dp = dpnp.eye(n, dtype=numpy.float64) * 2.0
        B_dp = dpnp.eye(n, dtype=numpy.float64) * 3.0
        opA = LinearOperator((n, n), matvec=lambda x: A_dp @ x)
        opB = LinearOperator((n, n), matvec=lambda x: B_dp @ x)
        opAB = opA * opB
        x_dp = dpnp.ones(n)
        y_dp = opAB.matvec(x_dp)
        assert_allclose(_to_numpy(y_dp), numpy.full(n, 6.0))

    def test_neg_operator(self):
        n = 4
        A_dp = dpnp.eye(n, dtype=numpy.float64)
        op = LinearOperator((n, n), matvec=lambda x: A_dp @ x)
        neg_op = -op
        x_dp = dpnp.ones(n)
        y_dp = neg_op.matvec(x_dp)
        assert_allclose(_to_numpy(y_dp), numpy.full(n, -1.0))

    def test_power_operator(self):
        n = 4
        A_dp = dpnp.eye(n, dtype=numpy.float64) * 2.0
        op = LinearOperator((n, n), matvec=lambda x: A_dp @ x)
        op3 = op ** 3
        x_dp = dpnp.ones(n)
        y_dp = op3.matvec(x_dp)
        # 2^3 * I * [1...] = 8
        assert_allclose(_to_numpy(y_dp), numpy.full(n, 8.0))

    # --- shape / error validation ---

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            LinearOperator((5,), matvec=lambda x: x)

    def test_matvec_wrong_input_dim_raises(self):
        n = 4
        A_dp = dpnp.eye(n, dtype=numpy.float64)
        op = LinearOperator((n, n), matvec=lambda x: A_dp @ x)
        with pytest.raises(ValueError):
            op.matvec(dpnp.ones(n + 1))

    # --- aslinearoperator ---

    def test_aslinearoperator_identity_if_already_lo(self):
        n = 4
        A_dp = dpnp.eye(n)
        op = LinearOperator((n, n), matvec=lambda x: A_dp @ x)
        assert aslinearoperator(op) is op

    def test_aslinearoperator_from_dense_dpnp(self):
        n = 6
        A_dp = dpnp.eye(n, dtype=numpy.float64)
        op = aslinearoperator(A_dp)
        x_dp = dpnp.ones(n)
        y_dp = op.matvec(x_dp)
        assert_allclose(_to_numpy(y_dp), numpy.ones(n))

    def test_aslinearoperator_from_numpy(self):
        n = 5
        A_np = numpy.eye(n, dtype=numpy.float64)
        op = aslinearoperator(A_np)
        x_dp = dpnp.ones(n)
        y_dp = op.matvec(x_dp)
        assert_allclose(_to_numpy(y_dp), numpy.ones(n))

    def test_aslinearoperator_invalid_raises(self):
        with pytest.raises(TypeError):
            aslinearoperator("not_an_array")

    def test_repr_string(self):
        n = 3
        op = LinearOperator((n, n), matvec=lambda x: x, dtype=numpy.float64)
        r = repr(op)
        assert "3x3" in r

    # --- IdentityOperator ---

    def test_identity_operator(self):
        from dpnp.scipy.sparse.linalg._interface import IdentityOperator

        n = 7
        op = IdentityOperator((n, n), dtype=numpy.float64)
        x_dp = dpnp.arange(n, dtype=numpy.float64)
        assert_array_equal(_to_numpy(op.matvec(x_dp)), numpy.arange(n))
        assert_array_equal(_to_numpy(op.rmatvec(x_dp)), numpy.arange(n))

    # --- complex dtype ---

    @pytest.mark.parametrize("dtype", [numpy.complex64, numpy.complex128])
    def test_complex_matvec(self, dtype):
        n = 6
        rng = numpy.random.default_rng(10)
        A_np = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))).astype(dtype)
        A_dp = dpnp.asarray(A_np)
        x_np = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(dtype)
        x_dp = dpnp.asarray(x_np)

        op = LinearOperator((n, n), matvec=lambda x: A_dp @ x, dtype=dtype)
        y_dp = op.matvec(x_dp)
        assert_allclose(_to_numpy(y_dp), A_np @ x_np, rtol=1e-4)


# ---------------------------------------------------------------------------
# TestCG
# ---------------------------------------------------------------------------

class TestCG:
    """Tests for dpnp.scipy.sparse.linalg.cg."""

    @pytest.mark.parametrize("n", [5, 10, 30])
    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    def test_cg_spd_convergence(self, n, dtype):
        rng = numpy.random.default_rng(100)
        A_np = _make_spd(n, dtype, rng)
        b_np = rng.standard_normal(n).astype(dtype)
        A_dp = dpnp.asarray(A_np)
        b_dp = dpnp.asarray(b_np)

        x_dp, info = cg(A_dp, b_dp, tol=1e-7, maxiter=500)
        assert info == 0, f"CG did not converge (info={info})"
        assert _rel_residual(A_np, x_dp, b_np) < 1e-5

    def test_cg_matches_numpy_solve(self):
        rng = numpy.random.default_rng(101)
        n = 15
        dtype = numpy.float64
        A_np = _make_spd(n, dtype, rng)
        b_np = rng.standard_normal(n).astype(dtype)
        A_dp = dpnp.asarray(A_np)
        b_dp = dpnp.asarray(b_np)

        x_ref = numpy.linalg.solve(A_np, b_np)
        x_dp, info = cg(A_dp, b_dp, tol=1e-10, maxiter=1000)
        assert info == 0
        assert_allclose(_to_numpy(x_dp), x_ref, rtol=1e-6)

    def test_cg_x0_initial_guess(self):
        rng = numpy.random.default_rng(102)
        n = 12
        dtype = numpy.float64
        A_np = _make_spd(n, dtype, rng)
        b_np = rng.standard_normal(n).astype(dtype)
        A_dp = dpnp.asarray(A_np)
        b_dp = dpnp.asarray(b_np)

        x_ref = numpy.linalg.solve(A_np, b_np)
        x0_dp = dpnp.asarray(x_ref)
        x_dp, info = cg(A_dp, b_dp, x0=x0_dp, tol=1e-10, maxiter=5)
        assert _rel_residual(A_np, x_dp, b_np) < 1e-8

    def test_cg_callback_called(self):
        rng = numpy.random.default_rng(103)
        n = 8
        dtype = numpy.float64
        A_np = _make_spd(n, dtype, rng)
        b_np = rng.standard_normal(n)
        A_dp = dpnp.asarray(A_np)
        b_dp = dpnp.asarray(b_np)

        calls = []
        def cb(xk):
            calls.append(1)

        x_dp, info = cg(A_dp, b_dp, tol=1e-8, maxiter=200, callback=cb)
        assert info == 0
        assert len(calls) > 0

    def test_cg_already_zero_rhs(self):
        n = 5
        A_dp = dpnp.eye(n, dtype=numpy.float64)
        b_dp = dpnp.zeros(n, dtype=numpy.float64)
        x_dp, info = cg(A_dp, b_dp)
        assert info == 0
        assert_allclose(_to_numpy(x_dp), numpy.zeros(n), atol=1e-14)

    def test_cg_returns_dpnp_array(self):
        n = 4
        A_dp = dpnp.eye(n, dtype=numpy.float64)
        b_dp = dpnp.ones(n, dtype=numpy.float64)
        x_dp, _ = cg(A_dp, b_dp)
        assert isinstance(x_dp, dpnp.ndarray)

    def test_cg_with_atol(self):
        rng = numpy.random.default_rng(104)
        n = 10
        dtype = numpy.float64
        A_np = _make_spd(n, dtype, rng)
        b_np = rng.standard_normal(n).astype(dtype)
        A_dp = dpnp.asarray(A_np)
        b_dp = dpnp.asarray(b_np)

        x_dp, info = cg(A_dp, b_dp, tol=0.0, atol=1e-8, maxiter=500)
        assert info == 0

    def test_cg_with_linear_operator(self):
        rng = numpy.random.default_rng(105)
        n = 10
        dtype = numpy.float64
        A_np = _make_spd(n, dtype, rng)
        A_dp = dpnp.asarray(A_np)
        b_np = rng.standard_normal(n).astype(dtype)
        b_dp = dpnp.asarray(b_np)

        op = LinearOperator((n, n), matvec=lambda x: A_dp @ x, dtype=dtype)
        x_dp, info = cg(op, b_dp, tol=1e-8, maxiter=500)
        assert info == 0
        assert _rel_residual(A_np, x_dp, b_np) < 1e-6

    def test_cg_maxiter_exhausted_returns_nonzero_info(self):
        rng = numpy.random.default_rng(106)
        n = 20
        dtype = numpy.float64
        A_np = _make_spd(n, dtype, rng)
        b_np = rng.standard_normal(n).astype(dtype)
        A_dp = dpnp.asarray(A_np)
        b_dp = dpnp.asarray(b_np)

        _, info = cg(A_dp, b_dp, tol=1e-20, maxiter=1)
        assert info != 0

    def test_cg_preconditioner_unsupported_raises(self):
        n = 4
        A_dp = dpnp.eye(n, dtype=numpy.float64)
        b_dp = dpnp.ones(n)
        M = dpnp.eye(n)
        with pytest.raises(NotImplementedError):
            cg(A_dp, b_dp, M=M)

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    def test_cg_dtype_preserved_in_output(self, dtype):
        n = 8
        rng = numpy.random.default_rng(107)
        A_np = _make_spd(n, dtype, rng)
        b_np = rng.standard_normal(n).astype(dtype)
        x_dp, _ = cg(dpnp.asarray(A_np), dpnp.asarray(b_np), tol=1e-6, maxiter=500)
        assert numpy.issubdtype(x_dp.dtype, numpy.floating)


# ---------------------------------------------------------------------------
# TestGMRES
# ---------------------------------------------------------------------------

class TestGMRES:
    """Tests for dpnp.scipy.sparse.linalg.gmres."""

    @pytest.mark.parametrize("n", [5, 10, 25])
    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    def test_gmres_nonsym_convergence(self, n, dtype):
        rng = numpy.random.default_rng(200)
        A_np = _make_nonsym(n, dtype, rng)
        b_np = rng.standard_normal(n).astype(dtype)
        A_dp = dpnp.asarray(A_np)
        b_dp = dpnp.asarray(b_np)

        x_dp, info = gmres(A_dp, b_dp, tol=1e-7, maxiter=50, restart=n)
        assert info == 0, f"GMRES did not converge (info={info})"
        assert _rel_residual(A_np, x_dp, b_np) < 1e-5

    def test_gmres_matches_numpy_solve(self):
        rng = numpy.random.default_rng(201)
        n = 12
        dtype = numpy.float64
        A_np = _make_nonsym(n, dtype, rng)
        b_np = rng.standard_normal(n).astype(dtype)
        A_dp = dpnp.asarray(A_np)
        b_dp = dpnp.asarray(b_np)

        x_ref = numpy.linalg.solve(A_np, b_np)
        x_dp, info = gmres(A_dp, b_dp, tol=1e-10, maxiter=50, restart=n)
        assert info == 0
        assert_allclose(_to_numpy(x_dp), x_ref, rtol=1e-5)

    def test_gmres_spd_matches_cg(self):
        """On an SPD system GMRES and CG should agree."""
        rng = numpy.random.default_rng(202)
        n = 15
        dtype = numpy.float64
        A_np = _make_spd(n, dtype, rng)
        b_np = rng.standard_normal(n).astype(dtype)
        A_dp = dpnp.asarray(A_np)
        b_dp = dpnp.asarray(b_np)

        x_gmres, _ = gmres(A_dp, b_dp, tol=1e-10, maxiter=100, restart=n)
        x_cg, _ = cg(A_dp, b_dp, tol=1e-10, maxiter=500)
        assert_allclose(_to_numpy(x_gmres), _to_numpy(x_cg), rtol=1e-5)

    def test_gmres_restart_parameter(self):
        """Restarted GMRES (restart < n) should still converge."""
        rng = numpy.random.default_rng(203)
        n = 20
        dtype = numpy.float64
        A_np = _make_nonsym(n, dtype, rng)
        b_np = rng.standard_normal(n).astype(dtype)
        A_dp = dpnp.asarray(A_np)
        b_dp = dpnp.asarray(b_np)

        x_dp, info = gmres(A_dp, b_dp, tol=1e-7, maxiter=20, restart=5)
        assert info == 0
        assert _rel_residual(A_np, x_dp, b_np) < 1e-5

    def test_gmres_x0_initial_guess(self):
        rng = numpy.random.default_rng(204)
        n = 10
        dtype = numpy.float64
        A_np = _make_nonsym(n, dtype, rng)
        b_np = rng.standard_normal(n).astype(dtype)
        A_dp = dpnp.asarray(A_np)
        b_dp = dpnp.asarray(b_np)

        x_ref = numpy.linalg.solve(A_np, b_np)
        x0_dp = dpnp.asarray(x_ref)
        x_dp, info = gmres(A_dp, b_dp, x0=x0_dp, tol=1e-10, maxiter=5, restart=n)
        assert _rel_residual(A_np, x_dp, b_np) < 1e-8

    def test_gmres_callback_called(self):
        rng = numpy.random.default_rng(205)
        n = 8
        A_np = _make_nonsym(n, numpy.float64, rng)
        b_np = rng.standard_normal(n)
        A_dp = dpnp.asarray(A_np)
        b_dp = dpnp.asarray(b_np)

        calls = []
        def cb(xk):
            calls.append(1)

        _, info = gmres(A_dp, b_dp, tol=1e-8, maxiter=20, callback=cb, restart=n)
        assert info == 0
        assert len(calls) > 0

    def test_gmres_already_zero_rhs(self):
        n = 5
        A_dp = dpnp.eye(n, dtype=numpy.float64)
        b_dp = dpnp.zeros(n, dtype=numpy.float64)
        x_dp, info = gmres(A_dp, b_dp)
        assert info == 0
        assert_allclose(_to_numpy(x_dp), numpy.zeros(n), atol=1e-14)

    def test_gmres_returns_dpnp_array(self):
        n = 4
        A_dp = dpnp.eye(n, dtype=numpy.float64)
        b_dp = dpnp.ones(n, dtype=numpy.float64)
        x_dp, _ = gmres(A_dp, b_dp)
        assert isinstance(x_dp, dpnp.ndarray)

    def test_gmres_with_atol(self):
        rng = numpy.random.default_rng(206)
        n = 10
        dtype = numpy.float64
        A_np = _make_nonsym(n, dtype, rng)
        b_np = rng.standard_normal(n).astype(dtype)
        x_dp, info = gmres(
            dpnp.asarray(A_np),
            dpnp.asarray(b_np),
            tol=0.0,
            atol=1e-7,
            maxiter=50,
            restart=n,
        )
        assert info == 0

    def test_gmres_with_linear_operator(self):
        rng = numpy.random.default_rng(207)
        n = 10
        dtype = numpy.float64
        A_np = _make_nonsym(n, dtype, rng)
        A_dp = dpnp.asarray(A_np)
        b_np = rng.standard_normal(n).astype(dtype)
        b_dp = dpnp.asarray(b_np)

        op = LinearOperator((n, n), matvec=lambda x: A_dp @ x, dtype=dtype)
        x_dp, info = gmres(op, b_dp, tol=1e-8, maxiter=50, restart=n)
        assert info == 0
        assert _rel_residual(A_np, x_dp, b_np) < 1e-6

    def test_gmres_maxiter_exhausted_returns_nonzero_info(self):
        rng = numpy.random.default_rng(208)
        n = 20
        dtype = numpy.float64
        A_np = _make_nonsym(n, dtype, rng)
        b_np = rng.standard_normal(n).astype(dtype)
        A_dp = dpnp.asarray(A_np)
        b_dp = dpnp.asarray(b_np)

        _, info = gmres(A_dp, b_dp, tol=1e-20, maxiter=1, restart=2)
        assert info != 0

    def test_gmres_preconditioner_unsupported_raises(self):
        n = 4
        A_dp = dpnp.eye(n, dtype=numpy.float64)
        b_dp = dpnp.ones(n)
        M = dpnp.eye(n)
        with pytest.raises(NotImplementedError):
            gmres(A_dp, b_dp, M=M)

    def test_gmres_callback_type_pr_norm_raises(self):
        n = 4
        A_dp = dpnp.eye(n, dtype=numpy.float64)
        b_dp = dpnp.ones(n)
        with pytest.raises(NotImplementedError):
            gmres(A_dp, b_dp, callback=lambda x: None, callback_type="pr_norm")

    def test_gmres_invalid_callback_type_raises(self):
        n = 4
        A_dp = dpnp.eye(n, dtype=numpy.float64)
        b_dp = dpnp.ones(n)
        with pytest.raises(ValueError):
            gmres(A_dp, b_dp, callback_type="bad_value")

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    def test_gmres_dtype_preserved_in_output(self, dtype):
        n = 6
        rng = numpy.random.default_rng(209)
        A_np = _make_nonsym(n, dtype, rng)
        b_np = rng.standard_normal(n).astype(dtype)
        x_dp, _ = gmres(
            dpnp.asarray(A_np),
            dpnp.asarray(b_np),
            tol=1e-6,
            maxiter=50,
            restart=n,
        )
        assert numpy.issubdtype(x_dp.dtype, numpy.floating)

    @pytest.mark.parametrize("n", [5, 15])
    def test_gmres_happy_breakdown(self, n):
        """Identity operator should yield happy breakdown (exact solution)."""
        A_dp = dpnp.eye(n, dtype=numpy.float64)
        b_dp = dpnp.arange(1, n + 1, dtype=numpy.float64)
        x_dp, info = gmres(A_dp, b_dp, tol=1e-12, maxiter=n, restart=n)
        assert info == 0
        assert_allclose(_to_numpy(x_dp), numpy.arange(1, n + 1), rtol=1e-10)


# ---------------------------------------------------------------------------
# TestMINRES
# ---------------------------------------------------------------------------

class TestMINRES:
    """Tests for dpnp.scipy.sparse.linalg.minres (SciPy-backed stub)."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_scipy(self):
        pytest.importorskip("scipy", reason="SciPy required for minres tests")

    @pytest.mark.parametrize("n", [5, 10, 20])
    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    def test_minres_spd_convergence(self, n, dtype):
        rng = numpy.random.default_rng(300)
        A_np = _make_spd(n, dtype, rng)
        b_np = rng.standard_normal(n).astype(dtype)
        A_dp = dpnp.asarray(A_np)
        b_dp = dpnp.asarray(b_np)

        x_dp, info = minres(A_dp, b_dp, tol=1e-7, maxiter=500)
        assert info == 0, f"MINRES did not converge (info={info})"
        assert _rel_residual(A_np, x_dp, b_np) < 1e-5

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    def test_minres_sym_indef_convergence(self, dtype):
        rng = numpy.random.default_rng(301)
        n = 12
        A_np = _make_sym_indef(n, dtype, rng)
        b_np = rng.standard_normal(n).astype(dtype)
        A_dp = dpnp.asarray(A_np)
        b_dp = dpnp.asarray(b_np)

        x_dp, info = minres(A_dp, b_dp, tol=1e-6, maxiter=500)
        assert info == 0
        assert _rel_residual(A_np, x_dp, b_np) < 1e-4

    def test_minres_matches_scipy(self):
        import scipy.sparse.linalg as sla

        rng = numpy.random.default_rng(302)
        n = 10
        dtype = numpy.float64
        A_np = _make_spd(n, dtype, rng)
        b_np = rng.standard_normal(n).astype(dtype)

        x_scipy, info_scipy = sla.minres(A_np, b_np, rtol=1e-10)
        x_dp, info_dp = minres(
            dpnp.asarray(A_np), dpnp.asarray(b_np), tol=1e-10
        )
        assert info_dp == 0
        assert_allclose(_to_numpy(x_dp), x_scipy, rtol=1e-6)

    def test_minres_x0_initial_guess(self):
        rng = numpy.random.default_rng(303)
        n = 8
        dtype = numpy.float64
        A_np = _make_spd(n, dtype, rng)
        b_np = rng.standard_normal(n).astype(dtype)
        A_dp = dpnp.asarray(A_np)
        b_dp = dpnp.asarray(b_np)

        x_ref = numpy.linalg.solve(A_np, b_np)
        x0_dp = dpnp.asarray(x_ref)
        x_dp, info = minres(A_dp, b_dp, x0=x0_dp, tol=1e-10, maxiter=5)
        assert _rel_residual(A_np, x_dp, b_np) < 1e-8

    def test_minres_returns_dpnp_array(self):
        n = 4
        A_dp = dpnp.eye(n, dtype=numpy.float64)
        b_dp = dpnp.ones(n, dtype=numpy.float64)
        x_dp, _ = minres(A_dp, b_dp)
        assert isinstance(x_dp, dpnp.ndarray)

    def test_minres_already_zero_rhs(self):
        n = 5
        A_dp = dpnp.eye(n, dtype=numpy.float64)
        b_dp = dpnp.zeros(n, dtype=numpy.float64)
        x_dp, info = minres(A_dp, b_dp)
        assert info == 0
        assert_allclose(_to_numpy(x_dp), numpy.zeros(n), atol=1e-14)

    def test_minres_non_square_raises(self):
        A_dp = dpnp.ones((4, 6), dtype=numpy.float64)
        b_dp = dpnp.ones(4, dtype=numpy.float64)
        with pytest.raises(ValueError, match="square"):
            minres(A_dp, b_dp)

    def test_minres_with_shift(self):
        rng = numpy.random.default_rng(304)
        n = 8
        dtype = numpy.float64
        A_np = _make_spd(n, dtype, rng)
        b_np = rng.standard_normal(n).astype(dtype)
        A_dp = dpnp.asarray(A_np)
        b_dp = dpnp.asarray(b_np)

        x_dp, info = minres(A_dp, b_dp, tol=1e-8, shift=0.0)
        assert info == 0
        assert _rel_residual(A_np, x_dp, b_np) < 1e-6

    def test_minres_with_linear_operator(self):
        rng = numpy.random.default_rng(305)
        n = 10
        dtype = numpy.float64
        A_np = _make_spd(n, dtype, rng)
        A_dp = dpnp.asarray(A_np)
        b_np = rng.standard_normal(n).astype(dtype)
        b_dp = dpnp.asarray(b_np)

        op = LinearOperator((n, n), matvec=lambda x: A_dp @ x, dtype=dtype)
        x_dp, info = minres(op, b_dp, tol=1e-8, maxiter=500)
        assert info == 0
        assert _rel_residual(A_np, x_dp, b_np) < 1e-6

    def test_minres_with_preconditioner(self):
        rng = numpy.random.default_rng(306)
        n = 10
        dtype = numpy.float64
        A_np = _make_spd(n, dtype, rng)
        A_dp = dpnp.asarray(A_np)
        b_np = rng.standard_normal(n).astype(dtype)
        b_dp = dpnp.asarray(b_np)

        diag_A = numpy.diag(A_np)
        M_np = numpy.diag(1.0 / diag_A)
        M_dp = dpnp.asarray(M_np)

        op_M = LinearOperator((n, n), matvec=lambda x: M_dp @ x, dtype=dtype)
        x_dp, info = minres(A_dp, b_dp, M=op_M, tol=1e-8, maxiter=500)
        assert info == 0
        assert _rel_residual(A_np, x_dp, b_np) < 1e-5


# ---------------------------------------------------------------------------
# Cross-solver consistency
# ---------------------------------------------------------------------------

class TestSolverConsistency:
    """Verify that CG, GMRES, and MINRES agree on SPD systems."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_scipy(self):
        pytest.importorskip("scipy", reason="SciPy required for minres in consistency tests")

    @pytest.mark.parametrize("n", [8, 16])
    def test_cg_gmres_minres_agree_spd(self, n):
        rng = numpy.random.default_rng(400)
        dtype = numpy.float64
        A_np = _make_spd(n, dtype, rng)
        b_np = rng.standard_normal(n).astype(dtype)
        A_dp = dpnp.asarray(A_np)
        b_dp = dpnp.asarray(b_np)

        x_cg, info_cg = cg(A_dp, b_dp, tol=1e-10, maxiter=500)
        x_gm, info_gm = gmres(A_dp, b_dp, tol=1e-10, maxiter=50, restart=n)
        x_mr, info_mr = minres(A_dp, b_dp, tol=1e-10, maxiter=500)

        assert info_cg == 0 and info_gm == 0 and info_mr == 0

        assert_allclose(_to_numpy(x_cg), _to_numpy(x_gm), rtol=1e-5,
                        err_msg="CG and GMRES disagree")
        assert_allclose(_to_numpy(x_cg), _to_numpy(x_mr), rtol=1e-5,
                        err_msg="CG and MINRES disagree")

    def test_all_solvers_vs_numpy_direct(self):
        rng = numpy.random.default_rng(401)
        n = 12
        dtype = numpy.float64
        A_np = _make_spd(n, dtype, rng)
        b_np = rng.standard_normal(n).astype(dtype)
        A_dp = dpnp.asarray(A_np)
        b_dp = dpnp.asarray(b_np)
        x_ref = numpy.linalg.solve(A_np, b_np)

        x_cg, _ = cg(A_dp, b_dp, tol=1e-11, maxiter=500)
        x_gm, _ = gmres(A_dp, b_dp, tol=1e-11, maxiter=50, restart=n)
        x_mr, _ = minres(A_dp, b_dp, tol=1e-11, maxiter=500)

        for name, x_dp in [("cg", x_cg), ("gmres", x_gm), ("minres", x_mr)]:
            assert_allclose(
                _to_numpy(x_dp), x_ref, rtol=1e-7,
                err_msg=f"{name} deviates from numpy.linalg.solve"
            )


# ---------------------------------------------------------------------------
# Import-level smoke test
# ---------------------------------------------------------------------------

def test_public_api_importable():
    """Verify all four public names are importable from the module."""
    from dpnp.scipy.sparse.linalg import (  # noqa: F401
        LinearOperator,
        aslinearoperator,
        cg,
        gmres,
        minres,
    )
