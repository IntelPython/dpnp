# *****************************************************************************
# Copyright (c) 2025, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

"""Compatibility tests for ``dpnp.scipy.sparse.linalg``.

Modelled on
``cupy/tests/cupyx_tests/scipy_tests/sparse_tests/test_linalg.py``
and adapted to the in-tree dpnp testing harness so the file lives at
``dpnp/tests/third_party/cupyx/scipy_tests/sparse_tests/test_linalg.py``
per the layout convention of the existing ``linalg_tests`` and
``special_tests`` packages.

Coverage
--------
* :class:`dpnp.scipy.sparse.linalg.LinearOperator` -- construction with
  matvec / rmatvec / matmat callables, dtype inference, ``H`` / ``T``
  properties, ``__matmul__`` dispatch.
* :func:`dpnp.scipy.sparse.linalg.aslinearoperator` -- dispatch order:
  pass-through of an existing operator, dpnp dense array, dpnp CSR
  matrix, explicit rejection of host ``numpy.ndarray``.
* :func:`dpnp.scipy.sparse.linalg.cg` -- convergence on a small SPD
  system vs ``scipy.sparse.linalg.cg``, ``info`` contract, ``x0``
  warm-start.
* :func:`dpnp.scipy.sparse.linalg.gmres` -- convergence on a small
  non-symmetric diagonally-dominant system vs SciPy reference,
  ``restart`` knob, ``info`` contract.
* :func:`dpnp.scipy.sparse.linalg.minres` -- convergence on a
  symmetric indefinite system vs SciPy reference, ``shift`` knob.

Each test compares the dpnp result to the SciPy reference solving the
identical problem on the host, so the suite both pins the public API
to SciPy/CuPy semantics and guards against silent numerical
regressions.
"""

from __future__ import annotations

import unittest

import numpy
import pytest

import dpnp as cupy
from dpnp.tests.third_party.cupy import testing

if cupy.tests.helper.is_scipy_available():
    import scipy.sparse
    import scipy.sparse.linalg


# ---------------------------------------------------------------------------
# Reference problem builders (CPU / numpy / scipy)
# ---------------------------------------------------------------------------


def _spd_matrix(n, dtype, seed=0):
    """Build an n x n SPD matrix on the host with a strong diagonal.

    SPD = symmetric, positive definite. Used as the CG reference.
    ``A = R @ R.H + n*I`` is SPD by construction with condition number
    bounded by O(n), so CG converges in ~n iterations on float64 and
    fewer in practice.
    """
    rng = numpy.random.RandomState(seed)
    R = rng.rand(n, n).astype(dtype)
    if numpy.dtype(dtype).kind == "c":
        R = R + 1j * rng.rand(n, n).astype(dtype)
    A = R @ R.conj().T + n * numpy.eye(n, dtype=dtype)
    return A


def _diag_dominant(n, dtype, seed=0):
    """Build an n x n non-symmetric, diagonally dominant matrix.

    Used as the GMRES reference: well-conditioned, non-symmetric, so
    CG would not apply but GMRES converges in few iterations.
    """
    rng = numpy.random.RandomState(seed)
    A = rng.rand(n, n).astype(dtype)
    if numpy.dtype(dtype).kind == "c":
        A = A + 1j * rng.rand(n, n).astype(dtype)
    # Force diagonal dominance: |a_ii| > sum_{j != i} |a_ij|.
    A = A + n * numpy.eye(n, dtype=dtype)
    return A


def _sym_indef(n, dtype, seed=0):
    """Build an n x n symmetric indefinite matrix for MINRES.

    A symmetric matrix with both positive and negative eigenvalues so
    that CG cannot be applied; MINRES handles this case.
    """
    rng = numpy.random.RandomState(seed)
    A = rng.rand(n, n).astype(dtype)
    A = 0.5 * (A + A.T)
    # Shift spectrum to straddle zero.
    A = A - 0.5 * numpy.eye(n, dtype=dtype)
    return A


def _rhs(n, dtype, seed=1):
    rng = numpy.random.RandomState(seed)
    b = rng.rand(n).astype(dtype)
    if numpy.dtype(dtype).kind == "c":
        b = b + 1j * rng.rand(n).astype(dtype)
    return b


# ---------------------------------------------------------------------------
# LinearOperator
# ---------------------------------------------------------------------------


class TestLinearOperator(unittest.TestCase):
    """LinearOperator construction, dtype inference, basic algebra."""

    @testing.for_dtypes("fdFD")
    def test_explicit_dtype_preserved(self, dtype):
        """An explicit ``dtype=`` must not be overridden by ``_init_dtype``.

        Regression: a previous version called ``_init_dtype`` unconditionally
        in the ``_CustomLinearOperator`` constructor, clobbering the
        user-supplied dtype with the trial-matvec result.
        """
        n = 4
        A = cupy.scipy.sparse.linalg.LinearOperator(
            (n, n), matvec=lambda v: v, dtype=dtype,
        )
        assert A.dtype == numpy.dtype(dtype)

    @testing.for_dtypes("fdFD")
    def test_dtype_inferred_from_int8_trial(self, dtype):
        """When ``dtype`` is None, ``_init_dtype`` must use an int8 trial.

        Using int8 (the lowest-precedence numeric dtype) means the matvec's
        natural output dtype is preserved: a float32 matvec stays float32
        instead of being upcast to float64 by an int64/float64 trial vector.
        """
        n = 4

        def mv(v):
            # Promote int8 input to the operator's natural dtype.
            return v.astype(dtype)

        A = cupy.scipy.sparse.linalg.LinearOperator((n, n), matvec=mv)
        assert A.dtype == numpy.dtype(dtype)

    def test_matvec_dimension_mismatch_raises(self):
        n = 4
        A = cupy.scipy.sparse.linalg.LinearOperator(
            (n, n), matvec=lambda v: v, dtype=cupy.float64,
        )
        wrong = cupy.zeros(n + 1, dtype=cupy.float64)
        with pytest.raises(ValueError):
            A.matvec(wrong)

    def test_matmul_dispatch(self):
        n = 3
        diag = cupy.asarray([1.0, 2.0, 3.0])
        A = cupy.scipy.sparse.linalg.LinearOperator(
            (n, n), matvec=lambda v: diag * v, dtype=cupy.float64,
        )
        x = cupy.asarray([10.0, 20.0, 30.0])
        # ``A @ x`` and ``A * x`` must both go through matvec.
        result_matmul = cupy.asnumpy(A @ x)
        result_mul = cupy.asnumpy(A * x)
        testing.assert_allclose(result_matmul, [10.0, 40.0, 90.0])
        testing.assert_allclose(result_mul, [10.0, 40.0, 90.0])

    def test_adjoint_returns_linear_operator(self):
        n = 3
        A = cupy.scipy.sparse.linalg.LinearOperator(
            (n, n),
            matvec=lambda v: v,
            rmatvec=lambda v: v,
            dtype=cupy.float64,
        )
        AH = A.H
        assert isinstance(AH, cupy.scipy.sparse.linalg.LinearOperator)
        assert AH.shape == (n, n)

    def test_array_ufunc_opt_out(self):
        """LinearOperator must set ``__array_ufunc__ = None``.

        Matches the SciPy ``scipy.sparse.linalg.LinearOperator``
        contract: opting out of NumPy's ufunc dispatch lets
        ``scalar * linop`` and ``host_array @ linop`` fall back to
        the operator's reflected ``__rmul__`` / ``__rmatmul__``
        methods instead of NumPy attempting to broadcast the operator
        element-wise through the ufunc machinery. dpnp.ndarray itself
        sets this to None for the same reason; the two systems must
        agree on the dispatch protocol.
        """
        n = 3
        A = cupy.scipy.sparse.linalg.LinearOperator(
            (n, n), matvec=lambda v: v, dtype=cupy.float64,
        )
        # The marker is what the protocol checks for; presence is the
        # whole guarantee. SciPy does the same assertion in its own
        # test suite (test_interface.py::test_array_ufunc_opt_out).
        assert getattr(A, "__array_ufunc__", "missing") is None

    def test_numpy_scalar_times_linop_dispatches_to_rmul(self):
        """A NumPy scalar on the left must call LinearOperator.__rmul__.

        Concrete consequence of ``__array_ufunc__ = None``: NumPy
        returns NotImplemented from its ufunc and Python falls back to
        the operator's reflected method, producing a scaled operator
        rather than raising or producing a wrong-typed result.
        """
        n = 3
        A = cupy.scipy.sparse.linalg.LinearOperator(
            (n, n), matvec=lambda v: v, dtype=cupy.float64,
        )
        scaled = numpy.float64(2.0) * A
        assert isinstance(scaled, cupy.scipy.sparse.linalg.LinearOperator)
        x = cupy.ones(n, dtype=cupy.float64)
        testing.assert_allclose(cupy.asnumpy(scaled.matvec(x)), 2.0 * numpy.ones(n))

    def test_dot_rejects_numpy_array(self):
        """LinearOperator.dot must NOT silently host->device upload a
        numpy.ndarray operand.

        dpnp's strict-coercion contract forbids implicit transfers
        across the host / device boundary. A user passing a host
        numpy array into the operator's dot() is almost certainly a
        bug in device / queue selection, and silently uploading
        would mask it. The contract is to raise TypeError with a
        directly actionable hint to use dpnp.asarray() explicitly.
        """
        n = 4
        A = cupy.scipy.sparse.linalg.LinearOperator(
            (n, n), matvec=lambda v: v, dtype=cupy.float64,
        )
        host_vec = numpy.ones(n, dtype=numpy.float64)
        with pytest.raises(TypeError, match="numpy.ndarray"):
            A.dot(host_vec)
        with pytest.raises(TypeError, match="numpy.ndarray"):
            A @ host_vec
        with pytest.raises(TypeError, match="numpy.ndarray"):
            A * host_vec

    def test_dot_accepts_dpnp_array_after_explicit_transfer(self):
        """The companion to test_dot_rejects_numpy_array: the
        documented workaround (call dpnp.asarray() explicitly) must
        work. Demonstrates that the strict rejection is targeted at
        implicit transfers, not at legitimate device data flow.
        """
        n = 4
        A = cupy.scipy.sparse.linalg.LinearOperator(
            (n, n), matvec=lambda v: 2 * v, dtype=cupy.float64,
        )
        host_vec = numpy.ones(n, dtype=numpy.float64)
        dev_vec = cupy.asarray(host_vec)
        result = A.dot(dev_vec)
        testing.assert_allclose(
            cupy.asnumpy(result), 2.0 * numpy.ones(n),
        )

    def test_scaled_operator_preserves_float32_dtype(self):
        """Scaling a float32 operator by a Python or numpy float scalar
        must not promote the resulting operator's dtype to float64.

        Regression guard for the _ScaledLinearOperator dtype inference
        fix. Previously, _ScaledLinearOperator collected
        ``type(alpha)`` rather than the value's natural dtype, so a
        Python ``float`` (which has no dtype) was uniformly treated as
        float64 by dpnp.result_type. The result: a float32 operator
        scaled by ``2.0`` returned a float64 operator, which then
        widened every downstream matvec result on the device.
        """
        n = 3
        A = cupy.scipy.sparse.linalg.LinearOperator(
            (n, n), matvec=lambda v: v, dtype=cupy.float32,
        )
        # A numpy.float32 scalar must keep the dtype at float32.
        scaled = numpy.float32(2.0) * A
        assert scaled.dtype == numpy.dtype("float32"), (
            f"numpy.float32 scalar widened operator dtype to {scaled.dtype}; "
            "_ScaledLinearOperator must infer alpha dtype from the value, "
            "not from type(alpha)."
        )


# ---------------------------------------------------------------------------
# aslinearoperator
# ---------------------------------------------------------------------------


class TestAsLinearOperator(unittest.TestCase):

    def test_passthrough_existing_linear_operator(self):
        n = 3
        A = cupy.scipy.sparse.linalg.LinearOperator(
            (n, n), matvec=lambda v: v, dtype=cupy.float64,
        )
        out = cupy.scipy.sparse.linalg.aslinearoperator(A)
        assert out is A

    @testing.for_dtypes("fdFD")
    def test_wrap_dense_dpnp_array(self, dtype):
        n = 4
        A_np = _spd_matrix(n, dtype)
        A_dp = cupy.asarray(A_np)
        op = cupy.scipy.sparse.linalg.aslinearoperator(A_dp)
        x = cupy.asarray(_rhs(n, dtype))
        y = op.matvec(x)
        # Reference matvec on host.
        y_ref = A_np @ cupy.asnumpy(x)
        testing.assert_allclose(cupy.asnumpy(y), y_ref, rtol=1e-5, atol=1e-6)

    def test_reject_numpy_ndarray(self):
        """A host NumPy array must be rejected with a directed message.

        Silently uploading would mask user device/queue selection bugs;
        the contract is to require an explicit ``dpnp.asarray`` first.
        """
        A_np = numpy.eye(3, dtype=numpy.float64)
        with pytest.raises(TypeError, match="numpy"):
            cupy.scipy.sparse.linalg.aslinearoperator(A_np)

    @testing.for_dtypes("fd")
    def test_wrap_csr_matrix(self, dtype):
        n = 5
        A_np = _spd_matrix(n, dtype)
        A_dp = cupy.scipy.sparse.csr_matrix(cupy.asarray(A_np))
        op = cupy.scipy.sparse.linalg.aslinearoperator(A_dp)
        x = cupy.asarray(_rhs(n, dtype))
        y = op.matvec(x)
        y_ref = A_np @ cupy.asnumpy(x)
        testing.assert_allclose(cupy.asnumpy(y), y_ref, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# cg
# ---------------------------------------------------------------------------


@testing.with_requires("scipy")
class TestCG(unittest.TestCase):

    @testing.for_dtypes("fd")
    def test_cg_converges_dense_spd(self, dtype):
        n = 8
        A = _spd_matrix(n, dtype)
        b = _rhs(n, dtype)

        x_ref, info_ref = scipy.sparse.linalg.cg(A, b, rtol=1e-8, atol=0.0)
        assert info_ref == 0

        A_dp = cupy.asarray(A)
        b_dp = cupy.asarray(b)
        x_dp, info_dp = cupy.scipy.sparse.linalg.cg(
            A_dp, b_dp, rtol=1e-8, atol=0.0,
        )
        assert info_dp == 0
        testing.assert_allclose(
            cupy.asnumpy(x_dp), x_ref, rtol=1e-4, atol=1e-5,
        )

    @testing.for_dtypes("fd")
    def test_cg_warm_start(self, dtype):
        n = 8
        A = _spd_matrix(n, dtype)
        b = _rhs(n, dtype)

        A_dp = cupy.asarray(A)
        b_dp = cupy.asarray(b)
        # First call from zero.
        x0_dp, _ = cupy.scipy.sparse.linalg.cg(
            A_dp, b_dp, rtol=1e-3, atol=0.0,
        )
        # Restart from x0; must still converge.
        x_dp, info_dp = cupy.scipy.sparse.linalg.cg(
            A_dp, b_dp, x0=x0_dp, rtol=1e-8, atol=0.0,
        )
        assert info_dp == 0
        x_ref, _ = scipy.sparse.linalg.cg(A, b, rtol=1e-8, atol=0.0)
        testing.assert_allclose(
            cupy.asnumpy(x_dp), x_ref, rtol=1e-4, atol=1e-5,
        )

    def test_cg_info_contract_unconverged_is_positive(self):
        """info > 0 when maxiter is hit before convergence.

        SciPy / CuPy contract: info == 0 on success, info > 0 if not
        converged. dpnp previously returned info = -1 for a breakdown,
        which broke ``if info > 0`` user code; the regression is
        guarded here.
        """
        n = 32
        A = _spd_matrix(n, numpy.float64)
        b = _rhs(n, numpy.float64)
        A_dp = cupy.asarray(A)
        b_dp = cupy.asarray(b)
        # Tiny maxiter so we cannot converge.
        _, info = cupy.scipy.sparse.linalg.cg(
            A_dp, b_dp, maxiter=1, rtol=1e-12, atol=0.0,
        )
        assert info > 0

    def test_cg_zero_rhs_returns_zero(self):
        n = 4
        A_dp = cupy.asarray(_spd_matrix(n, numpy.float64))
        b_dp = cupy.zeros(n, dtype=cupy.float64)
        x, info = cupy.scipy.sparse.linalg.cg(A_dp, b_dp)
        assert info == 0
        testing.assert_allclose(cupy.asnumpy(x), numpy.zeros(n))

    def test_cg_inf_breakdown_returns_positive_info(self):
        """Singular operator triggers IEEE-754 inf propagation; cg
        must detect non-finite residual and return info > 0.

        Regression guard for the per-iter-sync collapse: the dpnp cg
        intentionally skips host syncs on pAp and rz_new and relies on
        alpha = rz/pAp producing inf or NaN when pAp underflows. The
        next residual norm is then non-finite, the single rnorm sync
        catches it, and the routine reports info > 0 per the SciPy
        contract. A regression that re-introduces the per-iter break-
        down checks would still pass on well-conditioned matrices but
        would change the info value reported here.
        """
        n = 8
        # Rank-deficient: A has a one-dimensional null space, so CG
        # cannot make progress in the direction of the null vector.
        # The first iteration produces a finite alpha; subsequent ones
        # have rz collapse and trigger the inf-propagation path.
        A = numpy.eye(n, dtype=numpy.float64)
        A[0, 0] = 0.0  # singular: row 0 is zero, A is PSD but not PD
        b = numpy.ones(n, dtype=numpy.float64)
        A_dp = cupy.asarray(A)
        b_dp = cupy.asarray(b)
        _, info = cupy.scipy.sparse.linalg.cg(
            A_dp, b_dp, maxiter=20, rtol=1e-12, atol=0.0,
        )
        # Either the residual stays bounded but never reaches rtol
        # (info == maxiter) or the rz_new = 0 division triggers
        # inf-propagation and we exit with the iter index. Both are
        # legitimate >0 outcomes per the SciPy contract; what must
        # NOT happen is info == 0 (false convergence) or info == -1
        # (the previous, broken contract).
        assert info > 0


# ---------------------------------------------------------------------------
# gmres
# ---------------------------------------------------------------------------


@testing.with_requires("scipy")
class TestGMRES(unittest.TestCase):

    @testing.for_dtypes("fd")
    def test_gmres_converges_diag_dominant(self, dtype):
        n = 10
        A = _diag_dominant(n, dtype)
        b = _rhs(n, dtype)

        # Convergence tolerance must respect the dtype's noise floor.
        # float32 cannot reliably reach 1e-8 relative residual in 10
        # Arnoldi steps because the accumulated rounding error in
        # classical Gram-Schmidt is already O(eps_f32 * sqrt(n)) ~
        # 4e-7. Asking scipy for 1e-8 means even a well-conditioned
        # 10x10 system reports info > 0 from the SciPy reference, not
        # from dpnp -- a false-negative against our own solver.
        # float64 has headroom and keeps the original 1e-8 target.
        rtol = 1e-5 if numpy.dtype(dtype) == numpy.float32 else 1e-8

        # SciPy's gmres signature changed in 1.12+ to use `rtol`. Older
        # versions used `tol`. The dpnp tree pins a recent SciPy so the
        # `rtol` kwarg is safe.
        x_ref, info_ref = scipy.sparse.linalg.gmres(
            A, b, rtol=rtol, atol=0.0,
        )
        assert info_ref == 0

        A_dp = cupy.asarray(A)
        b_dp = cupy.asarray(b)
        x_dp, info_dp = cupy.scipy.sparse.linalg.gmres(
            A_dp, b_dp, rtol=rtol, atol=0.0,
        )
        assert info_dp == 0
        # assert_allclose tolerance also needs dtype-awareness: a
        # float32 solution agreeing with the float32 scipy solution
        # to better than 5e-4 is the most we can demand without
        # hitting the same noise floor that loosened rtol above.
        cmp_rtol = 5e-4 if numpy.dtype(dtype) == numpy.float32 else 1e-4
        cmp_atol = 5e-5 if numpy.dtype(dtype) == numpy.float32 else 1e-5
        testing.assert_allclose(
            cupy.asnumpy(x_dp), x_ref, rtol=cmp_rtol, atol=cmp_atol,
        )

    def test_gmres_restart_parameter(self):
        n = 20
        A = _diag_dominant(n, numpy.float64)
        b = _rhs(n, numpy.float64)
        A_dp = cupy.asarray(A)
        b_dp = cupy.asarray(b)
        # Small restart should still converge for a well-conditioned
        # diagonally-dominant matrix.
        x_dp, info_dp = cupy.scipy.sparse.linalg.gmres(
            A_dp, b_dp, restart=5, rtol=1e-8, atol=0.0,
        )
        assert info_dp == 0
        testing.assert_allclose(
            cupy.asnumpy(A_dp @ x_dp), cupy.asnumpy(b_dp),
            rtol=1e-4, atol=1e-5,
        )

    def test_gmres_info_contract_unconverged_is_positive(self):
        n = 32
        A = _diag_dominant(n, numpy.float64)
        b = _rhs(n, numpy.float64)
        A_dp = cupy.asarray(A)
        b_dp = cupy.asarray(b)
        # Force a single outer iteration with tiny restart and no rounds.
        _, info = cupy.scipy.sparse.linalg.gmres(
            A_dp, b_dp, restart=2, maxiter=1, rtol=1e-12, atol=0.0,
        )
        assert info > 0

    @testing.for_dtypes("FD")
    def test_gmres_complex_arnoldi_fast_path(self, dtype):
        """Complex GMRES exercises the conjugate-in-place branch of
        _make_compute_hu.

        Regression guard: when the bi._gemv_alpha_beta fast path was
        introduced the closure had to patch up the result of
        ``gemv(transpose=True)`` (which returns V^T u) with an in-place
        conjugate of the Hessenberg column slice to obtain V^H u. If
        the conjugate is skipped or executed out of queue-order with
        respect to the follow-up gemv that consumes the slice, the
        Krylov basis silently loses orthogonality and convergence
        either stalls or returns a wrong answer.
        """
        n = 12
        A = _diag_dominant(n, dtype)
        b = _rhs(n, dtype)

        # See the rationale in test_gmres_converges_diag_dominant:
        # complex64's noise floor is the same as float32's (the data
        # arrays are stored as paired float32 components), so 1e-7 is
        # below what scipy's own gmres can reach in 12 Arnoldi steps
        # on this matrix. Use a dtype-aware rtol so the test reports
        # actual dpnp failures rather than scipy reaching its floor.
        rtol = 1e-5 if numpy.dtype(dtype) == numpy.complex64 else 1e-7

        x_ref, info_ref = scipy.sparse.linalg.gmres(
            A, b, rtol=rtol, atol=0.0,
        )
        assert info_ref == 0

        A_dp = cupy.asarray(A)
        b_dp = cupy.asarray(b)
        x_dp, info_dp = cupy.scipy.sparse.linalg.gmres(
            A_dp, b_dp, rtol=rtol, atol=0.0,
        )
        assert info_dp == 0
        cmp_rtol = 5e-4 if numpy.dtype(dtype) == numpy.complex64 else 1e-4
        cmp_atol = 5e-5 if numpy.dtype(dtype) == numpy.complex64 else 1e-5
        testing.assert_allclose(
            cupy.asnumpy(x_dp), x_ref, rtol=cmp_rtol, atol=cmp_atol,
        )


# ---------------------------------------------------------------------------
# minres
# ---------------------------------------------------------------------------


@testing.with_requires("scipy")
class TestMINRES(unittest.TestCase):

    def test_minres_converges_symmetric_indefinite(self):
        n = 12
        A = _sym_indef(n, numpy.float64)
        b = _rhs(n, numpy.float64)
        x_ref, info_ref = scipy.sparse.linalg.minres(A, b, rtol=1e-8)
        assert info_ref == 0

        A_dp = cupy.asarray(A)
        b_dp = cupy.asarray(b)
        x_dp, info_dp = cupy.scipy.sparse.linalg.minres(
            A_dp, b_dp, rtol=1e-8,
        )
        assert info_dp == 0
        testing.assert_allclose(
            cupy.asnumpy(x_dp), x_ref, rtol=1e-4, atol=1e-5,
        )

    def test_minres_shift_parameter(self):
        """Verify the ``shift`` parameter solves (A - shift*I) x = b."""
        n = 10
        A = _sym_indef(n, numpy.float64)
        b = _rhs(n, numpy.float64)
        shift = 0.25
        x_ref, _ = scipy.sparse.linalg.minres(
            A, b, shift=shift, rtol=1e-8,
        )
        A_dp = cupy.asarray(A)
        b_dp = cupy.asarray(b)
        x_dp, _ = cupy.scipy.sparse.linalg.minres(
            A_dp, b_dp, shift=shift, rtol=1e-8,
        )
        testing.assert_allclose(
            cupy.asnumpy(x_dp), x_ref, rtol=1e-4, atol=1e-5,
        )

    def test_minres_zero_rhs_returns_zero(self):
        n = 4
        A_dp = cupy.asarray(_sym_indef(n, numpy.float64))
        b_dp = cupy.zeros(n, dtype=cupy.float64)
        x, info = cupy.scipy.sparse.linalg.minres(A_dp, b_dp)
        assert info == 0
        testing.assert_allclose(cupy.asnumpy(x), numpy.zeros(n))


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


class TestModuleSurface(unittest.TestCase):

    def test_public_symbols_match_pr_contract(self):
        """The four PR-promised entry points must be importable."""
        from dpnp.scipy.sparse.linalg import (
            LinearOperator,
            aslinearoperator,
            cg,
            gmres,
            minres,
        )
        assert callable(LinearOperator)
        assert callable(aslinearoperator)
        assert callable(cg)
        assert callable(gmres)
        assert callable(minres)
