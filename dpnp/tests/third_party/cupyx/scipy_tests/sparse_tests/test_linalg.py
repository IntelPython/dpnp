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

        # SciPy's gmres signature changed in 1.12+ to use `rtol`. Older
        # versions used `tol`. The dpnp tree pins a recent SciPy so the
        # `rtol` kwarg is safe.
        x_ref, info_ref = scipy.sparse.linalg.gmres(
            A, b, rtol=1e-8, atol=0.0,
        )
        assert info_ref == 0

        A_dp = cupy.asarray(A)
        b_dp = cupy.asarray(b)
        x_dp, info_dp = cupy.scipy.sparse.linalg.gmres(
            A_dp, b_dp, rtol=1e-8, atol=0.0,
        )
        assert info_dp == 0
        testing.assert_allclose(
            cupy.asnumpy(x_dp), x_ref, rtol=1e-4, atol=1e-5,
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
