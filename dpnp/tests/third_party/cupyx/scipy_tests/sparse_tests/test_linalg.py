from __future__ import annotations

import unittest

import numpy
import pytest

import dpnp as cupy
from dpnp.tests.third_party.cupy import testing

if cupy.tests.helper.is_scipy_available():
    import scipy.sparse
    import scipy.sparse.linalg


def _spd_matrix(n, dtype, seed=0):
    rng = numpy.random.RandomState(seed)
    R = rng.rand(n, n).astype(dtype)
    if numpy.dtype(dtype).kind == "c":
        R = R + 1j * rng.rand(n, n).astype(dtype)
    return R @ R.conj().T + n * numpy.eye(n, dtype=dtype)


def _diag_dominant(n, dtype, seed=0):
    rng = numpy.random.RandomState(seed)
    A = rng.rand(n, n).astype(dtype)
    if numpy.dtype(dtype).kind == "c":
        A = A + 1j * rng.rand(n, n).astype(dtype)
    return A + n * numpy.eye(n, dtype=dtype)


def _sym_indef(n, dtype, seed=0):
    rng = numpy.random.RandomState(seed)
    A = rng.rand(n, n).astype(dtype)
    A = 0.5 * (A + A.T)
    return A - 0.5 * numpy.eye(n, dtype=dtype)


def _rhs(n, dtype, seed=1):
    rng = numpy.random.RandomState(seed)
    b = rng.rand(n).astype(dtype)
    if numpy.dtype(dtype).kind == "c":
        b = b + 1j * rng.rand(n).astype(dtype)
    return b


class TestLinearOperator(unittest.TestCase):

    @testing.for_dtypes("fdFD")
    def test_explicit_dtype_preserved(self, dtype):
        n = 4
        A = cupy.scipy.sparse.linalg.LinearOperator(
            (n, n),
            matvec=lambda v: v,
            dtype=dtype,
        )
        assert A.dtype == numpy.dtype(dtype)

    @testing.for_dtypes("fdFD")
    def test_dtype_inferred_from_int8_trial(self, dtype):
        n = 4

        def mv(v):
            return v.astype(dtype)

        A = cupy.scipy.sparse.linalg.LinearOperator((n, n), matvec=mv)
        assert A.dtype == numpy.dtype(dtype)

    def test_matvec_dimension_mismatch_raises(self):
        n = 4
        A = cupy.scipy.sparse.linalg.LinearOperator(
            (n, n),
            matvec=lambda v: v,
            dtype=cupy.float64,
        )
        wrong = cupy.zeros(n + 1, dtype=cupy.float64)
        with pytest.raises(ValueError):
            A.matvec(wrong)

    def test_matmul_dispatch(self):
        n = 3
        diag = cupy.asarray([1.0, 2.0, 3.0])
        A = cupy.scipy.sparse.linalg.LinearOperator(
            (n, n),
            matvec=lambda v: diag * v,
            dtype=cupy.float64,
        )
        x = cupy.asarray([10.0, 20.0, 30.0])
        testing.assert_allclose(cupy.asnumpy(A @ x), [10.0, 40.0, 90.0])
        testing.assert_allclose(cupy.asnumpy(A * x), [10.0, 40.0, 90.0])

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
        n = 3
        A = cupy.scipy.sparse.linalg.LinearOperator(
            (n, n),
            matvec=lambda v: v,
            dtype=cupy.float64,
        )
        assert getattr(A, "__array_ufunc__", "missing") is None

    def test_numpy_scalar_times_linop_dispatches_to_rmul(self):
        n = 3
        A = cupy.scipy.sparse.linalg.LinearOperator(
            (n, n),
            matvec=lambda v: v,
            dtype=cupy.float64,
        )
        scaled = numpy.float64(2.0) * A
        assert isinstance(scaled, cupy.scipy.sparse.linalg.LinearOperator)
        x = cupy.ones(n, dtype=cupy.float64)
        testing.assert_allclose(
            cupy.asnumpy(scaled.matvec(x)),
            2.0 * numpy.ones(n),
        )

    def test_dot_rejects_numpy_array(self):
        n = 4
        A = cupy.scipy.sparse.linalg.LinearOperator(
            (n, n),
            matvec=lambda v: v,
            dtype=cupy.float64,
        )
        host_vec = numpy.ones(n, dtype=numpy.float64)
        with pytest.raises(TypeError, match="numpy.ndarray"):
            A.dot(host_vec)
        with pytest.raises(TypeError, match="numpy.ndarray"):
            A @ host_vec
        with pytest.raises(TypeError, match="numpy.ndarray"):
            A * host_vec

    def test_dot_accepts_dpnp_array_after_explicit_transfer(self):
        n = 4
        A = cupy.scipy.sparse.linalg.LinearOperator(
            (n, n),
            matvec=lambda v: 2 * v,
            dtype=cupy.float64,
        )
        host_vec = numpy.ones(n, dtype=numpy.float64)
        dev_vec = cupy.asarray(host_vec)
        result = A.dot(dev_vec)
        testing.assert_allclose(
            cupy.asnumpy(result),
            2.0 * numpy.ones(n),
        )

    def test_scaled_operator_preserves_float32_dtype(self):
        n = 3
        A = cupy.scipy.sparse.linalg.LinearOperator(
            (n, n),
            matvec=lambda v: v,
            dtype=cupy.float32,
        )
        scaled = numpy.float32(2.0) * A
        assert scaled.dtype == numpy.dtype("float32")


class TestAsLinearOperator(unittest.TestCase):

    def test_passthrough_existing_linear_operator(self):
        n = 3
        A = cupy.scipy.sparse.linalg.LinearOperator(
            (n, n),
            matvec=lambda v: v,
            dtype=cupy.float64,
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
        y_ref = A_np @ cupy.asnumpy(x)
        testing.assert_allclose(cupy.asnumpy(y), y_ref, rtol=1e-5, atol=1e-6)

    def test_reject_numpy_ndarray(self):
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
            A_dp,
            b_dp,
            rtol=1e-8,
            atol=0.0,
        )
        assert info_dp == 0
        testing.assert_allclose(
            cupy.asnumpy(x_dp),
            x_ref,
            rtol=1e-4,
            atol=1e-5,
        )

    @testing.for_dtypes("fd")
    def test_cg_warm_start(self, dtype):
        n = 8
        A = _spd_matrix(n, dtype)
        b = _rhs(n, dtype)

        A_dp = cupy.asarray(A)
        b_dp = cupy.asarray(b)
        x0_dp, _ = cupy.scipy.sparse.linalg.cg(
            A_dp,
            b_dp,
            rtol=1e-3,
            atol=0.0,
        )
        x_dp, info_dp = cupy.scipy.sparse.linalg.cg(
            A_dp,
            b_dp,
            x0=x0_dp,
            rtol=1e-8,
            atol=0.0,
        )
        assert info_dp == 0
        x_ref, _ = scipy.sparse.linalg.cg(A, b, rtol=1e-8, atol=0.0)
        testing.assert_allclose(
            cupy.asnumpy(x_dp),
            x_ref,
            rtol=1e-4,
            atol=1e-5,
        )

    def test_cg_info_contract_unconverged_is_positive(self):
        n = 32
        A = _spd_matrix(n, numpy.float64)
        b = _rhs(n, numpy.float64)
        A_dp = cupy.asarray(A)
        b_dp = cupy.asarray(b)
        _, info = cupy.scipy.sparse.linalg.cg(
            A_dp,
            b_dp,
            maxiter=1,
            rtol=1e-12,
            atol=0.0,
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
        n = 8
        # Rank-deficient: row 0 is zero, A is PSD but not PD.
        A = numpy.eye(n, dtype=numpy.float64)
        A[0, 0] = 0.0
        b = numpy.ones(n, dtype=numpy.float64)
        A_dp = cupy.asarray(A)
        b_dp = cupy.asarray(b)
        _, info = cupy.scipy.sparse.linalg.cg(
            A_dp,
            b_dp,
            maxiter=20,
            rtol=1e-12,
            atol=0.0,
        )
        assert info > 0


@testing.with_requires("scipy")
class TestGMRES(unittest.TestCase):

    @testing.for_dtypes("fd")
    def test_gmres_converges_diag_dominant(self, dtype):
        n = 10
        A = _diag_dominant(n, dtype)
        b = _rhs(n, dtype)

        # float32 cannot reliably reach 1e-8 in 10 Arnoldi steps;
        # the noise floor of classical Gram-Schmidt is O(eps*sqrt(n)).
        rtol = 1e-5 if numpy.dtype(dtype) == numpy.float32 else 1e-8

        x_ref, info_ref = scipy.sparse.linalg.gmres(
            A,
            b,
            rtol=rtol,
            atol=0.0,
        )
        assert info_ref == 0

        A_dp = cupy.asarray(A)
        b_dp = cupy.asarray(b)
        x_dp, info_dp = cupy.scipy.sparse.linalg.gmres(
            A_dp,
            b_dp,
            rtol=rtol,
            atol=0.0,
        )
        assert info_dp == 0
        cmp_rtol = 5e-4 if numpy.dtype(dtype) == numpy.float32 else 1e-4
        cmp_atol = 5e-5 if numpy.dtype(dtype) == numpy.float32 else 1e-5
        testing.assert_allclose(
            cupy.asnumpy(x_dp),
            x_ref,
            rtol=cmp_rtol,
            atol=cmp_atol,
        )

    def test_gmres_restart_parameter(self):
        n = 20
        A = _diag_dominant(n, numpy.float64)
        b = _rhs(n, numpy.float64)
        A_dp = cupy.asarray(A)
        b_dp = cupy.asarray(b)
        x_dp, info_dp = cupy.scipy.sparse.linalg.gmres(
            A_dp,
            b_dp,
            restart=5,
            rtol=1e-8,
            atol=0.0,
        )
        assert info_dp == 0
        testing.assert_allclose(
            cupy.asnumpy(A_dp @ x_dp),
            cupy.asnumpy(b_dp),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_gmres_info_contract_unconverged_is_positive(self):
        n = 32
        A = _diag_dominant(n, numpy.float64)
        b = _rhs(n, numpy.float64)
        A_dp = cupy.asarray(A)
        b_dp = cupy.asarray(b)
        _, info = cupy.scipy.sparse.linalg.gmres(
            A_dp,
            b_dp,
            restart=2,
            maxiter=1,
            rtol=1e-12,
            atol=0.0,
        )
        assert info > 0

    @testing.for_dtypes("FD")
    def test_gmres_complex_arnoldi_fast_path(self, dtype):
        n = 12
        A = _diag_dominant(n, dtype)
        b = _rhs(n, dtype)

        rtol = 1e-5 if numpy.dtype(dtype) == numpy.complex64 else 1e-7

        x_ref, info_ref = scipy.sparse.linalg.gmres(
            A,
            b,
            rtol=rtol,
            atol=0.0,
        )
        assert info_ref == 0

        A_dp = cupy.asarray(A)
        b_dp = cupy.asarray(b)
        x_dp, info_dp = cupy.scipy.sparse.linalg.gmres(
            A_dp,
            b_dp,
            rtol=rtol,
            atol=0.0,
        )
        assert info_dp == 0
        cmp_rtol = 5e-4 if numpy.dtype(dtype) == numpy.complex64 else 1e-4
        cmp_atol = 5e-5 if numpy.dtype(dtype) == numpy.complex64 else 1e-5
        testing.assert_allclose(
            cupy.asnumpy(x_dp),
            x_ref,
            rtol=cmp_rtol,
            atol=cmp_atol,
        )


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
            A_dp,
            b_dp,
            rtol=1e-8,
        )
        assert info_dp == 0
        testing.assert_allclose(
            cupy.asnumpy(x_dp),
            x_ref,
            rtol=1e-4,
            atol=1e-5,
        )

    def test_minres_shift_parameter(self):
        n = 10
        A = _sym_indef(n, numpy.float64)
        b = _rhs(n, numpy.float64)
        shift = 0.25
        x_ref, _ = scipy.sparse.linalg.minres(
            A,
            b,
            shift=shift,
            rtol=1e-8,
        )
        A_dp = cupy.asarray(A)
        b_dp = cupy.asarray(b)
        x_dp, _ = cupy.scipy.sparse.linalg.minres(
            A_dp,
            b_dp,
            shift=shift,
            rtol=1e-8,
        )
        testing.assert_allclose(
            cupy.asnumpy(x_dp),
            x_ref,
            rtol=1e-4,
            atol=1e-5,
        )

    def test_minres_zero_rhs_returns_zero(self):
        n = 4
        A_dp = cupy.asarray(_sym_indef(n, numpy.float64))
        b_dp = cupy.zeros(n, dtype=cupy.float64)
        x, info = cupy.scipy.sparse.linalg.minres(A_dp, b_dp)
        assert info == 0
        testing.assert_allclose(cupy.asnumpy(x), numpy.zeros(n))


class TestModuleSurface(unittest.TestCase):

    def test_public_symbols_match_pr_contract(self):
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
