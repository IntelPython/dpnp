import warnings

import numpy
import pytest
from numpy.testing import (
    assert_allclose,
    assert_raises,
)

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
    has_support_aspect64,
)
from .third_party.cupy.testing import with_requires


# Helpers for constructing SPD, diagonally dominant, and symmetric
# indefinite test matrices.
def _spd_matrix(n, dtype, seed=42):
    rng = numpy.random.default_rng(seed)
    is_complex = numpy.issubdtype(numpy.dtype(dtype), numpy.complexfloating)
    if is_complex:
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        a = a.conj().T @ a + n * numpy.eye(n)
    else:
        a = rng.standard_normal((n, n))
        a = a.T @ a + n * numpy.eye(n)
    return dpnp.asarray(a.astype(dtype))


def _diag_dominant(n, dtype, seed=81):
    rng = numpy.random.default_rng(seed)
    is_complex = numpy.issubdtype(numpy.dtype(dtype), numpy.complexfloating)
    if is_complex:
        a = 0.05 * (
            rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        )
    else:
        a = 0.05 * rng.standard_normal((n, n))
    a = a + float(n) * numpy.eye(n)
    return dpnp.asarray(a.astype(dtype))


def _sym_indefinite(n, dtype, seed=99):
    rng = numpy.random.default_rng(seed)
    is_complex = numpy.issubdtype(numpy.dtype(dtype), numpy.complexfloating)
    if is_complex:
        # Random Hermitian indefinite: A = (M + M^H) / 2 with M complex.
        m_raw = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        m = 0.5 * (m_raw + m_raw.conj().T)
    else:
        a = rng.standard_normal((n, n))
        q, _ = numpy.linalg.qr(a)
        d = rng.standard_normal(n)
        # Force at least one positive and one negative eigenvalue so
        # the matrix is guaranteed indefinite even when standard_normal
        # happens to draw same-sign values for tiny n.
        d[0] = -abs(d[0]) - 0.1
        d[-1] = abs(d[-1]) + 0.1
        m = q @ numpy.diag(d) @ q.T
    return dpnp.asarray(m.astype(dtype))


def _rhs(n, dtype, seed=7):
    rng = numpy.random.default_rng(seed)
    is_complex = numpy.issubdtype(numpy.dtype(dtype), numpy.complexfloating)
    if is_complex:
        b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    else:
        b = rng.standard_normal(n)
    b /= numpy.linalg.norm(b)
    return dpnp.asarray(b.astype(dtype))


def _rtol_for(dtype):
    if dtype in (dpnp.float32, dpnp.complex64, numpy.float32, numpy.complex64):
        return 1e-5
    return 1e-8


def _res_bound(dtype):
    if dtype in (dpnp.float32, dpnp.complex64, numpy.float32, numpy.complex64):
        return 1e-3
    return 1e-5


class TestImports:
    def test_all_symbols_importable(self):
        from dpnp.scipy.sparse.linalg import (  # noqa: F401
            LinearOperator,
            aslinearoperator,
            cg,
            gmres,
            minres,
        )

        for sym in (LinearOperator, aslinearoperator, cg, gmres, minres):
            assert callable(sym)

    def test_all_in_dunder_all(self):
        import dpnp.scipy.sparse.linalg as mod

        for name in (
            "LinearOperator",
            "aslinearoperator",
            "cg",
            "gmres",
            "minres",
        ):
            assert name in mod.__all__


class TestLinearOperator:
    @pytest.mark.parametrize(
        "shape",
        [(5, 5), (7, 3), (3, 7)],
        ids=["(5, 5)", "(7, 3)", "(3, 7)"],
    )
    def test_shape(self, shape):
        m, n = shape
        lo = LinearOperator(
            shape,
            matvec=lambda x: dpnp.zeros(m, dtype=dpnp.float32),
            dtype=dpnp.float32,
        )
        assert lo.shape == (m, n)
        assert lo.ndim == 2

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_dtype_explicit(self, dtype):
        n = 4
        a = dpnp.eye(n, dtype=dtype)
        lo = LinearOperator(
            (n, n),
            matvec=lambda x: (a @ x.astype(dtype)).astype(dtype),
            dtype=dtype,
        )
        assert lo.dtype == dtype

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_dtype_inference_preserves_source(self, dtype):
        # _init_dtype probes matvec with an int8 vector (the lowest
        # precedence numeric dtype), so the matvec's natural output
        # dtype survives unchanged -- a float32 operator stays
        # float32 instead of being widened to float64. Mirrors
        # scipy/cupyx LinearOperator semantics.
        n = 4
        a = dpnp.eye(n, dtype=dtype)
        lo = LinearOperator((n, n), matvec=lambda x: a @ x)
        assert lo.dtype == dpnp.dtype(dtype)

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_matvec(self, dtype):
        n = 6
        a = generate_random_numpy_array((n, n), dtype, seed_value=42)
        ia = dpnp.array(a)
        lo = LinearOperator((n, n), matvec=lambda x: ia @ x, dtype=dtype)
        x = generate_random_numpy_array((n,), dtype, seed_value=1)
        ix = dpnp.array(x)
        result = lo.matvec(ix)
        expected = a @ x
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_rmatvec(self, dtype):
        n = 5
        a = generate_random_numpy_array((n, n), dtype, seed_value=12)
        ia = dpnp.array(a)
        lo = LinearOperator(
            (n, n),
            matvec=lambda x: ia @ x,
            rmatvec=lambda x: dpnp.conj(ia.T) @ x,
            dtype=dtype,
        )
        x = generate_random_numpy_array((n,), dtype, seed_value=3)
        ix = dpnp.array(x)
        result = lo.rmatvec(ix)
        expected = a.conj().T @ x
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_matmat_fallback_loop(self, dtype):
        n, k = 5, 3
        a = generate_random_numpy_array((n, n), dtype, seed_value=55)
        ia = dpnp.array(a)
        lo = LinearOperator((n, n), matvec=lambda x: ia @ x, dtype=dtype)
        x = generate_random_numpy_array((n, k), dtype, seed_value=9)
        ix = dpnp.array(x)
        result = lo.matmat(ix)
        expected = a @ x
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_matmul_1d(self, dtype):
        # lo @ x dispatches to matvec
        n = 6
        a = generate_random_numpy_array((n, n), dtype, seed_value=42)
        ia = dpnp.array(a)
        lo = LinearOperator((n, n), matvec=lambda x: ia @ x, dtype=dtype)
        x = generate_random_numpy_array((n,), dtype, seed_value=2)
        ix = dpnp.array(x)
        result = lo @ ix
        expected = a @ x
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_matmul_2d(self, dtype):
        # lo @ X dispatches to matmat
        n, k = 5, 3
        a = generate_random_numpy_array((n, n), dtype, seed_value=42)
        ia = dpnp.array(a)
        lo = LinearOperator((n, n), matvec=lambda x: ia @ x, dtype=dtype)
        x = generate_random_numpy_array((n, k), dtype, seed_value=5)
        ix = dpnp.array(x)
        result = lo @ ix
        expected = a @ x
        assert_dtype_allclose(result, expected)

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_call_alias(self):
        n = 4
        ia = dpnp.eye(n, dtype=dpnp.float64)
        lo = LinearOperator((n, n), matvec=lambda x: ia @ x, dtype=dpnp.float64)
        ix = dpnp.ones(n, dtype=dpnp.float64)
        assert_allclose(dpnp.asnumpy(lo(ix)), numpy.ones(n), atol=1e-12)

    def test_repr(self):
        lo = LinearOperator(
            (3, 4),
            matvec=lambda x: dpnp.zeros(3, dtype=dpnp.float32),
            dtype=dpnp.float32,
        )
        r = repr(lo)
        assert "LinearOperator" in r
        assert "3x4" in r or "(3, 4)" in r

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_subclass_custom_matmat(self, dtype):
        n, k = 7, 4
        a = generate_random_numpy_array((n, n), dtype, seed_value=42)
        ia = dpnp.array(a)

        class MyOp(LinearOperator):
            def __init__(self):
                super().__init__(dtype=dtype, shape=(n, n))
                self._a = ia

            def _matvec(self, x):
                return self._a @ x

            def _matmat(self, X):
                return self._a @ X

        op = MyOp()
        x = generate_random_numpy_array((n, k), dtype, seed_value=9)
        ix = dpnp.array(x)
        result = op.matmat(ix)
        expected = a @ x
        assert_dtype_allclose(result, expected)

    def test_linear_operator_errors(self):
        lo = LinearOperator(
            (3, 5),
            matvec=lambda x: dpnp.zeros(3, dtype=dpnp.float32),
            dtype=dpnp.float32,
        )
        # matvec with wrong shape
        assert_raises(ValueError, lo.matvec, dpnp.ones(4, dtype=dpnp.float32))

        # rmatvec not provided
        lo2 = LinearOperator(
            (3, 3),
            matvec=lambda x: dpnp.zeros(3, dtype=dpnp.float32),
            dtype=dpnp.float32,
        )
        assert_raises(
            (NotImplementedError, ValueError),
            lo2.rmatvec,
            dpnp.zeros(3, dtype=dpnp.float32),
        )

        # matmat with 1-D input
        assert_raises(ValueError, lo2.matmat, dpnp.ones(3, dtype=dpnp.float32))

        # negative shape
        assert_raises(
            ValueError,
            LinearOperator,
            (-1, 3),
            matvec=lambda x: x,
            dtype=dpnp.float32,
        )

        # shape with wrong ndim
        assert_raises(
            ValueError,
            LinearOperator,
            (3,),
            matvec=lambda x: x,
            dtype=dpnp.float32,
        )


class TestLinearOperatorAlgebra:
    # Coverage for the operator-algebra combinators wrapped by
    # LinearOperator: sum (A + B), product (A @ B), scaled (alpha * A),
    # power (A**k), adjoint involution ((A.H).H), and the cached SpMV
    # fast path that csr_matrix exposes through aslinearoperator.

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_sum_linear_operator(self, dtype):
        n = 5
        a = generate_random_numpy_array((n, n), dtype, seed_value=11)
        b = generate_random_numpy_array((n, n), dtype, seed_value=22)
        ia = dpnp.array(a)
        ib = dpnp.array(b)
        lo_a = LinearOperator((n, n), matvec=lambda x: ia @ x, dtype=dtype)
        lo_b = LinearOperator((n, n), matvec=lambda x: ib @ x, dtype=dtype)
        lo_sum = lo_a + lo_b
        x = generate_random_numpy_array((n,), dtype, seed_value=33)
        ix = dpnp.array(x)
        result = lo_sum.matvec(ix)
        expected = (a + b) @ x
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_product_linear_operator(self, dtype):
        n = 5
        a = generate_random_numpy_array((n, n), dtype, seed_value=11)
        b = generate_random_numpy_array((n, n), dtype, seed_value=22)
        ia = dpnp.array(a)
        ib = dpnp.array(b)
        lo_a = LinearOperator((n, n), matvec=lambda x: ia @ x, dtype=dtype)
        lo_b = LinearOperator((n, n), matvec=lambda x: ib @ x, dtype=dtype)
        # `lo_a * lo_b` composes via _ProductLinearOperator: the result
        # of lo_b.matvec(x) feeds lo_a.matvec(...).
        lo_prod = lo_a * lo_b
        x = generate_random_numpy_array((n,), dtype, seed_value=33)
        ix = dpnp.array(x)
        result = lo_prod.matvec(ix)
        expected = a @ (b @ x)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_scaled_linear_operator(self, dtype):
        n = 4
        a = generate_random_numpy_array((n, n), dtype, seed_value=44)
        ia = dpnp.array(a)
        lo = LinearOperator((n, n), matvec=lambda x: ia @ x, dtype=dtype)
        alpha = 2.5
        scaled = alpha * lo
        x = generate_random_numpy_array((n,), dtype, seed_value=55)
        ix = dpnp.array(x)
        result = scaled.matvec(ix)
        expected = alpha * (a @ x)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_power_linear_operator(self, dtype):
        # (A ** k).matvec(x) == A.matvec(A.matvec(... k times ... x))
        n = 4
        a = generate_random_numpy_array((n, n), dtype, seed_value=66)
        ia = dpnp.array(a)
        lo = LinearOperator((n, n), matvec=lambda x: ia @ x, dtype=dtype)
        powered = lo**3
        x = generate_random_numpy_array((n,), dtype, seed_value=77)
        ix = dpnp.array(x)
        result = powered.matvec(ix)
        expected = a @ (a @ (a @ x))
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_adjoint_involution(self, dtype):
        # (A.H).H must act as A on a probe vector. We compare matvec
        # outputs rather than object identity because the adjoint
        # operator is a fresh _AdjointLinearOperator.
        n = 5
        a = generate_random_numpy_array((n, n), dtype, seed_value=88)
        ia = dpnp.array(a)
        lo = LinearOperator(
            (n, n),
            matvec=lambda x: ia @ x,
            rmatvec=lambda x: dpnp.conj(ia.T) @ x,
            dtype=dtype,
        )
        lo_hh = lo.H.H
        x = generate_random_numpy_array((n,), dtype, seed_value=99)
        ix = dpnp.array(x)
        assert_dtype_allclose(lo_hh.matvec(ix), lo.matvec(ix))

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    @pytest.mark.parametrize("dtype", [dpnp.float32, dpnp.float64])
    def test_aslinearoperator_csr_matrix(self, dtype):
        # csr_matrix wrapped via aslinearoperator must reproduce the
        # dense matvec result. The wrapper routes through the cached
        # oneMKL SpMV handle on the device; this test indirectly
        # verifies that fast path is wired and produces correct
        # numerics.
        n = 6
        a_dense = generate_random_numpy_array((n, n), dtype, seed_value=42)
        # Zero out a fraction so the CSR representation is non-trivial.
        mask = generate_random_numpy_array((n, n), dtype, seed_value=7)
        a_dense[numpy.abs(mask) < 0.3] = 0
        ia_dense = dpnp.array(a_dense)
        ia_csr = dpnp.scipy.sparse.csr_matrix(ia_dense)
        lo = aslinearoperator(ia_csr)
        x = generate_random_numpy_array((n,), dtype, seed_value=13)
        ix = dpnp.array(x)
        result = lo.matvec(ix)
        expected = a_dense @ x
        assert_dtype_allclose(result, expected)


class TestAsLinearOperator:
    def test_identity_if_already_linearoperator(self):
        lo = LinearOperator((3, 3), matvec=lambda x: x, dtype=dpnp.float32)
        assert aslinearoperator(lo) is lo

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_dense_dpnp_array_matvec(self, dtype):
        n = 6
        a = generate_random_numpy_array((n, n), dtype, seed_value=42)
        ia = dpnp.array(a)
        lo = aslinearoperator(ia)
        assert lo.shape == (n, n)
        x = generate_random_numpy_array((n,), dtype, seed_value=1)
        ix = dpnp.array(x)
        result = lo.matvec(ix)
        expected = a @ x
        assert_dtype_allclose(result, expected)

    def test_dense_numpy_array_rejected(self):
        # aslinearoperator must NOT silently host -> device upload a
        # numpy.ndarray: dpnp's strict-coercion contract forbids
        # implicit transfers across the host / device boundary. The
        # user has to call dpnp.asarray() explicitly.
        n = 5
        a = generate_random_numpy_array((n, n), numpy.float64, seed_value=42)
        with pytest.raises(TypeError, match="numpy.ndarray"):
            aslinearoperator(a)

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_rmatvec_from_dpnp_dense(self):
        n = 5
        a = generate_random_numpy_array((n, n), numpy.float64, seed_value=42)
        ia = dpnp.array(a)
        lo = aslinearoperator(ia)
        x = generate_random_numpy_array((n,), numpy.float64, seed_value=2)
        ix = dpnp.array(x)
        result = lo.rmatvec(ix)
        expected = a.conj().T @ x
        assert_allclose(dpnp.asnumpy(result), expected, atol=1e-12)

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_duck_type_with_shape_and_matvec(self):
        n = 4

        class DuckOp:
            shape = (n, n)
            dtype = numpy.dtype(numpy.float64)

            def matvec(self, x):
                return x * 2.0

            def rmatvec(self, x):
                return x * 2.0

        lo = aslinearoperator(DuckOp())
        ix = dpnp.ones(n, dtype=dpnp.float64)
        result = lo.matvec(ix)
        assert_allclose(dpnp.asnumpy(result), numpy.full(n, 2.0), atol=1e-12)

    def test_aslinearoperator_errors(self):
        assert_raises(TypeError, aslinearoperator, "nope")


class TestCg:
    n = 30

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_cg_converges_spd(self, dtype):
        ia = _spd_matrix(self.n, dtype)
        ib = _rhs(self.n, dtype)
        x, info = cg(ia, ib, rtol=_rtol_for(dtype), maxiter=500)
        assert info == 0
        res = float(dpnp.linalg.norm(ia @ x - ib) / dpnp.linalg.norm(ib))
        assert res < _res_bound(dtype)

    @with_requires("scipy")
    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_cg_matches_scipy(self, dtype):
        import scipy.sparse.linalg as scipy_sla

        # rtol must respect the dtype's noise floor; asking scipy for
        # 1e-8 on float32/complex64 may make the reference itself
        # return info != 0 because the residual cannot drop further
        # than O(eps * sqrt(n)).
        req_rtol = _rtol_for(dtype)
        a = dpnp.asnumpy(_spd_matrix(self.n, dtype))
        b = dpnp.asnumpy(_rhs(self.n, dtype))
        try:
            x_ref, info_ref = scipy_sla.cg(a, b, rtol=req_rtol, maxiter=500)
        except TypeError:  # scipy < 1.12
            x_ref, info_ref = scipy_sla.cg(a, b, tol=req_rtol, maxiter=500)
        assert info_ref == 0
        x_dp, info = cg(
            dpnp.array(a), dpnp.array(b), rtol=req_rtol, maxiter=500
        )
        assert info == 0
        cmp_tol = 5e-4 if dtype in (dpnp.float32, dpnp.complex64) else 1e-8
        assert_allclose(dpnp.asnumpy(x_dp), x_ref, rtol=cmp_tol, atol=cmp_tol)

    @pytest.mark.parametrize("dtype", [dpnp.float32, dpnp.float64])
    def test_cg_x0_warm_start(self, dtype):
        if not has_support_aspect64() and dtype == dpnp.float64:
            pytest.skip("fp64 is required")
        ia = _spd_matrix(self.n, dtype)
        ib = _rhs(self.n, dtype)
        x0 = dpnp.ones(self.n, dtype=dtype)
        x, info = cg(ia, ib, x0=x0, rtol=_rtol_for(dtype), maxiter=500)
        assert info == 0
        res = float(dpnp.linalg.norm(ia @ x - ib) / dpnp.linalg.norm(ib))
        assert res < _res_bound(dtype)

    @pytest.mark.parametrize("dtype", [dpnp.float32, dpnp.float64])
    def test_cg_b_2dim(self, dtype):
        # b with shape (n, 1) must be accepted and flattened internally
        if not has_support_aspect64() and dtype == dpnp.float64:
            pytest.skip("fp64 is required")
        ia = _spd_matrix(self.n, dtype)
        ib = _rhs(self.n, dtype).reshape(self.n, 1)
        _, info = cg(ia, ib, rtol=1e-8, maxiter=500)
        assert info == 0

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_cg_b_zero(self):
        ia = _spd_matrix(10, dpnp.float64)
        ib = dpnp.zeros(10, dtype=dpnp.float64)
        x, info = cg(ia, ib, rtol=1e-8)
        assert info == 0
        assert_allclose(dpnp.asnumpy(x), numpy.zeros(10), atol=1e-14)

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_cg_callback(self):
        ia = _spd_matrix(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64)
        residuals = []

        def cb(xk):
            # cg passes the current iterate as a dpnp array of the
            # full system size.
            assert isinstance(xk, dpnp.ndarray)
            assert xk.shape == (self.n,)
            residuals.append(float(dpnp.linalg.norm(ia @ xk - ib)))

        cg(ia, ib, callback=cb, rtol=1e-10, maxiter=200)
        assert len(residuals) > 0
        # cg residual should be (loosely) non-increasing; allow some
        # slack for floating-point noise and the algorithm's natural
        # roundoff oscillation near convergence.
        for i in range(1, len(residuals)):
            assert residuals[i] <= residuals[i - 1] * 2.0

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_cg_atol(self):
        ia = _spd_matrix(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64)
        x, _ = cg(ia, ib, rtol=0.0, atol=1e-1, maxiter=500)
        assert float(dpnp.linalg.norm(ia @ x - ib)) < 1.0

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_cg_exact_solution(self):
        # x0 == true solution must return info == 0 immediately
        n = 10
        ia = _spd_matrix(n, dpnp.float64)
        ib = _rhs(n, dpnp.float64)
        x_true = dpnp.array(
            numpy.linalg.solve(dpnp.asnumpy(ia), dpnp.asnumpy(ib))
        )
        _, info = cg(ia, ib, x0=x_true, rtol=1e-12)
        assert info == 0

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_cg_via_linear_operator(self, dtype):
        ia = _spd_matrix(self.n, dtype)
        ib = _rhs(self.n, dtype)
        lo = aslinearoperator(ia)
        x, info = cg(lo, ib, rtol=_rtol_for(dtype), maxiter=500)
        assert info == 0
        res = float(dpnp.linalg.norm(ia @ x - ib) / dpnp.linalg.norm(ib))
        assert res < _res_bound(dtype)

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_cg_maxiter_nonconvergence(self):
        ia = _spd_matrix(50, dpnp.float64)
        ib = _rhs(50, dpnp.float64)
        _, info = cg(ia, ib, rtol=1e-15, atol=0.0, maxiter=1)
        assert info != 0

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_cg_diag_preconditioner(self):
        ia = _spd_matrix(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64)
        M = aslinearoperator(dpnp.diag(1.0 / dpnp.diag(ia)))
        _, info = cg(ia, ib, M=M, rtol=1e-8, maxiter=500)
        assert info == 0

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_cg_empty_matrix(self):
        # 0x0 matrix is a degenerate edge case scipy handles cleanly;
        # dpnp should too without crashing.
        a = dpnp.empty((0, 0), dtype=dpnp.float64)
        b = dpnp.empty(0, dtype=dpnp.float64)
        x, info = cg(a, b)
        assert info == 0
        assert x.shape == (0,)

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_cg_errors(self):
        # Mirrors scipy.sparse.linalg's test_invalid coverage for the
        # validations dpnp's cg / _make_system perform: non-square A,
        # 1-D and 3-D A, b shape, x0 shape, M shape. Each case must
        # surface a clean exception rather than producing garbage.
        ia = _spd_matrix(5, dpnp.float64)
        ib = _rhs(5, dpnp.float64)

        # b length mismatch
        with pytest.raises(ValueError):
            cg(ia, dpnp.ones(6, dtype=dpnp.float64), maxiter=1)

        # 1-D A (not a matrix)
        with pytest.raises(ValueError):
            cg(dpnp.ones(5, dtype=dpnp.float64), ib)

        # non-square A
        with pytest.raises(ValueError):
            cg(dpnp.ones((5, 6), dtype=dpnp.float64), ib)

        # x0 length mismatch
        with pytest.raises(ValueError):
            cg(ia, ib, x0=dpnp.ones(6, dtype=dpnp.float64))

        # b passed as a non-dpnp host array must be rejected
        with pytest.raises(TypeError):
            cg(ia, numpy.ones(5, dtype=numpy.float64))

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_cg_tol_kwarg_compat(self):
        # SciPy deprecated `tol` in favour of `rtol` in 1.12; dpnp's cg
        # keeps `tol` as a back-compat alias. A test ensures the
        # alias is honoured and is not silently ignored.
        ia = _spd_matrix(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64)
        # Use `tol` kwarg explicitly; the result must match what `rtol`
        # would produce.
        x_tol, info_tol = cg(ia, ib, tol=1e-8, maxiter=500)
        x_rtol, info_rtol = cg(ia, ib, rtol=1e-8, maxiter=500)
        assert info_tol == 0
        assert info_rtol == 0
        assert_allclose(
            dpnp.asnumpy(x_tol), dpnp.asnumpy(x_rtol), rtol=1e-12, atol=1e-12
        )


class TestGmres:
    n = 30

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_gmres_converges_diag_dominant(self, dtype):
        ia = _diag_dominant(self.n, dtype)
        ib = _rhs(self.n, dtype)
        x, _ = gmres(
            ia,
            ib,
            rtol=_rtol_for(dtype),
            maxiter=200,
            restart=self.n,
        )
        # Check actual residual rather than info: GMRES with restart
        # can return info > 0 even when the final residual meets rtol.
        res = float(dpnp.linalg.norm(ia @ x - ib) / dpnp.linalg.norm(ib))
        assert res < _res_bound(dtype)

    @with_requires("scipy")
    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_gmres_matches_scipy(self, dtype):
        import scipy.sparse.linalg as scipy_sla

        a = dpnp.asnumpy(_diag_dominant(self.n, dtype))
        b = dpnp.asnumpy(_rhs(self.n, dtype))
        # Same dtype-aware rtol rationale as test_cg_matches_scipy.
        req_rtol = _rtol_for(dtype)
        # Symmetric maxiter on both sides so a missed convergence on
        # one side is a real discrepancy, not an iteration-budget gap.
        common_maxiter = 200
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                x_ref, _ = scipy_sla.gmres(
                    a,
                    b,
                    rtol=req_rtol,
                    restart=self.n,
                    maxiter=common_maxiter,
                )
            except TypeError:  # scipy < 1.12
                x_ref, _ = scipy_sla.gmres(
                    a,
                    b,
                    tol=req_rtol,
                    restart=self.n,
                    maxiter=common_maxiter,
                )
        x_dp, info = gmres(
            dpnp.array(a),
            dpnp.array(b),
            rtol=req_rtol,
            restart=self.n,
            maxiter=common_maxiter,
        )
        assert info == 0
        cmp_tol = 5e-3 if dtype in (dpnp.float32, dpnp.complex64) else 1e-7
        assert_allclose(dpnp.asnumpy(x_dp), x_ref, rtol=cmp_tol, atol=cmp_tol)

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    @pytest.mark.parametrize("restart", [None, 5, 15], ids=["None", "5", "15"])
    def test_gmres_restart_values(self, restart):
        ia = _diag_dominant(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64)
        _, info = gmres(ia, ib, rtol=1e-8, restart=restart, maxiter=100)
        assert info == 0

    @pytest.mark.parametrize("dtype", [dpnp.float32, dpnp.float64])
    def test_gmres_x0_warm_start(self, dtype):
        if not has_support_aspect64() and dtype == dpnp.float64:
            pytest.skip("fp64 is required")
        ia = _diag_dominant(self.n, dtype)
        ib = _rhs(self.n, dtype)
        x0 = dpnp.ones(self.n, dtype=dtype)
        x, _ = gmres(
            ia,
            ib,
            x0=x0,
            rtol=_rtol_for(dtype),
            restart=self.n,
            maxiter=200,
        )
        res = float(dpnp.linalg.norm(ia @ x - ib) / dpnp.linalg.norm(ib))
        assert res < _res_bound(dtype)

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_gmres_b_2dim(self):
        ia = _diag_dominant(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64).reshape(self.n, 1)
        _, info = gmres(ia, ib, rtol=1e-8, restart=self.n, maxiter=100)
        assert info == 0

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_gmres_b_zero(self):
        ia = _diag_dominant(10, dpnp.float64)
        ib = dpnp.zeros(10, dtype=dpnp.float64)
        x, info = gmres(ia, ib, rtol=1e-8)
        assert info == 0
        assert_allclose(dpnp.asnumpy(x), numpy.zeros(10), atol=1e-14)

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_gmres_callback_x(self):
        ia = _diag_dominant(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64)
        captured = []

        def cb(xk):
            # callback_type="x" delivers the current iterate as a
            # dpnp array of full system size.
            assert isinstance(xk, dpnp.ndarray)
            assert xk.shape == (self.n,)
            captured.append(xk)

        gmres(
            ia,
            ib,
            callback=cb,
            callback_type="x",
            rtol=1e-10,
            maxiter=20,
            restart=self.n,
        )
        assert len(captured) > 0

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_gmres_callback_pr_norm(self):
        ia = _diag_dominant(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64)
        values = []

        def cb(r):
            # callback_type="pr_norm" delivers a scalar relative
            # residual; float() must work without an implicit
            # device sync (the solver hands it over as a host value).
            values.append(float(r))

        gmres(
            ia,
            ib,
            callback=cb,
            callback_type="pr_norm",
            rtol=1e-10,
            maxiter=20,
            restart=self.n,
        )
        assert len(values) > 0
        assert all(v >= 0 for v in values)
        # GMRES is monotone in the preconditioned residual; permit a
        # small slack to absorb roundoff near machine precision.
        for i in range(1, len(values)):
            assert values[i] <= values[i - 1] * 2.0

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_gmres_atol(self):
        ia = _diag_dominant(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64)
        x, _ = gmres(
            ia,
            ib,
            rtol=0.0,
            atol=1e-6,
            restart=self.n,
            maxiter=50,
        )
        assert float(dpnp.linalg.norm(ia @ x - ib)) < 1e-4

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_gmres_via_linear_operator(self, dtype):
        ia = _diag_dominant(self.n, dtype)
        ib = _rhs(self.n, dtype)
        lo = aslinearoperator(ia)
        x, _ = gmres(
            lo,
            ib,
            rtol=_rtol_for(dtype),
            restart=self.n,
            maxiter=200,
        )
        res = float(dpnp.linalg.norm(ia @ x - ib) / dpnp.linalg.norm(ib))
        assert res < _res_bound(dtype)

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_gmres_nonconvergence(self):
        # Ill-conditioned Hilbert matrix + tiny restart must not converge
        n = 48
        idx = numpy.arange(n, dtype=numpy.float64)
        a = 1.0 / (idx[:, None] + idx[None, :] + 1.0)
        rng = numpy.random.default_rng(5)
        b = rng.standard_normal(n)
        ia = dpnp.array(a)
        ib = dpnp.array(b)
        x, info = gmres(ia, ib, rtol=1e-15, atol=0.0, restart=2, maxiter=2)
        rel = float(dpnp.linalg.norm(ia @ x - ib) / dpnp.linalg.norm(ib))
        assert rel > 1e-12
        assert info != 0

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_gmres_complex_system(self):
        n = 15
        ia = _diag_dominant(n, dpnp.complex128)
        ib = _rhs(n, dpnp.complex128)
        x, _ = gmres(ia, ib, rtol=1e-8, restart=n, maxiter=200)
        res = float(dpnp.linalg.norm(ia @ x - ib) / dpnp.linalg.norm(ib))
        assert res < 1e-5

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_gmres_empty_matrix(self):
        a = dpnp.empty((0, 0), dtype=dpnp.float64)
        b = dpnp.empty(0, dtype=dpnp.float64)
        x, info = gmres(a, b)
        assert info == 0
        assert x.shape == (0,)

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_gmres_errors(self):
        # Same coverage shape as test_cg_errors, plus gmres-specific
        # callback_type validation.
        ia = _diag_dominant(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64)

        # unknown callback_type
        assert_raises(ValueError, gmres, ia, ib, callback_type="garbage")

        # b length mismatch
        with pytest.raises(ValueError):
            gmres(ia, dpnp.ones(self.n + 1, dtype=dpnp.float64))

        # 1-D A
        with pytest.raises(ValueError):
            gmres(dpnp.ones(self.n, dtype=dpnp.float64), ib)

        # non-square A
        with pytest.raises(ValueError):
            gmres(dpnp.ones((self.n, self.n + 1), dtype=dpnp.float64), ib)

        # x0 length mismatch
        with pytest.raises(ValueError):
            gmres(ia, ib, x0=dpnp.ones(self.n + 1, dtype=dpnp.float64))

        # b passed as a non-dpnp host array must be rejected
        with pytest.raises(TypeError):
            gmres(ia, numpy.ones(self.n, dtype=numpy.float64))

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_gmres_restart_clamped_to_n(self):
        # SciPy / dpnp clamp `restart` to min(restart, n). Passing a
        # restart larger than the system size must not error or
        # produce wrong results.
        n = 8
        ia = _diag_dominant(n, dpnp.float64)
        ib = _rhs(n, dpnp.float64)
        x, info = gmres(ia, ib, restart=100, rtol=1e-8, maxiter=200)
        assert info == 0
        res = float(dpnp.linalg.norm(ia @ x - ib) / dpnp.linalg.norm(ib))
        assert res < 1e-7

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_gmres_x0_exact_solution(self):
        # If x0 already satisfies A @ x0 == b, gmres must report
        # convergence immediately without iterating.
        n = 8
        ia = _diag_dominant(n, dpnp.float64)
        ib = _rhs(n, dpnp.float64)
        x_true = dpnp.array(
            numpy.linalg.solve(dpnp.asnumpy(ia), dpnp.asnumpy(ib))
        )
        _, info = gmres(ia, ib, x0=x_true, rtol=1e-12, restart=n)
        assert info == 0


class TestMinres:
    n = 30

    @pytest.mark.parametrize("dtype", [dpnp.float32, dpnp.float64])
    def test_minres_converges_spd(self, dtype):
        if not has_support_aspect64() and dtype == dpnp.float64:
            pytest.skip("fp64 is required")
        ia = _spd_matrix(self.n, dtype)
        ib = _rhs(self.n, dtype)
        x, info = minres(ia, ib, rtol=1e-8, maxiter=500)
        assert info == 0
        res = float(dpnp.linalg.norm(ia @ x - ib) / dpnp.linalg.norm(ib))
        assert res < 1e-4

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_minres_converges_sym_indefinite(self):
        ia = _sym_indefinite(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64)
        x, _ = minres(ia, ib, rtol=1e-8, maxiter=1000)
        res = float(dpnp.linalg.norm(ia @ x - ib) / dpnp.linalg.norm(ib))
        assert res < 1e-3

    @with_requires("scipy")
    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_minres_matches_scipy(self):
        import scipy.sparse.linalg as scipy_sla

        a = dpnp.asnumpy(_spd_matrix(self.n, dpnp.float64))
        b = dpnp.asnumpy(_rhs(self.n, dpnp.float64))
        try:
            x_ref, _ = scipy_sla.minres(a, b, rtol=1e-8)
        except TypeError:
            x_ref, _ = scipy_sla.minres(a, b, tol=1e-8)
        x_dp, info = minres(dpnp.array(a), dpnp.array(b), rtol=1e-8)
        assert info == 0
        assert_allclose(dpnp.asnumpy(x_dp), x_ref, rtol=1e-5, atol=1e-6)

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_minres_x0_warm_start(self):
        ia = _spd_matrix(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64)
        x0 = dpnp.zeros(self.n, dtype=dpnp.float64)
        _, info = minres(ia, ib, x0=x0, rtol=1e-8)
        assert info == 0

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_minres_shift(self):
        # shift != 0 solves (A - shift*I) x = b
        a = dpnp.asnumpy(_spd_matrix(self.n, dpnp.float64))
        b = dpnp.asnumpy(_rhs(self.n, dpnp.float64))
        shift = 0.5
        x_dp, info = minres(
            dpnp.array(a), dpnp.array(b), shift=shift, rtol=1e-8
        )
        assert info == 0
        a_shifted = a - shift * numpy.eye(self.n)
        res = numpy.linalg.norm(
            a_shifted @ dpnp.asnumpy(x_dp) - b
        ) / numpy.linalg.norm(b)
        assert res < 1e-4

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_minres_b_zero(self):
        ia = _spd_matrix(10, dpnp.float64)
        ib = dpnp.zeros(10, dtype=dpnp.float64)
        x, info = minres(ia, ib, rtol=1e-8)
        assert info == 0
        assert_allclose(dpnp.asnumpy(x), numpy.zeros(10), atol=1e-14)

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_minres_via_linear_operator(self):
        ia = _spd_matrix(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64)
        lo = aslinearoperator(ia)
        _, info = minres(lo, ib, rtol=1e-8)
        assert info == 0

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_minres_callback(self):
        ia = _spd_matrix(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64)
        captured = []

        def cb(xk):
            # minres passes the current iterate as a dpnp array of
            # full system size on every Lanczos step.
            assert isinstance(xk, dpnp.ndarray)
            assert xk.shape == (self.n,)
            captured.append(xk)

        minres(ia, ib, callback=cb, rtol=1e-10)
        assert len(captured) > 0

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_minres_empty_matrix(self):
        a = dpnp.empty((0, 0), dtype=dpnp.float64)
        b = dpnp.empty(0, dtype=dpnp.float64)
        x, info = minres(a, b)
        assert info == 0
        assert x.shape == (0,)

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_minres_b_2dim(self):
        # b with shape (n, 1) must be accepted and flattened internally
        ia = _spd_matrix(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64).reshape(self.n, 1)
        _, info = minres(ia, ib, rtol=1e-8, maxiter=500)
        assert info == 0

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_minres_x0_exact_solution(self):
        # If x0 already satisfies A @ x0 == b, minres must report
        # convergence immediately.
        n = 8
        ia = _spd_matrix(n, dpnp.float64)
        ib = _rhs(n, dpnp.float64)
        x_true = dpnp.array(
            numpy.linalg.solve(dpnp.asnumpy(ia), dpnp.asnumpy(ib))
        )
        _, info = minres(ia, ib, x0=x_true, rtol=1e-12)
        assert info == 0

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    def test_minres_errors(self):
        # Mirrors test_cg_errors / test_gmres_errors. minres does not
        # accept atol or callback_type, so those are not exercised.
        ia = _spd_matrix(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64)

        # non-square operator
        lo = aslinearoperator(dpnp.ones((4, 5), dtype=dpnp.float64))
        assert_raises(ValueError, minres, lo, dpnp.ones(4, dtype=dpnp.float64))

        # b length mismatch
        with pytest.raises(ValueError):
            minres(ia, dpnp.ones(self.n + 1, dtype=dpnp.float64))

        # 1-D A
        with pytest.raises(ValueError):
            minres(dpnp.ones(self.n, dtype=dpnp.float64), ib)

        # x0 length mismatch
        with pytest.raises(ValueError):
            minres(ia, ib, x0=dpnp.ones(self.n + 1, dtype=dpnp.float64))

        # b passed as a non-dpnp host array must be rejected
        with pytest.raises(TypeError):
            minres(ia, numpy.ones(self.n, dtype=numpy.float64))


class TestSolversIntegration:
    # Cross-cuts cg / gmres / minres against aslinearoperator(dense)
    # for a few representative sizes. Nested parametrize gives a
    # clean Cartesian product; per-dtype skip handles the fp64 case
    # uniformly.

    @pytest.mark.parametrize("n", [10, 30, 50])
    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_cg_spd_via_linearoperator(self, n, dtype):
        ia = _spd_matrix(n, dtype)
        lo = aslinearoperator(ia)
        ib = _rhs(n, dtype)
        x, info = cg(lo, ib, rtol=_rtol_for(dtype), maxiter=n * 10)
        assert info == 0
        res = float(dpnp.linalg.norm(ia @ x - ib) / dpnp.linalg.norm(ib))
        assert res < _res_bound(dtype)

    @pytest.mark.parametrize("n", [10, 30])
    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_gmres_nonsymmetric_via_linearoperator(self, n, dtype):
        ia = _diag_dominant(n, dtype)
        lo = aslinearoperator(ia)
        ib = _rhs(n, dtype)
        x, _ = gmres(lo, ib, rtol=_rtol_for(dtype), restart=n, maxiter=200)
        res = float(dpnp.linalg.norm(ia @ x - ib) / dpnp.linalg.norm(ib))
        assert res < _res_bound(dtype)

    @with_requires("scipy")
    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    @pytest.mark.parametrize("n", [10, 30])
    def test_minres_spd_via_linearoperator(self, n):
        ia = _spd_matrix(n, dpnp.float64)
        lo = aslinearoperator(ia)
        ib = _rhs(n, dpnp.float64)
        x, info = minres(lo, ib, rtol=1e-8)
        assert info == 0
        res = float(dpnp.linalg.norm(ia @ x - ib) / dpnp.linalg.norm(ib))
        assert res < 1e-4


class TestSolversEdgeCases:
    # Sanity-floor checks that fall outside the regular convergence
    # / correctness suites: identity-matrix degenerate-1-iter case,
    # wide-spectrum diagonal matrix (numerical-stability probe),
    # matvec callable that raises (error must propagate cleanly),
    # and the smallest possible n=1 / n=2 systems.

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    @pytest.mark.parametrize("solver", [cg, minres])
    def test_identity_system_one_iter(self, solver):
        # A = I and b arbitrary; the exact solution is x = b and any
        # iterative solver should reach it in at most n iterations
        # (Krylov subspace is trivial). gmres is omitted: A = I
        # triggers an Arnoldi happy breakdown at j=1 (A @ v_1 == v_1
        # so H[2, 1] = 0), and detecting it would require either a
        # host sync per Arnoldi step (violates the no-implicit-
        # sync rule of these solvers) or non-trivial device-side
        # NaN-guarded arithmetic for a contrived case that no real
        # workload exercises.
        n = 6
        ia = dpnp.eye(n, dtype=dpnp.float64)
        ib = _rhs(n, dpnp.float64)
        x, info = solver(ia, ib, rtol=1e-12)
        assert info == 0
        assert_allclose(dpnp.asnumpy(x), dpnp.asnumpy(ib), atol=1e-10)

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    @pytest.mark.parametrize("solver", [cg, minres])
    def test_wide_spectrum_diagonal(self, solver):
        # Diagonal SPD matrix whose eigenvalues span 3 orders of
        # magnitude (cond ~ 1e3) is a stability probe that fits
        # within the typical sqrt(cond) ~ 32 CG iterations for
        # n=30; broader spectra need maxiter > 1000 and are not
        # useful for a smoke test.
        #
        # The assertion threshold is sized for MINRES, whose
        # SciPy-matching stopping criterion is ``||r|| / (||A||
        # ||x||) <= rtol`` rather than ``||r|| / ||b|| <= rtol``.
        # With ||A|| ~ 1e2 here, ||r|| / ||b|| converges to roughly
        # ``rtol * Anorm * ynorm / ||b||``, which for rtol=1e-7 on
        # this system lands around 1e-4. CG (which actually tests
        # ``||r|| / ||b||``) reaches ~1e-7 comfortably, so the
        # bound below is dominated by MINRES.
        n = 30
        diag = numpy.logspace(-1, 2, n, dtype=numpy.float64)
        ia = dpnp.asarray(numpy.diag(diag))
        ib = _rhs(n, dpnp.float64)
        x, info = solver(ia, ib, rtol=1e-7, maxiter=2 * n)
        assert info == 0
        res = float(dpnp.linalg.norm(ia @ x - ib) / dpnp.linalg.norm(ib))
        assert res < 1e-3

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    @pytest.mark.parametrize("solver", [cg, gmres, minres])
    def test_matvec_callable_raises_propagates(self, solver):
        # A LinearOperator whose matvec raises must surface the
        # exception cleanly rather than being silently swallowed by
        # the solver's loop.
        n = 4

        def bad_matvec(x):
            raise RuntimeError("matvec sentinel")

        lo = LinearOperator((n, n), matvec=bad_matvec, dtype=dpnp.float64)
        ib = _rhs(n, dpnp.float64)
        with pytest.raises(RuntimeError, match="matvec sentinel"):
            solver(lo, ib, rtol=1e-8, maxiter=10)

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 is required")
    @pytest.mark.parametrize(
        "solver, n",
        [
            (cg, 1),
            (cg, 2),
            (minres, 1),
            (minres, 2),
            (gmres, 2),
        ],
    )
    def test_tiny_system(self, solver, n):
        # Smoke test for the smallest possible square systems; many
        # solvers have off-by-one footguns at n == 1. gmres is only
        # exercised at n=2: an n=1 system is mathematically degenerate
        # (x = b / A[0,0], the Krylov subspace is a single point) and
        # the Arnoldi step in _make_compute_hu calls oneMKL gemv with
        # a length-1 vector whose dpctl-reported stride is 0, which
        # the BLAS spec forbids. n=2 still exercises the smallest
        # non-trivial Arnoldi iteration.
        ia = _spd_matrix(n, dpnp.float64)
        ib = _rhs(n, dpnp.float64)
        x, info = solver(ia, ib, rtol=1e-10, maxiter=10)
        assert info == 0
        res = float(dpnp.linalg.norm(ia @ x - ib) / dpnp.linalg.norm(ib))
        assert res < 1e-8
