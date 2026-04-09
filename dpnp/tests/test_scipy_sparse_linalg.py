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
from dpnp.tests.helper import (
    assert_dtype_allclose,
    generate_random_numpy_array,
    get_all_dtypes,
    get_float_complex_dtypes,
    has_support_aspect64,
    is_scipy_available,
)
from dpnp.tests.third_party.cupy import testing

if is_scipy_available():
    import scipy.sparse.linalg as scipy_sla


# Helpers for constructing SPD, diagonally dominant, and symmetric
# indefinite test matrices. Kept small and local, matching the style of
# vvsort() at the top of test_linalg.py.
def _spd_matrix(n, dtype):
    rng = numpy.random.default_rng(42)
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
    a = rng.standard_normal((n, n))
    q, _ = numpy.linalg.qr(a)
    d = rng.standard_normal(n)
    m = (q @ numpy.diag(d) @ q.T).astype(dtype)
    return dpnp.asarray(m)


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


# GMRES in dpnp.scipy.sparse.linalg._iterative uses real-valued Givens
# rotation formulas which are incorrect for complex Arnoldi, so GMRES
# returns wrong solutions for complex dtypes. Complex GMRES tests are
# xfailed below. When the Givens block is fixed the xfails will flip to
# XPASS and force an update here.
_GMRES_CPX_XFAIL = (
    "GMRES Givens rotation is real-valued; broken for complex dtypes"
)

_GMRES_DTYPES = [
    dpnp.float32,
    dpnp.float64,
    pytest.param(
        dpnp.complex64,
        marks=pytest.mark.xfail(reason=_GMRES_CPX_XFAIL, strict=False),
    ),
    pytest.param(
        dpnp.complex128,
        marks=pytest.mark.xfail(reason=_GMRES_CPX_XFAIL, strict=False),
    ),
]


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

    def test_dtype_inference_float64_default(self):
        # Dtype inference probes matvec with a float64 vector, so the
        # inferred dtype is float64 even when the underlying array is
        # float32. Pin the current behaviour as a regression guard.
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
        n = 4
        a = dpnp.eye(n, dtype=dpnp.float32)
        lo = LinearOperator((n, n), matvec=lambda x: a @ x)
        assert lo.dtype == dpnp.float64

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

    def test_call_alias(self):
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
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

    @pytest.mark.parametrize("dtype", [dpnp.float32, dpnp.float64])
    def test_subclass_custom_matmat(self, dtype):
        if not has_support_aspect64() and dtype == dpnp.float64:
            pytest.skip("float64 not supported on this device")
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
            (ValueError, Exception),
            LinearOperator,
            (-1, 3),
            matvec=lambda x: x,
            dtype=dpnp.float32,
        )

        # shape with wrong ndim
        assert_raises(
            (ValueError, Exception),
            LinearOperator,
            (3,),
            matvec=lambda x: x,
            dtype=dpnp.float32,
        )


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

    def test_dense_numpy_array_attributes_only(self):
        # aslinearoperator(numpy_array) wraps with lambda x: A @ x where A
        # remains a numpy array; calling matvec(dpnp_x) then fails because
        # dpnp __rmatmul__ refuses numpy LHS. Only attributes are checked.
        n = 5
        a = generate_random_numpy_array((n, n), numpy.float64, seed_value=42)
        lo = aslinearoperator(a)
        assert lo.shape == (n, n)

    def test_rmatvec_from_dpnp_dense(self):
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
        n = 5
        a = generate_random_numpy_array((n, n), numpy.float64, seed_value=42)
        ia = dpnp.array(a)
        lo = aslinearoperator(ia)
        x = generate_random_numpy_array((n,), numpy.float64, seed_value=2)
        ix = dpnp.array(x)
        result = lo.rmatvec(ix)
        expected = a.conj().T @ x
        assert_allclose(dpnp.asnumpy(result), expected, atol=1e-12)

    def test_duck_type_with_shape_and_matvec(self):
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
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
        assert_raises((TypeError, Exception), aslinearoperator, "nope")


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

    @pytest.mark.skipif(not is_scipy_available(), reason="SciPy not available")
    @pytest.mark.parametrize("dtype", [dpnp.float32, dpnp.float64])
    def test_cg_matches_scipy(self, dtype):
        if not has_support_aspect64() and dtype == dpnp.float64:
            pytest.skip("float64 not supported on this device")
        a = dpnp.asnumpy(_spd_matrix(self.n, dtype))
        b = dpnp.asnumpy(_rhs(self.n, dtype))
        try:
            x_ref, info_ref = scipy_sla.cg(a, b, rtol=1e-8, maxiter=500)
        except TypeError:  # scipy < 1.12
            x_ref, info_ref = scipy_sla.cg(a, b, tol=1e-8, maxiter=500)
        assert info_ref == 0
        x_dp, info = cg(dpnp.array(a), dpnp.array(b), rtol=1e-8, maxiter=500)
        assert info == 0
        tol = 1e-4 if dtype == dpnp.float32 else 1e-8
        assert_allclose(dpnp.asnumpy(x_dp), x_ref, rtol=tol, atol=tol)

    @pytest.mark.parametrize("dtype", [dpnp.float32, dpnp.float64])
    def test_cg_x0_warm_start(self, dtype):
        if not has_support_aspect64() and dtype == dpnp.float64:
            pytest.skip("float64 not supported on this device")
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
            pytest.skip("float64 not supported on this device")
        ia = _spd_matrix(self.n, dtype)
        ib = _rhs(self.n, dtype).reshape(self.n, 1)
        _, info = cg(ia, ib, rtol=1e-8, maxiter=500)
        assert info == 0

    def test_cg_b_zero(self):
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
        ia = _spd_matrix(10, dpnp.float64)
        ib = dpnp.zeros(10, dtype=dpnp.float64)
        x, info = cg(ia, ib, rtol=1e-8)
        assert info == 0
        assert_allclose(dpnp.asnumpy(x), numpy.zeros(10), atol=1e-14)

    def test_cg_callback(self):
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
        ia = _spd_matrix(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64)
        calls = []
        cg(
            ia,
            ib,
            callback=lambda xk: calls.append(float(dpnp.linalg.norm(xk))),
            rtol=1e-10,
            maxiter=200,
        )
        assert len(calls) > 0

    def test_cg_atol(self):
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
        ia = _spd_matrix(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64)
        x, _ = cg(ia, ib, rtol=0.0, atol=1e-1, maxiter=500)
        assert float(dpnp.linalg.norm(ia @ x - ib)) < 1.0

    def test_cg_exact_solution(self):
        # x0 == true solution must return info == 0 immediately
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
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

    def test_cg_maxiter_nonconvergence(self):
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
        ia = _spd_matrix(50, dpnp.float64)
        ib = _rhs(50, dpnp.float64)
        _, info = cg(ia, ib, rtol=1e-15, atol=0.0, maxiter=1)
        assert info != 0

    def test_cg_diag_preconditioner(self):
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
        ia = _spd_matrix(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64)
        M = aslinearoperator(dpnp.diag(1.0 / dpnp.diag(ia)))
        _, info = cg(ia, ib, M=M, rtol=1e-8, maxiter=500)
        assert info == 0

    def test_cg_errors(self):
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
        ia = _spd_matrix(5, dpnp.float64)
        ib = dpnp.ones(6, dtype=dpnp.float64)
        # b length mismatch
        with pytest.raises((ValueError, Exception)):
            cg(ia, ib, maxiter=1)


class TestGmres:
    n = 30

    @pytest.mark.parametrize("dtype", _GMRES_DTYPES)
    def test_gmres_converges_diag_dominant(self, dtype):
        if not has_support_aspect64() and dtype in (
            dpnp.float64,
            dpnp.complex128,
        ):
            pytest.skip("float64 not supported on this device")
        ia = _diag_dominant(self.n, dtype)
        ib = _rhs(self.n, dtype)
        x, _ = gmres(
            ia,
            ib,
            rtol=_rtol_for(dtype),
            maxiter=200,
            restart=self.n,
        )
        # Check actual residual rather than info: see comment above
        # _GMRES_CPX_XFAIL.
        res = float(dpnp.linalg.norm(ia @ x - ib) / dpnp.linalg.norm(ib))
        assert res < _res_bound(dtype)

    @pytest.mark.skipif(not is_scipy_available(), reason="SciPy not available")
    @pytest.mark.parametrize("dtype", [dpnp.float32, dpnp.float64])
    def test_gmres_matches_scipy(self, dtype):
        if not has_support_aspect64() and dtype == dpnp.float64:
            pytest.skip("float64 not supported on this device")
        a = dpnp.asnumpy(_diag_dominant(self.n, dtype))
        b = dpnp.asnumpy(_rhs(self.n, dtype))
        req_rtol = _rtol_for(dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                x_ref, _ = scipy_sla.gmres(
                    a, b, rtol=req_rtol, restart=self.n, maxiter=None
                )
            except TypeError:  # scipy < 1.12
                x_ref, _ = scipy_sla.gmres(
                    a, b, tol=req_rtol, restart=self.n, maxiter=None
                )
        x_dp, info = gmres(
            dpnp.array(a),
            dpnp.array(b),
            rtol=req_rtol,
            restart=self.n,
            maxiter=50,
        )
        assert info == 0
        tol = 1e-3 if dtype == dpnp.float32 else 1e-7
        assert_allclose(dpnp.asnumpy(x_dp), x_ref, rtol=tol, atol=tol)

    @pytest.mark.parametrize("restart", [None, 5, 15], ids=["None", "5", "15"])
    def test_gmres_restart_values(self, restart):
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
        ia = _diag_dominant(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64)
        _, info = gmres(ia, ib, rtol=1e-8, restart=restart, maxiter=100)
        assert info == 0

    @pytest.mark.parametrize("dtype", [dpnp.float32, dpnp.float64])
    def test_gmres_x0_warm_start(self, dtype):
        if not has_support_aspect64() and dtype == dpnp.float64:
            pytest.skip("float64 not supported on this device")
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

    def test_gmres_b_2dim(self):
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
        ia = _diag_dominant(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64).reshape(self.n, 1)
        _, info = gmres(ia, ib, rtol=1e-8, restart=self.n, maxiter=100)
        assert info == 0

    def test_gmres_b_zero(self):
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
        ia = _diag_dominant(10, dpnp.float64)
        ib = dpnp.zeros(10, dtype=dpnp.float64)
        x, info = gmres(ia, ib, rtol=1e-8)
        assert info == 0
        assert_allclose(dpnp.asnumpy(x), numpy.zeros(10), atol=1e-14)

    def test_gmres_callback_x(self):
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
        ia = _diag_dominant(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64)
        calls = []
        gmres(
            ia,
            ib,
            callback=lambda xk: calls.append(1),
            callback_type="x",
            rtol=1e-10,
            maxiter=20,
            restart=self.n,
        )
        assert len(calls) > 0

    def test_gmres_callback_pr_norm(self):
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
        ia = _diag_dominant(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64)
        values = []
        gmres(
            ia,
            ib,
            callback=lambda r: values.append(float(r)),
            callback_type="pr_norm",
            rtol=1e-10,
            maxiter=20,
            restart=self.n,
        )
        assert len(values) > 0
        assert all(v >= 0 for v in values)

    def test_gmres_atol(self):
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
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

    @pytest.mark.parametrize("dtype", _GMRES_DTYPES)
    def test_gmres_via_linear_operator(self, dtype):
        if not has_support_aspect64() and dtype in (
            dpnp.float64,
            dpnp.complex128,
        ):
            pytest.skip("float64 not supported on this device")
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

    def test_gmres_nonconvergence(self):
        # Ill-conditioned Hilbert matrix + tiny restart must not converge
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
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

    @pytest.mark.xfail(reason=_GMRES_CPX_XFAIL, strict=False)
    def test_gmres_complex_system(self):
        if not has_support_aspect64():
            pytest.skip("complex128 not supported on this device")
        n = 15
        ia = _diag_dominant(n, dpnp.complex128)
        ib = _rhs(n, dpnp.complex128)
        x, _ = gmres(ia, ib, rtol=1e-8, restart=n, maxiter=200)
        res = float(dpnp.linalg.norm(ia @ x - ib) / dpnp.linalg.norm(ib))
        assert res < 1e-5

    def test_gmres_errors(self):
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
        ia = _diag_dominant(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64)
        # unknown callback_type
        assert_raises(ValueError, gmres, ia, ib, callback_type="garbage")


class TestMinres:
    n = 30

    @pytest.mark.parametrize("dtype", [dpnp.float32, dpnp.float64])
    def test_minres_converges_spd(self, dtype):
        if not has_support_aspect64() and dtype == dpnp.float64:
            pytest.skip("float64 not supported on this device")
        ia = _spd_matrix(self.n, dtype)
        ib = _rhs(self.n, dtype)
        x, info = minres(ia, ib, rtol=1e-8, maxiter=500)
        assert info == 0
        res = float(dpnp.linalg.norm(ia @ x - ib) / dpnp.linalg.norm(ib))
        assert res < 1e-4

    def test_minres_converges_sym_indefinite(self):
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
        ia = _sym_indefinite(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64)
        x, _ = minres(ia, ib, rtol=1e-8, maxiter=1000)
        res = float(dpnp.linalg.norm(ia @ x - ib) / dpnp.linalg.norm(ib))
        assert res < 1e-3

    @pytest.mark.skipif(not is_scipy_available(), reason="SciPy not available")
    def test_minres_matches_scipy(self):
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
        a = dpnp.asnumpy(_spd_matrix(self.n, dpnp.float64))
        b = dpnp.asnumpy(_rhs(self.n, dpnp.float64))
        try:
            x_ref, _ = scipy_sla.minres(a, b, rtol=1e-8)
        except TypeError:
            x_ref, _ = scipy_sla.minres(a, b, tol=1e-8)
        x_dp, info = minres(dpnp.array(a), dpnp.array(b), rtol=1e-8)
        assert info == 0
        assert_allclose(dpnp.asnumpy(x_dp), x_ref, rtol=1e-5, atol=1e-6)

    def test_minres_x0_warm_start(self):
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
        ia = _spd_matrix(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64)
        x0 = dpnp.zeros(self.n, dtype=dpnp.float64)
        _, info = minres(ia, ib, x0=x0, rtol=1e-8)
        assert info == 0

    def test_minres_shift(self):
        # shift != 0 solves (A - shift*I) x = b
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
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

    def test_minres_b_zero(self):
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
        ia = _spd_matrix(10, dpnp.float64)
        ib = dpnp.zeros(10, dtype=dpnp.float64)
        x, info = minres(ia, ib, rtol=1e-8)
        assert info == 0
        assert_allclose(dpnp.asnumpy(x), numpy.zeros(10), atol=1e-14)

    def test_minres_via_linear_operator(self):
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
        ia = _spd_matrix(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64)
        lo = aslinearoperator(ia)
        _, info = minres(lo, ib, rtol=1e-8)
        assert info == 0

    def test_minres_callback(self):
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
        ia = _spd_matrix(self.n, dpnp.float64)
        ib = _rhs(self.n, dpnp.float64)
        calls = []
        minres(
            ia,
            ib,
            callback=lambda xk: calls.append(1),
            rtol=1e-10,
        )
        assert len(calls) > 0

    def test_minres_errors(self):
        if not has_support_aspect64():
            pytest.skip("float64 not supported on this device")
        lo = aslinearoperator(dpnp.ones((4, 5), dtype=dpnp.float64))
        ib = dpnp.ones(4, dtype=dpnp.float64)
        # non-square operator
        assert_raises((ValueError, Exception), minres, lo, ib)


class TestSolversIntegration:
    @pytest.mark.parametrize(
        "n, dtype",
        [
            (10, dpnp.float32),
            (10, dpnp.float64),
            (30, dpnp.float64),
            (50, dpnp.float64),
        ],
    )
    def test_cg_spd_via_linearoperator(self, n, dtype):
        if not has_support_aspect64() and dtype == dpnp.float64:
            pytest.skip("float64 not supported on this device")
        ia = _spd_matrix(n, dtype)
        lo = aslinearoperator(ia)
        ib = _rhs(n, dtype)
        x, info = cg(lo, ib, rtol=_rtol_for(dtype), maxiter=n * 10)
        assert info == 0
        res = float(dpnp.linalg.norm(ia @ x - ib) / dpnp.linalg.norm(ib))
        assert res < _res_bound(dtype)

    @pytest.mark.parametrize(
        "n, dtype",
        [
            (10, dpnp.float32),
            (10, dpnp.float64),
            (30, dpnp.float64),
        ],
    )
    def test_gmres_nonsymmetric_via_linearoperator(self, n, dtype):
        if not has_support_aspect64() and dtype == dpnp.float64:
            pytest.skip("float64 not supported on this device")
        ia = _diag_dominant(n, dtype)
        lo = aslinearoperator(ia)
        ib = _rhs(n, dtype)
        x, _ = gmres(lo, ib, rtol=_rtol_for(dtype), restart=n, maxiter=200)
        res = float(dpnp.linalg.norm(ia @ x - ib) / dpnp.linalg.norm(ib))
        assert res < _res_bound(dtype)

    @pytest.mark.skipif(
        not is_scipy_available(), reason="SciPy required for minres"
    )
    @pytest.mark.parametrize(
        "n, dtype",
        [
            (10, dpnp.float64),
            (30, dpnp.float64),
        ],
    )
    def test_minres_spd_via_linearoperator(self, n, dtype):
        if not has_support_aspect64() and dtype == dpnp.float64:
            pytest.skip("float64 not supported on this device")
        ia = _spd_matrix(n, dtype)
        lo = aslinearoperator(ia)
        ib = _rhs(n, dtype)
        x, info = minres(lo, ib, rtol=1e-8)
        assert info == 0
        res = float(dpnp.linalg.norm(ia @ x - ib) / dpnp.linalg.norm(ib))
        assert res < 1e-4
