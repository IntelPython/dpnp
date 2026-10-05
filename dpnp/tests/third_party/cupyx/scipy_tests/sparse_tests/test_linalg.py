from __future__ import annotations

import unittest

import numpy
import pytest

import dpnp as cupy
from dpnp.tests.helper import has_support_aspect64
from dpnp.tests.third_party.cupy import testing

if cupy.tests.helper.is_scipy_available():
    import scipy.sparse
    import scipy.sparse.linalg


def _inner_cases(sp, A, inner_modification):
    # Mirror of upstream's _inner_cases, with the 'sparse' branch
    # pinned to csr (the only format dpnp implements).
    def mv(x):
        return A.dot(x)

    def rmv(x):
        return A.T.conj().dot(x)

    linop_cls = sp.linalg.LinearOperator

    class BaseMatlike(linop_cls):
        def __init__(self):
            super().__init__(dtype=A.dtype, shape=A.shape)

        def _adjoint(self):
            shape = self.shape[1], self.shape[0]
            return linop_cls(
                matvec=rmv, rmatvec=mv, dtype=self.dtype, shape=shape
            )

    class HasMatvec(BaseMatlike):
        def _matvec(self, x):
            return mv(x)

    class HasMatmat(BaseMatlike):
        def _matmat(self, x):
            return mv(x)

    if inner_modification == "normal":
        return sp.linalg.aslinearoperator(A)
    if inner_modification == "sparse":
        return sp.linalg.aslinearoperator(sp.csr_matrix(A))
    if inner_modification == "linear_operator":
        return linop_cls(matvec=mv, rmatvec=rmv, dtype=A.dtype, shape=A.shape)
    if inner_modification == "class_matvec":
        return HasMatvec()
    if inner_modification == "class_matmat":
        return HasMatmat()
    raise AssertionError(inner_modification)


def _generate_linear_operator(sp, A, outer_modification, inner_modification):
    # NOTE: transpose/hermitian wrap the inner operator built from the
    # transposed array (mirrors upstream), so the final shape stays
    # (M, N) on every path.
    if outer_modification == "normal":
        return _inner_cases(sp, A, inner_modification)
    if outer_modification == "transpose":
        return _inner_cases(sp, A.T, inner_modification).T
    if outer_modification == "hermitian":
        return _inner_cases(sp, A.T.conj(), inner_modification).H
    raise AssertionError(outer_modification)


@testing.parameterize(
    *testing.product(
        {
            "dtype": [
                numpy.float32,
                numpy.float64,
                numpy.complex64,
                numpy.complex128,
            ],
            "outer_modification": ["normal", "transpose", "hermitian"],
            "inner_modification": [
                "normal",
                "sparse",
                "linear_operator",
                "class_matvec",
                "class_matmat",
            ],
            "M": [1, 6],
            "N": [1, 7],
        }
    )
)
@testing.fix_random()
@testing.with_requires("scipy")
class TestLinearOperator(unittest.TestCase):
    def _needs_csr_adjoint_skip(self):
        # dpnp implements only the forward SpMV path for csr-backed
        # operators; any .T/.H or rmat* on them raises. See
        # test_sparse_rmatvec_not_implemented in the own-scope suite.
        return self.inner_modification == "sparse" and (
            self.outer_modification in ("transpose", "hermitian")
        )

    def _make_pair(self):
        # dtype comes from parameterize (not for_dtypes), so skip
        # float64/complex128 explicitly on fp64-less devices.
        if numpy.dtype(self.dtype).char in "dD" and not has_support_aspect64():
            self.skipTest("fp64 is required")
        A_cpu = testing.shaped_random((self.M, self.N), numpy, self.dtype)
        A_gpu = cupy.asarray(A_cpu)
        return A_cpu, A_gpu

    def _linops(self, A_cpu, A_gpu):
        numpy_linop = _generate_linear_operator(
            scipy.sparse,
            A_cpu,
            self.outer_modification,
            self.inner_modification,
        )
        cupy_linop = _generate_linear_operator(
            cupy.scipy.sparse,
            A_gpu,
            self.outer_modification,
            self.inner_modification,
        )
        return numpy_linop, cupy_linop

    def _vec_shapes(self):
        # NOTE: dpnp's csr-backed operator is 1-D-only (cupyx parity:
        # csr_matrix.dot rejects 2-D x), so column-vector inputs are
        # only exercised on non-sparse inners.
        if self.inner_modification == "sparse":
            return ((self.N,),)
        return ((self.N,), (self.N, 1))

    def _dot_shapes(self):
        shapes = [(self.N,), (self.N, 8)]
        if self.inner_modification != "sparse":
            shapes.insert(1, (self.N, 1))
        return shapes

    # The `(N, 1)` case below is deprecated in SciPy 1.18, an error in 1.20.
    # TODO: call `matmat` for it before allowing SciPy 1.20.
    @pytest.mark.filterwarnings(
        "ignore:Calling `matvec` on 'column vectors':FutureWarning"
    )
    def test_matvec(self):
        if self._needs_csr_adjoint_skip():
            self.skipTest("csr-backed adjoint unsupported by dpnp")
        A_cpu, A_gpu = self._make_pair()
        numpy_linop, cupy_linop = self._linops(A_cpu, A_gpu)
        for shape in self._vec_shapes():
            x_cpu = testing.shaped_random(shape, numpy, self.dtype)
            x_gpu = cupy.asarray(x_cpu)
            testing.assert_allclose(
                cupy.asnumpy(cupy_linop.matvec(x_gpu)),
                numpy_linop.matvec(x_cpu),
                rtol=1e-6,
            )

    def test_matmat(self):
        if self._needs_csr_adjoint_skip():
            self.skipTest("csr-backed adjoint unsupported by dpnp")
        A_cpu, A_gpu = self._make_pair()
        numpy_linop, cupy_linop = self._linops(A_cpu, A_gpu)
        x_cpu = testing.shaped_random((self.N, 8), numpy, self.dtype)
        x_gpu = cupy.asarray(x_cpu)
        testing.assert_allclose(
            cupy.asnumpy(cupy_linop.matmat(x_gpu)),
            numpy_linop.matmat(x_cpu),
            rtol=1e-6,
        )

    # The `(M, 1)` case below is deprecated in SciPy 1.18, an error in 1.20.
    # TODO: call `rmatmat` for it before allowing SciPy 1.20.
    @pytest.mark.filterwarnings(
        "ignore:Calling `rmatvec` on 'column vectors':FutureWarning"
    )
    def test_rmatvec(self):
        if self.inner_modification == "sparse":
            self.skipTest("csr-backed adjoint unsupported by dpnp")
        A_cpu, A_gpu = self._make_pair()
        numpy_linop, cupy_linop = self._linops(A_cpu, A_gpu)
        for shape in ((self.M,), (self.M, 1)):
            x_cpu = testing.shaped_random(shape, numpy, self.dtype)
            x_gpu = cupy.asarray(x_cpu)
            testing.assert_allclose(
                cupy.asnumpy(cupy_linop.rmatvec(x_gpu)),
                numpy_linop.rmatvec(x_cpu),
                rtol=1e-6,
            )

    def test_rmatmat(self):
        if self.inner_modification == "sparse":
            self.skipTest("csr-backed adjoint unsupported by dpnp")
        A_cpu, A_gpu = self._make_pair()
        numpy_linop, cupy_linop = self._linops(A_cpu, A_gpu)
        x_cpu = testing.shaped_random((self.M, 8), numpy, self.dtype)
        x_gpu = cupy.asarray(x_cpu)
        testing.assert_allclose(
            cupy.asnumpy(cupy_linop.rmatmat(x_gpu)),
            numpy_linop.rmatmat(x_cpu),
            rtol=1e-6,
        )

    def test_dot(self):
        if self._needs_csr_adjoint_skip():
            self.skipTest("csr-backed adjoint unsupported by dpnp")
        A_cpu, A_gpu = self._make_pair()
        numpy_linop, cupy_linop = self._linops(A_cpu, A_gpu)
        for shape in self._dot_shapes():
            x_cpu = testing.shaped_random(shape, numpy, self.dtype)
            x_gpu = cupy.asarray(x_cpu)
            testing.assert_allclose(
                cupy.asnumpy(cupy_linop.dot(x_gpu)),
                numpy_linop.dot(x_cpu),
                rtol=1e-6,
            )

    def test_mul(self):
        if self._needs_csr_adjoint_skip():
            self.skipTest("csr-backed adjoint unsupported by dpnp")
        A_cpu, A_gpu = self._make_pair()
        numpy_linop, cupy_linop = self._linops(A_cpu, A_gpu)
        for shape in self._dot_shapes():
            x_cpu = testing.shaped_random(shape, numpy, self.dtype)
            x_gpu = cupy.asarray(x_cpu)
            testing.assert_allclose(
                cupy.asnumpy(cupy_linop * x_gpu),
                numpy_linop * x_cpu,
                rtol=1e-6,
            )


@testing.parameterize(
    *testing.product(
        {
            "x0": [None, "ones"],
            "M": [None, "jacobi"],
            "atol": [None, "select-by-dtype"],
            "b_ndim": [1, 2],
            "use_linear_operator": [False, True],
        }
    )
)
@testing.fix_random()
@testing.with_requires("scipy")
class TestCg(unittest.TestCase):
    n = 30
    density = 0.33
    _atol = {"f": 1e-5, "d": 1e-12}

    def _is_base_config(self):
        return (
            self.x0 is None
            and self.M is None
            and self.atol is None
            and self.use_linear_operator is False
        )

    def _make_matrix(self, dtype):
        dtype = numpy.dtype(dtype)
        shape = (self.n, 10)
        a = testing.shaped_random(
            shape, numpy, dtype=dtype.char.lower(), scale=1
        )
        if dtype.char in "FD":
            a = a + 1j * testing.shaped_random(
                shape, numpy, dtype=dtype.char.lower(), scale=1
            )
        mask = testing.shaped_random(shape, numpy, dtype="f", scale=1)
        a[mask > self.density] = 0
        a = a @ a.conj().T
        a = a + numpy.diag(numpy.ones((self.n,), dtype=dtype.char.lower()))
        m = None
        if self.M == "jacobi":
            m = numpy.diag(1.0 / numpy.diag(a))
        return a, m

    def _make_normalized_vector(self, dtype):
        b = testing.shaped_random((self.n,), numpy, dtype=dtype)
        return b / numpy.linalg.norm(b)

    def _resolve_atol(self, dtype):
        if self.atol == "select-by-dtype":
            return self._atol[numpy.dtype(dtype).char.lower()]
        return 0.0

    def _run_both(self, a_ref, b_cpu, x0_cpu, m_ref, a_gpu, m_gpu, atol):
        b_gpu = cupy.asarray(b_cpu)
        x0_gpu = None if x0_cpu is None else cupy.asarray(x0_cpu)
        if self.use_linear_operator:
            a_ref = scipy.sparse.linalg.aslinearoperator(a_ref)
            a_gpu = cupy.scipy.sparse.linalg.aslinearoperator(a_gpu)
            if m_gpu is not None:
                m_ref = scipy.sparse.linalg.aslinearoperator(m_ref)
                m_gpu = cupy.scipy.sparse.linalg.aslinearoperator(m_gpu)
        x_ref, info_ref = scipy.sparse.linalg.cg(
            a_ref, b_cpu, x0_cpu, M=m_ref, atol=atol
        )
        assert info_ref == 0
        x_dp, info_dp = cupy.scipy.sparse.linalg.cg(
            a_gpu, b_gpu, x0=x0_gpu, M=m_gpu, atol=atol
        )
        assert info_dp == 0
        testing.assert_allclose(cupy.asnumpy(x_dp), x_ref, rtol=1e-5, atol=1e-5)

    def _prep(self, dtype):
        a_cpu, m_cpu = self._make_matrix(dtype)
        b_cpu = self._make_normalized_vector(dtype)
        if self.b_ndim == 2:
            b_cpu = b_cpu.reshape(self.n, 1)
        x0_cpu = None
        if self.x0 == "ones":
            x0_cpu = numpy.ones((self.n,), dtype=dtype)
        return a_cpu, m_cpu, b_cpu, x0_cpu, self._resolve_atol(dtype)

    @testing.for_dtypes("fdFD")
    def test_dense(self, dtype):
        a_cpu, m_cpu, b_cpu, x0_cpu, atol = self._prep(dtype)
        a_gpu = cupy.asarray(a_cpu)
        m_gpu = None if m_cpu is None else cupy.asarray(m_cpu)
        self._run_both(a_cpu, b_cpu, x0_cpu, m_cpu, a_gpu, m_gpu, atol)

    @testing.for_dtypes("fdFD")
    def test_sparse(self, dtype):
        a_cpu, m_cpu, b_cpu, x0_cpu, atol = self._prep(dtype)
        a_ref = scipy.sparse.csr_matrix(a_cpu)
        m_ref = None if m_cpu is None else scipy.sparse.csr_matrix(m_cpu)
        a_gpu = cupy.scipy.sparse.csr_matrix(cupy.asarray(a_cpu))
        m_gpu = (
            None
            if m_cpu is None
            else cupy.scipy.sparse.csr_matrix(cupy.asarray(m_cpu))
        )
        self._run_both(a_ref, b_cpu, x0_cpu, m_ref, a_gpu, m_gpu, atol)

    @testing.with_requires("scipy")
    @testing.for_dtypes("fdFD")
    def test_empty(self, dtype):
        if not self._is_base_config():
            self.skipTest("base config only")
        a_cpu = numpy.empty((0, 0), dtype=dtype)
        b_cpu = numpy.empty((0,), dtype=dtype)
        x_ref, info_ref = scipy.sparse.linalg.cg(a_cpu, b_cpu)
        assert info_ref == 0
        x_dp, info_dp = cupy.scipy.sparse.linalg.cg(
            cupy.asarray(a_cpu), cupy.asarray(b_cpu)
        )
        assert info_dp == 0
        testing.assert_allclose(cupy.asnumpy(x_dp), x_ref)

    @testing.for_dtypes("fdFD")
    def test_callback(self, dtype):
        if not self._is_base_config():
            self.skipTest("base config only")
        a_cpu, _ = self._make_matrix(dtype)
        b_cpu = self._make_normalized_vector(dtype)
        a_gpu = cupy.asarray(a_cpu)
        b_gpu = cupy.asarray(b_cpu)
        is_called = False

        def callback(x):
            nonlocal is_called
            is_called = True

        cupy.scipy.sparse.linalg.cg(a_gpu, b_gpu, callback=callback)
        assert is_called

    def test_invalid(self):
        if not self._is_base_config():
            self.skipTest("base config only")
        for xp, sp in ((numpy, scipy.sparse), (cupy, cupy.scipy.sparse)):
            a = testing.shaped_random((self.n, self.n), numpy, dtype="f")
            b = testing.shaped_random((self.n,), numpy, dtype="f")
            if xp is not numpy:
                a = xp.asarray(a)
                b = xp.asarray(b)
            ng_a = xp.ones((self.n,), dtype="f")
            with pytest.raises(ValueError):
                sp.linalg.cg(ng_a, b, atol=self.atol)
            ng_a = xp.ones((self.n, self.n + 1), dtype="f")
            with pytest.raises(ValueError):
                sp.linalg.cg(ng_a, b, atol=self.atol)
            ng_a = xp.ones((self.n, self.n, 1), dtype="f")
            with pytest.raises(ValueError):
                sp.linalg.cg(ng_a, b, atol=self.atol)
            ng_b = xp.ones((self.n + 1,), dtype="f")
            with pytest.raises(ValueError):
                sp.linalg.cg(a, ng_b, atol=self.atol)
            ng_b = xp.ones((self.n, 2), dtype="f")
            with pytest.raises(ValueError):
                sp.linalg.cg(a, ng_b, atol=self.atol)
            ng_x0 = xp.ones((self.n + 1,), dtype="f")
            with pytest.raises(ValueError):
                sp.linalg.cg(a, b, ng_x0, atol=self.atol)
            ng_M = xp.diag(xp.ones((self.n + 1,), dtype="f"))
            with pytest.raises(ValueError):
                sp.linalg.cg(a, b, None, M=ng_M, atol=self.atol)
        # NOTE: no int-dtype TypeError check (unlike upstream): dpnp
        # deliberately promotes an int A against a float b instead of
        # rejecting it (see _make_system in _iterative.py).


@testing.parameterize(
    *testing.product(
        {
            "x0": [None, "ones"],
            "M": [None, "jacobi"],
            "atol": [None, "select-by-dtype"],
            "b_ndim": [1, 2],
            "restart": [None, 10],
            "use_linear_operator": [False, True],
        }
    )
)
@testing.fix_random()
@testing.with_requires("scipy")
class TestGmres(unittest.TestCase):
    n = 30
    density = 0.2
    _atol = {"f": 1e-5, "d": 1e-12}

    def _is_base_config(self):
        return (
            self.x0 is None
            and self.M is None
            and self.atol is None
            and self.restart is None
            and self.use_linear_operator is False
        )

    def _make_matrix(self, dtype):
        dtype = numpy.dtype(dtype)
        shape = (self.n, self.n)
        a = testing.shaped_random(shape, numpy, dtype=dtype, scale=1)
        mask = testing.shaped_random(shape, numpy, dtype="f", scale=1)
        a[mask > self.density] = 0
        diag = numpy.diag(
            testing.shaped_random(
                (self.n,), numpy, dtype=dtype.char.lower(), scale=1
            )
            + 1
        )
        a[diag > 0] = 0
        a = a + diag
        m = None
        if self.M == "jacobi":
            m = numpy.diag(1.0 / numpy.diag(a))
        return a, m

    def _make_normalized_vector(self, dtype):
        b = testing.shaped_random((self.n,), numpy, dtype=dtype, scale=1)
        return b / numpy.linalg.norm(b)

    def _resolve_atol(self, dtype):
        if self.atol == "select-by-dtype":
            return self._atol[numpy.dtype(dtype).char.lower()]
        return 0.0

    def _run_both(self, a_ref, b_cpu, x0_cpu, m_ref, a_gpu, m_gpu, atol):
        b_gpu = cupy.asarray(b_cpu)
        x0_gpu = None if x0_cpu is None else cupy.asarray(x0_cpu)
        if self.use_linear_operator:
            a_ref = scipy.sparse.linalg.aslinearoperator(a_ref)
            a_gpu = cupy.scipy.sparse.linalg.aslinearoperator(a_gpu)
            if m_gpu is not None:
                m_ref = scipy.sparse.linalg.aslinearoperator(m_ref)
                m_gpu = cupy.scipy.sparse.linalg.aslinearoperator(m_gpu)
        x_ref, info_ref = scipy.sparse.linalg.gmres(
            a_ref, b_cpu, x0=x0_cpu, restart=self.restart, M=m_ref, atol=atol
        )
        assert info_ref == 0
        x_dp, info_dp = cupy.scipy.sparse.linalg.gmres(
            a_gpu, b_gpu, x0=x0_gpu, restart=self.restart, M=m_gpu, atol=atol
        )
        assert info_dp == 0
        testing.assert_allclose(cupy.asnumpy(x_dp), x_ref, rtol=1e-5, atol=1e-5)

    def _prep(self, dtype):
        a_cpu, m_cpu = self._make_matrix(dtype)
        b_cpu = self._make_normalized_vector(dtype)
        if self.b_ndim == 2:
            b_cpu = b_cpu.reshape(self.n, 1)
        x0_cpu = None
        if self.x0 == "ones":
            x0_cpu = numpy.ones((self.n,), dtype=dtype)
        return a_cpu, m_cpu, b_cpu, x0_cpu, self._resolve_atol(dtype)

    @testing.for_dtypes("fdFD")
    def test_dense(self, dtype):
        a_cpu, m_cpu, b_cpu, x0_cpu, atol = self._prep(dtype)
        a_gpu = cupy.asarray(a_cpu)
        m_gpu = None if m_cpu is None else cupy.asarray(m_cpu)
        self._run_both(a_cpu, b_cpu, x0_cpu, m_cpu, a_gpu, m_gpu, atol)

    @testing.for_dtypes("fdFD")
    def test_sparse(self, dtype):
        a_cpu, m_cpu, b_cpu, x0_cpu, atol = self._prep(dtype)
        a_ref = scipy.sparse.csr_matrix(a_cpu)
        m_ref = None if m_cpu is None else scipy.sparse.csr_matrix(m_cpu)
        a_gpu = cupy.scipy.sparse.csr_matrix(cupy.asarray(a_cpu))
        m_gpu = (
            None
            if m_cpu is None
            else cupy.scipy.sparse.csr_matrix(cupy.asarray(m_cpu))
        )
        self._run_both(a_ref, b_cpu, x0_cpu, m_ref, a_gpu, m_gpu, atol)

    @testing.with_requires("scipy")
    @testing.for_dtypes("fdFD")
    def test_empty(self, dtype):
        if not self._is_base_config():
            self.skipTest("base config only")
        a_cpu = numpy.empty((0, 0), dtype=dtype)
        b_cpu = numpy.empty((0,), dtype=dtype)
        x_ref, info_ref = scipy.sparse.linalg.gmres(a_cpu, b_cpu)
        assert info_ref == 0
        x_dp, info_dp = cupy.scipy.sparse.linalg.gmres(
            cupy.asarray(a_cpu), cupy.asarray(b_cpu)
        )
        assert info_dp == 0
        testing.assert_allclose(cupy.asnumpy(x_dp), x_ref)

    @testing.for_dtypes("fdFD")
    def test_callback(self, dtype):
        if not self._is_base_config():
            self.skipTest("base config only")
        a_cpu, _ = self._make_matrix(dtype)
        b_cpu = self._make_normalized_vector(dtype)
        a_gpu = cupy.asarray(a_cpu)
        b_gpu = cupy.asarray(b_cpu)
        is_called = False

        def callback1(x):
            nonlocal is_called
            is_called = True

        cupy.scipy.sparse.linalg.gmres(
            a_gpu, b_gpu, callback=callback1, callback_type="x"
        )
        assert is_called
        is_called = False

        def callback2(pr_norm):
            nonlocal is_called
            is_called = True

        cupy.scipy.sparse.linalg.gmres(
            a_gpu, b_gpu, callback=callback2, callback_type="pr_norm"
        )
        assert is_called

    def test_invalid(self):
        if not self._is_base_config():
            self.skipTest("base config only")
        for xp, sp in ((numpy, scipy.sparse), (cupy, cupy.scipy.sparse)):
            a = testing.shaped_random((self.n, self.n), numpy, dtype="f")
            b = testing.shaped_random((self.n,), numpy, dtype="f")
            if xp is not numpy:
                a = xp.asarray(a)
                b = xp.asarray(b)
            ng_a = xp.ones((self.n,), dtype="f")
            with pytest.raises(ValueError):
                sp.linalg.gmres(ng_a, b)
            ng_a = xp.ones((self.n, self.n + 1), dtype="f")
            with pytest.raises(ValueError):
                sp.linalg.gmres(ng_a, b)
            ng_a = xp.ones((self.n, self.n, 1), dtype="f")
            with pytest.raises(ValueError):
                sp.linalg.gmres(ng_a, b)
            ng_b = xp.ones((self.n + 1,), dtype="f")
            with pytest.raises(ValueError):
                sp.linalg.gmres(a, ng_b)
            ng_b = xp.ones((self.n, 2), dtype="f")
            with pytest.raises(ValueError):
                sp.linalg.gmres(a, ng_b)
            ng_x0 = xp.ones((self.n + 1,), dtype="f")
            with pytest.raises(ValueError):
                sp.linalg.gmres(a, b, x0=ng_x0)
            ng_M = xp.diag(xp.ones((self.n + 1,), dtype="f"))
            with pytest.raises(ValueError):
                sp.linalg.gmres(a, b, M=ng_M)
            ng_callback_type = "?"
            with pytest.raises(ValueError):
                sp.linalg.gmres(a, b, callback_type=ng_callback_type)
        # NOTE: no int-dtype TypeError check (unlike upstream): dpnp
        # deliberately promotes an int A against a float b instead of
        # rejecting it (see _make_system in _iterative.py).


@testing.parameterize(
    *testing.product(
        {
            "m": [30, 40],
            "x0": [None, "ones"],
            "M": [None, "jacobi"],
            "shift": [0, 1],
            "use_linear_operator": [False, True],
        }
    )
)
@testing.fix_random()
@testing.with_requires("scipy")
class TestMinres(unittest.TestCase):
    density = 0.01

    def _is_base_config(self):
        return (
            self.x0 is None
            and self.M is None
            and self.use_linear_operator is False
        )

    def _float_dtype(self):
        # upstream builds a/b with shaped_random's default
        return numpy.dtype(cupy.default_float_type())

    def _make_matrix(self):
        dt = self._float_dtype()
        shape = (self.m, self.m)
        a = testing.shaped_random(shape, numpy, dtype=dt, scale=1)
        mask = testing.shaped_random(shape, numpy, dtype="f", scale=1)
        a[mask > self.density] = 0
        m = None
        if self.M == "jacobi":
            m = numpy.diag(1.0 / numpy.diag(a))
        return a, m

    def _make_normalized_vector(self):
        b = testing.shaped_random(
            (self.m,), numpy, dtype=self._float_dtype(), scale=1
        )
        return b / numpy.linalg.norm(b)

    def _run_both(self, a_ref, b_cpu, x0_cpu, m_ref, a_gpu, m_gpu):
        # NOTE: upstream parameterizes shift but never passes it;
        # dpnp supports shift, so it is forwarded on both sides.
        b_gpu = cupy.asarray(b_cpu)
        x0_gpu = None if x0_cpu is None else cupy.asarray(x0_cpu)
        if self.use_linear_operator:
            a_ref = scipy.sparse.linalg.aslinearoperator(a_ref)
            a_gpu = cupy.scipy.sparse.linalg.aslinearoperator(a_gpu)
            if m_gpu is not None:
                m_ref = scipy.sparse.linalg.aslinearoperator(m_ref)
                m_gpu = cupy.scipy.sparse.linalg.aslinearoperator(m_gpu)
        # NOTE: no info assertion here (unlike cg/gmres above): the
        # random test matrices are often far from symmetric positive
        # definite, so neither side reliably converges; parity is the
        # iterate itself, exactly like upstream.
        x_ref, _ = scipy.sparse.linalg.minres(
            a_ref, b_cpu, x0=x0_cpu, M=m_ref, shift=self.shift
        )
        x_dp, _ = cupy.scipy.sparse.linalg.minres(
            a_gpu, b_gpu, x0=x0_gpu, M=m_gpu, shift=self.shift
        )
        testing.assert_allclose(cupy.asnumpy(x_dp), x_ref, rtol=1e-5, atol=1e-5)

    def _prep(self):
        a_cpu, m_cpu = self._make_matrix()
        b_cpu = self._make_normalized_vector()
        x0_cpu = None
        if self.x0 == "ones":
            x0_cpu = numpy.ones((self.m,), dtype=self._float_dtype())
        return a_cpu, m_cpu, b_cpu, x0_cpu

    def test_dense(self):
        a_cpu, m_cpu, b_cpu, x0_cpu = self._prep()
        a_gpu = cupy.asarray(a_cpu)
        m_gpu = None if m_cpu is None else cupy.asarray(m_cpu)
        self._run_both(a_cpu, b_cpu, x0_cpu, m_cpu, a_gpu, m_gpu)

    def test_sparse(self):
        a_cpu, m_cpu, b_cpu, x0_cpu = self._prep()
        a_ref = scipy.sparse.csr_matrix(a_cpu)
        m_ref = None if m_cpu is None else scipy.sparse.csr_matrix(m_cpu)
        a_gpu = cupy.scipy.sparse.csr_matrix(cupy.asarray(a_cpu))
        m_gpu = (
            None
            if m_cpu is None
            else cupy.scipy.sparse.csr_matrix(cupy.asarray(m_cpu))
        )
        self._run_both(a_ref, b_cpu, x0_cpu, m_ref, a_gpu, m_gpu)

    def test_invalid(self):
        if not self._is_base_config():
            self.skipTest("base config only")
        dt = self._float_dtype()
        for xp, sp in ((numpy, scipy.sparse), (cupy, cupy.scipy.sparse)):
            a = testing.shaped_random((self.m, self.m), numpy, dtype=dt)
            b = testing.shaped_random((self.m,), numpy, dtype=dt)
            if xp is not numpy:
                a = xp.asarray(a)
                b = xp.asarray(b)
            ng_a = xp.ones((self.m,), dtype=dt)
            with pytest.raises(ValueError):
                sp.linalg.minres(ng_a, b)
            ng_a = xp.ones((self.m, self.m + 1), dtype=dt)
            with pytest.raises(ValueError):
                sp.linalg.minres(ng_a, b)
            ng_a = xp.ones((self.m, self.m, 1), dtype=dt)
            with pytest.raises(ValueError):
                sp.linalg.minres(ng_a, b)
            ng_b = xp.ones((self.m + 1,), dtype=dt)
            with pytest.raises(ValueError):
                sp.linalg.minres(a, ng_b)
            ng_b = xp.ones((self.m, 2), dtype=dt)
            with pytest.raises(ValueError):
                sp.linalg.minres(a, ng_b)
            ng_x0 = xp.ones((self.m + 1,), dtype=dt)
            with pytest.raises(ValueError):
                sp.linalg.minres(a, b, x0=ng_x0)
            ng_M = xp.diag(xp.ones((self.m + 1,), dtype=dt))
            with pytest.raises(ValueError):
                sp.linalg.minres(a, b, M=ng_M)

    def test_callback(self):
        if not self._is_base_config():
            self.skipTest("base config only")
        a_cpu, _ = self._make_matrix()
        b_cpu = self._make_normalized_vector()
        a_gpu = cupy.asarray(a_cpu)
        b_gpu = cupy.asarray(b_cpu)
        is_called = False

        def callback(x):
            nonlocal is_called
            is_called = True

        cupy.scipy.sparse.linalg.minres(a_gpu, b_gpu, callback=callback)
        assert is_called
