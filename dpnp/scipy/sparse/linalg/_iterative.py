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

"""Iterative sparse linear solvers for dpnp -- pure GPU/SYCL implementation.

All computation stays on the device (USM/oneMKL).  There is NO host-dispatch
fallback: transferring data to the CPU for small systems defeats the purpose
of keeping a live computation on GPU memory.

Solver coverage
---------------
cg     : Conjugate Gradient (Hermitian positive definite)
gmres  : Restarted GMRES (general non-symmetric)
minres : MINRES (symmetric possibly indefinite)

SpMV fast-path
--------------
When a CSR dpnp sparse matrix is passed as A or M, _make_fast_matvec()
constructs a _CachedSpMV object that:
  1. Calls _sparse_gemv_init() ONCE to create the oneMKL matrix_handle,
     register CSR pointers via set_csr_data, and run optimize_gemv
     (the expensive sparsity-analysis phase).
  2. Calls _sparse_gemv_compute() on every matvec -- only the cheap
     oneMKL sparse::gemv kernel fires; no handle setup overhead.
  3. Calls _sparse_gemv_release() in __del__ to free the handle.

This means optimize_gemv runs once per operator, not once per iteration,
which is the correct usage pattern for oneMKL sparse BLAS.

Supported dtypes for the oneMKL SpMV fast-path:
  values : float32, float64, complex64, complex128
  indices: int32, int64
Complex dtypes require oneMKL sparse BLAS support (available since
oneMKL 2023.x); if the dispatch table slot is nullptr (types_matrix.hpp
does not register the pair) a ValueError is raised by the C++ layer.
_make_fast_matvec catches this and falls back to A.dot(x).
"""

from __future__ import annotations

from typing import Callable

import dpctl.utils as dpu
import numpy

import dpnp
import dpnp.backend.extensions.blas._blas_impl as bi

from ._interface import IdentityOperator, LinearOperator, aslinearoperator

# ---------------------------------------------------------------------------
# oneMKL sparse SpMV hook -- cached-handle API
# ---------------------------------------------------------------------------

try:
    from dpnp.backend.extensions.sparse import _sparse_impl as _si

    _HAS_SPARSE_IMPL = True
except ImportError:
    _si = None
    _HAS_SPARSE_IMPL = False

_SUPPORTED_DTYPES = frozenset("fdFD")


def _np_dtype(dp_dtype) -> numpy.dtype:
    """Normalise any dtype-like (dpnp type/numpy type/string) to numpy.dtype."""
    return numpy.dtype(dp_dtype)


def _check_dtype(dtype, name: str) -> None:
    if _np_dtype(dtype).char not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"{name} has unsupported dtype {dtype}; "
            "only float32, float64, complex64, complex128 are accepted."
        )


class _CachedSpMV:
    """
    Wrap a CSR matrix with a persistent oneMKL matrix_handle.

    The handle is initialised (set_csr_data + optimize_gemv) exactly once
    in __init__. Subsequent calls to __call__ only invoke sparse::gemv,
    paying no analysis overhead. The handle is released in __del__.

    Parameters
    ----------
    A : dpnp CSR sparse matrix
    trans : int  0=N, 1=T, 2=C  (fixed at construction)
    """

    __slots__ = (
        "_A",
        "_exec_q",
        "_handle",
        "_trans",
        "_nrows",
        "_ncols",
        "_nnz",
        "_out_size",
        "_in_size",
        "_dtype",
        "_val_type_id",
    )

    def __init__(self, A, trans: int = 0):
        self._A = A  # keep alive so USM pointers stay valid
        self._trans = int(trans)
        self._nrows = int(A.shape[0])
        self._ncols = int(A.shape[1])
        self._nnz = int(A.data.shape[0])
        self._exec_q = A.data.sycl_queue
        self._dtype = A.data.dtype

        # Output and input lengths depend on transpose mode.
        # For trans=0 (N): y has nrows, x has ncols.
        # For trans=1/2 (T/C): y has ncols, x has nrows.
        if self._trans == 0:
            self._out_size = self._nrows
            self._in_size = self._ncols
        else:
            self._out_size = self._ncols
            self._in_size = self._nrows

        self._handle = None
        self._val_type_id = -1

        # init_matrix_handle + set_csr_data + optimize_gemv (once).
        # We must wait on optimize_gemv before any compute call can run;
        # this is the only place __init__/__call__ blocks.
        handle, val_type_id, ev = _si._sparse_gemv_init(
            self._exec_q,
            self._trans,
            A.indptr,
            A.indices,
            A.data,
            self._nrows,
            self._ncols,
            self._nnz,
            [],
        )
        ev.wait()
        self._handle = handle
        self._val_type_id = val_type_id

    def __call__(self, x: dpnp.ndarray) -> dpnp.ndarray:
        """Y = op(A) * x -- only sparse::gemv fires, fully async."""
        y = dpnp.empty(
            self._out_size, dtype=self._dtype, sycl_queue=self._exec_q
        )
        # Do NOT wait on the event -- subsequent dpnp ops on the same
        # queue will serialize behind it automatically. Blocking here
        # throws away async overlap and dominates small-problem runtime.
        _si._sparse_gemv_compute(
            self._exec_q,
            self._handle,
            self._val_type_id,
            self._trans,
            1.0,
            x,
            0.0,
            y,
            self._nrows,
            self._ncols,
            [],
        )
        return y

    def __del__(self):
        # Guard against partial construction: _handle may not be set if
        # __init__ raised before the assignment.
        handle = getattr(self, "_handle", None)
        if handle is not None and _si is not None:
            try:
                _si._sparse_gemv_release(self._exec_q, handle, [])
            except Exception:
                pass
            self._handle = None


class _CachedSpMVPair:
    """Holds forward and (lazily built) adjoint cached SpMV handles."""

    __slots__ = ("forward", "_A", "_adjoint")

    def __init__(self, A):
        self.forward = _CachedSpMV(A, trans=0)
        self._A = A
        self._adjoint = None

    def matvec(self, x):
        """Apply the operator to vector x."""
        return self.forward(x)

    def rmatvec(self, x):
        """Return the data type of the operator."""
        if self._adjoint is None:
            # Build conjtrans handle on first use. For real dtypes
            # this is equivalent to trans=1.
            is_cpx = dpnp.issubdtype(self._A.data.dtype, dpnp.complexfloating)
            self._adjoint = _CachedSpMV(self._A, trans=2 if is_cpx else 1)
        return self._adjoint(x)


def _make_fast_matvec(A):
    """Return a _CachedSpMVPair if A is a CSR matrix with oneMKL support,
    or None if A is not an eligible sparse matrix.

    Falls back to None (caller uses A.dot) on:
      - missing _sparse_impl extension
      - dtype not supported by the C++ dispatch table
      - any other C++ exception during handle initialisation
    """
    try:
        from dpnp.scipy import sparse as _sp

        if not (_sp.issparse(A) and A.format == "csr"):
            return None
    except (ImportError, AttributeError):
        return None

    if not _HAS_SPARSE_IMPL:
        return None

    # Only build the cached handle for supported dtypes.
    if _np_dtype(A.data.dtype).char not in _SUPPORTED_DTYPES:
        return None

    try:
        return _CachedSpMVPair(A)
    except Exception:
        return None


def _make_system(A, M, x0, b):
    """Validate and prepare (A_op, M_op, x, b, dtype) on device.

    dpnp-only policy: b, x0, and any dense operator inputs must already
    be dpnp arrays. No host->device promotion happens here.

    dtype promotion follows CuPy v14 rules: A.dtype is used when it is in
    {f,d,F,D}; otherwise b.dtype is promoted to float64 (real) or
    complex128 (complex).
    """
    if not isinstance(b, dpnp.ndarray):
        raise TypeError(f"b must be a dpnp.ndarray, got {type(b).__name__}")
    if x0 is not None and not isinstance(x0, dpnp.ndarray):
        raise TypeError(
            f"x0 must be a dpnp.ndarray or None, got {type(x0).__name__}"
        )

    A_op = aslinearoperator(A)
    if A_op.shape[0] != A_op.shape[1]:
        raise ValueError("A must be a square operator")
    n = A_op.shape[0]

    b = b.reshape(-1)
    if b.shape[0] != n:
        raise ValueError(
            f"b length {b.shape[0]} does not match operator dimension {n}"
        )

    # Dtype promotion: prefer A.dtype; fall back via b.dtype.
    if (
        A_op.dtype is not None
        and _np_dtype(A_op.dtype).char in _SUPPORTED_DTYPES
    ):
        dtype = A_op.dtype
    elif dpnp.issubdtype(b.dtype, dpnp.complexfloating):
        dtype = dpnp.complex128
    else:
        dtype = dpnp.float64

    b = b.astype(dtype, copy=False)
    _check_dtype(b.dtype, "b")

    if x0 is None:
        x = dpnp.zeros(n, dtype=dtype, sycl_queue=b.sycl_queue)
    else:
        x = x0.astype(dtype, copy=True).reshape(-1)
        if x.shape[0] != n:
            raise ValueError(f"x0 length {x.shape[0]} != n={n}")

    if M is None:
        M_op = IdentityOperator((n, n), dtype=dtype)
    else:
        M_op = aslinearoperator(M)
        if M_op.shape != A_op.shape:
            raise ValueError(
                f"preconditioner shape {M_op.shape} != "
                f"operator shape {A_op.shape}"
            )

        fast_mv_M = _make_fast_matvec(M)
        if fast_mv_M is not None:
            _orig_M = M_op

            class _FastMOp(LinearOperator):
                def __init__(self):
                    super().__init__(_orig_M.dtype, _orig_M.shape)

                def _matvec(self, x):
                    return fast_mv_M.matvec(x)

                def _rmatvec(self, x):
                    return fast_mv_M.rmatvec(x)

            M_op = _FastMOp()

    # Inject fast CSR SpMV for A if available.
    fast_mv = _make_fast_matvec(A)
    if fast_mv is not None:
        _orig = A_op

        class _FastOp(LinearOperator):
            def __init__(self):
                super().__init__(_orig.dtype, _orig.shape)

            def _matvec(self, x):
                return fast_mv.matvec(x)

            def _rmatvec(self, x):
                return fast_mv.rmatvec(x)

        A_op = _FastOp()

    return A_op, M_op, x, b, dtype


def _get_atol(b_norm: float, atol, rtol: float) -> float:
    """Absolute stopping tolerance: max(atol, rtol*||b||), mirroring SciPy."""
    if atol == "legacy" or atol is None:
        atol = 0.0
    atol = float(atol)
    if atol < 0:
        raise ValueError(
            f"atol={atol!r} is invalid; must be a real, non-negative number."
        )
    return max(atol, float(rtol) * float(b_norm))


def cg(
    A,
    b,
    x0: dpnp.ndarray | None = None,
    *,
    rtol: float = 1e-5,
    tol: float | None = None,
    maxiter: int | None = None,
    M=None,
    callback: Callable | None = None,
    atol=None,
) -> tuple[dpnp.ndarray, int]:
    """Conjugate Gradient -- pure dpnp/oneMKL, Hermitian positive definite A.

    Parameters
    ----------
    A       : array_like or LinearOperator -- HPD (n, n)
    b       : array_like -- right-hand side (n,)
    x0      : array_like, optional -- initial guess
    rtol    : float -- relative tolerance (default 1e-5)
    tol     : float, optional -- deprecated alias for rtol
    maxiter : int, optional -- max iterations (default 10*n)
    M       : LinearOperator or array_like, optional -- SPD preconditioner
    callback: callable, optional -- callback(xk) after each iteration
    atol    : float, optional -- absolute tolerance

    Returns
    -------
    x    : dpnp.ndarray
    info : int  0=converged  >0=maxiter  -1=breakdown
    """
    if tol is not None:
        rtol = tol

    A_op, M_op, x, b, dtype = _make_system(A, M, x0, b)
    n = b.shape[0]

    bnrm = dpnp.linalg.norm(b)
    bnrm_host = float(bnrm)
    if bnrm_host == 0.0:
        return dpnp.zeros_like(b), 0

    atol_eff_host = _get_atol(bnrm_host, atol=atol, rtol=rtol)

    if maxiter is None:
        maxiter = n * 10

    rhotol = float(numpy.finfo(_np_dtype(dtype)).eps ** 2)

    r = b - A_op.matvec(x) if x0 is not None else b.copy()
    z = M_op.matvec(r)
    p = z.copy()

    # rz is kept as a 0-D dpnp array on device.
    rz = dpnp.real(dpnp.vdot(r, z))

    # Single sync for the initial breakdown check.
    if float(dpnp.abs(rz)) < rhotol:
        return x, 0

    info = maxiter

    for _k in range(maxiter):
        # Convergence check (sync).
        rnorm = dpnp.linalg.norm(r)
        if float(rnorm) <= atol_eff_host:
            info = 0
            break

        Ap = A_op.matvec(p)
        pAp = dpnp.real(dpnp.vdot(p, Ap))  # 0-D on device

        if float(dpnp.abs(pAp)) < rhotol:
            info = -1
            break

        alpha = rz / pAp  # 0-D on device
        x = x + alpha * p  # fully on-device
        r = r - alpha * Ap

        if callback is not None:
            callback(x)

        z = M_op.matvec(r)
        rz_new = dpnp.real(dpnp.vdot(r, z))

        if float(dpnp.abs(rz_new)) < rhotol:
            info = 0
            break

        beta = rz_new / rz  # 0-D on device
        p = z + beta * p
        rz = rz_new
    else:
        info = maxiter

    return x, int(info)


def gmres(
    A,
    b,
    x0: dpnp.ndarray | None = None,
    *,
    rtol: float = 1e-5,
    atol: float = 0.0,
    restart: int | None = None,
    maxiter: int | None = None,
    M=None,
    callback: Callable | None = None,
    callback_type: str | None = None,
) -> tuple[dpnp.ndarray, int]:
    """Uses Generalized Minimal RESidual iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : LinearOperator, dpnp sparse matrix, or 2-D dpnp.ndarray
        The real or complex matrix of the linear system, shape (n, n).
    b : dpnp.ndarray
        Right-hand side of the linear system, shape (n,) or (n, 1).
    x0 : dpnp.ndarray, optional
        Starting guess for the solution.
    rtol, atol : float
        Tolerance for convergence: ``||r|| <= max(atol, rtol*||b||)``.
    restart : int, optional
        Number of iterations between restarts (default 20). Larger values
        increase iteration cost but may be necessary for convergence.
    maxiter : int, optional
        Maximum number of iterations (default 10*n).
    M : LinearOperator, dpnp sparse matrix, or 2-D dpnp.ndarray, optional
        Preconditioner for ``A``; should approximate the inverse of ``A``.
    callback : callable, optional
        User-specified function to call on every restart. Called as
        ``callback(arg)``, where ``arg`` is selected by ``callback_type``.
    callback_type : {'x', 'pr_norm'}, optional
        If 'x', the current solution vector is passed to the callback.
        If 'pr_norm', the relative (preconditioned) residual norm.
        Default is 'pr_norm' when a callback is supplied.

    Returns
    -------
    x : dpnp.ndarray
        The (approximate) solution. Note that this is M @ x in the
        right-preconditioned formulation, matching CuPy's return value.
    info : int
        0 if converged; iteration count if maxiter was reached.

    See Also
    --------
    scipy.sparse.linalg.gmres
    cupyx.scipy.sparse.linalg.gmres
    """
    A_op, M_op, x, b, dtype = _make_system(A, M, x0, b)
    matvec = A_op.matvec
    psolve = M_op.matvec

    n = A_op.shape[0]
    if n == 0:
        return dpnp.empty_like(b), 0
    b_norm = dpnp.linalg.norm(b)
    if b_norm == 0.0:
        return b, 0
    atol = max(float(atol), rtol * float(b_norm))

    if maxiter is None:
        maxiter = n * 10
    if restart is None:
        restart = 20
    restart = min(int(restart), n)

    if callback_type is None:
        callback_type = "pr_norm"
    if callback_type not in ("x", "pr_norm"):
        raise ValueError(f"Unknown callback_type: {callback_type!r}")
    if callback is None:
        callback_type = None

    queue = b.sycl_queue

    # Krylov basis V, Hessenberg H, and RHS e all live on device to
    # avoid host-device sync overhead (which dominates on Intel GPUs
    # even for small transfers).  CuPy keeps e on host and solves
    # lstsq on CPU, but for dpnp we keep everything on device.
    V = dpnp.empty((n, restart), dtype=dtype, sycl_queue=queue, order="F")
    H = dpnp.zeros(
        (restart + 1, restart), dtype=dtype, sycl_queue=queue, order="F"
    )
    e = dpnp.zeros(restart + 1, dtype=dtype, sycl_queue=queue)

    compute_hu = _make_compute_hu(V)

    iters = 0
    while True:
        mx = psolve(x)
        r = b - matvec(mx)
        r_norm = dpnp.linalg.norm(r)

        if callback_type == "x":
            callback(mx)
        elif callback_type == "pr_norm" and iters > 0:
            callback(r_norm / b_norm)

        if r_norm <= atol or iters >= maxiter:
            break

        v = r / r_norm
        V[:, 0] = v
        e[0] = r_norm

        # Arnoldi iteration
        for j in range(restart):
            z = psolve(v)
            u = matvec(z)
            H[: j + 1, j], u = compute_hu(u, j)
            H[j + 1, j] = dpnp.linalg.norm(u)
            if j + 1 < restart:
                v = u / H[j + 1, j]
                V[:, j + 1] = v

        # Solve the Hessenberg least-squares H y = e on device.
        # Tiny problem (~restart x restart), kept on-device to avoid sync.
        y, *_ = dpnp.linalg.lstsq(H, e, rcond=None)
        x = x + dpnp.dot(V, y)
        iters += restart

    info = 0
    if iters >= maxiter and not bool(r_norm <= atol):
        info = iters

    return mx, info


def minres(
    A,
    b,
    x0: dpnp.ndarray | None = None,
    *,
    rtol: float = 1e-5,
    shift: float = 0.0,
    tol: float | None = None,
    maxiter: int | None = None,
    M=None,
    callback: Callable | None = None,
    show: bool = False,
    check: bool = False,
) -> tuple[dpnp.ndarray, int]:
    """Uses MINimum RESidual iteration to solve ``Ax = b``.

    Solves the symmetric (possibly indefinite) system ``Ax = b`` or,
    if *shift* is nonzero, ``(A - shift*I)x = b``.  All computation
    stays on the SYCL device; only scalar recurrence coefficients and
    norms are transferred to the host for branching.

    The algorithm follows SciPy's MINRES (Paige & Saunders, 1975)
    line-for-line.  Three host syncs per iteration are unavoidable:
    ``alpha`` and ``beta`` (Lanczos inner products) and ``ynorm``
    (solution norm for stopping tests).

    Parameters
    ----------
    A : dpnp sparse matrix, 2-D dpnp.ndarray, or LinearOperator
        The real symmetric or complex Hermitian matrix, shape ``(n, n)``.
    b : dpnp.ndarray
        Right-hand side, shape ``(n,)`` or ``(n, 1)``.
    x0 : dpnp.ndarray, optional
        Starting guess for the solution.
    shift : float
        If nonzero, solve ``(A - shift*I)x = b``.  Default 0.
    rtol : float
        Relative tolerance for convergence.  Default 1e-5.
    tol : float, optional
        Deprecated alias for *rtol*.
    maxiter : int, optional
        Maximum number of iterations.  Default ``5*n``.
    M : dpnp sparse matrix, dpnp.ndarray, or LinearOperator, optional
        Preconditioner approximating the inverse of ``A``.
    callback : callable, optional
        Called as ``callback(xk)`` after each iteration.
    show : bool
        If True, print convergence summary each iteration.
    check : bool
        If True, verify that ``A`` and ``M`` are symmetric before
        iterating.  Costs extra matvecs.

    Returns
    -------
    x : dpnp.ndarray
        The converged (or best) solution.
    info : int
        0 if converged, ``maxiter`` if the iteration limit was reached.

    Notes
    -----
    This is a direct translation of the Paige--Saunders MINRES algorithm
    as implemented in SciPy, adapted for dpnp device arrays with the
    oneMKL SpMV cached-handle fast-path.

    See Also
    --------
    scipy.sparse.linalg.minres
    cupyx.scipy.sparse.linalg.minres
    """
    if tol is not None:
        rtol = tol

    A_op, M_op, x, b, dtype = _make_system(A, M, x0, b)
    matvec = A_op.matvec
    psolve = M_op.matvec

    n = A_op.shape[0]
    if maxiter is None:
        maxiter = 5 * n

    istop = 0
    itn = 0
    Anorm = 0
    Acond = 0
    rnorm = 0
    ynorm = 0

    xtype = dtype
    eps = dpnp.finfo(xtype).eps

    # ------------------------------------------------------------------
    # Set up y and v for the first Lanczos vector v1.
    #   y  = beta1 * P' * v1, where P = M**(-1).
    #   v  is really P' * v1.
    # ------------------------------------------------------------------

    Ax = matvec(x)
    r1 = b - Ax
    y = psolve(r1)

    # beta1 = <r1, y>   -- one host sync (setup only).
    # Transferred to host immediately because beta1 seeds ~5 host-side
    # scalars (beta, qrnorm, phibar, rhs1) used in Python arithmetic
    # and branches every iteration.  Keeping it as a 0-D device array
    # would cascade implicit syncs or 0-D allocations throughout the
    # recurrence.
    beta1 = dpnp.inner(r1, y)

    if beta1 < 0:
        raise ValueError("indefinite preconditioner")
    elif beta1 == 0:
        return (x, 0)

    beta1 = dpnp.sqrt(beta1)
    beta1 = float(beta1)

    if check:
        # See if A is symmetric.  All on device; only the bool syncs.
        w_chk = matvec(y)
        r2_chk = matvec(w_chk)
        s = dpnp.inner(w_chk, w_chk)
        t = dpnp.inner(y, r2_chk)
        if abs(s - t) > (s + eps) * eps ** (1.0 / 3.0):
            raise ValueError("non-symmetric matrix")

        # See if M is symmetric.
        r2_chk = psolve(y)
        s = dpnp.inner(y, y)
        t = dpnp.inner(r1, r2_chk)
        if abs(s - t) > (s + eps) * eps ** (1.0 / 3.0):
            raise ValueError("non-symmetric preconditioner")

    # Initialise remaining quantities (all host-side scalars).
    oldb = 0
    beta = beta1
    dbar = 0
    epsln = 0
    qrnorm = beta1
    phibar = beta1
    rhs1 = beta1
    rhs2 = 0
    tnorm2 = 0
    gmax = 0
    gmin = dpnp.finfo(xtype).max
    cs = -1
    sn = 0
    queue = b.sycl_queue
    w = dpnp.zeros(n, dtype=xtype, sycl_queue=queue)
    w2 = dpnp.zeros(n, dtype=xtype, sycl_queue=queue)
    r2 = r1

    # Main Lanczos loop.
    while itn < maxiter:
        itn += 1

        s = 1.0 / beta
        v = s * y  # on device

        y = matvec(v)
        y = y - shift * v

        if itn >= 2:
            y = y - (beta / oldb) * r1

        # alpha = <v, y>   -- host sync #1
        alpha = float(dpnp.inner(v, y))

        y = y - (alpha / beta) * r2
        r1 = r2
        r2 = y
        y = psolve(r2)
        oldb = beta

        # beta = sqrt(<r2, y>)   -- host sync #2
        beta = float(dpnp.inner(r2, y))
        if beta < 0:
            raise ValueError("non-symmetric matrix")
        beta = numpy.sqrt(beta)

        tnorm2 += alpha**2 + oldb**2 + beta**2

        if itn == 1:
            if beta / beta1 <= 10 * eps:
                istop = -1  # Terminate later

        # Apply previous rotation Q_{k-1} to get
        #   [delta_k  epsln_{k+1}] = [cs  sn] [dbar_k     0     ]
        #   [gbar_k   dbar_{k+1} ]   [sn -cs] [alpha_k  beta_{k+1}]
        oldeps = epsln
        delta = cs * dbar + sn * alpha
        gbar = sn * dbar - cs * alpha
        epsln = sn * beta
        dbar = -cs * beta
        root = numpy.sqrt(gbar**2 + dbar**2)

        # Compute the next plane rotation Q_k.
        gamma = numpy.sqrt(gbar**2 + beta**2)
        gamma = max(gamma, eps)
        cs = gbar / gamma
        sn = beta / gamma
        phi = cs * phibar
        phibar = sn * phibar

        # Update x  -- all on device.
        denom = 1.0 / gamma
        w1 = w2
        w2 = w
        w = (v - oldeps * w1 - delta * w2) * denom
        x = x + phi * w

        # Go round again.
        gmax = max(gmax, gamma)
        gmin = min(gmin, gamma)
        z = rhs1 / gamma
        rhs1 = rhs2 - delta * z
        rhs2 = -epsln * z

        # ----------------------------------------------------------
        # Estimate norms and test for convergence.
        # ----------------------------------------------------------
        Anorm = numpy.sqrt(tnorm2)
        ynorm = float(dpnp.linalg.norm(x))  # host sync #3
        epsa = Anorm * eps
        epsx = Anorm * ynorm * eps
        epsr = Anorm * ynorm * rtol
        diag = gbar
        if diag == 0:
            diag = epsa

        qrnorm = phibar
        rnorm = qrnorm
        if ynorm == 0 or Anorm == 0:
            test1 = numpy.inf
        else:
            test1 = rnorm / (Anorm * ynorm)  # ||r|| / (||A|| ||x||)
        if Anorm == 0:
            test2 = numpy.inf
        else:
            test2 = root / Anorm  # ||Ar|| / (||A|| ||r||)

        # Estimate cond(A).
        Acond = gmax / gmin

        # Stopping criteria (SciPy's istop codes).
        if istop == 0:
            t1 = 1 + test1
            t2 = 1 + test2
            if t2 <= 1:
                istop = 2
            if t1 <= 1:
                istop = 1

            if itn >= maxiter:
                istop = 6
            if Acond >= 0.1 / eps:
                istop = 4
            if epsx >= beta1:
                istop = 3
            if test2 <= rtol:
                istop = 2
            if test1 <= rtol:
                istop = 1

        if show:
            prnt = (
                n <= 40
                or itn <= 10
                or itn >= maxiter - 10
                or itn % 10 == 0
                or qrnorm <= 10 * epsx
                or qrnorm <= 10 * epsr
                or Acond <= 1e-2 / eps
                or istop != 0
            )
            if prnt:
                x1 = float(x[0])
                print(
                    f"{itn:6g} {x1:12.5e} {test1:10.3e}"
                    f" {test2:10.3e}"
                    f" {Anorm:8.1e} {Acond:8.1e}"
                    f" {gbar / Anorm if Anorm else 0:8.1e}"
                )
                if itn % 10 == 0:
                    print()

        if callback is not None:
            callback(x)

        if istop != 0:
            break

    if istop == 6:
        info = maxiter
    else:
        info = 0

    return (x, info)


def _make_compute_hu(V):
    """Factory mirroring cupyx's _make_compute_hu using oneMKL gemv directly.

    Returns a closure compute_hu(u, j) that performs:
        h = V[:, :j+1]^H @ u     (gemv with transpose=True)
        u = u - V[:, :j+1] @ h   (gemv with transpose=False, then subtract)

    The current bi._gemv binding hardcodes alpha=1, beta=0, so the second
    pass requires a temporary vector and an explicit subtraction.  To get
    CuPy's fused u -= V@h in one kernel, the C++ binding would need
    alpha/beta parameters.

    V must be column-major; sub-views V[:, :j+1] of an F-order array
    are themselves F-contiguous, so the same closure handles every j.
    """
    if V.ndim != 2 or not V.flags.f_contiguous:
        raise ValueError(
            "_make_compute_hu: V must be a 2-D column-major (F-order) "
            "dpnp array"
        )

    exec_q = V.sycl_queue
    dtype = V.dtype
    is_cpx = dpnp.issubdtype(dtype, dpnp.complexfloating)

    def compute_hu(u, j):
        # h = V[:, :j+1]^H @ u  (allocate fresh, length j+1)
        h = dpnp.empty(j + 1, dtype=dtype, sycl_queue=exec_q)

        # Sub-view: column-major slice of the trailing axis is F-contiguous.
        Vj = V[:, : j + 1]
        Vj_usm = dpnp.get_usm_ndarray(Vj)
        u_usm = dpnp.get_usm_ndarray(u)
        h_usm = dpnp.get_usm_ndarray(h)

        _manager = dpu.SequentialOrderManager[exec_q]

        # Pass 1: h = Vj^T @ u  (real) or  h = (Vj^T @ u) then conj  (complex)
        ht1, ev1 = bi._gemv(
            exec_q,
            Vj_usm,
            u_usm,
            h_usm,
            transpose=True,
            depends=_manager.submitted_events,
        )
        _manager.add_event_pair(ht1, ev1)

        if is_cpx:
            # h = conj(h) -- in-place, length j+1, negligible
            h = dpnp.conj(h, out=h)
            h_usm = dpnp.get_usm_ndarray(h)

        # Pass 2: tmp = Vj @ h, then u -= tmp
        # No fused AXPY available, so we still allocate tmp.
        tmp = dpnp.empty_like(u)
        tmp_usm = dpnp.get_usm_ndarray(tmp)
        ht2, ev2 = bi._gemv(
            exec_q,
            Vj_usm,
            h_usm,
            tmp_usm,
            transpose=False,
            depends=_manager.submitted_events,
        )
        _manager.add_event_pair(ht2, ev2)

        u -= tmp
        return h, u

    return compute_hu
