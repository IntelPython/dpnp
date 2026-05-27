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

# Math-heavy module: single-letter and CamelCase identifiers such as
# A, M, X, V, H, Ap, Ax, Anorm, Acond, Vj, A_op, M_op, fast_mv_M,
# _orig_M are part of the published numerical-linear-algebra API and
# mirror SciPy/CuPy verbatim, so the snake_case rule is intentionally
# relaxed for the whole file.
# pylint: disable=invalid-name

from __future__ import annotations

from typing import Callable

import dpctl.utils as dpu
import numpy

import dpnp

# _blas_impl is a compiled (.so / .pyd) C-extension produced by the
# dpnp build; pylint cannot statically introspect its exported symbols.
# pylint: disable-next=no-name-in-module
import dpnp.backend.extensions.blas._blas_impl as bi

from ._interface import IdentityOperator, LinearOperator, aslinearoperator

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


# pylint: disable-next=too-many-instance-attributes
class _CachedSpMV:
    """
    Wrap a CSR matrix with a persistent oneMKL matrix_handle.

    The handle is initialised (set_csr_data + optimize_gemv) exactly once
    in __init__. Subsequent calls to __call__ only invoke sparse::gemv,
    paying no analysis overhead. The handle is released in __del__.

    Parameters
    ----------
    A : dpnp CSR sparse matrix
    si : dpnp.backend.extensions.sparse._sparse_impl module
        Passed in from _make_fast_matvec to keep the import lazy and
        avoid a circular import during dpnp package initialization.
    trans : int  0=N, 1=T, 2=C  (fixed at construction)
    """

    __slots__ = (
        "_A",
        "_si",
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

    def __init__(self, A, si, trans: int = 0):
        self._A = A  # keep alive so USM pointers stay valid
        self._si = si
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
        # pylint: disable-next=protected-access
        handle, val_type_id, ev = self._si._sparse_gemv_init(
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
        # pylint: disable-next=protected-access
        self._si._sparse_gemv_compute(
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
        si = getattr(self, "_si", None)
        if handle is None or si is None:
            return

        # During interpreter shutdown the compiled extension may be
        # collected before this __del__ runs; in that case
        # ``si._sparse_gemv_release`` evaluates to ``None`` (or raises
        # AttributeError on some module proxies). Probe explicitly so
        # we can distinguish "extension already torn down -- leak the
        # handle, the OS will reclaim it" from "release call raised --
        # narrow except below" and not silence both with one broad
        # ``except Exception``.
        release_fn = getattr(si, "_sparse_gemv_release", None)
        if release_fn is None:
            self._handle = None
            return

        try:
            release_fn(self._exec_q, handle, [])
        except (AttributeError, TypeError):
            # Shutdown-mode races: queue or handle attribute access
            # may itself raise once the supporting dpctl / pybind11
            # state is gone. The handle is unrecoverable; leave the
            # OS to reclaim it at process exit.
            pass
        except Exception:  # pylint: disable=broad-exception-caught
            # Genuine backend error while the interpreter is still
            # healthy. Swallowing here is still required (raising
            # from __del__ produces an unraisable-exception warning
            # and serves no purpose -- the handle is gone either
            # way), but the explicit broad-except now documents the
            # intent rather than masking the shutdown race above.
            pass
        finally:
            self._handle = None


class _CachedSpMVPair:
    """Forward + lazily-built adjoint SpMV closures around a csr_matrix.

    The forward handle is owned by the ``csr_matrix`` itself (built via
    ``csr_matrix._ensure_spmv_handle()``) and therefore shared with any
    other call site -- including a user-issued ``A.dot(x)`` outside the
    solver. The adjoint handle is built on demand and owned by this
    pair instance; ``__del__`` releases it.
    """

    __slots__ = ("_A", "_si", "_adjoint")

    def __init__(self, A, si):
        self._A = A
        self._si = si
        self._adjoint = None

    def matvec(self, x):
        """Apply the forward operator A @ x via the csr's cached handle."""
        # _ensure_spmv_handle has already been validated by the caller
        # (_make_fast_matvec) before this pair was constructed, so it
        # cannot return None here. We re-fetch on every call only to
        # pick up the (immutable) handle pointer and exec_q without
        # caching them redundantly on this object.
        _si, handle, val_type_id, exec_q = self._A._ensure_spmv_handle()
        y = dpnp.empty(
            self._A.shape[0], dtype=self._A.data.dtype, sycl_queue=exec_q
        )
        # pylint: disable-next=protected-access
        _si._sparse_gemv_compute(
            exec_q,
            handle,
            val_type_id,
            0,    # trans=N
            1.0,  # alpha
            x,
            0.0,  # beta
            y,
            int(self._A.shape[0]),
            int(self._A.shape[1]),
            [],
        )
        return y

    def rmatvec(self, x):
        """Apply the conjugate-transpose operator A^H @ x."""
        if self._adjoint is None:
            # Build conjtrans handle on first use. For real dtypes
            # this is equivalent to trans=1.
            is_cpx = dpnp.issubdtype(self._A.data.dtype, dpnp.complexfloating)
            self._adjoint = _CachedSpMV(
                self._A, self._si, trans=2 if is_cpx else 1
            )
        return self._adjoint(x)


def _make_fast_matvec(A):
    """Return a _CachedSpMVPair if A is a CSR matrix with oneMKL support,
    or None if A is not an eligible sparse matrix.

    Falls back to None (caller uses A.dot) on:
      - A is not a dpnp CSR sparse matrix
      - the compiled backend extension is unavailable
      - the (value, index) dtype combination is not registered with
        the oneMKL dispatch table
      - handle initialisation raises for any other backend-specific
        reason
    """
    try:
        # Lazy import: dpnp.scipy.sparse may import this module during
        # package initialisation, so a top-level import would deadlock.
        # pylint: disable-next=import-outside-toplevel
        from dpnp.scipy import sparse as _sp

        if not (_sp.issparse(A) and A.format == "csr"):
            return None
    except (ImportError, AttributeError):
        return None

    # Probe the csr_matrix's own SpMV path. This either returns a
    # fully-built handle (cached on A for sharing with A.dot) or None
    # when the backend extension / dtype combination is unsupported.
    if not hasattr(A, "_ensure_spmv_handle"):
        return None
    handle_info = A._ensure_spmv_handle()
    if handle_info is None:
        return None

    _si, _handle, _val_type_id, _exec_q = handle_info
    return _CachedSpMVPair(A, _si)


def _make_system(A, M, x0, b):
    """Make a linear system Ax = b

    Args:
        A (dpnp.ndarray or dpnpx.scipy.sparse.spmatrix or
            dpnpx.scipy.sparse.LinearOperator): sparse or dense matrix.
        M (dpnp.ndarray or dpnpx.scipy.sparse.spmatrix or
            dpnpx.scipy.sparse.LinearOperator): preconditioner.
        x0 (dpnp.ndarray): initial guess to iterative method.
        b (dpnp.ndarray): right hand side.

    Returns:
        tuple:
            It returns (A, M, x, b).
            A (LinaerOperator): matrix of linear system
            M (LinearOperator): preconditioner
            x (dpnp.ndarray): initial guess
            b (dpnp.ndarray): right hand side.
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


# pylint: disable-next=too-many-locals,too-many-statements
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
    info : int
        ``info`` follows the SciPy / CuPy contract:

          * ``info == 0``           : converged successfully
          * ``info > 0``            : did not converge; value is the
                                       iteration count at which the
                                       solver stopped (equals
                                       ``maxiter`` when the iteration
                                       budget was exhausted, or the
                                       iteration index when a numerical
                                       breakdown short-circuited the
                                       loop).
          * ``info < 0``            : reserved for illegal-input
                                       errors; not produced by this
                                       implementation (illegal inputs
                                       raise ``ValueError`` instead).

        Previous versions of this routine returned ``-1`` for an
        ``rz``/``pAp`` breakdown, which violated the SciPy contract
        and broke user code that branched on ``info > 0``.
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

    # rz is kept as a 0-D dpnp array on device throughout the loop;
    # the only time we transfer it to the host is the initial
    # breakdown guard below (matches the CuPy contract -- a zero
    # initial preconditioned residual means we are already at the
    # solution and there is nothing further to do).
    rz = dpnp.real(dpnp.vdot(r, z))
    if float(dpnp.abs(rz)) < rhotol:
        return x, 0

    info = maxiter
    k = 0
    # Per-iter sync count: 1 (rnorm convergence check). The pAp and
    # rz_new breakdown checks are intentionally not transferred to
    # the host; IEEE-754 inf / NaN propagation through alpha = rz/pAp
    # makes pathological values poison the next residual norm, which
    # the single sync below detects via the `not isfinite(rnorm_host)`
    # branch. Mirrors CuPy / cuBLAS-style CG which also dispatches
    # one nrm2 + comparison per iteration.
    for k in range(maxiter):
        rnorm = dpnp.linalg.norm(r)
        rnorm_host = float(rnorm)
        if rnorm_host <= atol_eff_host:
            info = 0
            break
        if not numpy.isfinite(rnorm_host):
            # IEEE-propagated breakdown: pAp or rz collapsed in the
            # previous iteration, poisoning r via alpha=inf/NaN. The
            # current iterate is the best estimate we have; report
            # info > 0 per SciPy contract.
            info = k + 1
            break

        Ap = A_op.matvec(p)
        pAp = dpnp.real(dpnp.vdot(p, Ap))  # 0-D, stays on device

        # No sync on pAp -- division by a near-zero pAp will produce
        # alpha = inf/NaN, propagated below into r and caught by the
        # rnorm_host check at the top of the next iteration.
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        if callback is not None:
            callback(x)

        z = M_op.matvec(r)
        rz_new = dpnp.real(dpnp.vdot(r, z))

        # No sync on rz_new either; near-zero rz_new likewise yields
        # beta = inf/NaN and is caught at the next loop entry.
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new
    else:
        info = maxiter

    return x, int(info)


# pylint: disable-next=too-many-locals,too-many-statements,too-many-branches
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
    # b_norm is a 0-D device tensor; cast to host once so the
    # subsequent comparisons / atol arithmetic are pure-host floats
    # and do not trigger implicit __bool__ syncs every iteration.
    b_norm = float(dpnp.linalg.norm(b))
    if b_norm == 0.0:
        return b, 0
    atol = max(float(atol), rtol * b_norm)

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

    # Krylov basis V is F-ordered so column slices V[:, :k] are
    # F-contiguous USM views, a precondition of the bi._gemv_alpha_beta
    # binding used inside _make_compute_hu.
    V = dpnp.empty((n, restart), dtype=dtype, sycl_queue=queue, order="F")
    # H is F-ordered for the same reason: compute_hu writes Hessenberg
    # column slices H[:j+1, j] in-place via the gemv output pointer.
    # An RHS of length restart+1 is built on the host (e_host) because
    # we run the small (restart+1) x restart least-squares on the host
    # every restart -- the device-side SVD launch overhead dominates
    # for this size class on Intel GPUs, matching CuPy's CPU choice.
    H = dpnp.zeros(
        (restart + 1, restart), dtype=dtype, sycl_queue=queue, order="F"
    )

    compute_hu = _make_compute_hu(V, H)

    np_dtype = _np_dtype(dtype)
    e_host = numpy.zeros(restart + 1, dtype=np_dtype)

    iters = 0
    # r_norm_host tracks the latest residual norm as a Python float so
    # the convergence test and the final maxiter check below operate on
    # host scalars (one explicit sync per restart, not an implicit one
    # per comparison).
    r_norm_host = numpy.inf
    while True:
        mx = psolve(x)
        r = b - matvec(mx)
        r_norm = dpnp.linalg.norm(r)
        r_norm_host = float(r_norm)

        if callback_type == "x":
            callback(mx)
        elif callback_type == "pr_norm" and iters > 0:
            # b_norm is already host; r_norm_host / b_norm stays on host.
            callback(r_norm_host / b_norm)

        if r_norm_host <= atol or iters >= maxiter:
            break

        # Initialise the Arnoldi basis with the (normalised) residual.
        # Writing V[:, 0] in one slice is a contiguous USM-to-USM copy
        # of length n; same shape as CuPy's V[:, 0] = v.
        v = r / r_norm
        V[:, 0] = v
        # Clear the Hessenberg column data the lstsq will read this
        # restart. Only the upper (j+1) entries per column are written
        # by compute_hu; without this reset stale values from the
        # previous restart would leak into the system.
        H[:] = 0
        # RHS for the Hessenberg system is r_norm * e_1; the rest of
        # e_host stays zero from the numpy.zeros allocation above.
        e_host[0] = r_norm_host
        if iters > 0:
            # Clear stale tail from previous restart in case maxiter
            # exceeds restart and we re-enter with a non-zero e_host[1].
            e_host[1:] = 0

        # Arnoldi iteration
        last_j = restart - 1
        for j in range(restart):
            z = psolve(v)
            u = matvec(z)
            # compute_hu writes H[:j+1, j] in-place and returns the
            # orthogonalised u. No h temporary, no tmp buffer, two
            # oneMKL gemv calls per Arnoldi step.
            u = compute_hu(u, j)
            # H[j+1, j] = ||u||  -- one device norm, one slice store.
            # Stored as a device 0-D scalar; we only sync if we need
            # to read its value for the next v normalisation.
            h_norm = dpnp.linalg.norm(u)
            H[j + 1, j] = h_norm
            if j < last_j:
                # Normalise u into the next Krylov vector and store it
                # in V. The single in-place store V[:, j+1] = v writes
                # a contiguous column slice with a unit-stride layout.
                v = u / h_norm
                V[:, j + 1] = v

        # Solve the small Hessenberg least-squares  H y = e  on the
        # host. The matrix is (restart+1) x restart -- typically
        # 21 x 20 -- so the device SVD launch overhead dominates;
        # CuPy makes the same choice and ships y back as a device
        # array. Single host sync per restart, replacing the per-
        # restart device-side lstsq that allocated a workspace and
        # ran a tiny SVD kernel.
        H_host = dpnp.asnumpy(H)
        y_host, *_ = numpy.linalg.lstsq(H_host, e_host, rcond=None)
        y = dpnp.asarray(y_host, sycl_queue=queue)
        x = x + dpnp.dot(V, y)
        iters += restart

    info = 0
    if iters >= maxiter and r_norm_host > atol:
        info = iters

    return mx, info


# pylint: disable-next=too-many-locals,too-many-branches,too-many-statements
def minres(
    A,
    b,
    x0: dpnp.ndarray | None = None,
    *,
    rtol: float = 1e-5,
    shift: float = 0.0,
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
    # recurrence -- and the < 0 / == 0 guards below would each trigger
    # an implicit __bool__ sync of their own.
    beta1 = float(dpnp.inner(r1, y))

    if beta1 < 0:
        raise ValueError("indefinite preconditioner")
    if beta1 == 0:
        return (x, 0)

    beta1 = numpy.sqrt(beta1)

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


def _make_compute_hu(V, H):
    """Factory for the GMRES Arnoldi inner step on Intel GPU.

    Returns a closure ``compute_hu(u, j) -> u`` that performs
    classical Gram-Schmidt orthogonalisation of ``u`` against the
    first ``j+1`` columns of ``V`` and writes the projection
    coefficients into column ``j`` of ``H``:

        h = V[:, :j+1].conj().T @ u
        H[:j+1, j] = h
        u = u - V[:, :j+1] @ h

    Both calls are dispatched as single oneMKL ``gemv`` kernels via
    the ``bi._gemv_alpha_beta`` binding:

      * Pass 1 (project) -- ``gemv(transpose=True, alpha=1, beta=0)``
        with the *output* pointing at the Hessenberg column slice
        ``H[:j+1, j]``. No temporary ``h`` buffer is allocated; the
        result lands directly in the matrix.
      * Pass 2 (subtract) -- ``gemv(transpose=False, alpha=-1, beta=1)``
        with input ``H[:j+1, j]`` and in-place output ``u``. No
        temporary ``tmp`` buffer; the AXPY-style update is fused
        into the gemv kernel.

    For complex matrices, pass 1 with ``transpose=True`` returns
    ``V^T u`` (not the Hermitian ``V^H u`` we need). We patch this
    up with an in-place conjugate on the Hessenberg column slice --
    j+1 scalar ops, negligible compared with the m*(j+1) gemv.

    Parameters
    ----------
    V : dpnp.ndarray
        Krylov basis of shape ``(n, restart)``, must be F-contiguous.
    H : dpnp.ndarray
        Hessenberg matrix of shape ``(restart+1, restart)``, must be
        F-contiguous so column slices ``H[:k, j]`` are unit-stride
        contiguous USM views the C binding can write into.

    Returns
    -------
    closure : callable
        ``compute_hu(u, j) -> u`` -- updates ``H[:j+1, j]`` in place
        and returns the orthogonalised ``u``.
    """
    if V.ndim != 2 or not V.flags.f_contiguous:
        raise ValueError(
            "_make_compute_hu: V must be a 2-D column-major (F-order) "
            "dpnp array"
        )
    if H.ndim != 2 or not H.flags.f_contiguous:
        raise ValueError(
            "_make_compute_hu: H must be a 2-D column-major (F-order) "
            "dpnp array so column slices are unit-stride USM views"
        )
    if V.sycl_queue != H.sycl_queue:
        raise ValueError(
            "_make_compute_hu: V and H must share the same SYCL queue"
        )

    exec_q = V.sycl_queue
    dtype = V.dtype
    is_cpx = dpnp.issubdtype(dtype, dpnp.complexfloating)

    def compute_hu(u, j):
        Vj = V[:, : j + 1]
        h_slice = H[: j + 1, j]  # length-(j+1) F-contig column slice

        Vj_usm = dpnp.get_usm_ndarray(Vj)
        u_usm = dpnp.get_usm_ndarray(u)
        h_usm = dpnp.get_usm_ndarray(h_slice)

        _manager = dpu.SequentialOrderManager[exec_q]

        # Pass 1: H[:j+1, j] = Vj^T @ u   (alpha=1, beta=0 implicit)
        # Writes the projection coefficients directly into the
        # Hessenberg column -- no h temporary, no slice-assign copy.
        # pylint: disable-next=protected-access
        ht1, ev1 = bi._gemv_alpha_beta(
            exec_q,
            Vj_usm,
            u_usm,
            h_usm,
            transpose=True,
            alpha=1.0,
            beta=0.0,
            depends=_manager.submitted_events,
        )
        _manager.add_event_pair(ht1, ev1)

        if is_cpx:
            # Need V^H, but we only have V^T from the gemv. Conjugate
            # the (j+1) scalars in place. Trivial cost relative to the
            # n*(j+1) gemv we just dispatched.
            h_slice[...] = dpnp.conj(h_slice)
            # Re-fetch USM handle after in-place update (still the
            # same backing buffer; the handle itself is unchanged but
            # making the dependency explicit keeps the sequential-
            # order manager happy).
            h_usm = dpnp.get_usm_ndarray(h_slice)

        # Pass 2: u = -Vj @ H[:j+1, j] + 1 * u   (alpha=-1, beta=1)
        # Fused AXPY-gemv -- single oneMKL kernel, no tmp buffer.
        # pylint: disable-next=protected-access
        ht2, ev2 = bi._gemv_alpha_beta(
            exec_q,
            Vj_usm,
            h_usm,
            u_usm,
            transpose=False,
            alpha=-1.0,
            beta=1.0,
            depends=_manager.submitted_events,
        )
        _manager.add_event_pair(ht2, ev2)

        return u

    return compute_hu
