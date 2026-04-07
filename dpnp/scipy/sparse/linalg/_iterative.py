# Copyright (c) 2023 - 2025, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of Intel Corporation nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
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
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Iterative sparse linear solvers for dpnp -- pure GPU/SYCL implementation.

All computation stays on the device (USM/oneMKL).  There is NO host-dispatch
fallback: transferring data to the CPU for small systems defeats the purpose
of keeping a live computation on GPU memory.

Solver coverage
---------------
cg     : Conjugate Gradient (Hermitian positive definite)
gmres  : Restarted GMRES (general non-symmetric)
minres : MINRES (symmetric possibly indefinite)

All signatures match cupyx.scipy.sparse.linalg (CuPy v14.0.1) and
scipy.sparse.linalg, using ``rtol`` as the primary tolerance keyword
(``tol`` is accepted as a deprecated alias for backward compatibility).

Algorithm notes
---------------
* b == 0 early-exit (return x0 or zeros with info=0).
* Breakdown detection via machine-epsilon rhotol (CG, GMRES).
* atol normalisation: atol_eff = max(atol, rtol * ||b||).
* dtype promotion: A.dtype preferred when in fdFD; otherwise b.dtype
  promoted to float64/complex128 (CuPy v14 compatible).
* Preconditioner M supported for all three solvers; shape is validated
  against A inside _make_system; fast CSR SpMV injected for M too.
* GMRES: Givens-rotation Hessenberg QR on CPU scalars; all matvec and
  inner-product work stays on device.  V basis pre-allocated as
  (n, restart+1) Fortran-order for coalesced column access; no per-
  iteration stack().  callback_type 'x', 'pr_norm', and 'legacy' all
  implemented.  Happy breakdown detected via h_{j+1,j} < rhotol.
* MINRES: native Paige-Saunders (1975) recurrence -- no scipy round-trip.
  Full stopping battery: rnorm <= atol_eff, test1 (relative residual),
  test2 (residual in range of A), Acond (condition number estimate) --
  matches CuPy v14 / SciPy minres.py reference.
  Preconditioner SPD check: raw inner product tested for negativity
  BEFORE sqrt so the guard fires (abs() removed -- was dead code).
  Stagnation floor 10*eps; convergence check precedes stagnation check.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as _np
import dpnp as _dpnp

from ._interface import IdentityOperator, LinearOperator, aslinearoperator


# ---------------------------------------------------------------------------
# oneMKL sparse SpMV hook
# ---------------------------------------------------------------------------

try:
    from dpnp.backend.extensions.sparse import _sparse_impl as _si
    _HAS_SPARSE_IMPL = True
except ImportError:
    _si = None
    _HAS_SPARSE_IMPL = False

_SUPPORTED_DTYPES = frozenset("fdFD")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _np_dtype(dp_dtype) -> _np.dtype:
    """Normalise any dtype-like (dpnp type, numpy type, string) to np.dtype.

    dpnp dtype objects (e.g. dpnp.float64) are Python type objects with no
    .char attribute.  np.dtype() accepts all of them correctly.
    """
    return _np.dtype(dp_dtype)


def _check_dtype(dtype, name: str) -> None:
    if _np_dtype(dtype).char not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"{name} has unsupported dtype {dtype}; "
            "only float32, float64, complex64, complex128 are accepted."
        )


def _make_fast_matvec(A):
    """Return device-side CSR SpMV callable, or None."""
    try:
        from dpnp.scipy import sparse as _sp
        if not (_sp.issparse(A) and A.format == "csr"):
            return None
    except (ImportError, AttributeError):
        return None

    if _HAS_SPARSE_IMPL:
        indptr  = A.indptr
        indices = A.indices
        data    = A.data
        nrows   = int(A.shape[0])
        ncols   = int(A.shape[1])
        nnz     = int(data.shape[0])
        exec_q  = data.sycl_queue

        def _csr_spmv(x: _dpnp.ndarray) -> _dpnp.ndarray:
            y = _dpnp.zeros(nrows, dtype=data.dtype, sycl_queue=exec_q)
            _, ev = _si._sparse_gemv(
                exec_q, 0, 1.0, indptr, indices, data, x,
                0.0, y, nrows, ncols, nnz, [],
            )
            ev.wait()
            return y

        return _csr_spmv

    return lambda x: A.dot(x)


def _make_system(A, M, x0, b):
    """Validate and prepare (A_op, M_op, x, b, dtype) on device.

    dtype promotion follows CuPy v14 rules: A.dtype is used when it is in
    {f,d,F,D}; otherwise b.dtype is promoted to float64 (real) or
    complex128 (complex).  Preconditioners are always accepted and validated.
    """
    A_op = aslinearoperator(A)
    if A_op.shape[0] != A_op.shape[1]:
        raise ValueError("A must be a square operator")
    n = A_op.shape[0]

    b = _dpnp.asarray(b).reshape(-1)
    if b.shape[0] != n:
        raise ValueError(
            f"b length {b.shape[0]} does not match operator dimension {n}"
        )

    # Dtype promotion: prefer A.dtype; fall back via b.dtype.
    if A_op.dtype is not None and _np_dtype(A_op.dtype).char in _SUPPORTED_DTYPES:
        dtype = A_op.dtype
    elif _dpnp.issubdtype(b.dtype, _dpnp.complexfloating):
        dtype = _dpnp.complex128
    else:
        dtype = _dpnp.float64

    b = b.astype(dtype, copy=False)
    _check_dtype(b.dtype, "b")

    if x0 is None:
        x = _dpnp.zeros(n, dtype=dtype)
    else:
        x = _dpnp.asarray(x0, dtype=dtype).reshape(-1)
        if x.shape[0] != n:
            raise ValueError(f"x0 length {x.shape[0]} != n={n}")

    if M is None:
        M_op = IdentityOperator((n, n), dtype=dtype)
    else:
        M_op = aslinearoperator(M)
        if M_op.shape != A_op.shape:
            raise ValueError(
                f"preconditioner shape {M_op.shape} != operator shape {A_op.shape}"
            )
        fast_mv_M = _make_fast_matvec(M)
        if fast_mv_M is not None:
            _orig_M = M_op
            class _FastMOp(LinearOperator):
                def __init__(self):
                    super().__init__(_orig_M.dtype, _orig_M.shape)
                def _matvec(self, x):  return fast_mv_M(x)
                def _rmatvec(self, x): return _orig_M.rmatvec(x)
            M_op = _FastMOp()

    # Inject fast CSR SpMV for A if available.
    fast_mv = _make_fast_matvec(A)
    if fast_mv is not None:
        _orig = A_op
        class _FastOp(LinearOperator):
            def __init__(self):
                super().__init__(_orig.dtype, _orig.shape)
            def _matvec(self, x):  return fast_mv(x)
            def _rmatvec(self, x): return _orig.rmatvec(x)
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


# ---------------------------------------------------------------------------
# Conjugate Gradient
# ---------------------------------------------------------------------------

def cg(
    A,
    b,
    x0: Optional[_dpnp.ndarray] = None,
    *,
    rtol: float = 1e-5,
    tol: Optional[float] = None,
    maxiter: Optional[int] = None,
    M=None,
    callback: Optional[Callable] = None,
    atol=None,
) -> Tuple[_dpnp.ndarray, int]:
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

    bnrm = float(_dpnp.linalg.norm(b))
    if bnrm == 0.0:
        return _dpnp.zeros_like(b), 0

    atol_eff = _get_atol(bnrm, atol=atol, rtol=rtol)
    if maxiter is None:
        maxiter = n * 10

    rhotol = float(_np.finfo(_np_dtype(dtype)).eps ** 2)

    # Use `x0 is not None` rather than `_dpnp.any(x)` -- dpnp arrays raise
    # AmbiguousTruth when used as Python booleans.
    r  = b - A_op.matvec(x) if x0 is not None else b.copy()
    z  = M_op.matvec(r)
    p  = _dpnp.array(z, copy=True)
    rz = float(_dpnp.real(_dpnp.vdot(r, z)))

    if abs(rz) < rhotol:
        return x, 0

    info = maxiter
    for _ in range(maxiter):
        if float(_dpnp.linalg.norm(r)) <= atol_eff:
            info = 0
            break

        Ap  = A_op.matvec(p)
        pAp = float(_dpnp.real(_dpnp.vdot(p, Ap)))
        if abs(pAp) < rhotol:
            info = -1
            break

        alpha  = rz / pAp
        x      = x + alpha * p
        r      = r - alpha * Ap

        if callback is not None:
            callback(x)

        z      = M_op.matvec(r)
        rz_new = float(_dpnp.real(_dpnp.vdot(r, z)))
        if abs(rz_new) < rhotol:
            info = 0
            break
        p  = z + (rz_new / rz) * p
        rz = rz_new
    else:
        info = maxiter

    return x, int(info)


# ---------------------------------------------------------------------------
# Restarted GMRES
# ---------------------------------------------------------------------------

def gmres(
    A,
    b,
    x0: Optional[_dpnp.ndarray] = None,
    *,
    rtol: float = 1e-5,
    tol: Optional[float] = None,
    restart: Optional[int] = None,
    maxiter: Optional[int] = None,
    M=None,
    callback: Optional[Callable] = None,
    atol=None,
    callback_type: Optional[str] = None,
) -> Tuple[_dpnp.ndarray, int]:
    """Restarted GMRES -- pure dpnp/oneMKL, general non-symmetric A.

    Parameters
    ----------
    A             : array_like or LinearOperator -- (n, n)
    b             : array_like -- right-hand side (n,)
    x0            : array_like, optional
    rtol          : float -- relative tolerance (default 1e-5)
    tol           : float, optional -- deprecated alias for rtol
    restart       : int, optional -- Krylov subspace size (default min(20,n))
    maxiter       : int, optional -- max outer restart cycles (default max(n,1))
    M             : LinearOperator or array_like, optional -- preconditioner
    callback      : callable, optional
    atol          : float, optional
    callback_type : {None, 'x', 'pr_norm', 'legacy'}
                    None / 'x' / 'legacy' -- callback(xk) after each restart
                    'pr_norm'             -- callback(||r||/||b||) per restart

    Returns
    -------
    x    : dpnp.ndarray
    info : int  0=converged  >0=iterations used  -1=breakdown
    """
    if tol is not None:
        rtol = tol

    if callback_type not in (None, "x", "pr_norm", "legacy"):
        raise ValueError(
            "callback_type must be None, 'x', 'pr_norm', or 'legacy'"
        )
    if callback is not None and callback_type is None:
        callback_type = "x"

    A_op, M_op, x, b, dtype = _make_system(A, M, x0, b)
    n = b.shape[0]

    bnrm = float(_dpnp.linalg.norm(b))
    if bnrm == 0.0:
        return _dpnp.zeros_like(b), 0

    atol_eff = _get_atol(bnrm, atol=atol, rtol=rtol)
    if restart is None: restart = min(20, n)
    if maxiter is None: maxiter = max(n, 1)
    restart = int(restart)
    maxiter = int(maxiter)

    is_cpx  = _dpnp.issubdtype(dtype, _dpnp.complexfloating)
    H_dtype = _np.complex128 if is_cpx else _np.float64
    rhotol  = float(_np.finfo(_np_dtype(dtype)).eps ** 2)

    total_iters = 0
    info        = maxiter

    for _outer in range(maxiter):
        r    = M_op.matvec(b - A_op.matvec(x))
        beta = float(_dpnp.linalg.norm(r))
        if beta == 0.0 or beta <= atol_eff:
            info = 0
            break

        # Pre-allocate V Fortran-order: columns V[:,j] are contiguous
        # in device memory, avoiding strided (non-coalesced) access.
        V = _dpnp.zeros((n, restart + 1), dtype=dtype, order='F')
        V[:, 0] = r / beta

        H_np  = _np.zeros((restart + 1, restart), dtype=H_dtype)
        cs_np = _np.zeros(restart, dtype=H_dtype)
        sn_np = _np.zeros(restart, dtype=H_dtype)
        g_np  = _np.zeros(restart + 1, dtype=H_dtype)
        g_np[0] = beta

        j_final = 0
        happy   = False

        for j in range(restart):
            total_iters += 1

            w = M_op.matvec(A_op.matvec(V[:, j]))

            # Modified Gram-Schmidt: one device-to-host transfer per step
            # (pulls (j+1,) h vector via .asnumpy()) instead of j scalars.
            h_dp = _dpnp.dot(_dpnp.conj(V[:, :j + 1].T), w)
            h_np = h_dp.asnumpy()
            w    = w - _dpnp.dot(V[:, :j + 1],
                                 _dpnp.asarray(h_np, dtype=dtype))

            h_j1 = float(_dpnp.linalg.norm(w))

            H_np[:j + 1, j] = h_np
            H_np[j + 1,  j] = h_j1

            # Apply previous Givens rotations to column j of H.
            for i in range(j):
                tmp             =  cs_np[i] * H_np[i, j] + sn_np[i] * H_np[i + 1, j]
                H_np[i + 1, j] = -_np.conj(sn_np[i]) * H_np[i, j] + cs_np[i] * H_np[i + 1, j]
                H_np[i,     j] =  tmp

            h_jj  = H_np[j,     j]
            h_j1j = H_np[j + 1, j]
            denom = _np.sqrt(_np.abs(h_jj) ** 2 + _np.abs(h_j1j) ** 2)
            if denom < rhotol:
                info    = -1
                happy   = True
                j_final = j
                break
            cs_np[j]       = h_jj  / denom
            sn_np[j]       = h_j1j / denom
            H_np[j,     j] = cs_np[j] * h_jj + sn_np[j] * h_j1j
            H_np[j + 1, j] = 0.0
            g_np[j + 1]    = -_np.conj(sn_np[j]) * g_np[j]
            g_np[j]        =  cs_np[j] * g_np[j]

            res_norm = abs(g_np[j + 1])

            if h_j1 < rhotol:       # happy breakdown
                j_final = j
                happy   = True
                if res_norm <= atol_eff:
                    info = 0
                break

            if res_norm <= atol_eff:
                j_final = j
                info    = 0
                happy   = True
                break

            if j + 1 < restart:
                V[:, j + 1] = w / h_j1
            j_final = j

        # Back-substitution on upper-triangular H_np (already on CPU).
        k    = j_final + 1
        y_np = _np.zeros(k, dtype=H_dtype)
        for i in range(k - 1, -1, -1):
            y_np[i] = g_np[i]
            for ll in range(i + 1, k):
                y_np[i] -= H_np[i, ll] * y_np[ll]
            if abs(H_np[i, i]) < rhotol:
                y_np[i] = 0.0
            else:
                y_np[i] /= H_np[i, i]

        # Solution update: device matmul, no host round-trip.
        x = x + _dpnp.dot(V[:, :k], _dpnp.asarray(y_np, dtype=dtype))

        res_norm = float(_dpnp.linalg.norm(M_op.matvec(b - A_op.matvec(x))))

        if callback is not None:
            if callback_type in ("x", "legacy"):
                callback(x)
            elif callback_type == "pr_norm":
                callback(res_norm / bnrm)

        if res_norm <= atol_eff:
            info = 0
            break

        if happy and info != 0:
            break
    else:
        info = total_iters

    return x, int(info)


# ---------------------------------------------------------------------------
# MINRES -- Paige-Saunders recurrence, pure dpnp / oneMKL
# ---------------------------------------------------------------------------

def minres(
    A,
    b,
    x0: Optional[_dpnp.ndarray] = None,
    *,
    shift: float = 0.0,
    rtol: float = 1e-5,
    tol: Optional[float] = None,
    maxiter: Optional[int] = None,
    M=None,
    callback: Optional[Callable] = None,
    check: bool = False,
    atol=None,
) -> Tuple[_dpnp.ndarray, int]:
    """MINRES for symmetric (possibly indefinite) A -- pure dpnp/oneMKL.

    Implements Paige-Saunders (1975) MINRES via Lanczos tridiagonalisation
    with Givens QR.  All matvec, dot-products, and vector updates run on
    device; only scalar recurrence coefficients are on CPU.

    Stopping criteria (matches CuPy v14 / SciPy minres.py reference):
      1. rnorm       <= atol_eff                  (absolute residual)
      2. test1       <= rtol  where test1 = ||r|| / (||A|| * ||x||)
      3. test2       <= rtol  where test2 = ||Ar_k|| / ||A||
      4. Acond       >= 0.1 / eps                 (ill-conditioned stop)
      5. phi * denom < 10*eps                     (stagnation)
    Convergence (1-4) is always checked before stagnation (5).

    Preconditioner SPD check: the raw inner product <r, M*r> is tested
    for negativity BEFORE sqrt so the guard is live (not dead code as it
    would be if abs() were applied first).

    Parameters
    ----------
    A       : array_like or LinearOperator -- symmetric/Hermitian (n, n)
    b       : array_like -- right-hand side (n,)
    x0      : array_like, optional -- initial guess
    shift   : float -- solve (A - shift*I)x = b
    rtol    : float -- relative tolerance (default 1e-5)
    tol     : float, optional -- deprecated alias for rtol
    maxiter : int, optional -- max iterations (default 5*n)
    M       : LinearOperator, optional -- SPD preconditioner
    callback: callable, optional -- callback(xk) after each step
    check   : bool -- verify A symmetry before iterating
    atol    : float, optional -- absolute tolerance

    Returns
    -------
    x    : dpnp.ndarray
    info : int  0=converged  1=maxiter  2=stagnation
    """
    if tol is not None:
        rtol = tol

    A_op, M_op, x, b, dtype = _make_system(A, M, x0, b)
    n   = b.shape[0]
    eps = float(_np.finfo(_np_dtype(dtype)).eps)

    if maxiter is None:
        maxiter = 5 * n

    bnrm = float(_dpnp.linalg.norm(b))
    if bnrm == 0.0:
        return _dpnp.zeros_like(b), 0

    atol_eff = _get_atol(bnrm, atol=atol, rtol=rtol)

    # ------------------------------------------------------------------
    # Initialise Lanczos: beta1 = sqrt(<r0, M*r0>)
    # Test the raw inner product for negativity BEFORE sqrt so that a
    # non-SPD preconditioner is detected (abs() was removed -- it made
    # this check dead code).
    # ------------------------------------------------------------------
    r1          = b - A_op.matvec(x) if x0 is not None else b.copy()
    y           = M_op.matvec(r1)
    beta1_inner = float(_dpnp.real(_dpnp.vdot(r1, y)))
    if beta1_inner < 0.0:
        raise ValueError(
            "minres: preconditioner M is not positive semi-definite "
            f"(<r, M*r> = {beta1_inner:.6g} < 0)"
        )
    if beta1_inner == 0.0:
        return x, 0
    beta1 = _np.sqrt(beta1_inner)

    if check:
        Ay  = A_op.matvec(y) - shift * y
        lhs = float(_dpnp.linalg.norm(
            Ay - (_dpnp.real(_dpnp.vdot(y, Ay))
                  / _dpnp.real(_dpnp.vdot(y, y))) * y
        ))
        rhs = eps ** 0.5 * float(_dpnp.linalg.norm(Ay))
        if lhs > rhs:
            raise ValueError(
                "minres: A does not appear symmetric/Hermitian; "
                "set check=False to skip this test."
            )

    # ------------------------------------------------------------------
    # Paige-Saunders scalar state (all on CPU)
    # ------------------------------------------------------------------
    beta   = beta1
    oldb   = 0.0
    phibar = beta1
    cs     = -1.0
    sn     =  0.0
    dbar   =  0.0
    epsln  =  0.0

    # State for full stopping battery
    tnorm2 = 0.0
    gmax   = 0.0
    gmin   = _np.finfo(_np_dtype(dtype)).max

    # Solution update vectors (on device)
    w  = _dpnp.zeros(n, dtype=dtype)
    w2 = _dpnp.zeros(n, dtype=dtype)
    r2 = r1.copy()
    v  = y / beta1

    # 10*eps stagnation floor (SciPy minres.py convention).
    stag_eps = 10.0 * eps

    info = 1
    for itr in range(1, maxiter + 1):
        # ------------------------------------------------------------------
        # Lanczos step k
        # ------------------------------------------------------------------
        s  = 1.0 / beta
        v  = y * s
        y  = A_op.matvec(v) - shift * v
        if itr > 1:
            y = y - (beta / oldb) * r1

        alpha = float(_dpnp.real(_dpnp.vdot(v, y)))
        y     = y - (alpha / beta) * r2
        r1    = r2.copy()
        r2    = y.copy()
        y     = M_op.matvec(r2)
        oldb  = beta

        # SPD check on iteration inner product (live guard, no abs()).
        inner_r2y = float(_dpnp.real(_dpnp.vdot(r2, y)))
        if inner_r2y < 0.0:
            raise ValueError(
                "minres: preconditioner M is not positive semi-definite "
                f"(<r, M*r> = {inner_r2y:.6g} < 0 at iteration {itr})"
            )
        beta = _np.sqrt(inner_r2y)

        tnorm2 += alpha ** 2 + oldb ** 2 + beta ** 2

        # ------------------------------------------------------------------
        # QR step: Paige-Saunders two-rotation recurrence
        # ------------------------------------------------------------------
        oldeps = epsln
        delta  = cs * dbar + sn * alpha
        gbar_k = sn * dbar - cs * alpha
        epsln  = sn * beta
        dbar   = -cs * beta

        # root = ||Ar_k|| proxy used for test2
        root   = _np.hypot(gbar_k, dbar)

        gamma  = _np.hypot(gbar_k, beta)
        if gamma == 0.0:
            gamma = eps
        cs     = gbar_k / gamma
        sn     = beta   / gamma

        phi    = cs * phibar
        phibar = sn * phibar

        gmax = max(gmax, gamma)
        gmin = min(gmin, gamma)

        # ------------------------------------------------------------------
        # Solution update: three-term w recurrence (Paige-Saunders SS5)
        # ------------------------------------------------------------------
        denom = 1.0 / gamma
        w_new = (v - oldeps * w - delta * w2) * denom
        x     = x + phi * w_new
        w     = w2
        w2    = w_new

        rnorm = abs(phibar)
        Anorm = _np.sqrt(tnorm2)
        ynorm = float(_dpnp.linalg.norm(x))

        if callback is not None:
            callback(x)

        # Convergence checks run before stagnation so a boundary iteration
        # that satisfies both is reported as converged (info=0).
        if rnorm <= atol_eff:
            info = 0
            break

        if Anorm > 0.0 and ynorm > 0.0:
            if rnorm / (Anorm * ynorm) <= rtol:   # test1
                info = 0
                break

        if Anorm > 0.0:
            if root / Anorm <= rtol:               # test2
                info = 0
                break

        if Anorm > 0.0 and (gmax / gmin) >= 0.1 / eps:  # Acond stop
            info = 0
            break

        if phi * denom < stag_eps:                 # stagnation
            info = 2
            break
    else:
        info = 1

    return x, int(info)
