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

"""Iterative sparse linear solvers for dpnp — pure GPU/SYCL implementation.

All computation stays on the device (USM/oneMKL).  There is NO host-dispatch
fallback: transferring data to the CPU for small systems defeats the purpose
of keeping a live computation on GPU memory.

Solver coverage
---------------
cg     : Conjugate Gradient (Hermitian positive definite)
gmres  : Restarted GMRES (general non-symmetric)
minres : MINRES (symmetric possibly indefinite)

All signatures match cupyx.scipy.sparse.linalg (CuPy v14.0.1) and
scipy.sparse.linalg.

Corner-case coverage
---------------------
* b == 0 early-exit (return x0 or zeros with info=0)
* Breakdown detection via machine-epsilon rhotol (CG, GMRES)
* atol normalisation: atol = max(atol_arg, rtol * ||b||)
* dtype promotion: f/F stay in single, d/D in double (CuPy rules)
* Preconditioner (M != None): raises NotImplementedError for CG and GMRES
  until a full left-preconditioned implementation lands; MINRES supports M.
* GMRES: Givens-rotation Hessenberg QR, allocation-free scalar CPU side;
  all matvec + inner-product work stays on device.
* GMRES: happy breakdown via h_{j+1,j} == 0
* MINRES: native Paige-Saunders (1975) recurrence — no scipy host round-trip.
  QR step uses the exact two-rotation recurrence from SciPy minres.py:
    oldeps = epsln
    delta  = cs * dbar + sn * alpha   # apply previous Givens rotation
    gbar_k = sn * dbar - cs * alpha   # residual for new rotation
    epsln  = sn * beta
    dbar   = -cs * beta
    gamma  = hypot(gbar_k, beta)      # new rotation eliminates beta
  betacheck uses fixed floor eps*beta1 (not a decaying product).
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


def _make_system(A, M, x0, b, *, allow_M: bool = False):
    """Validate and prepare (A_op, M_op, x, b, dtype) on device.

    Parameters
    ----------
    allow_M : bool
        If False (default) and M is not None, raise NotImplementedError.
        Set True only for solvers that fully support preconditioning (minres).
    """
    # ------------------------------------------------------------------
    # Preconditioner guard — must come BEFORE aslinearoperator so that
    # passing a dpnp array as M still raises rather than silently wrapping.
    # ------------------------------------------------------------------
    if M is not None and not allow_M:
        raise NotImplementedError(
            "Preconditioner M is not yet supported for this solver. "
            "Pass M=None or use minres which supports M."
        )

    A_op = aslinearoperator(A)
    if A_op.shape[0] != A_op.shape[1]:
        raise ValueError("A must be a square operator")
    n = A_op.shape[0]

    b = _dpnp.asarray(b).reshape(-1)
    if b.shape[0] != n:
        raise ValueError(
            f"b length {b.shape[0]} does not match operator dimension {n}"
        )

    # Dtype promotion — matches CuPy v14.0.1 rules
    if _dpnp.issubdtype(b.dtype, _dpnp.complexfloating):
        dtype = _dpnp.complex128
    else:
        dtype = _dpnp.float64
    if A_op.dtype is not None and _np_dtype(A_op.dtype).char in "fF":
        dtype = _dpnp.complex64 if _np_dtype(A_op.dtype).char == "F" else _dpnp.float32

    b = b.astype(dtype, copy=False)
    _check_dtype(b.dtype, "b")

    if x0 is None:
        x = _dpnp.zeros(n, dtype=dtype)
    else:
        x = _dpnp.asarray(x0, dtype=dtype).reshape(-1)
        if x.shape[0] != n:
            raise ValueError(f"x0 length {x.shape[0]} != n={n}")

    M_op = IdentityOperator((n, n), dtype=dtype) if M is None else aslinearoperator(M)

    # Inject fast CSR SpMV if available
    fast_mv = _make_fast_matvec(A)
    if fast_mv is not None:
        orig = A_op
        class _FastOp(LinearOperator):
            def __init__(self):
                super().__init__(orig.dtype, orig.shape)
            def _matvec(self, x):  return fast_mv(x)
            def _rmatvec(self, x): return orig.rmatvec(x)
        A_op = _FastOp()

    return A_op, M_op, x, b, dtype


def _get_atol(name: str, b_norm: float, atol, rtol: float) -> float:
    """Absolute stopping tolerance: max(atol, rtol*||b||), mirroring SciPy."""
    if atol == "legacy" or atol is None:
        atol = 0.0
    atol = float(atol)
    if atol < 0:
        raise ValueError(
            f"'{name}' called with invalid atol={atol!r}; "
            "atol must be a real, non-negative number."
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
    tol: float = 1e-5,
    maxiter: Optional[int] = None,
    M=None,
    callback: Optional[Callable] = None,
    atol=None,
) -> Tuple[_dpnp.ndarray, int]:
    """Conjugate Gradient — pure dpnp/oneMKL, Hermitian positive definite A.

    Parameters
    ----------
    A       : array_like or LinearOperator — HPD (n, n)
    b       : array_like — right-hand side (n,)
    x0      : array_like, optional — initial guess
    tol     : float — relative tolerance (default 1e-5)
    maxiter : int, optional — max iterations (default 10*n)
    M       : None — preconditioner (unsupported; pass None)
    callback: callable, optional — callback(xk) after each iteration
    atol    : float, optional — absolute tolerance

    Returns
    -------
    x    : dpnp.ndarray
    info : int  0=converged  >0=maxiter  -1=breakdown
    """
    # allow_M=False: NotImplementedError raised inside _make_system if M!=None
    A_op, M_op, x, b, dtype = _make_system(A, M, x0, b, allow_M=False)
    n = b.shape[0]

    bnrm = float(_dpnp.linalg.norm(b))
    if bnrm == 0.0:
        return _dpnp.zeros_like(b), 0

    atol_eff = _get_atol("cg", bnrm, atol, tol)
    if maxiter is None:
        maxiter = n * 10

    rhotol = float(_np.finfo(_np_dtype(dtype)).eps ** 2)

    # FIX: use `x0 is not None` to detect a non-trivial initial guess instead
    # of `_dpnp.any(x)` which returns a dpnp array and raises AmbiguousTruth.
    r  = b - A_op.matvec(x) if x0 is not None else b.copy()
    z  = M_op.matvec(r)
    p  = _dpnp.array(z, copy=True)
    rz = float(_dpnp.vdot(r, z).real)

    if abs(rz) < rhotol:
        return x, 0

    info = maxiter
    for _ in range(maxiter):
        if float(_dpnp.linalg.norm(r)) <= atol_eff:
            info = 0
            break

        Ap  = A_op.matvec(p)
        pAp = float(_dpnp.vdot(p, Ap).real)
        if abs(pAp) < rhotol:
            info = -1
            break

        alpha  = rz / pAp
        x      = x + alpha * p
        r      = r - alpha * Ap

        if callback is not None:
            callback(x)

        z      = M_op.matvec(r)
        rz_new = float(_dpnp.vdot(r, z).real)
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
    tol: float = 1e-5,
    restart: Optional[int] = None,
    maxiter: Optional[int] = None,
    M=None,
    callback: Optional[Callable] = None,
    atol=None,
    callback_type: Optional[str] = None,
) -> Tuple[_dpnp.ndarray, int]:
    """Restarted GMRES — pure dpnp/oneMKL, general non-symmetric A.

    Parameters
    ----------
    A             : array_like or LinearOperator — (n, n)
    b             : array_like — right-hand side (n,)
    x0            : array_like, optional
    tol           : float — relative tolerance (default 1e-5)
    restart       : int, optional — Krylov subspace size (default min(20,n))
    maxiter       : int, optional — max outer restart cycles (default n)
    M             : None — preconditioner (unsupported; pass None)
    callback      : callable, optional
    atol          : float, optional
    callback_type : {'x', 'pr_norm', 'legacy', None}

    Returns
    -------
    x    : dpnp.ndarray
    info : int  0=converged  >0=iterations used  -1=breakdown
    """
    if callback_type not in (None, "x", "pr_norm", "legacy"):
        raise ValueError(
            "callback_type must be None, 'x', 'pr_norm', or 'legacy'"
        )
    if callback_type == "pr_norm":
        raise NotImplementedError(
            "callback_type='pr_norm' is not yet implemented in dpnp gmres."
        )

    # allow_M=False: NotImplementedError raised inside _make_system if M!=None
    A_op, M_op, x, b, dtype = _make_system(A, M, x0, b, allow_M=False)
    n = b.shape[0]

    bnrm = float(_dpnp.linalg.norm(b))
    if bnrm == 0.0:
        return _dpnp.zeros_like(b), 0

    atol_eff = _get_atol("gmres", bnrm, atol, tol)
    if restart is None: restart = min(20, n)
    if maxiter is None: maxiter = n
    restart = int(restart)
    maxiter = int(maxiter)

    if callback_type is None and callback is not None:
        callback_type = "x"

    is_cpx  = _dpnp.issubdtype(dtype, _dpnp.complexfloating)
    H_dtype = _np.complex128 if is_cpx else _np.float64
    rhotol  = float(_np.finfo(_np_dtype(dtype)).eps ** 2)

    total_iters = 0
    info        = maxiter

    for _outer in range(maxiter):
        # FIX: use x0 is not None for the outer-loop residual too; after the
        # first restart x has been updated so always compute the residual.
        r    = M_op.matvec(b - A_op.matvec(x))
        beta = float(_dpnp.linalg.norm(r))
        if beta == 0.0 or beta <= atol_eff:
            info = 0
            break

        V_cols = [r / beta]
        H_np   = _np.zeros((restart + 1, restart), dtype=H_dtype)
        cs_np  = _np.zeros(restart, dtype=H_dtype)
        sn_np  = _np.zeros(restart, dtype=H_dtype)
        g_np   = _np.zeros(restart + 1, dtype=H_dtype)
        g_np[0] = beta

        j_final = 0
        happy   = False

        for j in range(restart):
            total_iters += 1

            w     = M_op.matvec(A_op.matvec(V_cols[j]))
            V_mat = _dpnp.stack(V_cols, axis=1)

            # FIX: dpnp arrays have no .conj() method on transpose results;
            # use the module-level _dpnp.conj() instead.
            h_dp  = _dpnp.dot(_dpnp.conj(V_mat.T), w)
            h_np  = _dpnp.asnumpy(h_dp)  # FIX: asnumpy is a module-level fn, not a method
            w     = w - _dpnp.dot(V_mat, _dpnp.asarray(h_np, dtype=dtype))

            # FIX: float(_dpnp.linalg.norm(...)) — norm returns a 0-d dpnp
            # array; float() extracts the scalar correctly without .asnumpy().
            h_j1  = float(_dpnp.linalg.norm(w))

            # FIX: always assign h_np directly (it is already the right dtype
            # for both real and complex cases); avoid the .real strip which
            # would drop the imaginary component for complex Hessenberg entries.
            H_np[:j + 1, j] = h_np
            H_np[j + 1,  j] = h_j1

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

            V_cols.append(w / h_j1)
            j_final = j

        k    = j_final + 1
        y_np = _np.zeros(k, dtype=H_dtype)
        for i in range(k - 1, -1, -1):
            y_np[i] = g_np[i]
            for l in range(i + 1, k):
                y_np[i] -= H_np[i, l] * y_np[l]
            if abs(H_np[i, i]) < rhotol:
                y_np[i] = 0.0
            else:
                y_np[i] /= H_np[i, i]

        V_k = _dpnp.stack(V_cols[:k], axis=1)
        x   = x + _dpnp.dot(V_k, _dpnp.asarray(y_np, dtype=dtype))

        res_norm = float(_dpnp.linalg.norm(M_op.matvec(b - A_op.matvec(x))))

        if callback is not None:
            callback(x if callback_type in ("x", "legacy") else res_norm)

        if res_norm <= atol_eff:
            info = 0
            break

        if happy and info != 0:
            break
    else:
        info = total_iters

    return x, int(info)


# ---------------------------------------------------------------------------
# MINRES — Paige-Saunders recurrence, pure dpnp / oneMKL
# ---------------------------------------------------------------------------

def minres(
    A,
    b,
    x0: Optional[_dpnp.ndarray] = None,
    *,
    shift: float = 0.0,
    tol: float = 1e-5,
    maxiter: Optional[int] = None,
    M=None,
    callback: Optional[Callable] = None,
    check: bool = False,
) -> Tuple[_dpnp.ndarray, int]:
    """MINRES for symmetric (possibly indefinite) A — pure dpnp/oneMKL.

    Implements Paige-Saunders (1975) MINRES via Lanczos tridiagonalisation
    with Givens QR.  All matvec, dot-products, and vector updates run on
    device; only scalar recurrence coefficients are pulled to CPU.

    The QR step uses the exact two-rotation recurrence from SciPy minres.py:

      oldeps = epsln
      delta  = cs * dbar + sn * alpha    # apply previous Givens rotation
      gbar_k = sn * dbar - cs * alpha    # residual for new rotation
      epsln  = sn * beta
      dbar   = -cs * beta

      gamma  = hypot(gbar_k, beta)       # new rotation eliminates beta
      cs     = gbar_k / gamma
      sn     = beta   / gamma

    Parameters
    ----------
    A       : array_like or LinearOperator — symmetric/Hermitian (n, n)
    b       : array_like — right-hand side (n,)
    x0      : array_like, optional — initial guess
    shift   : float — solve (A - shift*I)x = b
    tol     : float — relative tolerance (default 1e-5)
    maxiter : int, optional — max iterations (default 5*n)
    M       : LinearOperator, optional — SPD preconditioner
    callback: callable, optional — callback(xk) after each step
    check   : bool — verify A symmetry before iterating

    Returns
    -------
    x    : dpnp.ndarray
    info : int  0=converged  1=maxiter  2=stagnation
    """
    # allow_M=True: MINRES fully supports SPD preconditioners
    A_op, M_op, x, b, dtype = _make_system(A, M, x0, b, allow_M=True)
    n      = b.shape[0]
    eps    = float(_np.finfo(_np_dtype(dtype)).eps)

    if maxiter is None:
        maxiter = 5 * n

    bnrm = float(_dpnp.linalg.norm(b))
    if bnrm == 0.0:
        return _dpnp.zeros_like(b), 0

    atol_eff = _get_atol("minres", bnrm, atol=None, rtol=tol)

    # ------------------------------------------------------------------
    # Initialise Lanczos: compute beta1 = ||M^{-1/2} r0||_M
    # ------------------------------------------------------------------
    r1     = b - A_op.matvec(x) if x0 is not None else b.copy()
    y      = M_op.matvec(r1)

    beta1  = float(_dpnp.sqrt(_dpnp.abs(_dpnp.real(_dpnp.vdot(r1, y)))))

    if beta1 == 0.0:
        return x, 0

    if check:
        Ay = A_op.matvec(y) - shift * y
        lhs = float(_dpnp.linalg.norm(
            Ay - (_dpnp.vdot(y, Ay) / _dpnp.vdot(y, y)) * y
        ))
        rhs = eps ** 0.5 * float(_dpnp.linalg.norm(Ay))
        if lhs > rhs:
            raise ValueError(
                "minres: A does not appear symmetric/Hermitian; "
                "set check=False to skip this test."
            )

    # ------------------------------------------------------------------
    # Paige-Saunders state variables (all scalars on CPU)
    # ------------------------------------------------------------------
    beta   = beta1
    oldb   = 0.0
    phibar = beta1

    # Givens rotation state carried between iterations (SciPy initialisation)
    cs   = -1.0   # cos of previous rotation
    sn   =  0.0   # sin of previous rotation
    dbar =  0.0   # sub-diagonal entry carried forward
    epsln = 0.0   # sub-sub-diagonal from two steps ago

    # w-vectors for the three-term solution update (on device)
    w  = _dpnp.zeros(n, dtype=dtype)
    w2 = _dpnp.zeros(n, dtype=dtype)

    # Lanczos vectors
    r2 = r1.copy()
    v  = y / beta1

    info = 1
    for itr in range(1, maxiter + 1):
        # ------------------------------------------------------------------
        # Lanczos step k: produces alpha_k, beta_{k+1}, v_k
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
        beta  = float(_dpnp.sqrt(_dpnp.abs(_dpnp.real(_dpnp.vdot(r2, y)))))

        if beta < 0.0:
            raise ValueError("minres: preconditioner M is not positive definite")

        # Stagnation: beta collapsed to machine-epsilon * beta1
        if beta <= eps * beta1:
            info = 2
            break

        # ------------------------------------------------------------------
        # QR step: correct Paige-Saunders (1975) two-rotation recurrence.
        #
        # Apply the PREVIOUS Givens rotation Q_{k-1} to the current
        # tridiagonal column.  The column is [dbar, (alpha-shift), beta].
        # (alpha already incorporates the shift via the Lanczos matvec above
        # so the column below uses plain `alpha`.)
        #
        # Previous rotation acts on rows (k-1, k):
        #   delta  = cs_{k-1} * dbar + sn_{k-1} * alpha   <- new diagonal
        #   gbar_k = sn_{k-1} * dbar - cs_{k-1} * alpha   <- residual
        #   epsln  = sn_{k-1} * beta                       <- sub-sub-diag
        #   dbar   = -cs_{k-1} * beta                      <- carry forward
        #
        # New rotation Q_k eliminates beta from [gbar_k, beta]:
        #   gamma = hypot(gbar_k, beta)
        #   cs_k  = gbar_k / gamma
        #   sn_k  = beta   / gamma
        # ------------------------------------------------------------------
        oldeps = epsln
        delta  = cs * dbar + sn * alpha    # apply previous rotation — diagonal
        gbar_k = sn * dbar - cs * alpha    # remaining entry -> new rotation
        epsln  = sn * beta                 # sub-sub-diagonal for next step
        dbar   = -cs * beta               # carry forward for next step

        gamma = _np.hypot(gbar_k, beta)
        if gamma == 0.0:
            gamma = eps
        cs = gbar_k / gamma               # new cos
        sn = beta   / gamma               # new sin

        phi    = cs * phibar
        phibar = sn * phibar

        # ------------------------------------------------------------------
        # Solution update: three-term w recurrence (Paige-Saunders §5)
        #   w_new = (v - oldeps * w_{k-2} - delta * w_{k-1}) / gamma
        #   x    += phi * w_new
        # ------------------------------------------------------------------
        denom = 1.0 / gamma
        w_new = (v - oldeps * w - delta * w2) * denom
        x     = x + phi * w_new
        w     = w2
        w2    = w_new

        rnorm = abs(phibar)

        if callback is not None:
            callback(x)

        if rnorm <= atol_eff:
            info = 0
            break

        # Stagnation: step size relative to solution norm
        if phi * denom < eps:
            info = 2
            break
    else:
        info = 1

    return x, int(info)
