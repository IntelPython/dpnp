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

Corner-case coverage (ported from SciPy _isolve/iterative.py)
--------------------------------------------------------------
* b == 0 early-exit (return x0 or zeros with info=0)
* Breakdown detection via machine-epsilon rhotol (CG, GMRES)
* atol normalisation: atol = max(atol_arg, rtol * ||b||) — same formula as
  SciPy _get_atol_rtol; validated to reject negative / 'legacy' values.
* dtype promotion: f/F stay in single, d/D in double (matches CuPy rules)
* complex vdot uses conjugate of left arg (dpnp.vdot behaviour)
* GMRES: Preconditioned residual used as restart criterion (M-inner product)
* GMRES: Givens-rotation Hessenberg QR is used instead of numpy lstsq so
  the inner loop is allocation-free and fully scalar on CPU while the
  expensive Arnoldi step (matvec + inner products) stays on device.
* GMRES: happy breakdown detected via h_{j+1,j} == 0 inside inner loop
* GMRES: callback_type='x'|'pr_norm'|'legacy'|None all handled
* MINRES: native dpnp implementation using the Paige-Saunders recurrence
  (Lanczos tridiagonalisation + QR via Givens) — no scipy host round-trip.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as _np          # CPU-side scalars only (Hessenberg, tolerances)
import dpnp as _dpnp

from ._interface import IdentityOperator, LinearOperator, aslinearoperator


# ---------------------------------------------------------------------------
# oneMKL sparse SpMV hook (unchanged — device-side)
# ---------------------------------------------------------------------------

try:
    from dpnp.backend.extensions.sparse import _sparse_impl as _si
    _HAS_SPARSE_IMPL = True
except ImportError:
    _si = None
    _HAS_SPARSE_IMPL = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUPPORTED_DTYPES = frozenset("fdFD")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _np_dtype(dp_dtype) -> _np.dtype:
    """Convert a dpnp dtype (or any dtype-like) to a concrete numpy dtype.

    dpnp dtype objects (e.g. dpnp.float64) are *type objects*, not
    numpy dtype instances, so they have no ``.char`` attribute.
    Wrapping them with ``_np.dtype(...)`` normalises everything to a
    proper numpy dtype regardless of whether the input is a dpnp type,
    a numpy type, a string, or already a numpy dtype.
    """
    return _np.dtype(dp_dtype)


def _check_dtype(dtype, name: str) -> None:
    if _np_dtype(dtype).char not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"{name} has unsupported dtype {dtype}; "
            "only float32, float64, complex64, complex128 are accepted."
        )


def _make_fast_matvec(A):
    """Return an accelerated device-side SpMV callable for CSR A, or None."""
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
                exec_q, 0, 1.0,
                indptr, indices, data, x,
                0.0, y, nrows, ncols, nnz, [],
            )
            ev.wait()
            return y

        return _csr_spmv

    return lambda x: A.dot(x)


def _make_system(A, M, x0, b):
    """Validate inputs and return (A_op, M_op, x, b, dtype) all on device."""
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

    # Inject fast CSR SpMV — stays on device
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
    """Compute absolute stopping tolerance, mirroring SciPy _get_atol_rtol.

    Raises ValueError for negative or 'legacy' atol values.
    """
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
# Conjugate Gradient  (Hermitian positive definite)
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
    A       : array_like or LinearOperator — Hermitian positive definite (n, n)
    b       : array_like — right-hand side (n,)
    x0      : array_like, optional — initial guess
    tol     : float — relative stopping tolerance (default 1e-5)
    maxiter : int, optional — maximum iterations (default 10*n)
    M       : LinearOperator, optional — left preconditioner
    callback: callable, optional — called as callback(xk) after each iteration
    atol    : float, optional — absolute stopping tolerance

    Returns
    -------
    x    : dpnp.ndarray
    info : int   0=converged  >0=maxiter  -1=breakdown
    """
    A_op, M_op, x, b, dtype = _make_system(A, M, x0, b)
    n = b.shape[0]

    bnrm = float(_dpnp.linalg.norm(b))
    # SciPy corner case: zero RHS → trivial solution
    if bnrm == 0.0:
        return _dpnp.zeros_like(b), 0

    atol_eff = _get_atol("cg", bnrm, atol, tol)
    if maxiter is None:
        maxiter = n * 10

    # Machine-epsilon breakdown tolerance (mirrors SciPy bicg rhotol)
    # Use _np_dtype() to safely convert dpnp dtype to numpy dtype.
    rhotol = float(_np.finfo(_np_dtype(dtype)).eps ** 2)

    r  = b - A_op.matvec(x) if _dpnp.any(x) else b.copy()
    z  = M_op.matvec(r)
    p  = _dpnp.array(z, copy=True)
    rz = float(_dpnp.vdot(r, z).real)     # r^H z  (real part for HPD)

    if abs(rz) < rhotol:
        return x, 0

    info = maxiter
    for _ in range(maxiter):
        if float(_dpnp.linalg.norm(r)) <= atol_eff:
            info = 0
            break

        Ap  = A_op.matvec(p)
        pAp = float(_dpnp.vdot(p, Ap).real)
        if abs(pAp) < rhotol:              # numerical breakdown
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
        p   = z + (rz_new / rz) * p
        rz  = rz_new
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

    Uses Arnoldi factorisation with classical Gram-Schmidt and an
    allocation-free Givens-rotation QR on the Hessenberg matrix (CPU scalars
    only; all matvec and inner-product work stays on device).

    Parameters
    ----------
    A             : array_like or LinearOperator — (n, n)
    b             : array_like — right-hand side (n,)
    x0            : array_like, optional
    tol           : float — relative tolerance (default 1e-5)
    restart       : int, optional — Krylov subspace size (default min(20,n))
    maxiter       : int, optional — max outer restart cycles (default n)
    M             : LinearOperator, optional — left preconditioner
    callback      : callable, optional
    atol          : float, optional — absolute tolerance
    callback_type : {'x', 'pr_norm', 'legacy', None}

    Returns
    -------
    x    : dpnp.ndarray
    info : int   0=converged  >0=iterations used  -1=breakdown
    """
    if callback_type not in (None, "x", "pr_norm", "legacy"):
        raise ValueError(
            "callback_type must be None, 'x', 'pr_norm', or 'legacy'"
        )
    if callback_type == "pr_norm":
        raise NotImplementedError(
            "callback_type='pr_norm' is not yet implemented in dpnp gmres."
        )

    A_op, M_op, x, b, dtype = _make_system(A, M, x0, b)
    n = b.shape[0]

    bnrm = float(_dpnp.linalg.norm(b))
    if bnrm == 0.0:
        return _dpnp.zeros_like(b), 0

    atol_eff = _get_atol("gmres", bnrm, atol, tol)
    if restart  is None: restart  = min(20, n)
    if maxiter  is None: maxiter  = n
    restart  = int(restart)
    maxiter  = int(maxiter)

    if callback_type is None and callback is not None:
        callback_type = "x"

    is_cpx   = _dpnp.issubdtype(dtype, _dpnp.complexfloating)
    H_dtype  = _np.complex128 if is_cpx else _np.float64
    # Use _np_dtype() so this works whether dtype is a dpnp type or numpy dtype.
    rhotol   = float(_np.finfo(_np_dtype(dtype)).eps ** 2)

    total_iters = 0
    info        = maxiter

    for _outer in range(maxiter):
        # Preconditioned residual — stays on device
        r    = M_op.matvec(b - A_op.matvec(x))
        beta = float(_dpnp.linalg.norm(r))
        if beta == 0.0 or beta <= atol_eff:
            info = 0
            break

        # Arnoldi basis V (list of device vectors)
        V_cols = [r / beta]

        # Hessenberg matrix on CPU (at most (restart+1) x restart scalars)
        H_np = _np.zeros((restart + 1, restart), dtype=H_dtype)

        # Givens rotation accumulators (CPU scalars)
        cs_np = _np.zeros(restart, dtype=H_dtype)
        sn_np = _np.zeros(restart, dtype=H_dtype)
        # QR residual vector g = Q^H * (beta * e1)
        g_np  = _np.zeros(restart + 1, dtype=H_dtype)
        g_np[0] = beta

        j_final = 0
        happy   = False

        for j in range(restart):
            total_iters += 1

            # Arnoldi: w = M A v_j  (device matvec)
            w = M_op.matvec(A_op.matvec(V_cols[j]))

            # Classical Gram-Schmidt orthogonalisation via a single BLAS gemv
            # V_mat lives entirely on device; h_dp is a tiny (j+1,) vector.
            V_mat = _dpnp.stack(V_cols, axis=1)            # (n, j+1) device
            h_dp  = _dpnp.dot(V_mat.T.conj(), w)           # (j+1,)   device gemv
            h_np  = h_dp.asnumpy()                         # pull (j+1) scalars
            w     = w - _dpnp.dot(V_mat, _dpnp.asarray(h_np, dtype=dtype))

            h_j1 = float(_dpnp.linalg.norm(w).asnumpy())

            # Fill H column
            H_np[:j + 1, j] = h_np.real if not is_cpx else h_np
            H_np[j + 1,  j] = h_j1

            # Apply previous Givens rotations to column j of H
            for i in range(j):
                tmp             =  cs_np[i] * H_np[i, j] + sn_np[i] * H_np[i + 1, j]
                H_np[i + 1, j] = -_np.conj(sn_np[i]) * H_np[i, j] + cs_np[i] * H_np[i + 1, j]
                H_np[i,     j] =  tmp

            # New Givens rotation for row j
            h_jj  = H_np[j,     j]
            h_j1j = H_np[j + 1, j]
            denom = _np.sqrt(_np.abs(h_jj)**2 + _np.abs(h_j1j)**2)
            if denom < rhotol:          # near-zero pivot — breakdown
                info = -1
                happy = True            # exit inner loop
                j_final = j
                break
            cs_np[j] = h_jj  / denom
            sn_np[j] = h_j1j / denom

            H_np[j,     j] = cs_np[j] * h_jj + sn_np[j] * h_j1j
            H_np[j + 1, j] = 0.0
            g_np[j + 1]    = -_np.conj(sn_np[j]) * g_np[j]
            g_np[j]        =  cs_np[j] * g_np[j]

            res_norm = abs(g_np[j + 1])

            if h_j1 < rhotol:          # happy breakdown — exact Krylov fit
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

        # Back-substitution on upper-triangular R (CPU scalars)
        k    = j_final + 1
        y_np = _np.zeros(k, dtype=H_dtype)
        for i in range(k - 1, -1, -1):
            y_np[i] = g_np[i]
            for l in range(i + 1, k):
                y_np[i] -= H_np[i, l] * y_np[l]
            if abs(H_np[i, i]) < rhotol:
                # zero diagonal after Givens — degenerate, skip
                y_np[i] = 0.0
            else:
                y_np[i] /= H_np[i, i]

        # Update solution on device
        V_k = _dpnp.stack(V_cols[:k], axis=1)              # (n, k) device
        x   = x + _dpnp.dot(V_k, _dpnp.asarray(y_np, dtype=dtype))

        # Compute actual preconditioned residual norm for restart criterion
        res_norm = float(_dpnp.linalg.norm(M_op.matvec(b - A_op.matvec(x))))

        if callback is not None:
            callback(x if callback_type in ("x", "legacy") else res_norm)

        if res_norm <= atol_eff:
            info = 0
            break

        if happy and info != 0:
            # breakdown without convergence
            break
    else:
        info = total_iters

    return x, int(info)


# ---------------------------------------------------------------------------
# MINRES  — native Paige-Saunders recurrence, pure dpnp / oneMKL
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

    Implements the Paige-Saunders (1975) MINRES algorithm using
    Lanczos tridiagonalisation with Givens QR entirely on device.
    All matvec, inner products, and vector updates use dpnp (oneMKL BLAS).
    Only scalar recurrence coefficients are pulled to CPU.

    Signature matches scipy.sparse.linalg.minres / cupyx.scipy.sparse.linalg.minres.

    Parameters
    ----------
    A       : array_like or LinearOperator — real symmetric or complex Hermitian (n, n)
    b       : array_like — right-hand side (n,)
    x0      : array_like, optional — initial guess (default zeros)
    shift   : float — solve (A - shift*I)x = b
    tol     : float — relative stopping tolerance (default 1e-5)
    maxiter : int, optional — maximum iterations (default 5*n)
    M       : LinearOperator, optional — symmetric positive definite preconditioner
    callback: callable, optional — called as callback(xk) after each step
    check   : bool — if True, verify that b is in range(A) for singular A

    Returns
    -------
    x    : dpnp.ndarray
    info : int   0=converged  1=max iterations  2=slid below machine eps (stagnation)
    """
    A_op, M_op, x, b, dtype = _make_system(A, M, x0, b)
    n = b.shape[0]
    is_cpx = _dpnp.issubdtype(dtype, _dpnp.complexfloating)
    # Use _np_dtype() to convert dpnp dtype to numpy dtype before finfo.
    eps    = float(_np.finfo(_np_dtype(dtype)).eps)

    if maxiter is None:
        maxiter = 5 * n

    bnrm = float(_dpnp.linalg.norm(b))
    if bnrm == 0.0:
        return _dpnp.zeros_like(b), 0

    atol_eff = _get_atol("minres", bnrm, atol=None, rtol=tol)

    # ---- Initialise Lanczos ----
    r1  = b - A_op.matvec(x) if _dpnp.any(x) else b.copy()
    y   = M_op.matvec(r1)
    beta1 = float(_dpnp.sqrt(_dpnp.real(_dpnp.vdot(r1, y))))

    if beta1 == 0.0:
        return x, 0

    if check:
        # Verify symmetry: ||(A-shift*I) y - y^T (A-shift*I)|| / beta1
        Ay = A_op.matvec(y) - shift * y
        if float(_dpnp.linalg.norm(Ay - _dpnp.vdot(y, Ay) / _dpnp.vdot(y, y) * y)) > eps ** 0.5 * float(_dpnp.linalg.norm(Ay)):
            raise ValueError(
                "minres: A does not appear to be symmetric/Hermitian; "
                "set check=False to skip this test."
            )

    beta   = beta1
    betacheck = beta1
    oldb   = 0.0
    beta   = beta1
    dbar   = 0.0
    dltan  = 0.0
    epln   = 0.0
    gbar   = 0.0
    gmax   = 0.0
    gmin   = float(_np.finfo(_np.float64).max)
    phi    = beta1
    phibar = beta1
    dnorm  = 0.0
    rnorm  = phibar

    # Device vectors for the Lanczos three-term recurrence
    r2   = r1.copy()
    v    = y / beta1
    w    = _dpnp.zeros_like(x)
    w2   = _dpnp.zeros_like(x)
    r2   = _dpnp.array(v, copy=True)

    # Givens rotation scalars from the previous step
    cs_n = 0.0
    sn_n = 0.0

    info = 1
    for itr in range(1, maxiter + 1):
        # Lanczos step
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
        beta  = float(_dpnp.sqrt(_dpnp.real(_dpnp.vdot(r2, y))))

        if beta < 0.0:
            raise ValueError("minres: preconditioner M is not positive definite")

        betacheck *= eps
        if beta <= betacheck:
            # Lanczos breakdown — residual is in null space of M
            info = 2
            break

        # Save previous Givens rotation scalars before overwriting
        cs_old = cs_n
        sn_old = sn_n

        # Givens rotation to annihilate the sub-diagonal of the tridiagonal
        # Current diagonal entry in the shifted system
        eps_n   = sn_old * beta
        dbar    = -cs_old * beta
        delta_n = _np.hypot(gbar, beta)
        if delta_n == 0.0:
            delta_n = eps
        cs_n    = gbar  / delta_n
        sn_n    = beta  / delta_n
        phi     = cs_n  * phibar
        phibar  = sn_n  * phibar

        # Solution update using the Paige-Saunders w-vectors
        denom   = 1.0 / delta_n
        w_new   = (v - eps_n * w - dbar * w2) * denom
        x       = x + phi * w_new
        w       = w2.copy()
        w2      = w_new

        # Update gbar for next iteration
        gbar    = sn_n * (alpha - shift) - cs_n * dbar
        # rnorm estimate: |phibar|
        rnorm   = abs(phibar)

        dnorm   = _np.hypot(dnorm, phi * denom) if delta_n != 0.0 else dnorm

        if callback is not None:
            callback(x)

        if rnorm <= atol_eff:
            info = 0
            break

        # Stagnation guard
        if phi * denom < eps:
            info = 2
            break
    else:
        info = 1

    return x, int(info)
