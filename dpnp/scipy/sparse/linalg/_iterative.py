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

"""Iterative sparse linear solvers for dpnp.

Implements cg, gmres, minres with interfaces matching
cupyx.scipy.sparse.linalg (CuPy v14.0.1) and scipy.sparse.linalg.

Performance strategy
--------------------
* n <= _HOST_N_THRESHOLD  -> delegate to scipy.sparse.linalg (CPU fast path,
  same philosophy as CuPy host-dispatch for small systems).
* n >  _HOST_N_THRESHOLD  -> pure dpnp path; dense operations dispatch to
  oneMKL via dpnp.dot / dpnp.linalg.norm / dpnp.vdot (BLAS level-2/3).
* CSR sparse input        -> _make_fast_matvec injects oneMKL sparse::gemv
  via the _sparse_impl pybind11 extension (dpnp.backend.extensions.sparse).
  Falls back to A.dot(x) if the extension is not yet built.
* GMRES Hessenberg lstsq  -> numpy.linalg.lstsq on CPU (the (restart x restart)
  matrix is tiny; same decision as CuPy).
* MINRES                  -> SciPy host stub (CuPy v14.0.1 has no GPU MINRES;
  a native oneMKL MINRES will be added in a future dpnp release).
"""

from __future__ import annotations

import inspect
from typing import Callable, Optional, Tuple

import numpy as _np
import dpnp as _dpnp

from ._interface import IdentityOperator, LinearOperator, aslinearoperator

# ---------------------------------------------------------------------------
# Try to import the compiled _sparse_impl extension (oneMKL sparse::gemv).
# If the extension has not been built yet the pure-Python / A.dot fallback
# is used transparently - no import error is raised at module load time.
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

# Route to scipy for systems smaller than this threshold, mirroring CuPy's
# host-dispatch heuristic for small linear systems.
_HOST_N_THRESHOLD = 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(x):
    """Convert a dpnp or numpy array to a numpy array safely."""
    if isinstance(x, _dpnp.ndarray):
        return x.asnumpy()
    return _np.asarray(x)


def _check_dtype(dtype, name: str) -> None:
    if dtype.char not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"{name} has unsupported dtype {dtype}; "
            "only float32, float64, complex64, complex128 are accepted."
        )


def _scipy_tol_kwarg(fn) -> str:
    """Return 'rtol' if SciPy >= 1.12 renamed tol, else 'tol'."""
    try:
        sig = inspect.signature(fn)
        return "rtol" if "rtol" in sig.parameters else "tol"
    except Exception:
        return "tol"


# ---------------------------------------------------------------------------
# oneMKL sparse SpMV hook
# Equivalent of _cusparse.spMV_make_fast_matvec for the SYCL/oneMKL backend.
# ---------------------------------------------------------------------------

def _make_fast_matvec(A):
    """Return an accelerated SpMV callable for CSR sparse A, or None.

    Priority order:
    1. _sparse_impl._sparse_gemv  (oneMKL sparse::gemv, fully async SYCL)
    2. A.dot                      (dpnp.scipy.sparse CSR dot, fallback)
    3. None                       (caller will use LinearOperator.matvec)
    """
    try:
        from dpnp.scipy import sparse as _sp
        if not (_sp.issparse(A) and A.format == "csr"):
            return None
    except (ImportError, AttributeError):
        return None

    if _HAS_SPARSE_IMPL:
        # --- fast path: oneMKL sparse::gemv via pybind11 ---
        # Pull CSR arrays once; they are already in USM device memory.
        indptr  = A.indptr          # row_ptr  - int32 or int64 USM array
        indices = A.indices         # col_ind  - int32 or int64 USM array
        data    = A.data            # values   - float32 or float64 USM array
        nrows   = int(A.shape[0])
        ncols   = int(A.shape[1])
        nnz     = int(data.shape[0])
        # Capture the SYCL queue from the matrix data array at closure-creation
        # time, not from x at call time.  This avoids queue mismatch when x is
        # constructed on a different (e.g. default CPU) queue.
        exec_q  = data.sycl_queue

        def _csr_spmv(x: _dpnp.ndarray) -> _dpnp.ndarray:
            y = _dpnp.zeros(nrows, dtype=data.dtype, sycl_queue=exec_q)
            _, ev = _si._sparse_gemv(
                exec_q,
                0,            # trans = NoTrans
                1.0,          # alpha
                indptr, indices, data,
                x,
                0.0,          # beta
                y,
                nrows, ncols, nnz,
                [],           # depends
            )
            ev.wait()
            return y

        return _csr_spmv

    # --- fallback: dpnp.scipy.sparse CSR dot ---
    return lambda x: A.dot(x)


# ---------------------------------------------------------------------------
# _make_system  (mirrors CuPy's _make_system)
# ---------------------------------------------------------------------------

def _make_system(A, M, x0, b):
    """Validate and normalise inputs; inject fast SpMV if available.

    Returns
    -------
    A_op, M_op, x0, b, dtype
    """
    A_op = aslinearoperator(A)
    n = A_op.shape[0]
    if A_op.shape[0] != A_op.shape[1]:
        raise ValueError("A must be a square operator")

    b = _dpnp.asarray(b).reshape(-1)
    if b.shape[0] != n:
        raise ValueError(
            f"b length mismatch: operator has shape {A_op.shape}, b has {b.shape[0]} entries"
        )

    # Determine working precision (matches CuPy dtype-promotion rules)
    if _dpnp.issubdtype(b.dtype, _dpnp.complexfloating):
        dtype = _dpnp.complex128
    else:
        dtype = _dpnp.float64
    if A_op.dtype is not None and A_op.dtype.char in "fF":
        dtype = _dpnp.complex64 if A_op.dtype.char == "F" else _dpnp.float32

    b = b.astype(dtype, copy=False)
    _check_dtype(b.dtype, "b")

    if x0 is None:
        x0 = _dpnp.zeros(n, dtype=dtype)
    else:
        x0 = _dpnp.asarray(x0, dtype=dtype).reshape(-1)
    if x0.shape[0] != n:
        raise ValueError(
            f"x0 length mismatch: expected {n}, got {x0.shape[0]}"
        )

    M_op = IdentityOperator((n, n), dtype=dtype) if M is None else aslinearoperator(M)

    # Inject fast CSR SpMV when available
    fast_mv = _make_fast_matvec(A)
    if fast_mv is not None:
        orig = A_op
        class _FastOp(LinearOperator):
            def __init__(self):
                super().__init__(orig.dtype, orig.shape)
            def _matvec(self, x):  return fast_mv(x)
            def _rmatvec(self, x): return orig.rmatvec(x)
        A_op = _FastOp()

    return A_op, M_op, x0, b, dtype


def _tol_to_atol(b, tol: float, atol) -> float:
    """Compute absolute stopping threshold matching SciPy / CuPy semantics."""
    bnrm = float(_dpnp.linalg.norm(b))
    return max(0.0 if atol is None else float(atol), float(tol) * bnrm)


# ---------------------------------------------------------------------------
# Conjugate Gradient
# ---------------------------------------------------------------------------

def cg(
    A,
    b,
    x0=None,
    *,
    tol: float = 1e-5,
    maxiter=None,
    M=None,
    callback=None,
    atol=None,
) -> Tuple[_dpnp.ndarray, int]:
    """Conjugate Gradient solver for Hermitian positive definite A.

    Signature matches cupyx.scipy.sparse.linalg.cg / scipy.sparse.linalg.cg.

    Parameters
    ----------
    A : array_like or LinearOperator  -- Hermitian positive definite, shape (n, n)
    b : array_like                    -- right-hand side, shape (n,)
    x0 : array_like, optional         -- initial guess
    tol : float                       -- relative tolerance (default 1e-5)
    maxiter : int, optional           -- maximum iterations (default 10*n)
    M : LinearOperator, optional      -- preconditioner
    callback : callable, optional     -- called as callback(xk) each iteration
    atol : float, optional            -- absolute tolerance

    Returns
    -------
    x : dpnp.ndarray
    info : int  (0 = converged, >0 = max iters reached, -1 = breakdown)
    """
    b = _dpnp.asarray(b).reshape(-1)
    n = b.shape[0]

    # --- small-system CPU fast path (mirrors CuPy host-dispatch) ---
    if n <= _HOST_N_THRESHOLD:
        try:
            import scipy.sparse.linalg as _sla
            _kw = {
                _scipy_tol_kwarg(_sla.cg): tol,
                "atol": 0.0 if atol is None else float(atol),
                "maxiter": maxiter,
            }
            A_np  = _to_numpy(A) if not hasattr(A, "matvec") else A
            b_np  = _to_numpy(b)
            x0_np = None if x0 is None else _to_numpy(_dpnp.asarray(x0))
            x_np, info = _sla.cg(A_np, b_np, x0=x0_np, callback=callback, **_kw)
            return _dpnp.asarray(x_np), int(info)
        except Exception:
            pass  # fall through to dpnp path

    # --- dpnp / oneMKL path ---
    A_op, M_op, x, b, dtype = _make_system(A, M, x0, b)
    if maxiter is None:
        maxiter = n * 10
    atol_eff = _tol_to_atol(b, tol, atol)

    r  = b - A_op.matvec(x)
    z  = M_op.matvec(r)
    p  = _dpnp.array(z, copy=True)
    rz = float(_dpnp.vdot(r, z).real)

    if rz == 0.0:
        return x, 0

    info = maxiter
    for _ in range(maxiter):
        Ap  = A_op.matvec(p)
        pAp = float(_dpnp.vdot(p, Ap).real)
        if pAp == 0.0:
            info = -1
            break

        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        if callback is not None:
            callback(x)

        if float(_dpnp.linalg.norm(r)) <= atol_eff:
            info = 0
            break

        z      = M_op.matvec(r)
        rz_new = float(_dpnp.vdot(r, z).real)
        p      = z + (rz_new / rz) * p
        rz     = rz_new
    else:
        info = maxiter

    return x, int(info)


# ---------------------------------------------------------------------------
# Restarted GMRES
# ---------------------------------------------------------------------------

def gmres(
    A,
    b,
    x0=None,
    *,
    tol: float = 1e-5,
    restart=None,
    maxiter=None,
    M=None,
    callback=None,
    atol=None,
    callback_type=None,
) -> Tuple[_dpnp.ndarray, int]:
    """Restarted GMRES with oneMKL-accelerated Arnoldi step.

    Signature matches cupyx.scipy.sparse.linalg.gmres / scipy.sparse.linalg.gmres.

    Parameters
    ----------
    A, b, x0, tol, maxiter, M, callback, atol
        See scipy.sparse.linalg.gmres documentation.
    restart : int, optional
        Krylov subspace dimension between restarts. Default: min(20, n).
    callback_type : {'x', 'pr_norm', None}
        'x'      -> callback(xk) at each restart (default when callback given).
        'pr_norm'-> callback(residual_norm) at each restart.
        None     -> no callback invocation.

    Returns
    -------
    x : dpnp.ndarray
    info : int  (0 = converged, >0 = iterations used, -1 = breakdown)
    """
    b = _dpnp.asarray(b).reshape(-1)
    n = b.shape[0]

    # --- small-system CPU fast path ---
    if n <= _HOST_N_THRESHOLD:
        try:
            import scipy.sparse.linalg as _sla
            _kw = {
                _scipy_tol_kwarg(_sla.gmres): tol,
                "atol":   0.0 if atol is None else float(atol),
                "restart": restart,
                "maxiter": maxiter,
            }
            sig = inspect.signature(_sla.gmres)
            if "callback_type" in sig.parameters and callback_type is not None:
                _kw["callback_type"] = callback_type
            A_np  = _to_numpy(A) if not hasattr(A, "matvec") else A
            b_np  = _to_numpy(b)
            x0_np = None if x0 is None else _to_numpy(_dpnp.asarray(x0))
            x_np, info = _sla.gmres(A_np, b_np, x0=x0_np, callback=callback, **_kw)
            return _dpnp.asarray(x_np), int(info)
        except Exception:
            pass

    if callback_type not in (None, "x", "pr_norm"):
        raise ValueError("callback_type must be None, 'x', or 'pr_norm'")

    A_op, M_op, x, b, dtype = _make_system(A, M, x0, b)
    if restart  is None: restart  = min(20, n)
    if maxiter  is None: maxiter  = n
    restart, maxiter = int(restart), int(maxiter)

    # Default callback_type when a callback is provided (matches CuPy)
    if callback_type is None:
        callback_type = "x" if callback is not None else None

    atol_eff = _tol_to_atol(b, tol, atol)
    is_cpx   = _dpnp.issubdtype(dtype, _dpnp.complexfloating)
    H_dtype  = _np.complex128 if is_cpx else _np.float64

    info         = 0
    total_iters  = 0

    for _outer in range(maxiter):
        r    = M_op.matvec(b - A_op.matvec(x))
        beta = float(_dpnp.linalg.norm(r))
        if beta == 0.0 or beta <= atol_eff:
            info = 0
            break

        V_cols = [r / beta]
        H_np   = _np.zeros((restart + 1, restart), dtype=H_dtype)
        e1_np  = _np.zeros(restart + 1, dtype=H_dtype)
        e1_np[0] = beta

        j_inner  = 0
        for j in range(restart):
            total_iters += 1
            w = M_op.matvec(A_op.matvec(V_cols[j]))

            # Arnoldi step: h = V_j^H w via single oneMKL BLAS gemv.
            V_mat  = _dpnp.stack(V_cols, axis=1)          # (n, j+1)
            h_dp   = _dpnp.dot(V_mat.T.conj(), w)         # (j+1,)  -- oneMKL gemv
            h_np   = h_dp.asnumpy()                        # pull tiny vector to CPU
            w      = w - _dpnp.dot(V_mat, _dpnp.asarray(h_np, dtype=dtype))

            h_j1 = float(_dpnp.linalg.norm(w))
            H_np[:j + 1, j] = h_np
            H_np[j + 1,  j] = h_j1

            if h_j1 == 0.0:            # happy breakdown
                j_inner = j
                break
            V_cols.append(w / h_j1)
            j_inner = j

        # Hessenberg least-squares on CPU (matrix is at most restart x restart)
        k = j_inner + 1
        y_np, _, _, _ = _np.linalg.lstsq(
            H_np[:k + 1, :k], e1_np[:k + 1], rcond=None
        )

        V_k = _dpnp.stack(V_cols[:k], axis=1)
        x   = x + _dpnp.dot(V_k, _dpnp.asarray(y_np, dtype=dtype))

        res_norm = float(_dpnp.linalg.norm(M_op.matvec(b - A_op.matvec(x))))

        if callback is not None:
            callback(x if callback_type == "x" else res_norm)

        if res_norm <= atol_eff:
            info = 0
            break
    else:
        info = total_iters

    return x, int(info)


# ---------------------------------------------------------------------------
# MINRES  (SciPy-backed stub)
# ---------------------------------------------------------------------------
# CuPy v14.0.1 does NOT include a GPU-native MINRES implementation.
# Using a SciPy host stub is therefore the correct parallel strategy.
# A native oneMKL-based MINRES will be added in a future dpnp release.

def minres(
    A,
    b,
    x0=None,
    *,
    shift: float = 0.0,
    tol: float = 1e-5,
    maxiter=None,
    M=None,
    callback=None,
    check: bool = False,
) -> Tuple[_dpnp.ndarray, int]:
    """MINRES for symmetric (possibly indefinite) A.

    Signature matches cupyx.scipy.sparse.linalg.minres / scipy.sparse.linalg.minres.

    Currently delegates to scipy.sparse.linalg.minres on the host with dpnp
    operator wrappers.  A native oneMKL implementation will replace this stub
    in a future release.

    Parameters
    ----------
    A : array_like or LinearOperator  -- symmetric, shape (n, n)
    b : array_like                    -- right-hand side
    x0 : array_like, optional
    shift : float                     -- solve (A - shift*I) x = b
    tol : float                       -- relative stopping tolerance
    maxiter : int, optional
    M : LinearOperator, optional      -- symmetric positive definite preconditioner
    callback : callable, optional     -- called as callback(xk) each iteration
    check : bool                      -- check that A is symmetric (default False)

    Returns
    -------
    x : dpnp.ndarray
    info : int  (0 = converged, >0 = stagnation / max iters)
    """
    try:
        import scipy.sparse.linalg as _sla
    except ImportError as exc:
        raise NotImplementedError(
            "dpnp.scipy.sparse.linalg.minres currently requires SciPy on the host. "
            "A native oneMKL MINRES will be added in a future dpnp release."
        ) from exc

    A_dp = aslinearoperator(A)
    if A_dp.shape[0] != A_dp.shape[1]:
        raise ValueError("minres requires a square operator")

    def _wrap_op(op):
        return _sla.LinearOperator(
            op.shape,
            matvec=lambda x: op.matvec(_dpnp.asarray(x)).asnumpy(),
            dtype=_np.dtype(op.dtype) if op.dtype is not None else _np.float64,
        )

    M_sci = None if M is None else _wrap_op(aslinearoperator(M))
    b_np  = _dpnp.asarray(b).reshape(-1).asnumpy()
    x0_np = None if x0 is None else _dpnp.asarray(x0).reshape(-1).asnumpy()

    tkw = _scipy_tol_kwarg(_sla.minres)
    x_np, info = _sla.minres(
        _wrap_op(A_dp),
        b_np,
        x0=x0_np,
        **{tkw: tol},
        shift=shift,
        maxiter=maxiter,
        M=M_sci,
        callback=callback,
        show=False,
        check=check,
    )
    return _dpnp.asarray(x_np), int(info)
