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

from __future__ import annotations

import inspect
from typing import Callable, Optional, Tuple

import dpnp as _dpnp

from ._interface import aslinearoperator


_ArrayLike = _dpnp.ndarray


_HOST_THRESHOLD_DEFAULT = 256


def _norm(x: _ArrayLike) -> float:
    return float(_dpnp.linalg.norm(x))


def _make_stop_criterion(b: _ArrayLike, tol: float, atol: Optional[float]) -> float:
    bnrm = _norm(b)
    atol_eff = 0.0 if atol is None else float(atol)
    return max(tol * bnrm, atol_eff)


def _has_scipy() -> bool:
    try:
        import scipy  # noqa: F401

        return True
    except Exception:
        return False


def _scipy_tol_kwarg(sla_func) -> str:
    """Return 'rtol' if the SciPy function accepts it (SciPy >= 1.12), else 'tol'."""
    try:
        sig = inspect.signature(sla_func)
        return "rtol" if "rtol" in sig.parameters else "tol"
    except (ValueError, TypeError):
        return "tol"


def _cpu_cg(A, b, x0, tol, maxiter, M, callback, atol):
    import numpy as _np
    import scipy.sparse.linalg as _sla

    from ._interface import aslinearoperator as _aslo

    A_dp = _aslo(A)

    def matvec_np(x_np):
        x_dp = _dpnp.asarray(x_np)
        y_dp = A_dp.matvec(x_dp)
        return _np.asarray(y_dp)

    A_sci = _sla.LinearOperator(
        shape=A_dp.shape, matvec=matvec_np, dtype=_np.dtype(A_dp.dtype)
    )

    if M is not None:
        M_dp = _aslo(M)

        def m_matvec_np(x_np):
            x_dp = _dpnp.asarray(x_np)
            y_dp = M_dp.matvec(x_dp)
            return _np.asarray(y_dp)

        M_sci = _sla.LinearOperator(
            shape=M_dp.shape, matvec=m_matvec_np, dtype=_np.dtype(M_dp.dtype)
        )
    else:
        M_sci = None

    b_np = _np.asarray(_dpnp.asarray(b).reshape(-1))
    x0_np = None if x0 is None else _np.asarray(_dpnp.asarray(x0).reshape(-1))

    # SciPy >= 1.12 renamed tol -> rtol; detect at call time to avoid DeprecationWarning.
    tol_kw = _scipy_tol_kwarg(_sla.cg)
    x_host, info = _sla.cg(
        A_sci,
        b_np,
        x0=x0_np,
        **{tol_kw: tol},
        maxiter=maxiter,
        M=M_sci,
        callback=callback,
        atol=0.0 if atol is None else atol,
    )

    x_dp = _dpnp.asarray(x_host)
    return x_dp, int(info)


def _cpu_gmres(A, b, x0, tol, restart, maxiter, M, callback, atol, callback_type):
    import numpy as _np
    import scipy.sparse.linalg as _sla

    from ._interface import aslinearoperator as _aslo

    A_dp = _aslo(A)

    def matvec_np(x_np):
        x_dp = _dpnp.asarray(x_np)
        y_dp = A_dp.matvec(x_dp)
        return _np.asarray(y_dp)

    A_sci = _sla.LinearOperator(
        shape=A_dp.shape, matvec=matvec_np, dtype=_np.dtype(A_dp.dtype)
    )

    if M is not None:
        M_dp = _aslo(M)

        def m_matvec_np(x_np):
            x_dp = _dpnp.asarray(x_np)
            y_dp = M_dp.matvec(x_dp)
            return _np.asarray(y_dp)

        M_sci = _sla.LinearOperator(
            shape=M_dp.shape, matvec=m_matvec_np, dtype=_np.dtype(M_dp.dtype)
        )
    else:
        M_sci = None

    b_np = _np.asarray(_dpnp.asarray(b).reshape(-1))
    x0_np = None if x0 is None else _np.asarray(_dpnp.asarray(x0).reshape(-1))

    # SciPy >= 1.12 renamed tol -> rtol; detect at call time.
    tol_kw = _scipy_tol_kwarg(_sla.gmres)

    # callback_type was added in SciPy 1.9; only pass it when supported.
    gmres_sig = inspect.signature(_sla.gmres)
    extra_kw = {}
    if "callback_type" in gmres_sig.parameters and callback_type is not None:
        extra_kw["callback_type"] = callback_type

    x_host, info = _sla.gmres(
        A_sci,
        b_np,
        x0=x0_np,
        **{tol_kw: tol},
        restart=restart,
        maxiter=maxiter,
        M=M_sci,
        callback=callback,
        atol=0.0 if atol is None else atol,
        **extra_kw,
    )

    x_dp = _dpnp.asarray(x_host)
    return x_dp, int(info)


def cg(
    A,
    b,
    x0: Optional[_ArrayLike] = None,
    *,
    tol: float = 1e-5,
    maxiter: Optional[int] = None,
    M=None,
    callback: Optional[Callable[[_ArrayLike], None]] = None,
    atol: Optional[float] = None,
):
    b = _dpnp.asarray(b).reshape(-1)
    n = b.size

    if n < _HOST_THRESHOLD_DEFAULT and _has_scipy():
        return _cpu_cg(A, b, x0, tol, maxiter, M, callback, atol)

    A = aslinearoperator(A)

    if M is not None:
        raise NotImplementedError("Preconditioner M is not implemented for cg yet")

    if x0 is None:
        x = _dpnp.zeros_like(b)
    else:
        x = _dpnp.asarray(x0).reshape(-1).copy()

    r = b - A.matvec(x)
    p = r.copy()
    rr_old = _dpnp.vdot(r, r).real
    if rr_old == 0.0:
        return x, 0

    if maxiter is None:
        maxiter = n * 10

    tol_th = _make_stop_criterion(b, tol, atol)

    info = 0

    for _ in range(maxiter):
        Ap = A.matvec(p)
        pAp = _dpnp.vdot(p, Ap).real
        if pAp == 0.0:
            info = -1
            break

        alpha = rr_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        if callback is not None:
            callback(x)

        rr_new = _dpnp.vdot(r, r).real
        res_norm = rr_new**0.5
        if res_norm <= tol_th:
            info = 0
            break

        beta = rr_new / rr_old
        p = r + beta * p
        rr_old = rr_new
    else:
        info = maxiter

    return x, int(info)


def gmres(
    A,
    b,
    x0: Optional[_ArrayLike] = None,
    *,
    tol: float = 1e-5,
    restart: Optional[int] = None,
    maxiter: Optional[int] = None,
    M=None,
    callback: Optional[Callable[[object], None]] = None,
    atol: Optional[float] = None,
    callback_type: Optional[str] = None,
):
    b = _dpnp.asarray(b).reshape(-1)
    n = b.size

    if n < _HOST_THRESHOLD_DEFAULT and _has_scipy():
        return _cpu_gmres(A, b, x0, tol, restart, maxiter, M, callback, atol, callback_type)

    if callback_type not in (None, "x", "pr_norm"):
        raise ValueError("callback_type must be None, 'x', or 'pr_norm'")
    if callback_type == "pr_norm":
        raise NotImplementedError("callback_type='pr_norm' is not implemented yet")

    A = aslinearoperator(A)

    if M is not None:
        raise NotImplementedError("Preconditioner M is not implemented for gmres yet")

    if x0 is None:
        x = _dpnp.zeros_like(b)
    else:
        x = _dpnp.asarray(x0).reshape(-1).copy()

    if restart is None:
        restart = min(20, n)
    if maxiter is None:
        maxiter = n

    restart = int(restart)
    maxiter = int(maxiter)

    tol_th = _make_stop_criterion(b, tol, atol)

    info = 0
    total_iter = 0

    for outer in range(maxiter):
        r = b - A.matvec(x)
        beta = _norm(r)
        if beta == 0.0:
            info = 0
            break
        if beta <= tol_th:
            info = 0
            break

        V = _dpnp.zeros((n, restart + 1), dtype=x.dtype)
        H = _dpnp.zeros((restart + 1, restart), dtype=_dpnp.float64)
        cs = _dpnp.zeros(restart, dtype=_dpnp.float64)
        sn = _dpnp.zeros(restart, dtype=_dpnp.float64)
        e1 = _dpnp.zeros(restart + 1, dtype=_dpnp.float64)
        e1[0] = 1.0

        V[:, 0] = r / beta
        g = beta * e1

        inner_converged = False

        for j in range(restart):
            total_iter += 1
            w = A.matvec(V[:, j])

            for i in range(j + 1):
                H[i, j] = float(_dpnp.vdot(V[:, i], w).real)
                w = w - H[i, j] * V[:, i]

            H[j + 1, j] = _norm(w)
            if H[j + 1, j] != 0.0:
                V[:, j + 1] = w / H[j + 1, j]
            else:
                for k in range(j + 1, restart + 1):
                    H[k, j] = 0.0
                j_max = j
                break
            j_max = j

            for i in range(j):
                temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = temp

            h_jj = H[j, j]
            h_j1j = H[j + 1, j]
            denom = (h_jj**2 + h_j1j**2) ** 0.5
            if denom == 0.0:
                cs[j] = 1.0
                sn[j] = 0.0
            else:
                cs[j] = h_jj / denom
                sn[j] = h_j1j / denom

            H[j, j] = cs[j] * h_jj + sn[j] * h_j1j
            H[j + 1, j] = 0.0

            g_j = g[j]
            g[j] = cs[j] * g_j
            g[j + 1] = -sn[j] * g_j

            res_norm = abs(g[j + 1])
            if res_norm <= tol_th:
                inner_converged = True
                j_max = j
                break

        k_dim = j_max + 1
        y = _dpnp.zeros(k_dim, dtype=_dpnp.float64)
        for i in range(k_dim - 1, -1, -1):
            s = g[i]
            for j2 in range(i + 1, k_dim):
                s -= H[i, j2] * y[j2]
            y[i] = s / H[i, i]

        x = x + V[:, :k_dim] @ y

        if callback is not None and (callback_type in (None, "x")):
            callback(x)

        r = b - A.matvec(x)
        if _norm(r) <= tol_th:
            info = 0
            break

        if not inner_converged and outer == maxiter - 1:
            info = total_iter

    return x, int(info)


def minres(
    A,
    b,
    x0: Optional[_ArrayLike] = None,
    *,
    shift: float = 0.0,
    tol: float = 1e-5,
    maxiter: Optional[int] = None,
    M=None,
    callback: Optional[Callable[[_ArrayLike], None]] = None,
    check: bool = False,
):
    try:
        import numpy as _np
        import scipy.sparse.linalg as _sla
    except Exception as exc:  # pragma: no cover - import guard
        raise NotImplementedError(
            "dpnp.scipy.sparse.linalg.minres currently requires SciPy on the host."
        ) from exc

    A_dp = aslinearoperator(A)
    m, n = A_dp.shape
    if m != n:
        raise ValueError("minres requires a square operator")

    def matvec_np(x_np):
        x_dp = _dpnp.asarray(x_np)
        y_dp = A_dp.matvec(x_dp)
        return _np.asarray(y_dp)

    A_sci = _sla.LinearOperator(
        shape=A_dp.shape, matvec=matvec_np, dtype=_np.dtype(A_dp.dtype)
    )

    if M is not None:
        M_dp = aslinearoperator(M)

        def m_matvec_np(x_np):
            x_dp = _dpnp.asarray(x_np)
            y_dp = M_dp.matvec(x_dp)
            return _np.asarray(y_dp)

        M_sci = _sla.LinearOperator(
            shape=M_dp.shape, matvec=m_matvec_np, dtype=_np.dtype(M_dp.dtype)
        )
    else:
        M_sci = None

    b_np = _np.asarray(_dpnp.asarray(b).reshape(-1))
    x0_np = None if x0 is None else _np.asarray(_dpnp.asarray(x0).reshape(-1))

    x_host, info = _sla.minres(
        A_sci,
        b_np,
        x0=x0_np,
        rtol=tol,
        shift=shift,
        maxiter=maxiter,
        M=M_sci,
        callback=callback,
        show=False,
        check=check,
    )

    x_dp = _dpnp.asarray(x_host)
    return x_dp, int(info)
