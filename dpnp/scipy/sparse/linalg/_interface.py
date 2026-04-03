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

"""LinearOperator and helpers for dpnp.scipy.sparse.linalg.

Aligned with CuPy v14.0.1 cupyx/scipy/sparse/linalg/_interface.py
so that code written for cupyx or scipy.sparse.linalg is portable.
"""

from __future__ import annotations

import warnings

import dpnp


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _isshape(shape):
    if not isinstance(shape, tuple) or len(shape) != 2:
        return False
    return all(isinstance(s, int) and s >= 0 for s in shape)


def _isintlike(x):
    try:
        return int(x) == x
    except (TypeError, ValueError):
        return False


def _get_dtype(operators, dtypes=None):
    if dtypes is None:
        dtypes = []
    for obj in operators:
        if obj is not None and hasattr(obj, "dtype"):
            dtypes.append(obj.dtype)
    return dpnp.result_type(*dtypes)


# ---------------------------------------------------------------------------
# LinearOperator base
# ---------------------------------------------------------------------------

class LinearOperator:
    """Drop-in replacement for cupyx/scipy LinearOperator backed by dpnp arrays.

    Supports the full operator algebra (addition, multiplication, scaling,
    power, adjoint, transpose) matching CuPy v14.0.1 semantics.
    """

    ndim = 2

    def __new__(cls, *args, **kwargs):
        if cls is LinearOperator:
            # Factory: bare LinearOperator(shape, matvec=...) returns a
            # _CustomLinearOperator, exactly as SciPy / CuPy do.
            return super().__new__(_CustomLinearOperator)
        else:
            obj = super().__new__(cls)
            if (type(obj)._matvec is LinearOperator._matvec
                    and type(obj)._matmat is LinearOperator._matmat):
                warnings.warn(
                    "LinearOperator subclass should implement at least one of "
                    "_matvec and _matmat.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return obj

    def __init__(self, dtype, shape):
        if dtype is not None:
            dtype = dpnp.dtype(dtype)
        shape = tuple(shape)
        if not _isshape(shape):
            raise ValueError(
                f"invalid shape {shape!r} (must be a length-2 tuple of non-negative ints)"
            )
        self.dtype = dtype
        self.shape = shape

    def _init_dtype(self):
        """Infer dtype by running a trial matvec on a zero int8 vector."""
        if self.dtype is None:
            v = dpnp.zeros(self.shape[-1])
            self.dtype = self.matvec(v).dtype

    # ------------------------------------------------------------------ #
    #  Abstract primitives — subclasses override at least one of these    #
    # ------------------------------------------------------------------ #

    def _matvec(self, x):
        """Default: call matmat on a column vector."""
        return self.matmat(x.reshape(-1, 1))

    def _matmat(self, X):
        """Default: stack matvec calls — slow fallback."""
        return dpnp.hstack(
            [self.matvec(col.reshape(-1, 1)) for col in X.T]
        )

    def _rmatvec(self, x):
        if type(self)._adjoint is LinearOperator._adjoint:
            raise NotImplementedError(
                "rmatvec is not defined for this LinearOperator"
            )
        return self.H.matvec(x)

    def _rmatmat(self, X):
        if type(self)._adjoint is LinearOperator._adjoint:
            return dpnp.hstack(
                [self.rmatvec(col.reshape(-1, 1)) for col in X.T]
            )
        return self.H.matmat(X)

    # ------------------------------------------------------------------ #
    #  Public multiply methods (shape-checked)                            #
    # ------------------------------------------------------------------ #

    def matvec(self, x):
        M, N = self.shape
        if x.shape not in ((N,), (N, 1)):
            raise ValueError(
                f"dimension mismatch: operator shape {self.shape}, vector shape {x.shape}"
            )
        y = self._matvec(x)
        return y.reshape(M) if x.ndim == 1 else y.reshape(M, 1)

    def rmatvec(self, x):
        M, N = self.shape
        if x.shape not in ((M,), (M, 1)):
            raise ValueError(
                f"dimension mismatch: operator shape {self.shape}, vector shape {x.shape}"
            )
        y = self._rmatvec(x)
        return y.reshape(N) if x.ndim == 1 else y.reshape(N, 1)

    def matmat(self, X):
        if X.ndim != 2:
            raise ValueError(f"expected 2-D array, got {X.ndim}-D")
        if X.shape[0] != self.shape[1]:
            raise ValueError(
                f"dimension mismatch: {self.shape!r} vs {X.shape!r}"
            )
        return self._matmat(X)

    def rmatmat(self, X):
        if X.ndim != 2:
            raise ValueError(f"expected 2-D array, got {X.ndim}-D")
        if X.shape[0] != self.shape[0]:
            raise ValueError(
                f"dimension mismatch: {self.shape!r} vs {X.shape!r}"
            )
        return self._rmatmat(X)

    # ------------------------------------------------------------------ #
    #  Operator algebra                                                   #
    # ------------------------------------------------------------------ #

    def dot(self, x):
        if isinstance(x, LinearOperator):
            return _ProductLinearOperator(self, x)
        elif dpnp.isscalar(x):
            return _ScaledLinearOperator(self, x)
        else:
            x = dpnp.asarray(x)
            if x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1):
                return self.matvec(x)
            elif x.ndim == 2:
                return self.matmat(x)
            raise ValueError(f"expected 1-D or 2-D array or LinearOperator, got {x!r}")

    def __call__(self, x):
        return self * x

    def __mul__(self, x):
        return self.dot(x)

    def __matmul__(self, x):
        if dpnp.isscalar(x):
            raise ValueError("Scalar operands are not allowed with '@'; use '*' instead")
        return self.__mul__(x)

    def __rmatmul__(self, x):
        if dpnp.isscalar(x):
            raise ValueError("Scalar operands are not allowed with '@'; use '*' instead")
        return self.__rmul__(x)

    def __rmul__(self, x):
        if dpnp.isscalar(x):
            return _ScaledLinearOperator(self, x)
        return NotImplemented

    def __pow__(self, p):
        if dpnp.isscalar(p):
            return _PowerLinearOperator(self, p)
        return NotImplemented

    def __add__(self, x):
        if isinstance(x, LinearOperator):
            return _SumLinearOperator(self, x)
        return NotImplemented

    def __neg__(self):
        return _ScaledLinearOperator(self, -1)

    def __sub__(self, x):
        return self.__add__(-x)

    # ------------------------------------------------------------------ #
    #  Adjoint / transpose                                                #
    # ------------------------------------------------------------------ #

    def adjoint(self):
        """Return the conjugate-transpose (Hermitian adjoint) operator."""
        return self._adjoint()

    #: Property alias for adjoint() — A.H gives the Hermitian adjoint.
    H = property(adjoint)

    def transpose(self):
        """Return the (non-conjugated) transpose operator."""
        return self._transpose()

    #: Property alias for transpose() — A.T gives the plain transpose.
    T = property(transpose)

    def _adjoint(self):
        return _AdjointLinearOperator(self)

    def _transpose(self):
        return _TransposedLinearOperator(self)

    def __repr__(self):
        dt = "unspecified dtype" if self.dtype is None else f"dtype={self.dtype}"
        return f"<{self.shape[0]}x{self.shape[1]} {self.__class__.__name__} with {dt}>"


# ---------------------------------------------------------------------------
# Concrete operator classes
# ---------------------------------------------------------------------------

class _CustomLinearOperator(LinearOperator):
    """Created when the user calls LinearOperator(shape, matvec=...) directly."""

    def __init__(self, shape, matvec, rmatvec=None, matmat=None,
                 dtype=None, rmatmat=None):
        super().__init__(dtype, shape)
        self.args = ()
        self.__matvec_impl  = matvec
        self.__rmatvec_impl = rmatvec
        self.__rmatmat_impl = rmatmat
        self.__matmat_impl  = matmat
        self._init_dtype()

    def _matvec(self, x):
        return self.__matvec_impl(x)

    def _matmat(self, X):
        if self.__matmat_impl is not None:
            return self.__matmat_impl(X)
        return super()._matmat(X)

    def _rmatvec(self, x):
        if self.__rmatvec_impl is None:
            raise NotImplementedError("rmatvec is not defined for this operator")
        return self.__rmatvec_impl(x)

    def _rmatmat(self, X):
        if self.__rmatmat_impl is not None:
            return self.__rmatmat_impl(X)
        return super()._rmatmat(X)

    def _adjoint(self):
        return _CustomLinearOperator(
            shape=(self.shape[1], self.shape[0]),
            matvec=self.__rmatvec_impl,
            rmatvec=self.__matvec_impl,
            matmat=self.__rmatmat_impl,
            rmatmat=self.__matmat_impl,
            dtype=self.dtype,
        )


class _AdjointLinearOperator(LinearOperator):
    def __init__(self, A):
        super().__init__(A.dtype, (A.shape[1], A.shape[0]))
        self.A = A
        self.args = (A,)

    def _matvec(self, x):  return self.A._rmatvec(x)
    def _rmatvec(self, x): return self.A._matvec(x)
    def _matmat(self, X):  return self.A._rmatmat(X)
    def _rmatmat(self, X): return self.A._matmat(X)


class _TransposedLinearOperator(LinearOperator):
    def __init__(self, A):
        super().__init__(A.dtype, (A.shape[1], A.shape[0]))
        self.A = A
        self.args = (A,)

    def _matvec(self, x):  return dpnp.conj(self.A._rmatvec(dpnp.conj(x)))
    def _rmatvec(self, x): return dpnp.conj(self.A._matvec(dpnp.conj(x)))
    def _matmat(self, X):  return dpnp.conj(self.A._rmatmat(dpnp.conj(X)))
    def _rmatmat(self, X): return dpnp.conj(self.A._matmat(dpnp.conj(X)))


class _SumLinearOperator(LinearOperator):
    def __init__(self, A, B):
        if A.shape != B.shape:
            raise ValueError(f"shape mismatch for addition: {A!r} + {B!r}")
        super().__init__(_get_dtype([A, B]), A.shape)
        self.args = (A, B)

    def _matvec(self, x):  return self.args[0].matvec(x)  + self.args[1].matvec(x)
    def _rmatvec(self, x): return self.args[0].rmatvec(x) + self.args[1].rmatvec(x)
    def _matmat(self, X):  return self.args[0].matmat(X)  + self.args[1].matmat(X)
    def _rmatmat(self, X): return self.args[0].rmatmat(X) + self.args[1].rmatmat(X)
    def _adjoint(self):    return self.args[0].H + self.args[1].H


class _ProductLinearOperator(LinearOperator):
    def __init__(self, A, B):
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"shape mismatch for multiply: {A!r} * {B!r}")
        super().__init__(_get_dtype([A, B]), (A.shape[0], B.shape[1]))
        self.args = (A, B)

    def _matvec(self, x):  return self.args[0].matvec(self.args[1].matvec(x))
    def _rmatvec(self, x): return self.args[1].rmatvec(self.args[0].rmatvec(x))
    def _matmat(self, X):  return self.args[0].matmat(self.args[1].matmat(X))
    def _rmatmat(self, X): return self.args[1].rmatmat(self.args[0].rmatmat(X))
    def _adjoint(self):    A, B = self.args; return B.H * A.H


class _ScaledLinearOperator(LinearOperator):
    def __init__(self, A, alpha):
        super().__init__(_get_dtype([A], [type(alpha)]), A.shape)
        self.args = (A, alpha)

    def _matvec(self, x):  return self.args[1] * self.args[0].matvec(x)
    def _rmatvec(self, x): return dpnp.conj(self.args[1]) * self.args[0].rmatvec(x)
    def _matmat(self, X):  return self.args[1] * self.args[0].matmat(X)
    def _rmatmat(self, X): return dpnp.conj(self.args[1]) * self.args[0].rmatmat(X)
    def _adjoint(self):
        A, alpha = self.args
        return A.H * dpnp.conj(alpha)


class _PowerLinearOperator(LinearOperator):
    def __init__(self, A, p):
        if A.shape[0] != A.shape[1]:
            raise ValueError("matrix power requires a square operator")
        if not _isintlike(p) or p < 0:
            raise ValueError("matrix power requires a non-negative integer exponent")
        super().__init__(_get_dtype([A]), A.shape)
        self.args = (A, int(p))

    def _power(self, f, x):
        res = dpnp.array(x, copy=True)
        for _ in range(self.args[1]):
            res = f(res)
        return res

    def _matvec(self, x):  return self._power(self.args[0].matvec, x)
    def _rmatvec(self, x): return self._power(self.args[0].rmatvec, x)
    def _matmat(self, X):  return self._power(self.args[0].matmat, X)
    def _rmatmat(self, X): return self._power(self.args[0].rmatmat, X)
    def _adjoint(self):
        A, p = self.args
        return A.H ** p


class MatrixLinearOperator(LinearOperator):
    """Wrap a dense dpnp matrix (or sparse matrix) as a LinearOperator."""

    def __init__(self, A):
        super().__init__(A.dtype, A.shape)
        self.A = A
        self.__adj = None
        self.args = (A,)

    def _matmat(self, X):  return self.A.dot(X)
    def _rmatmat(self, X): return dpnp.conj(self.A.T).dot(X)

    def _adjoint(self):
        if self.__adj is None:
            self.__adj = _AdjointMatrixOperator(self)
        return self.__adj


class _AdjointMatrixOperator(MatrixLinearOperator):
    def __init__(self, adjoint):
        self.A = dpnp.conj(adjoint.A.T)
        self.__adjoint = adjoint
        self.args = (adjoint,)
        self.shape = (adjoint.shape[1], adjoint.shape[0])

    @property
    def dtype(self):
        return self.__adjoint.dtype

    def _adjoint(self):
        return self.__adjoint


class IdentityOperator(LinearOperator):
    """Identity operator — used as default preconditioner in _make_system."""

    def __init__(self, shape, dtype=None):
        super().__init__(dtype, shape)

    def _matvec(self, x):  return x
    def _rmatvec(self, x): return x
    def _matmat(self, X):  return X
    def _rmatmat(self, X): return X
    def _adjoint(self):    return self


# ---------------------------------------------------------------------------
# aslinearoperator
# ---------------------------------------------------------------------------

def aslinearoperator(A) -> LinearOperator:
    """Wrap A as a LinearOperator if it is not already one.

    Handles (in order):
      - Already a LinearOperator — returned as-is.
      - dpnp / scipy sparse matrix — wrapped in MatrixLinearOperator.
      - Dense dpnp / numpy ndarray — wrapped in MatrixLinearOperator.
      - Duck-typed objects with .shape and .matvec or @ support.
    """
    if isinstance(A, LinearOperator):
        return A

    # sparse matrix (dpnp.scipy.sparse or scipy.sparse)
    try:
        from dpnp.scipy import sparse as _sp
        if _sp.issparse(A):
            return MatrixLinearOperator(A)
    except (ImportError, AttributeError):
        pass

    try:
        import scipy.sparse as _ssp
        if _ssp.issparse(A):
            return MatrixLinearOperator(dpnp.asarray(A.toarray()))
    except (ImportError, AttributeError):
        pass

    # dense ndarray
    try:
        arr = dpnp.asarray(A)
        if arr.ndim == 2:
            return MatrixLinearOperator(arr)
    except Exception:
        pass

    # duck-typed
    if hasattr(A, "shape") and len(A.shape) == 2:
        m, n    = int(A.shape[0]), int(A.shape[1])
        dtype   = getattr(A, "dtype", None)
        matvec  = A.matvec  if hasattr(A, "matvec")  else (lambda x: A @ x)
        rmatvec = A.rmatvec if hasattr(A, "rmatvec") else None
        matmat  = A.matmat  if hasattr(A, "matmat")  else None
        rmatmat = A.rmatmat if hasattr(A, "rmatmat") else None
        return LinearOperator(
            (m, n),
            matvec=matvec,
            rmatvec=rmatvec,
            matmat=matmat,
            dtype=dtype,
            rmatmat=rmatmat,
        )

    raise TypeError(f"Cannot convert object of type {type(A)!r} to a LinearOperator")
