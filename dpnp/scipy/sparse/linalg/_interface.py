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

"""LinearOperator and helpers for dpnp.scipy.sparse.linalg.

Aligned with SciPy main scipy/sparse/linalg/_interface.py and
CuPy v14.0.1 cupyx/scipy/sparse/linalg/_interface.py so that code
written for either library is portable to dpnp.
"""

# Math-heavy module: single-letter and CamelCase identifiers such as
# A, B, M, N, X, V, H are part of the published linear-algebra API and
# mirror SciPy/CuPy verbatim, so the snake_case rule is intentionally
# relaxed for the whole file.
# pylint: disable=invalid-name

from __future__ import annotations

import warnings

import dpnp

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _isshape(shape):
    """Return True if shape is a length-2 tuple of non-negative integers."""
    if not isinstance(shape, tuple) or len(shape) != 2:
        return False
    try:
        return all(int(s) >= 0 and int(s) == s for s in shape)
    except (TypeError, ValueError):
        return False


def _isintlike(x):
    try:
        return int(x) == x
    except (TypeError, ValueError):
        return False


def _get_dtype(operators, dtypes=None):
    if dtypes is None:
        dtypes = []
    for obj in operators:
        if obj is not None and hasattr(obj, "dtype") and obj.dtype is not None:
            dtypes.append(obj.dtype)
    return dpnp.result_type(*dtypes) if dtypes else None


class LinearOperator:
    """Drop-in replacement for cupyx/scipy LinearOperator backed by dpnp arrays.

    Supports the full operator algebra (addition, multiplication, scaling,
    power, adjoint A.H, transpose A.T) matching CuPy v14.0.1 and SciPy main.
    """

    ndim = 2

    # Opt out of NumPy's ufunc dispatch (NEP 13); defers ``host_array *
    # linop`` etc. to ``LinearOperator.__rmul__`` / ``__rmatmul__``.
    # Same convention as ``dpnp.ndarray``.
    __array_ufunc__ = None

    def __new__(cls, *args, **kwargs):
        if cls is LinearOperator:
            return super().__new__(_CustomLinearOperator)
        obj = super().__new__(cls)
        if (
            type(obj)._matvec is LinearOperator._matvec
            and type(obj)._matmat is LinearOperator._matmat
        ):
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
        shape = tuple(int(s) for s in shape)
        if not _isshape(shape):
            raise ValueError(
                f"invalid shape {shape!r} (must be a length-2 tuple of "
                "non-negative ints)"
            )
        self.dtype = dtype
        self.shape = shape

    def _init_dtype(self):
        """Infer dtype via a trial matvec on an int8 zero vector.

        Using ``int8`` (the lowest precedence numeric dtype) lets the
        matvec promote to its natural output type without artificially
        widening the result -- a float32 operator stays float32, a
        complex64 operator stays complex64, etc.  Mirrors the behaviour
        of ``scipy.sparse.linalg.LinearOperator._init_dtype`` and
        ``cupyx.scipy.sparse.linalg.LinearOperator._init_dtype``.

        A previous version used ``dpnp.float64`` here, which silently
        upcast every dtype-inferred operator to float64; that broke
        single-precision and complex-single workflows.
        """
        if self.dtype is not None:
            return
        v = dpnp.zeros(self.shape[-1], dtype=dpnp.int8)
        self.dtype = self.matvec(v).dtype

    def _matvec(self, x):
        return self.matmat(x.reshape(-1, 1))

    def _matmat(self, X):
        return dpnp.hstack([self.matvec(col.reshape(-1, 1)) for col in X.T])

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

    def matvec(self, x):
        """Apply the matrix-vector product."""
        M, N = self.shape
        if x.shape not in ((N,), (N, 1)):
            raise ValueError(
                f"dimension mismatch: operator shape {self.shape}, "
                f"vector shape {x.shape}"
            )
        y = self._matvec(x)
        return y.reshape(M) if x.ndim == 1 else y.reshape(M, 1)

    def rmatvec(self, x):
        """Apply the adjoint matrix-vector product."""
        M, N = self.shape
        if x.shape not in ((M,), (M, 1)):
            raise ValueError(
                f"dimension mismatch: operator shape {self.shape}, "
                f"vector shape {x.shape}"
            )
        y = self._rmatvec(x)
        return y.reshape(N) if x.ndim == 1 else y.reshape(N, 1)

    def matmat(self, X):
        """Apply the matrix-matrix product."""
        if X.ndim != 2:
            raise ValueError(f"expected 2-D array, got {X.ndim}-D")
        if X.shape[0] != self.shape[1]:
            raise ValueError(
                f"dimension mismatch: {self.shape!r} vs {X.shape!r}"
            )
        return self._matmat(X)

    def rmatmat(self, X):
        """Apply the adjoint matrix-matrix product."""
        if X.ndim != 2:
            raise ValueError(f"expected 2-D array, got {X.ndim}-D")
        if X.shape[0] != self.shape[0]:
            raise ValueError(
                f"dimension mismatch: {self.shape!r} vs {X.shape!r}"
            )
        return self._rmatmat(X)

    def dot(self, x):
        """Dispatch to matvec / matmat / scalar-scale / product.

        Strict-coercion contract (matches the rest of dpnp): the only
        accepted types are :class:`LinearOperator`, a true scalar
        (Python / NumPy / dpnp 0-D), or a :class:`dpnp.ndarray` of
        rank 1 or 2. A host :class:`numpy.ndarray` is rejected with
        a directed :class:`TypeError`; silently calling
        ``dpnp.asarray(x)`` here would upload the host array to the
        device on every matvec, masking real bugs in caller code
        about device / queue selection. The user must convert
        explicitly via ``dpnp.asarray(x)`` before passing in.
        """
        if isinstance(x, LinearOperator):
            return _ProductLinearOperator(self, x)
        if dpnp.isscalar(x):
            return _ScaledLinearOperator(self, x)
        if not isinstance(x, dpnp.ndarray):
            # pylint: disable-next=import-outside-toplevel
            import numpy as _np

            if isinstance(x, _np.ndarray):
                raise TypeError(
                    "LinearOperator.dot: got a numpy.ndarray. dpnp "
                    "does not perform implicit host -> device "
                    "copies; pass dpnp.asarray(x) explicitly."
                )
            raise TypeError(
                "LinearOperator.dot: expected a dpnp.ndarray, a "
                "scalar, or another LinearOperator; got "
                f"{type(x).__name__!r}."
            )
        if x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1):
            return self.matvec(x)
        if x.ndim == 2:
            return self.matmat(x)
        raise ValueError(
            f"LinearOperator.dot: expected 1-D or 2-D dpnp array, "
            f"got {x.ndim}-D"
        )

    def __call__(self, x):
        return self * x

    def __mul__(self, x):
        """Multiply operator by array x."""
        return self.dot(x)

    def __matmul__(self, x):
        if dpnp.isscalar(x):
            raise ValueError(
                "Scalar operands not allowed with '@'; use '*' instead"
            )
        return self.__mul__(x)

    def __rmatmul__(self, x):
        if dpnp.isscalar(x):
            raise ValueError(
                "Scalar operands not allowed with '@'; use '*' instead"
            )
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

    def _adjoint(self):
        """Return conjugate-transpose operator (override in subclasses)."""
        return _AdjointLinearOperator(self)

    def _transpose(self):
        """Return plain-transpose operator (override in subclasses)."""
        return _TransposedLinearOperator(self)

    def adjoint(self):
        """Hermitian adjoint A^H."""
        return self._adjoint()

    def transpose(self):
        """Plain (non-conjugated) transpose A^T."""
        return self._transpose()

    #: A.H — conjugate transpose
    H = property(adjoint)
    #: A.T — plain transpose
    T = property(transpose)

    def __repr__(self):
        dt = (
            "unspecified dtype" if self.dtype is None else f"dtype={self.dtype}"
        )
        return (
            f"<{self.shape[0]}x{self.shape[1]}"
            f" {self.__class__.__name__} with {dt}>"
        )


class _CustomLinearOperator(LinearOperator):
    """Created when the user calls LinearOperator(shape, matvec=...)"""

    def __init__(
        self, shape, matvec, rmatvec=None, matmat=None, dtype=None, rmatmat=None
    ):
        super().__init__(dtype, shape)
        self.args = ()
        self.__matvec_impl = matvec
        self.__rmatvec_impl = rmatvec
        self.__rmatmat_impl = rmatmat
        self.__matmat_impl = matmat
        self._init_dtype()

    def _matvec(self, x):
        return self.__matvec_impl(x)

    def _matmat(self, X):
        if self.__matmat_impl is not None:
            return self.__matmat_impl(X)
        return super()._matmat(X)

    def _rmatvec(self, x):
        if self.__rmatvec_impl is None:
            raise NotImplementedError(
                "rmatvec is not defined for this operator"
            )
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

    def _matvec(self, x):
        return self.A._rmatvec(x)  # pylint: disable=protected-access

    def _rmatvec(self, x):
        return self.A._matvec(x)  # pylint: disable=protected-access

    def _matmat(self, X):
        return self.A._rmatmat(X)  # pylint: disable=protected-access

    def _rmatmat(self, X):
        return self.A._matmat(X)  # pylint: disable=protected-access

    def _adjoint(self):
        return self.A


class _TransposedLinearOperator(LinearOperator):
    def __init__(self, A):
        super().__init__(A.dtype, (A.shape[1], A.shape[0]))
        self.A = A
        self.args = (A,)

    def _matvec(self, x):
        # pylint: disable=protected-access
        return dpnp.conj(self.A._rmatvec(dpnp.conj(x)))

    def _rmatvec(self, x):
        # pylint: disable=protected-access
        return dpnp.conj(self.A._matvec(dpnp.conj(x)))

    def _matmat(self, X):
        # pylint: disable=protected-access
        return dpnp.conj(self.A._rmatmat(dpnp.conj(X)))

    def _rmatmat(self, X):
        # pylint: disable=protected-access
        return dpnp.conj(self.A._matmat(dpnp.conj(X)))

    def _transpose(self):
        return self.A


class _SumLinearOperator(LinearOperator):
    def __init__(self, A, B):
        if A.shape != B.shape:
            raise ValueError(f"shape mismatch for addition: {A!r} + {B!r}")
        super().__init__(_get_dtype([A, B]), A.shape)
        self.args = (A, B)

    def _matvec(self, x):
        return self.args[0].matvec(x) + self.args[1].matvec(x)

    def _rmatvec(self, x):
        return self.args[0].rmatvec(x) + self.args[1].rmatvec(x)

    def _matmat(self, X):
        return self.args[0].matmat(X) + self.args[1].matmat(X)

    def _rmatmat(self, X):
        return self.args[0].rmatmat(X) + self.args[1].rmatmat(X)

    def _adjoint(self):
        return self.args[0].H + self.args[1].H


class _ProductLinearOperator(LinearOperator):
    def __init__(self, A, B):
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"shape mismatch for multiply: {A!r} * {B!r}")
        super().__init__(_get_dtype([A, B]), (A.shape[0], B.shape[1]))
        self.args = (A, B)

    def _matvec(self, x):
        return self.args[0].matvec(self.args[1].matvec(x))

    def _rmatvec(self, x):
        return self.args[1].rmatvec(self.args[0].rmatvec(x))

    def _matmat(self, X):
        return self.args[0].matmat(self.args[1].matmat(X))

    def _rmatmat(self, X):
        return self.args[1].rmatmat(self.args[0].rmatmat(X))

    def _adjoint(self):
        A, B = self.args
        return B.H * A.H


class _ScaledLinearOperator(LinearOperator):
    def __init__(self, A, alpha):
        alpha_dtype = getattr(alpha, "dtype", type(alpha))
        super().__init__(_get_dtype([A], [alpha_dtype]), A.shape)
        self.args = (A, alpha)

    def _matvec(self, x):
        return self.args[1] * self.args[0].matvec(x)

    def _rmatvec(self, x):
        return self.args[1].conjugate() * self.args[0].rmatvec(x)

    def _matmat(self, X):
        return self.args[1] * self.args[0].matmat(X)

    def _rmatmat(self, X):
        return self.args[1].conjugate() * self.args[0].rmatmat(X)

    def _adjoint(self):
        A, alpha = self.args
        return A.H * alpha.conjugate()


class _PowerLinearOperator(LinearOperator):
    def __init__(self, A, p):
        if A.shape[0] != A.shape[1]:
            raise ValueError("matrix power requires a square operator")
        if not _isintlike(p) or p < 0:
            raise ValueError(
                "matrix power requires a non-negative integer exponent"
            )
        super().__init__(_get_dtype([A]), A.shape)
        self.args = (A, int(p))

    def _power(self, f, x):
        res = x.copy()
        for _ in range(self.args[1]):
            res = f(res)
        return res

    def _matvec(self, x):
        return self._power(self.args[0].matvec, x)

    def _rmatvec(self, x):
        return self._power(self.args[0].rmatvec, x)

    def _matmat(self, X):
        return self._power(self.args[0].matmat, X)

    def _rmatmat(self, X):
        return self._power(self.args[0].rmatmat, X)

    def _adjoint(self):
        A, p = self.args
        return A.H**p


class MatrixLinearOperator(LinearOperator):
    """Wrap a dense dpnp matrix (or sparse matrix) as a LinearOperator."""

    def __init__(self, A):
        super().__init__(A.dtype, A.shape)
        self.A = A
        self.__adj = None
        self.args = (A,)

    def _matmat(self, X):
        return self.A.dot(X)

    def _rmatmat(self, X):
        return dpnp.conj(self.A.T).dot(X)

    def _adjoint(self):
        if self.__adj is None:
            self.__adj = _AdjointMatrixOperator(self)
        return self.__adj


class _AdjointMatrixOperator(MatrixLinearOperator):
    # super().__init__() is intentionally skipped: this operator stores its
    # own (adjoint-derived) A, shape and dtype, and must NOT re-validate
    # shape via the base ``MatrixLinearOperator.__init__`` path.
    # pylint: disable=super-init-not-called
    def __init__(self, adjoint):
        self.A = dpnp.conj(adjoint.A.T)
        self.__adjoint = adjoint
        self.args = (adjoint,)
        self.shape = (adjoint.shape[1], adjoint.shape[0])

    @property
    def dtype(self):
        """Inherit dtype from the wrapped operator."""
        return self.__adjoint.dtype

    def _adjoint(self):
        return self.__adjoint


class IdentityOperator(LinearOperator):
    """Identity operator — used as the default (no-op) preconditioner."""

    def __init__(self, shape, dtype=None):
        super().__init__(dtype, shape)

    def _matvec(self, x):
        """Apply matrix-vector product via stored array."""
        return x

    def _rmatvec(self, x):
        return x

    def _matmat(self, X):
        return X

    def _rmatmat(self, X):
        return X

    def _adjoint(self):
        return self

    def _transpose(self):
        return self


def aslinearoperator(A) -> LinearOperator:
    """Return ``A`` as a :class:`LinearOperator`.

    Dispatch order (matches ``cupyx.scipy.sparse.linalg.aslinearoperator``
    and ``scipy.sparse.linalg.aslinearoperator``):

      1. Already a :class:`LinearOperator` -- returned as-is.
      2. A ``dpnp.scipy.sparse`` sparse matrix (e.g. ``csr_matrix``)
         -- wrapped as :class:`MatrixLinearOperator`. Inside the iterative
         solvers this wrapper is further specialised to a cached oneMKL
         SpMV handle in ``_iterative._make_fast_matvec`` so the dense
         materialisation in ``csr_matrix.dot`` is bypassed.
      3. A dense 2-D :class:`dpnp.ndarray` -- wrapped as
         :class:`MatrixLinearOperator` after promotion via
         :func:`dpnp.atleast_2d`.
      4. A duck-typed object with ``.shape`` and ``.matvec``
         (optionally ``rmatvec`` / ``matmat`` / ``rmatmat`` / ``dtype``).

    Notes
    -----
    A :class:`numpy.ndarray` is explicitly rejected: silently promoting
    a host array would force a hidden host->device copy on every
    matvec, defeating the point of routing through dpnp. Callers must
    explicitly transfer with ``dpnp.asarray`` first.
    """
    # 1. Already a LinearOperator -- pass through.
    if isinstance(A, LinearOperator):
        return A

    # 2. dpnp sparse matrix. Import is at module-load time -- if
    # dpnp.scipy.sparse is unimportable then the package itself is
    # broken and a hard failure is preferable to silent fallthrough.
    # The local import avoids the package-init circularity that exists
    # while dpnp.scipy.sparse.__init__ is still executing (it imports
    # us via linalg/__init__.py).
    # pylint: disable-next=import-outside-toplevel
    from dpnp.scipy.sparse import issparse

    if issparse(A):
        return MatrixLinearOperator(A)

    # 3. Dense dpnp array.
    if isinstance(A, dpnp.ndarray):
        if A.ndim > 2:
            raise ValueError(
                f"aslinearoperator: dpnp array must be at most 2-D, "
                f"got {A.ndim}-D"
            )
        return MatrixLinearOperator(dpnp.atleast_2d(A))

    # pylint: disable-next=import-outside-toplevel
    import numpy as _np

    if isinstance(A, _np.ndarray):
        raise TypeError(
            "aslinearoperator: got a numpy.ndarray; transfer it to "
            "the target device with dpnp.asarray(A) first."
        )

    # 4. Duck-typed object with .shape and .matvec.
    if hasattr(A, "shape") and hasattr(A, "matvec"):
        shape = tuple(A.shape)
        if len(shape) != 2:
            raise ValueError(
                f"aslinearoperator: duck-typed operator must be 2-D, "
                f"got shape {shape!r}"
            )
        return LinearOperator(
            shape,
            matvec=A.matvec,
            rmatvec=getattr(A, "rmatvec", None),
            matmat=getattr(A, "matmat", None),
            rmatmat=getattr(A, "rmatmat", None),
            dtype=getattr(A, "dtype", None),
        )

    raise TypeError(
        f"aslinearoperator: cannot convert object of type {type(A).__name__!r} "
        "to a LinearOperator. Expected a LinearOperator, a dpnp sparse "
        "matrix, a 2-D dpnp.ndarray, or an object with .shape and .matvec."
    )
