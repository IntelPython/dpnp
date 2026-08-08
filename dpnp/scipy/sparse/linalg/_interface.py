# *****************************************************************************
# Copyright (c) 2026, Intel Corporation
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

import numpy as _np

import dpnp

from ..._lib._sparse import issparse

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
    """
    Common interface for performing matrix-vector products, backed by dpnp
    arrays.

    Iterative solvers (``cg``, ``gmres``, ``minres``) only require the
    matrix-vector product ``A @ v`` and never the individual matrix
    entries. This class is the abstract interface between such solvers and
    matrix-like objects. Construct it either by passing callables to the
    constructor, or by subclassing and implementing ``_matvec`` (and
    optionally ``_rmatvec`` / ``_matmat`` / ``_rmatmat``). It also supports
    the full operator algebra (``+``, ``@``, scaling, power, adjoint ``A.H``,
    transpose ``A.T``), each producing a new lazy ``LinearOperator``.

    For full documentation refer to :obj:`scipy.sparse.linalg.LinearOperator`.

    Parameters
    ----------
    shape : tuple of int
        Operator dimensions ``(M, N)``.
    matvec : callable
        Returns ``A @ v`` for a 1-D `v`.
    rmatvec : callable, optional
        Returns ``A^H @ v`` (conjugate transpose applied to `v`).
    matmat : callable, optional
        Returns ``A @ V`` for a dense 2-D `V` of shape ``(N, K)``.
    dtype : dtype, optional
        Data type of the operator. Inferred from a trial ``matvec`` when
        ``None``.
    rmatmat : callable, optional
        Returns ``A^H @ V`` for a dense 2-D `V` of shape ``(M, K)``.

    Attributes
    ----------
    args : tuple
        For composite operators (sum, product, ...), the operands of the
        binary operation.
    ndim : int
        Number of dimensions, always ``2``.
    """

    ndim = 2

    # Opt out of NumPy's ufunc (NEP 13) and function (NEP 18) dispatch;
    # defers ``host_array * linop`` / ``numpy.dot(linop, x)`` etc. to
    # ``LinearOperator``'s own operators instead of materializing a host
    # array. Same convention as ``dpnp.ndarray`` and SciPy's LinearOperator.
    __array_ufunc__ = None
    __array_function__ = None

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
        # newaxis (not ``.reshape``) so a bare usm_ndarray input works too.
        return self._matmat(x[..., None])[..., 0]

    def _matmat(self, X):
        return dpnp.stack(
            [self._matvec(X[:, i]) for i in range(X.shape[1])], axis=-1
        )

    def _rmatvec(self, x):
        if type(self)._adjoint is LinearOperator._adjoint:
            raise NotImplementedError(
                "rmatvec is not defined for this LinearOperator"
            )
        return self.H.matvec(x)

    def _rmatmat(self, X):
        if type(self)._adjoint is LinearOperator._adjoint:
            return dpnp.stack(
                [self._rmatvec(X[:, i]) for i in range(X.shape[1])], axis=-1
            )
        return self.H.matmat(X)

    def matvec(self, x):
        """
        Matrix-vector multiplication ``y = A @ x``.

        Parameters
        ----------
        x : {dpnp.ndarray, usm_ndarray}
            An array with shape ``(N,)`` or ``(N, 1)``.

        Returns
        -------
        out : dpnp.ndarray
            An array with shape ``(M,)`` or ``(M, 1)`` matching the rank
            of `x`.
        """
        M, N = self.shape
        if x.shape not in ((N,), (N, 1)):
            raise ValueError(
                f"dimension mismatch: operator shape {self.shape}, "
                f"vector shape {x.shape}"
            )
        y = self._matvec(x)
        return y.reshape(M) if x.ndim == 1 else y.reshape(M, 1)

    def rmatvec(self, x):
        """
        Adjoint matrix-vector multiplication ``y = A^H @ x``.

        Parameters
        ----------
        x : {dpnp.ndarray, usm_ndarray}
            An array with shape ``(M,)`` or ``(M, 1)``.

        Returns
        -------
        out : dpnp.ndarray
            An array with shape ``(N,)`` or ``(N, 1)`` matching the rank
            of `x`.
        """
        M, N = self.shape
        if x.shape not in ((M,), (M, 1)):
            raise ValueError(
                f"dimension mismatch: operator shape {self.shape}, "
                f"vector shape {x.shape}"
            )
        y = self._rmatvec(x)
        return y.reshape(N) if x.ndim == 1 else y.reshape(N, 1)

    def matmat(self, X):
        """
        Matrix-matrix multiplication ``Y = A @ X``.

        Parameters
        ----------
        X : {dpnp.ndarray, usm_ndarray}
            A 2-D array with shape ``(N, K)``.

        Returns
        -------
        out : dpnp.ndarray
            A 2-D array with shape ``(M, K)``.
        """
        if X.ndim != 2:
            raise ValueError(f"expected 2-D array, got {X.ndim}-D")
        if X.shape[0] != self.shape[1]:
            raise ValueError(
                f"dimension mismatch: {self.shape!r} vs {X.shape!r}"
            )
        return self._matmat(X)

    def rmatmat(self, X):
        """
        Adjoint matrix-matrix multiplication ``Y = A^H @ X``.

        Parameters
        ----------
        X : {dpnp.ndarray, usm_ndarray}
            A 2-D array with shape ``(M, K)``.

        Returns
        -------
        out : dpnp.ndarray
            A 2-D array with shape ``(N, K)``.
        """
        if X.ndim != 2:
            raise ValueError(f"expected 2-D array, got {X.ndim}-D")
        if X.shape[0] != self.shape[0]:
            raise ValueError(
                f"dimension mismatch: {self.shape!r} vs {X.shape!r}"
            )
        return self._rmatmat(X)

    def dot(self, x):
        """
        Matrix-matrix or matrix-vector multiplication.

        Parameters
        ----------
        x : {LinearOperator, scalar, dpnp.ndarray, usm_ndarray}
            Right operand. A 1-D or 2-D array is applied via ``matvec`` /
            ``matmat``; a scalar scales the operator; another
            ``LinearOperator`` forms a product operator.

        Returns
        -------
        out : {LinearOperator, dpnp.ndarray}
            The product operator (scalar / operator operands) or the
            resulting array (array operand).

        Notes
        -----
        A host :class:`numpy.ndarray` is rejected: dpnp does not perform
        implicit host-to-device copies. Transfer it with ``dpnp.asarray``
        first.
        """
        if isinstance(x, LinearOperator):
            return _ProductLinearOperator(self, x)
        if dpnp.isscalar(x):
            return _ScaledLinearOperator(self, x)
        if not dpnp.is_supported_array_type(x):
            if isinstance(x, _np.ndarray):
                raise TypeError(
                    "LinearOperator.dot: got a numpy.ndarray. dpnp "
                    "does not perform implicit host -> device "
                    "copies; pass dpnp.asarray(x) explicitly."
                )
            raise TypeError(
                "LinearOperator.dot: expected a dpnp or usm_ndarray, a "
                "scalar, or another LinearOperator; got "
                f"{type(x).__name__!r}."
            )
        if x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1):
            return self.matvec(x)
        if x.ndim == 2:
            return self.matmat(x)
        raise ValueError(
            f"LinearOperator.dot: expected 1-D or 2-D array, " f"got {x.ndim}-D"
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

    def _matvec(self, x):
        # csr_matrix.dot is 1-D-only (like cupyx); x is already 1-D here.
        if issparse(self.A):
            return self.A.dot(x)
        return super()._matvec(x)

    def _matmat(self, X):
        # No native SpMM: emulate as a column loop of 1-D SpMVs (no densify).
        if issparse(self.A):
            return dpnp.stack(
                [self.A.dot(X[:, i]) for i in range(X.shape[1])], axis=-1
            )
        return self.A.dot(X)

    def _rmatmat(self, X):
        if issparse(self.A):
            raise NotImplementedError(
                "rmatvec/adjoint is not supported for sparse csr_matrix "
                "operators; only the forward matvec is implemented."
            )
        return dpnp.conj(self.A.T).dot(X)

    def _adjoint(self):
        if issparse(self.A):
            raise NotImplementedError(
                "rmatvec/adjoint is not supported for sparse csr_matrix "
                "operators; only the forward matvec is implemented."
            )
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
    """
    Return `A` as a :class:`LinearOperator`.

    For full documentation refer to
    :obj:`scipy.sparse.linalg.aslinearoperator`.

    Parameters
    ----------
    A : object
        The object to wrap. It may be any of the following:

          * a :class:`LinearOperator` (returned unchanged);
          * a ``dpnp.scipy.sparse`` sparse matrix, e.g. ``csr_matrix``
            (the iterative solvers further specialise this to a cached
            oneMKL SpMV handle, bypassing densification);
          * a 2-D array, ``dpnp.ndarray`` or ``usm_ndarray`` (promoted
            via :func:`dpnp.atleast_2d`);
          * an object exposing ``.shape`` and ``.matvec`` (and optionally
            ``rmatvec`` / ``matmat`` / ``rmatmat`` / ``dtype``).

    Returns
    -------
    out : LinearOperator
        The `A` operand wrapped as a :class:`LinearOperator`.

    See Also
    --------
    :obj:`dpnp.scipy.sparse.linalg.LinearOperator` : The wrapped type.

    Notes
    -----
    A host :class:`numpy.ndarray` is rejected: dpnp does not perform
    implicit host-to-device copies, which would defeat routing through
    the device. Transfer it with ``dpnp.asarray`` first.
    """
    # 1. Already a LinearOperator -- pass through.
    if isinstance(A, LinearOperator):
        return A

    # 2. dpnp sparse matrix.
    if issparse(A):
        return MatrixLinearOperator(A)

    # 3. Dense dpnp.ndarray or usm_ndarray.
    if dpnp.is_supported_array_type(A):
        if A.ndim > 2:
            raise ValueError(
                f"aslinearoperator: array must be at most 2-D, "
                f"got {A.ndim}-D"
            )
        return MatrixLinearOperator(dpnp.atleast_2d(dpnp.asarray(A)))

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
