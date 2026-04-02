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

from typing import Callable, Optional, Tuple

import dpnp as _dpnp


class LinearOperator:
    """DPNP-compatible linear operator.

    This is a lightweight implementation of
    :class:`scipy.sparse.linalg.LinearOperator` that operates on DPNP arrays
    and can be used with the iterative solvers in :mod:`dpnp.scipy.sparse.linalg`.
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        matvec: Callable,
        rmatvec: Optional[Callable] = None,
        matmat: Optional[Callable] = None,
        dtype=None,
    ) -> None:
        if len(shape) != 2:
            raise ValueError("LinearOperator shape must be length-2")

        m, n = shape
        if m < 0 or n < 0:
            raise ValueError("LinearOperator shape entries must be non-negative")

        self._shape = (int(m), int(n))
        self._matvec = matvec
        self._rmatvec = rmatvec
        self._matmat = matmat
        self._dtype = dtype

        if self._dtype is None:
            x0 = _dpnp.zeros(self._shape[1], dtype=_dpnp.int8)
            y0 = self._matvec(x0)
            self._dtype = _dpnp.asarray(y0).dtype

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self) -> int:
        return 2

    def _matvec_impl(self, x):
        return self._matvec(x)

    def _rmatvec_impl(self, x):
        if self._rmatvec is None:
            raise NotImplementedError("rmatvec is not defined for this LinearOperator")
        return self._rmatvec(x)

    def _matmat_impl(self, X):
        if self._matmat is not None:
            return self._matmat(X)

        X = _dpnp.atleast_2d(X)
        n, k = X.shape
        y = _dpnp.empty((self.shape[0], k), dtype=self.dtype)
        for j in range(k):
            y[:, j] = self._matvec_impl(X[:, j])
        return y

    def matvec(self, x):
        x = _dpnp.asarray(x)
        if x.ndim != 1:
            x = x.reshape(-1)
        if x.shape[0] != self.shape[1]:
            raise ValueError(
                "dimension mismatch in matvec: expected ({},), got {}".format(
                    self.shape[1], x.shape
                )
            )

        y = self._matvec_impl(x)
        y = _dpnp.asarray(y)
        if y.ndim != 1:
            y = y.reshape(-1)
        if y.shape[0] != self.shape[0]:
            raise ValueError(
                "LinearOperator matvec returned wrong shape: expected ({},), got {}".format(
                    self.shape[0], y.shape
                )
            )
        return y

    def rmatvec(self, x):
        x = _dpnp.asarray(x)
        if x.ndim != 1:
            x = x.reshape(-1)
        if x.shape[0] != self.shape[0]:
            raise ValueError(
                "dimension mismatch in rmatvec: expected ({},), got {}".format(
                    self.shape[0], x.shape
                )
            )

        y = self._rmatvec_impl(x)
        y = _dpnp.asarray(y)
        if y.ndim != 1:
            y = y.reshape(-1)
        if y.shape[0] != self.shape[1]:
            raise ValueError(
                "LinearOperator rmatvec returned wrong shape: expected ({},), got {}".format(
                    self.shape[1], y.shape
                )
            )
        return y

    def matmat(self, X):
        X = _dpnp.asarray(X)
        if X.ndim != 2:
            raise ValueError("matmat expects a 2-D array")
        if X.shape[0] != self.shape[1]:
            raise ValueError(
                "dimension mismatch in matmat: expected ({}, K), got {}".format(
                    self.shape[1], X.shape
                )
            )
        return _dpnp.asarray(self._matmat_impl(X))

    def __matmul__(self, x):
        x = _dpnp.asarray(x)
        if x.ndim == 1:
            return self.matvec(x)
        if x.ndim == 2:
            return self.matmat(x)
        raise ValueError("__matmul__ only supports 1-D or 2-D operands")

    def __call__(self, x):
        return self.__matmul__(x)

    def __repr__(self) -> str:
        return (
            "<{}x{} dpnp.scipy.sparse.linalg.LinearOperator with dtype={}>".format(
                self.shape[0], self.shape[1], self.dtype
            )
        )


def aslinearoperator(A) -> LinearOperator:
    if isinstance(A, LinearOperator):
        return A

    try:
        arr = _dpnp.asarray(A)
        if arr.ndim == 2:
            m, n = arr.shape

            def matvec(x):
                return arr @ x

            def rmatvec(x):
                return _dpnp.conj(arr.T) @ x

            return LinearOperator((m, n), matvec=matvec, rmatvec=rmatvec, dtype=arr.dtype)
    except Exception:
        pass

    if hasattr(A, "shape") and len(A.shape) == 2:
        m, n = A.shape

        if hasattr(A, "matvec"):
            def matvec(x):
                return A.matvec(x)
        else:
            def matvec(x):
                return A @ x

        rmatvec = None
        if hasattr(A, "rmatvec"):
            rmatvec = lambda x: A.rmatvec(x)

        return LinearOperator((m, n), matvec=matvec, rmatvec=rmatvec, dtype=getattr(A, "dtype", None))

    raise TypeError("Cannot convert object of type {} to LinearOperator".format(type(A)))
