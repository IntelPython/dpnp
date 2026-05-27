"""CSR matrix backed by dpnp/USM arrays.

Minimal implementation supporting the operations exercised by
dpnp.scipy.sparse.linalg solvers (cg, gmres, minres) and
LinearOperator. Construction from dense arrays or raw CSR
components, plus dot/T for the solver fallback path.

Operations not required by the solvers (arithmetic, format
conversion, element-wise math, reductions, indexing) are
intentionally not implemented in this initial version.
"""

from __future__ import annotations

import numpy as _np
import dpnp as _dpnp

from ._base import SparseABC


class csr_matrix(SparseABC):
    """Compressed Sparse Row matrix on a SYCL device.

    Construction
    ------------
    csr_matrix(D)
        from a 2-D dpnp.ndarray.

    csr_matrix((data, indices, indptr), shape=(M, N))
        from raw CSR component arrays (1-D dpnp arrays on the same
        SYCL queue).

    csr_matrix(other_csr)
        copy of another csr_matrix.

    Attributes
    ----------
    data : dpnp.ndarray
        1-D array of nonzero values, shape (nnz,).
    indices : dpnp.ndarray
        1-D array of column indices, shape (nnz,).
    indptr : dpnp.ndarray
        1-D array of row pointers, shape (M+1,).
    shape : tuple of int
    dtype : dpnp dtype
    nnz : int
    format : str
        Always 'csr'.
    ndim : int
        Always 2.
    """

    format = "csr"
    ndim = 2

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        if isinstance(arg1, _dpnp.ndarray):
            self._init_from_dense(arg1, dtype=dtype)
        elif isinstance(arg1, csr_matrix):
            self._init_from_components(
                (arg1.data, arg1.indices, arg1.indptr),
                arg1.shape,
                dtype=dtype if dtype is not None else arg1.dtype,
                copy=True,
            )
        elif isinstance(arg1, tuple) and len(arg1) == 3:
            if shape is None:
                raise ValueError(
                    "csr_matrix: shape must be provided when constructing "
                    "from (data, indices, indptr) components"
                )
            self._init_from_components(arg1, shape, dtype=dtype, copy=copy)
        else:
            raise TypeError(
                f"csr_matrix: cannot construct from {type(arg1).__name__}; "
                "supported forms are a 2-D dpnp.ndarray, another csr_matrix, "
                "or a (data, indices, indptr) tuple with shape= kwarg."
            )

    def _init_from_components(self, arrays, shape, dtype=None, copy=False):
        data, indices, indptr = arrays

        if not (isinstance(data, _dpnp.ndarray)
                and isinstance(indices, _dpnp.ndarray)
                and isinstance(indptr, _dpnp.ndarray)):
            raise TypeError(
                "csr_matrix: data, indices, and indptr must be dpnp arrays"
            )
        if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
            raise ValueError(
                "csr_matrix: data, indices, and indptr must be 1-D"
            )
        if data.shape[0] != indices.shape[0]:
            raise ValueError(
                f"csr_matrix: data length {data.shape[0]} != "
                f"indices length {indices.shape[0]}"
            )

        nrows, ncols = int(shape[0]), int(shape[1])
        if indptr.shape[0] != nrows + 1:
            raise ValueError(
                f"csr_matrix: indptr length {indptr.shape[0]} != "
                f"nrows+1 ({nrows + 1})"
            )

        q = data.sycl_queue
        if indices.sycl_queue != q or indptr.sycl_queue != q:
            raise ValueError(
                "csr_matrix: data, indices, and indptr must be on the same "
                "SYCL queue"
            )

        idx_char = _np.dtype(indices.dtype).char
        if idx_char not in ('i', 'l', 'q'):
            raise TypeError(
                f"csr_matrix: indices dtype must be int32 or int64, "
                f"got {indices.dtype}"
            )
        if _np.dtype(indptr.dtype).char != idx_char:
            raise TypeError(
                f"csr_matrix: indptr dtype ({indptr.dtype}) must match "
                f"indices dtype ({indices.dtype})"
            )

        if dtype is not None and _np.dtype(dtype) != _np.dtype(data.dtype):
            data = data.astype(dtype, copy=True)
        elif copy:
            data = data.copy()
            indices = indices.copy()
            indptr = indptr.copy()

        self.data = data
        self.indices = indices
        self.indptr = indptr
        self._shape = (nrows, ncols)

    def _init_from_dense(self, D, dtype=None):
        if D.ndim != 2:
            raise ValueError(
                f"csr_matrix: dense input must be 2-D, got {D.ndim}-D"
            )
        if dtype is not None:
            D = D.astype(dtype, copy=False)

        nrows, ncols = D.shape
        q = D.sycl_queue

        rows, cols = _dpnp.nonzero(D)
        nnz = int(rows.shape[0])

        if nnz == 0:
            self.data = _dpnp.empty(0, dtype=D.dtype, sycl_queue=q)
            self.indices = _dpnp.empty(0, dtype=_dpnp.int64, sycl_queue=q)
            self.indptr = _dpnp.zeros(nrows + 1, dtype=_dpnp.int64,
                                       sycl_queue=q)
            self._shape = (nrows, ncols)
            return

        values = D[rows, cols]
        idx_dtype = _dpnp.int64
        row_counts = _dpnp.bincount(rows.astype(idx_dtype), minlength=nrows)
        indptr = _dpnp.empty(nrows + 1, dtype=idx_dtype, sycl_queue=q)
        indptr[0] = 0
        indptr[1:] = _dpnp.cumsum(row_counts)

        self.data = values
        self.indices = cols.astype(idx_dtype)
        self.indptr = indptr
        self._shape = (nrows, ncols)

    # --- read-only properties ------------------------------------------

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def nnz(self):
        return int(self.data.shape[0])

    @property
    def size(self):
        return self.nnz

    @property
    def T(self):
        """Transpose. Materializes via toarray() since CSC isn't implemented."""
        return csr_matrix(self.toarray().T)

    # --- methods used by the solver fallback path ----------------------

    def dot(self, x):
        """Compute A @ x. Fallback path for MatrixLinearOperator.

        The solver hot path bypasses this method entirely via
        _make_fast_matvec building a _CachedSpMV directly from
        the CSR component arrays.
        """
        if not isinstance(x, _dpnp.ndarray):
            raise TypeError(
                f"csr_matrix.dot: expected dpnp.ndarray, "
                f"got {type(x).__name__}"
            )
        if x.ndim not in (1, 2):
            raise ValueError(
                f"csr_matrix.dot: x must be 1-D or 2-D, got {x.ndim}-D"
            )
        # Dense fallback. Correct but materializes A; acceptable here
        # because this path only runs when oneMKL SpMV is unavailable.
        return _dpnp.dot(self.toarray(), x)

    def toarray(self):
        """Convert to a dense dpnp 2-D array."""
        nrows, ncols = self._shape
        q = self.data.sycl_queue
        D = _dpnp.zeros(self._shape, dtype=self.dtype, sycl_queue=q)
        if self.nnz == 0:
            return D

        row_lengths = self.indptr[1:] - self.indptr[:-1]
        rows = _dpnp.repeat(
            _dpnp.arange(nrows, dtype=self.indices.dtype, sycl_queue=q),
            row_lengths,
        )
        D[rows, self.indices] = self.data
        return D

    def copy(self):
        return csr_matrix(self)

    def __repr__(self):
        return (
            f"<{self._shape[0]}x{self._shape[1]} csr_matrix "
            f"of dtype {self.dtype} with {self.nnz} stored elements>"
        )
