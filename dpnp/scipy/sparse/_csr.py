"""CSR matrix backed by dpnp/USM arrays.

Minimal implementation supporting the operations exercised by
dpnp.scipy.sparse.linalg solvers (cg, gmres, minres) and
LinearOperator. Construction from dense arrays or raw CSR
components; ``dot`` routed through oneMKL ``sparse::gemv`` when the
compiled backend extension is available, with a dense fallback used
when it is not.

Operations not required by the solvers (arithmetic, format
conversion, element-wise math, reductions, indexing) are
intentionally not implemented in this initial version.

SpMV fast path
--------------
On first ``.dot(x)`` with a 1-D ``x`` of a supported dtype, the
instance lazily allocates an oneMKL ``matrix_handle`` via
``_sparse_gemv_init`` (which itself runs ``set_csr_data`` plus
``optimize_gemv`` -- the expensive sparsity-analysis phase). The
handle is cached on the instance and reused for every subsequent
matvec; ``__del__`` releases it. This matches the cupyx behaviour
where ``csr_matrix.dot`` calls cuSPARSE SpMV directly without
densification, and lets the iterative solvers in
``dpnp.scipy.sparse.linalg`` reuse the same handle through
``_make_fast_matvec`` without rebuilding it.
"""

from __future__ import annotations

import numpy as _np
import dpnp as _dpnp

from ._base import SparseABC

# Value dtypes the oneMKL sparse::gemv dispatch table registers
# (see dpnp/backend/extensions/sparse/types_matrix.hpp). Anything
# outside this set must take the dense fallback in ``dot``.
_SPMV_VALUE_DTYPES = frozenset("fdFD")
# Index dtypes oneMKL accepts (int32, int64). Matches the second
# dimension of SparseGemvInitTypePairSupportFactory.
_SPMV_INDEX_DTYPES = frozenset("ilq")


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
        # Lazy SpMV handle state. Assigned BEFORE the dispatch below so
        # that __del__ never sees a partially-constructed object (it can
        # be invoked if any of the _init_* helpers raise).
        self._spmv_handle = None
        self._spmv_val_type_id = -1
        self._spmv_si = None
        self._spmv_exec_q = None

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

    # --- SpMV fast-path internals --------------------------------------

    def _spmv_supported(self):
        """True iff value and index dtypes are in the oneMKL dispatch table."""
        return (
            _np.dtype(self.data.dtype).char in _SPMV_VALUE_DTYPES
            and _np.dtype(self.indices.dtype).char in _SPMV_INDEX_DTYPES
        )

    def _ensure_spmv_handle(self):
        """Lazily build the cached oneMKL matrix_handle for forward SpMV.

        Returns the ``(si, handle, val_type_id, exec_q)`` quadruple so
        callers can drive ``_sparse_gemv_compute`` directly. Returns
        ``None`` if the compiled backend extension is unavailable, the
        dtype combination is unsupported, or handle construction fails
        for any backend-specific reason (in which case the caller must
        fall back to a dense path).
        """
        if self._spmv_handle is not None:
            return (
                self._spmv_si,
                self._spmv_handle,
                self._spmv_val_type_id,
                self._spmv_exec_q,
            )

        if not self._spmv_supported():
            return None

        try:
            # Lazy import: keeps csr_matrix importable in builds that
            # did not compile the sparse backend extension (e.g. host-
            # only test matrices, doc builds).
            # pylint: disable-next=import-outside-toplevel
            from dpnp.backend.extensions.sparse import _sparse_impl as _si
        except ImportError:
            return None

        exec_q = self.data.sycl_queue
        try:
            # pylint: disable-next=protected-access
            handle, val_type_id, ev = _si._sparse_gemv_init(
                exec_q,
                0,  # trans=N (forward)
                self.indptr,
                self.indices,
                self.data,
                int(self._shape[0]),
                int(self._shape[1]),
                int(self.data.shape[0]),
                [],
            )
        except Exception:  # pylint: disable=broad-exception-caught
            # Backend dispatch may reject the (value, index) pair even
            # though the Python guard above accepted them (e.g. complex
            # support disabled in the linked oneMKL build). Fall through
            # to the dense path silently.
            return None

        # set_csr_data + optimize_gemv must complete before any compute
        # call can dispatch against the handle. This is the only blocking
        # sync; subsequent matvecs return without waiting.
        ev.wait()

        self._spmv_si = _si
        self._spmv_handle = handle
        self._spmv_val_type_id = val_type_id
        self._spmv_exec_q = exec_q
        return (_si, handle, val_type_id, exec_q)

    # --- public API: matvec via cached oneMKL handle -------------------

    def dot(self, x):
        """Compute ``A @ x``.

        For a 1-D ``x`` of a supported dtype, this dispatches to oneMKL
        ``sparse::gemv`` via a cached matrix handle (built lazily on the
        first call). Subsequent calls reuse the handle and pay only the
        SpMV kernel cost, matching the cupyx ``csr_matrix.dot`` behaviour.

        Falls back to ``dpnp.dot(self.toarray(), x)`` when:

          * the compiled sparse backend extension is not present,
          * the value/index dtype combination is not in the oneMKL
            dispatch table, or
          * ``x`` is 2-D (no batched SpMV binding exists yet -- batched
            SpMM is a different oneMKL entry point and intentionally not
            wired up here).

        The dense fallback materialises ``self`` and is therefore O(M*N)
        in memory; the fast path is O(nnz) and never densifies.
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

        nrows, ncols = self._shape

        if x.ndim == 1 and x.shape[0] == ncols:
            handle_info = self._ensure_spmv_handle()
            if handle_info is not None:
                _si, handle, val_type_id, exec_q = handle_info
                # Reject dtype mismatches deterministically here rather
                # than letting the C++ layer raise: callers expect a
                # clean TypeError for cross-dtype matvec.
                if x.dtype != self.data.dtype:
                    raise TypeError(
                        f"csr_matrix.dot: x dtype {x.dtype} does not "
                        f"match matrix dtype {self.data.dtype}"
                    )
                y = _dpnp.empty(
                    nrows, dtype=self.data.dtype, sycl_queue=exec_q
                )
                # Do NOT wait on the returned event: any subsequent dpnp
                # operation on the same queue will serialise behind it
                # automatically. Blocking here would dominate runtime
                # for small systems (same rationale as _CachedSpMV in
                # linalg/_iterative.py).
                # pylint: disable-next=protected-access
                _si._sparse_gemv_compute(
                    exec_q,
                    handle,
                    val_type_id,
                    0,    # trans=N
                    1.0,  # alpha
                    x,
                    0.0,  # beta
                    y,
                    nrows,
                    ncols,
                    [],
                )
                return y

        # Dense fallback. Materialises ``self`` once -- this path is
        # exercised only when SpMV is unavailable for this matrix.
        return _dpnp.dot(self.toarray(), x)

    def __matmul__(self, x):
        return self.dot(x)

    def __del__(self):
        # Release the cached oneMKL matrix_handle if one was built.
        # See ``_iterative._CachedSpMV.__del__`` for the rationale of
        # the staged except clauses below: during interpreter shutdown
        # the compiled ``_sparse_impl`` extension may be GC'd before
        # this __del__ runs, in which case ``si._sparse_gemv_release``
        # evaluates to ``None``. Probe explicitly so a real backend
        # error (extension still healthy) is not silenced by the same
        # ``except Exception`` that catches the shutdown race.
        handle = getattr(self, "_spmv_handle", None)
        si = getattr(self, "_spmv_si", None)
        if handle is None or si is None:
            return

        release_fn = getattr(si, "_sparse_gemv_release", None)
        if release_fn is None:
            self._spmv_handle = None
            return

        try:
            release_fn(self._spmv_exec_q, handle, [])
        except (AttributeError, TypeError):
            # Shutdown-mode races; handle is unrecoverable and the
            # OS will reclaim it at process exit.
            pass
        except Exception:  # pylint: disable=broad-exception-caught
            # Genuine backend error while the interpreter is healthy.
            # Raising from __del__ produces only an unraisable warning
            # and the handle is gone either way -- swallow it
            # deliberately, distinct from the shutdown branch above.
            pass
        finally:
            self._spmv_handle = None

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
