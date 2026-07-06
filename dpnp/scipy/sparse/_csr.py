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

import dpctl.utils as _dpu
import numpy as _np

import dpnp as _dpnp

from ._base import SparseABC

# Two short blocks intentionally mirror code in
# dpnp/scipy/sparse/linalg/_iterative.py: the cached-SpMV invocation
# and the __del__ shutdown-safe release pattern. Both are tightly
# coupled to oneMKL's contract; extracting a shared helper would add
# indirection without reducing real duplication.
# pylint: disable=duplicate-code

# Value dtypes the oneMKL sparse::gemv dispatch table registers
# (see dpnp/backend/extensions/sparse/types_matrix.hpp). Anything
# outside this set must take the dense fallback in ``dot``.
_SPMV_VALUE_DTYPES = frozenset("fdFD")
# Index dtypes oneMKL accepts (int32, int64). Matches the second
# dimension of SparseGemvInitTypePairSupportFactory.
_SPMV_INDEX_DTYPES = frozenset("ilq")


# pylint: disable=invalid-name,too-many-instance-attributes
# The class name ``csr_matrix`` is the public scipy/cupy API spelling and
# must stay lowercase. The instance-attribute count exceeds the default
# pylint cap because the lazily-built oneMKL handle adds four cache
# fields (handle, val_type_id, si, exec_q) on top of the CSR triple +
# shape; all are required.
class csr_matrix(SparseABC):
    """Compressed Sparse Row matrix on a SYCL device.

    Construction
    ------------
    csr_matrix(D)
        from a 2-D dpnp.ndarray.

    csr_matrix((data, indices, indptr), shape=(M, N))
        from raw CSR component arrays (1-D dpnp arrays on the same
        SYCL queue). Components are stored as given; indices are sorted
        lazily (see ``sort_indices``) when required by the SpMV path.

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
    has_sorted_indices : bool
        Whether column indices are sorted within each row.
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
        self._has_sorted_indices = None

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

        if not (
            isinstance(data, _dpnp.ndarray)
            and isinstance(indices, _dpnp.ndarray)
            and isinstance(indptr, _dpnp.ndarray)
        ):
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
        if idx_char not in ("i", "l", "q"):
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
        if copy:
            indices = indices.copy()
            indptr = indptr.copy()

        # Store components verbatim (matching scipy): the caller's column
        # order is preserved and copy=False aliasing is honoured. Sorting
        # is deferred to sort_indices(), invoked lazily by the SpMV path.
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self._shape = (nrows, ncols)
        self._has_sorted_indices = None

    @property
    def has_sorted_indices(self):
        """Whether column indices are sorted per row (scipy-compatible).

        The result is cached; an unknown state triggers a one-time check.
        """
        if self._has_sorted_indices is None:
            self._has_sorted_indices = self._check_sorted()
        return self._has_sorted_indices

    def _check_sorted(self):
        idx = self.indices
        if idx.shape[0] == 0:
            return True
        # Sorted iff no adjacent pair within the same row is decreasing.
        q = idx.sycl_queue
        nrows = self._shape[0]
        row_lengths = self.indptr[1:] - self.indptr[:-1]
        row_ids = _dpnp.repeat(
            _dpnp.arange(nrows, dtype=self.indptr.dtype, sycl_queue=q),
            row_lengths,
        )
        same_row = row_ids[1:] == row_ids[:-1]
        decreasing = idx[1:] < idx[:-1]
        return not bool(_dpnp.any(same_row & decreasing))

    def sort_indices(self):
        """Sort column indices within each row, in place (scipy-compatible).

        SpMV backends require sorted CSR; this is a no-op once the
        indices are known sorted.
        """
        if self.has_sorted_indices:
            return
        indices = self.indices
        nnz = indices.shape[0]
        if nnz == 0:
            self._has_sorted_indices = True
            return

        q = indices.sycl_queue
        nrows = self._shape[0]
        row_lengths = self.indptr[1:] - self.indptr[:-1]
        row_ids = _dpnp.repeat(
            _dpnp.arange(nrows, dtype=indices.dtype, sycl_queue=q),
            row_lengths,
        )
        # Lexsort by (row, col) via two stable passes.
        order = _dpnp.argsort(indices, kind="stable")
        order = order[_dpnp.argsort(row_ids[order], kind="stable")]

        self.data = self.data[order]
        self.indices = self.indices[order]
        self._has_sorted_indices = True

    def _init_from_dense(self, dense, dtype=None):
        if dense.ndim != 2:
            raise ValueError(
                f"csr_matrix: dense input must be 2-D, got {dense.ndim}-D"
            )
        if dtype is not None:
            dense = dense.astype(dtype, copy=False)

        nrows, ncols = dense.shape
        q = dense.sycl_queue

        rows, cols = _dpnp.nonzero(dense)
        nnz = int(rows.shape[0])

        if nnz == 0:
            self.data = _dpnp.empty(0, dtype=dense.dtype, sycl_queue=q)
            self.indices = _dpnp.empty(0, dtype=_dpnp.int64, sycl_queue=q)
            self.indptr = _dpnp.zeros(
                nrows + 1, dtype=_dpnp.int64, sycl_queue=q
            )
            self._shape = (nrows, ncols)
            self._has_sorted_indices = True
            return

        values = dense[rows, cols]
        idx_dtype = _dpnp.int64
        row_counts = _dpnp.bincount(rows.astype(idx_dtype), minlength=nrows)
        indptr = _dpnp.empty(nrows + 1, dtype=idx_dtype, sycl_queue=q)
        indptr[0] = 0
        indptr[1:] = _dpnp.cumsum(row_counts)

        self.data = values
        self.indices = cols.astype(idx_dtype)
        self.indptr = indptr
        self._shape = (nrows, ncols)
        # dpnp.nonzero yields row-major order, i.e. columns ascending
        # within each row.
        self._has_sorted_indices = True

    # --- read-only properties ------------------------------------------

    @property
    def shape(self):
        """Tuple of matrix dimensions ``(M, N)``."""
        return self._shape

    @property
    def dtype(self):
        """Data type of stored values."""
        return self.data.dtype

    @property
    def nnz(self):
        """Number of stored nonzero entries."""
        return int(self.data.shape[0])

    @property
    def size(self):
        """Alias for ``nnz`` (number of stored entries)."""
        return self.nnz

    @property
    # pylint: disable-next=invalid-name
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

        self.sort_indices()

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
                y = _dpnp.empty(nrows, dtype=self.data.dtype, sycl_queue=exec_q)
                _manager = _dpu.SequentialOrderManager[exec_q]
                # pylint: disable-next=protected-access
                ht_ev, comp_ev = _si._sparse_gemv_compute(
                    exec_q,
                    handle,
                    val_type_id,
                    0,  # trans=N
                    1.0,  # alpha
                    x,
                    0.0,  # beta
                    y,
                    nrows,
                    ncols,
                    _manager.submitted_events,
                )
                _manager.add_event_pair(ht_ev, comp_ev)
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
            # pylint: disable-next=not-callable
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
        nrows = self._shape[0]
        q = self.data.sycl_queue
        dense = _dpnp.zeros(self._shape, dtype=self.dtype, sycl_queue=q)
        if self.nnz == 0:
            return dense

        row_lengths = self.indptr[1:] - self.indptr[:-1]
        rows = _dpnp.repeat(
            _dpnp.arange(nrows, dtype=self.indices.dtype, sycl_queue=q),
            row_lengths,
        )
        dense[rows, self.indices] = self.data
        return dense

    def copy(self):
        """Return a deep copy of this matrix."""
        return csr_matrix(self)

    def __repr__(self):
        return (
            f"<{self._shape[0]}x{self._shape[1]} csr_matrix "
            f"of dtype {self.dtype} with {self.nnz} stored elements>"
        )
