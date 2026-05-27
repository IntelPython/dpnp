"""Sparse base class and predicate, mirroring scipy/_lib/_sparse.py.

Only the modern ``issparse`` predicate is exposed. The legacy
``isspmatrix`` / ``isspmatrix_csr`` family (kept by SciPy for the
``spmatrix`` vs ``sparray`` discriminator and slated for deprecation,
see :mod:`scipy.sparse`) is intentionally omitted -- dpnp has no
``spmatrix`` / ``sparray`` split, so the legacy names would have no
useful semantics. Format-specific checks should use
``issparse(A) and A.format == "csr"`` directly, which is what the
solver fast-path already does.
"""

from abc import ABC


class SparseABC(ABC):
    """Abstract base for all dpnp.scipy.sparse format classes."""

    pass


def issparse(x):
    """Return True if x is a dpnp sparse matrix.

    Mirrors :func:`scipy.sparse.issparse` semantics: returns True for any
    instance of :class:`SparseABC`, False otherwise.
    """
    return isinstance(x, SparseABC)
