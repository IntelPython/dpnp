"""Sparse base class and predicate, mirroring scipy/_lib/_sparse.py."""

from abc import ABC


class SparseABC(ABC):
    """Abstract base for all dpnp.scipy.sparse format classes."""
    pass


def issparse(x):
    """Return True if x is a dpnp sparse matrix.

    Mirrors scipy.sparse.issparse semantics: returns True for any
    instance of SparseABC, False otherwise.
    """
    return isinstance(x, SparseABC)
