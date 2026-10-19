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

"""Sparse base class and predicate, mirroring scipy/_lib/_sparse.py."""

from abc import ABC

__all__ = ["SparseABC", "issparse"]


# pylint: disable-next=too-few-public-methods
class SparseABC(ABC):
    """Abstract base for all dpnp.scipy.sparse format classes."""


def issparse(x):
    """
    Determine whether `x` is a dpnp sparse matrix type.

    For full documentation refer to :obj:`scipy.sparse.issparse`.

    Parameters
    ----------
    x : object
        Object to check for being a dpnp sparse matrix.

    Returns
    -------
    out : bool
        ``True`` if `x` is a dpnp sparse matrix, ``False`` otherwise.

    Examples
    --------
    >>> import dpnp
    >>> from dpnp.scipy.sparse import csr_matrix, issparse
    >>> issparse(csr_matrix(dpnp.eye(3)))
    True
    >>> issparse(dpnp.eye(3))
    False
    >>> issparse(5)
    False

    """
    return isinstance(x, SparseABC)
