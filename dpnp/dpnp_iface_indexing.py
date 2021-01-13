# cython: language_level=3
# distutils: language = c++
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2020, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
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

"""
Interface of the Indexing part of the DPNP

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""


import numpy

from dpnp.dpnp_algo import *
from dpnp.dparray import dparray
from dpnp.dpnp_utils import *
import dpnp


__all__ = [
    "diag_indices",
    "diag_indices_from",
    "diagonal",
    "fill_diagonal",
    "nonzero",
    "put",
    "take",
    "tril_indices",
    "tril_indices_from",
    "triu_indices",
    "triu_indices_from"
]


def diag_indices(n, ndim=2):
    """
    Return the indices to access the main diagonal of an array.

    This returns a tuple of indices that can be used to access the main
    diagonal of an array `a` with ``a.ndim >= 2`` dimensions and shape
    (n, n, ..., n). For ``a.ndim = 2`` this is the usual diagonal, for
    ``a.ndim > 2`` this is the set of indices to access ``a[i, i, ..., i]``
    for ``i = [0..n-1]``.

    For full documentation refer to :obj:`numpy.diag_indices`.

    See also
    --------
    :obj:`diag_indices_from` : Return the indices to access the main
                               diagonal of an n-dimensional array.

    Examples
    --------
    Create a set of indices to access the diagonal of a (4, 4) array:

    >>> import dpnp as np
    >>> di = np.diag_indices(4)
    >>> di
    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))
    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> a[di] = 100
    >>> a
    array([[100,   1,   2,   3],
           [  4, 100,   6,   7],
           [  8,   9, 100,  11],
           [ 12,  13,  14, 100]])

    Now, we create indices to manipulate a 3-D array:

    >>> d3 = np.diag_indices(2, 3)
    >>> d3
    (array([0, 1]), array([0, 1]), array([0, 1]))

    And use it to set the diagonal of an array of zeros to 1:

    >>> a = np.zeros((2, 2, 2), dtype=int)
    >>> a[d3] = 1
    >>> a
    array([[[1, 0],
            [0, 0]],
           [[0, 0],
            [0, 1]]])

    """

    if not use_origin_backend():
        return dpnp_diag_indices(n, ndim)

    return call_origin(numpy.diag_indices, n, ndim)


def diag_indices_from(arr):
    """
    Return the indices to access the main diagonal of an n-dimensional array.

    For full documentation refer to :obj:`numpy.diag_indices_from`.

    See also
    --------
    :obj:`diag_indices` : Return the indices to access the main
                          diagonal of an array.

    """

    is_a_dparray = isinstance(arr, dparray)

    if (not use_origin_backend(arr) and is_a_dparray):
        # original limitation
        if not arr.ndim >= 2:
            checker_throw_value_error("diag_indices_from", "arr.ndim", arr.ndim, "at least 2-d")

        # original limitation
        # For more than d=2, the strided formula is only valid for arrays with
        # all dimensions equal, so we check first.
        if not numpy.alltrue(numpy.diff(arr.shape) == 0):  # TODO: replace alltrue and diff funcs with dpnp own ones
            checker_throw_value_error("diag_indices_from", "arr.shape", arr.shape,
                                      "All dimensions of input must be of equal length")

        return dpnp_diag_indices(arr.shape[0], arr.ndim)

    return call_origin(numpy.diag_indices_from, arr)


def diagonal(input, offset=0, axis1=0, axis2=1):
    """
    Return specified diagonals.

    For full documentation refer to :obj:`numpy.diagonal`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Parameters ``axis1`` and ``axis2`` are supported only with default values.
    Otherwise the function will be executed sequentially on CPU.
    """

    if not use_origin_backend(input):
        if not isinstance(input, dparray):
            pass
        elif not isinstance(offset, int):
            pass
        elif offset < 0:
            pass
        elif axis1 != 0:
            pass
        elif axis2 != 1:
            pass
        else:
            return dpnp_diagonal(input, offset)

    return call_origin(numpy.diagonal, input, offset, axis1, axis2)


def fill_diagonal(input, val, wrap=False):
    """
    Fill the main diagonal of the given array of any dimensionality.

    For full documentation refer to :obj:`numpy.fill_diagonal`.

    Limitations
    -----------
    Parameter ``wrap`` is supported only with default values.

    See Also
    --------
    :obj:`dpnp.diag_indices` : Return the indices to access the main diagonal of an array.
    :obj:`dpnp.diag_indices_from` : Return the indices to access the main diagonal of an n-dimensional array.
    """

    if not use_origin_backend(input):
        if not isinstance(input, dparray):
            pass
        elif not dpnp.isscalar(val):
            pass
        elif wrap:
            pass
        else:
            return dpnp_fill_diagonal(input, val)

    return call_origin(numpy.fill_diagonal, input, val, wrap)


def nonzero(a):
    """
    Return the indices of the elements that are non-zero.

    For full documentation refer to :obj:`numpy.nonzero`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.flatnonzero` : Return indices that are non-zero in
                              the flattened version of the input array.
    :obj:`dpnp.count_nonzero` : Counts the number of non-zero elements
                                in the input array.

    Notes
    -----
    While the nonzero values can be obtained with ``a[nonzero(a)]``, it is
    recommended to use ``x[x.astype(bool)]`` or ``x[x != 0]`` instead, which
    will correctly handle 0-d arrays.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
    >>> out = np.nonzero(x)
    >>> for arr in out:
    >>>     [i for i in arr]
    [0, 1, 2, 2]
    [0, 1, 0, 1]

    >>> x2 = np.array([3, 0, 0, 0, 4, 0, 5, 6, 0])
    >>> out2 = np.nonzero(x2)
    >>> for arr in out2:
    >>>     [i for i in arr]
    [0, 4, 6, 7]

    """

    is_a_dparray = isinstance(a, dparray)

    if (not use_origin_backend(a) and is_a_dparray):
        return dpnp_nonzero(a)

    return call_origin(numpy.nonzero, a)


def put(input, ind, v, mode='raise'):
    """
    Replaces specified elements of an array with given values.
    For full documentation refer to :obj:`numpy.put`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Not supported parameter mode.
    """

    if not use_origin_backend(input):
        if not isinstance(input, dparray):
            pass
        elif mode != 'raise':
            pass
        elif type(ind) != type(v):
            pass
        elif numpy.max(ind) > input.size:
            pass
        else:
            return dpnp_put(input, ind, v)

    return call_origin(numpy.put, input, ind, v, mode)


def take(input, indices, axis=None, out=None, mode='raise'):
    """
    Take elements from an array.
    For full documentation refer to :obj:`numpy.take`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Parameters ``axis``, ``out`` and ``mode`` are supported only with default values.
    Parameter ``indices`` is supported as :obj:`dpnp.ndarray`.

    See Also
    --------
    :obj:`dpnp.compress` : Take elements using a boolean mask.
    :obj:`take_along_axis` : Take elements by matching the array and the index arrays.
    """

    if not use_origin_backend(input):
        if not isinstance(input, dparray):
            pass
        elif not isinstance(indices, dparray):
            pass
        elif axis is not None:
            pass
        elif out is not None:
            pass
        elif mode != 'raise':
            pass
        else:
            return dpnp_take(input, indices)

    return call_origin(numpy.take, input, indices, axis, out, mode)


def tril_indices(n, k=0, m=None):
    """
    Return the indices for the lower-triangle of an (n, m) array.

    Parameters
    ----------
    n : int
        The row dimension of the arrays for which the returned
        indices will be valid.

    k : int, optional
        Diagonal offset (see `tril` for details).

    m : int, optional
        The column dimension of the arrays for which the returned
        arrays will be valid.
        By default `m` is taken equal to `n`.

    Returns
    -------
    inds : tuple of arrays
        The indices for the triangle. The returned tuple contains two arrays,
        each with the indices along one dimension of the array.

    """

    if not use_origin_backend():
        if isinstance(n, int) and isinstance(k, int) \
                and (isinstance(m, int) or m is None):
            return dpnp_tril_indices(n, k, m)

    return call_origin(numpy.tril_indices, n, k, m)


def tril_indices_from(arr, k=0):
    """
    Return the indices for the lower-triangle of arr.
    See `tril_indices` for full details.

    Parameters
    ----------
    arr : array_like
        The indices will be valid for square arrays whose dimensions are
        the same as arr.

    k : int, optional
        Diagonal offset (see `tril` for details).
    """

    is_arr_dparray = isinstance(arr, dparray)

    if (not use_origin_backend(arr) and is_arr_dparray):
        if isinstance(k, int):
            return dpnp_tril_indices_from(arr, k)

    return call_origin(numpy.tril_indices_from, arr, k)


def triu_indices(n, k=0, m=None):
    """
    Return the indices for the upper-triangle of an (n, m) array.

    Parameters
    ----------
    n : int
        The size of the arrays for which the returned indices will
        be valid.

    k : int, optional
        Diagonal offset (see `triu` for details).

    m : int, optional
        The column dimension of the arrays for which the returned
        arrays will be valid.
        By default `m` is taken equal to `n`.

    Returns
    -------
    inds : tuple, shape(2) of ndarrays, shape(`n`)
        The indices for the triangle. The returned tuple contains two arrays,
        each with the indices along one dimension of the array.  Can be used
        to slice a ndarray of shape(`n`, `n`).
    """

    if not use_origin_backend():
        if isinstance(n, int) and isinstance(k, int) \
                and (isinstance(m, int) or m is None):
            return dpnp_triu_indices(n, k, m)

    return call_origin(numpy.triu_indices, n, k, m)


def triu_indices_from(arr, k=0):
    """
    Return the indices for the lower-triangle of arr.
    See `tril_indices` for full details.

    Parameters
    ----------
    arr : array_like
        The indices will be valid for square arrays whose dimensions are
        the same as arr.

    k : int, optional
        Diagonal offset (see `tril` for details).
    """

    is_arr_dparray = isinstance(arr, dparray)

    if (not use_origin_backend(arr) and is_arr_dparray):
        if isinstance(k, int):
            return dpnp_triu_indices_from(arr, k)

    return call_origin(numpy.triu_indices_from, arr, k)
