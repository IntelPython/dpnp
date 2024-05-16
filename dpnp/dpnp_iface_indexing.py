# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2024, Intel Corporation
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

import dpctl.tensor as dpt
import numpy
from numpy.core.numeric import normalize_axis_index

import dpnp

# pylint: disable=no-name-in-module
from .dpnp_algo import (
    dpnp_choose,
    dpnp_diagonal,
    dpnp_putmask,
    dpnp_select,
)
from .dpnp_array import dpnp_array
from .dpnp_utils import (
    call_origin,
    use_origin_backend,
)

__all__ = [
    "choose",
    "diag_indices",
    "diag_indices_from",
    "diagonal",
    "extract",
    "fill_diagonal",
    "indices",
    "mask_indices",
    "nonzero",
    "place",
    "put",
    "put_along_axis",
    "putmask",
    "select",
    "take",
    "take_along_axis",
    "tril_indices",
    "tril_indices_from",
    "triu_indices",
    "triu_indices_from",
]


def _build_along_axis_index(a, ind, axis):
    """
    Build a fancy index used by a family of `_along_axis` functions.

    The fancy index consists of orthogonal arranges, with the
    requested index inserted at the right location.

    The resulting index is going to be used inside `dpnp.put_along_axis`
    and `dpnp.take_along_axis` implementations.

    """

    if not dpnp.issubdtype(ind.dtype, dpnp.integer):
        raise IndexError("`indices` must be an integer array")

    # normalize array shape and input axis
    if axis is None:
        a_shape = (a.size,)
        axis = 0
    else:
        a_shape = a.shape
        axis = normalize_axis_index(axis, a.ndim)

    if len(a_shape) != ind.ndim:
        raise ValueError(
            "`indices` and `a` must have the same number of dimensions"
        )

    # compute dimensions to iterate over
    dest_dims = list(range(axis)) + [None] + list(range(axis + 1, ind.ndim))
    shape_ones = (1,) * ind.ndim

    # build the index
    fancy_index = []
    for dim, n in zip(dest_dims, a_shape):
        if dim is None:
            fancy_index.append(ind)
        else:
            ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim + 1 :]
            fancy_index.append(
                dpnp.arange(
                    n,
                    dtype=ind.dtype,
                    usm_type=ind.usm_type,
                    sycl_queue=ind.sycl_queue,
                ).reshape(ind_shape)
            )

    return tuple(fancy_index)


def choose(x1, choices, out=None, mode="raise"):
    """
    Construct an array from an index array and a set of arrays to choose from.

    For full documentation refer to :obj:`numpy.choose`.

    See also
    --------
    :obj:`dpnp.take_along_axis` : Preferable if choices is an array.

    """
    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)

    choices_list = []
    for choice in choices:
        choices_list.append(
            dpnp.get_dpnp_descriptor(choice, copy_when_nondefault_queue=False)
        )

    if x1_desc:
        if any(not desc for desc in choices_list):
            pass
        elif out is not None:
            pass
        elif mode != "raise":
            pass
        elif any(not choices[0].dtype == choice.dtype for choice in choices):
            pass
        elif not choices_list:
            pass
        else:
            size = x1_desc.size
            choices_size = choices_list[0].size
            if any(
                choice.size != choices_size or choice.size != size
                for choice in choices
            ):
                pass
            elif any(x >= choices_size for x in dpnp.asnumpy(x1)):
                pass
            else:
                return dpnp_choose(x1_desc, choices_list).get_pyobj()

    return call_origin(numpy.choose, x1, choices, out, mode)


def diag_indices(n, ndim=2, device=None, usm_type="device", sycl_queue=None):
    """
    Return the indices to access the main diagonal of an array.

    This returns a tuple of indices that can be used to access the main
    diagonal of an array `a` with ``a.ndim >= 2`` dimensions and shape
    (n, n, ..., n). For ``a.ndim = 2`` this is the usual diagonal, for
    ``a.ndim > 2`` this is the set of indices to access ``a[i, i, ..., i]``
    for ``i = [0..n-1]``.

    For full documentation refer to :obj:`numpy.diag_indices`.

    Parameters
    ----------
    n : int
        The size, along each dimension, of the arrays for which the returned
        indices can be used.
    ndim : int, optional
        The number of dimensions. Default: ``2``.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Return
    ------
    out : tuple of dpnp.ndarray
        The indices to access the main diagonal of an array.

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

    idx = dpnp.arange(
        n,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )
    return (idx,) * ndim


def diag_indices_from(arr, device=None, usm_type=None, sycl_queue=None):
    """
    Return the indices to access the main diagonal of an n-dimensional array.

    For full documentation refer to :obj:`numpy.diag_indices_from`.

    Parameters
    ----------
    arr : dpnp.ndarray
        Array at least 2-D
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Return
    ------
    out : tuple of dpnp.ndarray
        The indices to access the main diagonal of an n-dimensional array.

    See also
    --------
    :obj:`diag_indices` : Return the indices to access the main
                          diagonal of an array.

    Examples
    --------
    Create a 4 by 4 array.

    >>> import dpnp as np
    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])

    Get the indices of the diagonal elements.

    >>> di = np.diag_indices_from(a)
    >>> di
    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))

    >>> a[di]
    array([ 0,  5, 10, 15])

    This is simply syntactic sugar for diag_indices.

    >>> np.diag_indices(a.shape[0])
    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))

    """

    dpnp.check_supported_arrays_type(arr)

    if not arr.ndim >= 2:
        raise ValueError("input array must be at least 2-d")

    sh0 = arr.shape[0]
    for sh in arr.shape[1:]:
        if sh != sh0:
            raise ValueError("All dimensions of input must be of equal length")

    _usm_type = arr.usm_type if usm_type is None else usm_type
    _sycl_queue = dpnp.get_normalized_queue_device(
        arr, sycl_queue=sycl_queue, device=device
    )
    return diag_indices(
        sh0,
        arr.ndim,
        usm_type=_usm_type,
        sycl_queue=_sycl_queue,
    )


def diagonal(x1, offset=0, axis1=0, axis2=1):
    """
    Return specified diagonals.

    For full documentation refer to :obj:`numpy.diagonal`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Parameters `axis1` and `axis2` are supported only with default values.
    Otherwise the function will be executed sequentially on CPU.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc:
        if not isinstance(offset, int):
            pass
        elif offset < 0:
            pass
        elif axis1 != 0:
            pass
        elif axis2 != 1:
            pass
        else:
            return dpnp_diagonal(x1_desc, offset).get_pyobj()

    return call_origin(numpy.diagonal, x1, offset, axis1, axis2)


def extract(condition, x):
    """
    Return the elements of an array that satisfy some condition.

    For full documentation refer to :obj:`numpy.extract`.

    Returns
    -------
    out : dpnp.ndarray
        Rank 1 array of values from `x` where `condition` is True.

    Limitations
    -----------
    Parameters `condition` and `x` are supported either as
    :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameter `x` must be the same shape as `condition`.
    Otherwise the function will be executed sequentially on CPU.
    """

    if dpnp.is_supported_array_type(condition) and dpnp.is_supported_array_type(
        x
    ):
        if condition.shape != x.shape:
            pass
        else:
            dpt_condition = (
                condition.get_array()
                if isinstance(condition, dpnp_array)
                else condition
            )
            dpt_array = x.get_array() if isinstance(x, dpnp_array) else x
            return dpnp_array._create_from_usm_ndarray(
                dpt.extract(dpt_condition, dpt_array)
            )

    return call_origin(numpy.extract, condition, x)


def fill_diagonal(a, val, wrap=False):
    """
    Fill the main diagonal of the given array of any dimensionality.

    For full documentation refer to :obj:`numpy.fill_diagonal`.

    Parameters
    ----------
    a : array, at least 2-D.
        Array whose diagonal is to be filled in-place.
    val : {dpnp.ndarray, usm_ndarray, scalar}
        Value(s) to write on the diagonal. If `val` is scalar, the value is
        written along the diagonal. If array-like, the flattened `val` is
        written along the diagonal, repeating if necessary to fill all
        diagonal entries.
    wrap : bool
        The diagonal "wrapped" after N columns. You can have this behavior
        with this option. This affects only tall matrices. Default: ``False``.

    See Also
    --------
    :obj:`dpnp.diag_indices` : Return the indices to access the main diagonal
                               of an array.
    :obj:`dpnp.diag_indices_from` : Return the indices to access the main
                                    diagonal of an n-dimensional array.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.zeros((3, 3), dtype=int)
    >>> np.fill_diagonal(a, 5)
    >>> a
    array([[5, 0, 0],
           [0, 5, 0],
           [0, 0, 5]])

    The same function can operate on a 4-D array:

    >>> a = np.zeros((3, 3, 3, 3), dtype=int)
    >>> np.fill_diagonal(a, 4)

    We only show a few blocks for clarity:

    >>> a[0, 0]
    array([[4, 0, 0],
           [0, 0, 0],
           [0, 0, 0]])
    >>> a[1, 1]
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 0]])
    >>> a[2, 2]
    array([[0, 0, 0],
           [0, 0, 0],
           [0, 0, 4]])

    The wrap option affects only tall matrices:

    >>> # tall matrices no wrap
    >>> a = np.zeros((5, 3), dtype=int)
    >>> np.fill_diagonal(a, 4)
    >>> a
    array([[4, 0, 0],
           [0, 4, 0],
           [0, 0, 4],
           [0, 0, 0],
           [0, 0, 0]])

    >>> # tall matrices wrap
    >>> a = np.zeros((5, 3), dtype=int)
    >>> np.fill_diagonal(a, 4, wrap=True)
    >>> a
    array([[4, 0, 0],
           [0, 4, 0],
           [0, 0, 4],
           [0, 0, 0],
           [4, 0, 0]])

    >>> # wide matrices
    >>> a = np.zeros((3, 5), dtype=int)
    >>> np.fill_diagonal(a, 4, wrap=True)
    >>> a
    array([[4, 0, 0, 0, 0],
           [0, 4, 0, 0, 0],
           [0, 0, 4, 0, 0]])

    The anti-diagonal can be filled by reversing the order of elements
    using either `dpnp.flipud` or `dpnp.fliplr`.

    >>> a = np.zeros((3, 3), dtype=int)
    >>> val = np.array([1, 2, 3])
    >>> np.fill_diagonal(np.fliplr(a), val)  # Horizontal flip
    >>> a
    array([[0, 0, 1],
           [0, 2, 0],
           [3, 0, 0]])
    >>> np.fill_diagonal(np.flipud(a), val)  # Vertical flip
    >>> a
    array([[0, 0, 3],
           [0, 2, 0],
           [1, 0, 0]])

    """

    dpnp.check_supported_arrays_type(a, scalar_type=True, all_scalars=True)

    if a.ndim < 2:
        raise ValueError("array must be at least 2-d")
    end = a.size
    if a.ndim == 2:
        step = a.shape[1] + 1
        if not wrap and a.shape[0] > a.shape[1]:
            end = a.shape[1] * a.shape[1]
    else:
        sh0 = a.shape[0]
        for sh in a.shape[1:]:
            if sh != sh0:
                raise ValueError(
                    "All dimensions of input must be of equal length"
                )
        step = sum(sh0**x for x in range(a.ndim))

    # TODO: implement flatiter for slice key
    # a.flat[:end:step] = val
    fl = a.flat
    if dpnp.isscalar(val):
        for idx in range(0, end, step):
            fl[idx] = val
    else:
        flat_val = val.flatten()
        for i in range(0, flat_val.size):
            for idx in range((step * i), end, step * (1 + i)):
                fl[idx] = flat_val[i]


def indices(
    dimensions,
    dtype=int,
    sparse=False,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Return an array representing the indices of a grid.

    Compute an array where the subarrays contain index values 0, 1, …
    varying only along the corresponding axis.

    For full documentation refer to :obj:`numpy.indices`.

    Parameters
    ----------
    dimensions : sequence of ints
        The shape of the grid.
    dtype : {None, dtype}, optional
        Data type of the result.
    sparse : {None, boolean}, optional
        Return a sparse representation of the grid instead of a dense
        representation. Default is ``False``.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : one dpnp.ndarray or tuple of dpnp.ndarray
        If sparse is ``False``:
        Returns one array of grid indices,
        ``grid.shape = (len(dimensions),) + tuple(dimensions)``.

        If sparse is ``True``:
        Returns a tuple of arrays,
        with grid[i].shape = (1, ..., 1, dimensions[i], 1, ..., 1)
        with dimensions[i] in the i-th place.

    Examples
    --------
    >>> import dpnp as np
    >>> grid = np.indices((2, 3))
    >>> grid.shape
    (2, 2, 3)
    >>> grid[0]
    array([[0, 0, 0],
           [1, 1, 1]])
    >>> grid[1]
    array([[0, 1, 2],
           [0, 1, 2]])

    The indices can be used as an index into an array.

    >>> x = np.arange(20).reshape(5, 4)
    >>> row, col = np.indices((2, 3))
    >>> x[row, col]
    array([[0, 1, 2],
           [4, 5, 6]])

    Note that it would be more straightforward in the above example to
    extract the required elements directly with ``x[:2, :3]``.
    If sparse is set to ``True``, the grid will be returned in a sparse
    representation.

    >>> i, j = np.indices((2, 3), sparse=True)
    >>> i.shape
    (2, 1)
    >>> j.shape
    (1, 3)
    >>> i
    array([[0],
           [1]])
    >>> j
    array([[0, 1, 2]])

    """

    dimensions = tuple(dimensions)
    n = len(dimensions)
    shape = (1,) * n
    if sparse:
        res = ()
    else:
        res = dpnp.empty(
            (n,) + dimensions,
            dtype=dtype,
            device=device,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
        )
    for i, dim in enumerate(dimensions):
        idx = dpnp.arange(
            dim,
            dtype=dtype,
            device=device,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
        ).reshape(shape[:i] + (dim,) + shape[i + 1 :])
        if sparse:
            res = res + (idx,)
        else:
            res[i] = idx
    return res


def mask_indices(
    n,
    mask_func,
    k=0,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Return the indices to access (n, n) arrays, given a masking function.

    Assume `mask_func` is a function that, for a square array a of size
    ``(n, n)`` with a possible offset argument `k`, when called as
    ``mask_func(a, k=k)`` returns a new array with zeros in certain locations
    (functions like `triu` or `tril` do precisely this). Then this function
    returns the indices where the non-zero values would be located.

    Parameters
    ----------
    n : int
        The returned indices will be valid to access arrays of shape (n, n).
    mask_func : callable
        A function whose call signature is similar to that of `triu`, `tril`.
        That is, ``mask_func(x, k=k)`` returns a boolean array, shaped like `x`.
        `k` is an optional argument to the function.
    k : scalar
        An optional argument which is passed through to `mask_func`. Functions
        like `triu`, `tril` take a second argument that is interpreted as an
        offset.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    indices : tuple of dpnp.ndarray
        The `n` arrays of indices corresponding to the locations where
        ``mask_func(np.ones((n, n)), k)`` is True.

    Examples
    --------
    These are the indices that would allow you to access the upper triangular
    part of any 3x3 array:

    >>> import dpnp as np
    >>> iu = np.mask_indices(3, np.triu)

    For example, if `a` is a 3x3 array:

    >>> a = np.arange(9).reshape(3, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> a[iu]
    array([0, 1, 2, 4, 5, 8])

    An offset can be passed also to the masking function.  This gets us the
    indices starting on the first diagonal right of the main one:

    >>> iu1 = np.mask_indices(3, np.triu, 1)

    with which we now extract only three elements:

    >>> a[iu1]
    array([1, 2, 5])

    """

    m = dpnp.ones(
        (n, n),
        dtype=int,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )
    a = mask_func(m, k=k)
    return nonzero(a != 0)


def nonzero(a):
    """
    Return the indices of the elements that are non-zero.

    Returns a tuple of arrays, one for each dimension of `a`,
    containing the indices of the non-zero elements in that
    dimension. The values in `a` are always tested and returned in
    row-major, C-style order.

    For full documentation refer to :obj:`numpy.nonzero`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.

    Returns
    -------
    out : tuple[dpnp.ndarray]
        Indices of elements that are non-zero.

    See Also
    --------
    :obj:`dpnp.flatnonzero` : Return indices that are non-zero in
                              the flattened version of the input array.
    :obj:`dpnp.ndarray.nonzero` : Equivalent ndarray method.
    :obj:`dpnp.count_nonzero` : Counts the number of non-zero elements
                                in the input array.

    Notes
    -----
    While the nonzero values can be obtained with ``a[nonzero(a)]``, it is
    recommended to use ``a[a.astype(bool)]`` or ``a[a != 0]`` instead, which
    will correctly handle 0-d arrays.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
    >>> x
    array([[3, 0, 0],
           [0, 4, 0],
           [5, 6, 0]])
    >>> np.nonzero(x)
    (array([0, 1, 2, 2]), array([0, 1, 0, 1]))

    >>> x[np.nonzero(x)]
    array([3, 4, 5, 6])
    >>> np.stack(np.nonzero(x)).T
    array([[0, 0],
           [1, 1],
           [2, 0],
           [2, 1]])

    A common use for ``nonzero`` is to find the indices of an array, where
    a condition is ``True.``  Given an array `a`, the condition `a` > 3 is
    a boolean array and since ``False`` is interpreted as ``0``,
    ``np.nonzero(a > 3)`` yields the indices of the `a` where the condition is
    true.

    >>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> a > 3
    array([[False, False, False],
           [ True,  True,  True],
           [ True,  True,  True]])
    >>> np.nonzero(a > 3)
    (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

    Using this result to index `a` is equivalent to using the mask directly:

    >>> a[np.nonzero(a > 3)]
    array([4, 5, 6, 7, 8, 9])
    >>> a[a > 3]  # prefer this spelling
    array([4, 5, 6, 7, 8, 9])

    ``nonzero`` can also be called as a method of the array.

    >>> (a > 3).nonzero()
    (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

    """

    usx_a = dpnp.get_usm_ndarray(a)
    return tuple(
        dpnp_array._create_from_usm_ndarray(y) for y in dpt.nonzero(usx_a)
    )


def place(x, mask, vals, /):
    """
    Change elements of an array based on conditional and input values.

    For full documentation refer to :obj:`numpy.place`.

    Limitations
    -----------
    Parameters `x`, `mask` and `vals` are supported either as
    :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    """

    if (
        dpnp.is_supported_array_type(x)
        and dpnp.is_supported_array_type(mask)
        and dpnp.is_supported_array_type(vals)
    ):
        dpt_array = x.get_array() if isinstance(x, dpnp_array) else x
        dpt_mask = mask.get_array() if isinstance(mask, dpnp_array) else mask
        dpt_vals = vals.get_array() if isinstance(vals, dpnp_array) else vals
        return dpt.place(dpt_array, dpt_mask, dpt_vals)

    return call_origin(numpy.place, x, mask, vals, dpnp_inplace=True)


# pylint: disable=redefined-outer-name
def put(a, indices, vals, /, *, axis=None, mode="wrap"):
    """
    Puts values of an array into another array along a given axis.

    For full documentation refer to :obj:`numpy.put`.

    Limitations
    -----------
    Parameters `a` and `indices` are supported either as :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Parameter `indices` is supported as 1-D array of integer data type.
    Parameter `vals` must be broadcastable to the shape of `indices`
    and has the same data type as `a` if it is as :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Parameter `mode` is supported with ``wrap``, the default, and ``clip``
    values.
    Parameter `axis` is supported as integer only.
    Otherwise the function will be executed sequentially on CPU.

    See Also
    --------
    :obj:`dpnp.putmask` : Changes elements of an array based on conditional
                          and input values.
    :obj:`dpnp.place` : Change elements of an array based on conditional and
                        input values.
    :obj:`dpnp.put_along_axis` : Put values into the destination array
                                 by matching 1d index and data slices.

    Notes
    -----
    In contrast to :obj:`numpy.put` `wrap` mode which wraps indices around
    the array for cyclic operations, :obj:`dpnp.put` `wrap` mode clamps indices
    to a fixed range within the array boundaries (-n <= i < n).

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.arange(5)
    >>> indices = np.array([0, 1])
    >>> np.put(x, indices, [-44, -55])
    >>> x
    array([-44, -55,   2,   3,   4])

    >>> x = np.arange(5)
    >>> indices = np.array([22])
    >>> np.put(x, indices, -5, mode='clip')
    >>> x
    array([ 0,  1,  2,  3, -5])

    """

    if dpnp.is_supported_array_type(a) and dpnp.is_supported_array_type(
        indices
    ):
        if indices.ndim != 1 or not dpnp.issubdtype(
            indices.dtype, dpnp.integer
        ):
            pass
        elif mode not in ("clip", "wrap"):
            pass
        elif axis is not None and not isinstance(axis, int):
            raise TypeError(f"`axis` must be of integer type, got {type(axis)}")
        # TODO: remove when #1382(dpctl) is solved
        elif dpnp.is_supported_array_type(vals) and a.dtype != vals.dtype:
            pass
        else:
            if axis is None and a.ndim > 1:
                a = dpnp.reshape(a, -1)
            dpt_array = dpnp.get_usm_ndarray(a)
            dpt_indices = dpnp.get_usm_ndarray(indices)
            dpt_vals = (
                dpnp.get_usm_ndarray(vals)
                if isinstance(vals, dpnp_array)
                else vals
            )
            return dpt.put(
                dpt_array, dpt_indices, dpt_vals, axis=axis, mode=mode
            )

    return call_origin(numpy.put, a, indices, vals, mode, dpnp_inplace=True)


# pylint: disable=redefined-outer-name
def put_along_axis(a, indices, values, axis):
    """
    Put values into the destination array by matching 1d index and data slices.

    For full documentation refer to :obj:`numpy.put_along_axis`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}, (Ni..., M, Nk...)
        Destination array.
    indices : {dpnp.ndarray, usm_ndarray}, (Ni..., J, Nk...)
        Indices to change along each 1d slice of `a`. This must match the
        dimension of input array, but dimensions in ``Ni`` and ``Nj``
        may be 1 to broadcast against `a`.
    values : {scalar, array_like}, (Ni..., J, Nk...)
        Values to insert at those indices. Its shape and dimension are
        broadcast to match that of `indices`.
    axis : int
        The axis to take 1d slices along. If axis is ``None``, the destination
        array is treated as if a flattened 1d view had been created of it.

    See Also
    --------
    :obj:`dpnp.put` : Put values along an axis, using the same indices
                      for every 1d slice.
    :obj:`dpnp.take_along_axis` : Take values from the input array
                                  by matching 1d index and data slices.

    Examples
    --------
    For this sample array

    >>> import dpnp as np
    >>> a = np.array([[10, 30, 20], [60, 40, 50]])

    We can replace the maximum values with:

    >>> ai = np.argmax(a, axis=1, keepdims=True)
    >>> ai
    array([[1],
           [0]])
    >>> np.put_along_axis(a, ai, 99, axis=1)
    >>> a
    array([[10, 99, 20],
           [99, 40, 50]])

    """

    dpnp.check_supported_arrays_type(a, indices)

    # TODO: remove when #1382(dpctl) is resolved
    if dpnp.is_supported_array_type(values) and a.dtype != values.dtype:
        values = values.astype(a.dtype)

    if axis is None:
        a = a.ravel()

    a[_build_along_axis_index(a, indices, axis)] = values


def putmask(x1, mask, values):
    """
    Changes elements of an array based on conditional and input values.

    For full documentation refer to :obj:`numpy.putmask`.

    Limitations
    -----------
    Input arrays ``arr``, ``mask`` and ``values`` are supported
    as :obj:`dpnp.ndarray`.

    """

    x1_desc = dpnp.get_dpnp_descriptor(
        x1, copy_when_strides=False, copy_when_nondefault_queue=False
    )
    mask_desc = dpnp.get_dpnp_descriptor(mask, copy_when_nondefault_queue=False)
    values_desc = dpnp.get_dpnp_descriptor(
        values, copy_when_nondefault_queue=False
    )
    if x1_desc and mask_desc and values_desc:
        return dpnp_putmask(x1_desc, mask_desc, values_desc)

    return call_origin(numpy.putmask, x1, mask, values, dpnp_inplace=True)


def select(condlist, choicelist, default=0):
    """
    Return an array drawn from elements in `choicelist`, depending on
    conditions.

    For full documentation refer to :obj:`numpy.select`.

    Limitations
    -----------
    Arrays of input lists are supported as :obj:`dpnp.ndarray`.
    Parameter `default` is supported only with default values.
    """

    if not use_origin_backend():
        if not isinstance(condlist, list):
            pass
        elif not isinstance(choicelist, list):
            pass
        elif len(condlist) != len(choicelist):
            pass
        else:
            val = True
            size_ = condlist[0].size
            for cond, choice in zip(condlist, choicelist):
                if cond.size != size_ or choice.size != size_:
                    val = False
            if not val:
                pass
            else:
                return dpnp_select(condlist, choicelist, default).get_pyobj()

    return call_origin(numpy.select, condlist, choicelist, default)


# pylint: disable=redefined-outer-name
def take(x, indices, /, *, axis=None, out=None, mode="wrap"):
    """
    Take elements from an array along an axis.

    For full documentation refer to :obj:`numpy.take`.

    Returns
    -------
    out : dpnp.ndarray
        An array with shape x.shape[:axis] + indices.shape + x.shape[axis + 1:]
        filled with elements from `x`.

    Limitations
    -----------
    Parameters `x` and `indices` are supported either as :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Parameter `indices` is supported as 1-D array of integer data type.
    Parameter `out` is supported only with default value.
    Parameter `mode` is supported with ``wrap``, the default, and ``clip``
    values.
    Providing parameter `axis` is optional when `x` is a 1-D array.
    Otherwise the function will be executed sequentially on CPU.

    See Also
    --------
    :obj:`dpnp.compress` : Take elements using a boolean mask.
    :obj:`dpnp.take_along_axis` : Take elements by matching the array and
                                  the index arrays.

    Notes
    -----
    How out-of-bounds indices will be handled.
    "wrap" - clamps indices to (-n <= i < n), then wraps negative indices.
    "clip" - clips indices to (0 <= i < n)

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([4, 3, 5, 7, 6, 8])
    >>> indices = np.array([0, 1, 4])
    >>> np.take(x, indices)
    array([4, 3, 6])

    In this example "fancy" indexing can be used.

    >>> x[indices]
    array([4, 3, 6])

    >>> indices = dpnp.array([-1, -6, -7, 5, 6])
    >>> np.take(x, indices)
    array([8, 4, 4, 8, 8])

    >>> np.take(x, indices, mode="clip")
    array([4, 4, 4, 8, 8])

    """

    if dpnp.is_supported_array_type(x) and dpnp.is_supported_array_type(
        indices
    ):
        if indices.ndim != 1 or not dpnp.issubdtype(
            indices.dtype, dpnp.integer
        ):
            pass
        elif axis is None and x.ndim > 1:
            pass
        elif out is not None:
            pass
        elif mode not in ("clip", "wrap"):
            pass
        else:
            dpt_array = dpnp.get_usm_ndarray(x)
            dpt_indices = dpnp.get_usm_ndarray(indices)
            return dpnp_array._create_from_usm_ndarray(
                dpt.take(dpt_array, dpt_indices, axis=axis, mode=mode)
            )

    return call_origin(numpy.take, x, indices, axis, out, mode)


def take_along_axis(a, indices, axis):
    """
    Take values from the input array by matching 1d index and data slices.

    For full documentation refer to :obj:`numpy.take_along_axis`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}, (Ni..., M, Nk...)
        Source array
    indices : {dpnp.ndarray, usm_ndarray}, (Ni..., J, Nk...)
        Indices to take along each 1d slice of `a`. This must match the
        dimension of the input array, but dimensions ``Ni`` and ``Nj``
        only need to broadcast against `a`.
    axis : int
        The axis to take 1d slices along. If axis is ``None``, the input
        array is treated as if it had first been flattened to 1d,
        for consistency with `sort` and `argsort`.

    Returns
    -------
    out : dpnp.ndarray
        The indexed result.

    See Also
    --------
    :obj:`dpnp.take` : Take along an axis, using the same indices for
                       every 1d slice.
    :obj:`dpnp.put_along_axis` : Put values into the destination array
                                 by matching 1d index and data slices.
    :obj:`dpnp.argsort` : Return the indices that would sort an array.

    Examples
    --------
    For this sample array

    >>> import dpnp as np
    >>> a = np.array([[10, 30, 20], [60, 40, 50]])

    We can sort either by using :obj:`dpnp.sort` directly, or
    :obj:`dpnp.argsort` and this function:

    >>> np.sort(a, axis=1)
    array([[10, 20, 30],
           [40, 50, 60]])
    >>> ai = np.argsort(a, axis=1)
    >>> ai
    array([[0, 2, 1],
           [1, 2, 0]])
    >>> np.take_along_axis(a, ai, axis=1)
    array([[10, 20, 30],
           [40, 50, 60]])

    The same works for max and min, if you maintain the trivial dimension
    with ``keepdims``:

    >>> np.max(a, axis=1, keepdims=True)
    array([[30],
           [60]])
    >>> ai = np.argmax(a, axis=1, keepdims=True)
    >>> ai
    array([[1],
           [0]])
    >>> np.take_along_axis(a, ai, axis=1)
    array([[30],
           [60]])

    If we want to get the max and min at the same time, we can stack the
    indices first:

    >>> ai_min = np.argmin(a, axis=1, keepdims=True)
    >>> ai_max = np.argmax(a, axis=1, keepdims=True)
    >>> ai = np.concatenate([ai_min, ai_max], axis=1)
    >>> ai
    array([[0, 1],
           [1, 0]])
    >>> np.take_along_axis(a, ai, axis=1)
    array([[10, 30],
           [40, 60]])

    """

    dpnp.check_supported_arrays_type(a, indices)

    if axis is None:
        a = a.ravel()

    return a[_build_along_axis_index(a, indices, axis)]


def tril_indices(
    n,
    k=0,
    m=None,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Return the indices for the lower-triangle of an (n, m) array.

    For full documentation refer to :obj:`numpy.tril_indices`.

    Parameters
    ----------
    n : int
        The row dimension of the arrays for which the returned
        indices will be valid.
    k : int, optional
        Diagonal offset (see `tril` for details). Default: ``0``.
    m : int, optional
        The column dimension of the arrays for which the returned
        arrays will be valid.
        By default `m` is taken equal to `n`.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    inds : tuple of dpnp.ndarray
        The indices for the triangle. The returned tuple contains two arrays,
        each with the indices along one dimension of the array.

    Examples
    --------
    Compute two different sets of indices to access 4x4 arrays, one for the
    lower triangular part starting at the main diagonal, and one starting two
    diagonals further right:

    >>> import dpnp as np
    >>> il1 = np.tril_indices(4)
    >>> il2 = np.tril_indices(4, 2)

    Here is how they can be used with a sample array:

    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])

    Both for indexing:

    >>> a[il1]
    array([ 0,  4,  5, ..., 13, 14, 15])

    And for assigning values:

    >>> a[il1] = -1
    >>> a
    array([[-1,  1,  2,  3],
           [-1, -1,  6,  7],
           [-1, -1, -1, 11],
           [-1, -1, -1, -1]])

    These cover almost the whole array (two diagonals right of the main one):

    >>> a[il2] = -10
    >>> a
    array([[-10, -10, -10,   3],
           [-10, -10, -10, -10],
           [-10, -10, -10, -10],
           [-10, -10, -10, -10]])

    """

    tri_ = dpnp.tri(
        n,
        m,
        k=k,
        dtype=bool,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )

    return tuple(
        dpnp.broadcast_to(inds, tri_.shape)[tri_]
        for inds in indices(
            tri_.shape,
            sparse=True,
            device=device,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
        )
    )


def tril_indices_from(
    arr,
    k=0,
    device=None,
    usm_type=None,
    sycl_queue=None,
):
    """
    Return the indices for the lower-triangle of arr.

    For full documentation refer to :obj:`numpy.tril_indices_from`.

    Parameters
    ----------
    arr : {dpnp.ndarray, usm_ndarray}
        The indices will be valid for square arrays whose dimensions are
        the same as arr.
    k : int, optional
        Diagonal offset (see `tril` for details). Default: ``0``.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    inds : tuple of dpnp.ndarray
        The indices for the triangle. The returned tuple contains two arrays,
        each with the indices along one dimension of the array.

    Examples
    --------
    Create a 4 by 4 array.

    >>> import dpnp as np
    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])

    Pass the array to get the indices of the lower triangular elements.

    >>> trili = np.tril_indices_from(a)
    >>> trili
    (array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3]),
     array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3]))

    >>> a[trili]
    array([ 0,  4,  5,  8,  9, 10, 12, 13, 14, 15])

    This is syntactic sugar for tril_indices().

    >>> np.tril_indices(a.shape[0])
    (array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3]),
     array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3]))

    Use the `k` parameter to return the indices for the lower triangular array
    up to the k-th diagonal.

    >>> trili1 = np.tril_indices_from(a, k=1)
    >>> a[trili1]
    array([ 0,  1,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14, 15])

    """

    dpnp.check_supported_arrays_type(arr)

    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")

    _usm_type = arr.usm_type if usm_type is None else usm_type
    _sycl_queue = dpnp.get_normalized_queue_device(
        arr, sycl_queue=sycl_queue, device=device
    )

    return tril_indices(
        arr.shape[-2],
        k=k,
        m=arr.shape[-1],
        usm_type=_usm_type,
        sycl_queue=_sycl_queue,
    )


def triu_indices(
    n,
    k=0,
    m=None,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Return the indices for the upper-triangle of an (n, m) array.

    For full documentation refer to :obj:`numpy.triu_indices`.

    Parameters
    ----------
    n : int
        The size of the arrays for which the returned indices will
        be valid.
    k : int, optional
        Diagonal offset (see `triu` for details). Default: ``0``.
    m : int, optional
        The column dimension of the arrays for which the returned
        arrays will be valid.
        By default `m` is taken equal to `n`.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    inds : tuple of dpnp.ndarray
        The indices for the triangle. The returned tuple contains two arrays,
        each with the indices along one dimension of the array.  Can be used
        to slice a ndarray of shape(`n`, `n`).

    Examples
    --------
    Compute two different sets of indices to access 4x4 arrays, one for the
    upper triangular part starting at the main diagonal, and one starting two
    diagonals further right:

    >>> import dpnp as np
    >>> iu1 = np.triu_indices(4)
    >>> iu2 = np.triu_indices(4, 2)

    Here is how they can be used with a sample array:

    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])

    Both for indexing:

    >>> a[iu1]
    array([ 0,  1,  2, ..., 10, 11, 15])

    And for assigning values:

    >>> a[iu1] = -1
    >>> a
    array([[-1, -1, -1, -1],
           [ 4, -1, -1, -1],
           [ 8,  9, -1, -1],
           [12, 13, 14, -1]])

    These cover only a small part of the whole array (two diagonals right
    of the main one):

    >>> a[iu2] = -10
    >>> a
    array([[ -1,  -1, -10, -10],
           [  4,  -1,  -1, -10],
           [  8,   9,  -1,  -1],
           [ 12,  13,  14,  -1]])

    """

    tri_ = ~dpnp.tri(
        n,
        m,
        k=k - 1,
        dtype=bool,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )

    return tuple(
        dpnp.broadcast_to(inds, tri_.shape)[tri_]
        for inds in indices(
            tri_.shape,
            sparse=True,
            device=device,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
        )
    )


def triu_indices_from(
    arr,
    k=0,
    device=None,
    usm_type=None,
    sycl_queue=None,
):
    """
    Return the indices for the lower-triangle of arr.

    For full documentation refer to :obj:`numpy.triu_indices_from`.

    Parameters
    ----------
    arr : {dpnp.ndarray, usm_ndarray}
        The indices will be valid for square arrays whose dimensions are
        the same as arr.
    k : int, optional
        Diagonal offset (see `triu` for details). Default: ``0``.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    inds : tuple of dpnp.ndarray
        The indices for the triangle. The returned tuple contains two arrays,
        each with the indices along one dimension of the array.  Can be used
        to slice a ndarray of shape(`n`, `n`).

    Examples
    --------
    Create a 4 by 4 array.

    >>> import dpnp as np
    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])

    Pass the array to get the indices of the upper triangular elements.

    >>> triui = np.triu_indices_from(a)
    >>> triui
    (array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3]),
     array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3]))

    >>> a[triui]
    array([ 0,  1,  2,  3,  5,  6,  7, 10, 11, 15])

    This is syntactic sugar for triu_indices().

    >>> np.triu_indices(a.shape[0])
    (array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3]),
     array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3]))

    Use the `k` parameter to return the indices for the upper triangular array
    from the k-th diagonal.

    >>> triuim1 = np.triu_indices_from(a, k=1)
    >>> a[triuim1]
    array([ 1,  2,  3,  6,  7, 11])

    """

    dpnp.check_supported_arrays_type(arr)

    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")

    _usm_type = arr.usm_type if usm_type is None else usm_type
    _sycl_queue = dpnp.get_normalized_queue_device(
        arr, sycl_queue=sycl_queue, device=device
    )

    return triu_indices(
        arr.shape[-2],
        k=k,
        m=arr.shape[-1],
        usm_type=_usm_type,
        sycl_queue=_sycl_queue,
    )
