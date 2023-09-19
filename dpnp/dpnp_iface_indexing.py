# cython: language_level=3
# distutils: language = c++
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2023, Intel Corporation
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

import dpnp
from dpnp.dpnp_algo import *
from dpnp.dpnp_array import dpnp_array
from dpnp.dpnp_utils import *

__all__ = [
    "choose",
    "diag_indices",
    "diag_indices_from",
    "diagonal",
    "extract",
    "fill_diagonal",
    "indices",
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


def choose(x1, choices, out=None, mode="raise"):
    """
    Construct an array from an index array and a set of arrays to choose from.

    For full documentation refer to :obj:`numpy.choose`.

    See also
    --------
    :obj:`take_along_axis` : Preferable if choices is an array.
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
        elif not len(choices_list):
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


def diag_indices_from(x1):
    """
    Return the indices to access the main diagonal of an n-dimensional array.

    For full documentation refer to :obj:`numpy.diag_indices_from`.

    See also
    --------
    :obj:`diag_indices` : Return the indices to access the main
                          diagonal of an array.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc:
        # original limitation
        if not x1_desc.ndim >= 2:
            pass

        # original limitation
        # For more than d=2, the strided formula is only valid for arrays with
        # all dimensions equal, so we check first.
        elif not numpy.alltrue(
            numpy.diff(x1_desc.shape) == 0
        ):  # TODO: replace alltrue and diff funcs with dpnp own ones
            pass
        else:
            return dpnp_diag_indices(x1_desc.shape[0], x1_desc.ndim)

    return call_origin(numpy.diag_indices_from, x1)


def diagonal(x1, offset=0, axis1=0, axis2=1):
    """
    Return specified diagonals.

    For full documentation refer to :obj:`numpy.diagonal`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Parameters ``axis1`` and ``axis2`` are supported only with default values.
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
    y : dpnp.ndarray
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


def fill_diagonal(x1, val, wrap=False):
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

    x1_desc = dpnp.get_dpnp_descriptor(
        x1, copy_when_strides=False, copy_when_nondefault_queue=False
    )
    if x1_desc:
        if not dpnp.isscalar(val):
            pass
        elif wrap:
            pass
        else:
            return dpnp_fill_diagonal(x1_desc, val)

    return call_origin(numpy.fill_diagonal, x1, val, wrap, dpnp_inplace=True)


def indices(dimensions, dtype=int, sparse=False):
    """
    Return an array representing the indices of a grid.

    For full documentation refer to :obj:`numpy.indices`.

    Limitations
    -----------
    Parameters ``dtype`` and ``sparse`` are supported only with default values.
    Parameter ``dimensions`` is supported with len <=2.
    """

    if not isinstance(dimensions, (tuple, list)):
        pass
    elif len(dimensions) > 2 or len(dimensions) == 0:
        pass
    elif dtype != int:
        pass
    elif sparse:
        pass
    else:
        return dpnp_indices(dimensions)

    return call_origin(numpy.indices, dimensions, dtype, sparse)


def nonzero(x, /):
    """
    Return the indices of the elements that are non-zero.

    For full documentation refer to :obj:`numpy.nonzero`.

    Returns
    -------
    y : tuple[dpnp.ndarray]
        Indices of elements that are non-zero.

    Limitations
    -----------
    Parameters `x` is supported as either :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
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

    if isinstance(x, dpnp_array) or isinstance(x, dpt.usm_ndarray):
        dpt_array = x.get_array() if isinstance(x, dpnp_array) else x
        return tuple(
            dpnp_array._create_from_usm_ndarray(y)
            for y in dpt.nonzero(dpt_array)
        )

    return call_origin(numpy.nonzero, x)


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
    Parameter `mode` is supported with ``wrap``, the default, and ``clip`` values.
    Parameter `axis` is supported as integer only.
    Otherwise the function will be executed sequentially on CPU.

    See Also
    --------
    :obj:`dpnp.putmask` : Changes elements of an array based on conditional and input values.
    :obj:`dpnp.place` : Change elements of an array based on conditional and input values.
    :obj:`dpnp.put_along_axis` : Put values into the destination array by matching 1d index and data slices.

    Notes
    -----
    In contrast to :obj:`numpy.put` `wrap` mode which wraps indices around the array for cyclic operations,
    :obj:`dpnp.put` `wrap` mode clamps indices to a fixed range within the array boundaries (-n <= i < n).

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


def put_along_axis(x1, indices, values, axis):
    """
    Put values into the destination array by matching 1d index and data slices.

    For full documentation refer to :obj:`numpy.put_along_axis`.

    See Also
    --------
    :obj:`take_along_axis` : Take values from the input array by matching 1d index and data slices.
    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    indices_desc = dpnp.get_dpnp_descriptor(
        indices, copy_when_nondefault_queue=False
    )
    values_desc = dpnp.get_dpnp_descriptor(
        values, copy_when_nondefault_queue=False
    )
    if x1_desc and indices_desc and values_desc:
        if x1_desc.ndim != indices_desc.ndim:
            pass
        elif not isinstance(axis, int):
            pass
        elif axis >= x1_desc.ndim:
            pass
        elif indices_desc.size != values_desc.size:
            pass
        else:
            return dpnp_put_along_axis(x1_desc, indices_desc, values_desc, axis)

    return call_origin(
        numpy.put_along_axis, x1, indices, values, axis, dpnp_inplace=True
    )


def putmask(x1, mask, values):
    """
    Changes elements of an array based on conditional and input values.

    For full documentation refer to :obj:`numpy.putmask`.

    Limitations
    -----------
    Input arrays ``arr``, ``mask`` and ``values``  are supported as :obj:`dpnp.ndarray`.
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
    Return an array drawn from elements in choicelist, depending on conditions.

    For full documentation refer to :obj:`numpy.select`.

    Limitations
    -----------
    Arrays of input lists are supported as :obj:`dpnp.ndarray`.
    Parameter ``default`` are supported only with default values.
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
            for i in range(len(condlist)):
                if condlist[i].size != size_ or choicelist[i].size != size_:
                    val = False
            if not val:
                pass
            else:
                return dpnp_select(condlist, choicelist, default).get_pyobj()

    return call_origin(numpy.select, condlist, choicelist, default)


def take(x, indices, /, *, axis=None, out=None, mode="wrap"):
    """
    Take elements from an array along an axis.

    For full documentation refer to :obj:`numpy.take`.

    Returns
    -------
    dpnp.ndarray
        An array with shape x.shape[:axis] + indices.shape + x.shape[axis + 1:]
        filled with elements from `x`.

    Limitations
    -----------
    Parameters `x` and `indices` are supported either as :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Parameter `indices` is supported as 1-D array of integer data type.
    Parameter `out` is supported only with default value.
    Parameter `mode` is supported with ``wrap``, the default, and ``clip`` values.
    Providing parameter `axis` is optional when `x` is a 1-D array.
    Otherwise the function will be executed sequentially on CPU.

    See Also
    --------
    :obj:`dpnp.compress` : Take elements using a boolean mask.
    :obj:`take_along_axis` : Take elements by matching the array and the index arrays.

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


def take_along_axis(x1, indices, axis):
    """
    Take values from the input array by matching 1d index and data slices.

    For full documentation refer to :obj:`numpy.take_along_axis`.

    See Also
    --------
    :obj:`dpnp.take` : Take along an axis, using the same indices for every 1d slice.
    :obj:`put_along_axis` : Put values into the destination array by matching 1d index and data slices.
    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    indices_desc = dpnp.get_dpnp_descriptor(
        indices, copy_when_nondefault_queue=False
    )
    if x1_desc and indices_desc:
        if x1_desc.ndim != indices_desc.ndim:
            pass
        elif not isinstance(axis, int):
            pass
        elif axis >= x1_desc.ndim:
            pass
        elif x1_desc.ndim == indices_desc.ndim:
            val_list = []
            for i in list(indices_desc.shape)[:-1]:
                if i == 1:
                    val_list.append(True)
                else:
                    val_list.append(False)
            if not all(val_list):
                pass
            else:
                return dpnp_take_along_axis(x1, indices, axis)
        else:
            return dpnp_take_along_axis(x1, indices, axis)

    return call_origin(numpy.take_along_axis, x1, indices, axis)


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
        if (
            isinstance(n, int)
            and isinstance(k, int)
            and (isinstance(m, int) or m is None)
        ):
            return dpnp_tril_indices(n, k, m)

    return call_origin(numpy.tril_indices, n, k, m)


def tril_indices_from(x1, k=0):
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

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc:
        if isinstance(k, int):
            return dpnp_tril_indices_from(x1_desc, k)

    return call_origin(numpy.tril_indices_from, x1, k)


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
        if (
            isinstance(n, int)
            and isinstance(k, int)
            and (isinstance(m, int) or m is None)
        ):
            return dpnp_triu_indices(n, k, m)

    return call_origin(numpy.triu_indices, n, k, m)


def triu_indices_from(x1, k=0):
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

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc:
        if isinstance(k, int):
            return dpnp_triu_indices_from(x1_desc, k)

    return call_origin(numpy.triu_indices_from, x1, k)
