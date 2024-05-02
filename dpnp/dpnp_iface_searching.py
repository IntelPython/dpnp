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
Interface of the searching function of the dpnp

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""

# pylint: disable=no-name-in-module

import dpctl.tensor as dpt
import dpctl.tensor._tensor_impl as dti

import dpnp

from .dpnp_array import dpnp_array
from .dpnp_utils import (
    get_usm_allocations,
)
from .dpnp_utils.dpnp_utils_reduction import dpnp_wrap_reduction_call

__all__ = ["argmax", "argmin", "searchsorted", "where"]


def _get_search_res_dt(a, _dtype, out):
    """Get a data type used by dpctl for result array in search function."""

    # get a data type used by dpctl for result array in search function
    res_dt = dti.default_device_index_type(a.sycl_device)

    # numpy raises TypeError if "out" data type mismatch default index type
    if not dpnp.can_cast(out.dtype, res_dt, casting="safe"):
        raise TypeError(
            f"Cannot cast from {out.dtype} to {res_dt} "
            "according to the rule safe."
        )
    return res_dt


def argmax(a, axis=None, out=None, *, keepdims=False):
    """
    Returns the indices of the maximum values along an axis.

    For full documentation refer to :obj:`numpy.argmax`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int}, optional
        By default, the index is into the flattened array, otherwise along
        the specified axis.
        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        If provided, the result will be inserted into this array. It should be
        of the appropriate shape and dtype.
        Default: ``None``.
    keepdims : {None, bool}, optional
        If this is set to ``True``, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result will
        broadcast correctly against the array.
        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        Array of indices into the array. It has the same shape as ``a.shape``
        with the dimension along `axis` removed. If `keepdims` is set to
        ``True``, then the size of `axis` will be ``1`` with the resulting
        array having same shape as ``a.shape``.

    See Also
    --------
    :obj:`dpnp.ndarray.argmax` : Equivalent function.
    :obj:`dpnp.nanargmax` : Returns the indices of the maximum values along
                            an axis, ignoring NaNs.
    :obj:`dpnp.argmin` : Returns the indices of the minimum values
                         along an axis.
    :obj:`dpnp.max` : The maximum value along a given axis.
    :obj:`dpnp.unravel_index` : Convert a flat index into an index tuple.
    :obj:`dpnp.take_along_axis` : Apply ``np.expand_dims(index_array, axis)``
                    from :obj:`dpnp.argmax` to an array as if by calling max.

    Notes
    -----
    In case of multiple occurrences of the maximum values, the indices
    corresponding to the first occurrence are returned.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(6).reshape((2, 3)) + 10
    >>> a
    array([[10, 11, 12],
           [13, 14, 15]])
    >>> np.argmax(a)
    array(5)

    >>> np.argmax(a, axis=0)
    array([1, 1, 1])
    >>> np.argmax(a, axis=1)
    array([2, 2])

    >>> b = np.arange(6)
    >>> b[1] = 5
    >>> b
    array([0, 5, 2, 3, 4, 5])
    >>> np.argmax(b)  # Only the first occurrence is returned.
    array(1)

    >>> x = np.arange(24).reshape((2, 3, 4))
    >>> res = np.argmax(x, axis=1, keepdims=True) # Setting keepdims to True
    >>> res.shape
    (2, 1, 4)

    """

    usm_a = dpnp.get_usm_ndarray(a)
    return dpnp_wrap_reduction_call(
        a,
        out,
        dpt.argmax,
        _get_search_res_dt,
        usm_a,
        axis=axis,
        keepdims=keepdims,
    )


def argmin(a, axis=None, out=None, *, keepdims=False):
    """
    Returns the indices of the minimum values along an axis.

    For full documentation refer to :obj:`numpy.argmin`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int}, optional
        By default, the index is into the flattened array, otherwise along
        the specified axis.
        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        If provided, the result will be inserted into this array. It should be
        of the appropriate shape and dtype.
        Default: ``None``.
    keepdims : {None, bool}, optional
        If this is set to ``True``, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result will
        broadcast correctly against the array.
        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed. If `keepdims` is set to
        ``True``, then the size of `axis` will be ``1`` with the resulting
        array having same shape as `a.shape`.


    See Also
    --------
    :obj:`dpnp.ndarray.argmin` : Equivalent function.
    :obj:`dpnp.nanargmin` : Returns the indices of the minimum values
                            along an axis, ignoring NaNs.
    :obj:`dpnp.argmax` : Returns the indices of the maximum values
                         along an axis.
    :obj:`dpnp.min` : The minimum value along a given axis.
    :obj:`dpnp.unravel_index` : Convert a flat index into an index tuple.
    :obj:`dpnp.take_along_axis` : Apply ``np.expand_dims(index_array, axis)``
                    from :obj:`dpnp.argmin` to an array as if by calling min.

    Notes
    -----
    In case of multiple occurrences of the minimum values, the indices
    corresponding to the first occurrence are returned.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(6).reshape((2, 3)) + 10
    >>> a
    array([[10, 11, 12],
           [13, 14, 15]])
    >>> np.argmin(a)
    array(0)

    >>> np.argmin(a, axis=0)
    array([0, 0, 0])
    >>> np.argmin(a, axis=1)
    array([0, 0])

    >>> b = np.arange(6) + 10
    >>> b[4] = 10
    >>> b
    array([10, 11, 12, 13, 10, 15])
    >>> np.argmin(b)  # Only the first occurrence is returned.
    array(0)

    >>> x = np.arange(24).reshape((2, 3, 4))
    >>> res = np.argmin(x, axis=1, keepdims=True) # Setting keepdims to True
    >>> res.shape
    (2, 1, 4)

    """

    usm_a = dpnp.get_usm_ndarray(a)
    return dpnp_wrap_reduction_call(
        a,
        out,
        dpt.argmin,
        _get_search_res_dt,
        usm_a,
        axis=axis,
        keepdims=keepdims,
    )


def searchsorted(a, v, side="left", sorter=None):
    """
    Find indices where elements should be inserted to maintain order.

    For full documentation refer to :obj:`numpy.searchsorted`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input 1-D array. If `sorter` is ``None``, then it must be sorted in
        ascending order, otherwise `sorter` must be an array of indices that
        sort it.
    v : {dpnp.ndarray, usm_ndarray, scalar}
        Values to insert into `a`.
    side : {'left', 'right'}, optional
        If ``'left'``, the index of the first suitable location found is given.
        If ``'right'``, return the last such index. If there is no suitable
        index, return either 0 or N (where N is the length of `a`).
        Default is ``'left'``.
    sorter : {dpnp.ndarray, usm_ndarray}, optional
        Optional 1-D array of integer indices that sort array a into ascending
        order. They are typically the result of :obj:`dpnp.argsort`.
        Out of bound index values of `sorter` array are treated using `"wrap"`
        mode documented in :py:func:`dpnp.take`.
        Default is ``None``.

    Returns
    -------
    indices : dpnp.ndarray
        Array of insertion points with the same shape as `v`,
        or 0-D array if `v` is a scalar.

    See Also
    --------
    :obj:`dpnp.sort` : Return a sorted copy of an array.
    :obj:`dpnp.histogram` : Produce histogram from 1-D data.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([11,12,13,14,15])
    >>> np.searchsorted(a, 13)
    array(2)
    >>> np.searchsorted(a, 13, side='right')
    array(3)
    >>> v = np.array([-10, 20, 12, 13])
    >>> np.searchsorted(a, v)
    array([0, 5, 1, 2])

    """

    usm_a = dpnp.get_usm_ndarray(a)
    if dpnp.isscalar(v):
        usm_v = dpt.asarray(v, sycl_queue=a.sycl_queue, usm_type=a.usm_type)
    else:
        usm_v = dpnp.get_usm_ndarray(v)

    usm_sorter = None if sorter is None else dpnp.get_usm_ndarray(sorter)
    return dpnp_array._create_from_usm_ndarray(
        dpt.searchsorted(usm_a, usm_v, side=side, sorter=usm_sorter)
    )


def where(condition, x=None, y=None, /, *, order="K", out=None):
    """
    Return elements chosen from `x` or `y` depending on `condition`.

    When only `condition` is provided, this function is a shorthand for
    :obj:`dpnp.nonzero(condition)`.

    For full documentation refer to :obj:`numpy.where`.

    Parameters
    ----------
    condition : {dpnp.ndarray, usm_ndarray}
        When ``True``, yield `x`, otherwise yield `y`.
    x, y : {dpnp.ndarray, usm_ndarray, scalar}, optional
        Values from which to choose. `x`, `y` and `condition` need to be
        broadcastable to some shape.
    order : {"K", "C", "F", "A"}, optional
        Memory layout of the new output arra, if keyword `out` is ``None``.
        Default: ``"K"``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        The array into which the result is written. The data type of `out` must
        match the expected shape and the expected data type of the result.
        If ``None`` then a new array is returned.
        Default: ``None``.

    Returns
    -------
    y : dpnp.ndarray
        An array with elements from `x` when `condition` is ``True``, and
        elements from `y` elsewhere.

    See Also
    --------
    :obj:`dpnp.choose` : Construct an array from an index array and a list of
                         arrays to choose from.
    :obj:`dpnp.nonzero` : Return the indices of the elements that are non-zero.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> np.where(a < 5, a, 10*a)
    array([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])

    This can be used on multidimensional arrays too:

    >>> np.where(np.array([[True, False], [True, True]]),
    ...          np.array([[1, 2], [3, 4]]),
    ...          np.array([[9, 8], [7, 6]]))
    array([[1, 8],
           [3, 4]])

    The shapes of x, y, and the condition are broadcast together:

    >>> x, y = np.ogrid[:3, :4]
    >>> np.where(x < y, x, 10 + y)  # both x and 10+y are broadcast
    array([[10,  0,  0,  0],
           [10, 11,  1,  1],
           [10, 11, 12,  2]])

    >>> a = np.array([[0, 1, 2],
    ...               [0, 2, 4],
    ...               [0, 3, 6]])
    >>> np.where(a < 4, a, -1)  # -1 is broadcast
    array([[ 0,  1,  2],
           [ 0,  2, -1],
           [ 0,  3, -1]])

    """

    missing = (x is None, y is None).count(True)
    if missing == 1:
        raise ValueError("Must provide both 'x' and 'y' or neither.")

    if missing == 2:
        return dpnp.nonzero(condition)

    usm_x = dpnp.get_usm_ndarray_or_scalar(x)
    usm_y = dpnp.get_usm_ndarray_or_scalar(y)
    usm_condition = dpnp.get_usm_ndarray(condition)

    usm_type, queue = get_usm_allocations([condition, x, y])
    if dpnp.isscalar(usm_x):
        usm_x = dpt.asarray(usm_x, usm_type=usm_type, sycl_queue=queue)

    if dpnp.isscalar(usm_y):
        usm_y = dpt.asarray(usm_y, usm_type=usm_type, sycl_queue=queue)

    usm_out = None if out is None else dpnp.get_usm_ndarray(out)
    result = dpnp_array._create_from_usm_ndarray(
        dpt.where(usm_condition, usm_x, usm_y, order=order, out=usm_out)
    )
    return dpnp.get_result_array(result, out)
