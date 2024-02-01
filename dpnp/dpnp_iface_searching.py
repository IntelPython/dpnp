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


import dpctl.tensor as dpt
import numpy

import dpnp
from dpnp.dpnp_algo import *
from dpnp.dpnp_array import dpnp_array
from dpnp.dpnp_utils import *

__all__ = ["argmax", "argmin", "searchsorted", "where"]


def argmax(a, axis=None, out=None, *, keepdims=False):
    """
    Returns the indices of the maximum values along an axis.

    For full documentation refer to :obj:`numpy.argmax`.

    Parameters
    ----------
    a :  {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : int, optional
        Axis along which to search. If ``None``, the function must return
        the index of the maximum value of the flattened array.
        Default: ``None``.
    out :  {dpnp.ndarray, usm_ndarray}, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.
    keepdims : bool
        If ``True``, the reduced axes (dimensions) must be included in the
        result as singleton dimensions, and, accordingly, the result must be
        compatible with the input array. Otherwise, if ``False``, the reduced
        axes (dimensions) must not be included in the result.
        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        If `axis` is ``None``, a zero-dimensional array containing the index of
        the first occurrence of the maximum value; otherwise, a non-zero-dimensional
        array containing the indices of the minimum values. The returned array
        must have the default array index data type.

    See Also
    --------
    :obj:`dpnp.ndarray.argmax` : Equivalent function.
    :obj:`dpnp.nanargmax` : Returns the indices of the maximum values along an axis, igonring NaNs.
    :obj:`dpnp.argmin` : Returns the indices of the minimum values along an axis.
    :obj:`dpnp.max` : The maximum value along a given axis.
    :obj:`dpnp.unravel_index` : Convert a flat index into an index tuple.
    :obj:`dpnp.take_along_axis` : Apply ``np.expand_dims(index_array, axis)``
                                  from argmax to an array as if by calling max.

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

    dpt_array = dpnp.get_usm_ndarray(a)
    result = dpnp_array._create_from_usm_ndarray(
        dpt.argmax(dpt_array, axis=axis, keepdims=keepdims)
    )

    return dpnp.get_result_array(result, out)


def argmin(a, axis=None, out=None, *, keepdims=False):
    """
    Returns the indices of the minimum values along an axis.

    For full documentation refer to :obj:`numpy.argmin`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : int, optional
        Axis along which to search. If ``None``, the function must return
        the index of the minimum value of the flattened array.
        Default: ``None``.
    out : {dpnp.ndarray, usm_ndarray}, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.
    keepdims : bool, optional
        If ``True``, the reduced axes (dimensions) must be included in the
        result as singleton dimensions, and, accordingly, the result must be
        compatible with the input array. Otherwise, if ``False``, the reduced
        axes (dimensions) must not be included in the result.
        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        If `axis` is ``None``, a zero-dimensional array containing the index of
        the first occurrence of the minimum value; otherwise, a non-zero-dimensional
        array containing the indices of the minimum values. The returned array
        must have the default array index data type.

    See Also
    --------
    :obj:`dpnp.ndarray.argmin` : Equivalent function.
    :obj:`dpnp.nanargmin` : Returns the indices of the minimum values along an axis, igonring NaNs.
    :obj:`dpnp.argmax` : Returns the indices of the maximum values along an axis.
    :obj:`dpnp.min` : The minimum value along a given axis.
    :obj:`dpnp.unravel_index` : Convert a flat index into an index tuple.
    :obj:`dpnp.take_along_axis` : Apply ``np.expand_dims(index_array, axis)``
                                  from argmin to an array as if by calling min.

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

    dpt_array = dpnp.get_usm_ndarray(a)
    result = dpnp_array._create_from_usm_ndarray(
        dpt.argmin(dpt_array, axis=axis, keepdims=keepdims)
    )

    return dpnp.get_result_array(result, out)


def searchsorted(a, v, side="left", sorter=None):
    """
    Find indices where elements should be inserted to maintain order.

    For full documentation refer to :obj:`numpy.searchsorted`.

    """

    return call_origin(numpy.where, a, v, side, sorter)


def where(condition, x=None, y=None, /):
    """
    Return elements chosen from `x` or `y` depending on `condition`.

    When only `condition` is provided, this function is a shorthand for
    :obj:`dpnp.nonzero(condition)`.

    For full documentation refer to :obj:`numpy.where`.

    Returns
    -------
    y : dpnp.ndarray
        An array with elements from `x` where `condition` is True, and elements
        from `y` elsewhere.

    Limitations
    -----------
    Parameter `condition` is supported as either :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `x` and `y` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`
    Otherwise the function will be executed sequentially on CPU.
    Input array data types of `x` and `y` are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`nonzero` : The function that is called when `x` and `y`are omitted.

    Examples
    --------
    >>> import dpnp as dp
    >>> a = dp.arange(10)
    >>> d
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> dp.where(a < 5, a, 10*a)
    array([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])

    """

    missing = (x is None, y is None).count(True)
    if missing == 1:
        raise ValueError("Must provide both 'x' and 'y' or neither.")
    elif missing == 2:
        return dpnp.nonzero(condition)
    elif missing == 0:
        if dpnp.is_supported_array_type(condition):
            if numpy.isscalar(x) or numpy.isscalar(y):
                # get USM type and queue to copy scalar from the host memory into a USM allocation
                usm_type, queue = get_usm_allocations([condition, x, y])
                x = (
                    dpt.asarray(x, usm_type=usm_type, sycl_queue=queue)
                    if numpy.isscalar(x)
                    else x
                )
                y = (
                    dpt.asarray(y, usm_type=usm_type, sycl_queue=queue)
                    if numpy.isscalar(y)
                    else y
                )
            if dpnp.is_supported_array_type(x) and dpnp.is_supported_array_type(
                y
            ):
                dpt_condition = (
                    condition.get_array()
                    if isinstance(condition, dpnp_array)
                    else condition
                )
                dpt_x = x.get_array() if isinstance(x, dpnp_array) else x
                dpt_y = y.get_array() if isinstance(y, dpnp_array) else y
                return dpnp_array._create_from_usm_ndarray(
                    dpt.where(dpt_condition, dpt_x, dpt_y)
                )

    return call_origin(numpy.where, condition, x, y)
