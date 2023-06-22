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


def argmax(x1, axis=None, out=None):
    """
    Returns the indices of the maximum values along an axis.

    For full documentation refer to :obj:`numpy.argmax`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Parameter ``axis`` is supported only with default value ``None``.
    Parameter ``out`` is supported only with default value ``None``.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.argmin` : Returns the indices of the minimum values along an axis.
    :obj:`dpnp.amax` : The maximum value along a given axis.
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
    >>> a.shape
    (2, 3)
    >>> [i for i in a]
    [10, 11, 12, 13, 14, 15]
    >>> np.argmax(a)
    5

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc:
        if axis is not None:
            pass
        elif out is not None:
            pass
        else:
            result_obj = dpnp_argmax(x1_desc).get_pyobj()
            result = dpnp.convert_single_elem_array_to_scalar(result_obj)

            return result

    return call_origin(numpy.argmax, x1, axis, out)


def argmin(x1, axis=None, out=None):
    """
    Returns the indices of the minimum values along an axis.

    For full documentation refer to :obj:`numpy.argmin`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Parameter ``axis`` is supported only with default value ``None``.
    Parameter ``out`` is supported only with default value ``None``.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.argmax` : Returns the indices of the maximum values along an axis.
    :obj:`dpnp.amin` : The minimum value along a given axis.
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
    >>> a.shape
    (2, 3)
    >>> [i for i in a]
    [10, 11, 12, 13, 14, 15]
    >>> np.argmin(a)
    0

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc:
        if axis is not None:
            pass
        elif out is not None:
            pass
        else:
            result_obj = dpnp_argmin(x1_desc).get_pyobj()
            result = dpnp.convert_single_elem_array_to_scalar(result_obj)

            return result

    return call_origin(numpy.argmin, x1, axis, out)


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
        check_input_type = lambda x: isinstance(
            x, (dpnp_array, dpt.usm_ndarray)
        )
        if check_input_type(condition):
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
            if check_input_type(x) and check_input_type(y):
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
