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
Interface of the Binary operations of the DPNP

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
from dpnp.dpnp_utils import *

__all__ = [
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "invert",
    "left_shift",
    "right_shift",
]


def _check_nd_call(
    origin_func, dpnp_func, x1, x2, dtype=None, out=None, where=True, **kwargs
):
    """Choose function to call based on input and call chosen fucntion."""

    if kwargs:
        pass
    elif where is not True:
        pass
    elif dtype is not None:
        pass
    elif dpnp.isscalar(x1) and dpnp.isscalar(x2):
        # at least either x1 or x2 has to be an array
        pass
    else:
        # get USM type and queue to copy scalar from the host memory into a USM allocation
        if dpnp.isscalar(x1) or dpnp.isscalar(x2):
            usm_type, queue = (
                get_usm_allocations([x1, x2])
                if dpnp.isscalar(x1) or dpnp.isscalar(x2)
                else (None, None)
            )
            dtype = x1.dtype if not dpnp.isscalar(x1) else x2.dtype
        else:
            dtype, usm_type, queue = (None, None, None)

        x1_desc = dpnp.get_dpnp_descriptor(
            x1,
            copy_when_strides=False,
            copy_when_nondefault_queue=False,
            alloc_dtype=dtype,
            alloc_usm_type=usm_type,
            alloc_queue=queue,
        )
        x2_desc = dpnp.get_dpnp_descriptor(
            x2,
            copy_when_strides=False,
            copy_when_nondefault_queue=False,
            alloc_dtype=dtype,
            alloc_usm_type=usm_type,
            alloc_queue=queue,
        )
        if x1_desc and x2_desc:
            if out is not None:
                if not isinstance(out, (dpnp.ndarray, dpt.usm_ndarray)):
                    raise TypeError(
                        "return array must be of supported array type"
                    )
                out_desc = (
                    dpnp.get_dpnp_descriptor(
                        out, copy_when_nondefault_queue=False
                    )
                    or None
                )
            else:
                out_desc = None

            return dpnp_func(
                x1_desc, x2_desc, dtype=dtype, out=out_desc, where=where
            ).get_pyobj()

    return call_origin(
        origin_func, x1, x2, dtype=dtype, out=out, where=where, **kwargs
    )


def bitwise_and(x1, x2, dtype=None, out=None, where=True, **kwargs):
    """
    Compute the bit-wise AND of two arrays element-wise.

    For full documentation refer to :obj:`numpy.bitwise_and`.

    Returns
    -------
    y : dpnp.ndarray
        An array containing the element-wise results of positive square root.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `dtype` and `where` are supported with their default values.
    Keyword arguments `kwargs` are currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Data type of input arrays `x` and `y` are limited by :obj:`dpnp.bool`, :obj:`dpnp.int32`
    and :obj:`dpnp.int64`.

    See Also
    --------
    :obj:`dpnp.logical_and` : Compute the truth value of ``x1`` AND ``x2`` element-wise.
    :obj:`dpnp.bitwise_or`: Compute the bit-wise OR of two arrays element-wise.
    :obj:`dpnp.bitwise_xor` : Compute the bit-wise XOR of two arrays element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([2, 5, 255])
    >>> x2 = np.array([3,14,16])
    >>> out = np.bitwise_and(x1, x2)
    >>> [i for i in out]
    [2, 4, 16]

    """
    return _check_nd_call(
        numpy.bitwise_and,
        dpnp_bitwise_and,
        x1,
        x2,
        dtype=dtype,
        out=out,
        where=where,
        **kwargs,
    )


def bitwise_or(x1, x2, dtype=None, out=None, where=True, **kwargs):
    """
    Compute the bit-wise OR of two arrays element-wise.

    For full documentation refer to :obj:`numpy.bitwise_or`.

    Returns
    -------
    y : dpnp.ndarray
        An array containing the element-wise results.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `dtype` and `where` are supported with their default values.
    Keyword arguments `kwargs` are currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Data type of input arrays `x` and `y` are limited by :obj:`dpnp.bool`, :obj:`dpnp.int32`
    and :obj:`dpnp.int64`.

    See Also
    --------
    :obj:`dpnp.logical_or` : Compute the truth value of ``x1`` OR ``x2`` element-wise.
    :obj:`dpnp.bitwise_and`: Compute the bit-wise AND of two arrays element-wise.
    :obj:`dpnp.bitwise_xor` : Compute the bit-wise XOR of two arrays element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([2, 5, 255])
    >>> x2 = np.array([4, 4, 4])
    >>> out = np.bitwise_or(x1, x2)
    >>> [i for i in out]
    [6, 5, 255]

    """
    return _check_nd_call(
        numpy.bitwise_or,
        dpnp_bitwise_or,
        x1,
        x2,
        dtype=dtype,
        out=out,
        where=where,
        **kwargs,
    )


def bitwise_xor(x1, x2, dtype=None, out=None, where=True, **kwargs):
    """
    Compute the bit-wise XOR of two arrays element-wise.

    For full documentation refer to :obj:`numpy.bitwise_xor`.

    Returns
    -------
    y : dpnp.ndarray
        An array containing the element-wise results.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `dtype` and `where` are supported with their default values.
    Keyword arguments `kwargs` are currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Data type of input arrays `x` and `y` are limited by :obj:`dpnp.bool`, :obj:`dpnp.int32`
    and :obj:`dpnp.int64`.

    See Also
    --------
    :obj:`dpnp.logical_xor` : Compute the truth value of ``x1`` XOR `x2`, element-wise.
    :obj:`dpnp.bitwise_and`: Compute the bit-wise AND of two arrays element-wise.
    :obj:`dpnp.bitwise_or` : Compute the bit-wise OR of two arrays element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([31, 3])
    >>> x2 = np.array([5, 6])
    >>> out = np.bitwise_xor(x1, x2)
    >>> [i for i in out]
    [26, 5]

    """
    return _check_nd_call(
        numpy.bitwise_xor,
        dpnp_bitwise_xor,
        x1,
        x2,
        dtype=dtype,
        out=out,
        where=where,
        **kwargs,
    )


def invert(x, /, out=None, *, where=True, dtype=None, subok=True, **kwargs):
    """
    Compute bit-wise inversion, or bit-wise NOT, element-wise.

    For full documentation refer to :obj:`numpy.invert`.

    Returns
    -------
    y : dpnp.ndarray
        An array containing the element-wise results.

    Limitations
    -----------
    Parameter `x` is supported as either :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword arguments `kwargs` are currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Data type of input array `x` is limited by :obj:`dpnp.bool`, :obj:`dpnp.int32`
    and :obj:`dpnp.int64`.

    See Also
    --------
    :obj:`dpnp.bitwise_and`: Compute the bit-wise AND of two arrays element-wise.
    :obj:`dpnp.bitwise_or` : Compute the bit-wise OR of two arrays element-wise.
    :obj:`dpnp.bitwise_xor` : Compute the bit-wise XOR of two arrays element-wise.
    :obj:`dpnp.logical_not` : Compute the truth value of NOT x element-wise.

    Examples
    --------
    >>> import dpnp as dp
    >>> x = dp.array([13])
    >>> out = dp.invert(x)
    >>> out[0]
    -14

    """

    if kwargs:
        pass
    elif where is not True:
        pass
    elif dtype is not None:
        pass
    elif subok is not True:
        pass
    else:
        x1_desc = dpnp.get_dpnp_descriptor(x, copy_when_nondefault_queue=False)
        if x1_desc:
            if out is not None:
                if not isinstance(out, (dpnp.ndarray, dpt.usm_ndarray)):
                    raise TypeError(
                        "return array must be of supported array type"
                    )
                out_desc = (
                    dpnp.get_dpnp_descriptor(
                        out, copy_when_nondefault_queue=False
                    )
                    or None
                )
            else:
                out_desc = None
        return dpnp_invert(x1_desc, out_desc).get_pyobj()

    return call_origin(
        numpy.invert,
        x,
        out=out,
        where=where,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


bitwise_not = invert  # bitwise_not is an alias for invert


def left_shift(x1, x2, dtype=None, out=None, where=True, **kwargs):
    """
    Shift the bits of an integer to the left.

    For full documentation refer to :obj:`numpy.left_shift`.

    Returns
    -------
    y : dpnp.ndarray
        An array containing the element-wise results.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `dtype` and `where` are supported with their default values.
    Keyword arguments `kwargs` are currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input data is supported as integer only.

    See Also
    --------
    :obj:`dpnp.right_shift` : Shift the bits of an integer to the right.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([5, 5, 5])
    >>> x2 = np.array([1, 2, 3])
    >>> out = np.left_shift(x1, x2)
    >>> [i for i in out]
    [10, 20, 40]

    """
    return _check_nd_call(
        numpy.left_shift,
        dpnp_left_shift,
        x1,
        x2,
        dtype=dtype,
        out=out,
        where=where,
        **kwargs,
    )


def right_shift(x1, x2, dtype=None, out=None, where=True, **kwargs):
    """
    Shift the bits of an integer to the right.

    For full documentation refer to :obj:`numpy.right_shift`.

    Returns
    -------
    y : dpnp.ndarray
        An array containing the element-wise results.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `dtype` and `where` are supported with their default values.
    Keyword arguments `kwargs` are currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input data is supported as integer only.

    See Also
    --------
    :obj:`dpnp.left_shift` : Shift the bits of an integer to the left.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([10, 10, 10])
    >>> x2 = np.array([1, 2, 3])
    >>> out = np.right_shift(x1, x2)
    >>> [i for i in out]
    [5, 2, 1]

    """
    return _check_nd_call(
        numpy.right_shift,
        dpnp_right_shift,
        x1,
        x2,
        dtype=dtype,
        out=out,
        where=where,
        **kwargs,
    )
