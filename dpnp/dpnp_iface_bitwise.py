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


import numpy

from .dpnp_algo.dpnp_elementwise_common import (
    check_nd_call_func,
    dpnp_bitwise_and,
    dpnp_bitwise_or,
    dpnp_bitwise_xor,
    dpnp_invert,
    dpnp_left_shift,
    dpnp_right_shift,
)

__all__ = [
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "invert",
    "left_shift",
    "right_shift",
]


def bitwise_and(
    x1,
    x2,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Compute the bit-wise AND of two arrays element-wise.

    For full documentation refer to :obj:`numpy.bitwise_and`.

    Returns
    -------
    out : dpnp.ndarray
        An array containing the element-wise results.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword arguments `kwargs` are currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Data type of input arrays `x1` and `x2` has to be an integer or boolean data type.

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
    >>> np.bitwise_and(x1, x2)
    [2, 4, 16]

    >>> a = np.array([True, True])
    >>> b = np.array([False, True])
    >>> np.bitwise_and(a, b)
    array([False,  True])

    The ``&`` operator can be used as a shorthand for ``bitwise_and`` on
    :class:`dpnp.ndarray`.

    >>> x1 & x2
    array([ 2,  4, 16])
    """
    return check_nd_call_func(
        numpy.bitwise_and,
        dpnp_bitwise_and,
        x1,
        x2,
        out=out,
        order=order,
        where=where,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def bitwise_or(
    x1,
    x2,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Compute the bit-wise OR of two arrays element-wise.

    For full documentation refer to :obj:`numpy.bitwise_or`.

    Returns
    -------
    out : dpnp.ndarray
        An array containing the element-wise results.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword arguments `kwargs` are currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Data type of input arrays `x1` and `x2` has to be an integer or boolean data type.

    See Also
    --------
    :obj:`dpnp.logical_or` : Compute the truth value of ``x1`` OR ``x2`` element-wise.
    :obj:`dpnp.bitwise_and`: Compute the bit-wise AND of two arrays element-wise.
    :obj:`dpnp.bitwise_xor` : Compute the bit-wise XOR of two arrays element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([2, 5, 255])
    >>> x2 = np.array([4])
    >>> np.bitwise_or(x1, x2)
    array([  6,   5, 255])

    The ``|`` operator can be used as a shorthand for ``bitwise_or`` on
    :class:`dpnp.ndarray`.

    >>> x1 | x2
    array([  6,   5, 255])
    """
    return check_nd_call_func(
        numpy.bitwise_or,
        dpnp_bitwise_or,
        x1,
        x2,
        out=out,
        order=order,
        where=where,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def bitwise_xor(
    x1,
    x2,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Compute the bit-wise XOR of two arrays element-wise.

    For full documentation refer to :obj:`numpy.bitwise_xor`.

    Returns
    -------
    out : dpnp.ndarray
        An array containing the element-wise results.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword arguments `kwargs` are currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Data type of input arrays `x1` and `x2` has to be an integer or boolean data type.

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
    >>> np.bitwise_xor(x1, x2)
    array([26,  5])

    >>> a = np.array([True, True])
    >>> b = np.array([False, True])
    >>> np.bitwise_xor(a, b)
    array([ True, False])

    The ``^`` operator can be used as a shorthand for ``bitwise_xor`` on
    :class:`dpnp.ndarray`.

    >>> a ^ b
    array([ True, False])
    """
    return check_nd_call_func(
        numpy.bitwise_xor,
        dpnp_bitwise_xor,
        x1,
        x2,
        out=out,
        order=order,
        where=where,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def invert(
    x, /, out=None, *, order="K", dtype=None, where=True, subok=True, **kwargs
):
    """
    Compute bit-wise inversion, or bit-wise NOT, element-wise.

    For full documentation refer to :obj:`numpy.invert`.

    Returns
    -------
    out : dpnp.ndarray
        An array containing the element-wise results.

    Limitations
    -----------
    Parameter `x` is supported as either :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword arguments `kwargs` are currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Data type of input array `x` has to be an integer data type.

    See Also
    --------
    :obj:`dpnp.bitwise_and`: Compute the bit-wise AND of two arrays element-wise.
    :obj:`dpnp.bitwise_or` : Compute the bit-wise OR of two arrays element-wise.
    :obj:`dpnp.bitwise_xor` : Compute the bit-wise XOR of two arrays element-wise.
    :obj:`dpnp.logical_not` : Compute the truth value of NOT x element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([13])
    >>> np.invert(x)
    -14

    >>> a = np.array([True, False])
    >>> np.invert(a)
    array([False,  True])

    The ``~`` operator can be used as a shorthand for ``invert`` on
    :class:`dpnp.ndarray`.

    >>> ~a
    array([False,  True])
    """

    return check_nd_call_func(
        numpy.invert,
        dpnp_invert,
        x,
        out=out,
        order=order,
        where=where,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


bitwise_not = invert  # bitwise_not is an alias for invert


def left_shift(
    x1,
    x2,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Shift the bits of an integer to the left.

    For full documentation refer to :obj:`numpy.left_shift`.

    Returns
    -------
    out : dpnp.ndarray
        An array containing the element-wise results.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword arguments `kwargs` are currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input data is supported as integer only.

    See Also
    --------
    :obj:`dpnp.right_shift` : Shift the bits of an integer to the right.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([5])
    >>> x2 = np.array([1, 2, 3])
    >>> np.left_shift(x1, x2)
    array([10, 20, 40])

    The ``<<`` operator can be used as a shorthand for ``left_shift`` on
    :class:`dpnp.ndarray`.

    >>> x1 << x2
    array([10, 20, 40])
    """
    return check_nd_call_func(
        numpy.left_shift,
        dpnp_left_shift,
        x1,
        x2,
        out=out,
        order=order,
        where=where,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def right_shift(
    x1,
    x2,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Shift the bits of an integer to the right.

    For full documentation refer to :obj:`numpy.right_shift`.

    Returns
    -------
    out : dpnp.ndarray
        An array containing the element-wise results.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword arguments `kwargs` are currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input data is supported as integer only.

    See Also
    --------
    :obj:`dpnp.left_shift` : Shift the bits of an integer to the left.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([10])
    >>> x2 = np.array([1, 2, 3])
    >>> np.right_shift(x1, x2)
    array([5, 2, 1])

    The ``>>`` operator can be used as a shorthand for ``right_shift`` on
    :class:`dpnp.ndarray`.

    >>> x1 >> x2
    array([5, 2, 1])
    """
    return check_nd_call_func(
        numpy.right_shift,
        dpnp_right_shift,
        x1,
        x2,
        out=out,
        order=order,
        where=where,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )
