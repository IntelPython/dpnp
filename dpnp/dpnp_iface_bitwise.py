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


from dpnp.backend import *
from dpnp.dparray import dparray
from dpnp.dpnp_utils import *
import dpnp

__all__ = [
    'bitwise_and',
    'bitwise_or',
    'bitwise_xor',
    'invert',
    'bitwise_not',
    'left_shift',
    'right_shift',
]


def _check_nd_call(origin_func, dpnp_func, *input_arrays,
                   check_sizes=False, check_shapes=False, check_dtypes=False, **kwargs):
    """
    Choose function to call based on required input arrays types, data types and shapes
    and call chosen fucntion.

    Parameters
    ----------
    origin_func : function
        original function to call if at least one input array didn't meet the requirements
    dpnp_func : function
        dpnp function to call if all the input arrays met the requirements
    input_arrays : tuple(arrays)
        input arrays
    check_sizes : bool
        to check all input arrays sizes are equal
    check_shapes : bool
        to check all input arrays shapes are equal
    check_dtypes : bool
        to check all input arrays data types are equal
    kwargs : dict
        remaining input parameters of the function

    Returns
    -------
        result of the function call
    """
    x1, *_ = input_arrays
    if not use_origin_backend(x1) and not kwargs:
        for x in input_arrays:
            if not isinstance(x, dparray):
                break
        else:
            if check_sizes and len(set(x.size for x in input_arrays)) > 1:
                pass  # fallback to numpy in case of different sizes of input arrays
            elif check_shapes and len(set(x.shape for x in input_arrays)) > 1:
                pass  # fallback to numpy in case of different shapes of input arrays
            elif check_dtypes and len(set(x.dtype for x in input_arrays)) > 1:
                pass  # fallback to numpy in case of different dtypes of input arrays
            else:
                return dpnp_func(*input_arrays)

    return call_origin(origin_func, *input_arrays, **kwargs)


def bitwise_and(x1, x2, **kwargs):
    """
    Compute the bit-wise AND of two arrays element-wise.

    Computes the bit-wise AND of the underlying binary representation of the integers in the input arrays.

    Parameters
    ----------
    x1, x2: array_like or scalar
        Input arrays or scalars. Only integer and boolean types are handled.

    Returns
    -------
    out: ndarray or scalar
        Output array or scalar if both x1 and x2 are scalars.

    See Also
    --------
    :obj:`dpnp.logical_and`, :obj:`dpnp.bitwise_or`, :obj:`dpnp.bitwise_xor`, :obj:`dpnp.binary_repr`

    """
    return _check_nd_call(numpy.bitwise_and, dpnp_bitwise_and, x1, x2,
                          check_sizes=True, check_shapes=True, check_dtypes=True, **kwargs)


def bitwise_or(x1, x2, **kwargs):
    """
    Compute the bit-wise OR of two arrays element-wise.

    Computes the bit-wise OR of the underlying binary representation of the integers in the input arrays.

    Parameters
    ----------
    x1, x2: array_like or scalar
        Input arrays or scalars. Only integer and boolean types are handled.

    Returns
    -------
    out: ndarray or scalar
        Output array or scalar if both x1 and x2 are scalars.

    See Also
    --------
    :obj:`dpnp.logical_or`, :obj:`dpnp.bitwise_and`, :obj:`dpnp.bitwise_xor`, :obj:`dpnp.binary_repr`

    """
    return _check_nd_call(numpy.bitwise_or, dpnp_bitwise_or, x1, x2,
                          check_sizes=True, check_shapes=True, check_dtypes=True, **kwargs)


def bitwise_xor(x1, x2, **kwargs):
    """
    Compute the bit-wise XOR of two arrays element-wise.

    Computes the bit-wise XOR of the underlying binary representation of the integers in the input arrays.

    Parameters
    ----------
    x1, x2: array_like or scalar
        Input arrays or scalars. Only integer and boolean types are handled.

    Returns
    -------
    out: ndarray or scalar
        Output array or scalar if both x1 and x2 are scalars.

    See Also
    --------
    :obj:`dpnp.logical_xor`, :obj:`dpnp.bitwise_and`, :obj:`dpnp.bitwise_or`, :obj:`dpnp.binary_repr`

    """
    return _check_nd_call(numpy.bitwise_xor, dpnp_bitwise_xor, x1, x2,
                          check_sizes=True, check_shapes=True, check_dtypes=True, **kwargs)


def invert(x, **kwargs):
    """
    Compute bit-wise inversion, or bit-wise NOT, element-wise.

    Computes the bit-wise NOT of the underlying binary representation of the integers in the input arrays.

    Parameters
    ----------
    x1, x2: array_like or scalar
        Input arrays or scalars. Only integer and boolean types are handled.

    Returns
    -------
    out: ndarray or scalar
        Output array or scalar if both x1 and x2 are scalars.

    See Also
    --------
    :obj:`dpnp.bitwise_and`, :obj:`dpnp.bitwise_or`, :obj:`dpnp.bitwise_xor`,
    :obj:`dpnp.logical_not`, :obj:`dpnp.binary_repr`

    """
    return _check_nd_call(numpy.invert, dpnp_invert, x, **kwargs)


bitwise_not = invert  # bitwise_not is an alias for invert


def left_shift(x1, x2, **kwargs):
    """
    Shift the bits of an integer to the left.

    Bits are shifted to the left by appending x2 0s at the right of x1.

    Parameters
    ----------
    x1, x2: array_like or int
        Input values.
    x1, x2: array_like or int
        Number of zeros to append to x1. Has to be non-negative.

    Returns
    -------
    out: ndarray or int
        Output array or scalar if both x1 and x2 are scalars.

    See Also
    --------
    :obj:`dpnp.right_shift`, :obj:`dpnp.binary_repr`

    """
    return _check_nd_call(numpy.left_shift, dpnp_left_shift, x1, x2,
                          check_sizes=True, check_shapes=True, check_dtypes=True, **kwargs)


def right_shift(x1, x2, **kwargs):
    """
    Shift the bits of an integer to the right.

    Bits are shifted to the right x2.

    Parameters
    ----------
    x1, x2: array_like or int
        Input values.
    x1, x2: array_like or int
        Number of bits to remove at the right of x1.

    Returns
    -------
    out: ndarray or int
        Output array or scalar if both x1 and x2 are scalars.

    See Also
    --------
    :obj:`dpnp.left_shift`, :obj:`dpnp.binary_repr`

    """
    return _check_nd_call(numpy.right_shift, dpnp_right_shift, x1, x2,
                          check_sizes=True, check_shapes=True, check_dtypes=True, **kwargs)
