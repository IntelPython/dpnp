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
Interface of the Binary operations of the Intel NumPy

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
    ogical_and, bitwise_or, bitwise_xor, binary_repr

    """
    supported_dtypes = [numpy.int64, numpy.int32]
    return check_nd_call(numpy.bitwise_and, dpnp_bitwise_and, supported_dtypes, x1, x2,
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
    logical_or, bitwise_and, bitwise_xor, binary_repr

    """
    supported_dtypes = [numpy.int64, numpy.int32]
    return check_nd_call(numpy.bitwise_or, dpnp_bitwise_or, supported_dtypes, x1, x2,
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
    logical_xor, bitwise_and, bitwise_or, binary_repr

    """
    supported_dtypes = [numpy.int64, numpy.int32]
    return check_nd_call(numpy.bitwise_xor, dpnp_bitwise_xor, supported_dtypes, x1, x2,
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
    bitwise_and, bitwise_or, bitwise_xor, logical_not, binary_repr

    """
    supported_dtypes = [numpy.int64, numpy.int32]
    return check_nd_call(numpy.invert, dpnp_invert, supported_dtypes, x, **kwargs)


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
    right_shift, binary_repr

    """
    supported_dtypes = [numpy.int64, numpy.int32]
    return check_nd_call(numpy.left_shift, dpnp_left_shift, supported_dtypes, x1, x2,
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
    left_shift, binary_repr

    """
    supported_dtypes = [numpy.int64, numpy.int32]
    return check_nd_call(numpy.right_shift, dpnp_right_shift, supported_dtypes, x1, x2,
                         check_sizes=True, check_shapes=True, check_dtypes=True, **kwargs)
