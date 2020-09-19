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
Interface of the Linear Algebra part of the Intel NumPy

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""


import numpy

import dpnp
from dpnp.backend import *
from dpnp.dparray import dparray
from dpnp.dpnp_utils import checker_throw_value_error, use_origin_backend
import dpnp.config as config

__all__ = [
    'dot',
    "einsum",
    "einsum_path",
    "kron",
    "multi_dot",
    "outer"
]


def dot(in_array1, in_array2, out_array=None):
    """
    Dot product of two arrays. Specifically,

    - If both `a` and `b` are 1-D arrays, it is inner product of vectors
      (without complex conjugation).

    - If both `a` and `b` are 2-D arrays, it is matrix multiplication,
      but using :func:`matmul` or ``a @ b`` is preferred.

    - If either `a` or `b` is 0-D (scalar), it is equivalent to :func:`multiply`
      and using ``numpy.multiply(a, b)`` or ``a * b`` is preferred.

    - If `a` is an N-D array and `b` is a 1-D array, it is a sum product over
      the last axis of `a` and `b`.

    - If `a` is an N-D array and `b` is an M-D array (where ``M>=2``), it is a
      sum product over the last axis of `a` and the second-to-last axis of `b`::

        dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

    Parameters
    ----------
    a : array_like
        First argument.
    b : array_like
        Second argument.
    out : ndarray, optional
        Output argument. This must have the exact kind that would be returned
        if it was not used. In particular, it must have the right type, must be
        C-contiguous, and its dtype must be the dtype that would be returned
        for `dot(a,b)`. This is a performance feature. Therefore, if these
        conditions are not met, an exception is raised, instead of attempting
        to be flexible.

    Returns
    -------
    output : ndarray
        Returns the dot product of `a` and `b`.  If `a` and `b` are both
        scalars or both 1-D arrays then a scalar is returned; otherwise
        an array is returned.
        If `out` is given, then it is returned.

    """

    if (use_origin_backend()):
        return numpy.matmul(in_array1, in_array2)

    if out_array is not None:
        checker_throw_value_error("dot", "out_array", type(out_array), None)

    if (in_array1.dtype != in_array2.dtype):
        checker_throw_value_error("dot", "types", in_array2.dtype, in_array1.dtype)

    result = dpnp_dot(in_array1, in_array2)

    # scalar returned
    if result.shape == (1,):
        return result.dtype.type(result[0])

    return result


def einsum(*operands, **kwargs):
    """
    einsum(subscripts, *operands, dtype=False)

    Evaluates the Einstein summation convention on the operands.
    Using the Einstein summation convention, many common multi-dimensional
    array operations can be represented in a simple fashion. This function
    provides a way to compute such summations.

    See Also
    --------
    :meth:`numpy.einsum`

    """

    new_operands = []

    for item in operands:
        if isinstance(item, dparray):
            dpnp_array = dpnp.asnumpy(item)
            new_operands.append(dpnp_array)
        else:
            new_operands.append(item)

    return numpy.einsum(*new_operands, **kwargs)


def einsum_path(*operands, optimize='greedy', einsum_call=False):
    """
    einsum_path(subscripts, *operands, optimize='greedy')

    Evaluates the lowest cost contraction order for an einsum expression by
    considering the creation of intermediate arrays.

    See Also
    --------
    :meth:`numpy.einsum_path`

    """

    new_operands = []

    for item in operands:
        if isinstance(item, dparray):
            dpnp_array = dpnp.asnumpy(item)
            new_operands.append(dpnp_array)
        else:
            new_operands.append(item)

    return numpy.einsum_path(*new_operands, optimize=optimize, einsum_call=einsum_call)


def kron(input1, input2):
    """
    Returns the kronecker product of two arrays.

    .. seealso:: :func:`numpy.kron`

    """

    if isinstance(input1, dparray):
        input1_n = dpnp.asnumpy(input1)
    else:
        input1_n = input1

    if isinstance(input2, dparray):
        input2_n = dpnp.asnumpy(input2)
    else:
        input2_n = input2

    result = numpy.kron(input1_n, input2_n)

    return result


def multi_dot(arrays, out=None):
    """
    Compute the dot product of two or more arrays in a single function call

    Parameters
    ----------
    arrays : sequence of array_like
        If the first argument is 1-D it is treated as row vector.
        If the last argument is 1-D it is treated as column vector.
        The other arguments must be 2-D.
    out : ndarray, optional
        unsupported

    Returns
    -------
    output : ndarray
        Returns the dot product of the supplied arrays.

    See Also
    --------
    :meth:`numpy.multi_dot`

    """

    n = len(arrays)

    if n < 2:
        checker_throw_value_error("multi_dot", "arrays", n, ">1")

    result = arrays[0]
    for id in range(1, n):
        result = dot(result, arrays[id])

    return result


def outer(x1, x2, out=None):
    """
    Returns the outer product of two vectors.

    The input arrays are flattened into 1-D vectors and then it performs outer
    product of these vectors.

    .. seealso:: :func:`numpy.outer`

    """

    is_x1_dparray = isinstance(x1, dparray)
    is_x2_dparray = isinstance(x2, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray and (out is None)):
        return dpnp_outer(x1, x2)

    input1 = dpnp.asnumpy(x1) if is_x1_dparray else x1
    input2 = dpnp.asnumpy(x2) if is_x2_dparray else x2

    # TODO need to put dparray memory into NumPy call
    result_numpy = numpy.outer(input1, input2, out)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result
