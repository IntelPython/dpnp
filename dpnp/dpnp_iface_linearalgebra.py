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
Interface of the Linear Algebra part of the DPNP

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
import dpnp.config as config


__all__ = [
    "dot",
    "einsum",
    "einsum_path",
    "inner",
    "kron",
    "outer",
    "tensordot",
    "vdot"
]


def dot(x1, x2, **kwargs):
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

    is_x1_dparray = isinstance(x1, dparray)
    is_x2_dparray = isinstance(x2, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray and not kwargs):
        dim1 = x1.ndim
        dim2 = x2.ndim

        if not (dim1 >= 2 and dim2 == 1) and not (dim1 >= 2 and dim2 >= 2) and (x1.dtype == x2.dtype):
            result = dpnp_dot(x1, x2)

            # scalar returned
            if result.shape == (1,):
                return result.dtype.type(result[0])

            return result

    return call_origin(numpy.dot, x1, x2, **kwargs)


def einsum(*args, **kwargs):
    """
    einsum(subscripts, *operands, dtype=False)

    Evaluates the Einstein summation convention on the operands.
    Using the Einstein summation convention, many common multi-dimensional
    array operations can be represented in a simple fashion. This function
    provides a way to compute such summations.

    See Also
    --------
    :func:`numpy.einsum`

    """

    return call_origin(numpy.einsum, *args, **kwargs)


def einsum_path(*args, **kwargs):
    """
    einsum_path(subscripts, *operands, optimize='greedy')

    Evaluates the lowest cost contraction order for an einsum expression by
    considering the creation of intermediate arrays.

    See Also
    --------
    :meth:`numpy.einsum_path`

    """

    return call_origin(numpy.einsum_path, *args, **kwargs)


def inner(x1, x2, **kwargs):
    """
    Returns the inner product of two vectors.

    The input arrays are flattened into 1-D vectors and then it performs inner
    product of these vectors.

    .. seealso:: :func:`numpy.inner`

    """

    is_x1_dparray = isinstance(x1, dparray)
    is_x2_dparray = isinstance(x2, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray and not kwargs):
        return dpnp_inner(x1, x2)

    return call_origin(numpy.inner, x1, x2, **kwargs)


def kron(x1, x2, **kwargs):
    """
    Returns the kronecker product of two arrays.

    .. seealso:: :func:`numpy.kron`

    """

    return call_origin(numpy.kron, x1, x2, **kwargs)


def outer(x1, x2, **kwargs):
    """
    Returns the outer product of two vectors.

    The input arrays are flattened into 1-D vectors and then it performs outer
    product of these vectors.

    .. seealso:: :func:`numpy.outer`

    """

    is_x1_dparray = isinstance(x1, dparray)
    is_x2_dparray = isinstance(x2, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray and not kwargs):
        return dpnp_outer(x1, x2)

    return call_origin(numpy.outer, x1, x2, **kwargs)


def tensordot(x1, x2, axes=2):
    """
    Compute tensor dot product along specified axes.

    .. seealso:: :func:`numpy.tensordot`

    """

    is_x1_dparray = isinstance(x1, dparray)
    is_x2_dparray = isinstance(x2, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray and (axes == 1)):
        return dpnp_tensordot(x1, x2)  # dpnp_matmul

    return call_origin(numpy.tensordot, x1, x2, axes)


def vdot(*args, **kwargs):
    """
    Return the dot product of two vectors.

    See Also
    --------
    :func:`numpy.vdot`

    """

    return dpnp.dot(*args, **kwargs)
