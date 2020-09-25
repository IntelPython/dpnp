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


from dpnp.dparray import dparray
from dpnp.dpnp_utils import checker_throw_value_error, use_origin_backend
from dpnp.linalg.linalg import *
import dpnp
import numpy


__all__ = [
    "eig",
    "matrix_power"
]


def eig(in_array1):
    """
    Compute the eigenvalues and right eigenvectors of a square array.
    Parameters
    ----------
    a : (..., M, M) array
        Matrices for which the eigenvalues and right eigenvectors will
        be computed
    Returns
    -------
    w : (..., M) array
        The eigenvalues, each repeated according to its multiplicity.
        The eigenvalues are not necessarily ordered. The resulting
        array will be of complex type, unless the imaginary part is
        zero in which case it will be cast to a real type. When `a`
        is real the resulting eigenvalues will be real (0 imaginary
        part) or occur in conjugate pairs
    v : (..., M, M) array
        The normalized (unit "length") eigenvectors, such that the
        column ``v[:,i]`` is the eigenvector corresponding to the
        eigenvalue ``w[i]``.
    """

    if (use_origin_backend()):
        return numpy.linalg.eig(in_array1)

    return dpnp_eig(in_array1)


def matrix_power(input, count):
    """
    Raise a square matrix to the (integer) power `count`.

    Parameters
    ----------
    input : sequence of array_like

    Returns
    -------
    output : dparray
        Returns the dot product of the supplied arrays.

    See Also
    --------
    :meth:`numpy.linalg.matrix_power`

    """

    is_input_dparray = isinstance(input, dparray)

    if not use_origin_backend(input) and is_input_dparray and count > 0:
        result = input
        for id in range(count - 1):
            result = dpnp.matmul(result, input)

        return result

    input1 = dpnp.asnumpy(input) if is_input_dparray else input

    # TODO need to put dparray memory into NumPy call
    result_numpy = numpy.linalg.matrix_power(input1, count)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result
