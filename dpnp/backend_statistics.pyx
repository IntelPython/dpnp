# cython: language_level=3
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

"""Module Backend (Statistics part)

This module contains interface functions between C backend layer
and the rest of the library

"""


import numpy
from dpnp.dpnp_utils cimport checker_throw_type_error, normalize_axis


__all__ += [
    "dpnp_cov",
    "dpnp_mean"
]


cpdef dparray dpnp_cov(dparray array1):
    cdef dparray mean = dparray(array1.shape[0], dtype=array1.dtype)
    cdef dparray X = dparray(array1.shape, dtype=array1.dtype)

    # mean(array1, axis=1) #################################
    for i in range(array1.shape[0]):
        sum = 0.0
        for j in range(array1.shape[1]):
            sum += array1[i,j]
        mean[i] = sum/array1.shape[1]
    ########################################################
    #X = array1 - mean[:, None]
    #X = array1 - mean[:, numpy.newaxis]
    #X = array1 - mean.reshape((array1.shape[0], 1))
    for i in range(array1.shape[0]):
        for j in range(array1.shape[1]):
            X[i,j] = array1[i,j] - mean[i]
    ########################################################
    Y = X.transpose()
    res = dpnp_matmul(X,Y)
    return res/(array1.shape[1]-1)


cpdef dparray dpnp_mean(dparray a, axis):
    cdef dparray_shape_type shape_a = a.shape
    cdef long size_a = a.size
    cdef size_t dim_a = a.ndim

    if dim_a == 0:
        """
        Means scalar at input
        shape_a[0] will failed because it is list
        TODO need to copy it into new 'result' array
        """
        return a

    if a.dtype == numpy.float32:
        res_type = numpy.float32
    else:
        res_type = numpy.float64

    if dim_a > 2:
        raise NotImplementedError

    if axis is not None and axis >= dim_a:
        raise IndexError("Tuple index out of range")

    cdef float sum_val = 0

    if axis is None:
        if dim_a == 2:
            for i in range(shape_a[0]):
                for j in range(shape_a[1]):
                    sum_val += a[i, j]
        else:
            for i in range(shape_a[0]):
                sum_val += a[i]

        result = dparray(1, dtype=res_type)
        result[0] = sum_val / size_a

    elif axis == 0:
        if dim_a == 2:
            result = dparray(2, dtype=res_type)
            for i in range(shape_a[1]):
                sum_val = 0
                for j in range(shape_a[0]):
                    sum_val += a[j, i]
                result[i] = sum_val / shape_a[0]
        else:
            result = dparray(1, dtype=res_type)
            sum_val = 0
            for i in range(shape_a[0]):
                sum_val += a[i]
            result[0] = sum_val / size_a

    elif axis == 1:
        result = dparray(shape_a[0], dtype=res_type)
        for i in range(shape_a[0]):
            sum_val = 0
            for j in range(shape_a[1]):
                sum_val += a[i, j]
            result[i] = sum_val / shape_a[1]

    return result
