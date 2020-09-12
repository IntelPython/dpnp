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
from dpnp.backend cimport *


__all__ += [
    "dpnp_cov",
    "dpnp_mean"
]


cpdef dparray dpnp_cov(dparray array1):
    # behaviour of original numpy
    if array1.ndim > 2:
        raise ValueError("array has more than 2 dimensions")

    if array1.ndim < 2:
        raise NotImplementedError

    # numpy provide result as float64 for any input type
    cdef dparray mean = dparray(array1.shape[0], dtype=numpy.float64)
    cdef dparray X = dparray(array1.shape, dtype=numpy.float64)

    # mean(array1, axis=1) #################################
    # dpmp.mean throws: 'dpnp.dparray.dparray' object is not callable
    for i in range(array1.shape[0]):
        sum = 0.0
        for j in range(array1.shape[1]):
            sum += array1[i, j]
        mean[i] = sum / array1.shape[1]
    ########################################################
    #X = array1 - mean[:, None]
    #X = array1 - mean[:, numpy.newaxis]
    #X = array1 - mean.reshape((array1.shape[0], 1))
    for i in range(array1.shape[0]):
        for j in range(array1.shape[1]):
            X[i, j] = array1[i, j] - mean[i]
    ########################################################
    Y = X.transpose()
    res = dpnp_matmul(X, Y) / (array1.shape[1] - 1)

    return res


cpdef dparray dpnp_mean(dparray input, axis):
    cdef dparray_shape_type shape_input = input.shape
    cdef long size_input = input.size
    cdef size_t dim_input = input.ndim

    if input.dtype == numpy.float32:
        res_type = numpy.float32
    else:
        res_type = numpy.float64

    cdef float sum_val = 0

    if axis is None:
        sum_val = dpnp_sum(input)
        result = dparray((1, ), dtype=res_type)
        result[0] = sum_val
        return result / size_input
    else:
        axis_ = axis if axis >= 0 else -1 * axis
        if dim_input == 2:
            result = dparray(shape_input[~axis_], dtype=res_type)
            for i in range(shape_input[~axis_]):
                sum_val = 0
                for j in range(shape_input[axis_]):
                    index = (i, j)
                    sum_val += input[index[~axis_], index[axis_]]
                result[i] = sum_val
        else:
            result = dparray((1, ), dtype=res_type)
            sum_val = 0
            for i in range(shape_input[axis_]):
                sum_val += input[i]
            result[0] = sum_val
        return result / shape_input[axis_]
