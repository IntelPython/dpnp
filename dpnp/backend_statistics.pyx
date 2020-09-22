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
import dpnp
from dpnp.dpnp_utils cimport checker_throw_type_error, normalize_axis
from dpnp.backend cimport *


__all__ += [
    "dpnp_cov",
    "dpnp_mean",
    "dpnp_min"
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


cpdef dparray dpnp_min(dparray input, axis):
    cdef dparray_shape_type shape_input = input.shape
    cdef long size_input = input.size
    if isinstance(axis, int):
        axis_ = tuple([axis])
    else:
        axis_ = axis
    if axis_ is None:
        output_shape = dparray(1, dtype=numpy.int64)
        output_shape[0] = 1
    else:
        output_shape = dparray(len(shape_input) - len(axis_), dtype=numpy.int64)
        ind = 0
        for id, shape_axis in enumerate(shape_input):
            if id not in axis_:
                output_shape[ind] = shape_axis
                ind += 1
    cdef long prod = 1
    for i in range(len(output_shape)):
        if output_shape[i] != 0:
            prod *= output_shape[i]
    result_array = [None] * prod
    input_shape_offsets = [None] * len(shape_input)
    acc = 1
    for i in range(len(shape_input)):
        ind = len(shape_input) - 1 - i
        input_shape_offsets[ind] = acc
        acc *= shape_input[ind]
    output_shape_offsets = [None] * len(shape_input)
    acc = 1
    if axis_ is not None:
        for i in range(len(output_shape)):
            ind = len(output_shape) - 1 - i
            output_shape_offsets[ind] = acc
            acc *= output_shape[ind]
            result_offsets = input_shape_offsets[:] # need copy. not a reference
        for i in axis_:
            result_offsets[i] = 0

    for source_idx in range(size_input):

        # reconstruct x,y,z from linear source_idx
        xyz = []
        remainder = source_idx
        for i in input_shape_offsets:
            quotient, remainder = divmod(remainder, i)
            xyz.append(quotient)

        # extract result axis
        result_axis = []
        if axis_ is None:
            result_axis = xyz
        else:
            for idx, offset in enumerate(xyz):
                if idx not in axis_:
                    result_axis.append(offset)

        # Construct result offset
        result_offset = 0
        if axis_ is not None:
            for i, result_axis_val in enumerate(result_axis):
              result_offset += (output_shape_offsets[i] * result_axis_val)

        input_elem = input.item(source_idx)
        if axis_ is None:
            if result_array[0] is None:
                result_array[0] = input_elem
            else:
                result_array[0] = min(result_array[0], input_elem)
        else:
            if result_array[result_offset] is None:
                result_array[result_offset] = input_elem
            else:
                result_array[result_offset] = min(result_array[result_offset], input_elem)
    dpnp_array = dpnp.array(result_array, dtype=input.dtype)
    dpnp_result_array = dpnp_array.reshape(output_shape)
    return dpnp_result_array
