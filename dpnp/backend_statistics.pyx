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


import dpnp
import numpy
import dpnp
from dpnp.dpnp_utils cimport *
from dpnp.backend cimport *


__all__ += [
    "dpnp_average",
    "dpnp_cov",
    "dpnp_max",
    "dpnp_mean",
    "dpnp_median",
    "dpnp_min"
]


# C function pointer to the C library template functions
ctypedef void(*fptr_custom_cov_1in_1out_t)(void * , void * , size_t, size_t)


# C function pointer to the C library template functions
ctypedef void(*custom_statistic_1in_1out_func_ptr_t)(void * , void * , size_t * , size_t, size_t * , size_t)


# C function pointer to the C library template functions
ctypedef void(*custom_statistic_1in_1out_func_axis_ptr_t)(void * , void * , size_t * , size_t * , size_t, size_t, size_t * , size_t , size_t)


cpdef dpnp_average(dparray x1):
    array_sum = dpnp_sum(x1)

    """ Numpy interface inconsistency """
    return_type = numpy.float32 if (x1.dtype == numpy.float32) else numpy.float64

    return (return_type(array_sum / x1.size))


cpdef dparray dpnp_cov(dparray array1):
    cdef dparray_shape_type input_shape = array1.shape

    if array1.ndim == 1:
        input_shape.insert(input_shape.begin(), 1)

    # convert string type names (dparray.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_COV, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    # ceate result array with type given by FPTR data
    in_array = array1.astype(result_type)
    cdef dparray result = dparray((input_shape[0], input_shape[0]), dtype=result_type)

    cdef fptr_custom_cov_1in_1out_t func = <fptr_custom_cov_1in_1out_t > kernel_data.ptr
    # call FPTR function
    func(in_array.get_data(), result.get_data(), input_shape[0], input_shape[1])

    return result


cpdef dparray _dpnp_max(dparray input):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MAX, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    cdef dparray result = dparray((1,), dtype=result_type)

    cdef custom_statistic_1in_1out_func_ptr_t func = <custom_statistic_1in_1out_func_ptr_t > kernel_data.ptr

    # stub for interface support
    cdef dparray_shape_type axis
    cdef Py_ssize_t axis_size = 0

    func(input.get_data(), result.get_data(), < size_t * > input._dparray_shape.data(), input.ndim, < size_t * > axis.data(), axis_size)

    return result


cpdef dparray dpnp_max(dparray input, axis):
    if axis is None:
        return _dpnp_max(input)

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
            result_offsets = input_shape_offsets[:]  # need copy. not a reference
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
                result_array[0] = max(result_array[0], input_elem)
        else:
            if result_array[result_offset] is None:
                result_array[result_offset] = input_elem
            else:
                result_array[result_offset] = max(result_array[result_offset], input_elem)
    dpnp_array = dpnp.array(result_array, dtype=input.dtype)
    dpnp_result_array = dpnp_array.reshape(output_shape)
    return dpnp_result_array


cpdef dparray _dpnp_mean(dparray input):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MEAN, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    cdef dparray result = dparray((1,), dtype=result_type)

    cdef custom_statistic_1in_1out_func_ptr_t func = <custom_statistic_1in_1out_func_ptr_t > kernel_data.ptr

    # stub for interface support
    cdef dparray_shape_type axis
    cdef Py_ssize_t axis_size = 0

    func(input.get_data(), result.get_data(), < size_t * > input._dparray_shape.data(), input.ndim, < size_t * > axis.data(), axis_size)

    return result


cpdef dparray dpnp_mean(dparray input, axis):
    if axis is None:
        return _dpnp_mean(input)

    cdef long size_input = input.size
    cdef dparray_shape_type shape_input = input.shape

    if input.dtype == numpy.float32:
        res_type = numpy.float32
    else:
        res_type = numpy.float64

    if size_input == 0:
        return dpnp.array([numpy.nan], dtype=res_type)

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
            result_offsets = input_shape_offsets[:]  # need copy. not a reference
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
                result_array[0] += input_elem
        else:
            if result_array[result_offset] is None:
                result_array[result_offset] = input_elem
            else:
                result_array[result_offset] += input_elem

    del_ = size_input
    if axis_ is not None:
        for i in range(len(shape_input)):
            if i not in axis_:
                del_ = del_ / shape_input[i]
    dpnp_array = dpnp.array(result_array, dtype=input.dtype)
    dpnp_result_array = dpnp_array.reshape(output_shape)
    return dpnp_result_array / del_


cpdef dparray dpnp_median(dparray array1):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MEDIAN, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    cdef dparray result = dparray((1,), dtype=result_type)

    cdef custom_statistic_1in_1out_func_ptr_t func = <custom_statistic_1in_1out_func_ptr_t > kernel_data.ptr

    # stub for interface support
    cdef dparray_shape_type axis
    cdef Py_ssize_t axis_size = 0

    func(array1.get_data(), result.get_data(), < size_t * > array1._dparray_shape.data(), array1.ndim, < size_t * > axis.data(), axis_size)

    return result


cpdef dparray _dpnp_min(dparray input):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MIN, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    cdef dparray result = dparray((1,), dtype=result_type)

    cdef custom_statistic_1in_1out_func_ptr_t func = <custom_statistic_1in_1out_func_ptr_t > kernel_data.ptr

    # stub for interface support
    cdef dparray_shape_type axis
    cdef Py_ssize_t axis_size = 0

    func(input.get_data(), result.get_data(), < size_t * > input._dparray_shape.data(), input.ndim, < size_t * > axis.data(), axis_size)

    return result


cpdef dparray _dpnp_min_(dparray input, axis, output_shape):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MIN_AXIS, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    cdef dparray result = dparray(output_shape, dtype=result_type)

    cdef custom_statistic_1in_1out_func_axis_ptr_t func = <custom_statistic_1in_1out_func_axis_ptr_t > kernel_data.ptr

    cdef dparray_shape_type axis_
    axis_.reserve(len(axis))
    for shape_it in axis:
        if shape_it < 0:
            raise ValueError("Intel NumPy dparray::__init__(): Negative values in 'shape' are not allowed")
        axis_.push_back(shape_it)
    cdef Py_ssize_t axis_size = len(axis)
    cdef Py_ssize_t ind = len(output_shape)

    func(input.get_data(), result.get_data(), < size_t * > input._dparray_shape.data(), < size_t * > result._dparray_shape.data(), input.ndim, result.ndim, < size_t * > axis_.data(), axis_size, ind)

    dpnp_array = dpnp.array(result, dtype=input.dtype)
    dpnp_result_array = dpnp_array.reshape(output_shape)
    return dpnp_result_array


cpdef dparray dpnp_min(dparray input, axis):
    cdef dparray_shape_type shape_input = input.shape
    if axis is None:
        return _dpnp_min(input)
    else:
        if isinstance(axis, int):
            if axis < 0:
                axis_ = tuple([-1 * axis])
            else:
                axis_ = tuple([axis])
        else:
            _axis_ = []
            for i in range(len(axis)):
                if axis[i] < 0:
                    _axis_.append(-1 * axis[i])
                else:
                    _axis_.append(axis[i])
            axis_ = tuple(_axis_)

        output_shape = dparray(len(shape_input) - len(axis_), dtype=numpy.int64)
        ind = 0
        for id, shape_axis in enumerate(shape_input):
            if id not in axis_:
                output_shape[ind] = shape_axis
                ind += 1
        return _dpnp_min_(input, axis_, output_shape)
