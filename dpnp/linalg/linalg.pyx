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

"""Module Backend

This module contains interface functions between C backend layer
and the rest of the library

"""

import dpnp
from dpnp.dpnp_utils cimport *
from dpnp.backend cimport *
from dpnp.dparray cimport dparray, dparray_shape_type
import numpy
cimport numpy


__all__ = [
    "dpnp_cholesky",
    "dpnp_det",
    "dpnp_eig",
    "dpnp_eigvals",
    "dpnp_matrix_rank",
    "dpnp_norm"
]


# C function pointer to the C library template functions
ctypedef void(*custom_linalg_1in_1out_func_ptr_t)(void * , void * , size_t * , size_t)


# C function pointer to the C library template functions
ctypedef void(*custom_linalg_1in_1out_func_ptr_t_)(void *, void * , size_t * )


# C function pointer to the C library template functions
ctypedef void(*custom_linalg_1in_1out_with_size_func_ptr_t_)(void * , void * , size_t)


cpdef dparray dpnp_cholesky(dparray input):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_CHOLESKY, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    cdef dparray result = dparray(input.size, dtype=result_type)

    cdef custom_linalg_1in_1out_func_ptr_t_ func = <custom_linalg_1in_1out_func_ptr_t_ > kernel_data.ptr

    func(input.get_data(), result.get_data(), < size_t * > input._dparray_shape.data())
    l_result = result.reshape(input.shape)
    return l_result


cpdef dparray dpnp_det(dparray input):
    cdef size_t n = input.shape[-1]
    cdef size_t size_out = 1
    if input.ndim == 2:
        output_shape = (1, )
        size_out = 0
    else:
        output_shape = tuple((list(input.shape))[:-2])
        for i in range(len(output_shape)):
            size_out *= output_shape[i]

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_DET, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    cdef dparray result = dparray(size_out, dtype=result_type)

    cdef custom_linalg_1in_1out_func_ptr_t func = <custom_linalg_1in_1out_func_ptr_t > kernel_data.ptr

    func(input.get_data(), result.get_data(), < size_t * > input._dparray_shape.data(), input.ndim)

    if size_out > 1:
        dpnp_result = result.reshape(output_shape)
        return dpnp_result
    else:
        return result


cpdef tuple dpnp_eig(dparray x1):
    cdef dparray_shape_type x1_shape = x1.shape

    cdef size_t size = 0 if x1_shape.empty() else x1_shape.front()

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_EIG, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)

    cdef dparray res_val = dparray((size,), dtype=result_type)
    cdef dparray res_vec = dparray(x1_shape, dtype=result_type)

    cdef fptr_2in_1out_t func = <fptr_2in_1out_t > kernel_data.ptr
    # call FPTR function
    func(x1.get_data(), res_val.get_data(), res_vec.get_data(), size)

    return (res_val, res_vec)


cpdef dparray dpnp_eigvals(dparray input):
    cdef dparray_shape_type input_shape = input.shape

    cdef size_t size = 0 if input_shape.empty() else input_shape.front()

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_EIGVALS, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)

    cdef dparray res_val = dparray((size,), dtype=result_type)

    cdef custom_linalg_1in_1out_with_size_func_ptr_t_ func = <custom_linalg_1in_1out_with_size_func_ptr_t_ > kernel_data.ptr
    # call FPTR function
    func(input.get_data(), res_val.get_data(), size)

    return res_val


cpdef dparray dpnp_matrix_rank(dparray input):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MATRIX_RANK, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    cdef dparray result = dparray((1,), dtype=result_type)

    cdef custom_linalg_1in_1out_func_ptr_t func = <custom_linalg_1in_1out_func_ptr_t > kernel_data.ptr

    func(input.get_data(), result.get_data(), < size_t * > input._dparray_shape.data(), input.ndim)

    return result


cpdef dparray dpnp_norm(dparray input, ord=None, axis=None):
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

    if axis is None:
        ndim = input.ndim
        if ((ord is None)  or
            (ord in ('f', 'fro') and ndim ==2) or
            (ord == 2 and ndim == 1)):

            input = input.ravel(order='K')
            sqnorm = dpnp.dot(input, input)
            ret = dpnp.sqrt(sqnorm)
            return dpnp.array([ret], dtype=res_type)

    len_axis = 1 if axis is None else len(axis_)
    if len_axis == 1:
        if ord == numpy.inf:
            return dpnp.array([dpnp.abs(input).max(axis=axis)])
        elif ord == -numpy.inf:
            return dpnp.array([dpnp.abs(input).min(axis=axis)])
        elif ord == 0:
            return dpnp.array([(input != 0).astype(input.dtype).sum(axis=axis)])
        elif ord is None or ord == 2:
            s = input * input
            return dpnp.sqrt(dpnp.sum(s, axis=axis))
        elif isinstance(ord, str):
            raise ValueError(f"Invalid norm order '{ord}' for vectors")
        else:
            absx = dpnp.abs(input)
            absx **= ord
            ret = dpnp.sum(absx, axis=axis)
            ret **= (1 / ord)
            return ret

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
                result_array[0] = input_elem ** 2
            else:
                result_array[0] += input_elem ** 2
        else:
            if result_array[result_offset] is None:
                result_array[result_offset] = input_elem ** 2
            else:
                result_array[result_offset] += input_elem ** 2

    output_size = 1
    for i in output_shape:
        output_size *= i

    for i in range(output_size):
        result_array[i] **= 1/2

    dpnp_array = dpnp.array(result_array, dtype=res_type)
    dpnp_result_array = dpnp_array.reshape(output_shape)
    return dpnp_result_array
