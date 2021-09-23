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

"""Module Backend (array creation part)

This module contains interface functions between C backend layer
and the rest of the library

"""

# NO IMPORTs here. All imports must be placed into main "dpnp_algo.pyx" file

__all__ += [
    "dpnp_copy",
    "dpnp_diag",
    "dpnp_full",
    "dpnp_full_like",
    "dpnp_geomspace",
    "dpnp_identity",
    "dpnp_linspace",
    "dpnp_logspace",
    "dpnp_meshgrid",
    "dpnp_ones",
    "dpnp_ones_like",
    "dpnp_trace",
    "dpnp_tri",
    "dpnp_tril",
    "dpnp_triu",
    "dpnp_vander",
    "dpnp_zeros",
    "dpnp_zeros_like"
]


ctypedef void(*custom_1in_1out_func_ptr_t)(void * , void * , const int , size_t * , size_t * , const size_t, const size_t)
ctypedef void(*ftpr_custom_vander_1in_1out_t)(void *, void * , size_t, size_t, int)
ctypedef void(*custom_indexing_1out_func_ptr_t)(void *, const size_t , const size_t , const int)
ctypedef void(*fptr_dpnp_trace_t)(const void * , void * , const size_t * , const size_t)


cpdef utils.dpnp_descriptor dpnp_copy(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out(DPNP_FN_COPY, x1, x1.shape)


cpdef utils.dpnp_descriptor dpnp_diag(utils.dpnp_descriptor v, int k):
    cdef shape_type_c input_shape = v.shape

    if v.ndim == 1:
        n = v.shape[0] + abs(k)

        shape_result = (n, n)
    else:
        n = min(v.shape[0], v.shape[0] + k, v.shape[1], v.shape[1] - k)
        if n < 0:
            n = 0

        shape_result = (n, )

    result_obj = dpnp.zeros(shape_result, dtype=v.dtype)  # TODO need to call dpnp_zero instead
    cdef utils.dpnp_descriptor result = dpnp_descriptor(result_obj)

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(v.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_DIAG, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)

    cdef custom_1in_1out_func_ptr_t func = <custom_1in_1out_func_ptr_t > kernel_data.ptr
    cdef shape_type_c result_shape = result.shape

    func(v.get_data(), result.get_data(), k, < size_t * > input_shape.data(), < size_t * > result_shape.data(), v.ndim, result.ndim)

    return result


cpdef utils.dpnp_descriptor dpnp_full(result_shape, value_in, result_dtype):
    # Convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType dtype_in = dpnp_dtype_to_DPNPFuncType(result_dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_FULL, dtype_in, DPNP_FT_NONE)

    # Create single-element input fill array with type given by FPTR data
    cdef shape_type_c shape_in = (1,)
    cdef utils.dpnp_descriptor array_fill = utils.create_output_descriptor(shape_in, kernel_data.return_type, None)
    array_fill.get_pyobj()[0] = value_in

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape_c = utils._object_to_tuple(result_shape)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape_c, kernel_data.return_type, None)

    cdef fptr_1in_1out_t func = <fptr_1in_1out_t > kernel_data.ptr
    # Call FPTR function
    func(array_fill.get_data(), result.get_data(), result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_full_like(result_shape, value_in, result_dtype):
    # Convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType dtype_in = dpnp_dtype_to_DPNPFuncType(result_dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_FULL_LIKE, dtype_in, DPNP_FT_NONE)

    # Create single-element input fill array with type given by FPTR data
    cdef shape_type_c shape_in = (1,)
    cdef utils.dpnp_descriptor array_fill = utils.create_output_descriptor(shape_in, kernel_data.return_type, None)
    array_fill.get_pyobj()[0] = value_in

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape_c = utils._object_to_tuple(result_shape)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape_c, kernel_data.return_type, None)

    cdef fptr_1in_1out_t func = <fptr_1in_1out_t > kernel_data.ptr
    # Call FPTR function
    func(array_fill.get_data(), result.get_data(), result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_geomspace(start, stop, num, endpoint, dtype, axis):
    cdef shape_type_c obj_shape = utils._object_to_tuple(num)
    cdef utils.dpnp_descriptor result = utils_py.create_output_descriptor_py(obj_shape, dtype, None)

    if endpoint:
        steps_count = num - 1
    else:
        steps_count = num

    # if there are steps, then fill values
    if steps_count > 0:
        step = dpnp.power(dpnp.float64(stop) / start, 1.0 / steps_count)
        mult = step
        for i in range(1, result.size):
            result.get_pyobj()[i] = start * mult
            mult = mult * step
    else:
        step = dpnp.nan

    # if result is not empty, then fiil first and last elements
    if num > 0:
        result.get_pyobj()[0] = start
        if endpoint and result.size > 1:
            result.get_pyobj()[result.size - 1] = stop

    return result


cpdef utils.dpnp_descriptor dpnp_identity(n, result_dtype):
    cdef DPNPFuncType dtype_in = dpnp_dtype_to_DPNPFuncType(result_dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_IDENTITY, dtype_in, DPNP_FT_NONE)

    cdef shape_type_c shape_in = (n, n)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(shape_in, kernel_data.return_type, None)

    cdef fptr_1out_t func = <fptr_1out_t > kernel_data.ptr
    func(result.get_data(), n)

    return result


# TODO this function should work through dpnp_arange_c
cpdef tuple dpnp_linspace(start, stop, num, endpoint, retstep, dtype, axis):
    cdef shape_type_c obj_shape = utils._object_to_tuple(num)
    cdef utils.dpnp_descriptor result = utils_py.create_output_descriptor_py(obj_shape, dtype, None)

    if endpoint:
        steps_count = num - 1
    else:
        steps_count = num

    # if there are steps, then fill values
    if steps_count > 0:
        step = (dpnp.float64(stop) - start) / steps_count
        for i in range(1, result.size):
            result.get_pyobj()[i] = start + step * i
    else:
        step = dpnp.nan

    # if result is not empty, then fiil first and last elements
    if num > 0:
        result.get_pyobj()[0] = start
        if endpoint and result.size > 1:
            result.get_pyobj()[result.size - 1] = stop

    return (result.get_pyobj(), step)


cpdef utils.dpnp_descriptor dpnp_logspace(start, stop, num, endpoint, base, dtype, axis):
    temp = dpnp.linspace(start, stop, num=num, endpoint=endpoint)
    return dpnp.get_dpnp_descriptor(dpnp.astype(dpnp.power(base, temp), dtype))


cpdef list dpnp_meshgrid(xi, copy, sparse, indexing):
    input_count = len(xi)

    # simple case
    if input_count == 0:
        return []

    # simple case
    if input_count == 1:
        return [dpnp_copy(dpnp.get_dpnp_descriptor(xi[0])).get_pyobj()]

    shape_mult = 1
    for i in range(input_count):
        shape_mult = shape_mult * xi[i].size

    shape_list = []
    for i in range(input_count):
        shape_list.append(xi[i].size)
    if indexing == "xy":
        temp = shape_list[0]
        shape_list[0] = shape_list[1]
        shape_list[1] = temp

    steps = []
    for i in range(input_count):
        shape_mult = shape_mult // shape_list[i]
        steps.append(shape_mult)
    if indexing == "xy":
        temp = steps[0]
        steps[0] = steps[1]
        steps[1] = temp

    shape = tuple(shape_list)

    cdef utils.dpnp_descriptor res_item
    result = []
    for i in range(input_count):
        res_item = utils_py.create_output_descriptor_py(shape, xi[i].dtype, None)

        for j in range(res_item.size):
            res_item.get_pyobj()[j] = xi[i][(j // steps[i]) % xi[i].size]

        result.append(res_item.get_pyobj())

    return result


cpdef utils.dpnp_descriptor dpnp_ones(result_shape, result_dtype):
    return call_fptr_1out(DPNP_FN_ONES, utils._object_to_tuple(result_shape), result_dtype)


cpdef utils.dpnp_descriptor dpnp_ones_like(result_shape, result_dtype):
    return call_fptr_1out(DPNP_FN_ONES_LIKE, utils._object_to_tuple(result_shape), result_dtype)


cpdef utils.dpnp_descriptor dpnp_trace(utils.dpnp_descriptor arr, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    if dtype is None:
        dtype_ = arr.dtype
    else:
        dtype_ = dtype

    cdef utils.dpnp_descriptor diagonal_arr = dpnp_diagonal(arr, offset)
    cdef size_t diagonal_ndim = diagonal_arr.ndim
    cdef shape_type_c diagonal_shape = diagonal_arr.shape

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(arr.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(dtype_)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_TRACE, param1_type, param2_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = diagonal_shape[:-1]
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef fptr_dpnp_trace_t func = <fptr_dpnp_trace_t > kernel_data.ptr

    func(diagonal_arr.get_data(), result.get_data(), < size_t * > diagonal_shape.data(), diagonal_ndim)

    return result


cpdef utils.dpnp_descriptor dpnp_tri(N, M=None, k=0, dtype=numpy.float):
    if M is None:
        M = N

    if dtype == numpy.float:
        dtype = numpy.float64

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_TRI, param1_type, param1_type)

    cdef shape_type_c shape_in = (N, M)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(shape_in, kernel_data.return_type, None)

    cdef custom_indexing_1out_func_ptr_t func = <custom_indexing_1out_func_ptr_t > kernel_data.ptr

    func(result.get_data(), N, M, k)

    return result


cpdef utils.dpnp_descriptor dpnp_tril(utils.dpnp_descriptor m, int k):
    cdef shape_type_c input_shape = m.shape
    cdef shape_type_c result_shape

    if m.ndim == 1:
        result_shape = (m.shape[0], m.shape[0])
    else:
        result_shape = m.shape

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(m.dtype)
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_TRIL, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef custom_1in_1out_func_ptr_t func = <custom_1in_1out_func_ptr_t > kernel_data.ptr
    func(m.get_data(), result.get_data(), k, < size_t * > input_shape.data(), < size_t * > result_shape.data(), m.ndim, result.ndim)

    return result


cpdef utils.dpnp_descriptor dpnp_triu(utils.dpnp_descriptor m, int k):
    cdef shape_type_c input_shape = m.shape
    cdef shape_type_c result_shape

    if m.ndim == 1:
        result_shape = (m.shape[0], m.shape[0])
    else:
        result_shape = m.shape

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(m.dtype)
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_TRIU, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef custom_1in_1out_func_ptr_t func = <custom_1in_1out_func_ptr_t > kernel_data.ptr
    func(m.get_data(), result.get_data(), k, < size_t * > input_shape.data(), < size_t * > result_shape.data(), m.ndim, result.ndim)

    return result


cpdef utils.dpnp_descriptor dpnp_vander(utils.dpnp_descriptor x1, int N, int increasing):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_VANDER, param1_type, DPNP_FT_NONE)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = (x1.size, N)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef ftpr_custom_vander_1in_1out_t func = <ftpr_custom_vander_1in_1out_t > kernel_data.ptr
    func(x1.get_data(), result.get_data(), x1.size, N, increasing)

    return result


cpdef utils.dpnp_descriptor dpnp_zeros(result_shape, result_dtype):
    return call_fptr_1out(DPNP_FN_ZEROS, utils._object_to_tuple(result_shape), result_dtype)


cpdef utils.dpnp_descriptor dpnp_zeros_like(result_shape, result_dtype):
    return call_fptr_1out(DPNP_FN_ZEROS_LIKE, utils._object_to_tuple(result_shape), result_dtype)
