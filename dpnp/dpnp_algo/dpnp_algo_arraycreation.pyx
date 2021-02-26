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


import dpnp
import numpy

from dpnp.dpnp_utils cimport *
from dpnp.dpnp_algo cimport *


__all__ += [
    "dpnp_copy",
    "dpnp_diag",
    "dpnp_full",
    "dpnp_geomspace",
    "dpnp_linspace",
    "dpnp_logspace",
    "dpnp_meshgrid",
    "dpnp_tri",
    "dpnp_tril",
    "dpnp_triu",
]


ctypedef void(*custom_1in_1out_func_ptr_t)(void * , void * , const int , size_t * , size_t * , const size_t, const size_t)


cpdef dparray dpnp_copy(dparray x1, order, subok):
    return call_fptr_1in_1out(DPNP_FN_COPY, x1, x1.shape)


cpdef dparray dpnp_diag(dparray v, int k):
    if v.ndim == 1:
        n = v.shape[0] + abs(k)

        shape_result = (n, n)
    else:
        n = min(v.shape[0], v.shape[0] + k, v.shape[1], v.shape[1] - k)
        if n < 0:
            n = 0

        shape_result = (n, )

    cdef dparray result = dpnp.zeros(shape_result, dtype=v.dtype)

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(v.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_DIAG, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)

    cdef custom_1in_1out_func_ptr_t func = <custom_1in_1out_func_ptr_t > kernel_data.ptr

    func(v.get_data(), result.get_data(), k, < size_t * > v._dparray_shape.data(), < size_t * > result._dparray_shape.data(), v.ndim, result.ndim)

    return result


cpdef dparray dpnp_full(result_shape, value_in, result_dtype):
    # Convert string type names (dparray.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType dtype_in = dpnp_dtype_to_DPNPFuncType(result_dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_FULL, dtype_in, DPNP_FT_NONE)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    # Create single-element input array with type given by FPTR data
    cdef dparray_shape_type shape_in = (1,)
    cdef dparray array_in = dparray(shape_in, dtype=result_type)
    array_in[0] = value_in
    # Create result array with type given by FPTR data
    cdef dparray result = dparray(result_shape, dtype=result_type)

    cdef fptr_1in_1out_t func = <fptr_1in_1out_t > kernel_data.ptr
    # Call FPTR function
    func(array_in.get_data(), result.get_data(), result.size)

    return result


cpdef dparray dpnp_geomspace(start, stop, num, endpoint, dtype, axis):
    cdef dparray result = dparray(num, dtype=dtype)

    if endpoint:
        steps_count = num - 1
    else:
        steps_count = num

    # if there are steps, then fill values
    if steps_count > 0:
        step = dpnp.power(dpnp.float64(stop) / start, 1.0 / steps_count)
        mult = step
        for i in range(1, result.size):
            result[i] = start * mult
            mult = mult * step
    else:
        step = dpnp.nan

    # if result is not empty, then fiil first and last elements
    if num > 0:
        result[0] = start
        if endpoint and result.size > 1:
            result[result.size - 1] = stop

    return result


# TODO this function should work through dpnp_arange_c
cpdef tuple dpnp_linspace(start, stop, num, endpoint, retstep, dtype, axis):
    cdef dparray result = dparray(num, dtype=dtype)

    if endpoint:
        steps_count = num - 1
    else:
        steps_count = num

    # if there are steps, then fill values
    if steps_count > 0:
        step = (dpnp.float64(stop) - start) / steps_count
        for i in range(1, result.size):
            result[i] = start + step * i
    else:
        step = dpnp.nan

    # if result is not empty, then fiil first and last elements
    if num > 0:
        result[0] = start
        if endpoint and result.size > 1:
            result[result.size - 1] = stop

    return (result, step)


cpdef dparray dpnp_logspace(start, stop, num, endpoint, base, dtype, axis):
    temp = dpnp.linspace(start, stop, num=num, endpoint=endpoint)
    return dpnp.power(base, temp).astype(dtype)


cpdef list dpnp_meshgrid(xi, copy, sparse, indexing):
    cdef dparray res_item

    input_count = len(xi)

    # simple case
    if input_count == 0:
        return []

    # simple case
    if input_count == 1:
        return [dpnp.copy(xi[0])]

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

    result = []
    for i in range(input_count):
        res_item = dparray(shape=shape, dtype=xi[i].dtype)

        for j in range(res_item.size):
            res_item[j] = xi[i][(j // steps[i]) % xi[i].size]

        result.append(res_item)

    return result


cpdef dparray dpnp_tri(N, M, k, dtype):
    cdef dparray result

    if M is None:
        M = N

    result = dparray(shape=(N, M), dtype=dtype)

    for i in range(N):
        diag_idx = max(0, i + k + 1)
        diag_idx = min(diag_idx, M)
        for j in range(diag_idx):
            result[i, j] = 1
        for j in range(diag_idx, M):
            result[i, j] = 0

    return result


cpdef dparray dpnp_tril(dparray m, int k):
    if m.ndim == 1:
        result_shape = (m.shape[0], m.shape[0])
    else:
        result_shape = m.shape

    result_ndim = len(result_shape)
    cdef dparray result = dparray(result_shape, dtype=m.dtype)

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(m.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_TRIL, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)

    cdef custom_1in_1out_func_ptr_t func = <custom_1in_1out_func_ptr_t > kernel_data.ptr

    func(m.get_data(), result.get_data(), k, < size_t * > m._dparray_shape.data(), < size_t * > result._dparray_shape.data(), m.ndim, result.ndim)

    return result


cpdef dparray dpnp_triu(m, k):
    cdef dparray result
    if m.ndim == 1:

        result = dparray(shape=(m.shape[0], m.shape[0]), dtype=m.dtype)

        for i in range(result.size):
            ids = get_axis_indeces(i, result.shape)

            diag_idx = max(-1, ids[result.ndim - 2] + k)
            diag_idx = min(diag_idx, result.shape[result.ndim - 1])

            if ids[result.ndim - 1] >= diag_idx:
                result[i] = m[ids[result.ndim - 1]]
            else:
                result[i] = 0
    else:
        result = dparray(shape=m.shape, dtype=m.dtype)

        for i in range(result.size):
            ids = get_axis_indeces(i, result.shape)

            diag_idx = max(-1, ids[result.ndim - 2] + k)
            diag_idx = min(diag_idx, result.shape[result.ndim - 1])

            if ids[result.ndim - 1] >= diag_idx:
                result[i] = m[i]
            else:
                result[i] = 0

    return result
