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

"""Module Backend (Indexing part)

This module contains interface functions between C backend layer
and the rest of the library

"""


import numpy
from dpnp.dpnp_utils cimport *
from dpnp.dpnp_iface_counting import count_nonzero


__all__ += [
    "dpnp_choose",
    "dpnp_diag_indices",
    "dpnp_diagonal",
    "dpnp_fill_diagonal",
    "dpnp_indices",
    "dpnp_nonzero",
    "dpnp_place",
    "dpnp_put",
    "dpnp_put_along_axis",
    "dpnp_putmask",
    "dpnp_select",
    "dpnp_take",
    "dpnp_take_along_axis",
    "dpnp_tril_indices",
    "dpnp_tril_indices_from",
    "dpnp_triu_indices",
    "dpnp_triu_indices_from"
]


ctypedef void(*custom_indexing_2in_1out_func_ptr_t)(void *, void * , void * , size_t)
ctypedef void(*custom_indexing_2in_1out_func_ptr_t_)(void * , void * , const size_t, size_t * , size_t * , const size_t)
ctypedef void(*custom_indexing_2in_func_ptr_t)(void * , void * , size_t * , const size_t)
ctypedef void(*custom_indexing_3in_func_ptr_t)(void * , void * , void *, const size_t, const size_t)
ctypedef void(*custom_indexing_6in_func_ptr_t)(void * , void * , void * , const size_t, const size_t, const size_t)


cpdef dparray dpnp_choose(input, choices):
    res_array = dparray(len(input), dtype=choices[0].dtype)
    for i in range(len(input)):
        res_array[i] = (choices[input[i]])[i]
    return res_array


cpdef tuple dpnp_diag_indices(n, ndim):
    cdef dparray res_item = dpnp.arange(n, dtype=dpnp.int64)

    # yes, all are the same item
    result = []
    for i in range(ndim):
        result.append(res_item)

    return tuple(result)


cpdef dparray dpnp_diagonal(dparray input, offset=0):
    n = min(input.shape[0], input.shape[1])
    res_shape = [None] * (input.ndim - 1)

    if input.ndim > 2:
        for i in range(input.ndim - 2):
            res_shape[i] = input.shape[i + 2]

    if (n + offset) > input.shape[1]:
        res_shape[-1] = input.shape[1] - offset
    elif (n + offset) > input.shape[0]:
        res_shape[-1] = input.shape[0]
    else:
        res_shape[-1] = n + offset

    res_ndim = len(res_shape)

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_DIAGONAL, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef dparray result = dparray(res_shape, dtype=result_type)

    cdef custom_indexing_2in_1out_func_ptr_t_ func = <custom_indexing_2in_1out_func_ptr_t_ > kernel_data.ptr

    func(input.get_data(), result.get_data(), offset, < size_t * > input._dparray_shape.data(), < size_t * > result._dparray_shape.data(), res_ndim)

    return result


cpdef dpnp_fill_diagonal(dparray input, val):
    val_arr = dparray(1, dtype=input.dtype)
    val_arr[0] = val

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_FILL_DIAGONAL, param1_type, param1_type)

    cdef custom_indexing_2in_func_ptr_t func = <custom_indexing_2in_func_ptr_t > kernel_data.ptr

    func(input.get_data(), val_arr.get_data(), < size_t * > input._dparray_shape.data(), input.ndim)


cpdef dparray dpnp_indices(dimensions):
    len_dimensions = len(dimensions)
    res_shape = []
    res_shape.append(len_dimensions)
    for i in range(len_dimensions):
        res_shape.append(dimensions[i])

    result = []
    if len_dimensions == 1:
        res = []
        for i in range(dimensions[0]):
            res.append(i)
        result.append(res)
    else:
        res1 = []
        for i in range(dimensions[0]):
            res = []
            for j in range(dimensions[1]):
                res.append(i)
            res1.append(res)
        result.append(res1)

        res2 = []
        for i in range(dimensions[0]):
            res = []
            for j in range(dimensions[1]):
                res.append(j)
            res2.append(res)
        result.append(res2)

    dpnp_result = dpnp.array(result)
    return dpnp_result


cpdef tuple dpnp_nonzero(dparray in_array1):
    res_count = in_array1.ndim

    # have to go through array one extra time to count size of result arrays
    res_size = count_nonzero(in_array1)

    res_list = []
    for i in range(res_count):
        res_list.append(dparray((res_size, ), dtype=dpnp.int64))
    result = _object_to_tuple(res_list)

    idx = 0
    for i in range(in_array1.size):
        if in_array1[i] != 0:
            ids = get_axis_indeces(i, in_array1.shape)
            for j in range(res_count):
                result[j][idx] = ids[j]
            idx = idx + 1

    return result


cpdef dpnp_place(dparray arr, dparray mask, dparray vals):
    mask_ = dparray(mask.size, dtype=dpnp.int64)
    for i in range(mask.size):
        if mask[i]:
            mask_[i] = 1
        else:
            mask_[i] = 0
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(arr.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_PLACE, param1_type, param1_type)

    cdef custom_indexing_3in_func_ptr_t func = <custom_indexing_3in_func_ptr_t > kernel_data.ptr

    func(arr.get_data(), mask_.get_data(), vals.get_data(), arr.size, vals.size)


cpdef dpnp_put(dparray input, ind, v):
    ind_is_list = isinstance(ind, list)

    if dpnp.isscalar(ind):
        ind_size = 1
    else:
        ind_size = len(ind)
    ind_array = dparray(ind_size, dtype=dpnp.int64)
    if dpnp.isscalar(ind):
        ind_array[0] = ind
    else:
        for i in range(ind_size):
            ind_array[i] = ind[i]

    if dpnp.isscalar(v):
        v_size = 1
    else:
        v_size = len(v)
    v_array = dparray(v_size, dtype=input.dtype)
    if dpnp.isscalar(v):
        v_array[0] = v
    else:
        for i in range(v_size):
            v_array[i] = v[i]

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_PUT, param1_type, param1_type)

    cdef custom_indexing_6in_func_ptr_t func = <custom_indexing_6in_func_ptr_t > kernel_data.ptr

    func(input.get_data(), ind_array.get_data(), v_array.get_data(), input.size, ind_array.size, v_array.size)


cpdef dpnp_put_along_axis(dparray arr, dparray indices, values, axis):
    cdef long size_arr = arr.size
    cdef dparray_shape_type shape_arr = arr.shape
    cdef long size_indices = indices.size

    if axis != arr.ndim - 1:
        output_shape = dparray(len(shape_arr) - 1, dtype=numpy.int64)
        ind = 0
        for id, shape_axis in enumerate(shape_arr):
            if id != axis:
                output_shape[ind] = shape_axis
                ind += 1

        prod = 1
        for i in range(len(output_shape)):
            if output_shape[i] != 0:
                prod *= output_shape[i]

        ind_array = [None] * prod
        arr_shape_offsets = [None] * len(shape_arr)
        acc = 1

        for i in range(len(shape_arr)):
            ind = len(shape_arr) - 1 - i
            arr_shape_offsets[ind] = acc
            acc *= shape_arr[ind]

        output_shape_offsets = [None] * len(shape_arr)
        acc = 1

        for i in range(len(output_shape)):
            ind = len(output_shape) - 1 - i
            output_shape_offsets[ind] = acc
            acc *= output_shape[ind]

        for source_idx in range(size_arr):

            # reconstruct x,y,z from linear source_idx
            xyz = []
            remainder = source_idx
            for i in arr_shape_offsets:
                quotient, remainder = divmod(remainder, i)
                xyz.append(quotient)

            # extract result axis
            result_axis = []
            for idx, offset in enumerate(xyz):
                if idx != axis:
                    result_axis.append(offset)

            # Construct result offset
            result_offset = 0
            for i, result_axis_val in enumerate(result_axis):
                result_offset += (output_shape_offsets[i] * result_axis_val)

            arr_elem = arr.item(source_idx)
            if ind_array[result_offset] is None:
                ind_array[result_offset] = 0
            else:
                ind_array[result_offset] += 1

            if ind_array[result_offset] % size_indices == indices[result_offset % size_indices]:
                arr[source_idx] = values

    else:
        for i in range(size_arr):
            ind = size_indices * (i // size_indices) + indices[i % size_indices]
            arr[ind] = values


cpdef dpnp_putmask(dparray arr, dparray mask, dparray values):
    cpdef int values_size = values.size
    for i in range(arr.size):
        if mask[i]:
            arr[i] = values[i % values_size]


cpdef dparray dpnp_select(condlist, choicelist, default):
    size_ = condlist[0].size
    res_array = dparray(size_, dtype=choicelist[0].dtype)
    pass_val = {a: default for a in range(size_)}
    for i in range(len(condlist)):
        for j in range(size_):
            if (condlist[i])[j]:
                res_array[j] = (choicelist[i])[j]
                pass_val.pop(j)

    for ind, val in pass_val.items():
        res_array[ind] = val

    return res_array.reshape(condlist[0].shape)


cpdef dparray dpnp_take(dparray input, dparray indices):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_TAKE, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef dparray result = dparray(indices.shape, dtype=result_type)

    cdef custom_indexing_2in_1out_func_ptr_t func = <custom_indexing_2in_1out_func_ptr_t > kernel_data.ptr

    func(input.get_data(), indices.get_data(), result.get_data(), indices.size)

    return result


cpdef dparray dpnp_take_along_axis(dparray arr, dparray indices, int axis):
    cdef long size_arr = arr.size
    cdef dparray_shape_type shape_arr = arr.shape
    cdef long size_indices = indices.size
    res_type = arr.dtype

    if axis != arr.ndim - 1:
        res_shape_list = list(shape_arr)
        res_shape_list[axis] = 1
        res_shape = tuple(res_shape_list)

        output_shape = dparray(len(shape_arr) - 1, dtype=numpy.int64)
        ind = 0
        for id, shape_axis in enumerate(shape_arr):
            if id != axis:
                output_shape[ind] = shape_axis
                ind += 1

        prod = 1
        for i in range(len(output_shape)):
            if output_shape[i] != 0:
                prod *= output_shape[i]

        result_array = [None] * prod
        ind_array = [None] * prod
        arr_shape_offsets = [None] * len(shape_arr)
        acc = 1

        for i in range(len(shape_arr)):
            ind = len(shape_arr) - 1 - i
            arr_shape_offsets[ind] = acc
            acc *= shape_arr[ind]

        output_shape_offsets = [None] * len(shape_arr)
        acc = 1

        for i in range(len(output_shape)):
            ind = len(output_shape) - 1 - i
            output_shape_offsets[ind] = acc
            acc *= output_shape[ind]
            result_offsets = arr_shape_offsets[:]  # need copy. not a reference
        result_offsets[axis] = 0

        for source_idx in range(size_arr):

            # reconstruct x,y,z from linear source_idx
            xyz = []
            remainder = source_idx
            for i in arr_shape_offsets:
                quotient, remainder = divmod(remainder, i)
                xyz.append(quotient)

            # extract result axis
            result_axis = []
            for idx, offset in enumerate(xyz):
                if idx != axis:
                    result_axis.append(offset)

            # Construct result offset
            result_offset = 0
            for i, result_axis_val in enumerate(result_axis):
                result_offset += (output_shape_offsets[i] * result_axis_val)

            arr_elem = arr.item(source_idx)
            if ind_array[result_offset] is None:
                ind_array[result_offset] = 0
            else:
                ind_array[result_offset] += 1

            if ind_array[result_offset] % size_indices == indices[result_offset % size_indices]:
                result_array[result_offset] = arr_elem

        dpnp_array = dpnp.array(result_array, dtype=res_type)
        dpnp_result_array = dpnp_array.reshape(res_shape)
        return dpnp_result_array

    else:
        result_array = dparray(shape_arr, dtype=res_type)
        for i in range(size_arr):
            ind = size_indices * (i // size_indices) + indices[i % size_indices]
            result_array[i] = arr[ind]
        return result_array


cpdef tuple dpnp_tril_indices(n, k=0, m=None):
    array1 = []
    array2 = []
    if m is None:
        for i in range(n):
            for j in range(i + 1 + k):
                if j >= n:
                    continue
                else:
                    array1.append(i)
                    array2.append(j)
    else:
        for i in range(n):
            for j in range(i + 1 + k):
                if j < m:
                    array1.append(i)
                    array2.append(j)

    dparray1 = dpnp.array(array1, dtype=dpnp.int64)
    dparray2 = dpnp.array(array2, dtype=dpnp.int64)
    return (dparray1, dparray2)


cpdef tuple dpnp_tril_indices_from(arr, k=0):
    m = arr.shape[0]
    n = arr.shape[1]
    array1 = []
    array2 = []
    if m is None:
        for i in range(n):
            for j in range(i + 1 + k):
                if j >= n:
                    continue
                else:
                    array1.append(i)
                    array2.append(j)
    else:
        for i in range(n):
            for j in range(i + 1 + k):
                if j < m:
                    array1.append(i)
                    array2.append(j)

    dparray1 = dpnp.array(array1, dtype=dpnp.int64)
    dparray2 = dpnp.array(array2, dtype=dpnp.int64)
    return (dparray1, dparray2)


cpdef tuple dpnp_triu_indices(n, k=0, m=None):
    array1 = []
    array2 = []
    if m is None:
        for i in range(n):
            for j in range(i + k, n):
                array1.append(i)
                array2.append(j)
    else:
        for i in range(n):
            for j in range(i + k, m):
                array1.append(i)
                array2.append(j)

    dparray1 = dpnp.array(array1, dtype=dpnp.int64)
    dparray2 = dpnp.array(array2, dtype=dpnp.int64)
    return (dparray1, dparray2)


cpdef tuple dpnp_triu_indices_from(arr, k=0):
    m = arr.shape[0]
    n = arr.shape[1]
    array1 = []
    array2 = []
    if m is None:
        for i in range(n):
            for j in range(i + k, n):
                array1.append(i)
                array2.append(j)
    else:
        for i in range(n):
            for j in range(i + k, m):
                array1.append(i)
                array2.append(j)

    dparray1 = dpnp.array(array1, dtype=dpnp.int64)
    dparray2 = dpnp.array(array2, dtype=dpnp.int64)
    return (dparray1, dparray2)
