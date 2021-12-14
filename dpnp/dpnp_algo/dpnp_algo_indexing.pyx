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

# NO IMPORTs here. All imports must be placed into main "dpnp_algo.pyx" file

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

ctypedef void(*fptr_dpnp_choose_t)(void * , void * , void ** , size_t, size_t, size_t)
ctypedef void(*fptr_dpnp_diag_indices)(void *, size_t)
ctypedef void(*custom_indexing_2in_1out_func_ptr_t)(void * , const size_t, void * , void * , size_t)
ctypedef void(*custom_indexing_2in_1out_func_ptr_t_)(void *, const size_t, void * , const size_t, shape_elem_type * ,
                                                     shape_elem_type * , const size_t)
ctypedef void(*custom_indexing_2in_func_ptr_t)(void * , void * , shape_elem_type * , const size_t)
ctypedef void(*custom_indexing_3in_func_ptr_t)(void *, void * , void * , const size_t, const size_t)
ctypedef void(*custom_indexing_3in_with_axis_func_ptr_t)(void *, void * , void * , const size_t, shape_elem_type * ,
                                                         const size_t, const size_t, const size_t,)
ctypedef void(*custom_indexing_6in_func_ptr_t)(void * , void * , void * , const size_t, const size_t, const size_t)
ctypedef void(*fptr_dpnp_nonzero_t)(const void *, void * , const size_t, const shape_elem_type * , const size_t ,
                                    const size_t)


cpdef utils.dpnp_descriptor dpnp_choose(utils.dpnp_descriptor input, list choices1):
    cdef vector[void *] choices
    cdef utils.dpnp_descriptor choice
    for desc in choices1:
        choice = desc
        choices.push_back(choice.get_data())

    cdef shape_type_c input_shape = input.shape
    cdef size_t choice_size = choices1[0].size

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(choices1[0].dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_CHOOSE, param1_type, param2_type)

    cdef utils.dpnp_descriptor res_array = utils.create_output_descriptor(input_shape, kernel_data.return_type, None)

    cdef fptr_dpnp_choose_t func = <fptr_dpnp_choose_t > kernel_data.ptr

    func(res_array.get_data(),
         input.get_data(),
         choices.data(),
         input_shape[0],
         choices.size(),
         choice_size)

    return res_array


cpdef tuple dpnp_diag_indices(n, ndim):
    cdef size_t res_size = 0 if n < 0 else n

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(dpnp.int64)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_DIAG_INDICES, param1_type, param1_type)

    cdef fptr_dpnp_diag_indices func = <fptr_dpnp_diag_indices > kernel_data.ptr

    res_list = []
    cdef utils.dpnp_descriptor res_arr
    cdef shape_type_c result_shape = utils._object_to_tuple(res_size)
    for i in range(ndim):
        res_arr = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

        func(res_arr.get_data(), res_size)

        res_list.append(res_arr.get_pyobj())

    return tuple(res_list)

cpdef utils.dpnp_descriptor dpnp_diagonal(dpnp_descriptor input, offset=0):
    cdef shape_type_c input_shape = input.shape

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

    cdef shape_type_c result_shape = res_shape
    res_ndim = len(res_shape)

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_DIAGONAL, param1_type, param1_type)

    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef custom_indexing_2in_1out_func_ptr_t_ func = <custom_indexing_2in_1out_func_ptr_t_ > kernel_data.ptr

    func(input.get_data(),
         input.size,
         result.get_data(),
         offset,
         input_shape.data(),
         result_shape.data(),
         res_ndim)

    return result


cpdef dpnp_fill_diagonal(dpnp_descriptor input, val):
    cdef shape_type_c input_shape = input.shape
    cdef utils.dpnp_descriptor val_arr = utils_py.create_output_descriptor_py((1,), input.dtype, None)
    val_arr.get_pyobj()[0] = val

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_FILL_DIAGONAL, param1_type, param1_type)

    cdef custom_indexing_2in_func_ptr_t func = <custom_indexing_2in_func_ptr_t > kernel_data.ptr

    func(input.get_data(), val_arr.get_data(), input_shape.data(), input.ndim)


cpdef object dpnp_indices(dimensions):
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


cpdef tuple dpnp_nonzero(utils.dpnp_descriptor in_array1):
    cdef shape_type_c shape_arr = in_array1.shape
    res_count = in_array1.ndim

    # have to go through array one extra time to count size of result arrays
    res_size_obj = dpnp_count_nonzero(in_array1)
    cdef size_t res_size = dpnp.convert_single_elem_array_to_scalar(res_size_obj.get_pyobj())

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(in_array1.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_NONZERO, param1_type, param1_type)

    cdef fptr_dpnp_nonzero_t func = <fptr_dpnp_nonzero_t > kernel_data.ptr

    array1_obj = in_array1.get_array()

    res_list = []
    cdef utils.dpnp_descriptor res_arr
    cdef shape_type_c result_shape
    for j in range(res_count):
        result_shape = utils._object_to_tuple(res_size)
        res_arr = utils_py.create_output_descriptor_py(result_shape,
                                                       dpnp.int64,
                                                       None,
                                                       device=array1_obj.sycl_device,
                                                       usm_type=array1_obj.usm_type,
                                                       sycl_queue=array1_obj.sycl_queue)

        func(in_array1.get_data(), res_arr.get_data(), res_arr.size, shape_arr.data(), in_array1.ndim, j)

        res_list.append(res_arr.get_pyobj())

    result = utils._object_to_tuple(res_list)

    return result


cpdef dpnp_place(dpnp_descriptor arr, object mask, dpnp_descriptor vals):
    cdef utils.dpnp_descriptor mask_ = utils_py.create_output_descriptor_py((mask.size,), dpnp.int64, None)
    for i in range(mask.size):
        if mask[i]:
            mask_.get_pyobj()[i] = 1
        else:
            mask_.get_pyobj()[i] = 0
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(arr.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_PLACE, param1_type, param1_type)

    cdef custom_indexing_3in_func_ptr_t func = <custom_indexing_3in_func_ptr_t > kernel_data.ptr

    func(arr.get_data(), mask_.get_data(), vals.get_data(), arr.size, vals.size)


cpdef dpnp_put(dpnp_descriptor input, object ind, v):
    ind_is_list = isinstance(ind, list)

    if dpnp.isscalar(ind):
        ind_size = 1
    else:
        ind_size = len(ind)
    cdef utils.dpnp_descriptor ind_array = utils_py.create_output_descriptor_py((ind_size,), dpnp.int64, None)
    if dpnp.isscalar(ind):
        ind_array.get_pyobj()[0] = ind
    else:
        for i in range(ind_size):
            ind_array.get_pyobj()[i] = ind[i]

    if dpnp.isscalar(v):
        v_size = 1
    else:
        v_size = len(v)
    cdef utils.dpnp_descriptor v_array = utils_py.create_output_descriptor_py((v_size,), input.dtype, None)
    if dpnp.isscalar(v):
        v_array.get_pyobj()[0] = v
    else:
        for i in range(v_size):
            v_array.get_pyobj()[i] = v[i]

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_PUT, param1_type, param1_type)

    cdef custom_indexing_6in_func_ptr_t func = <custom_indexing_6in_func_ptr_t > kernel_data.ptr

    func(input.get_data(), ind_array.get_data(), v_array.get_data(), input.size, ind_array.size, v_array.size)


cpdef dpnp_put_along_axis(dpnp_descriptor arr, dpnp_descriptor indices, dpnp_descriptor values, int axis):
    cdef shape_type_c arr_shape = arr.shape
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(arr.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_PUT_ALONG_AXIS, param1_type, param1_type)

    cdef custom_indexing_3in_with_axis_func_ptr_t func = <custom_indexing_3in_with_axis_func_ptr_t > kernel_data.ptr

    func(arr.get_data(), indices.get_data(), values.get_data(), axis, arr_shape.data(), arr.ndim, indices.size, values.size)


cpdef dpnp_putmask(utils.dpnp_descriptor arr, utils.dpnp_descriptor mask, utils.dpnp_descriptor values):
    cdef int values_size = values.size
    for i in range(arr.size):
        if mask.get_pyobj()[numpy.unravel_index(i, mask.shape)]:
            arr.get_pyobj()[numpy.unravel_index(i, arr.shape)] = values.get_pyobj()[numpy.unravel_index(i % values_size, values.shape)]


cpdef utils.dpnp_descriptor dpnp_select(list condlist, list choicelist, default):
    cdef size_t size_ = condlist[0].size
    cdef utils.dpnp_descriptor res_array = utils_py.create_output_descriptor_py(condlist[0].shape, choicelist[0].dtype, None)

    pass_val = {a: default for a in range(size_)}
    for i in range(len(condlist)):
        for j in range(size_):
            if (condlist[i])[j]:
                res_array.get_pyobj()[j] = (choicelist[i])[j]
                pass_val.pop(j)

    for ind, val in pass_val.items():
        res_array.get_pyobj()[ind] = val

    return res_array


cpdef utils.dpnp_descriptor dpnp_take(utils.dpnp_descriptor input, utils.dpnp_descriptor indices):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_TAKE, param1_type, param1_type)

    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(indices.shape, kernel_data.return_type, None)

    cdef custom_indexing_2in_1out_func_ptr_t func = <custom_indexing_2in_1out_func_ptr_t > kernel_data.ptr

    func(input.get_data(), input.size, indices.get_data(), result.get_data(), indices.size)

    return result


cpdef object dpnp_take_along_axis(object arr, object indices, int axis):
    cdef long size_arr = arr.size
    cdef shape_type_c shape_arr = arr.shape
    cdef shape_type_c output_shape
    cdef long size_indices = indices.size
    res_type = arr.dtype

    if axis != arr.ndim - 1:
        res_shape_list = list(shape_arr)
        res_shape_list[axis] = 1
        res_shape = tuple(res_shape_list)

        output_shape = (0,) * (len(shape_arr) - 1)
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
        dpnp_result_array = dpnp.reshape(dpnp_array, res_shape)
        return dpnp_result_array

    else:
        result_array = utils_py.create_output_descriptor_py(shape_arr, res_type, None).get_pyobj()
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

    array1 = dpnp.array(array1, dtype=dpnp.int64)
    array2 = dpnp.array(array2, dtype=dpnp.int64)
    return (array1, array2)


cpdef tuple dpnp_tril_indices_from(dpnp_descriptor arr, k=0):
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

    array1 = dpnp.array(array1, dtype=dpnp.int64)
    array2 = dpnp.array(array2, dtype=dpnp.int64)
    return (array1, array2)


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

    array1 = dpnp.array(array1, dtype=dpnp.int64)
    array2 = dpnp.array(array2, dtype=dpnp.int64)
    return (array1, array2)


cpdef tuple dpnp_triu_indices_from(dpnp_descriptor arr, k=0):
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

    array1 = dpnp.array(array1, dtype=dpnp.int64)
    array2 = dpnp.array(array2, dtype=dpnp.int64)
    return (array1, array2)
