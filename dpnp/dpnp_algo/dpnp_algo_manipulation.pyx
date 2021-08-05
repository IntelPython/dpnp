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

"""Module Backend (Array manipulation routines)

This module contains interface functions between C backend layer
and the rest of the library

"""

# NO IMPORTs here. All imports must be placed into main "dpnp_algo.pyx" file

__all__ += [
    "dpnp_atleast_2d",
    "dpnp_atleast_3d",
    "dpnp_copyto",
    "dpnp_expand_dims",
    "dpnp_repeat",
    "dpnp_transpose",
    "dpnp_squeeze",
]


# C function pointer to the C library template functions
ctypedef void(*fptr_custom_elemwise_transpose_1in_1out_t)(void * , size_t * , size_t * ,
                                                          size_t * , size_t, void * , size_t)
ctypedef void(*fptr_dpnp_repeat_t)(const void *, void * , const size_t , const size_t)


cpdef dparray dpnp_atleast_2d(dparray arr):
    cdef size_t arr_ndim = arr.ndim
    cdef long arr_size = arr.size
    if arr_ndim == 1:
        result = dparray((1, arr_size), dtype=arr.dtype)
        for i in range(arr_size):
            result[0, i] = arr[i]
        return result
    else:
        return arr


cpdef dparray dpnp_atleast_3d(dparray arr):
    cdef size_t arr_ndim = arr.ndim
    cdef dparray_shape_type arr_shape = arr.shape
    cdef long arr_size = arr.size
    if arr_ndim == 1:
        result = dparray((1, 1, arr_size), dtype=arr.dtype)
        for i in range(arr_size):
            result[0, 0, i] = arr[i]
        return result
    elif arr_ndim == 2:
        result = dparray((1, arr_shape[0], arr_shape[1]), dtype=arr.dtype)
        for i in range(arr_shape[0]):
            for j in range(arr_shape[1]):
                result[0, i, j] = arr[i, j]
        return result
    else:
        return arr


cpdef dpnp_copyto(utils.dpnp_descriptor dst, utils.dpnp_descriptor src, where=True):
    # Convert string type names (dparray.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType dst_type = dpnp_dtype_to_DPNPFuncType(dst.dtype)
    cdef DPNPFuncType src_type = dpnp_dtype_to_DPNPFuncType(src.dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_COPYTO, dst_type, src_type)

    cdef fptr_1in_1out_t func = <fptr_1in_1out_t > kernel_data.ptr
    # Call FPTR function
    func(dst.get_data(), src.get_data(), dst.size)


cpdef utils.dpnp_descriptor dpnp_expand_dims(utils.dpnp_descriptor in_array, axis):
    axis_tuple = utils._object_to_tuple(axis)
    result_ndim = len(axis_tuple) + in_array.ndim

    if len(axis_tuple) == 0:
        axis_ndim = 0
    else:
        axis_ndim = max(-min(0, min(axis_tuple)), max(0, max(axis_tuple))) + 1

    axis_norm = utils._object_to_tuple(utils.normalize_axis(axis_tuple, result_ndim))

    if axis_ndim - len(axis_norm) > in_array.ndim:
        utils.checker_throw_axis_error("dpnp_expand_dims", "axis", axis, axis_ndim)

    if len(axis_norm) > len(set(axis_norm)):
        utils.checker_throw_value_error("dpnp_expand_dims", "axis", axis, "no repeated axis")

    cdef dparray_shape_type shape_list
    axis_idx = 0
    for i in range(result_ndim):
        if i in axis_norm:
            shape_list.push_back(1)
        else:
            shape_list.push_back(in_array.shape[axis_idx])
            axis_idx = axis_idx + 1

    cdef utils.dpnp_descriptor result = dpnp.get_dpnp_descriptor(dpnp_copy(in_array).get_pyobj().reshape(shape_list))

    return result


cpdef dparray dpnp_repeat(utils.dpnp_descriptor array1, repeats, axes=None):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_REPEAT, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef long new_size = array1.size * repeats
    cdef dparray result = dparray((new_size, ), dtype=array1.dtype)

    cdef fptr_dpnp_repeat_t func = <fptr_dpnp_repeat_t > kernel_data.ptr
    func(array1.get_data(), result.get_data(), repeats, array1.size)

    return result


cpdef dparray dpnp_transpose(utils.dpnp_descriptor array1, axes=None):
    cdef dparray_shape_type input_shape = array1.shape
    cdef size_t input_shape_size = array1.ndim
    cdef dparray_shape_type result_shape = dparray_shape_type(input_shape_size, 1)

    cdef dparray_shape_type permute_axes
    if axes is None:
        """
        template to do transpose a tensor
        input_shape=[2, 3, 4]
        permute_axes=[2, 1, 0]
        after application `permute_axes` to `input_shape` result:
        result_shape=[4, 3, 2]

        'do nothing' axes variable is `permute_axes=[0, 1, 2]`

        test: pytest tests/third_party/cupy/manipulation_tests/test_transpose.py::TestTranspose::test_external_transpose_all
        """
        permute_axes = list(reversed([i for i in range(input_shape_size)]))
    else:
        permute_axes = utils.normalize_axis(axes, input_shape_size)

    for i in range(input_shape_size):
        """ construct output shape """
        result_shape[i] = input_shape[permute_axes[i]]

    # convert string type names (dparray.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_TRANSPOSE, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    # ceate result array with type given by FPTR data
    cdef dparray result = dparray(result_shape, dtype=result_type)

    cdef fptr_custom_elemwise_transpose_1in_1out_t func = <fptr_custom_elemwise_transpose_1in_1out_t > kernel_data.ptr
    # call FPTR function
    func(array1.get_data(), < size_t * > input_shape.data(), < size_t * > result_shape.data(),
         < size_t * > permute_axes.data(), input_shape_size, result.get_data(), array1.size)

    return result


cpdef utils.dpnp_descriptor dpnp_squeeze(utils.dpnp_descriptor in_array, axis):
    cdef dparray_shape_type shape_list
    if axis is None:
        for i in range(in_array.ndim):
            if in_array.shape[i] != 1:
                shape_list.push_back(in_array.shape[i])
    else:
        axis_norm = utils._object_to_tuple(utils.normalize_axis(utils._object_to_tuple(axis), in_array.ndim))
        for i in range(in_array.ndim):
            if i in axis_norm:
                if in_array.shape[i] != 1:
                    utils.checker_throw_value_error("dpnp_squeeze", "axis", axis, "axis has size not equal to one")
            else:
                shape_list.push_back(in_array.shape[i])

    cdef utils.dpnp_descriptor result = dpnp.get_dpnp_descriptor(dpnp_copy(in_array).get_pyobj().reshape(shape_list))

    return result
