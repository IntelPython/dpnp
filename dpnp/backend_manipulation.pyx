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


from dpnp.dpnp_utils cimport *


__all__ += [
    "dpnp_atleast_2d",
    "dpnp_atleast_3d",
    "dpnp_copyto",
    "dpnp_repeat",
    "dpnp_transpose"
]


# C function pointer to the C library template functions
ctypedef void(*fptr_custom_elemwise_transpose_1in_1out_t)(void * , dparray_shape_type & , dparray_shape_type & ,
                                                          dparray_shape_type &, void * , size_t)


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


cpdef dparray dpnp_copyto(dparray dst, dparray src, where=True):
    cdef dparray_shape_type shape_src = src.shape
    cdef long size_src = src.size
    output_shape = dparray(len(shape_src), dtype=numpy.int64)
    for id, shape_ in enumerate(shape_src):
        output_shape[id] = shape_
    cdef long prod = 1
    for i in range(len(output_shape)):
        if output_shape[i] != 0:
            prod *= output_shape[i]
    result_array = [None] * prod
    src_shape_offsets = [None] * len(shape_src)
    acc = 1
    for i in range(len(shape_src)):
        ind = len(shape_src) - 1 - i
        src_shape_offsets[ind] = acc
        acc *= shape_src[ind]
    output_shape_offsets = [None] * len(shape_src)
    acc = 1
    for i in range(len(output_shape)):
        ind = len(output_shape) - 1 - i
        output_shape_offsets[ind] = acc
        acc *= output_shape[ind]
        result_offsets = src_shape_offsets[:]  # need copy. not a reference

    for source_idx in range(size_src):

        # reconstruct x,y,z from linear source_idx
        xyz = []
        remainder = source_idx
        for i in src_shape_offsets:
            quotient, remainder = divmod(remainder, i)
            xyz.append(quotient)

        result_indexes = []
        for idx, offset in enumerate(xyz):
            result_indexes.append(offset)

        result_offset = 0
        for i, result_indexes_val in enumerate(result_indexes):
            result_offset += (output_shape_offsets[i] * result_indexes_val)

        src_elem = src.item(source_idx)
        dst[source_idx] = src_elem

    return dst


cpdef dparray dpnp_repeat(dparray array1, repeats, axes=None):
    cdef long new_size = array1.size * repeats
    cdef dparray result = dparray((new_size, ), dtype=array1.dtype)

    for idx2 in range(array1.size):
        for idx1 in range(repeats):
            result[(idx2 * repeats) + idx1] = array1[idx2]

    return result


cpdef dparray dpnp_transpose(dparray array1, axes=None):
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
    func(array1.get_data(), input_shape, result_shape, permute_axes, result.get_data(), array1.size)

    return result
