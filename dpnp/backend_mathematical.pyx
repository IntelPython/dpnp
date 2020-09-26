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

"""Module Backend (Mathematical part)

This module contains interface functions between C backend layer
and the rest of the library

"""


from libcpp.map cimport map
from libcpp.string cimport string
from dpnp.dpnp_utils cimport checker_throw_type_error

import dpnp
import numpy
cimport numpy

__all__ += [
    "dpnp_absolute",
    "dpnp_add",
    'dpnp_arctan2',
    "dpnp_ceil",
    "dpnp_divide",
    "dpnp_fabs",
    "dpnp_floor",
    "dpnp_floor_divide",
    "dpnp_fmod",
    'dpnp_hypot',
    "dpnp_maximum",
    "dpnp_minimum",
    "dpnp_multiply",
    "dpnp_negative",
    "dpnp_power",
    "dpnp_prod",
    "dpnp_sign",
    "dpnp_subtract",
    "dpnp_sum",
    "dpnp_trunc"
]

# binary names definition because they are unicode in Python
cdef string float64_name = numpy.float64.__name__.encode()
cdef string float32_name = numpy.float32.__name__.encode()
cdef string int64_name = numpy.int64.__name__.encode()
cdef string int32_name = numpy.int32.__name__.encode()

# C function pointer to the C library template functions
ctypedef void * void_ptr
ctypedef void(*custom_math_2in_1out_func_ptr_t)(void_ptr, void_ptr, void_ptr, size_t)
ctypedef void(*custom_math_1in_1out_func_ptr_t)(void_ptr, void_ptr, size_t)

cdef struct custom_math_2in_1out:
    string return_type  # return type identifier which expected by the `ptr` function
    custom_math_2in_1out_func_ptr_t ptr  # C function pointer

ctypedef custom_math_2in_1out custom_math_2in_1out_t


IF 0:  # can't compile to make it more clear
    pass
    # ctypedef map[string, custom_math_2in_1out_t] 2param_map_t
    # ctypedef map[string, 2param_map_t] 1param_map_t
    # cdef map[string, 1param_map_t] func_map
ELSE:
    cdef map[string, map[string, map[string, custom_math_2in_1out_t]]] func_map


cdef custom_math_2in_1out_t _elementwise_2arg_3type(string name1, string name2, string name3):
    """
    Returns C library kernel pointer and auxiliary data by corresponding type parameters
    """

    # cdef custom_math_2in_1out_t kernel_data = func_map.at(add_name).at(array1_typename).at(array2_typename)
    cdef custom_math_2in_1out_t kernel_data = func_map[name1][name2][name3]
    # TODO need to add a check for map::end()
    # the exception will be ignored because C exception from cdef function

    return kernel_data


cpdef dparray dpnp_absolute(dparray input):
    cdef dparray_shape_type shape_input = input.shape
    cdef long size_input = input.size
    output_shape = dparray(len(shape_input), dtype=numpy.int64)
    for id, shape_ in enumerate(shape_input):
        output_shape[id] = shape_
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
    for i in range(len(output_shape)):
        ind = len(output_shape) - 1 - i
        output_shape_offsets[ind] = acc
        acc *= output_shape[ind]
        result_offsets = input_shape_offsets[:]  # need copy. not a reference

    for source_idx in range(size_input):

        # reconstruct x,y,z from linear source_idx
        xyz = []
        remainder = source_idx
        for i in input_shape_offsets:
            quotient, remainder = divmod(remainder, i)
            xyz.append(quotient)

        result_indexes = []
        for idx, offset in enumerate(xyz):
            result_indexes.append(offset)

        result_offset = 0
        for i, result_indexes_val in enumerate(result_indexes):
            result_offset += (output_shape_offsets[i] * result_indexes_val)

        input_elem = input.item(source_idx)
        result_array[result_offset] = input_elem if input_elem >= 0 else -1 * input_elem

    dpnp_array = dpnp.array(result_array, dtype=input.dtype)
    dpnp_result_array = dpnp_array.reshape(output_shape)
    return dpnp_result_array


cpdef dparray dpnp_add(dparray array1, dparray array2):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(array2.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_ADD, param1_type, param2_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef dparray result = dparray(array1.shape, dtype=result_type)

    cdef custom_math_2in_1out_func_ptr_t func = <custom_math_2in_1out_func_ptr_t > kernel_data.ptr
    func(array1.get_data(), array2.get_data(), result.get_data(), array1.size)

    return result


cpdef dparray dpnp_arctan2(dparray array1, dparray array2):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(array2.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_ARCTAN2, param1_type, param2_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef dparray result = dparray(array1.shape, dtype=result_type)

    cdef custom_math_2in_1out_func_ptr_t func = <custom_math_2in_1out_func_ptr_t > kernel_data.ptr
    func(array1.get_data(), array2.get_data(), result.get_data(), array1.size)

    return result


cpdef dparray dpnp_ceil(dparray array1):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_CEIL, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef dparray result = dparray(array1.shape, dtype=result_type)

    cdef custom_math_1in_1out_func_ptr_t func = <custom_math_1in_1out_func_ptr_t > kernel_data.ptr
    func(array1.get_data(), result.get_data(), array1.size)

    return result


cpdef dparray dpnp_divide(dparray array1, dparray array2):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(array2.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_DIVIDE, param1_type, param2_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef dparray result = dparray(array1.shape, dtype=result_type)

    cdef custom_math_2in_1out_func_ptr_t func = <custom_math_2in_1out_func_ptr_t > kernel_data.ptr
    func(array1.get_data(), array2.get_data(), result.get_data(), array1.size)

    return result


cpdef dparray dpnp_fabs(dparray array1):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_FABS, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef dparray result = dparray(array1.shape, dtype=result_type)

    cdef custom_math_1in_1out_func_ptr_t func = <custom_math_1in_1out_func_ptr_t > kernel_data.ptr
    func(array1.get_data(), result.get_data(), array1.size)

    return result


cpdef dparray dpnp_floor(dparray array1):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_FLOOR, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef dparray result = dparray(array1.shape, dtype=result_type)

    cdef custom_math_1in_1out_func_ptr_t func = <custom_math_1in_1out_func_ptr_t > kernel_data.ptr
    func(array1.get_data(), result.get_data(), array1.size)

    return result


cpdef dparray dpnp_floor_divide(dparray array1, dparray array2):
    cdef dparray result = dparray(array1.shape, dtype=array1.dtype)

    for i in range(result.size):
        result[i] = (array1[i] // array2[i])

    return result


cpdef dparray dpnp_hypot(dparray array1, dparray array2):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(array2.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_HYPOT, param1_type, param2_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef dparray result = dparray(array1.shape, dtype=result_type)

    cdef custom_math_2in_1out_func_ptr_t func = <custom_math_2in_1out_func_ptr_t > kernel_data.ptr
    func(array1.get_data(), array2.get_data(), result.get_data(), array1.size)

    return result


cpdef dparray dpnp_maximum(dparray array1, dparray array2):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(array2.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MAXIMUM, param1_type, param2_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef dparray result = dparray(array1.shape, dtype=result_type)

    cdef custom_math_2in_1out_func_ptr_t func = <custom_math_2in_1out_func_ptr_t > kernel_data.ptr
    func(array1.get_data(), array2.get_data(), result.get_data(), array1.size)

    return result


cpdef dparray dpnp_minimum(dparray array1, dparray array2):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(array2.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MINIMUM, param1_type, param2_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef dparray result = dparray(array1.shape, dtype=result_type)

    cdef custom_math_2in_1out_func_ptr_t func = <custom_math_2in_1out_func_ptr_t > kernel_data.ptr
    func(array1.get_data(), array2.get_data(), result.get_data(), array1.size)

    return result


cpdef dparray dpnp_multiply(dparray array1, dparray array2):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(array2.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MULTIPLY, param1_type, param2_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef dparray result = dparray(array1.shape, dtype=result_type)

    cdef custom_math_2in_1out_func_ptr_t func = <custom_math_2in_1out_func_ptr_t > kernel_data.ptr
    func(array1.get_data(), array2.get_data(), result.get_data(), array1.size)

    return result


cpdef dparray dpnp_negative(dparray array1):
    cdef dparray result = dparray(array1.shape, dtype=array1.dtype)

    for i in range(result.size):
        result[i] = -(array1[i])

    return result


cpdef dparray dpnp_power(dparray array1, dparray array2):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(array2.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_POWER, param1_type, param2_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef dparray result = dparray(array1.shape, dtype=result_type)

    cdef custom_math_2in_1out_func_ptr_t func = <custom_math_2in_1out_func_ptr_t > kernel_data.ptr
    func(array1.get_data(), array2.get_data(), result.get_data(), array1.size)

    return result


cpdef dparray dpnp_fmod(dparray array1, dparray array2):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(array2.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_FMOD, param1_type, param2_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef dparray result = dparray(array1.shape, dtype=result_type)

    cdef custom_math_2in_1out_func_ptr_t func = <custom_math_2in_1out_func_ptr_t > kernel_data.ptr
    func(array1.get_data(), array2.get_data(), result.get_data(), array1.size)

    return result


cpdef dpnp_prod(dparray array1):
    return_type = array1.dtype

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_PROD, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef dparray result = dparray((1), dtype=result_type)

    cdef custom_math_1in_1out_func_ptr_t func = <custom_math_1in_1out_func_ptr_t > kernel_data.ptr
    func(array1.get_data(), result.get_data(), array1.size)

    if array1.dtype == numpy.int32:
        """ Numpy interface inconsistency """
        return_type = numpy.dtype(numpy.int64)

    return return_type.type(result[0])


cpdef dparray dpnp_sign(dparray array1):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_SIGN, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef dparray result = dparray(array1.shape, dtype=result_type)

    cdef custom_math_1in_1out_func_ptr_t func = <custom_math_1in_1out_func_ptr_t > kernel_data.ptr
    func(array1.get_data(), result.get_data(), array1.size)

    return result


cpdef dparray dpnp_subtract(dparray array1, dparray array2):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(array2.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_SUBTRACT, param1_type, param2_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef dparray result = dparray(array1.shape, dtype=result_type)

    cdef custom_math_2in_1out_func_ptr_t func = <custom_math_2in_1out_func_ptr_t > kernel_data.ptr
    func(array1.get_data(), array2.get_data(), result.get_data(), array1.size)

    return result


cpdef dpnp_sum(dparray array):
    call_type = array.dtype
    return_type = call_type
    cdef dparray result = dparray((1), dtype=call_type)

    if call_type == numpy.float64:
        custom_sum_c[double](array.get_data(), result.get_data(), array.size)
    elif call_type == numpy.float32:
        custom_sum_c[float](array.get_data(), result.get_data(), array.size)
    elif call_type == numpy.int64:
        custom_sum_c[long](array.get_data(), result.get_data(), array.size)
    elif call_type == numpy.int32:
        custom_sum_c[int](array.get_data(), result.get_data(), array.size)
        """ Numpy interface inconsistency """
        return_type = numpy.dtype(numpy.int64)
    else:
        checker_throw_type_error("dpnp_sum", call_type)

    return return_type.type(result[0])


cpdef dparray dpnp_trunc(dparray array1):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_TRUNC, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef dparray result = dparray(array1.shape, dtype=result_type)

    cdef custom_math_1in_1out_func_ptr_t func = <custom_math_1in_1out_func_ptr_t > kernel_data.ptr
    func(array1.get_data(), result.get_data(), array1.size)

    return result
