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

import numpy
cimport numpy

__all__ += [
    "dpnp_absolute",
    "dpnp_add",
    'dpnp_arctan2',
    "dpnp_divide",
    'dpnp_hypot',
    "dpnp_maximum",
    "dpnp_minimum",
    "dpnp_multiply",
    "dpnp_negative",
    "dpnp_power",
    "dpnp_subtract",
    "dpnp_sum"
]

# binary names definition because they are unicode in Python
cdef string float64_name = numpy.float64.__name__.encode()
cdef string float32_name = numpy.float32.__name__.encode()
cdef string int64_name = numpy.int64.__name__.encode()
cdef string int32_name = numpy.int32.__name__.encode()

# C function pointer to the C library template functions
ctypedef void (*custom_math_2in_1out_func_ptr_t) (void *, void * , void * , size_t)

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


cpdef dparray dpnp_absolute(dparray x):
    cdef dparray_shape_type shape_x = x.shape
    cdef size_t dim_x = x.ndim

    result = dparray(shape_x, dtype=x.dtype)

    if dim_x > 2:
        raise NotImplementedError

    if dim_x == 2:
        for i in range(shape_x[0]):
            for j in range(shape_x[1]):
                elem = x[i, j]
                result[i, j] = elem if elem >= 0 else -1 * elem
    else:
        for i in range(shape_x[0]):
            elem = x[i]
            result[i] = elem if elem >= 0 else -1 * elem

    return result


cpdef dparray dpnp_add(dparray array1, dparray array2):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(array2.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_ADD, param1_type, param2_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    cdef dparray result = dparray(array1.shape, dtype=result_type)

    cdef custom_math_2in_1out_func_ptr_t func = <custom_math_2in_1out_func_ptr_t > kernel_data.ptr
    func(array1.get_data(), array2.get_data(), result.get_data(), array1.size)

    return result


cdef string arctan2_name = "arctan2".encode()
func_map[arctan2_name][float64_name][float64_name] = [float64_name, & custom_elemwise_arctan2_c[double, double, double]]
func_map[arctan2_name][float64_name][float32_name] = [float64_name, & custom_elemwise_arctan2_c[double, float, double]]
func_map[arctan2_name][float64_name][int64_name] = [float64_name, & custom_elemwise_arctan2_c[double, long, double]]
func_map[arctan2_name][float64_name][int32_name] = [float64_name, & custom_elemwise_arctan2_c[double, int, double]]

func_map[arctan2_name][float32_name][float64_name] = [float64_name, & custom_elemwise_arctan2_c[float, double, double]]
func_map[arctan2_name][float32_name][float32_name] = [float32_name, & custom_elemwise_arctan2_c[float, float, float]]
func_map[arctan2_name][float32_name][int64_name] = [float64_name, & custom_elemwise_arctan2_c[float, long, double]]
func_map[arctan2_name][float32_name][int32_name] = [float64_name, & custom_elemwise_arctan2_c[float, int, double]]

func_map[arctan2_name][int64_name][float64_name] = [float64_name, & custom_elemwise_arctan2_c[long, double, double]]
func_map[arctan2_name][int64_name][float32_name] = [float64_name, & custom_elemwise_arctan2_c[long, float, double]]
func_map[arctan2_name][int64_name][int64_name] = [float64_name, & custom_elemwise_arctan2_c[long, long, double]]
func_map[arctan2_name][int64_name][int32_name] = [float64_name, & custom_elemwise_arctan2_c[long, int, double]]

func_map[arctan2_name][int32_name][float64_name] = [float64_name, & custom_elemwise_arctan2_c[int, double, double]]
func_map[arctan2_name][int32_name][float32_name] = [float64_name, & custom_elemwise_arctan2_c[int, float, double]]
func_map[arctan2_name][int32_name][int64_name] = [float64_name, & custom_elemwise_arctan2_c[int, long, double]]
func_map[arctan2_name][int32_name][int32_name] = [float64_name, & custom_elemwise_arctan2_c[int, int, double]]

cpdef dparray dpnp_arctan2(dparray array1, dparray array2):
    cdef dparray result
    param1_type = array1.dtype
    param2_type = array2.dtype

    cdef custom_math_2in_1out_t kernel_data = _elementwise_2arg_3type(arctan2_name, param1_type.name.encode(), param2_type.name.encode())
    if ( < long > kernel_data.ptr == 0):
        checker_throw_type_error("dpnp_arctan2", (param1_type, param2_type))

    result = dparray(array1.shape, dtype=kernel_data.return_type)
    kernel_data.ptr(array1.get_data(), array2.get_data(), result.get_data(), array1.size)

    return result


cdef string divide_name = "divide".encode()
func_map[divide_name][float64_name][float64_name] = [float64_name, & custom_elemwise_divide_c[double, double, double]]
func_map[divide_name][float64_name][float32_name] = [float64_name, & custom_elemwise_divide_c[double, float, double]]
func_map[divide_name][float64_name][int64_name] = [float64_name, & custom_elemwise_divide_c[double, long, double]]
func_map[divide_name][float64_name][int32_name] = [float64_name, & custom_elemwise_divide_c[double, int, double]]

func_map[divide_name][float32_name][float64_name] = [float64_name, & custom_elemwise_divide_c[float, double, double]]
func_map[divide_name][float32_name][float32_name] = [float32_name, & custom_elemwise_divide_c[float, float, float]]
func_map[divide_name][float32_name][int64_name] = [float64_name, & custom_elemwise_divide_c[float, long, double]]
func_map[divide_name][float32_name][int32_name] = [float64_name, & custom_elemwise_divide_c[float, int, double]]

func_map[divide_name][int64_name][float64_name] = [float64_name, & custom_elemwise_divide_c[long, double, double]]
func_map[divide_name][int64_name][float32_name] = [float64_name, & custom_elemwise_divide_c[long, float, double]]
func_map[divide_name][int64_name][int64_name] = [int64_name, & custom_elemwise_divide_c[long, long, long]]
func_map[divide_name][int64_name][int32_name] = [int64_name, & custom_elemwise_divide_c[long, int, long]]

func_map[divide_name][int32_name][float64_name] = [float64_name, & custom_elemwise_divide_c[int, double, double]]
func_map[divide_name][int32_name][float32_name] = [float64_name, & custom_elemwise_divide_c[int, float, double]]
func_map[divide_name][int32_name][int64_name] = [int64_name, & custom_elemwise_divide_c[int, long, long]]
func_map[divide_name][int32_name][int32_name] = [int32_name, & custom_elemwise_divide_c[int, int, int]]

cpdef dparray dpnp_divide(dparray array1, dparray array2):
    cdef dparray result
    param1_type = array1.dtype
    param2_type = array2.dtype

    cdef custom_math_2in_1out_t kernel_data = _elementwise_2arg_3type(divide_name, param1_type.name.encode(), param2_type.name.encode())
    if ( < long > kernel_data.ptr == 0):
        checker_throw_type_error("dpnp_divide", (param1_type, param2_type))

    result = dparray(array1.shape, dtype=kernel_data.return_type)
    kernel_data.ptr(array1.get_data(), array2.get_data(), result.get_data(), array1.size)

    return result


cdef string hypot_name = "hypot".encode()
func_map[hypot_name][float64_name][float64_name] = [float64_name, & custom_elemwise_hypot_c[double, double, double]]
func_map[hypot_name][float64_name][float32_name] = [float64_name, & custom_elemwise_hypot_c[double, float, double]]
func_map[hypot_name][float64_name][int64_name] = [float64_name, & custom_elemwise_hypot_c[double, long, double]]
func_map[hypot_name][float64_name][int32_name] = [float64_name, & custom_elemwise_hypot_c[double, int, double]]

func_map[hypot_name][float32_name][float64_name] = [float64_name, & custom_elemwise_hypot_c[float, double, double]]
func_map[hypot_name][float32_name][float32_name] = [float32_name, & custom_elemwise_hypot_c[float, float, float]]
func_map[hypot_name][float32_name][int64_name] = [float64_name, & custom_elemwise_hypot_c[float, long, double]]
func_map[hypot_name][float32_name][int32_name] = [float64_name, & custom_elemwise_hypot_c[float, int, double]]

func_map[hypot_name][int64_name][float64_name] = [float64_name, & custom_elemwise_hypot_c[long, double, double]]
func_map[hypot_name][int64_name][float32_name] = [float64_name, & custom_elemwise_hypot_c[long, float, double]]
func_map[hypot_name][int64_name][int64_name] = [float64_name, & custom_elemwise_hypot_c[long, long, double]]
func_map[hypot_name][int64_name][int32_name] = [float64_name, & custom_elemwise_hypot_c[long, int, double]]

func_map[hypot_name][int32_name][float64_name] = [float64_name, & custom_elemwise_hypot_c[int, double, double]]
func_map[hypot_name][int32_name][float32_name] = [float64_name, & custom_elemwise_hypot_c[int, float, double]]
func_map[hypot_name][int32_name][int64_name] = [float64_name, & custom_elemwise_hypot_c[int, long, double]]
func_map[hypot_name][int32_name][int32_name] = [float64_name, & custom_elemwise_hypot_c[int, int, double]]

cpdef dparray dpnp_hypot(dparray array1, dparray array2):
    cdef dparray result
    param1_type = array1.dtype
    param2_type = array2.dtype

    cdef custom_math_2in_1out_t kernel_data = _elementwise_2arg_3type(hypot_name, param1_type.name.encode(), param2_type.name.encode())
    if ( < long > kernel_data.ptr == 0):
        checker_throw_type_error("dpnp_hypot", (param1_type, param2_type))

    result = dparray(array1.shape, dtype=kernel_data.return_type)
    kernel_data.ptr(array1.get_data(), array2.get_data(), result.get_data(), array1.size)

    return result


cpdef dparray dpnp_maximum(dparray array1, dparray array2):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(array2.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MAXIMUM, param1_type, param2_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    cdef dparray result = dparray(array1.shape, dtype=result_type)

    cdef custom_math_2in_1out_func_ptr_t func = <custom_math_2in_1out_func_ptr_t > kernel_data.ptr
    func(array1.get_data(), array2.get_data(), result.get_data(), array1.size)

    return result


cpdef dparray dpnp_minimum(dparray array1, dparray array2):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(array2.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MINIMUM, param1_type, param2_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    cdef dparray result = dparray(array1.shape, dtype=result_type)

    cdef custom_math_2in_1out_func_ptr_t func = <custom_math_2in_1out_func_ptr_t > kernel_data.ptr
    func(array1.get_data(), array2.get_data(), result.get_data(), array1.size)

    return result


cdef string multiply_name = "multiply".encode()
func_map[multiply_name][float64_name][float64_name] = [float64_name, & custom_elemwise_multiply_c[double, double, double]]
func_map[multiply_name][float64_name][float32_name] = [float64_name, & custom_elemwise_multiply_c[double, float, double]]
func_map[multiply_name][float64_name][int64_name] = [float64_name, & custom_elemwise_multiply_c[double, long, double]]
func_map[multiply_name][float64_name][int32_name] = [float64_name, & custom_elemwise_multiply_c[double, int, double]]

func_map[multiply_name][float32_name][float64_name] = [float64_name, & custom_elemwise_multiply_c[float, double, double]]
func_map[multiply_name][float32_name][float32_name] = [float32_name, & custom_elemwise_multiply_c[float, float, float]]
func_map[multiply_name][float32_name][int64_name] = [float64_name, & custom_elemwise_multiply_c[float, long, double]]
func_map[multiply_name][float32_name][int32_name] = [float64_name, & custom_elemwise_multiply_c[float, int, double]]

func_map[multiply_name][int64_name][float64_name] = [float64_name, & custom_elemwise_multiply_c[long, double, double]]
func_map[multiply_name][int64_name][float32_name] = [float64_name, & custom_elemwise_multiply_c[long, float, double]]
func_map[multiply_name][int64_name][int64_name] = [int64_name, & custom_elemwise_multiply_c[long, long, long]]
func_map[multiply_name][int64_name][int32_name] = [int64_name, & custom_elemwise_multiply_c[long, int, long]]

func_map[multiply_name][int32_name][float64_name] = [float64_name, & custom_elemwise_multiply_c[int, double, double]]
func_map[multiply_name][int32_name][float32_name] = [float64_name, & custom_elemwise_multiply_c[int, float, double]]
func_map[multiply_name][int32_name][int64_name] = [int64_name, & custom_elemwise_multiply_c[int, long, long]]
func_map[multiply_name][int32_name][int32_name] = [int32_name, & custom_elemwise_multiply_c[int, int, int]]

cpdef dparray dpnp_multiply(dparray array1, dparray array2):
    cdef dparray result
    param1_type = array1.dtype
    param2_type = array2.dtype

    cdef custom_math_2in_1out_t kernel_data = _elementwise_2arg_3type(multiply_name, param1_type.name.encode(), param2_type.name.encode())
    if ( < long > kernel_data.ptr == 0):
        checker_throw_type_error("dpnp_multiply", (param1_type, param2_type))

    result = dparray(array1.shape, dtype=kernel_data.return_type)
    kernel_data.ptr(array1.get_data(), array2.get_data(), result.get_data(), array1.size)

    return result


cpdef dparray dpnp_negative(dparray array1):
    cdef dparray result = dparray(array1.shape, dtype=array1.dtype)

    for i in range(result.size):
        result[i] = -(array1[i])

    return result


cdef string power_name = "power".encode()
func_map[power_name][float64_name][float64_name] = [float64_name, & custom_elemwise_power_c[double, double, double]]
func_map[power_name][float64_name][float32_name] = [float64_name, & custom_elemwise_power_c[double, float, double]]
func_map[power_name][float64_name][int64_name] = [float64_name, & custom_elemwise_power_c[double, long, double]]
func_map[power_name][float64_name][int32_name] = [float64_name, & custom_elemwise_power_c[double, int, double]]

func_map[power_name][float32_name][float64_name] = [float64_name, & custom_elemwise_power_c[float, double, double]]
func_map[power_name][float32_name][float32_name] = [float32_name, & custom_elemwise_power_c[float, float, float]]
func_map[power_name][float32_name][int64_name] = [float64_name, & custom_elemwise_power_c[float, long, double]]
func_map[power_name][float32_name][int32_name] = [float64_name, & custom_elemwise_power_c[float, int, double]]

func_map[power_name][int64_name][float64_name] = [float64_name, & custom_elemwise_power_c[long, double, double]]
func_map[power_name][int64_name][float32_name] = [float64_name, & custom_elemwise_power_c[long, float, double]]
func_map[power_name][int64_name][int64_name] = [int64_name, & custom_elemwise_power_c[long, long, long]]
func_map[power_name][int64_name][int32_name] = [int64_name, & custom_elemwise_power_c[long, int, long]]

func_map[power_name][int32_name][float64_name] = [float64_name, & custom_elemwise_power_c[int, double, double]]
func_map[power_name][int32_name][float32_name] = [float64_name, & custom_elemwise_power_c[int, float, double]]
func_map[power_name][int32_name][int64_name] = [int64_name, & custom_elemwise_power_c[int, long, long]]
func_map[power_name][int32_name][int32_name] = [int32_name, & custom_elemwise_power_c[int, int, int]]

cpdef dparray dpnp_power(dparray array1, dparray array2):
    cdef dparray result
    param1_type = array1.dtype
    param2_type = array2.dtype

    cdef custom_math_2in_1out_t kernel_data = _elementwise_2arg_3type(power_name, param1_type.name.encode(), param2_type.name.encode())
    if ( < long > kernel_data.ptr == 0):
        checker_throw_type_error("dpnp_power", (param1_type, param2_type))

    result = dparray(array1.shape, dtype=kernel_data.return_type)
    kernel_data.ptr(array1.get_data(), array2.get_data(), result.get_data(), array1.size)

    return result


cdef string subtract_name = "subtract".encode()
func_map[subtract_name][float64_name][float64_name] = [float64_name, & custom_elemwise_subtract_c[double, double, double]]
func_map[subtract_name][float64_name][float32_name] = [float64_name, & custom_elemwise_subtract_c[double, float, double]]
func_map[subtract_name][float64_name][int64_name] = [float64_name, & custom_elemwise_subtract_c[double, long, double]]
func_map[subtract_name][float64_name][int32_name] = [float64_name, & custom_elemwise_subtract_c[double, int, double]]

func_map[subtract_name][float32_name][float64_name] = [float64_name, & custom_elemwise_subtract_c[float, double, double]]
func_map[subtract_name][float32_name][float32_name] = [float32_name, & custom_elemwise_subtract_c[float, float, float]]
func_map[subtract_name][float32_name][int64_name] = [float64_name, & custom_elemwise_subtract_c[float, long, double]]
func_map[subtract_name][float32_name][int32_name] = [float64_name, & custom_elemwise_subtract_c[float, int, double]]

func_map[subtract_name][int64_name][float64_name] = [float64_name, & custom_elemwise_subtract_c[long, double, double]]
func_map[subtract_name][int64_name][float32_name] = [float64_name, & custom_elemwise_subtract_c[long, float, double]]
func_map[subtract_name][int64_name][int64_name] = [int64_name, & custom_elemwise_subtract_c[long, long, long]]
func_map[subtract_name][int64_name][int32_name] = [int64_name, & custom_elemwise_subtract_c[long, int, long]]

func_map[subtract_name][int32_name][float64_name] = [float64_name, & custom_elemwise_subtract_c[int, double, double]]
func_map[subtract_name][int32_name][float32_name] = [float64_name, & custom_elemwise_subtract_c[int, float, double]]
func_map[subtract_name][int32_name][int64_name] = [int64_name, & custom_elemwise_subtract_c[int, long, long]]
func_map[subtract_name][int32_name][int32_name] = [int32_name, & custom_elemwise_subtract_c[int, int, int]]

cpdef dparray dpnp_subtract(dparray array1, dparray array2):
    cdef dparray result
    param1_type = array1.dtype
    param2_type = array2.dtype

    cdef custom_math_2in_1out_t kernel_data = _elementwise_2arg_3type(subtract_name, param1_type.name.encode(), param2_type.name.encode())
    if ( < long > kernel_data.ptr == 0):
        checker_throw_type_error("dpnp_subtract", (param1_type, param2_type))

    result = dparray(array1.shape, dtype=kernel_data.return_type)
    kernel_data.ptr(array1.get_data(), array2.get_data(), result.get_data(), array1.size)

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
