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

"""Module Backend (Logic part)

This module contains interface functions between C backend layer
and the rest of the library

"""

# NO IMPORTs here. All imports must be placed into main "dpnp_algo.pyx" file

__all__ += [
    "dpnp_all",
    "dpnp_allclose",
    "dpnp_any",
    "dpnp_equal",
    "dpnp_greater",
    "dpnp_greater_equal",
    "dpnp_isclose",
    "dpnp_isfinite",
    "dpnp_isinf",
    "dpnp_isnan",
    "dpnp_less",
    "dpnp_less_equal",
    "dpnp_logical_and",
    "dpnp_logical_not",
    "dpnp_logical_or",
    "dpnp_logical_xor",
    "dpnp_not_equal"
]


ctypedef void(*custom_logic_1in_1out_func_ptr_t)(void * , void * , const size_t)
ctypedef void(*custom_allclose_1in_1out_func_ptr_t)(void * , void * , void *, const size_t, double, double)


cpdef dparray dpnp_all(dpnp_descriptor array1):
    cdef dparray result = dparray((1,), dtype=numpy.bool)

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_ALL, param1_type, param1_type)

    cdef custom_logic_1in_1out_func_ptr_t func = <custom_logic_1in_1out_func_ptr_t > kernel_data.ptr

    func(array1.get_data(), result.get_data(), array1.size)

    return result


cpdef dparray dpnp_allclose(dpnp_descriptor array1, dpnp_descriptor array2, double rtol, double atol):
    cdef dparray result = dparray((1,), dtype=numpy.bool)

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(array2.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_ALLCLOSE, param1_type, param2_type)

    cdef custom_allclose_1in_1out_func_ptr_t func = <custom_allclose_1in_1out_func_ptr_t > kernel_data.ptr

    func(array1.get_data(), array2.get_data(), result.get_data(), array1.size, rtol, atol)

    return result


cpdef dparray dpnp_any(dpnp_descriptor array1):
    cdef dparray result = dparray((1,), dtype=numpy.bool)

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_ANY, param1_type, param1_type)

    cdef custom_logic_1in_1out_func_ptr_t func = <custom_logic_1in_1out_func_ptr_t > kernel_data.ptr

    func(array1.get_data(), result.get_data(), array1.size)

    return result


cpdef dparray dpnp_equal(object array1, object input2):
    cdef dparray result = dparray(array1.shape, dtype=numpy.bool)

    if isinstance(input2, int):
        for i in range(result.size):
            result[i] = numpy.bool(array1[i] == input2)
    else:
        for i in range(result.size):
            result[i] = numpy.bool(array1[i] == input2[i])

    return result


cpdef dparray dpnp_greater(object input1, object input2):
    input2_is_scalar = dpnp.isscalar(input2)

    cdef dparray result = dparray(input1.shape, dtype=numpy.bool)

    if input2_is_scalar:
        for i in range(result.size):
            result[i] = dpnp.bool(input1[i] > input2)
    else:
        for i in range(result.size):
            result[i] = dpnp.bool(input1[i] > input2[i])

    return result


cpdef dparray dpnp_greater_equal(object input1, object input2):
    input2_is_scalar = dpnp.isscalar(input2)

    cdef dparray result = dparray(input1.shape, dtype=numpy.bool)

    if input2_is_scalar:
        for i in range(result.size):
            result[i] = dpnp.bool(input1[i] >= input2)
    else:
        for i in range(result.size):
            result[i] = dpnp.bool(input1[i] >= input2[i])

    return result


cpdef dparray dpnp_isclose(object input1, object input2, double rtol=1e-05, double atol=1e-08, cpp_bool equal_nan=False):
    cdef dparray result = dparray(input1.shape, dtype=numpy.bool)

    if isinstance(input2, int):
        for i in range(result.size):
            result[i] = numpy.isclose(input1[i], input2, rtol, atol, equal_nan)
    else:
        for i in range(result.size):
            result[i] = numpy.isclose(input1[i], input2[i], rtol, atol, equal_nan)

    return result


cpdef dparray dpnp_isfinite(object input1):
    cdef dparray result = dparray(input1.shape, dtype=numpy.bool)

    for i in range(result.size):
        result[i] = numpy.isfinite(input1[i])

    return result


cpdef dparray dpnp_isinf(object input1):
    cdef dparray result = dparray(input1.shape, dtype=numpy.bool)

    for i in range(result.size):
        result[i] = numpy.isinf(input1[i])

    return result


cpdef dparray dpnp_isnan(object input1):
    cdef dparray result = dparray(input1.shape, dtype=numpy.bool)

    for i in range(result.size):
        result[i] = numpy.isnan(input1[i])

    return result


cpdef dparray dpnp_less(object input1, object input2):
    input2_is_scalar = dpnp.isscalar(input2)

    cdef dparray result = dparray(input1.shape, dtype=numpy.bool)

    if input2_is_scalar:
        for i in range(result.size):
            result[i] = dpnp.bool(input1[i] < input2)
    else:
        for i in range(result.size):
            result[i] = dpnp.bool(input1[i] < input2[i])

    return result


cpdef dparray dpnp_less_equal(object input1, object input2):
    input2_is_scalar = dpnp.isscalar(input2)

    cdef dparray result = dparray(input1.shape, dtype=numpy.bool)

    if input2_is_scalar:
        for i in range(result.size):
            result[i] = dpnp.bool(input1[i] <= input2)
    else:
        for i in range(result.size):
            result[i] = dpnp.bool(input1[i] <= input2[i])

    return result


cpdef dparray dpnp_logical_and(object input1, object input2):
    cdef dparray result = dparray(input1.shape, dtype=numpy.bool)

    for i in range(result.size):
        result[i] = numpy.logical_and(input1[i], input2[i])

    return result


cpdef dparray dpnp_logical_not(object input1):
    cdef dparray result = dparray(input1.shape, dtype=numpy.bool)

    for i in range(result.size):
        result[i] = numpy.logical_not(input1[i])

    return result


cpdef dparray dpnp_logical_or(object input1, object input2):
    cdef dparray result = dparray(input1.shape, dtype=numpy.bool)

    for i in range(result.size):
        result[i] = numpy.logical_or(input1[i], input2[i])

    return result


cpdef dparray dpnp_logical_xor(object input1, object input2):
    cdef dparray result = dparray(input1.shape, dtype=numpy.bool)

    for i in range(result.size):
        result[i] = numpy.logical_xor(input1[i], input2[i])

    return result


cpdef dparray dpnp_not_equal(object input1, object input2):
    input2_is_scalar = dpnp.isscalar(input2)

    cdef dparray result = dparray(input1.shape, dtype=numpy.bool)

    if input2_is_scalar:
        for i in range(result.size):
            result[i] = dpnp.bool(input1[i] != input2)
    else:
        for i in range(result.size):
            result[i] = dpnp.bool(input1[i] != input2[i])

    return result
