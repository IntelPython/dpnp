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

"""Module Backend (Searching part)

This module contains interface functions between C backend layer
and the rest of the library

"""

import cython
import numpy
from dpnp.dpnp_utils cimport checker_throw_type_error, normalize_axis


__all__ += [
    "dpnp_argmax",
    "dpnp_argmin"
]


# C function pointer to the C library template functions
ctypedef void(*custom_search_1in_1out_func_ptr_t)(void * , void * , size_t)


cdef struct custom_math_2in_1out:
    string return_type  # return type identifier which expected by the `ptr` function
    custom_search_1in_1out_func_ptr_t ptr  # C function pointer


cpdef dparray dpnp_argmax(dparray in_array1):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(in_array1.dtype)
    cdef DPNPFuncType output_type = dpnp_dtype_to_DPNPFuncType(numpy.int64)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_ARGMAX, param1_type, output_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    cdef dparray result = dparray((1,), dtype=result_type)

    cdef custom_search_1in_1out_func_ptr_t func = <custom_search_1in_1out_func_ptr_t > kernel_data.ptr

    func(in_array1.get_data(), result.get_data(), in_array1.size)

    return result


cpdef dparray dpnp_argmin(dparray in_array1):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(in_array1.dtype)
    cdef DPNPFuncType output_type = dpnp_dtype_to_DPNPFuncType(numpy.int64)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_ARGMIN, param1_type, output_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    cdef dparray result = dparray((1,), dtype=result_type)

    cdef custom_search_1in_1out_func_ptr_t func = <custom_search_1in_1out_func_ptr_t > kernel_data.ptr

    func(in_array1.get_data(), result.get_data(), in_array1.size)

    return result
