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

# NO IMPORTs here. All imports must be placed into main "dpnp_algo.pyx" file

__all__ += [
    "dpnp_argmax",
    "dpnp_argmin"
]


# C function pointer to the C library template functions
ctypedef void(*custom_search_1in_1out_func_ptr_t)(void * , void * , size_t)


cpdef utils.dpnp_descriptor dpnp_argmax(utils.dpnp_descriptor in_array1):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(in_array1.dtype)
    cdef DPNPFuncType output_type = dpnp_dtype_to_DPNPFuncType(dpnp.int64)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_ARGMAX, param1_type, output_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = (1,)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef custom_search_1in_1out_func_ptr_t func = <custom_search_1in_1out_func_ptr_t > kernel_data.ptr

    func(in_array1.get_data(), result.get_data(), in_array1.size)

    return result


cpdef utils.dpnp_descriptor dpnp_argmin(utils.dpnp_descriptor in_array1):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(in_array1.dtype)
    cdef DPNPFuncType output_type = dpnp_dtype_to_DPNPFuncType(dpnp.int64)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_ARGMIN, param1_type, output_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = (1,)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef custom_search_1in_1out_func_ptr_t func = <custom_search_1in_1out_func_ptr_t > kernel_data.ptr

    func(in_array1.get_data(), result.get_data(), in_array1.size)

    return result
