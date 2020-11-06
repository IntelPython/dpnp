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

"""Module Backend

This module contains interface functions between C backend layer
and the rest of the library

"""

import dpnp
from dpnp.dpnp_utils cimport *
from dpnp.backend cimport *
from dpnp.dparray cimport dparray, dparray_shape_type
import numpy
cimport numpy


__all__ = [
    "dpnp_cholesky",
    "dpnp_det",
    "dpnp_eig",
    "dpnp_matrix_rank"
]


# C function pointer to the C library template functions
ctypedef void(*custom_linalg_1in_1out_func_ptr_t)(void *, void * , size_t * , size_t)


# C function pointer to the C library template functions
ctypedef void(*custom_linalg_1in_1out_func_ptr_t_)(void *, void * , size_t *)


cpdef dparray dpnp_cholesky(dparray input):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_CHOLESKY, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef dparray result = dparray(input.size, dtype=result_type)

    cdef custom_linalg_1in_1out_func_ptr_t_ func = <custom_linalg_1in_1out_func_ptr_t_ > kernel_data.ptr

    func(input.get_data(), result.get_data(), < size_t * > input._dparray_shape.data())
    l_result = result.reshape(input.shape)
    return l_result


cpdef dparray dpnp_det(dparray input):
    cdef size_t n = input.shape[-1]
    cdef size_t size_out = 1
    if input.ndim == 2:
        output_shape = (1, )
        size_out = 0
    else:
        output_shape = tuple((list(input.shape))[:-2])
        for i in range(len(output_shape)):
            size_out *= output_shape[i]

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_DET, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef dparray result = dparray(size_out, dtype=result_type)

    cdef custom_linalg_1in_1out_func_ptr_t func = <custom_linalg_1in_1out_func_ptr_t > kernel_data.ptr

    func(input.get_data(), result.get_data(), < size_t * > input._dparray_shape.data(), input.ndim)

    if size_out > 1:
        dpnp_result = result.reshape(output_shape)
        return dpnp_result
    else:
        return result


cpdef tuple dpnp_eig(dparray x1):
    cdef dparray_shape_type x1_shape = x1.shape

    cdef size_t size = 0 if x1_shape.empty() else x1_shape.front()

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_EIG, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)

    cdef dparray res_val = dparray((size,), dtype=result_type)
    cdef dparray res_vec = dparray(x1_shape, dtype=result_type)

    cdef fptr_2in_1out_t func = <fptr_2in_1out_t > kernel_data.ptr
    # call FPTR function
    func(x1.get_data(), res_val.get_data(), res_vec.get_data(), size)

    return (res_val, res_vec)


cpdef dparray dpnp_matrix_rank(dparray input):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MATRIX_RANK, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef dparray result = dparray((1,), dtype=result_type)

    cdef custom_linalg_1in_1out_func_ptr_t func = <custom_linalg_1in_1out_func_ptr_t > kernel_data.ptr

    func(input.get_data(), result.get_data(), < size_t * > input._dparray_shape.data(), input.ndim)

    return result
