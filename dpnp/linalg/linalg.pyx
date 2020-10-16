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

from dpnp.dpnp_utils cimport *
from dpnp.backend cimport *
from dpnp.dparray cimport dparray, dparray_shape_type
import numpy
cimport numpy


__all__ = [
    "dpnp_eig",
    "dpnp_matrix_rank"
]


cpdef tuple dpnp_eig(dparray in_array1):
    cdef dparray_shape_type shape1 = in_array1.shape

    cdef size_t size1 = 0
    if not shape1.empty():
        size1 = shape1.front()

    # convert string type names (dparray.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(in_array1.dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_EIG, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    # this array is used as input for MKL and will be overwritten with eigen vectors
    res_vec = in_array1.astype(result_type)
    # ceate result array with type given by FPTR data
    cdef dparray res_val = dparray((size1,), dtype=result_type)

    cdef fptr_1in_1out_t func = <fptr_1in_1out_t > kernel_data.ptr
    # call FPTR function
    func(res_vec.get_data(), res_val.get_data(), size1)

    return res_val, res_vec


cpdef dparray dpnp_matrix_rank(dparray input):
    cdef dparray_shape_type shape_input = input.shape

    cdef size_t elems = 1
    cdef dparray result = dparray((1,), dtype=input.dtype)
    cdef long rank_val = 0
    if input.ndim > 1:
        elems = min(shape_input)
    for i in range(elems):
        ind_ = []
        for j in range(input.ndim):
            ind_.append(i)
        ind = tuple(ind_)
        rank_val += input[ind]

    result[0] = rank_val
    return result
