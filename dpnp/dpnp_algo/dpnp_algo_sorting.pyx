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

"""Module Backend (Sorting part)

This module contains interface functions between C backend layer
and the rest of the library

"""

# NO IMPORTs here. All imports must be placed into main "dpnp_algo.pyx" file

__all__ += [
    "dpnp_argsort",
    "dpnp_partition",
    "dpnp_searchsorted",
    "dpnp_sort"
]


ctypedef void(*fptr_dpnp_partition_t)(void * , void * , void * , const size_t , const size_t * , const size_t)
ctypedef void(*fptr_dpnp_searchsorted_t)(void * , const void * , const void * , bool , const size_t , const size_t )


cpdef utils.dpnp_descriptor dpnp_argsort(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out(DPNP_FN_ARGSORT, x1, x1.shape)


cpdef utils.dpnp_descriptor dpnp_partition(utils.dpnp_descriptor arr, int kth, axis=-1, kind='introselect', order=None):
    cdef shape_type_c shape1 = arr.shape

    cdef size_t kth_ = kth if kth >= 0 else (arr.ndim + kth)
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(arr.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_PARTITION, param1_type, param1_type)

    cdef utils.dpnp_descriptor arr2 = dpnp_copy(arr)

    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(arr.shape, kernel_data.return_type, None)

    cdef fptr_dpnp_partition_t func = <fptr_dpnp_partition_t > kernel_data.ptr

    func(arr.get_data(), arr2.get_data(), result.get_data(), kth_, < size_t * > shape1.data(), arr.ndim)

    return result


cpdef utils.dpnp_descriptor dpnp_searchsorted(utils.dpnp_descriptor arr, utils.dpnp_descriptor v, side='left'):
    if side is 'left':
        side_ = True
    else:
        side_ = False

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(arr.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_SEARCHSORTED, param1_type, param1_type)

    cdef utils.dpnp_descriptor result = utils_py.create_output_descriptor_py(v.shape, dpnp.int64, None)

    cdef fptr_dpnp_searchsorted_t func = <fptr_dpnp_searchsorted_t > kernel_data.ptr

    func(arr.get_data(), v.get_data(), result.get_data(), side_, arr.size, v.size)

    return result


cpdef utils.dpnp_descriptor dpnp_sort(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out(DPNP_FN_SORT, x1, x1.shape)
