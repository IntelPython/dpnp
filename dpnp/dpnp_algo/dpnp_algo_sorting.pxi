# cython: language_level=3
# cython: linetrace=True
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2024, Intel Corporation
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
    "dpnp_partition",
    "dpnp_searchsorted",
]


ctypedef c_dpctl.DPCTLSyclEventRef(*fptr_dpnp_partition_t)(c_dpctl.DPCTLSyclQueueRef,
                                                           void * ,
                                                           void * ,
                                                           void * ,
                                                           const size_t,
                                                           const shape_elem_type * ,
                                                           const size_t,
                                                           const c_dpctl.DPCTLEventVectorRef)
ctypedef c_dpctl.DPCTLSyclEventRef(*fptr_dpnp_searchsorted_t)(c_dpctl.DPCTLSyclQueueRef,
                                                              void * ,
                                                              const void * ,
                                                              const void * ,
                                                              bool,
                                                              const size_t,
                                                              const size_t,
                                                              const c_dpctl.DPCTLEventVectorRef)


cpdef utils.dpnp_descriptor dpnp_partition(utils.dpnp_descriptor arr, int kth, axis=-1, kind='introselect', order=None):
    cdef shape_type_c shape1 = arr.shape

    cdef size_t kth_ = kth if kth >= 0 else (arr.ndim + kth)
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(arr.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_PARTITION_EXT, param1_type, param1_type)

    cdef utils.dpnp_descriptor arr2 = dpnp_copy(arr)

    arr_obj = arr.get_array()

    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(arr.shape,
                                                                       kernel_data.return_type,
                                                                       None,
                                                                       device=arr_obj.sycl_device,
                                                                       usm_type=arr_obj.usm_type,
                                                                       sycl_queue=arr_obj.sycl_queue)

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef fptr_dpnp_partition_t func = <fptr_dpnp_partition_t > kernel_data.ptr

    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    arr.get_data(),
                                                    arr2.get_data(),
                                                    result.get_data(),
                                                    kth_,
                                                    shape1.data(),
                                                    arr.ndim,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result


cpdef utils.dpnp_descriptor dpnp_searchsorted(utils.dpnp_descriptor arr, utils.dpnp_descriptor v, side='left'):
    if side is 'left':
        side_ = True
    else:
        side_ = False

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(arr.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_SEARCHSORTED_EXT, param1_type, param1_type)

    arr_obj = arr.get_array()

    cdef utils.dpnp_descriptor result = utils_py.create_output_descriptor_py(v.shape,
                                                                             dpnp.int64,
                                                                             None,
                                                                             device=arr_obj.sycl_device,
                                                                             usm_type=arr_obj.usm_type,
                                                                             sycl_queue=arr_obj.sycl_queue)

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef fptr_dpnp_searchsorted_t func = <fptr_dpnp_searchsorted_t > kernel_data.ptr

    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    arr.get_data(),
                                                    v.get_data(),
                                                    result.get_data(),
                                                    side_,
                                                    arr.size,
                                                    v.size,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result
