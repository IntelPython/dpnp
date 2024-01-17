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

"""Module Backend (array creation part)

This module contains interface functions between C backend layer
and the rest of the library

"""

# NO IMPORTs here. All imports must be placed into main "dpnp_algo.pyx" file

__all__ += [
    "dpnp_copy",
    "dpnp_trace",
]


ctypedef c_dpctl.DPCTLSyclEventRef(*custom_1in_1out_func_ptr_t)(c_dpctl.DPCTLSyclQueueRef,
                                                                void *,
                                                                void * ,
                                                                const int ,
                                                                shape_elem_type * ,
                                                                shape_elem_type * ,
                                                                const size_t,
                                                                const size_t,
                                                                const c_dpctl.DPCTLEventVectorRef)
ctypedef c_dpctl.DPCTLSyclEventRef(*ftpr_custom_vander_1in_1out_t)(c_dpctl.DPCTLSyclQueueRef,
                                                                   void * , void * , size_t, size_t, int,
                                                                   const c_dpctl.DPCTLEventVectorRef) except +
ctypedef c_dpctl.DPCTLSyclEventRef(*custom_arraycreation_1in_1out_func_ptr_t)(c_dpctl.DPCTLSyclQueueRef,
                                                                              void *,
                                                                              const size_t,
                                                                              const size_t,
                                                                              const shape_elem_type*,
                                                                              const shape_elem_type*,
                                                                              void *,
                                                                              const size_t,
                                                                              const size_t,
                                                                              const shape_elem_type*,
                                                                              const shape_elem_type*,
                                                                              const shape_elem_type *,
                                                                              const size_t,
                                                                              const c_dpctl.DPCTLEventVectorRef)
ctypedef c_dpctl.DPCTLSyclEventRef(*custom_indexing_1out_func_ptr_t)(c_dpctl.DPCTLSyclQueueRef,
                                                                     void * ,
                                                                     const size_t ,
                                                                     const size_t ,
                                                                     const int,
                                                                     const c_dpctl.DPCTLEventVectorRef) except +
ctypedef c_dpctl.DPCTLSyclEventRef(*fptr_dpnp_trace_t)(c_dpctl.DPCTLSyclQueueRef,
                                                       const void *,
                                                       void * ,
                                                       const shape_elem_type * ,
                                                       const size_t,
                                                       const c_dpctl.DPCTLEventVectorRef) except +


cpdef utils.dpnp_descriptor dpnp_copy(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_COPY_EXT, x1)


cpdef utils.dpnp_descriptor dpnp_trace(utils.dpnp_descriptor arr, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    if dtype is None:
        dtype_ = arr.dtype
    else:
        dtype_ = dtype

    cdef utils.dpnp_descriptor diagonal_arr = dpnp_diagonal(arr, offset)
    cdef size_t diagonal_ndim = diagonal_arr.ndim
    cdef shape_type_c diagonal_shape = diagonal_arr.shape

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(arr.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(dtype_)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_TRACE_EXT, param1_type, param2_type)

    arr_obj = arr.get_array()

    # create result array with type given by FPTR data
    cdef shape_type_c result_shape = diagonal_shape[:-1]
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape,
                                                                       kernel_data.return_type,
                                                                       None,
                                                                       device=arr_obj.sycl_device,
                                                                       usm_type=arr_obj.usm_type,
                                                                       sycl_queue=arr_obj.sycl_queue)

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef fptr_dpnp_trace_t func = <fptr_dpnp_trace_t > kernel_data.ptr

    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    diagonal_arr.get_data(),
                                                    result.get_data(),
                                                    diagonal_shape.data(),
                                                    diagonal_ndim,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result
