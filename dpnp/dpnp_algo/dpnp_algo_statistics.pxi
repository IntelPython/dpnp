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

"""Module Backend (Statistics part)

This module contains interface functions between C backend layer
and the rest of the library

"""

# NO IMPORTs here. All imports must be placed into main "dpnp_algo.pyx" file

__all__ += [
    "dpnp_correlate",
    "dpnp_median",
]


# C function pointer to the C library template functions
ctypedef c_dpctl.DPCTLSyclEventRef(*custom_statistic_1in_1out_func_ptr_t)(c_dpctl.DPCTLSyclQueueRef,
                                                                          void *, void * , shape_elem_type * , size_t,
                                                                          shape_elem_type * , size_t,
                                                                          const c_dpctl.DPCTLEventVectorRef)


cpdef utils.dpnp_descriptor dpnp_correlate(utils.dpnp_descriptor x1, utils.dpnp_descriptor x2):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(x2.dtype)

    cdef shape_type_c x1_shape = x1.shape
    cdef shape_type_c x2_shape = x2.shape

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_CORRELATE_EXT, param1_type, param2_type)

    result_sycl_device, result_usm_type, result_sycl_queue = utils.get_common_usm_allocation(x1, x2)

    # create result array with type given by FPTR data
    cdef shape_type_c result_shape = (1,)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape,
                                                                       kernel_data.return_type,
                                                                       None,
                                                                       device=result_sycl_device,
                                                                       usm_type=result_usm_type,
                                                                       sycl_queue=result_sycl_queue)

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef fptr_2in_1out_t func = <fptr_2in_1out_t > kernel_data.ptr

    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    result.get_data(),
                                                    x1.get_data(),
                                                    x1.size,
                                                    x1_shape.data(),
                                                    x1_shape.size(),
                                                    x2.get_data(),
                                                    x2.size,
                                                    x2_shape.data(),
                                                    x2_shape.size(),
                                                    NULL,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result


cpdef utils.dpnp_descriptor dpnp_median(utils.dpnp_descriptor array1):
    cdef shape_type_c x1_shape = array1.shape
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MEDIAN_EXT, param1_type, param1_type)

    array1_obj = array1.get_array()

    cdef (DPNPFuncType, void *) ret_type_and_func = utils.get_ret_type_and_func(kernel_data,
                                                                                array1_obj.sycl_device.has_aspect_fp64)
    cdef DPNPFuncType return_type = ret_type_and_func[0]
    cdef custom_statistic_1in_1out_func_ptr_t func = < custom_statistic_1in_1out_func_ptr_t > ret_type_and_func[1]

    cdef utils.dpnp_descriptor result = utils.create_output_descriptor((1,),
                                                                       return_type,
                                                                       None,
                                                                       device=array1_obj.sycl_device,
                                                                       usm_type=array1_obj.usm_type,
                                                                       sycl_queue=array1_obj.sycl_queue)

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    # stub for interface support
    cdef shape_type_c axis
    cdef Py_ssize_t axis_size = 0

    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    array1.get_data(),
                                                    result.get_data(),
                                                    x1_shape.data(),
                                                    array1.ndim,
                                                    axis.data(),
                                                    axis_size,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result
