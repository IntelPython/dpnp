# cython: language_level=3
# cython: linetrace=True
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2023, Intel Corporation
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

"""Module Backend (FFT part)

This module contains interface functions between C backend layer
and the rest of the library

"""


cimport dpnp.dpnp_utils as utils
from dpnp.dpnp_algo cimport *

__all__ = [
    "dpnp_fft",
    "dpnp_rfft"
]

ctypedef c_dpctl.DPCTLSyclEventRef(*fptr_dpnp_fft_fft_t)(c_dpctl.DPCTLSyclQueueRef, void *, void * ,
                                                         shape_elem_type * , shape_elem_type * , size_t, long,
                                                         long, size_t, size_t, const c_dpctl.DPCTLEventVectorRef)


cpdef utils.dpnp_descriptor dpnp_fft(utils.dpnp_descriptor input,
                                     size_t input_boundarie,
                                     size_t output_boundarie,
                                     long axis,
                                     size_t inverse,
                                     size_t norm):

    cdef shape_type_c input_shape = input.shape
    cdef shape_type_c output_shape = input_shape

    cdef long axis_norm = utils.normalize_axis((axis,), input_shape.size())[0]
    output_shape[axis_norm] = output_boundarie

    # convert string type names (dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_FFT_FFT_EXT, param1_type, param1_type)

    input_obj = input.get_array()

    # get FPTR function and return type
    cdef (DPNPFuncType, void *) ret_type_and_func = utils.get_ret_type_and_func(kernel_data,
                                                                                input_obj.sycl_device.has_aspect_fp64)
    cdef DPNPFuncType return_type = ret_type_and_func[0]
    cdef fptr_dpnp_fft_fft_t func = < fptr_dpnp_fft_fft_t > ret_type_and_func[1]

    # ceate result array with type given by FPTR data
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(output_shape,
                                                                       return_type,
                                                                       None,
                                                                       device=input_obj.sycl_device,
                                                                       usm_type=input_obj.usm_type,
                                                                       sycl_queue=input_obj.sycl_queue)

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    # call FPTR function
    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    input.get_data(),
                                                    result.get_data(),
                                                    input_shape.data(),
                                                    output_shape.data(),
                                                    input_shape.size(),
                                                    axis_norm,
                                                    input_boundarie,
                                                    inverse,
                                                    norm,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result


cpdef utils.dpnp_descriptor dpnp_rfft(utils.dpnp_descriptor input,
                                      size_t input_boundarie,
                                      size_t output_boundarie,
                                      long axis,
                                      size_t inverse,
                                      size_t norm):

    cdef shape_type_c input_shape = input.shape
    cdef shape_type_c output_shape = input_shape

    cdef long axis_norm = utils.normalize_axis((axis,), input_shape.size())[0]
    output_shape[axis_norm] = output_boundarie

    # convert string type names (dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_FFT_RFFT_EXT, param1_type, param1_type)

    input_obj = input.get_array()

    # ceate result array with type given by FPTR data
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(output_shape,
                                                                       kernel_data.return_type,
                                                                       None,
                                                                       device=input_obj.sycl_device,
                                                                       usm_type=input_obj.usm_type,
                                                                       sycl_queue=input_obj.sycl_queue)

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef fptr_dpnp_fft_fft_t func = <fptr_dpnp_fft_fft_t > kernel_data.ptr
    # call FPTR function
    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    input.get_data(),
                                                    result.get_data(),
                                                    input_shape.data(),
                                                    output_shape.data(),
                                                    input_shape.size(),
                                                    axis_norm,
                                                    input_boundarie,
                                                    inverse,
                                                    norm,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result
