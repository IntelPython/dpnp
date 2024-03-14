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

"""Module Backend

This module contains interface functions between C backend layer
and the rest of the library

"""

import numpy

from dpnp.dpnp_algo cimport *

import dpnp
import dpnp.dpnp_utils as utils_py

cimport numpy

cimport dpnp.dpnp_utils as utils

__all__ = [
    "dpnp_eig",
    "dpnp_eigvals",
]


# C function pointer to the C library template functions
ctypedef c_dpctl.DPCTLSyclEventRef(*custom_linalg_1in_1out_with_size_func_ptr_t_)(c_dpctl.DPCTLSyclQueueRef,
                                                                                  void *, void * , size_t,
                                                                                  const c_dpctl.DPCTLEventVectorRef)
ctypedef c_dpctl.DPCTLSyclEventRef(*custom_linalg_2in_1out_func_ptr_t)(c_dpctl.DPCTLSyclQueueRef,
                                                                       void *, void * , void * , size_t,
                                                                       const c_dpctl.DPCTLEventVectorRef)


cpdef tuple dpnp_eig(utils.dpnp_descriptor x1):
    cdef shape_type_c x1_shape = x1.shape

    cdef size_t size = 0 if x1_shape.empty() else x1_shape.front()

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_EIG_EXT, param1_type, param1_type)

    x1_obj = x1.get_array()

    cdef (DPNPFuncType, void *) ret_type_and_func = utils.get_ret_type_and_func(kernel_data,
                                                                                x1_obj.sycl_device.has_aspect_fp64)
    cdef DPNPFuncType return_type = ret_type_and_func[0]
    cdef custom_linalg_2in_1out_func_ptr_t func = < custom_linalg_2in_1out_func_ptr_t > ret_type_and_func[1]

    cdef utils.dpnp_descriptor res_val = utils.create_output_descriptor((size,),
                                                                        return_type,
                                                                        None,
                                                                        device=x1_obj.sycl_device,
                                                                        usm_type=x1_obj.usm_type,
                                                                        sycl_queue=x1_obj.sycl_queue)
    cdef utils.dpnp_descriptor res_vec = utils.create_output_descriptor(x1_shape,
                                                                        return_type,
                                                                        None,
                                                                        device=x1_obj.sycl_device,
                                                                        usm_type=x1_obj.usm_type,
                                                                        sycl_queue=x1_obj.sycl_queue)

    result_sycl_queue = res_val.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    # call FPTR function
    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    x1.get_data(),
                                                    res_val.get_data(),
                                                    res_vec.get_data(),
                                                    size,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return (res_val.get_pyobj(), res_vec.get_pyobj())


cpdef utils.dpnp_descriptor dpnp_eigvals(utils.dpnp_descriptor input):
    cdef shape_type_c input_shape = input.shape

    cdef size_t size = 0 if input_shape.empty() else input_shape.front()

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_EIGVALS_EXT, param1_type, param1_type)

    input_obj = input.get_array()

    cdef (DPNPFuncType, void *) ret_type_and_func = utils.get_ret_type_and_func(kernel_data,
                                                                                input_obj.sycl_device.has_aspect_fp64)
    cdef DPNPFuncType return_type = ret_type_and_func[0]
    cdef custom_linalg_1in_1out_with_size_func_ptr_t_ func = < custom_linalg_1in_1out_with_size_func_ptr_t_ > ret_type_and_func[1]

    # create result array with type given by FPTR data
    cdef utils.dpnp_descriptor res_val = utils.create_output_descriptor((size,),
                                                                         return_type,
                                                                         None,
                                                                         device=input_obj.sycl_device,
                                                                         usm_type=input_obj.usm_type,
                                                                         sycl_queue=input_obj.sycl_queue)

    result_sycl_queue = res_val.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    # call FPTR function
    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    input.get_data(),
                                                    res_val.get_data(),
                                                    size,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return res_val
