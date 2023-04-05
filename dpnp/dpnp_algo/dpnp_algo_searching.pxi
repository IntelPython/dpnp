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

"""Module Backend (Searching part)

This module contains interface functions between C backend layer
and the rest of the library

"""

# NO IMPORTs here. All imports must be placed into main "dpnp_algo.pyx" file

__all__ += [
    "dpnp_argmax",
    "dpnp_argmin",
    "dpnp_where"
]


# C function pointer to the C library template functions
ctypedef c_dpctl.DPCTLSyclEventRef(*custom_search_1in_1out_func_ptr_t)(c_dpctl.DPCTLSyclQueueRef,
                                                                       void * , void * , size_t,
                                                                       const c_dpctl.DPCTLEventVectorRef)

ctypedef c_dpctl.DPCTLSyclEventRef(*where_func_ptr_t)(c_dpctl.DPCTLSyclQueueRef,
                                                      void *,
                                                      const size_t,
                                                      const size_t,
                                                      const shape_elem_type * ,
                                                      const shape_elem_type * ,
                                                      void *,
                                                      const size_t,
                                                      const size_t,
                                                      const shape_elem_type * ,
                                                      const shape_elem_type * ,
                                                      void *,
                                                      const size_t,
                                                      const size_t,
                                                      const shape_elem_type * ,
                                                      const shape_elem_type * ,
                                                      void *,
                                                      const size_t,
                                                      const size_t,
                                                      const shape_elem_type * ,
                                                      const shape_elem_type * ,
                                                      const c_dpctl.DPCTLEventVectorRef) except +


cpdef utils.dpnp_descriptor dpnp_argmax(utils.dpnp_descriptor in_array1):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(in_array1.dtype)
    cdef DPNPFuncType output_type = dpnp_dtype_to_DPNPFuncType(dpnp.int64)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_ARGMAX_EXT, param1_type, output_type)

    in_array1_obj = in_array1.get_array()

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = (1,)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape,
                                                                       kernel_data.return_type,
                                                                       None,
                                                                       device=in_array1_obj.sycl_device,
                                                                       usm_type=in_array1_obj.usm_type,
                                                                       sycl_queue=in_array1_obj.sycl_queue)

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef custom_search_1in_1out_func_ptr_t func = <custom_search_1in_1out_func_ptr_t > kernel_data.ptr

    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    in_array1.get_data(),
                                                    result.get_data(),
                                                    in_array1.size,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result


cpdef utils.dpnp_descriptor dpnp_argmin(utils.dpnp_descriptor in_array1):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(in_array1.dtype)
    cdef DPNPFuncType output_type = dpnp_dtype_to_DPNPFuncType(dpnp.int64)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_ARGMIN_EXT, param1_type, output_type)

    in_array1_obj = in_array1.get_array()

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = (1,)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape,
                                                                       kernel_data.return_type,
                                                                       None,
                                                                       device=in_array1_obj.sycl_device,
                                                                       usm_type=in_array1_obj.usm_type,
                                                                       sycl_queue=in_array1_obj.sycl_queue)

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef custom_search_1in_1out_func_ptr_t func = <custom_search_1in_1out_func_ptr_t > kernel_data.ptr

    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    in_array1.get_data(),
                                                    result.get_data(),
                                                    in_array1.size,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result


cpdef utils.dpnp_descriptor dpnp_where(utils.dpnp_descriptor cond_obj,
                                       utils.dpnp_descriptor x_obj,
                                       utils.dpnp_descriptor y_obj):
    # Convert object type to C enum DPNPFuncType
    cdef DPNPFuncType cond_c_type = dpnp_dtype_to_DPNPFuncType(cond_obj.dtype)
    cdef DPNPFuncType x_c_type = dpnp_dtype_to_DPNPFuncType(x_obj.dtype)
    cdef DPNPFuncType y_c_type = dpnp_dtype_to_DPNPFuncType(y_obj.dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_WHERE_EXT, x_c_type, y_c_type)

    # Create result array
    cdef shape_type_c cond_shape = cond_obj.shape
    cdef shape_type_c x_shape = x_obj.shape
    cdef shape_type_c y_shape = y_obj.shape

    cdef shape_type_c cond_strides = utils.strides_to_vector(cond_obj.strides, cond_shape)
    cdef shape_type_c x_strides = utils.strides_to_vector(x_obj.strides, x_shape)
    cdef shape_type_c y_strides = utils.strides_to_vector(y_obj.strides, y_shape)

    cdef shape_type_c cond_x_shape = utils.get_common_shape(cond_shape, x_shape)
    cdef shape_type_c cond_y_shape = utils.get_common_shape(cond_shape, y_shape)
    cdef shape_type_c result_shape = utils.get_common_shape(cond_x_shape, cond_y_shape)
    cdef utils.dpnp_descriptor result

    result_usm_type, result_sycl_queue = utils_py.get_usm_allocations([cond_obj.get_array(),
                                                                       x_obj.get_array(),
                                                                       y_obj.get_array()])

    # get FPTR function and return type
    cdef where_func_ptr_t func = < where_func_ptr_t > kernel_data.ptr
    cdef DPNPFuncType return_type = kernel_data.return_type

    """ Create result array with type given by FPTR data """
    result = utils.create_output_descriptor(result_shape,
                                            return_type,
                                            None,
                                            device=None,
                                            usm_type=result_usm_type,
                                            sycl_queue=result_sycl_queue)

    cdef shape_type_c result_strides = utils.strides_to_vector(result.strides, result_shape)

    result_obj = result.get_array()

    cdef c_dpctl.SyclQueue q = < c_dpctl.SyclQueue > result_obj.sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    """ Call FPTR function """
    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    result.get_data(),
                                                    result.size,
                                                    result.ndim,
                                                    result_shape.data(),
                                                    result_strides.data(),
                                                    cond_obj.get_data(),
                                                    cond_obj.size,
                                                    cond_obj.ndim,
                                                    cond_shape.data(),
                                                    cond_strides.data(),
                                                    x_obj.get_data(),
                                                    x_obj.size,
                                                    x_obj.ndim,
                                                    x_shape.data(),
                                                    x_strides.data(),
                                                    y_obj.get_data(),
                                                    y_obj.size,
                                                    y_obj.ndim,
                                                    y_shape.data(),
                                                    y_strides.data(),
                                                    NULL)  # dep_events_ref)

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result
