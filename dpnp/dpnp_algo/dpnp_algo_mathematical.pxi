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

"""Module Backend (Mathematical part)

This module contains interface functions between C backend layer
and the rest of the library

"""

# NO IMPORTs here. All imports must be placed into main "dpnp_algo.pyx" file

__all__ += [
    "dpnp_ediff1d",
    "dpnp_fmax",
    "dpnp_fmin",
    "dpnp_modf",
]


ctypedef c_dpctl.DPCTLSyclEventRef(*fptr_1in_2out_t)(c_dpctl.DPCTLSyclQueueRef,
                                                     void * , void * , void * , size_t,
                                                     const c_dpctl.DPCTLEventVectorRef)


cpdef utils.dpnp_descriptor dpnp_ediff1d(utils.dpnp_descriptor x1):

    if x1.size <= 1:
        return utils.dpnp_descriptor(dpnp.empty(0, dtype=x1.dtype))  # TODO need to call dpnp_empty instead

    # Convert type (x1.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_EDIFF1D_EXT, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)

    # Currently shape and strides of the input array are not took into account for the function ediff1d
    cdef shape_type_c x1_shape = (x1.size,)
    cdef shape_type_c x1_strides = utils.strides_to_vector(None, x1_shape)

    x1_obj = x1.get_array()

    cdef shape_type_c result_shape = (x1.size - 1,)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape,
                                                                       kernel_data.return_type,
                                                                       None,
                                                                       device=x1_obj.sycl_device,
                                                                       usm_type=x1_obj.usm_type,
                                                                       sycl_queue=x1_obj.sycl_queue)

    cdef shape_type_c result_strides = utils.strides_to_vector(result.strides, result_shape)

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    # Call FPTR function
    cdef fptr_1in_1out_strides_t func = <fptr_1in_1out_strides_t > kernel_data.ptr
    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    result.get_data(),
                                                    result.size,
                                                    result.ndim,
                                                    result_shape.data(),
                                                    result_strides.data(),
                                                    x1.get_data(),
                                                    x1.size,
                                                    x1.ndim,
                                                    x1_shape.data(),
                                                    x1_strides.data(),
                                                    NULL,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result


cpdef utils.dpnp_descriptor dpnp_fmax(utils.dpnp_descriptor x1_obj,
                                         utils.dpnp_descriptor x2_obj,
                                         object dtype=None,
                                         utils.dpnp_descriptor out=None,
                                         object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_MAXIMUM_EXT, x1_obj, x2_obj, dtype, out, where)


cpdef utils.dpnp_descriptor dpnp_fmin(utils.dpnp_descriptor x1_obj,
                                         utils.dpnp_descriptor x2_obj,
                                         object dtype=None,
                                         utils.dpnp_descriptor out=None,
                                         object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_MINIMUM_EXT, x1_obj, x2_obj, dtype, out, where)


cpdef tuple dpnp_modf(utils.dpnp_descriptor x1):
    """ Convert string type names (array.dtype) to C enum DPNPFuncType """
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)

    """ get the FPTR data structure """
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MODF_EXT, param1_type, DPNP_FT_NONE)

    x1_obj = x1.get_array()

    # create result array with type given by FPTR data
    cdef shape_type_c result_shape = x1.shape
    cdef utils.dpnp_descriptor result1 = utils.create_output_descriptor(result_shape,
                                                                        kernel_data.return_type,
                                                                        None,
                                                                        device=x1_obj.sycl_device,
                                                                        usm_type=x1_obj.usm_type,
                                                                        sycl_queue=x1_obj.sycl_queue)
    cdef utils.dpnp_descriptor result2 = utils.create_output_descriptor(result_shape,
                                                                        kernel_data.return_type,
                                                                        None,
                                                                        device=x1_obj.sycl_device,
                                                                        usm_type=x1_obj.usm_type,
                                                                        sycl_queue=x1_obj.sycl_queue)

    _, _, result_sycl_queue = utils.get_common_usm_allocation(result1, result2)

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef fptr_1in_2out_t func = <fptr_1in_2out_t > kernel_data.ptr
    """ Call FPTR function """
    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    x1.get_data(),
                                                    result1.get_data(),
                                                    result2.get_data(),
                                                    x1.size,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return (result1.get_pyobj(), result2.get_pyobj())
