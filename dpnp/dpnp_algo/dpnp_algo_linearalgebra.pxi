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

"""Module Backend (linear algebra routines)

This module contains interface functions between C backend layer
and the rest of the library

"""

# NO IMPORTs here. All imports must be placed into main "dpnp_algo.pyx" file

__all__ += [
    "dpnp_kron",
]


# C function pointer to the C library template functions
ctypedef c_dpctl.DPCTLSyclEventRef(*fptr_2in_1out_shapes_t)(c_dpctl.DPCTLSyclQueueRef,
                                                            void *, void * , void * , shape_elem_type * ,
                                                            shape_elem_type *, shape_elem_type * , size_t,
                                                            const c_dpctl.DPCTLEventVectorRef)


cpdef utils.dpnp_descriptor dpnp_kron(dpnp_descriptor in_array1, dpnp_descriptor in_array2):
    cdef size_t ndim = max(in_array1.ndim, in_array2.ndim)

    cdef shape_type_c in_array1_shape
    if in_array1.ndim < ndim:
        for i in range(ndim - in_array1.ndim):
            in_array1_shape.push_back(1)
    for i in range(in_array1.ndim):
        in_array1_shape.push_back(in_array1.shape[i])

    cdef shape_type_c in_array2_shape
    if in_array2.ndim < ndim:
        for i in range(ndim - in_array2.ndim):
            in_array2_shape.push_back(1)
    for i in range(in_array2.ndim):
        in_array2_shape.push_back(in_array2.shape[i])

    cdef shape_type_c result_shape
    for i in range(ndim):
        result_shape.push_back(in_array1_shape[i] * in_array2_shape[i])

    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(in_array1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(in_array2.dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_KRON_EXT, param1_type, param2_type)

    result_sycl_device, result_usm_type, result_sycl_queue = utils.get_common_usm_allocation(in_array1, in_array2)

    # create result array with type given by FPTR data
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape,
                                                                       kernel_data.return_type,
                                                                       None,
                                                                       device=result_sycl_device,
                                                                       usm_type=result_usm_type,
                                                                       sycl_queue=result_sycl_queue)

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef fptr_2in_1out_shapes_t func = <fptr_2in_1out_shapes_t > kernel_data.ptr
    # call FPTR function
    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    in_array1.get_data(),
                                                    in_array2.get_data(),
                                                    result.get_data(),
                                                    in_array1_shape.data(),
                                                    in_array2_shape.data(),
                                                    result_shape.data(),
                                                    ndim,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result
