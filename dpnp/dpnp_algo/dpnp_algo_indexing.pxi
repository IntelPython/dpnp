# cython: language_level=3
# cython: linetrace=True
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2025, Intel Corporation
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

"""Module Backend (Indexing part)

This module contains interface functions between C backend layer
and the rest of the library

"""

# NO IMPORTs here. All imports must be placed into main "dpnp_algo.pyx" file

__all__ += [
    "dpnp_choose",
    "dpnp_putmask",
]

ctypedef c_dpctl.DPCTLSyclEventRef(*fptr_dpnp_choose_t)(c_dpctl.DPCTLSyclQueueRef,
                                                        void *, void * , void ** , size_t, size_t, size_t,
                                                        const c_dpctl.DPCTLEventVectorRef)

cpdef utils.dpnp_descriptor dpnp_choose(utils.dpnp_descriptor x1, list choices1):
    cdef vector[void * ] choices
    cdef utils.dpnp_descriptor choice
    for desc in choices1:
        choice = desc
        choices.push_back(choice.get_data())

    cdef shape_type_c x1_shape = x1.shape
    cdef size_t choice_size = choices1[0].size

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)

    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(choices1[0].dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_CHOOSE_EXT, param1_type, param2_type)

    x1_obj = x1.get_array()

    cdef utils.dpnp_descriptor res_array = utils.create_output_descriptor(x1_shape,
                                                                          kernel_data.return_type,
                                                                          None,
                                                                          device=x1_obj.sycl_device,
                                                                          usm_type=x1_obj.usm_type,
                                                                          sycl_queue=x1_obj.sycl_queue)

    result_sycl_queue = res_array.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef fptr_dpnp_choose_t func = <fptr_dpnp_choose_t > kernel_data.ptr

    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    res_array.get_data(),
                                                    x1.get_data(),
                                                    choices.data(),
                                                    x1_shape[0],
                                                    choices.size(),
                                                    choice_size,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return res_array


cpdef dpnp_putmask(utils.dpnp_descriptor arr, utils.dpnp_descriptor mask, utils.dpnp_descriptor values):
    cdef int values_size = values.size

    mask_flatiter = mask.get_pyobj().flat
    arr_flatiter = arr.get_pyobj().flat
    values_flatiter = values.get_pyobj().flat

    for i in range(arr.size):
        if mask_flatiter[i]:
            arr_flatiter[i] = values_flatiter[i % values_size]
