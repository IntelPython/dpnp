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

"""Module Backend (Logic part)

This module contains interface functions between C backend layer
and the rest of the library

"""

# NO IMPORTs here. All imports must be placed into main "dpnp_algo.pyx" file

__all__ += [
    "dpnp_all",
    "dpnp_allclose",
    "dpnp_any",
    "dpnp_equal",
    "dpnp_greater",
    "dpnp_greater_equal",
    "dpnp_isclose",
    "dpnp_isfinite",
    "dpnp_isinf",
    "dpnp_isnan",
    "dpnp_less",
    "dpnp_less_equal",
    "dpnp_logical_and",
    "dpnp_logical_not",
    "dpnp_logical_or",
    "dpnp_logical_xor",
    "dpnp_not_equal"
]


ctypedef c_dpctl.DPCTLSyclEventRef(*custom_logic_1in_1out_func_ptr_t)(c_dpctl.DPCTLSyclQueueRef,
                                                                      void *, void * , const size_t,
                                                                      const c_dpctl.DPCTLEventVectorRef)
ctypedef c_dpctl.DPCTLSyclEventRef(*custom_allclose_1in_1out_func_ptr_t)(c_dpctl.DPCTLSyclQueueRef,
                                                                         void * ,
                                                                         void * ,
                                                                         void * ,
                                                                         const size_t,
                                                                         double,
                                                                         double,
                                                                         const c_dpctl.DPCTLEventVectorRef)


cpdef utils.dpnp_descriptor dpnp_all(utils.dpnp_descriptor array1):
    array1_obj = array1.get_array()

    cdef utils.dpnp_descriptor result = utils_py.create_output_descriptor_py((1,),
                                                                             dpnp.bool,
                                                                             None,
                                                                             device=array1_obj.sycl_device,
                                                                             usm_type=array1_obj.usm_type,
                                                                             sycl_queue=array1_obj.sycl_queue)

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_ALL_EXT, param1_type, param1_type)

    cdef custom_logic_1in_1out_func_ptr_t func = <custom_logic_1in_1out_func_ptr_t > kernel_data.ptr

    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref, array1.get_data(), result.get_data(), array1.size, NULL)

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result


cpdef utils.dpnp_descriptor dpnp_allclose(utils.dpnp_descriptor array1,
                                          utils.dpnp_descriptor array2,
                                          double rtol_val,
                                          double atol_val):
    result_sycl_device, result_usm_type, result_sycl_queue = utils.get_common_usm_allocation(array1, array2)

    cdef utils.dpnp_descriptor result = utils_py.create_output_descriptor_py((1,),
                                                                             dpnp.bool,
                                                                             None,
                                                                             device=result_sycl_device,
                                                                             usm_type=result_usm_type,
                                                                             sycl_queue=result_sycl_queue)

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(array2.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_ALLCLOSE_EXT, param1_type, param2_type)

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef custom_allclose_1in_1out_func_ptr_t func = <custom_allclose_1in_1out_func_ptr_t > kernel_data.ptr

    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    array1.get_data(),
                                                    array2.get_data(),
                                                    result.get_data(),
                                                    array1.size,
                                                    rtol_val,
                                                    atol_val,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result


cpdef utils.dpnp_descriptor dpnp_any(utils.dpnp_descriptor array1):
    array1_obj = array1.get_array()

    cdef utils.dpnp_descriptor result = utils_py.create_output_descriptor_py((1,),
                                                                             dpnp.bool,
                                                                             None,
                                                                             device=array1_obj.sycl_device,
                                                                             usm_type=array1_obj.usm_type,
                                                                             sycl_queue=array1_obj.sycl_queue)

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_ANY_EXT, param1_type, param1_type)

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef custom_logic_1in_1out_func_ptr_t func = <custom_logic_1in_1out_func_ptr_t > kernel_data.ptr

    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref, array1.get_data(), result.get_data(), array1.size, NULL)

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result


cpdef utils.dpnp_descriptor dpnp_equal(utils.dpnp_descriptor x1_obj,
                                       utils.dpnp_descriptor x2_obj,
                                       object dtype=None,
                                       utils.dpnp_descriptor out=None,
                                       object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_EQUAL_EXT, x1_obj, x2_obj, dtype, out, where, func_name="equal")


cpdef utils.dpnp_descriptor dpnp_greater(utils.dpnp_descriptor x1_obj,
                                         utils.dpnp_descriptor x2_obj,
                                         object dtype=None,
                                         utils.dpnp_descriptor out=None,
                                         object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_GREATER_EXT, x1_obj, x2_obj, dtype, out, where, func_name="greater")


cpdef utils.dpnp_descriptor dpnp_greater_equal(utils.dpnp_descriptor x1_obj,
                                               utils.dpnp_descriptor x2_obj,
                                               object dtype=None,
                                               utils.dpnp_descriptor out=None,
                                               object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_GREATER_EQUAL_EXT, x1_obj, x2_obj, dtype, out, where, func_name="greater_equal")


cpdef utils.dpnp_descriptor dpnp_isclose(utils.dpnp_descriptor input1,
                                         utils.dpnp_descriptor input2,
                                         double rtol=1e-05,
                                         double atol=1e-08,
                                         cpp_bool equal_nan=False):
    result_sycl_device, result_usm_type, result_sycl_queue = utils.get_common_usm_allocation(input1, input2)
    cdef utils.dpnp_descriptor result = utils_py.create_output_descriptor_py(input1.shape,
                                                                             dpnp.bool,
                                                                             None,
                                                                             device=result_sycl_device,
                                                                             usm_type=result_usm_type,
                                                                             sycl_queue=result_sycl_queue)

    for i in range(result.size):
        result.get_pyobj()[i] = numpy.isclose(input1.get_pyobj()[i], input2.get_pyobj()[i], rtol, atol, equal_nan)

    return result


cpdef utils.dpnp_descriptor dpnp_isfinite(utils.dpnp_descriptor input1):
    input1_obj = input1.get_array()
    cdef utils.dpnp_descriptor result = utils_py.create_output_descriptor_py(input1.shape,
                                                                             dpnp.bool,
                                                                             None,
                                                                             device=input1_obj.sycl_device,
                                                                             usm_type=input1_obj.usm_type,
                                                                             sycl_queue=input1_obj.sycl_queue)

    for i in range(result.size):
        result.get_pyobj()[i] = numpy.isfinite(input1.get_pyobj()[i])

    return result


cpdef utils.dpnp_descriptor dpnp_isinf(utils.dpnp_descriptor input1):
    input1_obj = input1.get_array()
    cdef utils.dpnp_descriptor result = utils_py.create_output_descriptor_py(input1.shape,
                                                                             dpnp.bool,
                                                                             None,
                                                                             device=input1_obj.sycl_device,
                                                                             usm_type=input1_obj.usm_type,
                                                                             sycl_queue=input1_obj.sycl_queue)

    for i in range(result.size):
        result.get_pyobj()[i] = numpy.isinf(input1.get_pyobj()[i])

    return result


cpdef utils.dpnp_descriptor dpnp_isnan(utils.dpnp_descriptor input1):
    input1_obj = input1.get_array()
    cdef utils.dpnp_descriptor result = utils_py.create_output_descriptor_py(input1.shape,
                                                                             dpnp.bool,
                                                                             None,
                                                                             device=input1_obj.sycl_device,
                                                                             usm_type=input1_obj.usm_type,
                                                                             sycl_queue=input1_obj.sycl_queue)

    for i in range(result.size):
        result.get_pyobj()[i] = numpy.isnan(input1.get_pyobj()[i])

    return result


cpdef utils.dpnp_descriptor dpnp_less(utils.dpnp_descriptor x1_obj,
                                      utils.dpnp_descriptor x2_obj,
                                      object dtype=None,
                                      utils.dpnp_descriptor out=None,
                                      object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_LESS_EXT, x1_obj, x2_obj, dtype, out, where, func_name="less")


cpdef utils.dpnp_descriptor dpnp_less_equal(utils.dpnp_descriptor x1_obj,
                                            utils.dpnp_descriptor x2_obj,
                                            object dtype=None,
                                            utils.dpnp_descriptor out=None,
                                            object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_LESS_EQUAL_EXT, x1_obj, x2_obj, dtype, out, where, func_name="less_equal")


cpdef utils.dpnp_descriptor dpnp_logical_and(utils.dpnp_descriptor x1_obj,
                                             utils.dpnp_descriptor x2_obj,
                                             object dtype=None,
                                             utils.dpnp_descriptor out=None,
                                             object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_LOGICAL_AND_EXT, x1_obj, x2_obj, dtype, out, where, func_name="logical_and")


cpdef utils.dpnp_descriptor dpnp_logical_not(utils.dpnp_descriptor x_obj,
                                            object dtype=None,
                                            utils.dpnp_descriptor out=None,
                                            object where=True):
    return call_fptr_1in_1out_strides(DPNP_FN_LOGICAL_NOT_EXT, x_obj, dtype, out, where, func_name="logical_not")


cpdef utils.dpnp_descriptor dpnp_logical_or(utils.dpnp_descriptor x1_obj,
                                            utils.dpnp_descriptor x2_obj,
                                            object dtype=None,
                                            utils.dpnp_descriptor out=None,
                                            object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_LOGICAL_OR_EXT, x1_obj, x2_obj, dtype, out, where, func_name="logical_or")


cpdef utils.dpnp_descriptor dpnp_logical_xor(utils.dpnp_descriptor x1_obj,
                                             utils.dpnp_descriptor x2_obj,
                                             object dtype=None,
                                             utils.dpnp_descriptor out=None,
                                             object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_LOGICAL_XOR_EXT, x1_obj, x2_obj, dtype, out, where, func_name="logical_xor")


cpdef utils.dpnp_descriptor dpnp_not_equal(utils.dpnp_descriptor x1_obj,
                                           utils.dpnp_descriptor x2_obj,
                                           object dtype=None,
                                           utils.dpnp_descriptor out=None,
                                           object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_NOT_EQUAL_EXT, x1_obj, x2_obj, dtype, out, where, func_name="not_equal")
