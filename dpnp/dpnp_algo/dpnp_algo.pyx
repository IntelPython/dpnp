# cython: language_level=3
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

"""Module Backend

This module contains interface functions between C backend layer
and the rest of the library

"""

from libc.time cimport time, time_t
from libcpp.vector cimport vector
import dpnp
import dpnp.config as config
import dpnp.dpnp_container as dpnp_container
import dpnp.dpnp_utils as utils_py
from dpnp.dpnp_array import dpnp_array

import dpctl

cimport cpython
cimport dpnp.dpnp_utils as utils

cimport numpy
import numpy


__all__ = [
    "dpnp_astype",
    "dpnp_flatten",
    "dpnp_init_val",
    "dpnp_queue_initialize",
    "dpnp_queue_is_cpu"
]


include "dpnp_algo_arraycreation.pyx"
include "dpnp_algo_bitwise.pyx"
include "dpnp_algo_counting.pyx"
include "dpnp_algo_indexing.pyx"
include "dpnp_algo_linearalgebra.pyx"
include "dpnp_algo_logic.pyx"
include "dpnp_algo_manipulation.pyx"
include "dpnp_algo_mathematical.pyx"
include "dpnp_algo_searching.pyx"
include "dpnp_algo_sorting.pyx"
include "dpnp_algo_special.pyx"
include "dpnp_algo_statistics.pyx"
include "dpnp_algo_trigonometric.pyx"


ctypedef c_dpctl.DPCTLSyclEventRef(*fptr_dpnp_astype_t)(c_dpctl.DPCTLSyclQueueRef,
                                                        const void *, void * , const size_t,
                                                        const c_dpctl.DPCTLEventVectorRef)
ctypedef c_dpctl.DPCTLSyclEventRef(*fptr_dpnp_flatten_t)(c_dpctl.DPCTLSyclQueueRef,
                                                         void *, const size_t, const size_t,
                                                         const shape_elem_type * , const shape_elem_type * ,
                                                         void *, const size_t, const size_t,
                                                         const shape_elem_type * , const shape_elem_type * ,
                                                         const long * ,
                                                         const c_dpctl.DPCTLEventVectorRef)
ctypedef c_dpctl.DPCTLSyclEventRef(*fptr_dpnp_initval_t)(c_dpctl.DPCTLSyclQueueRef,
                                                         void *, void * , size_t,
                                                         const c_dpctl.DPCTLEventVectorRef)


cpdef utils.dpnp_descriptor dpnp_astype(utils.dpnp_descriptor x1, dtype):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_ASTYPE_EXT, param1_type, param2_type)

    x1_obj = x1.get_array()

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = x1.shape
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape,
                                                                       kernel_data.return_type,
                                                                       None,
                                                                       device=x1_obj.sycl_device,
                                                                       usm_type=x1_obj.usm_type,
                                                                       sycl_queue=x1_obj.sycl_queue)

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef fptr_dpnp_astype_t func = <fptr_dpnp_astype_t > kernel_data.ptr
    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref, x1.get_data(), result.get_data(), x1.size, NULL)

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result


cpdef utils.dpnp_descriptor dpnp_flatten(utils.dpnp_descriptor x1):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_FLATTEN_EXT, param1_type, param1_type)

    cdef shape_type_c x1_shape = x1.shape
    cdef shape_type_c x1_strides = utils.strides_to_vector(x1.strides, x1_shape)

    x1_obj = x1.get_array()

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = (x1.size,)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape,
                                                                       kernel_data.return_type,
                                                                       None,
                                                                       device=x1_obj.sycl_device,
                                                                       usm_type=x1_obj.usm_type,
                                                                       sycl_queue=x1_obj.sycl_queue)

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef shape_type_c result_strides = utils.strides_to_vector(result.strides, result_shape)

    cdef fptr_dpnp_flatten_t func = <fptr_dpnp_flatten_t > kernel_data.ptr
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


cpdef utils.dpnp_descriptor dpnp_init_val(shape, dtype, value):
    """
    same as dpnp_full(). TODO remove code dumplication
    """
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_INITVAL_EXT, param1_type, param1_type)

    cdef utils.dpnp_descriptor result = utils_py.create_output_descriptor_py(shape, dtype, None)

    result_obj = result.get_array()

    # TODO: find better way to pass single value with type conversion
    cdef utils.dpnp_descriptor val_arr = utils_py.create_output_descriptor_py((1, ),
                                                                              dtype,
                                                                              None,
                                                                              device=result_obj.sycl_device,
                                                                              usm_type=result_obj.usm_type,
                                                                              sycl_queue=result_obj.sycl_queue)
    val_arr.get_pyobj()[0] = value

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_obj.sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef fptr_dpnp_initval_t func = <fptr_dpnp_initval_t > kernel_data.ptr
    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref, result.get_data(), val_arr.get_data(), result.size, NULL)

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result


cpdef dpnp_queue_initialize():
    """
    Initialize SYCL queue which will be used for any library operations.
    It takes visible time and needs to be done in the module loading procedure.
    """
    cdef time_t seed_from_time
    cdef QueueOptions queue_type = CPU_SELECTOR

    if (config.__DPNP_QUEUE_GPU__):
        queue_type = GPU_SELECTOR

    dpnp_queue_initialize_c(queue_type)
    dpnp_python_constants_initialize_c(< void*> None,
                                        < void * > dpnp.nan)

    # TODO:
    # choose seed number as is in numpy
    seed_from_time = time(NULL)
    dpnp_rng_srand_c(seed_from_time)


cpdef dpnp_queue_is_cpu():
    """Return 1 if current queue is CPU. Return 0 otherwise.

    """
    return dpnp_queue_is_cpu_c()


"""
Internal functions
"""
cdef DPNPFuncType dpnp_dtype_to_DPNPFuncType(dtype):
    dt_c = numpy.dtype(dtype).char
    kind = numpy.dtype(dtype).kind
    if isinstance(kind, int):
        kind = chr(kind)
    itemsize = numpy.dtype(dtype).itemsize

    if dt_c == 'd':
        return DPNP_FT_DOUBLE
    elif dt_c == 'f':
        return DPNP_FT_FLOAT
    elif kind == 'i':
        if itemsize == 8:
            return DPNP_FT_LONG
        elif itemsize == 4:
            return DPNP_FT_INT
        else:
            utils.checker_throw_type_error("dpnp_dtype_to_DPNPFuncType", dtype)
    elif dt_c == 'F':
        return DPNP_FT_CMPLX64
    elif dt_c == 'D':
        return DPNP_FT_CMPLX128
    elif dt_c == '?':
        return DPNP_FT_BOOL
    else:
        utils.checker_throw_type_error("dpnp_dtype_to_DPNPFuncType", dtype)


cdef dpnp_DPNPFuncType_to_dtype(size_t type):
    """
    Type 'size_t' used instead 'DPNPFuncType' because Cython has lack of Enum support (0.29)
    TODO needs to use DPNPFuncType here
    """
    if type == <size_t > DPNP_FT_DOUBLE:
        return numpy.float64
    elif type == <size_t > DPNP_FT_FLOAT:
        return numpy.float32
    elif type == <size_t > DPNP_FT_LONG:
        return numpy.int64
    elif type == <size_t > DPNP_FT_INT:
        return numpy.int32
    elif type == <size_t > DPNP_FT_CMPLX64:
        return numpy.complex64
    elif type == <size_t > DPNP_FT_CMPLX128:
        return numpy.complex128
    elif type == <size_t > DPNP_FT_BOOL:
        return numpy.bool_
    else:
        utils.checker_throw_type_error("dpnp_DPNPFuncType_to_dtype", type)


cdef utils.dpnp_descriptor call_fptr_1in_1out(DPNPFuncName fptr_name,
                                              utils.dpnp_descriptor x1,
                                              shape_type_c result_shape,
                                              utils.dpnp_descriptor out=None,
                                              func_name=None):

    """ Convert type (x1.dtype) to C enum DPNPFuncType """
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)

    """ get the FPTR data structure """
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(fptr_name, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)

    cdef utils.dpnp_descriptor result

    if out is None:
        """ Create result array with type given by FPTR data """
        x1_obj = x1.get_array()
        result = utils.create_output_descriptor(result_shape,
                                                kernel_data.return_type,
                                                None,
                                                device=x1_obj.sycl_device,
                                                usm_type=x1_obj.usm_type,
                                                sycl_queue=x1_obj.sycl_queue)
    else:
        if out.dtype != result_type:
            utils.checker_throw_value_error(func_name, 'out.dtype', out.dtype, result_type)
        if out.shape != result_shape:
            utils.checker_throw_value_error(func_name, 'out.shape', out.shape, result_shape)

        result = out

        utils.get_common_usm_allocation(x1, result)  # check USM allocation is common

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef fptr_1in_1out_t func = <fptr_1in_1out_t > kernel_data.ptr

    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref, x1.get_data(), result.get_data(), x1.size, NULL)

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result


cdef utils.dpnp_descriptor call_fptr_1in_1out_strides(DPNPFuncName fptr_name,
                                                      utils.dpnp_descriptor x1,
                                                      object dtype=None,
                                                      utils.dpnp_descriptor out=None,
                                                      object where=True,
                                                      func_name=None):

    """ Convert type (x1.dtype) to C enum DPNPFuncType """
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)

    """ get the FPTR data structure """
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(fptr_name, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)

    cdef shape_type_c x1_shape = x1.shape
    cdef shape_type_c x1_strides = utils.strides_to_vector(x1.strides, x1_shape)

    cdef shape_type_c result_shape = x1_shape
    cdef utils.dpnp_descriptor result

    if out is None:
        """ Create result array with type given by FPTR data """
        x1_obj = x1.get_array()
        result = utils.create_output_descriptor(result_shape,
                                                kernel_data.return_type,
                                                None,
                                                device=x1_obj.sycl_device,
                                                usm_type=x1_obj.usm_type,
                                                sycl_queue=x1_obj.sycl_queue)
    else:
        if out.dtype != result_type:
            utils.checker_throw_value_error(func_name, 'out.dtype', out.dtype, result_type)
        if out.shape != result_shape:
            utils.checker_throw_value_error(func_name, 'out.shape', out.shape, result_shape)

        result = out

        utils.get_common_usm_allocation(x1, result)  # check USM allocation is common

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef shape_type_c result_strides = utils.strides_to_vector(result.strides, result_shape)

    """ Call FPTR function """
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


cdef utils.dpnp_descriptor call_fptr_2in_1out(DPNPFuncName fptr_name,
                                              utils.dpnp_descriptor x1_obj,
                                              utils.dpnp_descriptor x2_obj,
                                              object dtype=None,
                                              utils.dpnp_descriptor out=None,
                                              object where=True,
                                              func_name=None):

    # Convert type (x1_obj.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType x1_c_type = dpnp_dtype_to_DPNPFuncType(x1_obj.dtype)
    cdef DPNPFuncType x2_c_type = dpnp_dtype_to_DPNPFuncType(x2_obj.dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(fptr_name, x1_c_type, x2_c_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)

    # Create result array
    cdef shape_type_c x1_shape = x1_obj.shape
    cdef shape_type_c x2_shape = x2_obj.shape
    cdef shape_type_c result_shape = utils.get_common_shape(x1_shape, x2_shape)
    cdef utils.dpnp_descriptor result

    result_sycl_device, result_usm_type, result_sycl_queue = utils.get_common_usm_allocation(x1_obj, x2_obj)

    if out is None:
        """ Create result array with type given by FPTR data """
        result = utils.create_output_descriptor(result_shape,
                                                kernel_data.return_type,
                                                None,
                                                device=result_sycl_device,
                                                usm_type=result_usm_type,
                                                sycl_queue=result_sycl_queue)
    else:
        if out.dtype != result_type:
            utils.checker_throw_value_error(func_name, 'out.dtype', out.dtype, result_type)
        if out.shape != result_shape:
            utils.checker_throw_value_error(func_name, 'out.shape', out.shape, result_shape)

        result = out

        utils.get_common_usm_allocation(x1_obj, result)  # check USM allocation is common

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    """ Call FPTR function """
    cdef fptr_2in_1out_t func = <fptr_2in_1out_t > kernel_data.ptr
    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    result.get_data(),
                                                    x1_obj.get_data(),
                                                    x1_obj.size,
                                                    x1_shape.data(),
                                                    x1_shape.size(),
                                                    x2_obj.get_data(),
                                                    x2_obj.size,
                                                    x2_shape.data(),
                                                    x2_shape.size(),
                                                    NULL,
                                                    NULL)  # dep_events_ref)

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result

cdef utils.dpnp_descriptor call_fptr_2in_1out_strides(DPNPFuncName fptr_name,
                                                      utils.dpnp_descriptor x1_obj,
                                                      utils.dpnp_descriptor x2_obj,
                                                      object dtype=None,
                                                      utils.dpnp_descriptor out=None,
                                                      object where=True,
                                                      func_name=None):

    # Convert type (x1_obj.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType x1_c_type = dpnp_dtype_to_DPNPFuncType(x1_obj.dtype)
    cdef DPNPFuncType x2_c_type = dpnp_dtype_to_DPNPFuncType(x2_obj.dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(fptr_name, x1_c_type, x2_c_type)

    # Create result array
    cdef shape_type_c x1_shape = x1_obj.shape

    cdef shape_type_c x1_strides = utils.strides_to_vector(x1_obj.strides, x1_shape)
    cdef shape_type_c x2_shape = x2_obj.shape
    cdef shape_type_c x2_strides = utils.strides_to_vector(x2_obj.strides, x2_shape)

    cdef shape_type_c result_shape = utils.get_common_shape(x1_shape, x2_shape)
    cdef utils.dpnp_descriptor result

    result_sycl_device, result_usm_type, result_sycl_queue = utils.get_common_usm_allocation(x1_obj, x2_obj)

    # get FPTR function and return type
    cdef fptr_2in_1out_strides_t func = NULL
    cdef DPNPFuncType return_type = DPNP_FT_NONE
    if fptr_name != DPNP_FN_DIVIDE_EXT or result_sycl_device.has_aspect_fp64:
        return_type = kernel_data.return_type
        func = < fptr_2in_1out_strides_t > kernel_data.ptr
    else:
        return_type = kernel_data.return_type_no_fp64
        func = < fptr_2in_1out_strides_t > kernel_data.ptr_no_fp64

    if out is None:
        """ Create result array with type given by FPTR data """
        result = utils.create_output_descriptor(result_shape,
                                                return_type,
                                                None,
                                                device=result_sycl_device,
                                                usm_type=result_usm_type,
                                                sycl_queue=result_sycl_queue)
    else:
        result_type = dpnp_DPNPFuncType_to_dtype(< size_t > return_type)
        if out.dtype != result_type:
            utils.checker_throw_value_error(func_name, 'out.dtype', out.dtype, result_type)
        if out.shape != result_shape:
            utils.checker_throw_value_error(func_name, 'out.shape', out.shape, result_shape)

        result = out

        utils.get_common_usm_allocation(x1_obj, result)  # check USM allocation is common

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
                                                    x1_obj.get_data(),
                                                    x1_obj.size,
                                                    x1_obj.ndim,
                                                    x1_shape.data(),
                                                    x1_strides.data(),
                                                    x2_obj.get_data(),
                                                    x2_obj.size,
                                                    x2_obj.ndim,
                                                    x2_shape.data(),
                                                    x2_strides.data(),
                                                    NULL,
                                                    NULL)  # dep_events_ref)

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result
