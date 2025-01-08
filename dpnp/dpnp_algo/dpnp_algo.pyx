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

"""Module Backend

This module contains interface functions between C backend layer
and the rest of the library

"""

from libc.time cimport time, time_t
from libcpp.vector cimport vector

import dpctl

import dpnp
import dpnp.config as config
import dpnp.dpnp_container as dpnp_container
import dpnp.dpnp_utils as utils_py
from dpnp.dpnp_array import dpnp_array

cimport cpython
cimport numpy

cimport dpnp.dpnp_utils as utils

import operator

import numpy

__all__ = [
]


include "dpnp_algo_indexing.pxi"
include "dpnp_algo_mathematical.pxi"
include "dpnp_algo_sorting.pxi"
include "dpnp_algo_special.pxi"


"""
Internal functions
"""
cdef DPNPFuncType dpnp_dtype_to_DPNPFuncType(dtype):
    dt_c = dpnp.dtype(dtype).char
    kind = dpnp.dtype(dtype).kind
    if isinstance(kind, int):
        kind = chr(kind)
    itemsize = dpnp.dtype(dtype).itemsize

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
        return dpnp.float64
    elif type == <size_t > DPNP_FT_FLOAT:
        return dpnp.float32
    elif type == <size_t > DPNP_FT_LONG:
        return dpnp.int64
    elif type == <size_t > DPNP_FT_INT:
        return dpnp.int32
    elif type == <size_t > DPNP_FT_CMPLX64:
        return dpnp.complex64
    elif type == <size_t > DPNP_FT_CMPLX128:
        return dpnp.complex128
    elif type == <size_t > DPNP_FT_BOOL:
        return dpnp.bool
    else:
        utils.checker_throw_type_error("dpnp_DPNPFuncType_to_dtype", type)


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

    x1_obj = x1.get_array()

    # get FPTR function and return type
    cdef (DPNPFuncType, void *) ret_type_and_func = utils.get_ret_type_and_func(kernel_data,
                                                                                x1_obj.sycl_device.has_aspect_fp64)
    cdef DPNPFuncType return_type = ret_type_and_func[0]
    cdef fptr_1in_1out_strides_t func = < fptr_1in_1out_strides_t > ret_type_and_func[1]

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > return_type)

    cdef shape_type_c x1_shape = x1.shape
    cdef shape_type_c x1_strides = utils.strides_to_vector(x1.strides, x1_shape)

    cdef shape_type_c result_shape = x1_shape
    cdef utils.dpnp_descriptor result

    if out is None:
        """ Create result array with type given by FPTR data """
        result = utils.create_output_descriptor(result_shape,
                                                return_type,
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
