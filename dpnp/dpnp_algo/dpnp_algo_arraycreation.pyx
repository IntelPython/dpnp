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

"""Module Backend (array creation part)

This module contains interface functions between C backend layer
and the rest of the library

"""

# NO IMPORTs here. All imports must be placed into main "dpnp_algo.pyx" file

__all__ += [
    "dpnp_copy",
    "dpnp_diag",
    "dpnp_geomspace",
    "dpnp_identity",
    "dpnp_linspace",
    "dpnp_logspace",
    "dpnp_meshgrid",
    "dpnp_ptp",
    "dpnp_trace",
    "dpnp_tri",
    "dpnp_vander",
]


ctypedef c_dpctl.DPCTLSyclEventRef(*custom_1in_1out_func_ptr_t)(c_dpctl.DPCTLSyclQueueRef,
                                                                void *,
                                                                void * ,
                                                                const int ,
                                                                shape_elem_type * ,
                                                                shape_elem_type * ,
                                                                const size_t,
                                                                const size_t,
                                                                const c_dpctl.DPCTLEventVectorRef)
ctypedef c_dpctl.DPCTLSyclEventRef(*ftpr_custom_vander_1in_1out_t)(c_dpctl.DPCTLSyclQueueRef,
                                                                   void * , void * , size_t, size_t, int,
                                                                   const c_dpctl.DPCTLEventVectorRef) except +
ctypedef c_dpctl.DPCTLSyclEventRef(*custom_arraycreation_1in_1out_func_ptr_t)(c_dpctl.DPCTLSyclQueueRef,
                                                                              void *,
                                                                              const size_t,
                                                                              const size_t,
                                                                              const shape_elem_type*,
                                                                              const shape_elem_type*,
                                                                              void *,
                                                                              const size_t,
                                                                              const size_t,
                                                                              const shape_elem_type*,
                                                                              const shape_elem_type*,
                                                                              const shape_elem_type *,
                                                                              const size_t,
                                                                              const c_dpctl.DPCTLEventVectorRef)
ctypedef c_dpctl.DPCTLSyclEventRef(*custom_indexing_1out_func_ptr_t)(c_dpctl.DPCTLSyclQueueRef,
                                                                     void * ,
                                                                     const size_t ,
                                                                     const size_t ,
                                                                     const int,
                                                                     const c_dpctl.DPCTLEventVectorRef) except +
ctypedef c_dpctl.DPCTLSyclEventRef(*fptr_dpnp_trace_t)(c_dpctl.DPCTLSyclQueueRef,
                                                       const void *,
                                                       void * ,
                                                       const shape_elem_type * ,
                                                       const size_t,
                                                       const c_dpctl.DPCTLEventVectorRef) except +


cpdef utils.dpnp_descriptor dpnp_copy(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_COPY_EXT, x1)


cpdef utils.dpnp_descriptor dpnp_diag(utils.dpnp_descriptor v, int k):
    cdef shape_type_c input_shape = v.shape
    cdef shape_type_c result_shape

    if v.ndim == 1:
        n = v.shape[0] + abs(k)

        result_shape = (n, n)
    else:
        n = min(v.shape[0], v.shape[0] + k, v.shape[1], v.shape[1] - k)
        if n < 0:
            n = 0

        result_shape = (n, )

    v_obj = v.get_array()

    result_obj = dpnp_container.zeros(result_shape, dtype=v.dtype, device=v_obj.sycl_device)
    cdef utils.dpnp_descriptor result = dpnp_descriptor(result_obj)

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(v.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_DIAG_EXT, param1_type, param1_type)

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef custom_1in_1out_func_ptr_t func = <custom_1in_1out_func_ptr_t > kernel_data.ptr

    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    v.get_data(),
                                                    result.get_data(),
                                                    k,
                                                    input_shape.data(),
                                                    result_shape.data(),
                                                    v.ndim,
                                                    result.ndim,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result


cpdef utils.dpnp_descriptor dpnp_geomspace(start, stop, num, endpoint, dtype, axis):
    cdef shape_type_c obj_shape = utils._object_to_tuple(num)
    cdef utils.dpnp_descriptor result = utils_py.create_output_descriptor_py(obj_shape, dtype, None)

    if endpoint:
        steps_count = num - 1
    else:
        steps_count = num

    # if there are steps, then fill values
    if steps_count > 0:
        step = dpnp.power(dpnp.float64(stop) / start, 1.0 / steps_count)
        mult = step
        for i in range(1, result.size):
            result.get_pyobj()[i] = start * mult
            mult = mult * step
    else:
        step = dpnp.nan

    # if result is not empty, then fiil first and last elements
    if num > 0:
        result.get_pyobj()[0] = start
        if endpoint and result.size > 1:
            result.get_pyobj()[result.size - 1] = stop

    return result


cpdef utils.dpnp_descriptor dpnp_identity(n, result_dtype):
    cdef DPNPFuncType dtype_in = dpnp_dtype_to_DPNPFuncType(result_dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_IDENTITY_EXT, dtype_in, DPNP_FT_NONE)

    cdef shape_type_c shape_in = (n, n)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(shape_in, kernel_data.return_type, None)

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef fptr_1out_t func = <fptr_1out_t > kernel_data.ptr
    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref, result.get_data(), n, NULL)

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result


cpdef utils.dpnp_descriptor dpnp_linspace(start, stop, num, dtype=None, device=None, usm_type=None, sycl_queue=None, endpoint=True, retstep=False, axis=0):
    start_isarray = isinstance(start, (dpnp.ndarray, dpctl.tensor.usm_ndarray))
    stop_isarray = isinstance(stop, (dpnp.ndarray, dpctl.tensor.usm_ndarray))

    if sycl_queue is None and device is None:
        sycl_queue = utils_py.get_common_allocation_queue([start, stop])
    sycl_queue_normalized = dpnp.get_normalized_queue_device(sycl_queue=sycl_queue, device=device)

    if usm_type is None:
        if start_isarray and stop_isarray:
            usm_type = dpctl.utils.get_coerced_usm_type([start.usm_type, stop.usm_type])
        elif start_isarray:
            usm_type = start.usm_type
        elif stop_isarray:
            usm_type = stop.usm_type
        else:
            usm_type = "device"

    start_isarray = start_isarray or isinstance(start, numpy.ndarray)
    stop_isarray = stop_isarray or isinstance(stop, numpy.ndarray)

    dt = None
    if start_isarray and stop_isarray:
        dt = numpy.result_type(start.dtype, stop.dtype)
    elif start_isarray:
        dt = start.dtype
    elif stop_isarray:
        dt = stop.dtype

    if dt == None or numpy.issubdtype(dt, dpnp.integer):
        dt = numpy.result_type(float(num), dt)

    dt = utils_py.map_dtype_to_device(dt, sycl_queue_normalized.sycl_device)

    if dtype is None:
        dtype = dt

    if dpnp.isscalar(start) and dpnp.isscalar(stop):
        res = dpnp_container.linspace(start,
                                      stop,
                                      num,
                                      dtype=dt,
                                      device=device,
                                      usm_type=usm_type,
                                      sycl_queue=sycl_queue_normalized,
                                      endpoint=endpoint)
    else:
        #num = operator.index(num)
        if num < 0:
            raise ValueError("Number of points must be non-negative")

        #FIXME: When subtraction with scalar will be implemented
        start_isscalar = dpnp.isscalar(start)
        if start_isscalar:
            start = [start]
        elif start_isarray:
            if start.ndim == 0:
                start = start.reshape((1))
                start_isscalar = True
        stop_isscalar = dpnp.isscalar(stop)
        if stop_isscalar:
            stop = [stop]
        elif stop_isarray:
            if stop.ndim == 0:
                stop = stop.reshape((1))
                stop_isscalar = True

        _start = dpnp.asarray(start, dtype=dt, usm_type=usm_type, sycl_queue=sycl_queue_normalized)
        _stop = dpnp.asarray(stop, dtype=dt, usm_type=usm_type, sycl_queue=sycl_queue_normalized)

        _num = dpnp.asarray([(num - 1) if endpoint else num], dtype=dt,
                            usm_type=usm_type, sycl_queue=sycl_queue_normalized)

        step = (_stop - _start) / _num

        res = dpnp_container.arange(0,
                                    stop=num,
                                    step=1,
                                    dtype=dt,
                                    usm_type=usm_type,
                                    sycl_queue=sycl_queue_normalized)

        res = res.reshape((-1,) + (1,) * step.ndim)
        res = res * step + _start

        if endpoint and num > 1:
            res[-1] = dpnp_container.full(step.shape, _stop)

        if stop_isscalar and start_isscalar:
            res = res.reshape(-1)

    if numpy.issubdtype(dtype, dpnp.integer):
        res = res + dpnp.asarray([0.000000001], dtype=dt, usm_type=usm_type, sycl_queue=sycl_queue_normalized)
        dpnp.floor(res, out=res)
    return dpnp.get_dpnp_descriptor(res.astype(dtype), copy_when_nondefault_queue=False)


cpdef utils.dpnp_descriptor dpnp_logspace(start, stop, num, endpoint, base, dtype, axis):
    temp = dpnp.linspace(start, stop, num=num, endpoint=endpoint)
    return dpnp.get_dpnp_descriptor(dpnp.astype(dpnp.power(base, temp), dtype))


cpdef list dpnp_meshgrid(xi, copy, sparse, indexing):
    input_count = len(xi)

    # simple case
    if input_count == 0:
        return []

    # simple case
    if input_count == 1:
        return [dpnp_copy(dpnp.get_dpnp_descriptor(xi[0])).get_pyobj()]

    shape_mult = 1
    for i in range(input_count):
        shape_mult = shape_mult * xi[i].size

    shape_list = []
    for i in range(input_count):
        shape_list.append(xi[i].size)
    if indexing == "xy":
        temp = shape_list[0]
        shape_list[0] = shape_list[1]
        shape_list[1] = temp

    steps = []
    for i in range(input_count):
        shape_mult = shape_mult // shape_list[i]
        steps.append(shape_mult)
    if indexing == "xy":
        temp = steps[0]
        steps[0] = steps[1]
        steps[1] = temp

    shape = tuple(shape_list)

    cdef utils.dpnp_descriptor res_item
    result = []
    for i in range(input_count):
        res_item = utils_py.create_output_descriptor_py(shape, xi[i].dtype, None)

        for j in range(res_item.size):
            res_item.get_pyobj()[j] = xi[i][(j // steps[i]) % xi[i].size]

        result.append(res_item.get_pyobj())

    return result


cpdef dpnp_ptp(utils.dpnp_descriptor arr, axis=None):
    cdef shape_type_c shape_arr = arr.shape
    cdef shape_type_c output_shape
    if axis is None:
        axis_ = axis
        output_shape = (1,)
    else:
        if isinstance(axis, int):
            if axis < 0:
                axis_ = tuple([arr.ndim - axis])
            else:
                axis_ = tuple([axis])
        else:
            _axis_ = []
            for i in range(len(axis)):
                if axis[i] < 0:
                    _axis_.append(arr.ndim - axis[i])
                else:
                    _axis_.append(axis[i])
            axis_ = tuple(_axis_)

        out_shape = []
        ind = 0
        for id, shape_axis in enumerate(shape_arr):
            if id not in axis_:
                out_shape.append(shape_axis)
        output_shape = tuple(out_shape)

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(arr.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_PTP_EXT, param1_type, param1_type)

    arr_obj = arr.get_array()

    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(output_shape,
                                                                       kernel_data.return_type,
                                                                       None,
                                                                       device=arr_obj.sycl_device,
                                                                       usm_type=arr_obj.usm_type,
                                                                       sycl_queue=arr_obj.sycl_queue)

    cdef shape_type_c axis1
    cdef Py_ssize_t axis_size = 0
    cdef shape_type_c axis2 = axis1
    if axis_ is not None:
        axis1 = axis_
        axis2.reserve(len(axis1))
        for shape_it in axis1:
            if shape_it < 0:
                raise ValueError("DPNP dparray::__init__(): Negative values in 'shape' are not allowed")
            axis2.push_back(shape_it)
        axis_size = len(axis1)

    cdef shape_type_c result_strides = utils.strides_to_vector(result.strides, result.shape)
    cdef shape_type_c arr_strides = utils.strides_to_vector(arr.strides, arr.shape)

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef custom_arraycreation_1in_1out_func_ptr_t func = <custom_arraycreation_1in_1out_func_ptr_t > kernel_data.ptr
    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    result.get_data(),
                                                    result.size,
                                                    result.ndim,
                                                    output_shape.data(),
                                                    result_strides.data(),
                                                    arr.get_data(),
                                                    arr.size,
                                                    arr.ndim,
                                                    shape_arr.data(),
                                                    arr_strides.data(),
                                                    axis2.data(),
                                                    axis_size,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result


cpdef utils.dpnp_descriptor dpnp_trace(utils.dpnp_descriptor arr, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    if dtype is None:
        dtype_ = arr.dtype
    else:
        dtype_ = dtype

    cdef utils.dpnp_descriptor diagonal_arr = dpnp_diagonal(arr, offset)
    cdef size_t diagonal_ndim = diagonal_arr.ndim
    cdef shape_type_c diagonal_shape = diagonal_arr.shape

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(arr.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(dtype_)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_TRACE_EXT, param1_type, param2_type)

    arr_obj = arr.get_array()

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = diagonal_shape[:-1]
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape,
                                                                       kernel_data.return_type,
                                                                       None,
                                                                       device=arr_obj.sycl_device,
                                                                       usm_type=arr_obj.usm_type,
                                                                       sycl_queue=arr_obj.sycl_queue)

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef fptr_dpnp_trace_t func = <fptr_dpnp_trace_t > kernel_data.ptr

    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    diagonal_arr.get_data(),
                                                    result.get_data(),
                                                    diagonal_shape.data(),
                                                    diagonal_ndim,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result


cpdef utils.dpnp_descriptor dpnp_tri(N, M=None, k=0, dtype=dpnp.float):
    if M is None:
        M = N

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_TRI_EXT, param1_type, param1_type)

    cdef shape_type_c shape_in = (N, M)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(shape_in, kernel_data.return_type, None)

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef custom_indexing_1out_func_ptr_t func = <custom_indexing_1out_func_ptr_t > kernel_data.ptr

    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref, result.get_data(), N, M, k, NULL)

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result


cpdef utils.dpnp_descriptor dpnp_vander(utils.dpnp_descriptor x1, int N, int increasing):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_VANDER_EXT, param1_type, DPNP_FT_NONE)

    x1_obj = x1.get_array()

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = (x1.size, N)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape,
                                                                       kernel_data.return_type,
                                                                       None,
                                                                       device=x1_obj.sycl_device,
                                                                       usm_type=x1_obj.usm_type,
                                                                       sycl_queue=x1_obj.sycl_queue)

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef ftpr_custom_vander_1in_1out_t func = <ftpr_custom_vander_1in_1out_t > kernel_data.ptr
    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    x1.get_data(),
                                                    result.get_data(),
                                                    x1.size,
                                                    N,
                                                    increasing,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result
