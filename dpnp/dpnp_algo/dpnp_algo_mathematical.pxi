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

"""Module Backend (Mathematical part)

This module contains interface functions between C backend layer
and the rest of the library

"""

# NO IMPORTs here. All imports must be placed into main "dpnp_algo.pyx" file

__all__ += [
    "dpnp_absolute",
    "dpnp_arctan2",
    "dpnp_around",
    "dpnp_ceil",
    "dpnp_conjugate",
    "dpnp_copysign",
    "dpnp_cross",
    "dpnp_cumprod",
    "dpnp_cumsum",
    "dpnp_diff",
    "dpnp_ediff1d",
    "dpnp_fabs",
    "dpnp_floor",
    "dpnp_floor_divide",
    "dpnp_fmod",
    "dpnp_gradient",
    'dpnp_hypot',
    "dpnp_maximum",
    "dpnp_minimum",
    "dpnp_modf",
    "dpnp_nancumprod",
    "dpnp_nancumsum",
    "dpnp_nanprod",
    "dpnp_nansum",
    "dpnp_negative",
    "dpnp_prod",
    "dpnp_remainder",
    "dpnp_sign",
    "dpnp_sum",
    "dpnp_trapz",
    "dpnp_trunc"
]


ctypedef c_dpctl.DPCTLSyclEventRef(*fptr_custom_elemwise_absolute_1in_1out_t)(c_dpctl.DPCTLSyclQueueRef,
                                                                              void * , void * , size_t,
                                                                              const c_dpctl.DPCTLEventVectorRef)
ctypedef c_dpctl.DPCTLSyclEventRef(*fptr_1in_2out_t)(c_dpctl.DPCTLSyclQueueRef,
                                                     void * , void * , void * , size_t,
                                                     const c_dpctl.DPCTLEventVectorRef)
ctypedef c_dpctl.DPCTLSyclEventRef(*ftpr_custom_trapz_2in_1out_with_2size_t)(c_dpctl.DPCTLSyclQueueRef,
                                                                             void *, void * , void * , double, size_t, size_t,
                                                                             const c_dpctl.DPCTLEventVectorRef)
ctypedef c_dpctl.DPCTLSyclEventRef(*ftpr_custom_around_1in_1out_t)(c_dpctl.DPCTLSyclQueueRef,
                                                                   const void * , void * , const size_t, const int,
                                                                   const c_dpctl.DPCTLEventVectorRef)


cpdef utils.dpnp_descriptor dpnp_absolute(utils.dpnp_descriptor x1):
    cdef shape_type_c x1_shape = x1.shape
    cdef size_t x1_shape_size = x1.ndim

    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_ABSOLUTE_EXT, param1_type, param1_type)

    x1_obj = x1.get_array()

    # ceate result array with type given by FPTR data
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(x1_shape,
                                                                       kernel_data.return_type,
                                                                       None,
                                                                       device=x1_obj.sycl_device,
                                                                       usm_type=x1_obj.usm_type,
                                                                       sycl_queue=x1_obj.sycl_queue)

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef fptr_custom_elemwise_absolute_1in_1out_t func = <fptr_custom_elemwise_absolute_1in_1out_t > kernel_data.ptr
    # call FPTR function
    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref, x1.get_data(), result.get_data(), x1.size, NULL)

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result


cpdef utils.dpnp_descriptor dpnp_arctan2(utils.dpnp_descriptor x1_obj,
                                         utils.dpnp_descriptor x2_obj,
                                         object dtype=None,
                                         utils.dpnp_descriptor out=None,
                                         object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_ARCTAN2_EXT, x1_obj, x2_obj, dtype, out, where, func_name="arctan2")


cpdef utils.dpnp_descriptor dpnp_around(utils.dpnp_descriptor x1, int decimals):

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_AROUND_EXT, param1_type, param1_type)

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

    cdef ftpr_custom_around_1in_1out_t func = <ftpr_custom_around_1in_1out_t > kernel_data.ptr

    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref, x1.get_data(), result.get_data(), x1.size, decimals, NULL)

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result


cpdef utils.dpnp_descriptor dpnp_ceil(utils.dpnp_descriptor x1, utils.dpnp_descriptor out):
    return call_fptr_1in_1out_strides(DPNP_FN_CEIL_EXT, x1, dtype=None, out=out, where=True, func_name='ceil')


cpdef utils.dpnp_descriptor dpnp_conjugate(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_CONJIGUATE_EXT, x1)


cpdef utils.dpnp_descriptor dpnp_copysign(utils.dpnp_descriptor x1_obj,
                                          utils.dpnp_descriptor x2_obj,
                                          object dtype=None,
                                          utils.dpnp_descriptor out=None,
                                          object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_COPYSIGN_EXT, x1_obj, x2_obj, dtype, out, where)


cpdef utils.dpnp_descriptor dpnp_cross(utils.dpnp_descriptor x1_obj,
                                       utils.dpnp_descriptor x2_obj,
                                       object dtype=None,
                                       utils.dpnp_descriptor out=None,
                                       object where=True):
    return call_fptr_2in_1out(DPNP_FN_CROSS_EXT, x1_obj, x2_obj, dtype, out, where)


cpdef utils.dpnp_descriptor dpnp_cumprod(utils.dpnp_descriptor x1):
    # instead of x1.shape, (x1.size, ) is passed to the function
    # due to the following:
    # >>> import numpy
    # >>> a = numpy.array([[1, 2], [2, 3]])
    # >>> res = numpy.cumprod(a)
    # >>> res.shape
    # (4,)

    return call_fptr_1in_1out(DPNP_FN_CUMPROD_EXT, x1, (x1.size,))


cpdef utils.dpnp_descriptor dpnp_cumsum(utils.dpnp_descriptor x1):
    # instead of x1.shape, (x1.size, ) is passed to the function
    # due to the following:
    # >>> import numpy
    # >>> a = numpy.array([[1, 2], [2, 3]])
    # >>> res = numpy.cumsum(a)
    # >>> res.shape
    # (4,)

    return call_fptr_1in_1out(DPNP_FN_CUMSUM_EXT, x1, (x1.size,))


cpdef utils.dpnp_descriptor dpnp_diff(utils.dpnp_descriptor x1, int n):
    cdef utils.dpnp_descriptor res

    x1_obj = x1.get_array()

    if x1.size - n < 1:
        res_obj = dpnp_container.empty(0,
                                       dtype=x1.dtype,
                                       device=x1_obj.sycl_device,
                                       usm_type=x1_obj.usm_type,
                                       sycl_queue=x1_obj.sycl_queue)
        res = utils.dpnp_descriptor(res_obj)
        return res

    res_obj = dpnp_container.empty(x1.size - 1,
                                   dtype=x1.dtype,
                                   device=x1_obj.sycl_device,
                                   usm_type=x1_obj.usm_type,
                                   sycl_queue=x1_obj.sycl_queue)
    res = utils.dpnp_descriptor(res_obj)
    for i in range(res.size):
        res.get_pyobj()[i] = x1.get_pyobj()[i + 1] - x1.get_pyobj()[i]

    if n == 1:
        return res

    return dpnp_diff(res, n - 1)


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


cpdef utils.dpnp_descriptor dpnp_fabs(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_FABS_EXT, x1)


cpdef utils.dpnp_descriptor dpnp_floor(utils.dpnp_descriptor x1, utils.dpnp_descriptor out):
    return call_fptr_1in_1out_strides(DPNP_FN_FLOOR_EXT, x1, dtype=None, out=out, where=True, func_name='floor')


cpdef utils.dpnp_descriptor dpnp_floor_divide(utils.dpnp_descriptor x1_obj,
                                              utils.dpnp_descriptor x2_obj,
                                              object dtype=None,
                                              utils.dpnp_descriptor out=None,
                                              object where=True):
    return call_fptr_2in_1out(DPNP_FN_FLOOR_DIVIDE_EXT, x1_obj, x2_obj, dtype, out, where)


cpdef utils.dpnp_descriptor dpnp_fmod(utils.dpnp_descriptor x1_obj,
                                      utils.dpnp_descriptor x2_obj,
                                      object dtype=None,
                                      utils.dpnp_descriptor out=None,
                                      object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_FMOD_EXT, x1_obj, x2_obj, dtype, out, where)


cpdef utils.dpnp_descriptor dpnp_gradient(utils.dpnp_descriptor y1, int dx=1):

    cdef size_t size = y1.size

    y1_obj = y1.get_array()

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils_py.create_output_descriptor_py(result_shape,
                                                                             dpnp.float64,
                                                                             None,
                                                                             device=y1_obj.sycl_device,
                                                                             usm_type=y1_obj.usm_type,
                                                                             sycl_queue=y1_obj.sycl_queue)

    cdef double cur = (y1.get_pyobj()[1] - y1.get_pyobj()[0]) / dx

    result.get_pyobj().flat[0] = cur

    cur = (y1.get_pyobj()[-1] - y1.get_pyobj()[-2]) / dx

    result.get_pyobj().flat[size - 1] = cur

    for i in range(1, size - 1):
        cur = (y1.get_pyobj()[i + 1] - y1.get_pyobj()[i - 1]) / (2 * dx)
        result.get_pyobj().flat[i] = cur

    return result


cpdef utils.dpnp_descriptor dpnp_hypot(utils.dpnp_descriptor x1_obj,
                                       utils.dpnp_descriptor x2_obj,
                                       object dtype=None,
                                       utils.dpnp_descriptor out=None,
                                       object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_HYPOT_EXT, x1_obj, x2_obj, dtype, out, where)


cpdef utils.dpnp_descriptor dpnp_maximum(utils.dpnp_descriptor x1_obj,
                                         utils.dpnp_descriptor x2_obj,
                                         object dtype=None,
                                         utils.dpnp_descriptor out=None,
                                         object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_MAXIMUM_EXT, x1_obj, x2_obj, dtype, out, where)


cpdef utils.dpnp_descriptor dpnp_minimum(utils.dpnp_descriptor x1_obj,
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

    # ceate result array with type given by FPTR data
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


cpdef utils.dpnp_descriptor dpnp_nancumprod(utils.dpnp_descriptor x1):
    cur_x1 = dpnp_copy(x1).get_pyobj()

    cur_x1_flatiter = cur_x1.flat

    for i in range(cur_x1.size):
        if dpnp.isnan(cur_x1_flatiter[i]):
            cur_x1_flatiter[i] = 1

    x1_desc = dpnp.get_dpnp_descriptor(cur_x1, copy_when_nondefault_queue=False)
    return dpnp_cumprod(x1_desc)


cpdef utils.dpnp_descriptor dpnp_nancumsum(utils.dpnp_descriptor x1):
    cur_x1 = dpnp_copy(x1).get_pyobj()

    cur_x1_flatiter = cur_x1.flat

    for i in range(cur_x1.size):
        if dpnp.isnan(cur_x1_flatiter[i]):
            cur_x1_flatiter[i] = 0

    x1_desc = dpnp.get_dpnp_descriptor(cur_x1, copy_when_nondefault_queue=False)
    return dpnp_cumsum(x1_desc)


cpdef utils.dpnp_descriptor dpnp_nanprod(utils.dpnp_descriptor x1):
    x1_obj = x1.get_array()
    cdef utils.dpnp_descriptor result = utils_py.create_output_descriptor_py(x1.shape,
                                                                             x1.dtype,
                                                                             None,
                                                                             device=x1_obj.sycl_device,
                                                                             usm_type=x1_obj.usm_type,
                                                                             sycl_queue=x1_obj.sycl_queue)

    for i in range(result.size):
        input_elem = x1.get_pyobj().flat[i]

        if dpnp.isnan(input_elem):
            result.get_pyobj().flat[i] = 1
        else:
            result.get_pyobj().flat[i] = input_elem

    return dpnp_prod(result)


cpdef utils.dpnp_descriptor dpnp_nansum(utils.dpnp_descriptor x1):
    x1_obj = x1.get_array()
    cdef utils.dpnp_descriptor result = utils_py.create_output_descriptor_py(x1.shape,
                                                                             x1.dtype,
                                                                             None,
                                                                             device=x1_obj.sycl_device,
                                                                             usm_type=x1_obj.usm_type,
                                                                             sycl_queue=x1_obj.sycl_queue)

    for i in range(result.size):
        input_elem = x1.get_pyobj().flat[i]

        if dpnp.isnan(input_elem):
            result.get_pyobj().flat[i] = 0
        else:
            result.get_pyobj().flat[i] = input_elem

    return dpnp_sum(result)


cpdef utils.dpnp_descriptor dpnp_negative(dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_NEGATIVE_EXT, x1)


cpdef utils.dpnp_descriptor dpnp_prod(utils.dpnp_descriptor x1,
                                      object axis=None,
                                      object dtype=None,
                                      utils.dpnp_descriptor out=None,
                                      cpp_bool keepdims=False,
                                      object initial=None,
                                      object where=True):
    """
    input:float64   : outout:float64   : name:prod
    input:float32   : outout:float32   : name:prod
    input:int64     : outout:int64     : name:prod
    input:int32     : outout:int64     : name:prod
    input:bool      : outout:int64     : name:prod
    input:complex64 : outout:complex64 : name:prod
    input:complex128: outout:complex128: name:prod
    """

    cdef shape_type_c x1_shape = x1.shape
    cdef DPNPFuncType x1_c_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)

    cdef shape_type_c axis_shape = utils._object_to_tuple(axis)

    cdef shape_type_c result_shape = utils.get_reduction_output_shape(x1_shape, axis, keepdims)
    cdef DPNPFuncType result_c_type = utils.get_output_c_type(DPNP_FN_PROD_EXT, x1_c_type, out, dtype)

    """ select kernel """
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_PROD_EXT, x1_c_type, result_c_type)

    x1_obj = x1.get_array()

    """ Create result array """
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape,
                                                                       result_c_type,
                                                                       out,
                                                                       device=x1_obj.sycl_device,
                                                                       usm_type=x1_obj.usm_type,
                                                                       sycl_queue=x1_obj.sycl_queue)
    cdef dpnp_reduction_c_t func = <dpnp_reduction_c_t > kernel_data.ptr

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    """ Call FPTR interface function """
    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    result.get_data(),
                                                    x1.get_data(),
                                                    x1_shape.data(),
                                                    x1_shape.size(),
                                                    axis_shape.data(),
                                                    axis_shape.size(),
                                                    NULL,
                                                    NULL,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result


cpdef utils.dpnp_descriptor dpnp_remainder(utils.dpnp_descriptor x1_obj,
                                           utils.dpnp_descriptor x2_obj,
                                           object dtype=None,
                                           utils.dpnp_descriptor out=None,
                                           object where=True):
    return call_fptr_2in_1out(DPNP_FN_REMAINDER_EXT, x1_obj, x2_obj, dtype, out, where)


cpdef utils.dpnp_descriptor dpnp_sign(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_SIGN_EXT, x1)


cpdef utils.dpnp_descriptor dpnp_sum(utils.dpnp_descriptor x1,
                                     object axis=None,
                                     object dtype=None,
                                     utils.dpnp_descriptor out=None,
                                     cpp_bool keepdims=False,
                                     object initial=None,
                                     object where=True):

    cdef shape_type_c x1_shape = x1.shape
    cdef DPNPFuncType x1_c_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)

    cdef shape_type_c axis_shape = utils._object_to_tuple(axis)

    cdef shape_type_c result_shape = utils.get_reduction_output_shape(x1_shape, axis, keepdims)
    cdef DPNPFuncType result_c_type = utils.get_output_c_type(DPNP_FN_SUM_EXT, x1_c_type, out, dtype)

    """ select kernel """
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_SUM_EXT, x1_c_type, result_c_type)

    x1_obj = x1.get_array()

    """ Create result array """
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape,
                                                                       result_c_type,
                                                                       out,
                                                                       device=x1_obj.sycl_device,
                                                                       usm_type=x1_obj.usm_type,
                                                                       sycl_queue=x1_obj.sycl_queue)

    if x1.size == 0 and axis is None:
        return result

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    """ Call FPTR interface function """
    cdef dpnp_reduction_c_t func = <dpnp_reduction_c_t > kernel_data.ptr
    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    result.get_data(),
                                                    x1.get_data(),
                                                    x1_shape.data(),
                                                    x1_shape.size(),
                                                    axis_shape.data(),
                                                    axis_shape.size(),
                                                    NULL,
                                                    NULL,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result


cpdef utils.dpnp_descriptor dpnp_trapz(utils.dpnp_descriptor y1, utils.dpnp_descriptor x1, double dx):

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(y1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_TRAPZ_EXT, param1_type, param2_type)

    result_sycl_device, result_usm_type, result_sycl_queue = utils.get_common_usm_allocation(y1, x1)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = (1,)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape,
                                                                       kernel_data.return_type,
                                                                       None,
                                                                       device=result_sycl_device,
                                                                       usm_type=result_usm_type,
                                                                       sycl_queue=result_sycl_queue)

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef ftpr_custom_trapz_2in_1out_with_2size_t func = <ftpr_custom_trapz_2in_1out_with_2size_t > kernel_data.ptr
    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    y1.get_data(),
                                                    x1.get_data(),
                                                    result.get_data(),
                                                    dx,
                                                    y1.size,
                                                    x1.size,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result


cpdef utils.dpnp_descriptor dpnp_trunc(utils.dpnp_descriptor x1, utils.dpnp_descriptor out):
    return call_fptr_1in_1out_strides(DPNP_FN_TRUNC_EXT, x1, dtype=None, out=out, where=True, func_name='trunc')
