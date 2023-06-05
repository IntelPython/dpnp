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

"""Module Backend (linear algebra routines)

This module contains interface functions between C backend layer
and the rest of the library

"""

# NO IMPORTs here. All imports must be placed into main "dpnp_algo.pyx" file

__all__ += [
    "dpnp_dot",
    "dpnp_inner",
    "dpnp_kron",
    "dpnp_matmul",
    "dpnp_outer"
]


# C function pointer to the C library template functions
ctypedef c_dpctl.DPCTLSyclEventRef(*fptr_2in_1out_shapes_t)(c_dpctl.DPCTLSyclQueueRef,
                                                            void *, void * , void * , shape_elem_type * ,
                                                            shape_elem_type *, shape_elem_type * , size_t,
                                                            const c_dpctl.DPCTLEventVectorRef)
ctypedef c_dpctl.DPCTLSyclEventRef(*fptr_2in_1out_dot_t)(c_dpctl.DPCTLSyclQueueRef,
                                                         void * , const size_t, const size_t,
                                                         const shape_elem_type *, const shape_elem_type * ,
                                                         void * , const size_t, const size_t,
                                                         const shape_elem_type *, const shape_elem_type * ,
                                                         void * , const size_t, const size_t,
                                                         const shape_elem_type *, const shape_elem_type * ,
                                                         const c_dpctl.DPCTLEventVectorRef)
ctypedef c_dpctl.DPCTLSyclEventRef(*fptr_2in_1out_matmul_t)(c_dpctl.DPCTLSyclQueueRef,
                                                            void * , const size_t, const size_t,
                                                            const shape_elem_type *, const shape_elem_type * ,
                                                            void * , const size_t, const size_t,
                                                            const shape_elem_type *, const shape_elem_type * ,
                                                            void * , const size_t, const size_t,
                                                            const shape_elem_type *, const shape_elem_type * ,
                                                            const c_dpctl.DPCTLEventVectorRef)

cpdef utils.dpnp_descriptor dpnp_dot(utils.dpnp_descriptor in_array1,
                                     utils.dpnp_descriptor in_array2,
                                     utils.dpnp_descriptor out=None):
    cdef shape_type_c shape1, shape2

    shape1 = in_array1.shape
    shape2 = in_array2.shape

    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(in_array1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(in_array2.dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_DOT_EXT, param1_type, param2_type)
    cdef utils.dpnp_descriptor result

    ndim1 = in_array1.ndim
    ndim2 = in_array2.ndim
    cdef shape_type_c result_shape
    if ndim1 == 0:
        result_shape = shape2
    elif ndim2 == 0:
        result_shape = shape1
    elif ndim1 == 1 and ndim2 == 1:
        result_shape = ()
    elif ndim1 == 1:  # ndim2 > 1
        result_shape = shape2[::-2] if ndim2 == 2 else shape2[::2]
    elif ndim2 == 1:  # ndim1 > 1
        result_shape = shape1[:-1]
    else:
        if ndim1 == 1:
            shape1 = (1, shape1[0])
        if ndim2 == 1:
            shape2 = (shape1[0], 1)
        result_shape = shape1[:-1] + shape2[:-2] + shape2[-1:]

    result_sycl_device, result_usm_type, result_sycl_queue = utils.get_common_usm_allocation(in_array1, in_array2)

    if out is None:
        # create result array with type given by FPTR data
        result = utils.create_output_descriptor(result_shape,
                                                kernel_data.return_type,
                                                None,
                                                device=result_sycl_device,
                                                usm_type=result_usm_type,
                                                sycl_queue=result_sycl_queue)
    else:
        result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
        if out.dtype != result_type:
            utils.checker_throw_value_error('dot', 'out.dtype', out.dtype, result_type)
        if out.shape != result_shape:
            utils.checker_throw_value_error('dot', 'out.shape', out.shape, result_shape)

        result = out

        utils.get_common_usm_allocation(in_array1, result)  # check USM allocation is common

    cdef shape_type_c result_strides = utils.strides_to_vector(result.strides, result.shape)
    cdef shape_type_c in_array1_shape = in_array1.shape
    cdef shape_type_c in_array1_strides = utils.strides_to_vector(in_array1.strides, in_array1.shape)
    cdef shape_type_c in_array2_shape = in_array2.shape
    cdef shape_type_c in_array2_strides = utils.strides_to_vector(in_array2.strides, in_array2.shape)

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef fptr_2in_1out_dot_t func = <fptr_2in_1out_dot_t > kernel_data.ptr
    # call FPTR function
    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    result.get_data(),
                                                    result.size,
                                                    result.ndim,
                                                    result_shape.data(),
                                                    result_strides.data(),
                                                    in_array1.get_data(),
                                                    in_array1.size,
                                                    in_array1.ndim,
                                                    in_array1_shape.data(),
                                                    in_array1_strides.data(),
                                                    in_array2.get_data(),
                                                    in_array2.size,
                                                    in_array2.ndim,
                                                    in_array2_shape.data(),
                                                    in_array2_strides.data(),
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result


cpdef utils.dpnp_descriptor dpnp_inner(dpnp_descriptor array1, dpnp_descriptor array2):
    result_type = numpy.promote_types(array1.dtype, array1.dtype)

    assert(len(array1.shape) == len(array2.shape))

    cdef shape_type_c array1_no_last_axes = array1.shape[:-1]
    cdef shape_type_c array2_no_last_axes = array2.shape[:-1]

    cdef shape_type_c result_shape = array1_no_last_axes
    result_shape.insert(result_shape.end(), array2_no_last_axes.begin(), array2_no_last_axes.end())

    result_sycl_device, result_usm_type, result_sycl_queue = utils.get_common_usm_allocation(array1, array2)

    # ceate result array with type given by FPTR data
    cdef utils.dpnp_descriptor result = utils_py.create_output_descriptor_py(result_shape,
                                                                             result_type,
                                                                             None,
                                                                             device=result_sycl_device,
                                                                             usm_type=result_usm_type,
                                                                             sycl_queue=result_sycl_queue)

    # calculate input arrays offsets
    cdef shape_type_c array1_offsets = [1] * len(array1.shape)
    cdef shape_type_c array2_offsets = [1] * len(array2.shape)
    cdef size_t acc1 = 1
    cdef size_t acc2 = 1
    for axis in range(len(array1.shape) - 1, -1, -1):
        array1_offsets[axis] = acc1
        array2_offsets[axis] = acc2
        acc1 *= array1.shape[axis]
        acc2 *= array2.shape[axis]

    cdef shape_type_c result_shape_offsets = [1] * len(result.shape)
    acc = 1
    for i in range(len(result.shape) - 1, -1, -1):
        result_shape_offsets[i] = acc
        acc *= result.shape[i]

    cdef shape_type_c xyz
    cdef size_t array1_lin_index_base
    cdef size_t array2_lin_index_base
    cdef size_t axis2
    cdef long remainder
    cdef long quotient

    result_flatiter = result.get_pyobj().flat
    array1_flatiter = array1.get_pyobj().flat
    array2_flatiter = array2.get_pyobj().flat

    for idx1 in range(result.size):
        # reconstruct x,y,z from linear index
        xyz.clear()
        remainder = idx1
        for i in result_shape_offsets:
            quotient, remainder = divmod(remainder, i)
            xyz.push_back(quotient)

        # calculate linear base input index
        array1_lin_index_base = 0
        array2_lin_index_base = 0
        for axis in range(len(array1_offsets) - 1):
            axis2 = axis + (len(xyz) / 2)
            array1_lin_index_base += array1_offsets[axis] * xyz[axis]
            array2_lin_index_base += array2_offsets[axis] * xyz[axis2]

        # do inner product
        result_flatiter[idx1] = 0
        for idx2 in range(array1.shape[-1]):
            result_flatiter[idx1] += array1_flatiter[array1_lin_index_base + idx2] * \
                array2_flatiter[array2_lin_index_base + idx2]

    return result


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

    # ceate result array with type given by FPTR data
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


cpdef utils.dpnp_descriptor dpnp_matmul(utils.dpnp_descriptor in_array1, utils.dpnp_descriptor in_array2, utils.dpnp_descriptor out=None):

    cdef shape_type_c shape_result

    cdef shape_type_c shape1 = in_array1.shape
    cdef shape_type_c shape2 = in_array2.shape

    cdef size_t size_m = 0
    cdef size_t size_n = 0
    cdef size_t size_k = 0

    # Calling this function on an empty container causes undefined behavior.
    if not shape1.empty():
        size_m = shape1.front()
    if not shape2.empty():
        size_n = shape2.back()
    if not shape1.empty():
        size_k = shape1.back()

    cdef size_t ndim_max = max(in_array1.ndim, in_array2.ndim)

    if in_array1.ndim < ndim_max or ndim_max == 1:
        """
        shape1(2,), shape2(2,4)
        test: pytest tests/test_matmul.py::test_matmul[shape_pair4-types0] -v -s
        or
        shape1(2,), shape2(2,)
        test: pytest tests/test_matmul.py::test_matmul[shape_pair8-types0] -v -s
        """
        size_m = 1

    if in_array2.ndim < ndim_max or ndim_max == 1:
        """
        shape1(5,2), shape2(2,)
        test: pytest tests/test_matmul.py::test_matmul[shape_pair6-types0] -v -s
        or
        shape1(3,), shape2(3,)
        test: pytest tests/test_matmul.py::test_matmul[shape_pair8-types0] -v -s
        """
        size_n = 1

    shape_result = shape1[:-1] + shape2[-1:]

    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(in_array1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(in_array2.dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MATMUL_EXT, param1_type, param2_type)

    # ceate result array with type given by FPTR data
    result_sycl_device, result_usm_type, result_sycl_queue = utils.get_common_usm_allocation(in_array1, in_array2)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(shape_result,
                                                                       kernel_data.return_type,
                                                                       out,
                                                                       device=result_sycl_device,
                                                                       usm_type=result_usm_type,
                                                                       sycl_queue=result_sycl_queue)
    if result.size == 0:
        return result

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef fptr_2in_1out_matmul_t func = <fptr_2in_1out_matmul_t > kernel_data.ptr
    # call FPTR function
    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    result.get_data(),
                                                    result.size,
                                                    result.ndim,
                                                    NULL,  # result_shape
                                                    NULL,  # result_strides
                                                    in_array1.get_data(),
                                                    in_array1.size,
                                                    in_array1.ndim,
                                                    shape1.data(),
                                                    NULL,  # in_array1_strides
                                                    in_array2.get_data(),
                                                    in_array2.size,
                                                    in_array2.ndim,
                                                    shape2.data(),
                                                    NULL,  # in_array2_strides
                                                    NULL)  # dep_event_vec_ref
    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result


cpdef utils.dpnp_descriptor dpnp_outer(utils.dpnp_descriptor array1, utils.dpnp_descriptor array2):
    cdef shape_type_c result_shape = (array1.size, array2.size)
    result_type = numpy.promote_types(array1.dtype, array1.dtype)

    result_sycl_device, result_usm_type, result_sycl_queue = utils.get_common_usm_allocation(array1, array2)

    cdef utils.dpnp_descriptor result = utils_py.create_output_descriptor_py(result_shape,
                                                                             result_type,
                                                                             None,
                                                                             device=result_sycl_device,
                                                                             usm_type=result_usm_type,
                                                                             sycl_queue=result_sycl_queue)

    result_flatiter = result.get_pyobj().flat
    array1_flatiter = array1.get_pyobj().flat
    array2_flatiter = array2.get_pyobj().flat

    for idx1 in range(array1.size):
        for idx2 in range(array2.size):
            result_flatiter[idx1 * array2.size + idx2] = array1_flatiter[idx1] * array2_flatiter[idx2]

    return result
