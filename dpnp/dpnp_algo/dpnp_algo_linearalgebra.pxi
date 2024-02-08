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
    "dpnp_inner",
    "dpnp_kron",
]


# C function pointer to the C library template functions
ctypedef c_dpctl.DPCTLSyclEventRef(*fptr_2in_1out_shapes_t)(c_dpctl.DPCTLSyclQueueRef,
                                                            void *, void * , void * , shape_elem_type * ,
                                                            shape_elem_type *, shape_elem_type * , size_t,
                                                            const c_dpctl.DPCTLEventVectorRef)


cpdef utils.dpnp_descriptor dpnp_inner(dpnp_descriptor array1, dpnp_descriptor array2):
    result_type = numpy.promote_types(array1.dtype, array1.dtype)

    assert(len(array1.shape) == len(array2.shape))

    cdef shape_type_c array1_no_last_axes = array1.shape[:-1]
    cdef shape_type_c array2_no_last_axes = array2.shape[:-1]

    cdef shape_type_c result_shape = array1_no_last_axes
    result_shape.insert(result_shape.end(), array2_no_last_axes.begin(), array2_no_last_axes.end())

    result_sycl_device, result_usm_type, result_sycl_queue = utils.get_common_usm_allocation(array1, array2)

    # create result array with type given by FPTR data
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
