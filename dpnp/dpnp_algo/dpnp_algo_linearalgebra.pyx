# cython: language_level=3
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2020, Intel Corporation
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
ctypedef void(*fptr_2in_1out_shapes_t)(void *, void * , void * , size_t * , size_t * , size_t * , size_t)


cpdef dparray dpnp_dot(dpnp_descriptor in_array1, dpnp_descriptor in_array2):

    cdef dparray_shape_type shape1, shape2

    shape1 = in_array1.shape
    shape2 = in_array2.shape

    cdef size_t dim1 = in_array1.ndim
    cdef size_t dim2 = in_array2.ndim

    # matrix
    if dim1 == 2 and dim2 == 2:
        return dpnp_matmul(in_array1, in_array2)

    # scalar
    if dim1 == 0 or dim2 == 0:
        x1_desc = dpnp.get_dpnp_descriptor(in_array1)
        x2_desc = dpnp.get_dpnp_descriptor(in_array2)
        return dpnp_multiply(x1_desc, x2_desc)

    cdef size_t size1 = 0
    cdef size_t size2 = 0
    if not shape1.empty():
        size1 = shape1.front()
    if not shape1.empty():
        size2 = shape2.front()

    # vector
    # test: pytest tests/third_party/cupy/linalg_tests/test_product.py::TestProduct::test_dot_vec1 -v -s
    if size1 != size2:
        utils.checker_throw_runtime_error("dpnp_dot", "input vectors must be of equal size")

    # convert string type names (dparray.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(in_array1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(in_array2.dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_DOT, param1_type, param2_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    # ceate result array with type given by FPTR data
    cdef dparray result = dparray((1,), dtype=result_type)

    cdef fptr_2in_1out_t func = <fptr_2in_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), in_array1.get_data(), in_array1.size, shape1.data(), shape1.size(),
         in_array2.get_data(), in_array2.size, shape2.data(), shape2.size(), NULL)

    return result


cpdef dparray dpnp_inner(dpnp_descriptor array1, dpnp_descriptor array2):
    result_type = numpy.promote_types(array1.dtype, array1.dtype)

    assert(len(array1.shape) == len(array2.shape))

    cdef dparray_shape_type array1_no_last_axes = array1.shape[:-1]
    cdef dparray_shape_type array2_no_last_axes = array2.shape[:-1]

    cdef dparray_shape_type result_shape = array1_no_last_axes
    result_shape.insert(result_shape.end(), array2_no_last_axes.begin(), array2_no_last_axes.end())

    cdef dparray result = dparray(result_shape, dtype=result_type)

    # calculate input arrays offsets
    cdef dparray_shape_type array1_offsets = [1] * len(array1.shape)
    cdef dparray_shape_type array2_offsets = [1] * len(array2.shape)
    cdef size_t acc1 = 1
    cdef size_t acc2 = 1
    for axis in range(len(array1.shape) - 1, -1, -1):
        array1_offsets[axis] = acc1
        array2_offsets[axis] = acc2
        acc1 *= array1.shape[axis]
        acc2 *= array2.shape[axis]

    cdef dparray_shape_type result_shape_offsets = [1] * len(result.shape)
    acc = 1
    for i in range(len(result.shape) - 1, -1, -1):
        result_shape_offsets[i] = acc
        acc *= result.shape[i]

    cdef dparray_shape_type xyz
    cdef size_t array1_lin_index_base
    cdef size_t array2_lin_index_base
    cdef size_t axis2
    cdef long remainder
    cdef long quotient
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
        result[idx1] = 0
        for idx2 in range(array1.shape[-1]):
            result[idx1] += array1[array1_lin_index_base + idx2] * array2[array2_lin_index_base + idx2]

    return result


cpdef dparray dpnp_kron(dpnp_descriptor in_array1, dpnp_descriptor in_array2):
    cdef size_t ndim = max(in_array1.ndim, in_array2.ndim)

    cdef dparray_shape_type in_array1_shape
    if in_array1.ndim < ndim:
        for i in range(ndim - in_array1.ndim):
            in_array1_shape.push_back(1)
    for i in range(in_array1.ndim):
        in_array1_shape.push_back(in_array1.shape[i])

    cdef dparray_shape_type in_array2_shape
    if in_array2.ndim < ndim:
        for i in range(ndim - in_array2.ndim):
            in_array2_shape.push_back(1)
    for i in range(in_array2.ndim):
        in_array2_shape.push_back(in_array2.shape[i])

    cdef dparray_shape_type result_shape
    for i in range(ndim):
        result_shape.push_back(in_array1_shape[i] * in_array2_shape[i])

    # convert string type names (dparray.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(in_array1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(in_array2.dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_KRON, param1_type, param2_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    # ceate result array with type given by FPTR data
    cdef dparray result = dparray(result_shape, dtype=result_type)

    cdef fptr_2in_1out_shapes_t func = <fptr_2in_1out_shapes_t > kernel_data.ptr
    # call FPTR function
    func(in_array1.get_data(), in_array2.get_data(), result.get_data(), < size_t * > in_array1_shape.data(), < size_t * > in_array2_shape.data(), < size_t * > result_shape.data(), ndim)

    return result


cpdef dparray dpnp_matmul(dpnp_descriptor in_array1, dpnp_descriptor in_array2, dparray out=None):

    cdef dparray_shape_type shape_result

    cdef dparray_shape_type shape1 = in_array1.shape
    cdef dparray_shape_type shape2 = in_array2.shape

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

    if ndim_max > 2:
        """
        shape1(5, 3, 2) * shape2(5, 2, 4) -> result(5, 3, 4)
        test: pytest tests/test_matmul.py::test_matmul[shape_pair10-types0] -v -s
        """
        shape_result = shape1[:-1] + [shape2.back()]
    else:
        """
        shape1(5,2) * shape2(2,3) -> result(5,3)
        test: pytest tests/test_matmul.py::test_matmul[shape_pair0-types0] -v -s
        """
        shape_result = shape1[:-1] + shape2[1:]

    # convert string type names (dparray.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(in_array1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(in_array2.dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MATMUL, param1_type, param2_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)

    cdef dparray result

    if out is not None:
        if out.dtype != result_type:
            utils.checker_throw_value_error('matmul', 'out.dtype', out.dtype, result_type)
        if out.shape != shape_result:
            utils.checker_throw_value_error('matmul', 'out.shape', out.shape, shape_result)
        result = out
    else:
        result = dparray(shape_result, dtype=result_type)

    if result.size == 0:
        return result

    cdef fptr_blas_gemm_2in_1out_t func = <fptr_blas_gemm_2in_1out_t > kernel_data.ptr
    # call FPTR function
    func(in_array1.get_data(), in_array2.get_data(), result.get_data(), size_m, size_n, size_k)

    return result


cpdef dparray dpnp_outer(dpnp_descriptor array1, dpnp_descriptor array2):
    cdef dparray_shape_type result_shape = (array1.size, array2.size)
    result_type = numpy.promote_types(array1.dtype, array1.dtype)

    cdef dparray result = dparray(result_shape, dtype=result_type)

    for idx1 in range(array1.size):
        for idx2 in range(array2.size):
            result[idx1 * array2.size + idx2] = array1[idx1] * array2[idx2]

    return result
