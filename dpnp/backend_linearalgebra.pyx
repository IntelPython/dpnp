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

from dpnp.dpnp_utils cimport checker_throw_type_error, normalize_axis
cimport numpy


__all__ += [
    "dpnp_dot",
    "dpnp_inner",
    "dpnp_outer"
]


cpdef dparray dpnp_dot(dparray in_array1, dparray in_array2):
    cdef vector[Py_ssize_t] shape1 = in_array1.shape
    cdef vector[Py_ssize_t] shape2 = in_array2.shape

    call_type = in_array1.dtype

    cdef size_t dim1 = in_array1.ndim
    cdef size_t dim2 = in_array2.ndim

    # matrix
    if dim1 == 2 and dim2 == 2:
        return dpnp_matmul(in_array1, in_array2)

    # scalar
    if dim1 == 0 or dim2 == 0:
        return dpnp_multiply(in_array1, in_array2)

    cdef size_t size1 = 0
    cdef size_t size2 = 0
    if not shape1.empty():
        size1 = shape1.front()
    if not shape1.empty():
        size2 = shape2.front()

    # vector
    # test: pytest tests/third_party/cupy/linalg_tests/test_product.py::TestProduct::test_dot_vec1 -v -s
    if size1 != size2:
        raise checker_throw_runtime_error("dpnp_dot", "input vectors must be of equal size")

    cdef dparray result = dparray((1,), dtype=call_type)
    if call_type == numpy.float64:
        mkl_blas_dot_c[double](in_array1.get_data(), in_array2.get_data(), result.get_data(), size1)
    elif call_type == numpy.float32:
        mkl_blas_dot_c[float](in_array1.get_data(), in_array2.get_data(), result.get_data(), size1)
    elif call_type == numpy.int64:
        custom_blas_dot_c[long](in_array1.get_data(), in_array2.get_data(), result.get_data(), size1)
    elif call_type == numpy.int32:
        custom_blas_dot_c[int](in_array1.get_data(), in_array2.get_data(), result.get_data(), size1)
    else:
        checker_throw_type_error("dpnp_dot", call_type)

    return result


cpdef dparray dpnp_inner(dparray array1, dparray array2):
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


cpdef dparray dpnp_outer(dparray array1, dparray array2):
    cdef dparray_shape_type result_shape = (array1.size, array2.size)
    result_type = numpy.promote_types(array1.dtype, array1.dtype)

    cdef dparray result = dparray(result_shape, dtype=result_type)

    for idx1 in range(array1.size):
        for idx2 in range(array2.size):
            result[idx1 * array2.size + idx2] = array1[idx1] * array2[idx2]

    return result
