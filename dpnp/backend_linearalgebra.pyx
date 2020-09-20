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

    if dim1 >= 2 and dim2 == 1:
        raise NotImplementedError

    if dim1 >= 2 and dim2 >= 2:
        raise NotImplementedError

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


cpdef dparray dpnp_outer(dparray array1, dparray array2):
    cdef dparray_shape_type result_shape = (array1.size, array2.size)
    result_type = numpy.promote_types(array1.dtype, array1.dtype)

    cdef dparray result = dparray(result_shape, dtype=result_type)

    for idx1 in range(array1.size):
        for idx2 in range(array2.size):
            result[idx1 * array2.size + idx2] = array1[idx1] * array2[idx2]

    return result
