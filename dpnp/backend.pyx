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

"""Module Backend

This module contains interface functions between C backend layer
and the rest of the library

"""

import dpnp.config as config
import numpy

cimport cpython
cimport dpnp.dpnp_utils as utils
cimport numpy


__all__ = [
    "dpnp_arange",
    "dpnp_array",
    "dpnp_astype",
    "dpnp_init_val",
    "dpnp_matmul",
    "dpnp_queue_initialize",
    "dpnp_remainder"
]


include "backend_bitwise.pyx"
include "backend_counting.pyx"
include "backend_linearalgebra.pyx"
include "backend_logic.pyx"
include "backend_manipulation.pyx"
include "backend_mathematical.pyx"
include "backend_searching.pyx"
include "backend_sorting.pyx"
include "backend_statistics.pyx"
include "backend_trigonometric.pyx"


cpdef dparray dpnp_arange(start, stop, step, dtype):

    if step is not 1:
        raise ValueError("Intel NumPy dpnp_arange(): `step` is not supported")

    obj_len = int(numpy.ceil((stop - start) / step))
    if obj_len < 0:
        raise ValueError(f"Intel NumPy dpnp_arange(): Negative array size (start={start},stop={stop},step={step})")

    cdef dparray result = dparray(obj_len, dtype=dtype)

    for i in range(result.size):
        result[i] = start + i

    return result


cpdef dparray dpnp_array(obj, dtype=None):
    cdef dparray result
    cdef elem_dtype
    cdef dparray_shape_type obj_shape

    if not cpython.PySequence_Check(obj):
        raise TypeError(f"Intel NumPy array(): Unsupported non-sequence obj={type(obj)}")

    obj_shape, elem_dtype = get_shape_dtype(obj)
    if dtype is not None:
        """ Set type from parameter. result might be empty array """
        result = dparray(obj_shape, dtype=dtype)
    else:
        if obj_shape.empty():
            """ Empty object (ex. empty list) and no type provided """
            result = dparray(obj_shape)
        else:
            result = dparray(obj_shape, dtype=elem_dtype)

    copy_values_to_dparray(result, obj)

    return result


cpdef dparray dpnp_astype(dparray array1, dtype_target):
    cdef dparray result = dparray(array1.shape, dtype=dtype_target)

    for i in range(result.size):
        result[i] = array1[i]

    return result


cpdef dparray dpnp_init_val(shape, dtype, value):
    cdef dparray result = dparray(shape, dtype=dtype)

    for i in range(result.size):
        result[i] = value

    return result


cpdef dparray dpnp_matmul(dparray in_array1, dparray in_array2):
    cdef vector[Py_ssize_t] shape_result

    call_type = in_array1.dtype

    cdef vector[Py_ssize_t] shape1 = in_array1.shape
    cdef vector[Py_ssize_t] shape2 = in_array2.shape

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

    cdef dparray result = dparray(shape_result, dtype=call_type)
    if result.size == 0:
        return result

    if call_type == numpy.float64:
        dpnp_blas_gemm_c[double](in_array1.get_data(), in_array2.get_data(),
                                 result.get_data(), size_m, size_n, size_k)
    elif call_type == numpy.float32:
        dpnp_blas_gemm_c[float](in_array1.get_data(), in_array2.get_data(),
                                result.get_data(), size_m, size_n, size_k)
    elif call_type == numpy.int64:
        custom_blas_gemm_c[long](in_array1.get_data(), in_array2.get_data(),
                                 result.get_data(), size_m, size_n, size_k)
    elif call_type == numpy.int32:
        custom_blas_gemm_c[int](in_array1.get_data(), in_array2.get_data(),
                                result.get_data(), size_m, size_n, size_k)
    else:
        raise TypeError(
            f"Intel NumPy {call_type} matmul({call_type} A, {call_type} B, {call_type} result) is not supported.")

    return result


cpdef dpnp_queue_initialize():
    """Initialize SYCL queue which will be used for any library operations
    It takes visible time and needs to be done in the module loading procedure

    """

    cdef QueueOptions queue_type = CPU_SELECTOR

    if (config.__DPNP_QUEUE_GPU__):
        queue_type = GPU_SELECTOR

    dpnp_queue_initialize_c(queue_type)


cpdef dparray dpnp_remainder(dparray array1, int scalar):

    cdef dparray result = dparray(array1.shape, dtype=array1.dtype)

    for i in range(result.size):
        result[i] = (array1[i] % scalar)

    return result

"""
Internal functions
"""
cpdef DPNPFuncType dpnp_dtype_to_DPNPFuncType(dtype):

    if dtype == numpy.float64:
        return DPNP_FT_DOUBLE
    elif dtype == numpy.float32:
        return DPNP_FT_FLOAT
    elif dtype == numpy.int64:
        return DPNP_FT_LONG
    elif dtype == numpy.int32:
        return DPNP_FT_INT
    else:
        checker_throw_type_error("dpnp_dtype_to_DPNPFuncType", dtype)

cpdef dpnp_DPNPFuncType_to_dtype(size_t type):
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
    else:
        checker_throw_type_error("dpnp_DPNPFuncType_to_dtype", type)


cdef dparray call_fptr_1in_1out(DPNPFuncName fptr_name, dparray x1, dparray_shape_type result_shape):

    """ Convert string type names (dparray.dtype) to C enum DPNPFuncType """
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)

    """ get the FPTR data structure """
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(fptr_name, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    """ Create result array with type given by FPTR data """
    cdef dparray result = dparray(result_shape, dtype=result_type)

    cdef fptr_1in_1out_t func = <fptr_1in_1out_t > kernel_data.ptr
    """ Call FPTR function """
    func(x1.get_data(), result.get_data(), x1.size)

    return result


cdef dparray call_fptr_2in_1out(DPNPFuncName fptr_name, dparray x1, dparray x2, dparray_shape_type result_shape):

    """ Convert string type names (dparray.dtype) to C enum DPNPFuncType """
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(x2.dtype)

    """ get the FPTR data structure """
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(fptr_name, param1_type, param2_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    """ Create result array with type given by FPTR data """
    cdef dparray result = dparray(result_shape, dtype=result_type)

    cdef fptr_2in_1out_t func = <fptr_2in_1out_t > kernel_data.ptr
    """ Call FPTR function """
    func(x1.get_data(), x2.get_data(), result.get_data(), x1.size)

    return result
