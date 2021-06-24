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

from libc.time cimport time, time_t
import dpnp
import dpnp.config as config
import numpy

cimport cpython
cimport dpnp.dpnp_utils as utils
cimport numpy


__all__ = [
    "dpnp_arange",
    "dpnp_array",
    "dpnp_astype",
    "dpnp_flatten",
    "dpnp_init_val",
    "dpnp_matmul",
    "dpnp_queue_initialize",
    "dpnp_queue_is_cpu",
    "dpnp_remainder"
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


ctypedef void(*fptr_dpnp_arange_t)(size_t, size_t, void * , size_t)
ctypedef void(*fptr_dpnp_astype_t)(const void * , void * , const size_t)
ctypedef void(*fptr_dpnp_flatten_t)(const void * , void * , const size_t)
ctypedef void(*fptr_dpnp_initval_t)(void * , void * , size_t)


cpdef dparray dpnp_arange(start, stop, step, dtype):

    if step is not 1:
        raise ValueError("DPNP dpnp_arange(): `step` is not supported")

    obj_len = int(numpy.ceil((stop - start) / step))
    if obj_len < 0:
        raise ValueError(f"DPNP dpnp_arange(): Negative array size (start={start},stop={stop},step={step})")

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(dtype)
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_ARANGE, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef dparray result = dparray(obj_len, dtype=result_type)

    # for i in range(result.size):
    #     result[i] = start + i

    cdef fptr_dpnp_arange_t func = <fptr_dpnp_arange_t > kernel_data.ptr
    func(start, step, result.get_data(), result.size)

    return result


cpdef dparray dpnp_array(obj, dtype=None):
    cdef dparray result
    cdef elem_dtype
    cdef dparray_shape_type obj_shape

    # convert scalar to tuple
    if dpnp.isscalar(obj):
        obj = (obj, )

    if not cpython.PySequence_Check(obj):
        raise TypeError(f"DPNP array(): Unsupported non-sequence obj={type(obj)}")

    obj_shape, elem_dtype = utils.get_shape_dtype(obj)
    if dtype is not None:
        """ Set type from parameter. result might be empty array """
        result = dparray(obj_shape, dtype=dtype)
    else:
        if obj_shape.empty():
            """ Empty object (ex. empty list) and no type provided """
            result = dparray(obj_shape)
        else:
            result = dparray(obj_shape, dtype=elem_dtype)

    utils.copy_values_to_dparray(result, obj)

    return result


cpdef dparray dpnp_astype(dparray array1, dtype_target):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(dtype_target)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_ASTYPE, param1_type, param2_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    cdef dparray result = dparray(array1.shape, dtype=result_type)

    cdef fptr_dpnp_astype_t func = <fptr_dpnp_astype_t > kernel_data.ptr
    func(array1.get_data(), result.get_data(), array1.size)

    return result


cpdef dparray dpnp_flatten(dparray array_):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array_.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_FLATTEN, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    cdef dparray result = dparray(array_.size, dtype=result_type)

    cdef fptr_dpnp_flatten_t func = <fptr_dpnp_flatten_t > kernel_data.ptr
    func(array_.get_data(), result.get_data(), array_.size)

    return result


cpdef dparray dpnp_init_val(shape, dtype, value):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_INITVAL, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    cdef dparray result = dparray(shape, dtype=dtype)

    # TODO: find better way to pass single value with type conversion
    cdef dparray val_arr = dparray((1, ), dtype=dtype)
    val_arr[0] = value

    cdef fptr_dpnp_initval_t func = <fptr_dpnp_initval_t > kernel_data.ptr
    func(result.get_data(), val_arr.get_data(), result.size)

    return result


cpdef dparray dpnp_matmul(dparray in_array1, dparray in_array2, dparray out=None):

    cdef vector[Py_ssize_t] shape_result

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

    # TODO:
    # choose seed number as is in numpy
    seed_from_time = time(NULL)
    dpnp_rng_srand_c(seed_from_time)


cpdef dpnp_queue_is_cpu():
    """Return 1 if current queue is CPU or HOST. Return 0 otherwise.

    """
    return dpnp_queue_is_cpu_c()


"""
Internal functions
"""
cpdef DPNPFuncType dpnp_dtype_to_DPNPFuncType(dtype):

    if dtype in [numpy.float64, 'float64']:
        return DPNP_FT_DOUBLE
    elif dtype in [numpy.float32, 'float32', 'f4']:
        return DPNP_FT_FLOAT
    elif dtype in [numpy.int64, 'int64', 'int', int]:
        return DPNP_FT_LONG
    elif dtype in [numpy.int32, 'int32']:
        return DPNP_FT_INT
    elif dtype in [numpy.complex128, 'complex128']:
        return DPNP_FT_CMPLX128
    elif dtype in [numpy.bool, numpy.bool_, 'bool', '?']:
        return DPNP_FT_BOOL
    else:
        utils.checker_throw_type_error("dpnp_dtype_to_DPNPFuncType", dtype)

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
    elif type == <size_t > DPNP_FT_CMPLX128:
        return numpy.complex128
    elif type == <size_t > DPNP_FT_BOOL:
        return numpy.bool
    else:
        utils.checker_throw_type_error("dpnp_DPNPFuncType_to_dtype", type)


cdef dparray call_fptr_1out(DPNPFuncName fptr_name, result_shape, result_dtype):

    # Convert string type names (dparray.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType dtype_in = dpnp_dtype_to_DPNPFuncType(result_dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(fptr_name, dtype_in, dtype_in)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)

    # Create result array with type given by FPTR data
    cdef dparray result = dparray(result_shape, dtype=result_type)

    cdef fptr_1out_t func = <fptr_1out_t > kernel_data.ptr
    # Call FPTR function
    func(result.get_data(), result.size)

    return result


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


cdef dparray call_fptr_2in_1out(DPNPFuncName fptr_name, object x1_obj, object x2_obj,
                                object dtype=None, dparray out=None, object where=True):
    cdef dparray_shape_type x1_shape, x2_shape, result_shape

    cdef bint x1_obj_is_dparray = isinstance(x1_obj, dparray)
    cdef bint x2_obj_is_dparray = isinstance(x2_obj, dparray)

    cdef dparray x1_dparray, x2_dparray

    common_type = utils.find_common_type(x1_obj, x2_obj)

    if x1_obj_is_dparray:
        x1_dparray = x1_obj
    else:
        x1_dparray = dparray((1,), dtype=common_type)
        utils.copy_values_to_dparray(x1_dparray, (x1_obj,))

    if x2_obj_is_dparray:
        x2_dparray = x2_obj
    else:
        x2_dparray = dparray((1,), dtype=common_type)
        utils.copy_values_to_dparray(x2_dparray, (x2_obj,))

    x1_shape = x1_dparray.shape
    x2_shape = x2_dparray.shape
    result_shape = utils.get_common_shape(x1_shape, x2_shape)

    # Convert string type names (dparray.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType x1_c_type = dpnp_dtype_to_DPNPFuncType(x1_dparray.dtype)
    cdef DPNPFuncType x2_c_type = dpnp_dtype_to_DPNPFuncType(x2_dparray.dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(fptr_name, x1_c_type, x2_c_type)

    # TODO: apply parameters out and dtype after reafactoring fmap (required 4th level nesting)

    # Create result array
    cdef dparray result = utils.create_output_array(result_shape, kernel_data.return_type, out)

    """ Call FPTR function """
    cdef fptr_2in_1out_t func = <fptr_2in_1out_t > kernel_data.ptr
    func(result.get_data(), x1_dparray.get_data(), x1_dparray.size, x1_shape.data(), x1_shape.size(),
         x2_dparray.get_data(), x2_dparray.size, x2_shape.data(), x2_shape.size(), NULL)

    return result

# this is replacement for "call_fptr_2in_1out". original function must be deleted after transotion.
cdef dparray call_fptr_2in_1out_new(DPNPFuncName fptr_name, utils.dpnp_descriptor x1_obj, utils.dpnp_descriptor x2_obj,
                                    object dtype=None, dparray out=None, object where=True):
    # Convert string type names (dparray.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType x1_c_type = dpnp_dtype_to_DPNPFuncType(x1_obj.dtype)
    cdef DPNPFuncType x2_c_type = dpnp_dtype_to_DPNPFuncType(x2_obj.dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(fptr_name, x1_c_type, x2_c_type)

    # Create result array
    cdef dparray_shape_type x1_shape = x1_obj.shape
    cdef dparray_shape_type x2_shape = x2_obj.shape
    cdef dparray_shape_type result_shape = utils.get_common_shape(x1_shape, x2_shape)
    cdef dparray result = utils.create_output_array(result_shape, kernel_data.return_type, out)

    """ Call FPTR function """
    cdef fptr_2in_1out_t func = <fptr_2in_1out_t > kernel_data.ptr
    func(result.get_data(), x1_obj.get_data(), x1_obj.size, x1_shape.data(), x1_shape.size(),
         x2_obj.get_data(), x2_obj.size, x2_shape.data(), x2_shape.size(), NULL)

    return result
