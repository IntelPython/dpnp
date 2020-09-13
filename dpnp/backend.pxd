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

from libcpp.vector cimport vector
from libcpp cimport bool
from dpnp.dparray cimport dparray, dparray_shape_type

cdef extern from "backend/backend_iface_fptr.hpp" namespace "DPNPFuncName":  # need this namespace for Enum import
    cdef enum DPNPFuncName "DPNPFuncName":
        DPNP_FN_ADD
        DPNP_FN_DOT

cdef extern from "backend/backend_iface_fptr.hpp" namespace "DPNPFuncType":  # need this namespace for Enum import
    cdef enum DPNPFuncType "DPNPFuncType":
        DPNP_FT_NONE
        DPNP_FT_INT
        DPNP_FT_LONG
        DPNP_FT_FLOAT
        DPNP_FT_DOUBLE

cdef extern from "backend/backend_iface_fptr.hpp":
    struct DPNPFuncData:
        DPNPFuncType return_type
        void* ptr

    DPNPFuncData get_dpnp_function_ptr(DPNPFuncName name, DPNPFuncType first_type, DPNPFuncType second_type)


cdef extern from "backend/backend_iface.hpp" namespace "QueueOptions":  # need this namespace for Enum import
    cdef enum QueueOptions "QueueOptions":
        CPU_SELECTOR
        GPU_SELECTOR

cdef extern from "backend/backend_iface.hpp":
    void dpnp_queue_initialize_c(QueueOptions selector)

    char * dpnp_memory_alloc_c(size_t size_in_bytes)
    void dpnp_memory_free_c(void * ptr)
    void dpnp_memory_memcpy_c(void * dst, const void * src, size_t size_in_bytes)

    void dpnp_blas_gemm_c[_DataType](void * array1, void * array2, void * result1, size_t size_m, size_t size_n, size_t size_k)
    void custom_blas_gemm_c[_DataType](void * array1, void * array2, void * result1, size_t size_m, size_t size_n, size_t size_k)

    # Linear Algebra part
    void custom_blas_dot_c[_DataType](void * array1, void * array2, void * result1, size_t size)
    void mkl_blas_dot_c[_DataType](void * array1, void * array2, void * result1, size_t size)
    void mkl_lapack_syevd_c[_DataType](void * array1, void * result1, size_t size)

    # Trigonometric part
    void custom_elemwise_acos_c[_DataType_input, _DataType_output](void * array1, void * result1, size_t size)
    void custom_elemwise_acosh_c[_DataType_input, _DataType_output](void * array1, void * result1, size_t size)
    void custom_elemwise_asin_c[_DataType_input, _DataType_output](void * array1, void * result1, size_t size)
    void custom_elemwise_asinh_c[_DataType_input, _DataType_output](void * array1, void * result1, size_t size)
    void custom_elemwise_atan_c[_DataType_input, _DataType_output](void * array1, void * result1, size_t size)
    void custom_elemwise_atanh_c[_DataType_input, _DataType_output](void * array1, void * result1, size_t size)
    void custom_elemwise_cbrt_c[_DataType_input, _DataType_output](void * array1, void * result1, size_t size)
    void custom_elemwise_cos_c[_DataType_input, _DataType_output](void * array1, void * result1, size_t size)
    void custom_elemwise_cosh_c[_DataType_input, _DataType_output](void * array1, void * result1, size_t size)
    void custom_elemwise_degrees_c[_DataType_input, _DataType_output](void * array1, void * result1, size_t size)
    void custom_elemwise_exp2_c[_DataType_input, _DataType_output](void * array1, void * result1, size_t size)
    void custom_elemwise_exp_c[_DataType_input, _DataType_output](void * array1, void * result1, size_t size)
    void custom_elemwise_expm1_c[_DataType_input, _DataType_output](void * array1, void * result1, size_t size)
    void custom_elemwise_log10_c[_DataType_input, _DataType_output](void * array1, void * result1, size_t size)
    void custom_elemwise_log1p_c[_DataType_input, _DataType_output](void * array1, void * result1, size_t size)
    void custom_elemwise_log2_c[_DataType_input, _DataType_output](void * array1, void * result1, size_t size)
    void custom_elemwise_log_c[_DataType_input, _DataType_output](void * array1, void * result1, size_t size)
    void custom_elemwise_radians_c[_DataType_input, _DataType_output](void * array1, void * result1, size_t size)
    void custom_elemwise_recip_c[_DataType](void * array1, void * result1, size_t size)
    void custom_elemwise_sin_c[_DataType_input, _DataType_output](void * array1, void * result1, size_t size)
    void custom_elemwise_sinh_c[_DataType_input, _DataType_output](void * array1, void * result1, size_t size)
    void custom_elemwise_sqrt_c[_DataType_input, _DataType_output](void * array1, void * result1, size_t size)
    void custom_elemwise_square_c[_DataType](void * array1, void * result1, size_t size)
    void custom_elemwise_tan_c[_DataType_input, _DataType_output](void * array1, void * result1, size_t size)
    void custom_elemwise_tanh_c[_DataType_input, _DataType_output](void * array1, void * result1, size_t size)

    # Mathematical part
    void custom_elemwise_add_c[_DataType_input1, _DataType_input2, _DataType_output](void * array1, void * array2, void * result1, size_t size)
    void custom_elemwise_arctan2_c[_DataType_input1, _DataType_input2, _DataType_output](void * array1, void * array2, void * result1, size_t size)
    void custom_elemwise_divide_c[_DataType_input1, _DataType_input2, _DataType_output](void * array1, void * array2, void * result1, size_t size)
    void custom_elemwise_hypot_c[_DataType_input1, _DataType_input2, _DataType_output](void * array1, void * array2, void * result1, size_t size)
    void custom_elemwise_multiply_c[_DataType_input1, _DataType_input2, _DataType_output](void * array1, void * array2, void * result1, size_t size)
    void custom_elemwise_power_c[_DataType_input1, _DataType_input2, _DataType_output](void * array1, void * array2, void * result1, size_t size)
    void custom_elemwise_subtract_c[_DataType_input1, _DataType_input2, _DataType_output](void * array1, void * array2, void * result1, size_t size)
    void custom_sum_c[_DataType](void * array, void * result, size_t size)

    # array manipulation routines
    void custom_elemwise_transpose_c[_DataType](void * array1_in, dparray_shape_type & input_shape, dparray_shape_type & result_shape, dparray_shape_type & permute_axes, void * result1, size_t size)

    # Random module routines
    void mkl_rng_gaussian[_DataType](void * result, size_t size)
    void mkl_rng_uniform[_DataType](void * result, size_t size)
    void mkl_rng_uniform_mt19937[_DataType](void * result, long low, long high, size_t size)

    # Sorting routines
    void custom_argsort_c[_DataType, _idx_DataType](void * array, void * result, size_t size)
    void custom_sort_c[_DataType](void * array, void * result, size_t size)


cpdef dparray dpnp_remainder(dparray array1, int scalar)
cpdef dparray dpnp_astype(dparray array1, dtype_target)


"""
Internal functions
"""
cpdef DPNPFuncType dpnp_dtype_to_DPNPFuncType(dtype)
cpdef dpnp_DPNPFuncType_to_dtype(size_t type)


"""
Logic functions
"""
cpdef dparray dpnp_equal(dparray array1, input2)
cpdef dparray dpnp_greater(dparray input1, dparray input2)
cpdef dparray dpnp_greater_equal(dparray input1, dparray input2)
cpdef dparray dpnp_isclose(dparray input1, input2, double rtol=*, double atol=*, bool equal_nan=*)
cpdef dparray dpnp_less(dparray input1, dparray input2)
cpdef dparray dpnp_less_equal(dparray input1, dparray input2)
cpdef dparray dpnp_logical_and(dparray input1, dparray input2)
cpdef dparray dpnp_logical_not(dparray input1)
cpdef dparray dpnp_logical_or(dparray input1, dparray input2)
cpdef dparray dpnp_logical_xor(dparray input1, dparray input2)
cpdef dparray dpnp_not_equal(dparray input1, dparray input2)


"""
Linear algebra
"""
cpdef dparray dpnp_matmul(dparray in_array1, dparray in_array2)


"""
Array creation routines
"""
cpdef dparray dpnp_arange(start, stop, step, dtype)
cpdef dparray dpnp_array(obj, dtype=*)
cpdef dparray dpnp_init_val(shape, dtype, value)


"""
Mathematical functions
"""
cpdef dparray dpnp_add(dparray array1, dparray array2)
cpdef dparray dpnp_arctan2(dparray array1, dparray array2)
cpdef dparray dpnp_cos(dparray array1)
cpdef dparray dpnp_divide(dparray array1, dparray array2)
cpdef dparray dpnp_hypot(dparray array1, dparray array2)
cpdef dparray dpnp_multiply(dparray array1, dparray array2)
cpdef dparray dpnp_negative(dparray array1)
cpdef dparray dpnp_power(dparray array1, dparray array2)
cpdef dparray dpnp_sin(dparray array1)
cpdef dparray dpnp_subtract(dparray array1, dparray array2)


"""
Array manipulation routines
"""
cpdef dparray dpnp_repeat(dparray array1, repeats, axes=*)
cpdef dparray dpnp_transpose(dparray array1, axes=*)


"""
Statistics functions
"""
cpdef dparray dpnp_mean(dparray a, axis)


"""
Sorting functions
"""
cpdef dparray dpnp_argsort(dparray array1)
cpdef dparray dpnp_sort(dparray array1)
