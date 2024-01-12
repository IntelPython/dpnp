# cython: language_level=3
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
cimport dpctl as c_dpctl
from libcpp cimport bool as cpp_bool

from dpnp.dpnp_algo cimport shape_elem_type, shape_type_c
from dpnp.dpnp_utils.dpnp_algo_utils cimport dpnp_descriptor


cdef extern from "dpnp_iface_fptr.hpp" namespace "DPNPFuncName":  # need this namespace for Enum import
    cdef enum DPNPFuncName "DPNPFuncName":
        DPNP_FN_ALLCLOSE
        DPNP_FN_ALLCLOSE_EXT
        DPNP_FN_ARANGE
        DPNP_FN_ARGSORT
        DPNP_FN_ARGSORT_EXT
        DPNP_FN_CHOLESKY
        DPNP_FN_CHOLESKY_EXT
        DPNP_FN_CHOOSE
        DPNP_FN_CHOOSE_EXT
        DPNP_FN_COPY
        DPNP_FN_COPY_EXT
        DPNP_FN_CORRELATE
        DPNP_FN_CORRELATE_EXT
        DPNP_FN_CROSS
        DPNP_FN_CROSS_EXT
        DPNP_FN_CUMPROD
        DPNP_FN_CUMPROD_EXT
        DPNP_FN_CUMSUM
        DPNP_FN_CUMSUM_EXT
        DPNP_FN_DEGREES
        DPNP_FN_DEGREES_EXT
        DPNP_FN_DET
        DPNP_FN_DET_EXT
        DPNP_FN_DIAG_INDICES
        DPNP_FN_DIAG_INDICES_EXT
        DPNP_FN_DIAGONAL
        DPNP_FN_DIAGONAL_EXT
        DPNP_FN_DOT
        DPNP_FN_DOT_EXT
        DPNP_FN_EDIFF1D
        DPNP_FN_EDIFF1D_EXT
        DPNP_FN_EIG
        DPNP_FN_EIG_EXT
        DPNP_FN_EIGVALS
        DPNP_FN_EIGVALS_EXT
        DPNP_FN_ERF
        DPNP_FN_ERF_EXT
        DPNP_FN_EYE
        DPNP_FN_EYE_EXT
        DPNP_FN_FABS
        DPNP_FN_FABS_EXT
        DPNP_FN_FFT_FFT
        DPNP_FN_FFT_FFT_EXT
        DPNP_FN_FFT_RFFT
        DPNP_FN_FFT_RFFT_EXT
        DPNP_FN_FILL_DIAGONAL
        DPNP_FN_FILL_DIAGONAL_EXT
        DPNP_FN_FMOD
        DPNP_FN_FMOD_EXT
        DPNP_FN_FULL
        DPNP_FN_FULL_LIKE
        DPNP_FN_INV
        DPNP_FN_INV_EXT
        DPNP_FN_KRON
        DPNP_FN_KRON_EXT
        DPNP_FN_MATMUL
        DPNP_FN_MATMUL_EXT
        DPNP_FN_MATRIX_RANK
        DPNP_FN_MATRIX_RANK_EXT
        DPNP_FN_MAXIMUM
        DPNP_FN_MAXIMUM_EXT
        DPNP_FN_MEDIAN
        DPNP_FN_MEDIAN_EXT
        DPNP_FN_MINIMUM
        DPNP_FN_MINIMUM_EXT
        DPNP_FN_MODF
        DPNP_FN_MODF_EXT
        DPNP_FN_NONZERO
        DPNP_FN_ONES
        DPNP_FN_ONES_LIKE
        DPNP_FN_PARTITION
        DPNP_FN_PARTITION_EXT
        DPNP_FN_PLACE
        DPNP_FN_QR
        DPNP_FN_QR_EXT
        DPNP_FN_RADIANS
        DPNP_FN_RADIANS_EXT
        DPNP_FN_RECIP
        DPNP_FN_RECIP_EXT
        DPNP_FN_RNG_BETA
        DPNP_FN_RNG_BETA_EXT
        DPNP_FN_RNG_BINOMIAL
        DPNP_FN_RNG_BINOMIAL_EXT
        DPNP_FN_RNG_CHISQUARE
        DPNP_FN_RNG_CHISQUARE_EXT
        DPNP_FN_RNG_EXPONENTIAL
        DPNP_FN_RNG_EXPONENTIAL_EXT
        DPNP_FN_RNG_F
        DPNP_FN_RNG_F_EXT
        DPNP_FN_RNG_GAMMA
        DPNP_FN_RNG_GAMMA_EXT
        DPNP_FN_RNG_GAUSSIAN
        DPNP_FN_RNG_GAUSSIAN_EXT
        DPNP_FN_RNG_GEOMETRIC
        DPNP_FN_RNG_GEOMETRIC_EXT
        DPNP_FN_RNG_GUMBEL
        DPNP_FN_RNG_GUMBEL_EXT
        DPNP_FN_RNG_HYPERGEOMETRIC
        DPNP_FN_RNG_HYPERGEOMETRIC_EXT
        DPNP_FN_RNG_LAPLACE
        DPNP_FN_RNG_LAPLACE_EXT
        DPNP_FN_RNG_LOGISTIC
        DPNP_FN_RNG_LOGISTIC_EXT
        DPNP_FN_RNG_LOGNORMAL
        DPNP_FN_RNG_LOGNORMAL_EXT
        DPNP_FN_RNG_MULTINOMIAL
        DPNP_FN_RNG_MULTINOMIAL_EXT
        DPNP_FN_RNG_MULTIVARIATE_NORMAL
        DPNP_FN_RNG_MULTIVARIATE_NORMAL_EXT
        DPNP_FN_RNG_NEGATIVE_BINOMIAL
        DPNP_FN_RNG_NEGATIVE_BINOMIAL_EXT
        DPNP_FN_RNG_NONCENTRAL_CHISQUARE
        DPNP_FN_RNG_NONCENTRAL_CHISQUARE_EXT
        DPNP_FN_RNG_NORMAL
        DPNP_FN_RNG_NORMAL_EXT
        DPNP_FN_RNG_PARETO
        DPNP_FN_RNG_PARETO_EXT
        DPNP_FN_RNG_POISSON
        DPNP_FN_RNG_POISSON_EXT
        DPNP_FN_RNG_POWER
        DPNP_FN_RNG_POWER_EXT
        DPNP_FN_RNG_RAYLEIGH
        DPNP_FN_RNG_RAYLEIGH_EXT
        DPNP_FN_RNG_SHUFFLE
        DPNP_FN_RNG_SHUFFLE_EXT
        DPNP_FN_RNG_SRAND
        DPNP_FN_RNG_SRAND_EXT
        DPNP_FN_RNG_STANDARD_CAUCHY
        DPNP_FN_RNG_STANDARD_CAUCHY_EXT
        DPNP_FN_RNG_STANDARD_EXPONENTIAL
        DPNP_FN_RNG_STANDARD_EXPONENTIAL_EXT
        DPNP_FN_RNG_STANDARD_GAMMA
        DPNP_FN_RNG_STANDARD_GAMMA_EXT
        DPNP_FN_RNG_STANDARD_NORMAL
        DPNP_FN_RNG_STANDARD_T
        DPNP_FN_RNG_STANDARD_T_EXT
        DPNP_FN_RNG_TRIANGULAR
        DPNP_FN_RNG_TRIANGULAR_EXT
        DPNP_FN_RNG_UNIFORM
        DPNP_FN_RNG_UNIFORM_EXT
        DPNP_FN_RNG_VONMISES
        DPNP_FN_RNG_VONMISES_EXT
        DPNP_FN_RNG_WALD
        DPNP_FN_RNG_WALD_EXT
        DPNP_FN_RNG_WEIBULL
        DPNP_FN_RNG_WEIBULL_EXT
        DPNP_FN_RNG_ZIPF
        DPNP_FN_RNG_ZIPF_EXT
        DPNP_FN_SEARCHSORTED
        DPNP_FN_SEARCHSORTED_EXT
        DPNP_FN_SORT
        DPNP_FN_SORT_EXT
        DPNP_FN_SVD
        DPNP_FN_SVD_EXT
        DPNP_FN_TRACE
        DPNP_FN_TRACE_EXT
        DPNP_FN_TRANSPOSE
        DPNP_FN_TRAPZ
        DPNP_FN_TRAPZ_EXT
        DPNP_FN_TRIL
        DPNP_FN_TRIL_EXT
        DPNP_FN_TRIU
        DPNP_FN_TRIU_EXT
        DPNP_FN_ZEROS
        DPNP_FN_ZEROS_LIKE

cdef extern from "dpnp_iface_fptr.hpp" namespace "DPNPFuncType":  # need this namespace for Enum import
    cdef enum DPNPFuncType "DPNPFuncType":
        DPNP_FT_NONE
        DPNP_FT_INT
        DPNP_FT_LONG
        DPNP_FT_FLOAT
        DPNP_FT_DOUBLE
        DPNP_FT_CMPLX64
        DPNP_FT_CMPLX128
        DPNP_FT_BOOL

cdef extern from "dpnp_iface_fptr.hpp":
    struct DPNPFuncData:
        DPNPFuncType return_type
        void * ptr
        DPNPFuncType return_type_no_fp64
        void *ptr_no_fp64

    DPNPFuncData get_dpnp_function_ptr(DPNPFuncName name, DPNPFuncType first_type, DPNPFuncType second_type) except +


cdef extern from "dpnp_iface.hpp" namespace "QueueOptions":  # need this namespace for Enum import
    cdef enum QueueOptions "QueueOptions":
        CPU_SELECTOR
        GPU_SELECTOR
        AUTO_SELECTOR

cdef extern from "constants.hpp":
    void dpnp_python_constants_initialize_c(void * py_none, void * py_nan)

cdef extern from "dpnp_iface.hpp":
    void dpnp_queue_initialize_c(QueueOptions selector)

    char * dpnp_memory_alloc_c(size_t size_in_bytes) except +
    void dpnp_memory_free_c(void * ptr)
    void dpnp_memory_memcpy_c(void * dst, const void * src, size_t size_in_bytes)
    void dpnp_rng_srand_c(size_t seed)


# C function pointer to the C library template functions
ctypedef c_dpctl.DPCTLSyclEventRef(*fptr_1out_t)(c_dpctl.DPCTLSyclQueueRef,
                                                 void * , size_t,
                                                 const c_dpctl.DPCTLEventVectorRef) except +
ctypedef c_dpctl.DPCTLSyclEventRef(*fptr_1in_1out_t)(c_dpctl.DPCTLSyclQueueRef,
                                                     void *, void * , size_t,
                                                     const c_dpctl.DPCTLEventVectorRef)
ctypedef c_dpctl.DPCTLSyclEventRef(*fptr_1in_1out_strides_t)(c_dpctl.DPCTLSyclQueueRef,
                                                             void *, const size_t, const size_t,
                                                             const shape_elem_type * , const shape_elem_type * ,
                                                             void *, const size_t, const size_t,
                                                             const shape_elem_type * , const shape_elem_type * ,
                                                             const long * ,
                                                             const c_dpctl.DPCTLEventVectorRef)
ctypedef c_dpctl.DPCTLSyclEventRef(*fptr_2in_1out_t)(c_dpctl.DPCTLSyclQueueRef,
                                                     void * ,
                                                     const void * ,
                                                     const size_t,
                                                     const shape_elem_type * ,
                                                     const size_t,
                                                     const void *,
                                                     const size_t,
                                                     const shape_elem_type * ,
                                                     const size_t,
                                                     const long * ,
                                                     const c_dpctl.DPCTLEventVectorRef)
ctypedef c_dpctl.DPCTLSyclEventRef(*fptr_2in_1out_strides_t)(c_dpctl.DPCTLSyclQueueRef,
                                                             void *,
                                                             const size_t,
                                                             const size_t,
                                                             const shape_elem_type * ,
                                                             const shape_elem_type * ,
                                                             void *,
                                                             const size_t,
                                                             const size_t,
                                                             const shape_elem_type * ,
                                                             const shape_elem_type * ,
                                                             void *,
                                                             const size_t, const size_t,
                                                             const shape_elem_type * ,
                                                             const shape_elem_type * ,
                                                             const long * ,
                                                             const c_dpctl.DPCTLEventVectorRef) except +
ctypedef void(*fptr_blas_gemm_2in_1out_t)(void *, void * , void * , size_t, size_t, size_t)


"""
Internal functions
"""
cdef DPNPFuncType dpnp_dtype_to_DPNPFuncType(dtype)
cdef dpnp_DPNPFuncType_to_dtype(size_t type)


"""
Logic functions
"""
cpdef dpnp_descriptor dpnp_isclose(dpnp_descriptor input1, dpnp_descriptor input2,
                                   double rtol=*, double atol=*, cpp_bool equal_nan=*)


"""
Linear algebra
"""
cpdef dpnp_descriptor dpnp_dot(dpnp_descriptor in_array1, dpnp_descriptor in_array2)
cpdef dpnp_descriptor dpnp_matmul(dpnp_descriptor in_array1, dpnp_descriptor in_array2, dpnp_descriptor out=*)


"""
Array creation routines
"""
cpdef dpnp_descriptor dpnp_copy(dpnp_descriptor x1)

"""
Mathematical functions
"""
cpdef dpnp_descriptor dpnp_fmax(dpnp_descriptor x1_obj, dpnp_descriptor x2_obj, object dtype=*,
                                   dpnp_descriptor out=*, object where=*)
cpdef dpnp_descriptor dpnp_fmin(dpnp_descriptor x1_obj, dpnp_descriptor x2_obj, object dtype=*,
                                   dpnp_descriptor out=*, object where=*)


"""
Sorting functions
"""
cpdef dpnp_descriptor dpnp_argsort(dpnp_descriptor array1)
cpdef dpnp_descriptor dpnp_sort(dpnp_descriptor array1)

"""
Trigonometric functions
"""
cpdef dpnp_descriptor dpnp_degrees(dpnp_descriptor array1)
cpdef dpnp_descriptor dpnp_radians(dpnp_descriptor array1)
cpdef dpnp_descriptor dpnp_recip(dpnp_descriptor array1)
