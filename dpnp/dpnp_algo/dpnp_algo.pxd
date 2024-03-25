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

from dpnp.dpnp_algo cimport shape_elem_type
from dpnp.dpnp_utils.dpnp_algo_utils cimport dpnp_descriptor


cdef extern from "dpnp_iface_fptr.hpp" namespace "DPNPFuncName":  # need this namespace for Enum import
    cdef enum DPNPFuncName "DPNPFuncName":
        DPNP_FN_ALLCLOSE_EXT
        DPNP_FN_CHOOSE_EXT
        DPNP_FN_COPY_EXT
        DPNP_FN_CORRELATE_EXT
        DPNP_FN_CUMPROD_EXT
        DPNP_FN_CUMSUM_EXT
        DPNP_FN_DEGREES_EXT
        DPNP_FN_DIAG_INDICES_EXT
        DPNP_FN_DIAGONAL_EXT
        DPNP_FN_EDIFF1D_EXT
        DPNP_FN_EIG_EXT
        DPNP_FN_EIGVALS_EXT
        DPNP_FN_ERF_EXT
        DPNP_FN_FABS_EXT
        DPNP_FN_FFT_FFT_EXT
        DPNP_FN_FFT_RFFT_EXT
        DPNP_FN_FILL_DIAGONAL_EXT
        DPNP_FN_FMOD_EXT
        DPNP_FN_MAXIMUM_EXT
        DPNP_FN_MEDIAN_EXT
        DPNP_FN_MINIMUM_EXT
        DPNP_FN_MODF_EXT
        DPNP_FN_PARTITION_EXT
        DPNP_FN_RADIANS_EXT
        DPNP_FN_RNG_BETA_EXT
        DPNP_FN_RNG_BINOMIAL_EXT
        DPNP_FN_RNG_CHISQUARE_EXT
        DPNP_FN_RNG_EXPONENTIAL_EXT
        DPNP_FN_RNG_F_EXT
        DPNP_FN_RNG_GAMMA_EXT
        DPNP_FN_RNG_GAUSSIAN_EXT
        DPNP_FN_RNG_GEOMETRIC_EXT
        DPNP_FN_RNG_GUMBEL_EXT
        DPNP_FN_RNG_HYPERGEOMETRIC_EXT
        DPNP_FN_RNG_LAPLACE_EXT
        DPNP_FN_RNG_LOGISTIC_EXT
        DPNP_FN_RNG_LOGNORMAL_EXT
        DPNP_FN_RNG_MULTINOMIAL_EXT
        DPNP_FN_RNG_MULTIVARIATE_NORMAL
        DPNP_FN_RNG_MULTIVARIATE_NORMAL_EXT
        DPNP_FN_RNG_NEGATIVE_BINOMIAL_EXT
        DPNP_FN_RNG_NONCENTRAL_CHISQUARE_EXT
        DPNP_FN_RNG_NORMAL_EXT
        DPNP_FN_RNG_PARETO_EXT
        DPNP_FN_RNG_POISSON_EXT
        DPNP_FN_RNG_POWER_EXT
        DPNP_FN_RNG_RAYLEIGH_EXT
        DPNP_FN_RNG_SHUFFLE_EXT
        DPNP_FN_RNG_SRAND
        DPNP_FN_RNG_SRAND_EXT
        DPNP_FN_RNG_STANDARD_CAUCHY_EXT
        DPNP_FN_RNG_STANDARD_EXPONENTIAL_EXT
        DPNP_FN_RNG_STANDARD_GAMMA_EXT
        DPNP_FN_RNG_STANDARD_NORMAL
        DPNP_FN_RNG_STANDARD_T_EXT
        DPNP_FN_RNG_TRIANGULAR_EXT
        DPNP_FN_RNG_UNIFORM_EXT
        DPNP_FN_RNG_VONMISES_EXT
        DPNP_FN_RNG_WALD_EXT
        DPNP_FN_RNG_WEIBULL_EXT
        DPNP_FN_RNG_ZIPF_EXT
        DPNP_FN_SEARCHSORTED_EXT
        DPNP_FN_TRACE_EXT
        DPNP_FN_TRAPZ_EXT

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


cdef extern from "constants.hpp":
    void dpnp_python_constants_initialize_c(void * py_none, void * py_nan)

cdef extern from "dpnp_iface.hpp":

    char * dpnp_memory_alloc_c(size_t size_in_bytes) except +
    void dpnp_memory_free_c(void * ptr)
    void dpnp_memory_memcpy_c(void * dst, const void * src, size_t size_in_bytes)
    void dpnp_rng_srand_c(size_t seed)


# C function pointer to the C library template functions
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
Trigonometric functions
"""
cpdef dpnp_descriptor dpnp_degrees(dpnp_descriptor array1)
cpdef dpnp_descriptor dpnp_radians(dpnp_descriptor array1)
