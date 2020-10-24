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

from dpnp.backend cimport *
from dpnp.dparray cimport dparray
from dpnp.dpnp_utils cimport *
import numpy
cimport numpy


__all__ = [
    "dpnp_chisquare",
    "dpnp_exponential",
    "dpnp_gamma",
    "dpnp_randn",
    "dpnp_random",
    "dpnp_srand",
    "dpnp_uniform"
]


ctypedef void(*fptr_custom_rng_chi_square_c_1out_t)(void *, int, size_t)
ctypedef void(*fptr_custom_rng_exponential_c_1out_t)(void *, double, size_t)
ctypedef void(*fptr_custom_rng_gamma_c_1out_t)(void *, double, double, size_t)
ctypedef void(*fptr_custom_rng_gaussian_c_1out_t)(void *, double, double, size_t)
ctypedef void(*fptr_custom_rng_uniform_c_1out_t)(void *, long, long, size_t)


cpdef dparray dpnp_chisquare(int df, size):
    """
    Return a random matrix with data from the chi-square distribution.

    `dpnp_chisquare` generates a matrix filled with random floats sampled from a
    univariate "chi-square" distribution for a given number of degrees of freedom.

    """

    # convert string type names (dparray.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(numpy.float64)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_CHISQUARE, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    # ceate result array with type given by FPTR data
    cdef dparray result = dparray(size, dtype=result_type)

    cdef fptr_custom_rng_chi_square_c_1out_t func = <fptr_custom_rng_chi_square_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), df, result.size)

    return result


cpdef dparray dpnp_exponential(double beta, size):
    """
    Return a random matrix with data from the "exponential" distribution.

    `dpnp_exponential` generates a matrix filled with random floats sampled from a
    univariate "exponential" distribution of `beta`.

    """

    dtype = numpy.float64
    # convert string type names (dparray.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_EXPONENTIAL, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    # ceate result array with type given by FPTR data
    cdef dparray result = dparray(size, dtype=dtype)

    cdef fptr_custom_rng_exponential_c_1out_t func = <fptr_custom_rng_exponential_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), beta, result.size)

    return result


cpdef dparray dpnp_gamma(double shape, double scale, size):
    """
    Returns an array populated with samples from gamma distribution.

    `dpnp_gamma` generates a matrix filled with random floats sampled from a
    univariate gamma distribution of `shape` and `scale`.

    """

    dtype = numpy.float64
    # convert string type names (dparray.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_GAMMA, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    # ceate result array with type given by FPTR data
    cdef dparray result = dparray(size, dtype=dtype)

    cdef fptr_custom_rng_gamma_c_1out_t func = <fptr_custom_rng_gamma_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), shape, scale, result.size)

    return result


cpdef dparray dpnp_randn(dims):
    """
    Return a random matrix with data from the "standard normal" distribution.

    `dpnp_randn` generates a matrix filled with random floats sampled from a
    univariate "normal" (Gaussian) distribution of mean 0 and variance 1.

    """
    cdef double mean = 0.0
    cdef double stddev = 1.0

    # convert string type names (dparray.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(numpy.float64)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_GAUSSIAN, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    # ceate result array with type given by FPTR data
    cdef dparray result = dparray(dims, dtype=result_type)

    cdef fptr_custom_rng_gaussian_c_1out_t func = <fptr_custom_rng_gaussian_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), mean, stddev, result.size)

    return result


cpdef dparray dpnp_random(dims):
    """
    Create an array of the given shape and populate it
    with random samples from a uniform distribution over [0, 1).

    """
    cdef long low = 0
    cdef long high = 1

    # convert string type names (dparray.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(numpy.float64)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_UNIFORM, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    # ceate result array with type given by FPTR data
    cdef dparray result = dparray(dims, dtype=result_type)

    cdef fptr_custom_rng_uniform_c_1out_t func = <fptr_custom_rng_uniform_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), low, high, result.size)

    return result


cpdef dpnp_srand(seed):
    """
    Initialize basic random number generator.

    """
    dpnp_srand_c(seed)


cpdef dparray dpnp_uniform(long low, long high, size, dtype=numpy.int32):
    """
    Return a random matrix with data from the uniform distribution.

    Generates a matrix filled with random numbers sampled from a
    uniform distribution of the certain left (low) and right (high)
    bounds.

    """
    # convert string type names (dparray.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_UNIFORM, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    # ceate result array with type given by FPTR data
    cdef dparray result = dparray(size, dtype=result_type)

    cdef fptr_custom_rng_uniform_c_1out_t func = <fptr_custom_rng_uniform_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), low, high, result.size)

    return result
