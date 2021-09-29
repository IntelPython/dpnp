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


import numpy
import dpnp.config as config
from dpnp.dpnp_algo cimport *

cimport dpnp.dpnp_utils as utils

cimport numpy


__all__ = [
    "dpnp_rng_beta",
    "dpnp_rng_binomial",
    "dpnp_rng_chisquare",
    "dpnp_rng_exponential",
    "dpnp_rng_f",
    "dpnp_rng_gamma",
    "dpnp_rng_geometric",
    "dpnp_rng_gumbel",
    "dpnp_rng_hypergeometric",
    "dpnp_rng_laplace",
    "dpnp_rng_lognormal",
    "dpnp_rng_logistic",
    "dpnp_rng_multinomial",
    "dpnp_rng_multivariate_normal",
    "dpnp_rng_negative_binomial",
    "dpnp_rng_noncentral_chisquare",
    "dpnp_rng_normal",
    "dpnp_rng_pareto",
    "dpnp_rng_poisson",
    "dpnp_rng_power",
    "dpnp_rng_randn",
    "dpnp_rng_random",
    "dpnp_rng_rayleigh",
    "dpnp_rng_shuffle",
    "dpnp_rng_srand",
    "dpnp_rng_standard_cauchy",
    "dpnp_rng_standard_exponential",
    "dpnp_rng_standard_gamma",
    "dpnp_rng_standard_normal",
    "dpnp_rng_standard_t",
    "dpnp_rng_triangular",
    "dpnp_rng_uniform",
    "dpnp_rng_vonmises",
    "dpnp_rng_wald",
    "dpnp_rng_weibull",
    "dpnp_rng_zipf"
]


ctypedef void(*fptr_dpnp_rng_beta_c_1out_t)(void * , const double, const double, const size_t) except +
ctypedef void(*fptr_dpnp_rng_binomial_c_1out_t)(void * , const int, const double, const size_t) except +
ctypedef void(*fptr_dpnp_rng_chisquare_c_1out_t)(void * , const int, const size_t) except +
ctypedef void(*fptr_dpnp_rng_exponential_c_1out_t)(void * , const double, const size_t) except +
ctypedef void(*fptr_dpnp_rng_f_c_1out_t)(void * , const double, const double, const size_t) except +
ctypedef void(*fptr_dpnp_rng_gamma_c_1out_t)(void * , const double, const double, const size_t) except +
ctypedef void(*fptr_dpnp_rng_geometric_c_1out_t)(void * , const float, const size_t) except +
ctypedef void(*fptr_dpnp_rng_gaussian_c_1out_t)(void * , const double, const double, const size_t) except +
ctypedef void(*fptr_dpnp_rng_gumbel_c_1out_t)(void * , const double, const double, const size_t) except +
ctypedef void(*fptr_dpnp_rng_hypergeometric_c_1out_t)(void * , const int, const int, const int, const size_t) except +
ctypedef void(*fptr_dpnp_rng_laplace_c_1out_t)(void * , const double, const double, const size_t) except +
ctypedef void(*fptr_dpnp_rng_logistic_c_1out_t)(void * , const double, const double, const size_t) except +
ctypedef void(*fptr_dpnp_rng_lognormal_c_1out_t)(void * , const double, const double, const size_t) except +
ctypedef void(*fptr_dpnp_rng_multinomial_c_1out_t)(void * result,
                                                   const int,
                                                   const double * ,
                                                   const size_t,
                                                   const size_t) except +
ctypedef void(*fptr_dpnp_rng_multivariate_normal_c_1out_t)(void * ,
                                                           const int,
                                                           const double * ,
                                                           const size_t,
                                                           const double * ,
                                                           const size_t,
                                                           const size_t) except +
ctypedef void(*fptr_dpnp_rng_negative_binomial_c_1out_t)(void * , const double, const double, const size_t) except +
ctypedef void(*fptr_dpnp_rng_noncentral_chisquare_c_1out_t)(void * , const double, const double, const size_t) except +
ctypedef void(*fptr_dpnp_rng_normal_c_1out_t)(void * , const double, const double, const size_t) except +
ctypedef void(*fptr_dpnp_rng_pareto_c_1out_t)(void * , const double, const size_t) except +
ctypedef void(*fptr_dpnp_rng_poisson_c_1out_t)(void * , const double, const size_t) except +
ctypedef void(*fptr_dpnp_rng_power_c_1out_t)(void * , const double, const size_t) except +
ctypedef void(*fptr_dpnp_rng_rayleigh_c_1out_t)(void * , const double, const size_t) except +
ctypedef void(*fptr_dpnp_rng_shuffle_c_1out_t)(void * ,
                                               const size_t,
                                               const size_t,
                                               const size_t,
                                               const size_t) except +
ctypedef void(*fptr_dpnp_rng_srand_c_1out_t)(const size_t) except +
ctypedef void(*fptr_dpnp_rng_standard_cauchy_c_1out_t)(void * , const size_t) except +
ctypedef void(*fptr_dpnp_rng_standard_exponential_c_1out_t)(void * , const size_t) except +
ctypedef void(*fptr_dpnp_rng_standard_gamma_c_1out_t)(void * , const double, const size_t) except +
ctypedef void(*fptr_dpnp_rng_standard_normal_c_1out_t)(void * , const size_t) except +
ctypedef void(*fptr_dpnp_rng_standard_t_c_1out_t)(void * , const double, const size_t) except +
ctypedef void(*fptr_dpnp_rng_triangular_c_1out_t)(void * ,
                                                  const double,
                                                  const double,
                                                  const double,
                                                  const size_t) except +
ctypedef void(*fptr_dpnp_rng_uniform_c_1out_t)(void * , const long, const long, const size_t) except +
ctypedef void(*fptr_dpnp_rng_vonmises_c_1out_t)(void * , const double, const double, const size_t) except +
ctypedef void(*fptr_dpnp_rng_wald_c_1out_t)(void *, const double, const double, const size_t) except +
ctypedef void(*fptr_dpnp_rng_weibull_c_1out_t)(void * , const double, const size_t) except +
ctypedef void(*fptr_dpnp_rng_zipf_c_1out_t)(void * , const double, const size_t) except +


cpdef utils.dpnp_descriptor dpnp_rng_beta(double a, double b, size):
    """
    Returns an array populated with samples from beta distribution.
    `dpnp_rng_beta` generates a matrix filled with random floats sampled from a
    univariate beta distribution.

    """

    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(numpy.float64)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_BETA, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef fptr_dpnp_rng_beta_c_1out_t func = <fptr_dpnp_rng_beta_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), a, b, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_binomial(int ntrial, double p, size):
    """
    Returns an array populated with samples from binomial distribution.
    `dpnp_rng_binomial` generates a matrix filled with random floats sampled from a
    univariate binomial distribution for a given number of independent trials and
    success probability p of a single trial.

    """

    dtype = numpy.int32
    cdef DPNPFuncType param1_type
    cdef DPNPFuncData kernel_data
    cdef fptr_dpnp_rng_binomial_c_1out_t func

    # convert string type names (array.dtype) to C enum DPNPFuncType
    param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

    # get the FPTR data structure
    kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_BINOMIAL, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    func = <fptr_dpnp_rng_binomial_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), ntrial, p, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_chisquare(int df, size):
    """
    Returns an array populated with samples from chi-square distribution.
    `dpnp_rng_chisquare` generates a matrix filled with random floats sampled from a
    univariate chi-square distribution for a given number of degrees of freedom.

    """

    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(numpy.float64)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_CHISQUARE, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef fptr_dpnp_rng_chisquare_c_1out_t func = <fptr_dpnp_rng_chisquare_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), df, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_exponential(double beta, size):
    """
    Returns an array populated with samples from exponential distribution.
    `dpnp_rng_exponential` generates a matrix filled with random floats sampled from a
    univariate exponential distribution of `beta`.

    """

    dtype = numpy.float64
    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_EXPONENTIAL, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef fptr_dpnp_rng_exponential_c_1out_t func = <fptr_dpnp_rng_exponential_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), beta, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_f(double df_num, double df_den, size):
    """
    Returns an array populated with samples from F distribution.
    `dpnp_rng_f` generates a matrix filled with random floats sampled from a
    univariate F distribution.
    """

    dtype = numpy.float64
    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_F, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef fptr_dpnp_rng_f_c_1out_t func = <fptr_dpnp_rng_f_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), df_num, df_den, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_gamma(double shape, double scale, size):
    """
    Returns an array populated with samples from gamma distribution.
    `dpnp_rng_gamma` generates a matrix filled with random floats sampled from a
    univariate gamma distribution of `shape` and `scale`.

    """

    dtype = numpy.float64
    cdef DPNPFuncType param1_type
    cdef DPNPFuncData kernel_data
    cdef fptr_dpnp_rng_gamma_c_1out_t func

    param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

    kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_GAMMA, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    func = <fptr_dpnp_rng_gamma_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), shape, scale, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_geometric(float p, size):
    """
    Returns an array populated with samples from geometric distribution.
    `dpnp_rng_geometric` generates a matrix filled with random floats sampled from a
    univariate geometric distribution for a success probability p of a single
    trial.

    """

    dtype = numpy.int32
    cdef DPNPFuncType param1_type
    cdef DPNPFuncData kernel_data
    cdef fptr_dpnp_rng_geometric_c_1out_t func

    # convert string type names (array.dtype) to C enum DPNPFuncType
    param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

    # get the FPTR data structure
    kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_GEOMETRIC, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    func = <fptr_dpnp_rng_geometric_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), p, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_gumbel(double loc, double scale, size):
    """
    Returns an array populated with samples from gumbel distribution.
    `dpnp_rng_gumbel` generates a matrix filled with random floats sampled from a
    univariate Gumbel distribution.

    """

    dtype = numpy.float64
    cdef DPNPFuncType param1_type
    cdef DPNPFuncData kernel_data
    cdef fptr_dpnp_rng_gumbel_c_1out_t func

    param1_type = dpnp_dtype_to_DPNPFuncType(dtype)
    kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_GUMBEL, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    func = <fptr_dpnp_rng_gumbel_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), loc, scale, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_hypergeometric(int l, int s, int m, size):
    """
    Returns an array populated with samples from hypergeometric distribution.
    `dpnp_rng_hypergeometric` generates a matrix filled with random floats sampled from a
    univariate hypergeometric distribution.

    """

    dtype = numpy.int32
    cdef DPNPFuncType param1_type
    cdef DPNPFuncData kernel_data
    cdef fptr_dpnp_rng_hypergeometric_c_1out_t func

    # convert string type names (array.dtype) to C enum DPNPFuncType
    param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

    # get the FPTR data structure
    kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_HYPERGEOMETRIC, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    func = <fptr_dpnp_rng_hypergeometric_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), l, s, m, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_laplace(double loc, double scale, size):
    """
    Returns an array populated with samples from laplace distribution.
    `dpnp_rng_laplace` generates a matrix filled with random floats sampled from a
    univariate laplace distribution.

    """

    dtype = numpy.float64
    cdef DPNPFuncType param1_type
    cdef DPNPFuncData kernel_data
    cdef fptr_dpnp_rng_laplace_c_1out_t func

    # convert string type names (array.dtype) to C enum DPNPFuncType
    param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

    # get the FPTR data structure
    kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_LAPLACE, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    func = <fptr_dpnp_rng_laplace_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), loc, scale, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_logistic(double loc, double scale, size):
    """
    Returns an array populated with samples from logistic distribution.
    `dpnp_rng_logistic` generates a matrix filled with random floats sampled from a
    univariate logistic distribution.

    """

    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(numpy.float64)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_LOGISTIC, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef fptr_dpnp_rng_logistic_c_1out_t func = < fptr_dpnp_rng_logistic_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), loc, scale, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_lognormal(double mean, double stddev, size):
    """
    Returns an array populated with samples from lognormal distribution.
    `dpnp_rng_lognormal` generates a matrix filled with random floats sampled from a
    univariate lognormal distribution.

    """

    dtype = numpy.float64
    cdef DPNPFuncType param1_type
    cdef DPNPFuncData kernel_data
    cdef fptr_dpnp_rng_lognormal_c_1out_t func

    # convert string type names (array.dtype) to C enum DPNPFuncType
    param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

    # get the FPTR data structure
    kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_LOGNORMAL, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    func = <fptr_dpnp_rng_lognormal_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), mean, stddev, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_multinomial(int ntrial, p, size):
    """
    Returns an array populated with samples from multinomial distribution.

    `dpnp_rng_multinomial` generates a matrix filled with random floats sampled from a
    univariate multinomial distribution for a given number of independent trials and
    probabilities of each of the ``p`` different outcome.

    """

    dtype = numpy.int32
    cdef DPNPFuncType param1_type
    cdef DPNPFuncData kernel_data
    cdef fptr_dpnp_rng_multinomial_c_1out_t func
    p = numpy.asarray(p, dtype=numpy.float64)

    cdef double * p_vector = <double * > numpy.PyArray_DATA(p)
    cdef size_t p_vector_size = len(p)

    # convert string type names (array.dtype) to C enum DPNPFuncType
    param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

    # get the FPTR data structure
    kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_MULTINOMIAL, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    func = <fptr_dpnp_rng_multinomial_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), ntrial, p_vector, p_vector_size, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_multivariate_normal(numpy.ndarray mean, numpy.ndarray cov, size):
    """
    Returns an array populated with samples from multivariate normal distribution.
    `dpnp_rng_multivariate_normal` generates a matrix filled with random floats sampled from a
    multivariate normal distribution.

    """

    dtype = numpy.float64
    cdef int dimen
    cdef double * mean_vector
    cdef double * cov_vector
    cdef size_t mean_vector_size
    cdef size_t cov_vector_size

    cdef DPNPFuncType param1_type
    cdef DPNPFuncData kernel_data
    cdef fptr_dpnp_rng_multivariate_normal_c_1out_t func

    # mean and cov expected numpy.ndarray in C order (row major)
    mean_vector = <double * > numpy.PyArray_DATA(mean)
    cov_vector = <double * > numpy.PyArray_DATA(cov)

    mean_vector_size = mean.size
    cov_vector_size = cov.size

    # convert string type names (array.dtype) to C enum DPNPFuncType
    param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

    # get the FPTR data structure
    kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_MULTIVARIATE_NORMAL, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    dimen = len(mean)

    func = <fptr_dpnp_rng_multivariate_normal_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), dimen, mean_vector, mean_vector_size, cov_vector, cov_vector_size, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_negative_binomial(double a, double p, size):
    """
    Returns an array populated with samples from negative binomial distribution.

    `negative_binomial` generates a matrix filled with random floats sampled from a
    univariate negative binomial distribution for a given parameter of the distribution
    `a` and success probability `p` of a single trial.

    """

    dtype = numpy.int32
    cdef utils.dpnp_descriptor result
    cdef DPNPFuncType param1_type
    cdef DPNPFuncData kernel_data
    cdef fptr_dpnp_rng_negative_binomial_c_1out_t func
    cdef shape_type_c result_shape

    if p == 0.0:
        filled_val = numpy.iinfo(dtype).min
        return dpnp_full(size, filled_val, dtype)
    elif p == 1.0:
        return dpnp_full(size, 0, dtype)
    else:
        # convert string type names (array.dtype) to C enum DPNPFuncType
        param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

        # get the FPTR data structure
        kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_NEGATIVE_BINOMIAL, param1_type, param1_type)

        # ceate result array with type given by FPTR data
        result_shape = utils._object_to_tuple(size)
        result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

        func = <fptr_dpnp_rng_negative_binomial_c_1out_t > kernel_data.ptr
        # call FPTR function
        func(result.get_data(), a, p, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_noncentral_chisquare(double df, double nonc, size):
    """
    Returns an array populated with samples from noncentral chisquare distribution.
    `dpnp_rng_noncentral_chisquare` generates a matrix filled with random floats sampled from a
    univariate noncentral chisquare distribution.

    """

    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(numpy.float64)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_NONCENTRAL_CHISQUARE, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef fptr_dpnp_rng_noncentral_chisquare_c_1out_t func = < fptr_dpnp_rng_noncentral_chisquare_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), df, nonc, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_normal(double loc, double scale, size):
    """
    Returns an array populated with samples from normal distribution.
    `dpnp_rng_normal` generates a matrix filled with random floats sampled from a
    normal distribution.

    """

    dtype = numpy.float64
    cdef shape_type_c result_shape
    cdef utils.dpnp_descriptor result
    cdef DPNPFuncType param1_type
    cdef DPNPFuncData kernel_data
    cdef fptr_dpnp_rng_normal_c_1out_t func

    if scale == 0.0:
        return dpnp_full(size, loc, dtype)
    else:
        # convert string type names (array.dtype) to C enum DPNPFuncType
        param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

        # get the FPTR data structure
        kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_NORMAL, param1_type, param1_type)

        # ceate result array with type given by FPTR data
        result_shape = utils._object_to_tuple(size)
        result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

        func = <fptr_dpnp_rng_normal_c_1out_t > kernel_data.ptr
        # call FPTR function
        func(result.get_data(), loc, scale, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_pareto(double alpha, size):
    """
    Returns an array populated with samples from Pareto distribution.
    `dpnp_rng_pareto` generates a matrix filled with random floats sampled from a
    univariate Pareto distribution of `alpha`.

    """

    dtype = numpy.float64
    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_PARETO, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef fptr_dpnp_rng_pareto_c_1out_t func = <fptr_dpnp_rng_pareto_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), alpha, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_poisson(double lam, size):
    """
    Returns an array populated with samples from Poisson distribution.
    `dpnp_rng_poisson` generates a matrix filled with random floats sampled from a
    univariate Poisson distribution for a given number of independent trials and
    success probability p of a single trial.

    """

    dtype = numpy.int32
    cdef shape_type_c result_shape
    cdef utils.dpnp_descriptor result
    cdef DPNPFuncType param1_type
    cdef DPNPFuncData kernel_data
    cdef fptr_dpnp_rng_poisson_c_1out_t func

    if lam == 0:
        return dpnp_full(size, 0, dtype)
    else:
        # convert string type names (array.dtype) to C enum DPNPFuncType
        param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

        # get the FPTR data structure
        kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_POISSON, param1_type, param1_type)

        # ceate result array with type given by FPTR data
        result_shape = utils._object_to_tuple(size)
        result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

        func = <fptr_dpnp_rng_poisson_c_1out_t > kernel_data.ptr
        # call FPTR function
        func(result.get_data(), lam, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_power(double alpha, size):
    """
    Returns an array populated with samples from power distribution.
    `dpnp_rng_power` generates a matrix filled with random floats sampled from a
    univariate power distribution of `alpha`.
    """

    dtype = numpy.float64
    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_POWER, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef fptr_dpnp_rng_power_c_1out_t func = <fptr_dpnp_rng_power_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), alpha, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_randn(dims):
    """
    Returns an array populated with samples from standard normal distribution.
    `dpnp_rng_randn` generates a matrix filled with random floats sampled from a
    univariate normal (Gaussian) distribution of mean 0 and variance 1.

    """
    cdef double mean = 0.0
    cdef double stddev = 1.0

    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(numpy.float64)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_GAUSSIAN, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(dims)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef fptr_dpnp_rng_gaussian_c_1out_t func = <fptr_dpnp_rng_gaussian_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), mean, stddev, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_random(dims):
    """
    Create an array of the given shape and populate it
    with random samples from a uniform distribution over [0, 1).

    """
    cdef long low = 0
    cdef long high = 1

    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(numpy.float64)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_UNIFORM, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(dims)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef fptr_dpnp_rng_uniform_c_1out_t func = <fptr_dpnp_rng_uniform_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), low, high, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_rayleigh(double scale, size):
    """
    Returns an array populated with samples from Rayleigh distribution.
    `dpnp_rayleigh` generates a matrix filled with random floats sampled from a
    univariate Rayleigh distribution of `scale`.

    """

    dtype = numpy.float64
    cdef shape_type_c result_shape
    cdef utils.dpnp_descriptor result
    cdef DPNPFuncType param1_type
    cdef DPNPFuncData kernel_data
    cdef fptr_dpnp_rng_rayleigh_c_1out_t func

    if scale == 0.0:
        return dpnp_full(size, 0.0, dtype)
    else:
        # convert string type names (array.dtype) to C enum DPNPFuncType
        param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

        # get the FPTR data structure
        kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_RAYLEIGH, param1_type, param1_type)

        # ceate result array with type given by FPTR data
        result_shape = utils._object_to_tuple(size)
        result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

        func = <fptr_dpnp_rng_rayleigh_c_1out_t > kernel_data.ptr
        # call FPTR function
        func(result.get_data(), scale, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_shuffle(utils.dpnp_descriptor x1):
    """
    Modify a sequence in-place by shuffling its contents.

    """

    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype.type)
    cdef size_t itemsize = x1.dtype.itemsize
    cdef size_t ndim = x1.ndim
    cdef size_t high_dim_size = x1.get_pyobj().size

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_SHUFFLE, param1_type, param1_type)

    cdef fptr_dpnp_rng_shuffle_c_1out_t func = < fptr_dpnp_rng_shuffle_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(x1.get_data(), itemsize, ndim, high_dim_size, x1.size)

    return x1


cpdef dpnp_rng_srand(seed):
    """
    Initialize basic random number generator.

    """

    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(numpy.float64)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_SRAND, param1_type, param1_type)

    cdef fptr_dpnp_rng_srand_c_1out_t func = < fptr_dpnp_rng_srand_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(seed)


cpdef utils.dpnp_descriptor dpnp_rng_standard_cauchy(size):
    """
    Returns an array populated with samples from standard cauchy distribution.
    `dpnp_standard_cauchy` generates a matrix filled with random floats sampled from a
    univariate standard cauchy distribution.

    """

    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(numpy.float64)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_STANDARD_CAUCHY, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef fptr_dpnp_rng_standard_cauchy_c_1out_t func = < fptr_dpnp_rng_standard_cauchy_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_standard_exponential(size):
    """
    Returns an array populated with samples from standard exponential distribution.
    `dpnp_standard_exponential` generates a matrix filled with random floats sampled from a
    standard exponential distribution.

    """

    cdef fptr_dpnp_rng_standard_exponential_c_1out_t func

    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(numpy.float64)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_STANDARD_EXPONENTIAL, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    func = < fptr_dpnp_rng_standard_exponential_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_standard_gamma(double shape, size):
    """
    Returns an array populated with samples from standard gamma distribution.
    `dpnp_standard_gamma` generates a matrix filled with random floats sampled from a
    univariate standard gamma distribution.

    """

    dtype = numpy.float64
    cdef shape_type_c result_shape
    cdef utils.dpnp_descriptor result
    cdef DPNPFuncType param1_type
    cdef DPNPFuncData kernel_data
    cdef fptr_dpnp_rng_standard_gamma_c_1out_t func

    if shape == 0.0:
        return dpnp_full(size, 0.0, dtype)
    else:
        # convert string type names (array.dtype) to C enum DPNPFuncType
        param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

        # get the FPTR data structure
        kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_STANDARD_GAMMA, param1_type, param1_type)

        # ceate result array with type given by FPTR data
        result_shape = utils._object_to_tuple(size)
        result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

        func = <fptr_dpnp_rng_standard_gamma_c_1out_t > kernel_data.ptr
        # call FPTR function
        func(result.get_data(), shape, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_standard_normal(size):
    """
    Returns an array populated with samples from standard normal(Gaussian) distribution.
    `dpnp_standard_normal` generates a matrix filled with random floats sampled from a
    univariate standard normal(Gaussian) distribution.

    """

    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(numpy.float64)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_STANDARD_NORMAL, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef fptr_dpnp_rng_standard_normal_c_1out_t func = < fptr_dpnp_rng_standard_normal_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), result.size)

    return result

cpdef utils.dpnp_descriptor dpnp_rng_standard_t(double df, size):
    """
    Returns an array populated with samples from standard t distribution.
    `dpnp_standard_t` generates a matrix filled with random floats sampled from a
    univariate standard t distribution for a given number of degrees of freedom.

    """

    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(numpy.float64)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_STANDARD_T, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef fptr_dpnp_rng_standard_t_c_1out_t func = <fptr_dpnp_rng_standard_t_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), df, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_triangular(double left, double mode, double right, size):
    """
    Returns an array populated with samples from triangular distribution.
    `dpnp_rng_triangular` generates a matrix filled with random floats sampled from a
    univariate triangular distribution.

    """

    dtype = numpy.float64
    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_TRIANGULAR, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef fptr_dpnp_rng_triangular_c_1out_t func = <fptr_dpnp_rng_triangular_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), left, mode, right, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_uniform(long low, long high, size, dtype):
    """
    Returns an array populated with samples from standard uniform distribution.
    Generates a matrix filled with random numbers sampled from a
    uniform distribution of the certain left (low) and right (high)
    bounds.

    """

    cdef shape_type_c result_shape
    cdef utils.dpnp_descriptor result
    cdef DPNPFuncType param1_type
    cdef DPNPFuncData kernel_data
    cdef fptr_dpnp_rng_uniform_c_1out_t func

    if low == high:
        return dpnp_full(size, low, dtype)
    else:
        # convert string type names (array.dtype) to C enum DPNPFuncType
        param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

        # get the FPTR data structure
        kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_UNIFORM, param1_type, param1_type)

        # ceate result array with type given by FPTR data
        result_shape = utils._object_to_tuple(size)
        result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

        func = <fptr_dpnp_rng_uniform_c_1out_t > kernel_data.ptr
        # call FPTR function
        func(result.get_data(), low, high, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_vonmises(double mu, double kappa, size):
    """
    Returns an array populated with samples from Vonmises distribution.
    `dpnp_rng_vonmises` generates a matrix filled with random floats sampled from a
    univariate Vonmises distribution.

    """

    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(numpy.float64)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_VONMISES, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef fptr_dpnp_rng_vonmises_c_1out_t func = <fptr_dpnp_rng_vonmises_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), mu, kappa, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_wald(double mean, double scale, size):
    """
    Returns an array populated with samples from Wald's distribution.
    `dpnp_rng_wald` generates a matrix filled with random floats sampled from a
    univariate Wald's distribution.

    """

    dtype = numpy.float64
    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_WALD, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef fptr_dpnp_rng_wald_c_1out_t func = <fptr_dpnp_rng_wald_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), mean, scale, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_weibull(double a, size):
    """
    Returns an array populated with samples from weibull distribution.
    `dpnp_weibull` generates a matrix filled with random floats sampled from a
    univariate weibull distribution.

    """

    dtype = numpy.float64
    cdef DPNPFuncType param1_type
    cdef DPNPFuncData kernel_data
    cdef fptr_dpnp_rng_weibull_c_1out_t func

    # convert string type names (array.dtype) to C enum DPNPFuncType
    param1_type = dpnp_dtype_to_DPNPFuncType(numpy.float64)

    # get the FPTR data structure
    kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_WEIBULL, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    func = <fptr_dpnp_rng_weibull_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), a, result.size)

    return result


cpdef utils.dpnp_descriptor dpnp_rng_zipf(double a, size):
    """
    Returns an array populated with samples from Zipf distribution.
    `dpnp_rng_zipf` generates a matrix filled with random floats sampled from a
    univariate Zipf distribution.

    """

    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(numpy.float64)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_RNG_ZIPF, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef fptr_dpnp_rng_zipf_c_1out_t func = <fptr_dpnp_rng_zipf_c_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), a, result.size)

    return result
