//*****************************************************************************
// Copyright (c) 2016-2025, Intel Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
// THE POSSIBILITY OF SUCH DAMAGE.
//*****************************************************************************

/*
 * This header file is for interface Cython with C++.
 * It should not contains any backend specific headers (like SYCL or math
 * library) because all included headers will be exposed in Cython compilation
 * procedure
 *
 * We would like to avoid backend specific things in higher level Cython
 * modules. Any backend interface functions and types should be defined here.
 *
 * Also, this file should contains documentation on functions and types
 * which are used in the interface
 */

#pragma once
#ifndef BACKEND_IFACE_FPTR_H // Cython compatibility
#define BACKEND_IFACE_FPTR_H

#include <dpnp_iface.hpp>

/**
 * @defgroup BACKEND_FUNC_PTR_API Backend C++ library runtime interface API
 * @{
 * This section describes Backend API for runtime function pointers
 * @}
 */

/**
 * @ingroup BACKEND_FUNC_PTR_API
 * @brief Function names to request via this interface
 *
 * The structure defines the parameters that are used
 * by @ref get_dpnp_function_ptr "get_dpnp_function_ptr".
 */
enum class DPNPFuncName : size_t
{
    DPNP_FN_NONE,    /**< Very first element of the enumeration */
    DPNP_FN_ERF,     /**< Used in scipy.special.erf impl  */
    DPNP_FN_ERF_EXT, /**< Used in scipy.special.erf impl, requires extra
                        parameters */
    DPNP_FN_INITVAL, /**< Used in numpy ones, ones_like, zeros, zeros_like impls
                      */
    DPNP_FN_INITVAL_EXT, /**< Used in numpy ones, ones_like, zeros, zeros_like
                            impls  */
    DPNP_FN_MODF,        /**< Used in numpy.modf() impl  */
    DPNP_FN_MODF_EXT,  /**< Used in numpy.modf() impl, requires extra parameters
                        */
    DPNP_FN_ONES,      /**< Used in numpy.ones() impl */
    DPNP_FN_ONES_LIKE, /**< Used in numpy.ones_like() impl */
    DPNP_FN_PARTITION, /**< Used in numpy.partition() impl */
    DPNP_FN_PARTITION_EXT, /**< Used in numpy.partition() impl, requires extra
                              parameters */
    DPNP_FN_RNG_BETA,      /**< Used in numpy.random.beta() impl  */
    DPNP_FN_RNG_BETA_EXT,  /**< Used in numpy.random.beta() impl, requires extra
                              parameters */
    DPNP_FN_RNG_BINOMIAL,  /**< Used in numpy.random.binomial() impl  */
    DPNP_FN_RNG_BINOMIAL_EXT,  /**< Used in numpy.random.binomial() impl,
                                  requires extra parameters */
    DPNP_FN_RNG_CHISQUARE,     /**< Used in numpy.random.chisquare() impl  */
    DPNP_FN_RNG_CHISQUARE_EXT, /**< Used in numpy.random.chisquare() impl,
                                  requires extra parameters */
    DPNP_FN_RNG_EXPONENTIAL,   /**< Used in numpy.random.exponential() impl  */
    DPNP_FN_RNG_EXPONENTIAL_EXT, /**< Used in numpy.random.exponential() impl,
                                    requires extra parameters */
    DPNP_FN_RNG_F,               /**< Used in numpy.random.f() impl  */
    DPNP_FN_RNG_F_EXT,        /**< Used in numpy.random.f() impl, requires extra
                                 parameters */
    DPNP_FN_RNG_GAMMA,        /**< Used in numpy.random.gamma() impl  */
    DPNP_FN_RNG_GAMMA_EXT,    /**< Used in numpy.random.gamma() impl, requires
                                 extra parameters */
    DPNP_FN_RNG_GAUSSIAN,     /**< Used in numpy.random.randn() impl  */
    DPNP_FN_RNG_GAUSSIAN_EXT, /**< Used in numpy.random.randn() impl, requires
                                 extra parameters */
    DPNP_FN_RNG_GEOMETRIC,    /**< Used in numpy.random.geometric() impl  */
    DPNP_FN_RNG_GEOMETRIC_EXT, /**< Used in numpy.random.geometric() impl,
                                  requires extra parameters */
    DPNP_FN_RNG_GUMBEL,        /**< Used in numpy.random.gumbel() impl  */
    DPNP_FN_RNG_GUMBEL_EXT,    /**< Used in numpy.random.gumbel() impl, requires
                                  extra parameters */
    DPNP_FN_RNG_HYPERGEOMETRIC, /**< Used in numpy.random.hypergeometric() impl
                                 */
    DPNP_FN_RNG_HYPERGEOMETRIC_EXT, /**< Used in numpy.random.hypergeometric()
                                       impl, requires extra parameters */
    DPNP_FN_RNG_LAPLACE,            /**< Used in numpy.random.laplace() impl  */
    DPNP_FN_RNG_LAPLACE_EXT,        /**< Used in numpy.random.laplace() impl  */
    DPNP_FN_RNG_LOGISTIC,      /**< Used in numpy.random.logistic() impl  */
    DPNP_FN_RNG_LOGISTIC_EXT,  /**< Used in numpy.random.logistic() impl,
                                  requires extra parameters */
    DPNP_FN_RNG_LOGNORMAL,     /**< Used in numpy.random.lognormal() impl  */
    DPNP_FN_RNG_LOGNORMAL_EXT, /**< Used in numpy.random.lognormal() impl,
                                  requires extra parameters */
    DPNP_FN_RNG_MULTINOMIAL,   /**< Used in numpy.random.multinomial() impl  */
    DPNP_FN_RNG_MULTINOMIAL_EXT, /**< Used in numpy.random.multinomial() impl,
                                    requires extra parameters */
    DPNP_FN_RNG_MULTIVARIATE_NORMAL,     /**< Used in
                                            numpy.random.multivariate_normal() impl
                                          */
    DPNP_FN_RNG_MULTIVARIATE_NORMAL_EXT, /**< Used in
                                            numpy.random.multivariate_normal()
                                            impl  */
    DPNP_FN_RNG_NEGATIVE_BINOMIAL, /**< Used in numpy.random.negative_binomial()
                                      impl  */
    DPNP_FN_RNG_NEGATIVE_BINOMIAL_EXT,    /**< Used in
                                             numpy.random.negative_binomial() impl
                                           */
    DPNP_FN_RNG_NONCENTRAL_CHISQUARE,     /**< Used in
                                             numpy.random.noncentral_chisquare()
                                             impl  */
    DPNP_FN_RNG_NONCENTRAL_CHISQUARE_EXT, /**< Used in
                                             numpy.random.noncentral_chisquare()
                                             impl  */
    DPNP_FN_RNG_NORMAL,       /**< Used in numpy.random.normal() impl  */
    DPNP_FN_RNG_NORMAL_EXT,   /**< Used in numpy.random.normal() impl, requires
                                 extra parameters */
    DPNP_FN_RNG_PARETO,       /**< Used in numpy.random.pareto() impl  */
    DPNP_FN_RNG_PARETO_EXT,   /**< Used in numpy.random.pareto() impl, requires
                                 extra parameters */
    DPNP_FN_RNG_POISSON,      /**< Used in numpy.random.poisson() impl  */
    DPNP_FN_RNG_POISSON_EXT,  /**< Used in numpy.random.poisson() impl, requires
                                 extra parameters */
    DPNP_FN_RNG_POWER,        /**< Used in numpy.random.power() impl  */
    DPNP_FN_RNG_POWER_EXT,    /**< Used in numpy.random.power() impl, requires
                                 extra parameters */
    DPNP_FN_RNG_RAYLEIGH,     /**< Used in numpy.random.rayleigh() impl  */
    DPNP_FN_RNG_RAYLEIGH_EXT, /**< Used in numpy.random.rayleigh() impl,
                                 requires extra parameters */
    DPNP_FN_RNG_SRAND,        /**< Used in numpy.random.seed() impl  */
    DPNP_FN_RNG_SRAND_EXT, /**< Used in numpy.random.seed() impl, requires extra
                              parameters */
    DPNP_FN_RNG_SHUFFLE,   /**< Used in numpy.random.shuffle() impl  */
    DPNP_FN_RNG_SHUFFLE_EXT, /**< Used in numpy.random.shuffle() impl, requires
                                extra parameters */
    DPNP_FN_RNG_STANDARD_CAUCHY,     /**< Used in numpy.random.standard_cauchy()
                                        impl  */
    DPNP_FN_RNG_STANDARD_CAUCHY_EXT, /**< Used in numpy.random.standard_cauchy()
                                        impl  */
    DPNP_FN_RNG_STANDARD_EXPONENTIAL,     /**< Used in
                                             numpy.random.standard_exponential()
                                             impl  */
    DPNP_FN_RNG_STANDARD_EXPONENTIAL_EXT, /**< Used in
                                             numpy.random.standard_exponential()
                                             impl  */
    DPNP_FN_RNG_STANDARD_GAMMA, /**< Used in numpy.random.standard_gamma() impl
                                 */
    DPNP_FN_RNG_STANDARD_GAMMA_EXT, /**< Used in numpy.random.standard_gamma()
                                       impl, requires extra parameters */
    DPNP_FN_RNG_STANDARD_NORMAL,    /**< Used in numpy.random.standard_normal()
                                       impl  */
    DPNP_FN_RNG_STANDARD_T,     /**< Used in numpy.random.standard_t() impl  */
    DPNP_FN_RNG_STANDARD_T_EXT, /**< Used in numpy.random.standard_t() impl,
                                   requires extra parameters */
    DPNP_FN_RNG_TRIANGULAR,     /**< Used in numpy.random.triangular() impl  */
    DPNP_FN_RNG_TRIANGULAR_EXT, /**< Used in numpy.random.triangular() impl,
                                   requires extra parameters */
    DPNP_FN_RNG_UNIFORM,        /**< Used in numpy.random.uniform() impl  */
    DPNP_FN_RNG_UNIFORM_EXT,  /**< Used in numpy.random.uniform() impl, requires
                                 extra parameters */
    DPNP_FN_RNG_VONMISES,     /**< Used in numpy.random.vonmises() impl  */
    DPNP_FN_RNG_VONMISES_EXT, /**< Used in numpy.random.vonmises() impl,
                                 requires extra parameters */
    DPNP_FN_RNG_WALD,         /**< Used in numpy.random.wald() impl  */
    DPNP_FN_RNG_WALD_EXT, /**< Used in numpy.random.wald() impl, requires extra
                             parameters */
    DPNP_FN_RNG_WEIBULL,  /**< Used in numpy.random.weibull() impl  */
    DPNP_FN_RNG_WEIBULL_EXT, /**< Used in numpy.random.weibull() impl, requires
                                extra parameters */
    DPNP_FN_RNG_ZIPF,        /**< Used in numpy.random.zipf() impl  */
    DPNP_FN_RNG_ZIPF_EXT, /**< Used in numpy.random.zipf() impl, requires extra
                             parameters */
    DPNP_FN_ZEROS,        /**< Used in numpy.zeros() impl */
    DPNP_FN_ZEROS_LIKE,   /**< Used in numpy.zeros_like() impl */
    DPNP_FN_LAST,         /**< The latest element of the enumeration */
};

/**
 * @ingroup BACKEND_FUNC_PTR_API
 * @brief Template types which are used in this interface
 *
 * The structure defines the types that are used
 * by @ref get_dpnp_function_ptr "get_dpnp_function_ptr".
 */
enum class DPNPFuncType : size_t
{
    DPNP_FT_NONE,    /**< Very first element of the enumeration */
    DPNP_FT_BOOL,    /**< analog of numpy.bool_ or bool */
    DPNP_FT_INT,     /**< analog of numpy.int32 or int */
    DPNP_FT_LONG,    /**< analog of numpy.int64 or long */
    DPNP_FT_FLOAT,   /**< analog of numpy.float32 or float */
    DPNP_FT_DOUBLE,  /**< analog of numpy.float32 or double */
    DPNP_FT_CMPLX64, /**< analog of numpy.complex64 or std::complex<float> */
    DPNP_FT_CMPLX128 /**< analog of numpy.complex128 or std::complex<double> */
};

/**
 * This operator is needed for compatibility with Cython 0.29 which has a bug in
 * Enum handling
 * TODO needs to be deleted in future
 */
INP_DLLEXPORT
size_t operator-(DPNPFuncType lhs, DPNPFuncType rhs);

/**
 * @ingroup BACKEND_FUNC_PTR_API
 * @brief Contains information about the C++ backend function
 *
 * The structure defines the types that are used
 * by @ref get_dpnp_function_ptr "get_dpnp_function_ptr".
 */
typedef struct DPNPFuncData
{
    DPNPFuncData(const DPNPFuncType gen_type,
                 void *gen_ptr,
                 const DPNPFuncType type_no_fp64,
                 void *ptr_no_fp64)
        : return_type(gen_type), ptr(gen_ptr),
          return_type_no_fp64(type_no_fp64), ptr_no_fp64(ptr_no_fp64)
    {
    }
    DPNPFuncData(const DPNPFuncType gen_type, void *gen_ptr)
        : DPNPFuncData(gen_type, gen_ptr, DPNPFuncType::DPNP_FT_NONE, nullptr)
    {
    }
    DPNPFuncData() : DPNPFuncData(DPNPFuncType::DPNP_FT_NONE, nullptr) {}

    DPNPFuncType return_type; /**< return type identifier which expected by the
                                 @ref ptr function */
    void *ptr;                /**< C++ backend function pointer */
    DPNPFuncType return_type_no_fp64; /**< alternative return type identifier
                                         when no fp64 support by device */
    void *ptr_no_fp64; /**< alternative C++ backend function pointer when no
                          fp64 support by device */
} DPNPFuncData_t;

/**
 * @ingroup BACKEND_API
 * @brief get runtime pointer to selected function
 *
 * Runtime pointer to the backend API function from storage map<name,
 * map<first_type, map<second_type, DPNPFuncData_t>>>
 *
 * @param [in]  name         Name of the function in storage
 * @param [in]  first_type   First type of the storage
 * @param [in]  second_type  Second type of the storage
 *
 * @return Struct @ref DPNPFuncData_t with information about the backend API
 * function.
 */
INP_DLLEXPORT
DPNPFuncData_t get_dpnp_function_ptr(
    DPNPFuncName name,
    DPNPFuncType first_type,
    DPNPFuncType second_type = DPNPFuncType::DPNP_FT_NONE);

/**
 * @ingroup BACKEND_API
 * @brief get runtime pointer to selected function
 *
 * Same interface function as @ref get_dpnp_function_ptr with a bit different
 * interface
 *
 * @param [out] result_type  Type of the result provided by the backend API
 * function
 * @param [in]  name         Name of the function in storage
 * @param [in]  first_type   First type of the storage
 * @param [in]  second_type  Second type of the storage
 *
 * @return pointer to the backend API function.
 */
INP_DLLEXPORT
void *get_dpnp_function_ptr1(
    DPNPFuncType &result_type,
    DPNPFuncName name,
    DPNPFuncType first_type,
    DPNPFuncType second_type = DPNPFuncType::DPNP_FT_NONE);

#endif // BACKEND_IFACE_FPTR_H
