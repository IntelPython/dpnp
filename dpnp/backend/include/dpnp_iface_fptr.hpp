//*****************************************************************************
// Copyright (c) 2016-2024, Intel Corporation
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
    DPNP_FN_NONE,         /**< Very first element of the enumeration */
    DPNP_FN_ABSOLUTE,     /**< Used in numpy.absolute() impl  */
    DPNP_FN_ADD,          /**< Used in numpy.add() impl  */
    DPNP_FN_ALL,          /**< Used in numpy.all() impl  */
    DPNP_FN_ALLCLOSE,     /**< Used in numpy.allclose() impl  */
    DPNP_FN_ALLCLOSE_EXT, /**< Used in numpy.allclose() impl, requires extra
                             parameters */
    DPNP_FN_ANY,          /**< Used in numpy.any() impl  */
    DPNP_FN_ARANGE,       /**< Used in numpy.arange() impl  */
    DPNP_FN_ARCCOS,       /**< Used in numpy.arccos() impl  */
    DPNP_FN_ARCCOSH,      /**< Used in numpy.arccosh() impl  */
    DPNP_FN_ARCSIN,       /**< Used in numpy.arcsin() impl  */
    DPNP_FN_ARCSINH,      /**< Used in numpy.arcsinh() impl  */
    DPNP_FN_ARCTAN,       /**< Used in numpy.arctan() impl  */
    DPNP_FN_ARCTAN2,      /**< Used in numpy.arctan2() impl  */
    DPNP_FN_ARCTANH,      /**< Used in numpy.arctanh() impl  */
    DPNP_FN_ARGMAX,       /**< Used in numpy.argmax() impl  */
    DPNP_FN_ARGMIN,       /**< Used in numpy.argmin() impl  */
    DPNP_FN_ARGSORT,      /**< Used in numpy.argsort() impl  */
    DPNP_FN_AROUND,       /**< Used in numpy.around() impl  */
    DPNP_FN_ASTYPE,       /**< Used in numpy.astype() impl  */
    DPNP_FN_BITWISE_AND,  /**< Used in numpy.bitwise_and() impl  */
    DPNP_FN_BITWISE_OR,   /**< Used in numpy.bitwise_or() impl  */
    DPNP_FN_BITWISE_XOR,  /**< Used in numpy.bitwise_xor() impl  */
    DPNP_FN_CBRT,         /**< Used in numpy.cbrt() impl  */
    DPNP_FN_CEIL,         /**< Used in numpy.ceil() impl  */
    DPNP_FN_CHOLESKY,     /**< Used in numpy.linalg.cholesky() impl  */
    DPNP_FN_CONJUGATE,    /**< Used in numpy.conjugate() impl  */
    DPNP_FN_CHOOSE,       /**< Used in numpy.choose() impl  */
    DPNP_FN_CHOOSE_EXT,   /**< Used in numpy.choose() impl, requires extra
                             parameters */
    DPNP_FN_COPY,         /**< Used in numpy.copy() impl  */
    DPNP_FN_COPY_EXT, /**< Used in numpy.copy() impl, requires extra parameters
                       */
    DPNP_FN_COPYSIGN, /**< Used in numpy.copysign() impl  */
    DPNP_FN_COPYTO,   /**< Used in numpy.copyto() impl  */
    DPNP_FN_COPYTO_EXT,    /**< Used in numpy.copyto() impl, requires extra
                              parameters */
    DPNP_FN_CORRELATE,     /**< Used in numpy.correlate() impl  */
    DPNP_FN_CORRELATE_EXT, /**< Used in numpy.correlate() impl, requires extra
                              parameters */
    DPNP_FN_COS,           /**< Used in numpy.cos() impl  */
    DPNP_FN_COSH,          /**< Used in numpy.cosh() impl  */
    DPNP_FN_COUNT_NONZERO, /**< Used in numpy.count_nonzero() impl  */
    DPNP_FN_COV,           /**< Used in numpy.cov() impl  */
    DPNP_FN_CROSS,         /**< Used in numpy.cross() impl  */
    DPNP_FN_CUMPROD,       /**< Used in numpy.cumprod() impl  */
    DPNP_FN_CUMSUM,        /**< Used in numpy.cumsum() impl  */
    DPNP_FN_DEGREES,       /**< Used in numpy.degrees() impl  */
    DPNP_FN_DEGREES_EXT,   /**< Used in numpy.degrees() impl, requires extra
                              parameters */
    DPNP_FN_DET,           /**< Used in numpy.linalg.det() impl  */
    DPNP_FN_DIAG,          /**< Used in numpy.diag() impl  */
    DPNP_FN_DIAG_INDICES,  /**< Used in numpy.diag_indices() impl  */
    DPNP_FN_DIAGONAL,      /**< Used in numpy.diagonal() impl  */
    DPNP_FN_DIVIDE,        /**< Used in numpy.divide() impl  */
    DPNP_FN_DOT,           /**< Used in numpy.dot() impl  */
    DPNP_FN_DOT_EXT, /**< Used in numpy.dot() impl, requires extra parameters */
    DPNP_FN_EDIFF1D, /**< Used in numpy.ediff1d() impl  */
    DPNP_FN_EDIFF1D_EXT, /**< Used in numpy.ediff1d() impl, requires extra
                            parameters */
    DPNP_FN_EIG,         /**< Used in numpy.linalg.eig() impl  */
    DPNP_FN_EIGVALS,     /**< Used in numpy.linalg.eigvals() impl  */
    DPNP_FN_ERF,         /**< Used in scipy.special.erf impl  */
    DPNP_FN_ERF_EXT,     /**< Used in scipy.special.erf impl, requires extra
                            parameters */
    DPNP_FN_EYE,         /**< Used in numpy.eye() impl  */
    DPNP_FN_EXP,         /**< Used in numpy.exp() impl  */
    DPNP_FN_EXP2,        /**< Used in numpy.exp2() impl  */
    DPNP_FN_EXPM1,       /**< Used in numpy.expm1() impl  */
    DPNP_FN_FABS,        /**< Used in numpy.fabs() impl  */
    DPNP_FN_FABS_EXT, /**< Used in numpy.fabs() impl, requires extra parameters
                       */
    DPNP_FN_FFT_FFT,  /**< Used in numpy.fft.fft() impl  */
    DPNP_FN_FFT_FFT_EXT,   /**< Used in numpy.fft.fft() impl, requires extra
                              parameters */
    DPNP_FN_FFT_RFFT,      /**< Used in numpy.fft.rfft() impl  */
    DPNP_FN_FFT_RFFT_EXT,  /**< Used in numpy.fft.rfft() impl, requires extra
                              parameters */
    DPNP_FN_FILL_DIAGONAL, /**< Used in numpy.fill_diagonal() impl  */
    DPNP_FN_FLATTEN,       /**< Used in numpy.flatten() impl  */
    DPNP_FN_FLOOR,         /**< Used in numpy.floor() impl  */
    DPNP_FN_FLOOR_DIVIDE,  /**< Used in numpy.floor_divide() impl  */
    DPNP_FN_FMOD,          /**< Used in numpy.fmod() impl  */
    DPNP_FN_FMOD_EXT,  /**< Used in numpy.fmod() impl, requires extra parameters
                        */
    DPNP_FN_FULL,      /**< Used in numpy.full() impl  */
    DPNP_FN_FULL_LIKE, /**< Used in numpy.full_like() impl  */
    DPNP_FN_HYPOT,     /**< Used in numpy.hypot() impl  */
    DPNP_FN_IDENTITY,  /**< Used in numpy.identity() impl  */
    DPNP_FN_INITVAL, /**< Used in numpy ones, ones_like, zeros, zeros_like impls
                      */
    DPNP_FN_INITVAL_EXT, /**< Used in numpy ones, ones_like, zeros, zeros_like
                            impls  */
    DPNP_FN_INV,         /**< Used in numpy.linalg.inv() impl  */
    DPNP_FN_INVERT,      /**< Used in numpy.invert() impl  */
    DPNP_FN_KRON,        /**< Used in numpy.kron() impl  */
    DPNP_FN_LEFT_SHIFT,  /**< Used in numpy.left_shift() impl  */
    DPNP_FN_LOG,         /**< Used in numpy.log() impl  */
    DPNP_FN_LOG10,       /**< Used in numpy.log10() impl  */
    DPNP_FN_LOG2,        /**< Used in numpy.log2() impl  */
    DPNP_FN_LOG1P,       /**< Used in numpy.log1p() impl  */
    DPNP_FN_MATMUL,      /**< Used in numpy.matmul() impl  */
    DPNP_FN_MATRIX_RANK, /**< Used in numpy.linalg.matrix_rank() impl  */
    DPNP_FN_MAX,         /**< Used in numpy.max() impl  */
    DPNP_FN_MAXIMUM,     /**< Used in numpy.fmax() impl  */
    DPNP_FN_MAXIMUM_EXT, /**< Used in numpy.fmax() impl , requires extra
                            parameters */
    DPNP_FN_MEAN,        /**< Used in numpy.mean() impl  */
    DPNP_FN_MEDIAN,      /**< Used in numpy.median() impl  */
    DPNP_FN_MEDIAN_EXT,  /**< Used in numpy.median() impl, requires extra
                            parameters */
    DPNP_FN_MIN,         /**< Used in numpy.min() impl  */
    DPNP_FN_MINIMUM,     /**< Used in numpy.fmin() impl  */
    DPNP_FN_MINIMUM_EXT, /**< Used in numpy.fmax() impl, requires extra
                            parameters */
    DPNP_FN_MODF,        /**< Used in numpy.modf() impl  */
    DPNP_FN_MODF_EXT,  /**< Used in numpy.modf() impl, requires extra parameters
                        */
    DPNP_FN_MULTIPLY,  /**< Used in numpy.multiply() impl  */
    DPNP_FN_NANVAR,    /**< Used in numpy.nanvar() impl  */
    DPNP_FN_NEGATIVE,  /**< Used in numpy.negative() impl  */
    DPNP_FN_NONZERO,   /**< Used in numpy.nonzero() impl  */
    DPNP_FN_ONES,      /**< Used in numpy.ones() impl */
    DPNP_FN_ONES_LIKE, /**< Used in numpy.ones_like() impl */
    DPNP_FN_PARTITION, /**< Used in numpy.partition() impl */
    DPNP_FN_PARTITION_EXT,  /**< Used in numpy.partition() impl, requires extra
                               parameters */
    DPNP_FN_PLACE,          /**< Used in numpy.place() impl  */
    DPNP_FN_POWER,          /**< Used in numpy.power() impl  */
    DPNP_FN_PROD,           /**< Used in numpy.prod() impl  */
    DPNP_FN_PTP,            /**< Used in numpy.ptp() impl  */
    DPNP_FN_PUT,            /**< Used in numpy.put() impl  */
    DPNP_FN_PUT_ALONG_AXIS, /**< Used in numpy.put_along_axis() impl  */
    DPNP_FN_QR,             /**< Used in numpy.linalg.qr() impl  */
    DPNP_FN_RADIANS,        /**< Used in numpy.radians() impl  */
    DPNP_FN_RADIANS_EXT,    /**< Used in numpy.radians() impl, requires extra
                               parameters */
    DPNP_FN_REMAINDER,      /**< Used in numpy.remainder() impl  */
    DPNP_FN_RECIP,          /**< Used in numpy.recip() impl  */
    DPNP_FN_REPEAT,         /**< Used in numpy.repeat() impl  */
    DPNP_FN_RIGHT_SHIFT,    /**< Used in numpy.right_shift() impl  */
    DPNP_FN_RNG_BETA,       /**< Used in numpy.random.beta() impl  */
    DPNP_FN_RNG_BETA_EXT, /**< Used in numpy.random.beta() impl, requires extra
                             parameters */
    DPNP_FN_RNG_BINOMIAL, /**< Used in numpy.random.binomial() impl  */
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
    DPNP_FN_SEARCHSORTED, /**< Used in numpy.searchsorted() impl  */
    DPNP_FN_SIGN,         /**< Used in numpy.sign() impl  */
    DPNP_FN_SIN,          /**< Used in numpy.sin() impl  */
    DPNP_FN_SINH,         /**< Used in numpy.sinh() impl  */
    DPNP_FN_SORT,         /**< Used in numpy.sort() impl  */
    DPNP_FN_SQRT,         /**< Used in numpy.sqrt() impl  */
    DPNP_FN_SQRT_EXT, /**< Used in numpy.sqrt() impl, requires extra parameters
                       */
    DPNP_FN_SQUARE,   /**< Used in numpy.square() impl  */
    DPNP_FN_STD,      /**< Used in numpy.std() impl  */
    DPNP_FN_SUBTRACT, /**< Used in numpy.subtract() impl  */
    DPNP_FN_SUBTRACT_EXT, /**< Used in numpy.subtract() impl, requires extra
                             parameters */
    DPNP_FN_SUM,          /**< Used in numpy.sum() impl  */
    DPNP_FN_SVD,          /**< Used in numpy.linalg.svd() impl  */
    DPNP_FN_TAKE,         /**< Used in numpy.take() impl  */
    DPNP_FN_TAN,          /**< Used in numpy.tan() impl  */
    DPNP_FN_TANH,         /**< Used in numpy.tanh() impl  */
    DPNP_FN_TRANSPOSE,    /**< Used in numpy.transpose() impl  */
    DPNP_FN_TRACE,        /**< Used in numpy.trace() impl  */
    DPNP_FN_TRAPZ,        /**< Used in numpy.trapz() impl  */
    DPNP_FN_TRAPZ_EXT,    /**< Used in numpy.trapz() impl, requires extra
                             parameters */
    DPNP_FN_TRI,          /**< Used in numpy.tri() impl  */
    DPNP_FN_TRIL,         /**< Used in numpy.tril() impl  */
    DPNP_FN_TRIU,         /**< Used in numpy.triu() impl  */
    DPNP_FN_TRUNC,        /**< Used in numpy.trunc() impl  */
    DPNP_FN_VANDER,       /**< Used in numpy.vander() impl  */
    DPNP_FN_VAR,          /**< Used in numpy.var() impl  */
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

/**
 * DEPRECATED.
 * Experimental interface. DO NOT USE IT!
 *
 * parameter @ref type_name will be converted into var_args or char *[] with
 * extra length parameter
 */
INP_DLLEXPORT
void *get_backend_function_name(const char *func_name, const char *type_name);

#endif // BACKEND_IFACE_FPTR_H
