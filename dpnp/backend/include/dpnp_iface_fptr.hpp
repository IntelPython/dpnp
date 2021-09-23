//*****************************************************************************
// Copyright (c) 2016-2020, Intel Corporation
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
 * It should not contains any backend specific headers (like SYCL or math library) because
 * all included headers will be exposed in Cython compilation procedure
 *
 * We would like to avoid backend specific things in higher level Cython modules.
 * Any backend interface functions and types should be defined here.
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
    DPNP_FN_NONE,                     /**< Very first element of the enumeration */
    DPNP_FN_ABSOLUTE,                 /**< Used in numpy.absolute() implementation  */
    DPNP_FN_ADD,                      /**< Used in numpy.add() implementation  */
    DPNP_FN_ALL,                      /**< Used in numpy.all() implementation  */
    DPNP_FN_ALLCLOSE,                 /**< Used in numpy.allclose() implementation  */
    DPNP_FN_ANY,                      /**< Used in numpy.any() implementation  */
    DPNP_FN_ARANGE,                   /**< Used in numpy.arange() implementation  */
    DPNP_FN_ARCCOS,                   /**< Used in numpy.arccos() implementation  */
    DPNP_FN_ARCCOSH,                  /**< Used in numpy.arccosh() implementation  */
    DPNP_FN_ARCSIN,                   /**< Used in numpy.arcsin() implementation  */
    DPNP_FN_ARCSINH,                  /**< Used in numpy.arcsinh() implementation  */
    DPNP_FN_ARCTAN,                   /**< Used in numpy.arctan() implementation  */
    DPNP_FN_ARCTAN2,                  /**< Used in numpy.arctan2() implementation  */
    DPNP_FN_ARCTANH,                  /**< Used in numpy.arctanh() implementation  */
    DPNP_FN_ARGMAX,                   /**< Used in numpy.argmax() implementation  */
    DPNP_FN_ARGMIN,                   /**< Used in numpy.argmin() implementation  */
    DPNP_FN_ARGSORT,                  /**< Used in numpy.argsort() implementation  */
    DPNP_FN_AROUND,                   /**< Used in numpy.around() implementation  */
    DPNP_FN_ASTYPE,                   /**< Used in numpy.astype() implementation  */
    DPNP_FN_BITWISE_AND,              /**< Used in numpy.bitwise_and() implementation  */
    DPNP_FN_BITWISE_OR,               /**< Used in numpy.bitwise_or() implementation  */
    DPNP_FN_BITWISE_XOR,              /**< Used in numpy.bitwise_xor() implementation  */
    DPNP_FN_CBRT,                     /**< Used in numpy.cbrt() implementation  */
    DPNP_FN_CEIL,                     /**< Used in numpy.ceil() implementation  */
    DPNP_FN_CHOLESKY,                 /**< Used in numpy.linalg.cholesky() implementation  */
    DPNP_FN_CONJIGUATE,               /**< Used in numpy.conjugate() implementation  */
    DPNP_FN_CHOOSE,                   /**< Used in numpy.choose() implementation  */
    DPNP_FN_COPY,                     /**< Used in numpy.copy() implementation  */
    DPNP_FN_COPYSIGN,                 /**< Used in numpy.copysign() implementation  */
    DPNP_FN_COPYTO,                   /**< Used in numpy.copyto() implementation  */
    DPNP_FN_CORRELATE,                /**< Used in numpy.correlate() implementation  */
    DPNP_FN_COS,                      /**< Used in numpy.cos() implementation  */
    DPNP_FN_COSH,                     /**< Used in numpy.cosh() implementation  */
    DPNP_FN_COUNT_NONZERO,            /**< Used in numpy.count_nonzero() implementation  */
    DPNP_FN_COV,                      /**< Used in numpy.cov() implementation  */
    DPNP_FN_CROSS,                    /**< Used in numpy.cross() implementation  */
    DPNP_FN_CUMPROD,                  /**< Used in numpy.cumprod() implementation  */
    DPNP_FN_CUMSUM,                   /**< Used in numpy.cumsum() implementation  */
    DPNP_FN_DEGREES,                  /**< Used in numpy.degrees() implementation  */
    DPNP_FN_DET,                      /**< Used in numpy.linalg.det() implementation  */
    DPNP_FN_DIAG,                     /**< Used in numpy.diag() implementation  */
    DPNP_FN_DIAG_INDICES,             /**< Used in numpy.diag_indices() implementation  */
    DPNP_FN_DIAGONAL,                 /**< Used in numpy.diagonal() implementation  */
    DPNP_FN_DIVIDE,                   /**< Used in numpy.divide() implementation  */
    DPNP_FN_DOT,                      /**< Used in numpy.dot() implementation  */
    DPNP_FN_EDIFF1D,                  /**< Used in numpy.ediff1d() implementation  */
    DPNP_FN_EIG,                      /**< Used in numpy.linalg.eig() implementation  */
    DPNP_FN_EIGVALS,                  /**< Used in numpy.linalg.eigvals() implementation  */
    DPNP_FN_ERF,                      /**< Used in scipy.special.erf implementation  */
    DPNP_FN_EXP,                      /**< Used in numpy.exp() implementation  */
    DPNP_FN_EXP2,                     /**< Used in numpy.exp2() implementation  */
    DPNP_FN_EXPM1,                    /**< Used in numpy.expm1() implementation  */
    DPNP_FN_FABS,                     /**< Used in numpy.fabs() implementation  */
    DPNP_FN_FFT_FFT,                  /**< Used in numpy.fft.fft() implementation  */
    DPNP_FN_FILL_DIAGONAL,            /**< Used in numpy.fill_diagonal() implementation  */
    DPNP_FN_FLATTEN,                  /**< Used in numpy.flatten() implementation  */
    DPNP_FN_FLOOR,                    /**< Used in numpy.floor() implementation  */
    DPNP_FN_FLOOR_DIVIDE,             /**< Used in numpy.floor_divide() implementation  */
    DPNP_FN_FMOD,                     /**< Used in numpy.fmod() implementation  */
    DPNP_FN_FULL,                     /**< Used in numpy.full() implementation  */
    DPNP_FN_FULL_LIKE,                /**< Used in numpy.full_like() implementation  */
    DPNP_FN_HYPOT,                    /**< Used in numpy.hypot() implementation  */
    DPNP_FN_IDENTITY,                 /**< Used in numpy.identity() implementation  */
    DPNP_FN_INITVAL,                  /**< Used in numpy ones, ones_like, zeros, zeros_like, full, full_like impl  */
    DPNP_FN_INV,                      /**< Used in numpy.linalg.inv() implementation  */
    DPNP_FN_INVERT,                   /**< Used in numpy.invert() implementation  */
    DPNP_FN_KRON,                     /**< Used in numpy.kron() implementation  */
    DPNP_FN_LEFT_SHIFT,               /**< Used in numpy.left_shift() implementation  */
    DPNP_FN_LOG,                      /**< Used in numpy.log() implementation  */
    DPNP_FN_LOG10,                    /**< Used in numpy.log10() implementation  */
    DPNP_FN_LOG2,                     /**< Used in numpy.log2() implementation  */
    DPNP_FN_LOG1P,                    /**< Used in numpy.log1p() implementation  */
    DPNP_FN_MATMUL,                   /**< Used in numpy.matmul() implementation  */
    DPNP_FN_MATRIX_RANK,              /**< Used in numpy.linalg.matrix_rank() implementation  */
    DPNP_FN_MAX,                      /**< Used in numpy.max() implementation  */
    DPNP_FN_MAXIMUM,                  /**< Used in numpy.maximum() implementation  */
    DPNP_FN_MEAN,                     /**< Used in numpy.mean() implementation  */
    DPNP_FN_MEDIAN,                   /**< Used in numpy.median() implementation  */
    DPNP_FN_MIN,                      /**< Used in numpy.min() implementation  */
    DPNP_FN_MINIMUM,                  /**< Used in numpy.minimum() implementation  */
    DPNP_FN_MODF,                     /**< Used in numpy.modf() implementation  */
    DPNP_FN_MULTIPLY,                 /**< Used in numpy.multiply() implementation  */
    DPNP_FN_NANVAR,                   /**< Used in numpy.nanvar() implementation  */
    DPNP_FN_NEGATIVE,                 /**< Used in numpy.negative() implementation  */
    DPNP_FN_NONZERO,                  /**< Used in numpy.nonzero() implementation  */
    DPNP_FN_ONES,                     /**< Used in numpy.ones() implementation */
    DPNP_FN_ONES_LIKE,                /**< Used in numpy.ones_like() implementation */
    DPNP_FN_PARTITION,                /**< Used in numpy.partition() implementation */
    DPNP_FN_PLACE,                    /**< Used in numpy.place() implementation  */
    DPNP_FN_POWER,                    /**< Used in numpy.power() implementation  */
    DPNP_FN_PROD,                     /**< Used in numpy.prod() implementation  */
    DPNP_FN_PUT,                      /**< Used in numpy.put() implementation  */
    DPNP_FN_PUT_ALONG_AXIS,           /**< Used in numpy.put_along_axis() implementation  */
    DPNP_FN_QR,                       /**< Used in numpy.linalg.qr() implementation  */
    DPNP_FN_RADIANS,                  /**< Used in numpy.radians() implementation  */
    DPNP_FN_REMAINDER,                /**< Used in numpy.remainder() implementation  */
    DPNP_FN_RECIP,                    /**< Used in numpy.recip() implementation  */
    DPNP_FN_REPEAT,                   /**< Used in numpy.repeat() implementation  */
    DPNP_FN_RIGHT_SHIFT,              /**< Used in numpy.right_shift() implementation  */
    DPNP_FN_RNG_BETA,                 /**< Used in numpy.random.beta() implementation  */
    DPNP_FN_RNG_BINOMIAL,             /**< Used in numpy.random.binomial() implementation  */
    DPNP_FN_RNG_CHISQUARE,            /**< Used in numpy.random.chisquare() implementation  */
    DPNP_FN_RNG_EXPONENTIAL,          /**< Used in numpy.random.exponential() implementation  */
    DPNP_FN_RNG_F,                    /**< Used in numpy.random.f() implementation  */
    DPNP_FN_RNG_GAMMA,                /**< Used in numpy.random.gamma() implementation  */
    DPNP_FN_RNG_GAUSSIAN,             /**< Used in numpy.random.randn() implementation  */
    DPNP_FN_RNG_GEOMETRIC,            /**< Used in numpy.random.geometric() implementation  */
    DPNP_FN_RNG_GUMBEL,               /**< Used in numpy.random.gumbel() implementation  */
    DPNP_FN_RNG_HYPERGEOMETRIC,       /**< Used in numpy.random.hypergeometric() implementation  */
    DPNP_FN_RNG_LAPLACE,              /**< Used in numpy.random.laplace() implementation  */
    DPNP_FN_RNG_LOGISTIC,             /**< Used in numpy.random.logistic() implementation  */
    DPNP_FN_RNG_LOGNORMAL,            /**< Used in numpy.random.lognormal() implementation  */
    DPNP_FN_RNG_MULTINOMIAL,          /**< Used in numpy.random.multinomial() implementation  */
    DPNP_FN_RNG_MULTIVARIATE_NORMAL,  /**< Used in numpy.random.multivariate_normal() implementation  */
    DPNP_FN_RNG_NEGATIVE_BINOMIAL,    /**< Used in numpy.random.negative_binomial() implementation  */
    DPNP_FN_RNG_NONCENTRAL_CHISQUARE, /**< Used in numpy.random.noncentral_chisquare() implementation  */
    DPNP_FN_RNG_NORMAL,               /**< Used in numpy.random.normal() implementation  */
    DPNP_FN_RNG_PARETO,               /**< Used in numpy.random.pareto() implementation  */
    DPNP_FN_RNG_POISSON,              /**< Used in numpy.random.poisson() implementation  */
    DPNP_FN_RNG_POWER,                /**< Used in numpy.random.power() implementation  */
    DPNP_FN_RNG_RAYLEIGH,             /**< Used in numpy.random.rayleigh() implementation  */
    DPNP_FN_RNG_SRAND,                /**< Used in numpy.random.seed() implementation  */
    DPNP_FN_RNG_SHUFFLE,              /**< Used in numpy.random.shuffle() implementation  */
    DPNP_FN_RNG_STANDARD_CAUCHY,      /**< Used in numpy.random.standard_cauchy() implementation  */
    DPNP_FN_RNG_STANDARD_EXPONENTIAL, /**< Used in numpy.random.standard_exponential() implementation  */
    DPNP_FN_RNG_STANDARD_GAMMA,       /**< Used in numpy.random.standard_gamma() implementation  */
    DPNP_FN_RNG_STANDARD_NORMAL,      /**< Used in numpy.random.standard_normal() implementation  */
    DPNP_FN_RNG_STANDARD_T,           /**< Used in numpy.random.standard_t() implementation  */
    DPNP_FN_RNG_TRIANGULAR,           /**< Used in numpy.random.triangular() implementation  */
    DPNP_FN_RNG_UNIFORM,              /**< Used in numpy.random.uniform() implementation  */
    DPNP_FN_RNG_VONMISES,             /**< Used in numpy.random.vonmises() implementation  */
    DPNP_FN_RNG_WALD,                 /**< Used in numpy.random.wald() implementation  */
    DPNP_FN_RNG_WEIBULL,              /**< Used in numpy.random.weibull() implementation  */
    DPNP_FN_RNG_ZIPF,                 /**< Used in numpy.random.zipf() implementation  */
    DPNP_FN_SEARCHSORTED,             /**< Used in numpy.searchsorted() implementation  */
    DPNP_FN_SIGN,                     /**< Used in numpy.sign() implementation  */
    DPNP_FN_SIN,                      /**< Used in numpy.sin() implementation  */
    DPNP_FN_SINH,                     /**< Used in numpy.sinh() implementation  */
    DPNP_FN_SORT,                     /**< Used in numpy.sort() implementation  */
    DPNP_FN_SQRT,                     /**< Used in numpy.sqrt() implementation  */
    DPNP_FN_SQUARE,                   /**< Used in numpy.square() implementation  */
    DPNP_FN_STD,                      /**< Used in numpy.std() implementation  */
    DPNP_FN_SUBTRACT,                 /**< Used in numpy.subtract() implementation  */
    DPNP_FN_SUM,                      /**< Used in numpy.sum() implementation  */
    DPNP_FN_SVD,                      /**< Used in numpy.linalg.svd() implementation  */
    DPNP_FN_TAKE,                     /**< Used in numpy.take() implementation  */
    DPNP_FN_TAN,                      /**< Used in numpy.tan() implementation  */
    DPNP_FN_TANH,                     /**< Used in numpy.tanh() implementation  */
    DPNP_FN_TRANSPOSE,                /**< Used in numpy.transpose() implementation  */
    DPNP_FN_TRACE,                    /**< Used in numpy.trace() implementation  */
    DPNP_FN_TRAPZ,                    /**< Used in numpy.trapz() implementation  */
    DPNP_FN_TRI,                      /**< Used in numpy.tri() implementation  */
    DPNP_FN_TRIL,                     /**< Used in numpy.tril() implementation  */
    DPNP_FN_TRIU,                     /**< Used in numpy.triu() implementation  */
    DPNP_FN_TRUNC,                    /**< Used in numpy.trunc() implementation  */
    DPNP_FN_VANDER,                   /**< Used in numpy.vander() implementation  */
    DPNP_FN_VAR,                      /**< Used in numpy.var() implementation  */
    DPNP_FN_ZEROS,                    /**< Used in numpy.zeros() implementation */
    DPNP_FN_ZEROS_LIKE,               /**< Used in numpy.zeros_like() implementation */
    DPNP_FN_LAST                      /**< The latest element of the enumeration */
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
    DPNP_FT_NONE,     /**< Very first element of the enumeration */
    DPNP_FT_INT,      /**< analog of numpy.int32 or int */
    DPNP_FT_LONG,     /**< analog of numpy.int64 or long */
    DPNP_FT_FLOAT,    /**< analog of numpy.float32 or float */
    DPNP_FT_DOUBLE,   /**< analog of numpy.float32 or double */
    DPNP_FT_CMPLX64,  /**< analog of numpy.complex64 or std::complex<float> */
    DPNP_FT_CMPLX128, /**< analog of numpy.complex128 or std::complex<double> */
    DPNP_FT_BOOL      /**< analog of numpy.bool or numpy.bool_ or bool */
};

/**
 * This operator is needed for compatibility with Cython 0.29 which has a bug in Enum handling
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
    DPNPFuncType return_type; /**< return type identifier which expected by the @ref ptr function */
    void* ptr;                /**< C++ backend function pointer */
} DPNPFuncData_t;

/**
 * @ingroup BACKEND_API
 * @brief get runtime pointer to selected function
 *
 * Runtime pointer to the backend API function from storage map<name, map<first_type, map<second_type, DPNPFuncData_t>>>
 *
 * @param [in]  name         Name of the function in storage
 * @param [in]  first_type   First type of the storage
 * @param [in]  second_type  Second type of the storage
 *
 * @return Struct @ref DPNPFuncData_t with information about the backend API function.
 */
INP_DLLEXPORT
DPNPFuncData_t get_dpnp_function_ptr(DPNPFuncName name,
                                     DPNPFuncType first_type,
                                     DPNPFuncType second_type = DPNPFuncType::DPNP_FT_NONE);

/**
 * @ingroup BACKEND_API
 * @brief get runtime pointer to selected function
 *
 * Same interface function as @ref get_dpnp_function_ptr with a bit diffrent interface
 *
 * @param [out] result_type  Type of the result provided by the backend API function
 * @param [in]  name         Name of the function in storage
 * @param [in]  first_type   First type of the storage
 * @param [in]  second_type  Second type of the storage
 *
 * @return pointer to the backend API function.
 */
INP_DLLEXPORT
void* get_dpnp_function_ptr1(DPNPFuncType& result_type,
                             DPNPFuncName name,
                             DPNPFuncType first_type,
                             DPNPFuncType second_type = DPNPFuncType::DPNP_FT_NONE);

/**
 * DEPRECATED.
 * Experimental interface. DO NOT USE IT!
 *
 * parameter @ref type_name will be converted into var_args or char *[] with extra length parameter
 */
INP_DLLEXPORT
void* get_backend_function_name(const char* func_name, const char* type_name);

#endif // BACKEND_IFACE_FPTR_H
