//*****************************************************************************
// Copyright (c) 2023-2024, Intel Corporation
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

#pragma once

#include <type_traits>

// dpctl tensor headers
#include "utils/type_dispatch.hpp"

// dpctl namespace for operations with types
namespace dpctl_td_ns = dpctl::tensor::type_dispatch;

namespace dpnp::extensions::lapack::types
{
/**
 * @brief A factory to define pairs of supported types for which
 * MKL LAPACK library provides support in oneapi::mkl::lapack::geqrf_batch<T>
 * function.
 *
 * @tparam T Type of array containing the input matrices to be QR factorized in
 * batch mode. Upon execution, each matrix in the batch is transformed to output
 * arrays representing their respective orthogonal matrix Q and upper triangular
 * matrix R.
 */
template <typename T>
struct GeqrfBatchTypePairSupportFactory
{
    static constexpr bool is_defined = std::disjunction<
        dpctl_td_ns::TypePairDefinedEntry<T, double, T, double>,
        dpctl_td_ns::TypePairDefinedEntry<T, float, T, float>,
        dpctl_td_ns::TypePairDefinedEntry<T,
                                          std::complex<float>,
                                          T,
                                          std::complex<float>>,
        dpctl_td_ns::TypePairDefinedEntry<T,
                                          std::complex<double>,
                                          T,
                                          std::complex<double>>,
        // fall-through
        dpctl_td_ns::NotDefinedEntry>::is_defined;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL LAPACK library provides support in oneapi::mkl::lapack::geqrf<T>
 * function.
 *
 * @tparam T Type of array containing the input matrix to be QR factorized.
 * Upon execution, this matrix is transformed to output arrays representing
 * the orthogonal matrix Q and the upper triangular matrix R.
 */
template <typename T>
struct GeqrfTypePairSupportFactory
{
    static constexpr bool is_defined = std::disjunction<
        dpctl_td_ns::TypePairDefinedEntry<T, double, T, double>,
        dpctl_td_ns::TypePairDefinedEntry<T, float, T, float>,
        dpctl_td_ns::TypePairDefinedEntry<T,
                                          std::complex<float>,
                                          T,
                                          std::complex<float>>,
        dpctl_td_ns::TypePairDefinedEntry<T,
                                          std::complex<double>,
                                          T,
                                          std::complex<double>>,
        // fall-through
        dpctl_td_ns::NotDefinedEntry>::is_defined;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL LAPACK library provides support in oneapi::mkl::lapack::gesv<T>
 * function.
 *
 * @tparam T Type of array containing the coefficient matrix A and
 * the array of multiple dependent variables. Upon execution, the array of
 * multiple dependent variables will be overwritten with the solution.
 */
template <typename T>
struct GesvTypePairSupportFactory
{
    static constexpr bool is_defined = std::disjunction<
        dpctl_td_ns::TypePairDefinedEntry<T, float, T, float>,
        dpctl_td_ns::TypePairDefinedEntry<T, double, T, double>,
        dpctl_td_ns::TypePairDefinedEntry<T,
                                          std::complex<float>,
                                          T,
                                          std::complex<float>>,
        dpctl_td_ns::TypePairDefinedEntry<T,
                                          std::complex<double>,
                                          T,
                                          std::complex<double>>,
        // fall-through
        dpctl_td_ns::NotDefinedEntry>::is_defined;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL LAPACK library provides support in oneapi::mkl::lapack::gesvd<T, RealT>
 * function.
 *
 * @tparam T Type of array containing input matrix A and output matrices U and
 * VT of singular vectors.
 * @tparam RealT Type of output array containing singular values of A.
 */
template <typename T, typename RealT>
struct GesvdTypePairSupportFactory
{
    static constexpr bool is_defined = std::disjunction<
        dpctl_td_ns::TypePairDefinedEntry<T, float, RealT, float>,
        dpctl_td_ns::TypePairDefinedEntry<T, double, RealT, double>,
        dpctl_td_ns::TypePairDefinedEntry<T, std::complex<float>, RealT, float>,
        dpctl_td_ns::
            TypePairDefinedEntry<T, std::complex<double>, RealT, double>,
        // fall-through
        dpctl_td_ns::NotDefinedEntry>::is_defined;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL LAPACK library provides support in oneapi::mkl::lapack::getrf<T>
 * function.
 *
 * @tparam T Type of array containing input matrix,
 * as well as the output array for storing the LU factorization.
 */
template <typename T>
struct GetrfTypePairSupportFactory
{
    static constexpr bool is_defined = std::disjunction<
        dpctl_td_ns::TypePairDefinedEntry<T, double, T, double>,
        dpctl_td_ns::TypePairDefinedEntry<T, float, T, float>,
        dpctl_td_ns::TypePairDefinedEntry<T,
                                          std::complex<float>,
                                          T,
                                          std::complex<float>>,
        dpctl_td_ns::TypePairDefinedEntry<T,
                                          std::complex<double>,
                                          T,
                                          std::complex<double>>,
        // fall-through
        dpctl_td_ns::NotDefinedEntry>::is_defined;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL LAPACK library provides support in oneapi::mkl::lapack::getrf_batch<T>
 * function.
 *
 * @tparam T Type of array containing input matrix,
 * as well as the output array for storing the LU factorization.
 */
template <typename T>
struct GetrfBatchTypePairSupportFactory
{
    static constexpr bool is_defined = std::disjunction<
        dpctl_td_ns::TypePairDefinedEntry<T, double, T, double>,
        dpctl_td_ns::TypePairDefinedEntry<T, float, T, float>,
        dpctl_td_ns::TypePairDefinedEntry<T,
                                          std::complex<float>,
                                          T,
                                          std::complex<float>>,
        dpctl_td_ns::TypePairDefinedEntry<T,
                                          std::complex<double>,
                                          T,
                                          std::complex<double>>,
        // fall-through
        dpctl_td_ns::NotDefinedEntry>::is_defined;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL LAPACK library provides support in oneapi::mkl::lapack::getri_batch<T>
 * function.
 *
 * @tparam T Type of array containing input matrix (LU-factored form),
 * as well as the output array for storing the inverse of the matrix.
 */
template <typename T>
struct GetriBatchTypePairSupportFactory
{
    static constexpr bool is_defined = std::disjunction<
        dpctl_td_ns::TypePairDefinedEntry<T, double, T, double>,
        dpctl_td_ns::TypePairDefinedEntry<T, float, T, float>,
        dpctl_td_ns::TypePairDefinedEntry<T,
                                          std::complex<float>,
                                          T,
                                          std::complex<float>>,
        dpctl_td_ns::TypePairDefinedEntry<T,
                                          std::complex<double>,
                                          T,
                                          std::complex<double>>,
        // fall-through
        dpctl_td_ns::NotDefinedEntry>::is_defined;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL LAPACK library provides support in oneapi::mkl::lapack::getrs<T>
 * function.
 *
 * @tparam T Type of array containing input matrix (LU-factored form)
 * and the array of multiple dependent variables,
 * as well as the output array for storing the solutions to a system of linear
 * equations.
 */
template <typename T>
struct GetrsTypePairSupportFactory
{
    static constexpr bool is_defined = std::disjunction<
        dpctl_td_ns::TypePairDefinedEntry<T, double, T, double>,
        dpctl_td_ns::TypePairDefinedEntry<T, float, T, float>,
        dpctl_td_ns::TypePairDefinedEntry<T,
                                          std::complex<float>,
                                          T,
                                          std::complex<float>>,
        dpctl_td_ns::TypePairDefinedEntry<T,
                                          std::complex<double>,
                                          T,
                                          std::complex<double>>,
        // fall-through
        dpctl_td_ns::NotDefinedEntry>::is_defined;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL LAPACK library provides support in oneapi::mkl::lapack::heevd<T, RealT>
 * function.
 *
 * @tparam T Type of array containing input matrix A and an output array with
 * eigenvectors.
 * @tparam RealT Type of output array containing eigenvalues of A.
 */
template <typename T, typename RealT>
struct HeevdTypePairSupportFactory
{
    static constexpr bool is_defined = std::disjunction<
        dpctl_td_ns::
            TypePairDefinedEntry<T, std::complex<double>, RealT, double>,
        dpctl_td_ns::TypePairDefinedEntry<T, std::complex<float>, RealT, float>,
        // fall-through
        dpctl_td_ns::NotDefinedEntry>::is_defined;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL LAPACK library provides support in oneapi::mkl::lapack::orgqr_batch<T>
 * function.
 *
 * @tparam T Type of array containing the matrix A,
 * each from a separate instance in the batch, from which the
 * elementary reflectors were generated (as in QR factorization).
 * Upon execution, each array in the batch is overwritten with
 * its respective orthonormal matrix Q.
 */
template <typename T>
struct OrgqrBatchTypePairSupportFactory
{
    static constexpr bool is_defined = std::disjunction<
        dpctl_td_ns::TypePairDefinedEntry<T, double, T, double>,
        dpctl_td_ns::TypePairDefinedEntry<T, float, T, float>,
        // fall-through
        dpctl_td_ns::NotDefinedEntry>::is_defined;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL LAPACK library provides support in oneapi::mkl::lapack::orgqr<T>
 * function.
 *
 * @tparam T Type of array containing the matrix A from which the
 * elementary reflectors were generated (as in QR factorization).
 * Upon execution, the array is overwritten with the orthonormal matrix Q.
 */
template <typename T>
struct OrgqrTypePairSupportFactory
{
    static constexpr bool is_defined = std::disjunction<
        dpctl_td_ns::TypePairDefinedEntry<T, double, T, double>,
        dpctl_td_ns::TypePairDefinedEntry<T, float, T, float>,
        // fall-through
        dpctl_td_ns::NotDefinedEntry>::is_defined;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL LAPACK library provides support in oneapi::mkl::lapack::potrf<T>
 * function.
 *
 * @tparam T Type of array containing input matrix,
 * as well as the output array for storing the Cholesky factor L.
 */
template <typename T>
struct PotrfTypePairSupportFactory
{
    static constexpr bool is_defined = std::disjunction<
        dpctl_td_ns::TypePairDefinedEntry<T, double, T, double>,
        dpctl_td_ns::TypePairDefinedEntry<T, float, T, float>,
        dpctl_td_ns::TypePairDefinedEntry<T,
                                          std::complex<float>,
                                          T,
                                          std::complex<float>>,
        dpctl_td_ns::TypePairDefinedEntry<T,
                                          std::complex<double>,
                                          T,
                                          std::complex<double>>,
        // fall-through
        dpctl_td_ns::NotDefinedEntry>::is_defined;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL LAPACK library provides support in oneapi::mkl::lapack::potrf<T>
 * function.
 *
 * @tparam T Type of array containing input matrices,
 * as well as the output arrays for storing the Cholesky factor L.
 */
template <typename T>
struct PotrfBatchTypePairSupportFactory
{
    static constexpr bool is_defined = std::disjunction<
        dpctl_td_ns::TypePairDefinedEntry<T, double, T, double>,
        dpctl_td_ns::TypePairDefinedEntry<T, float, T, float>,
        dpctl_td_ns::TypePairDefinedEntry<T,
                                          std::complex<float>,
                                          T,
                                          std::complex<float>>,
        dpctl_td_ns::TypePairDefinedEntry<T,
                                          std::complex<double>,
                                          T,
                                          std::complex<double>>,
        // fall-through
        dpctl_td_ns::NotDefinedEntry>::is_defined;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL LAPACK library provides support in oneapi::mkl::lapack::syevd<T>
 * function.
 *
 * @tparam T Type of array containing input matrix A and an output arrays with
 * eigenvectors and eigenvectors.
 */
template <typename T, typename RealT>
struct SyevdTypePairSupportFactory
{
    static constexpr bool is_defined = std::disjunction<
        dpctl_td_ns::TypePairDefinedEntry<T, double, RealT, double>,
        dpctl_td_ns::TypePairDefinedEntry<T, float, RealT, float>,
        // fall-through
        dpctl_td_ns::NotDefinedEntry>::is_defined;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL LAPACK library provides support in oneapi::mkl::lapack::ungqr_batch<T>
 * function.
 *
 * @tparam T Type of array containing the matrix A,
 * each from a separate instance in the batch, from which the
 * elementary reflectors were generated (as in QR factorization).
 * Upon execution, each array in the batch is overwritten with
 * its respective complex unitary matrix Q.
 */
template <typename T>
struct UngqrBatchTypePairSupportFactory
{
    static constexpr bool is_defined = std::disjunction<
        dpctl_td_ns::TypePairDefinedEntry<T,
                                          std::complex<float>,
                                          T,
                                          std::complex<float>>,
        dpctl_td_ns::TypePairDefinedEntry<T,
                                          std::complex<double>,
                                          T,
                                          std::complex<double>>,
        // fall-through
        dpctl_td_ns::NotDefinedEntry>::is_defined;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL LAPACK library provides support in oneapi::mkl::lapack::ungqr<T>
 * function.
 *
 * @tparam T Type of array containing the matrix A from which the
 * elementary reflectors were generated (as in QR factorization).
 * Upon execution, the array is overwritten with the complex unitary matrix Q.
 */
template <typename T>
struct UngqrTypePairSupportFactory
{
    static constexpr bool is_defined = std::disjunction<
        dpctl_td_ns::TypePairDefinedEntry<T,
                                          std::complex<float>,
                                          T,
                                          std::complex<float>>,
        dpctl_td_ns::TypePairDefinedEntry<T,
                                          std::complex<double>,
                                          T,
                                          std::complex<double>>,
        // fall-through
        dpctl_td_ns::NotDefinedEntry>::is_defined;
};
} // namespace dpnp::extensions::lapack::types
