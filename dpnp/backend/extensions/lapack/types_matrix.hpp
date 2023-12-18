//*****************************************************************************
// Copyright (c) 2023, Intel Corporation
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

namespace dpnp
{
namespace backend
{
namespace ext
{
namespace lapack
{
namespace types
{
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
 * MKL LAPACK library provides support in oneapi::mkl::lapack::syevd<T>
 * function.
 *
 * @tparam T Type of array containing input matrix A and an output arrays with
 * eigenvectors and eigenvectors.
 */
template <typename T>
struct SyevdTypePairSupportFactory
{
    static constexpr bool is_defined = std::disjunction<
        dpctl_td_ns::TypePairDefinedEntry<T, double, T, double>,
        dpctl_td_ns::TypePairDefinedEntry<T, float, T, float>,
        // fall-through
        dpctl_td_ns::NotDefinedEntry>::is_defined;
};
} // namespace types
} // namespace lapack
} // namespace ext
} // namespace backend
} // namespace dpnp
