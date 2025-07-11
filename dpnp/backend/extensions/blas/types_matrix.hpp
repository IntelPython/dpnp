//*****************************************************************************
// Copyright (c) 2024-2025, Intel Corporation
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

namespace dpnp::extensions::blas::types
{
/**
 * @brief A factory to define pairs of supported types for which
 * MKL BLAS library provides support in oneapi::mkl::blas::dot<T>
 * function.
 *
 * @tparam T Type of input and output arrays.
 */
template <typename T>
struct DotTypePairSupportFactory
{
    static constexpr bool is_defined = std::disjunction<
        dpctl_td_ns::TypePairDefinedEntry<T, float, T, float>,
        dpctl_td_ns::TypePairDefinedEntry<T, double, T, double>,
        // fall-through
        dpctl_td_ns::NotDefinedEntry>::is_defined;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL BLAS library provides support in oneapi::mkl::blas::dotc<T>
 * function.
 *
 * @tparam T Type of input and output arrays.
 */
template <typename T>
struct DotcTypePairSupportFactory
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
 * MKL BLAS library provides support in oneapi::mkl::blas::dotu<T>
 * function.
 *
 * @tparam T Type of input and output arrays.
 */
template <typename T>
struct DotuTypePairSupportFactory
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
 * MKL BLAS library provides support in oneapi::mkl::blas::gemm<Tab, Tc>
 * function.
 *
 * @tparam Tab Type of arrays containing input matrices A and B.
 * @tparam Tc Type of array containing output matrix C.
 */
template <typename Tab, typename Tc>
struct GemmTypePairSupportFactory
{
    static constexpr bool is_defined = std::disjunction<
#if !defined(USE_ONEMATH)
        dpctl_td_ns::TypePairDefinedEntry<Tab, std::int8_t, Tc, std::int32_t>,
        dpctl_td_ns::TypePairDefinedEntry<Tab, std::int8_t, Tc, float>,
#endif // USE_ONEMATH
        dpctl_td_ns::TypePairDefinedEntry<Tab, sycl::half, Tc, float>,
        dpctl_td_ns::TypePairDefinedEntry<Tab, sycl::half, Tc, sycl::half>,
        dpctl_td_ns::TypePairDefinedEntry<Tab, float, Tc, float>,
        dpctl_td_ns::TypePairDefinedEntry<Tab, double, Tc, double>,
        dpctl_td_ns::TypePairDefinedEntry<Tab,
                                          std::complex<float>,
                                          Tc,
                                          std::complex<float>>,
        dpctl_td_ns::TypePairDefinedEntry<Tab,
                                          std::complex<double>,
                                          Tc,
                                          std::complex<double>>,
        // fall-through
        dpctl_td_ns::NotDefinedEntry>::is_defined;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL BLAS library provides support in
 * oneapi::mkl::blas::gemm_batch<Tab, Tc> function.
 *
 * @tparam Tab Type of arrays containing input matrices A and B.
 * @tparam Tc Type of array containing output matrix C.
 */
template <typename Tab, typename Tc>
struct GemmBatchTypePairSupportFactory
{
    static constexpr bool is_defined = std::disjunction<
#if !defined(USE_ONEMATH)
        dpctl_td_ns::TypePairDefinedEntry<Tab, std::int8_t, Tc, std::int32_t>,
        dpctl_td_ns::TypePairDefinedEntry<Tab, std::int8_t, Tc, float>,
#endif // USE_ONEMATH
        dpctl_td_ns::TypePairDefinedEntry<Tab, sycl::half, Tc, float>,
        dpctl_td_ns::TypePairDefinedEntry<Tab, sycl::half, Tc, sycl::half>,
        dpctl_td_ns::TypePairDefinedEntry<Tab, float, Tc, float>,
        dpctl_td_ns::TypePairDefinedEntry<Tab, double, Tc, double>,
        dpctl_td_ns::TypePairDefinedEntry<Tab,
                                          std::complex<float>,
                                          Tc,
                                          std::complex<float>>,
        dpctl_td_ns::TypePairDefinedEntry<Tab,
                                          std::complex<double>,
                                          Tc,
                                          std::complex<double>>,
        // fall-through
        dpctl_td_ns::NotDefinedEntry>::is_defined;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL BLAS library provides support in oneapi::mkl::blas::gemv<T>
 * function.
 *
 * @tparam T Type of input and output arrays.
 */
template <typename T>
struct GemvTypePairSupportFactory
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
 * MKL BLAS library provides support in oneapi::mkl::blas::syrk<T>
 * function.
 *
 * @tparam T Type of input and output arrays.
 */
template <typename T>
struct SyrkTypePairSupportFactory
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
} // namespace dpnp::extensions::blas::types
