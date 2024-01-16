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

// dpctl namespace for types dispatching
namespace dpctl_td_ns = dpctl::tensor::type_dispatch;

namespace dpnp
{
namespace backend
{
namespace ext
{
namespace vm
{
namespace types
{
/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::abs<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct AbsOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<double>, double>,
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<float>, float>,
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::acos<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct AcosOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<double>>,
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<float>>,
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::acosh<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct AcoshOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<double>>,
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<float>>,
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::add<T> function.
 *
 * @tparam T Type of input vectors `a` and `b` and of result vector `y`.
 */
template <typename T>
struct AddOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::BinaryTypeMapResultEntry<T,
                                              std::complex<double>,
                                              T,
                                              std::complex<double>,
                                              std::complex<double>>,
        dpctl_td_ns::BinaryTypeMapResultEntry<T,
                                              std::complex<float>,
                                              T,
                                              std::complex<float>,
                                              std::complex<float>>,
        dpctl_td_ns::BinaryTypeMapResultEntry<T, double, T, double, double>,
        dpctl_td_ns::BinaryTypeMapResultEntry<T, float, T, float, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::asin<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct AsinOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<double>>,
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<float>>,
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::asinh<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct AsinhOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<double>>,
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<float>>,
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::atan<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct AtanOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<double>>,
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<float>>,
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::atan2<T> function.
 *
 * @tparam T Type of input vectors `a` and `b` and of result vector `y`.
 */
template <typename T>
struct Atan2OutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::BinaryTypeMapResultEntry<T, double, T, double, double>,
        dpctl_td_ns::BinaryTypeMapResultEntry<T, float, T, float, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::atanh<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct AtanhOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<double>>,
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<float>>,
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::cbrt<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct CbrtOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::ceil<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct CeilOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::conj<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct ConjOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<double>>,
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<float>>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::cos<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct CosOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<double>>,
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<float>>,
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::cosh<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct CoshOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<double>>,
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<float>>,
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::div<T> function.
 *
 * @tparam T Type of input vectors `a` and `b` and of result vector `y`.
 */
template <typename T>
struct DivOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::BinaryTypeMapResultEntry<T,
                                              std::complex<double>,
                                              T,
                                              std::complex<double>,
                                              std::complex<double>>,
        dpctl_td_ns::BinaryTypeMapResultEntry<T,
                                              std::complex<float>,
                                              T,
                                              std::complex<float>,
                                              std::complex<float>>,
        dpctl_td_ns::BinaryTypeMapResultEntry<T, double, T, double, double>,
        dpctl_td_ns::BinaryTypeMapResultEntry<T, float, T, float, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::exp<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct ExpOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<double>>,
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<float>>,
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::exp2<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct Exp2OutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::expm1<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct Expm1OutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::floor<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct FloorOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::hypot<T> function.
 *
 * @tparam T Type of input vectors `a` and `b` and of result vector `y`.
 */
template <typename T>
struct HypotOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::BinaryTypeMapResultEntry<T, double, T, double, double>,
        dpctl_td_ns::BinaryTypeMapResultEntry<T, float, T, float, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::ln<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct LnOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<double>>,
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<float>>,
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::log10<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct Log10OutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<double>>,
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<float>>,
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::log1p<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct Log1pOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::log2<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct Log2OutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::mul<T> function.
 *
 * @tparam T Type of input vectors `a` and `b` and of result vector `y`.
 */
template <typename T>
struct MulOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::BinaryTypeMapResultEntry<T,
                                              std::complex<double>,
                                              T,
                                              std::complex<double>,
                                              std::complex<double>>,
        dpctl_td_ns::BinaryTypeMapResultEntry<T,
                                              std::complex<float>,
                                              T,
                                              std::complex<float>,
                                              std::complex<float>>,
        dpctl_td_ns::BinaryTypeMapResultEntry<T, double, T, double, double>,
        dpctl_td_ns::BinaryTypeMapResultEntry<T, float, T, float, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::pow<T> function.
 *
 * @tparam T Type of input vectors `a` and `b` and of result vector `y`.
 */
template <typename T>
struct PowOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::BinaryTypeMapResultEntry<T,
                                              std::complex<double>,
                                              T,
                                              std::complex<double>,
                                              std::complex<double>>,
        dpctl_td_ns::BinaryTypeMapResultEntry<T,
                                              std::complex<float>,
                                              T,
                                              std::complex<float>,
                                              std::complex<float>>,
        dpctl_td_ns::BinaryTypeMapResultEntry<T, double, T, double, double>,
        dpctl_td_ns::BinaryTypeMapResultEntry<T, float, T, float, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::rint<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct RoundOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::sin<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct SinOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<double>>,
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<float>>,
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::sinh<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct SinhOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<double>>,
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<float>>,
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::sqr<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct SqrOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::sqrt<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct SqrtOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<double>>,
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<float>>,
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::sub<T> function.
 *
 * @tparam T Type of input vectors `a` and `b` and of result vector `y`.
 */
template <typename T>
struct SubOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::BinaryTypeMapResultEntry<T,
                                              std::complex<double>,
                                              T,
                                              std::complex<double>,
                                              std::complex<double>>,
        dpctl_td_ns::BinaryTypeMapResultEntry<T,
                                              std::complex<float>,
                                              T,
                                              std::complex<float>,
                                              std::complex<float>>,
        dpctl_td_ns::BinaryTypeMapResultEntry<T, double, T, double, double>,
        dpctl_td_ns::BinaryTypeMapResultEntry<T, float, T, float, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::tan<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct TanOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<double>>,
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<float>>,
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::tanh<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct TanhOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<double>>,
        dpctl_td_ns::TypeMapResultEntry<T, std::complex<float>>,
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::trunc<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct TruncOutputType
{
    using value_type = typename std::disjunction<
        dpctl_td_ns::TypeMapResultEntry<T, double>,
        dpctl_td_ns::TypeMapResultEntry<T, float>,
        dpctl_td_ns::DefaultResultEntry<void>>::result_type;
};

} // namespace types
} // namespace vm
} // namespace ext
} // namespace backend
} // namespace dpnp
