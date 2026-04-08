//*****************************************************************************
// Copyright (c) 2026, Intel Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// - Neither the name of the copyright holder nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
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
///
/// \file
/// This file defines functions for value casting.
//===----------------------------------------------------------------------===//

#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <sycl/sycl.hpp>

namespace dpctl::tensor::type_utils
{
template <typename T, typename = void>
struct is_complex : public std::false_type
{
};

template <typename T>
struct is_complex<
    T,
    std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, std::complex<float>> ||
                     std::is_same_v<std::remove_cv_t<T>, std::complex<double>>>>
    : public std::true_type
{
};

template <typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

template <typename dstTy, typename srcTy>
dstTy convert_impl(const srcTy &v)
{
    if constexpr (std::is_same_v<dstTy, srcTy>) {
        return v;
    }
    else if constexpr (std::is_same_v<dstTy, bool>) {
        if constexpr (is_complex_v<srcTy>) {
            // bool(complex_v) ==
            //     (complex_v.real() != 0) && (complex_v.imag() !=0)
            return (convert_impl<bool, typename srcTy::value_type>(v.real()) ||
                    convert_impl<bool, typename srcTy::value_type>(v.imag()));
        }
        else {
            return static_cast<dstTy>(v != srcTy{0});
        }
    }
    else if constexpr (std::is_same_v<srcTy, bool>) {
        // C++ interprets a byte of storage behind bool by only
        // testing is least significant bit, leading to both
        // 0x00 and 0x02 interpreted as False, while 0x01 and 0xFF
        // interpreted as True. NumPy's interpretation of underlying
        // storage is different: any bit set is interpreted as True,
        // no bits set as False, see gh-2121
        const std::uint8_t &u = sycl::bit_cast<std::uint8_t>(v);
        if constexpr (is_complex_v<dstTy>) {
            return (u == 0) ? dstTy{} : dstTy{1, 0};
        }
        else {
            return (u == 0) ? dstTy{} : dstTy{1};
        }
    }
    else if constexpr (is_complex_v<srcTy> && !is_complex_v<dstTy>) {
        // real_t(complex_v) == real_t(complex_v.real())
        return convert_impl<dstTy, typename srcTy::value_type>(v.real());
    }
    else if constexpr (!std::is_integral_v<srcTy> &&
                       !std::is_same_v<dstTy, bool> &&
                       std::is_integral_v<dstTy> && std::is_unsigned_v<dstTy>)
    {
        // first cast to signed variant, the cast to unsigned one
        using signedT = typename std::make_signed_t<dstTy>;
        return static_cast<dstTy>(convert_impl<signedT, srcTy>(v));
    }
    else {
        return static_cast<dstTy>(v);
    }
}

template <typename T>
void validate_type_for_device(const sycl::device &d)
{
    if constexpr (std::is_same_v<T, double>) {
        if (!d.has(sycl::aspect::fp64)) {
            throw std::runtime_error("Device " +
                                     d.get_info<sycl::info::device::name>() +
                                     " does not support type 'float64'");
        }
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>) {
        if (!d.has(sycl::aspect::fp64)) {
            throw std::runtime_error("Device " +
                                     d.get_info<sycl::info::device::name>() +
                                     " does not support type 'complex128'");
        }
    }
    else if constexpr (std::is_same_v<T, sycl::half>) {
        if (!d.has(sycl::aspect::fp16)) {
            throw std::runtime_error("Device " +
                                     d.get_info<sycl::info::device::name>() +
                                     " does not support type 'float16'");
        }
    }
}

template <typename T>
void validate_type_for_device(const sycl::queue &q)
{
    validate_type_for_device<T>(q.get_device());
}

template <typename Op, typename Vec, std::size_t... I>
auto vec_cast_impl(const Vec &v, std::index_sequence<I...>)
{
    return Op{v[I]...};
}

template <typename dstT,
          typename srcT,
          std::size_t N,
          typename Indices = std::make_index_sequence<N>>
auto vec_cast(const sycl::vec<srcT, N> &s)
{
    if constexpr (std::is_same_v<srcT, dstT>) {
        return s;
    }
    else {
        return vec_cast_impl<sycl::vec<dstT, N>, sycl::vec<srcT, N>>(s,
                                                                     Indices{});
    }
}
} // namespace dpctl::tensor::type_utils
