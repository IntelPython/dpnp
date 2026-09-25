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
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines utilities shared by radix sort and select kernels.
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include <sycl/sycl.hpp>

#include "utils/type_utils.hpp"

namespace dpnp::tensor::kernels::radix_utils
{

/*! @brief Computes smallest exponent such that `n <= (1 << exponent)` */
template <typename SizeT,
          std::enable_if_t<std::is_unsigned_v<SizeT> &&
                               sizeof(SizeT) == sizeof(std::uint64_t),
                           int> = 0>
std::uint32_t ceil_log2(SizeT n)
{
    // if n > 2^b, n = q * 2^b + r for q > 0 and 0 <= r < 2^b
    // floor_log2(q * 2^b + r) == floor_log2(q * 2^b) == q + floor_log2(n1)
    // ceil_log2(n) == 1 + floor_log2(n-1)
    if (n <= 1)
        return std::uint32_t{1};

    std::uint32_t exp{1};
    --n;
    if (n >= (SizeT{1} << 32)) {
        n >>= 32;
        exp += 32;
    }
    if (n >= (SizeT{1} << 16)) {
        n >>= 16;
        exp += 16;
    }
    if (n >= (SizeT{1} << 8)) {
        n >>= 8;
        exp += 8;
    }
    if (n >= (SizeT{1} << 4)) {
        n >>= 4;
        exp += 4;
    }
    if (n >= (SizeT{1} << 2)) {
        n >>= 2;
        exp += 2;
    }
    if (n >= (SizeT{1} << 1)) {
        n >>= 1;
        ++exp;
    }
    return exp;
}

//----------------------------------------------------------
// bitwise order-preserving conversions to unsigned integers
//----------------------------------------------------------

template <bool is_ascending>
bool order_preserving_cast(const bool &val)
{
    // by reference: a bool copy lets the compiler assume a 0/1 byte, and the
    // bucket index below reads only the low radix bits, see gh-2121
    const bool v = dpnp::tensor::type_utils::normalize_bool(val);
    if constexpr (is_ascending)
        return v;
    else
        return !v;
}

template <bool is_ascending,
          typename UIntT,
          std::enable_if_t<std::is_unsigned_v<UIntT>, int> = 0>
UIntT order_preserving_cast(UIntT val)
{
    if constexpr (is_ascending) {
        return val;
    }
    else {
        // bitwise invert
        return (~val);
    }
}

template <bool is_ascending,
          typename IntT,
          std::enable_if_t<std::is_integral_v<IntT> && std::is_signed_v<IntT>,
                           int> = 0>
std::make_unsigned_t<IntT> order_preserving_cast(IntT val)
{
    using UIntT = std::make_unsigned_t<IntT>;
    const UIntT uint_val = sycl::bit_cast<UIntT>(val);

    if constexpr (is_ascending) {
        // ascending_mask: 100..0
        static constexpr UIntT ascending_mask =
            (UIntT(1) << std::numeric_limits<IntT>::digits);
        return (uint_val ^ ascending_mask);
    }
    else {
        // descending_mask: 011..1
        static constexpr UIntT descending_mask =
            (std::numeric_limits<UIntT>::max() >> 1);
        return (uint_val ^ descending_mask);
    }
}

template <bool is_ascending>
std::uint16_t order_preserving_cast(sycl::half val)
{
    using UIntT = std::uint16_t;

    const UIntT uint_val = sycl::bit_cast<UIntT>(
        (sycl::isnan(val)) ? std::numeric_limits<sycl::half>::quiet_NaN()
                           : val);
    UIntT mask;

    // test the sign bit of the original value
    const bool zero_fp_sign_bit = (UIntT(0) == (uint_val >> 15));

    static constexpr UIntT zero_mask = UIntT(0x8000u);
    static constexpr UIntT nonzero_mask = UIntT(0xFFFFu);

    static constexpr UIntT inv_zero_mask = static_cast<UIntT>(~zero_mask);
    static constexpr UIntT inv_nonzero_mask = static_cast<UIntT>(~nonzero_mask);

    if constexpr (is_ascending) {
        mask = (zero_fp_sign_bit) ? zero_mask : nonzero_mask;
    }
    else {
        mask = (zero_fp_sign_bit) ? (inv_zero_mask) : (inv_nonzero_mask);
    }

    return (uint_val ^ mask);
}

template <bool is_ascending,
          typename FloatT,
          std::enable_if_t<std::is_floating_point_v<FloatT> &&
                               sizeof(FloatT) == sizeof(std::uint32_t),
                           int> = 0>
std::uint32_t order_preserving_cast(FloatT val)
{
    using UIntT = std::uint32_t;

    UIntT uint_val = sycl::bit_cast<UIntT>(
        (sycl::isnan(val)) ? std::numeric_limits<FloatT>::quiet_NaN() : val);

    UIntT mask;

    // test the sign bit of the original value
    const bool zero_fp_sign_bit = (UIntT(0) == (uint_val >> 31));

    static constexpr UIntT zero_mask = UIntT(0x80000000u);
    static constexpr UIntT nonzero_mask = UIntT(0xFFFFFFFFu);

    if constexpr (is_ascending)
        mask = (zero_fp_sign_bit) ? zero_mask : nonzero_mask;
    else
        mask = (zero_fp_sign_bit) ? (~zero_mask) : (~nonzero_mask);

    return (uint_val ^ mask);
}

template <bool is_ascending,
          typename FloatT,
          std::enable_if_t<std::is_floating_point_v<FloatT> &&
                               sizeof(FloatT) == sizeof(std::uint64_t),
                           int> = 0>
std::uint64_t order_preserving_cast(FloatT val)
{
    using UIntT = std::uint64_t;

    UIntT uint_val = sycl::bit_cast<UIntT>(
        (sycl::isnan(val)) ? std::numeric_limits<FloatT>::quiet_NaN() : val);
    UIntT mask;

    // test the sign bit of the original value
    const bool zero_fp_sign_bit = (UIntT(0) == (uint_val >> 63));

    static constexpr UIntT zero_mask = UIntT(0x8000000000000000u);
    static constexpr UIntT nonzero_mask = UIntT(0xFFFFFFFFFFFFFFFFu);

    if constexpr (is_ascending)
        mask = (zero_fp_sign_bit) ? zero_mask : nonzero_mask;
    else
        mask = (zero_fp_sign_bit) ? (~zero_mask) : (~nonzero_mask);

    return (uint_val ^ mask);
}

//-----------------
// bucket functions
//-----------------

template <typename T>
constexpr std::size_t number_of_bits_in_type()
{
    constexpr std::size_t type_bits =
        (sizeof(T) * std::numeric_limits<unsigned char>::digits);
    return type_bits;
}

// the number of buckets (size of radix bits) in T
template <typename T>
constexpr std::uint32_t number_of_buckets_in_type(std::uint32_t radix_bits)
{
    constexpr std::size_t type_bits = number_of_bits_in_type<T>();
    return (type_bits + radix_bits - 1) / radix_bits;
}

// get bits value (bucket) in a certain radix position
template <std::uint32_t radix_mask, typename T>
std::uint32_t get_bucket_id(T val, std::uint32_t radix_offset)
{
    static_assert(std::is_unsigned_v<T>);

    return (val >> radix_offset) & T(radix_mask);
}

//--------------------------------------------------------------
// keys which are equal if and only if the values compare equal
//--------------------------------------------------------------

template <std::size_t>
struct uint_of_size;

template <>
struct uint_of_size<1>
{
    using type = std::uint8_t;
};

template <>
struct uint_of_size<2>
{
    using type = std::uint16_t;
};

template <>
struct uint_of_size<4>
{
    using type = std::uint32_t;
};

template <>
struct uint_of_size<8>
{
    using type = std::uint64_t;
};

/*! @brief Unsigned integer type of radix keys for values of type `T` */
template <typename T>
using radix_key_t = typename uint_of_size<sizeof(T)>::type;

template <typename T>
T normalize_signed_zero(T val)
{
    if constexpr (std::is_floating_point_v<T> ||
                  std::is_same_v<T, sycl::half>) {
        return (val == T(0)) ? T(0) : val;
    }
    else {
        return val;
    }
}

/*! @brief Order-preserving unsigned key of `val`, unlike
 * `order_preserving_cast` equal exactly when the values compare equal, so that
 * -0.0 and +0.0 as well as all NaNs share a key */
template <bool is_ascending, typename T>
radix_key_t<T> ordered_radix_key(const T &val)
{
    using KeyT = radix_key_t<T>;
    if constexpr (std::is_same_v<T, bool>) {
        // by reference, see order_preserving_cast for bool
        return KeyT(order_preserving_cast<is_ascending>(val));
    }
    else {
        return KeyT(
            order_preserving_cast<is_ascending>(normalize_signed_zero(val)));
    }
}

//-----------
// projections
//-----------

struct IdentityProj
{
    constexpr IdentityProj() {}

    template <typename T>
    constexpr T operator()(T val) const
    {
        return val;
    }
};

template <typename ValueT, typename IndexT>
struct ValueProj
{
    constexpr ValueProj() {}

    constexpr ValueT operator()(const std::pair<ValueT, IndexT> &pair) const
    {
        return pair.first;
    }
};

template <typename IndexT, typename ValueT, typename ProjT>
struct IndexedProj
{
    IndexedProj(const ValueT *arg_ptr) : ptr(arg_ptr), value_projector{} {}

    IndexedProj(const ValueT *arg_ptr, const ProjT &proj_op)
        : ptr(arg_ptr), value_projector(proj_op)
    {
    }

    auto operator()(IndexT i) const
    {
        // normalize the value read from memory: for bool a byte other than
        // 0x00/0x01 would otherwise order by its raw value, see gh-2121
        return value_projector(
            dpnp::tensor::type_utils::normalize_bool(ptr[i]));
    }

private:
    const ValueT *ptr;
    ProjT value_projector;
};

} // namespace dpnp::tensor::kernels::radix_utils
