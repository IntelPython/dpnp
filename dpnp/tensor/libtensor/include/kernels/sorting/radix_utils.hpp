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
/// This file defines utilities for radix-based algorithms.
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>

namespace dpnp::tensor::kernels::radix_utils
{

template <bool is_ascending, typename T, typename Enable = void>
struct RadixTypeConfig
{
};

template <bool is_ascending>
struct RadixTypeConfig<is_ascending, bool>
{
    typedef bool RadixType;

    static inline RadixType encode(bool val)
    {
        if constexpr (is_ascending)
            return val;
        else
            return !val;
    }

    static inline bool decode(RadixType val)
    {
        if constexpr (is_ascending)
            return val;
        else
            return !val;
    }
};

template <bool is_ascending, typename UIntT>
struct RadixTypeConfig<is_ascending,
                       UIntT,
                       std::enable_if_t<std::is_unsigned_v<UIntT>>>
{
    typedef UIntT RadixType;

    static inline RadixType encode(UIntT val)
    {
        if constexpr (is_ascending) {
            return val;
        }
        else {
            // bitwise invert
            return (~val);
        }
    }

    static inline UIntT decode(RadixType val)
    {
        if constexpr (is_ascending) {
            return val;
        }
        else {
            // bitwise invert
            return (~val);
        }
    }
};

template <bool is_ascending, typename IntT>
struct RadixTypeConfig<
    is_ascending,
    IntT,
    std::enable_if_t<std::is_integral_v<IntT> && std::is_signed_v<IntT>>>
{
    typedef std::make_unsigned_t<IntT> RadixType;

    static inline RadixType encode(IntT val)
    {
        // ascending_mask: 100..0
        constexpr RadixType ascending_mask =
            (RadixType(1) << std::numeric_limits<IntT>::digits);
        // descending_mask: 011..1
        constexpr RadixType descending_mask =
            (std::numeric_limits<RadixType>::max() >> 1);

        constexpr RadixType mask =
            (is_ascending) ? ascending_mask : descending_mask;
        const RadixType uint_val = sycl::bit_cast<RadixType>(val);

        return (uint_val ^ mask);
    }

    static inline IntT decode(RadixType val)
    {
        // ascending_mask: 100..0
        constexpr RadixType ascending_mask =
            (RadixType(1) << std::numeric_limits<IntT>::digits);
        // descending_mask: 011..1
        constexpr RadixType descending_mask =
            (std::numeric_limits<RadixType>::max() >> 1);

        constexpr RadixType mask =
            (is_ascending) ? ascending_mask : descending_mask;
        const IntT int_val = sycl::bit_cast<IntT>(val);

        return (int_val ^ mask);
    }
};

template <bool is_ascending>
struct RadixTypeConfig<is_ascending, sycl::half>
{
    typedef std::uint16_t RadixType;

    static inline RadixType encode(sycl::half val)
    {
        const RadixType uint_val = sycl::bit_cast<RadixType>(
            (sycl::isnan(val)) ? std::numeric_limits<sycl::half>::quiet_NaN()
                               : val);
        RadixType mask;

        // test the sign bit of the original value
        const bool zero_fp_sign_bit = (RadixType(0) == (uint_val >> 15));

        constexpr RadixType zero_mask = RadixType(0x8000u);
        constexpr RadixType nonzero_mask = RadixType(0xFFFFu);

        constexpr RadixType inv_zero_mask = static_cast<RadixType>(~zero_mask);
        constexpr RadixType inv_nonzero_mask =
            static_cast<RadixType>(~nonzero_mask);

        if constexpr (is_ascending) {
            mask = (zero_fp_sign_bit) ? zero_mask : nonzero_mask;
        }
        else {
            mask = (zero_fp_sign_bit) ? (inv_zero_mask) : (inv_nonzero_mask);
        }

        return (uint_val ^ mask);
    }

    static inline sycl::half decode(RadixType uint_val)
    {
        RadixType mask;

        // test the sign bit of the original value
        const bool zero_fp_sign_bit = (RadixType(0) == (uint_val >> 15));

        constexpr RadixType nonzero_mask = RadixType(0x8000u);
        constexpr RadixType zero_mask = RadixType(0xFFFFu);

        constexpr RadixType inv_nonzero_mask =
            static_cast<RadixType>(~nonzero_mask);
        constexpr RadixType inv_zero_mask = static_cast<RadixType>(~zero_mask);

        if constexpr (is_ascending) {
            mask = (zero_fp_sign_bit) ? zero_mask : nonzero_mask;
        }
        else {
            mask = (zero_fp_sign_bit) ? (inv_zero_mask) : (inv_nonzero_mask);
        }

        const RadixType masked = uint_val ^ mask;
        return sycl::bit_cast<sycl::half>(masked);
    }
};

template <bool is_ascending, typename FloatT>
struct RadixTypeConfig<
    is_ascending,
    FloatT,
    std::enable_if_t<std::is_floating_point_v<FloatT> &&
                     sizeof(FloatT) == sizeof(std::uint32_t)>>
{
    typedef std::uint32_t RadixType;

    static inline RadixType encode(FloatT val)
    {
        RadixType uint_val = sycl::bit_cast<RadixType>(
            (sycl::isnan(val)) ? std::numeric_limits<FloatT>::quiet_NaN()
                               : val);

        RadixType mask;

        // test the sign bit of the original value
        const bool zero_fp_sign_bit = (RadixType(0) == (uint_val >> 31));

        constexpr RadixType zero_mask = RadixType(0x80000000u);
        constexpr RadixType nonzero_mask = RadixType(0xFFFFFFFFu);

        if constexpr (is_ascending)
            mask = (zero_fp_sign_bit) ? zero_mask : nonzero_mask;
        else
            mask = (zero_fp_sign_bit) ? (~zero_mask) : (~nonzero_mask);

        return (uint_val ^ mask);
    }

    static inline FloatT decode(RadixType uint_val)
    {
        RadixType mask;

        // test the sign bit of the original value
        const bool zero_fp_sign_bit = (RadixType(0) == (uint_val >> 31));

        constexpr RadixType zero_mask = RadixType(0xFFFFFFFFu);
        constexpr RadixType nonzero_mask = RadixType(0x80000000u);

        if constexpr (is_ascending)
            mask = (zero_fp_sign_bit) ? zero_mask : nonzero_mask;
        else
            mask = (zero_fp_sign_bit) ? (~zero_mask) : (~nonzero_mask);

        const RadixType masked = uint_val ^ mask;
        return sycl::bit_cast<FloatT>(masked);
    }
};

template <bool is_ascending, typename FloatT>
struct RadixTypeConfig<
    is_ascending,
    FloatT,
    std::enable_if_t<std::is_floating_point_v<FloatT> &&
                     sizeof(FloatT) == sizeof(std::uint64_t)>>
{
    typedef std::uint64_t RadixType;

    static inline RadixType encode(FloatT val)
    {

        RadixType uint_val = sycl::bit_cast<RadixType>(
            (sycl::isnan(val)) ? std::numeric_limits<FloatT>::quiet_NaN()
                               : val);
        RadixType mask;

        // test the sign bit of the original value
        const bool zero_fp_sign_bit = (RadixType(0) == (uint_val >> 63));

        constexpr RadixType zero_mask = RadixType(0x8000000000000000u);
        constexpr RadixType nonzero_mask = RadixType(0xFFFFFFFFFFFFFFFFu);

        if constexpr (is_ascending)
            mask = (zero_fp_sign_bit) ? zero_mask : nonzero_mask;
        else
            mask = (zero_fp_sign_bit) ? (~zero_mask) : (~nonzero_mask);

        return (uint_val ^ mask);
    }

    static inline FloatT decode(RadixType uint_val)
    {
        RadixType mask;

        // test the sign bit of the original value
        const bool zero_fp_sign_bit = (RadixType(0) == (uint_val >> 63));

        constexpr RadixType zero_mask = RadixType(0xFFFFFFFFFFFFFFFFu);
        constexpr RadixType nonzero_mask = RadixType(0x8000000000000000u);

        if constexpr (is_ascending)
            mask = (zero_fp_sign_bit) ? zero_mask : nonzero_mask;
        else
            mask = (zero_fp_sign_bit) ? (~zero_mask) : (~nonzero_mask);

        const RadixType masked = uint_val ^ mask;
        return sycl::bit_cast<FloatT>(masked);
    }
};

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

template <std::uint32_t radix_mask, typename T>
T set_bucket_id(T val, T insert, std::uint32_t radix_offset)
{
    static_assert(std::is_unsigned_v<T>);

    T m = radix_mask;
    insert &= m;
    insert <<= radix_offset;
    m <<= radix_offset;
    return (val & ~m) | insert;
}

} // namespace dpnp::tensor::kernels::radix_utils
