//*****************************************************************************
// Copyright (c) 2024, Intel Corporation
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

#include <complex>
#include <functional>
#include <tuple>
#include <type_traits>

#include <sycl/sycl.hpp>

#include "utils/math_utils.hpp"

namespace statistics
{
namespace common
{

template <typename N, typename D>
constexpr auto CeilDiv(N n, D d)
{
    return (n + d - 1) / d;
}

template <typename N, typename D>
constexpr auto Align(N n, D d)
{
    return CeilDiv(n, d) * d;
}

template <typename T, sycl::memory_order Order, sycl::memory_scope Scope>
struct AtomicOp
{
    static void add(T &lhs, const T value)
    {
        sycl::atomic_ref<T, Order, Scope> lh(lhs);
        lh += value;
    }
};

template <typename T, sycl::memory_order Order, sycl::memory_scope Scope>
struct AtomicOp<std::complex<T>, Order, Scope>
{
    static void add(std::complex<T> &lhs, const std::complex<T> value)
    {
        T *_lhs = reinterpret_cast<T(&)[2]>(lhs);
        const T *_val = reinterpret_cast<const T(&)[2]>(value);
        sycl::atomic_ref<T, Order, Scope> lh0(_lhs[0]);
        lh0 += _val[0];
        sycl::atomic_ref<T, Order, Scope> lh1(_lhs[1]);
        lh1 += _val[1];
    }
};

template <typename T>
struct Less
{
    bool operator()(const T &lhs, const T &rhs) const
    {
        return std::less{}(lhs, rhs);
    }
};

template <typename T>
struct Less<std::complex<T>>
{
    bool operator()(const std::complex<T> &lhs,
                    const std::complex<T> &rhs) const
    {
        return dpctl::tensor::math_utils::less_complex(lhs, rhs);
    }
};

template <typename T>
struct IsNan
{
    static bool isnan(const T &v)
    {
        if constexpr (std::is_floating_point<T>::value) {
            return sycl::isnan(v);
        }

        return false;
    }
};

template <typename T>
struct IsNan<std::complex<T>>
{
    static bool isnan(const std::complex<T> &v)
    {
        T real1 = std::real(v);
        T imag1 = std::imag(v);
        return sycl::isnan(real1) || sycl::isnan(imag1);
    }
};

size_t get_max_local_size(const sycl::device &device);
size_t get_max_local_size(const sycl::device &device,
                          int cpu_local_size_limit,
                          int gpu_local_size_limit);

inline size_t get_max_local_size(const sycl::queue &queue)
{
    return get_max_local_size(queue.get_device());
}

inline size_t get_max_local_size(const sycl::queue &queue,
                                 int cpu_local_size_limit,
                                 int gpu_local_size_limit)
{
    return get_max_local_size(queue.get_device(), cpu_local_size_limit,
                              gpu_local_size_limit);
}

size_t get_local_mem_size_in_bytes(const sycl::device &device);
size_t get_local_mem_size_in_bytes(const sycl::device &device, size_t reserve);

inline size_t get_local_mem_size_in_bytes(const sycl::queue &queue)
{
    return get_local_mem_size_in_bytes(queue.get_device());
}

inline size_t get_local_mem_size_in_bytes(const sycl::queue &queue,
                                          size_t reserve)
{
    return get_local_mem_size_in_bytes(queue.get_device(), reserve);
}

template <typename T>
size_t get_local_mem_size_in_items(const sycl::device &device)
{
    return get_local_mem_size_in_bytes(device) / sizeof(T);
}

template <typename T>
size_t get_local_mem_size_in_items(const sycl::device &device, size_t reserve)
{
    return get_local_mem_size_in_bytes(device, sizeof(T) * reserve) / sizeof(T);
}

template <int Dims>
sycl::nd_range<Dims> make_ndrange(const sycl::range<Dims> &global_range,
                                  const sycl::range<Dims> &local_range,
                                  const sycl::range<Dims> &work_per_item)
{
    sycl::range<Dims> aligned_global_range;

    for (int i = 0; i < Dims; ++i) {
        aligned_global_range[i] =
            Align(CeilDiv(global_range[i], work_per_item[i]), local_range[i]);
    }

    return sycl::nd_range<Dims>(aligned_global_range, local_range);
}

sycl::nd_range<1>
    make_ndrange(size_t global_size, size_t local_range, size_t work_per_item);

} // namespace common
} // namespace statistics
