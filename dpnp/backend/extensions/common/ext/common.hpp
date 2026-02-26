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

#pragma once

#include <complex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <sycl/sycl.hpp>

// dpctl tensor headers
#include "utils/math_utils.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

namespace type_utils = dpctl::tensor::type_utils;
namespace type_dispatch = dpctl::tensor::type_dispatch;

namespace ext::common
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
    static void add(T &lhs, const T &value)
    {
        if constexpr (type_utils::is_complex_v<T>) {
            using vT = typename T::value_type;
            vT *_lhs = reinterpret_cast<vT(&)[2]>(lhs);
            const vT *_val = reinterpret_cast<const vT(&)[2]>(value);

            AtomicOp<vT, Order, Scope>::add(_lhs[0], _val[0]);
            AtomicOp<vT, Order, Scope>::add(_lhs[1], _val[1]);
        }
        else {
            sycl::atomic_ref<T, Order, Scope> lh(lhs);
            lh += value;
        }
    }
};

template <typename T>
struct IsNan
{
    static bool isnan(const T &v)
    {
        if constexpr (type_utils::is_complex_v<T>) {
            using vT = typename T::value_type;

            const vT real1 = std::real(v);
            const vT imag1 = std::imag(v);

            return IsNan<vT>::isnan(real1) || IsNan<vT>::isnan(imag1);
        }
        else if constexpr (std::is_floating_point_v<T> ||
                           std::is_same_v<T, sycl::half>) {
            return sycl::isnan(v);
        }

        return false;
    }
};

template <typename T>
struct Less
{
    bool operator()(const T &lhs, const T &rhs) const
    {
        if constexpr (type_utils::is_complex_v<T>) {
            return IsNan<T>::isnan(rhs) ||
                   dpctl::tensor::math_utils::less_complex(lhs, rhs);
        }
        else {
            return IsNan<T>::isnan(rhs) || std::less{}(lhs, rhs);
        }
    }
};

template <typename T, bool hasValueType>
struct value_type_of_impl;

template <typename T>
struct value_type_of_impl<T, false>
{
    using type = T;
};

template <typename T>
struct value_type_of_impl<T, true>
{
    using type = typename T::value_type;
};

template <typename T>
using value_type_of = value_type_of_impl<T, type_utils::is_complex_v<T>>;

template <typename T>
using value_type_of_t = typename value_type_of<T>::type;

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

template <typename T>
inline size_t get_local_mem_size_in_items(const sycl::queue &queue)
{
    return get_local_mem_size_in_items<T>(queue.get_device());
}

template <typename T>
inline size_t get_local_mem_size_in_items(const sycl::queue &queue,
                                          size_t reserve)
{
    return get_local_mem_size_in_items<T>(queue.get_device(), reserve);
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

// This function is a copy from dpctl because it is not available in the public
// headers of dpctl.
pybind11::dtype dtype_from_typenum(int dst_typenum);

template <typename dispatchT,
          template <typename fnT, typename T>
          typename factoryT,
          int _num_types = type_dispatch::num_types>
inline void init_dispatch_vector(dispatchT dispatch_vector[])
{
    type_dispatch::DispatchVectorBuilder<dispatchT, factoryT, _num_types> dvb;
    dvb.populate_dispatch_vector(dispatch_vector);
}

template <typename dispatchT,
          template <typename fnT, typename D, typename S>
          typename factoryT,
          int _num_types = type_dispatch::num_types>
inline void init_dispatch_table(dispatchT dispatch_table[][_num_types])
{
    type_dispatch::DispatchTableBuilder<dispatchT, factoryT, _num_types> dtb;
    dtb.populate_dispatch_table(dispatch_table);
}
} // namespace ext::common

#include "ext/details/common_internal.hpp"
