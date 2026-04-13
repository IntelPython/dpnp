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
//===---------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpctl.tensor._tensor_reductions_impl
/// extension.
//===---------------------------------------------------------------------===//

#pragma once
#include <type_traits>

#include <sycl/sycl.hpp>

#include "utils/type_utils.hpp"

namespace dpctl::tensor::py_internal::atomic_support
{

typedef bool (*atomic_support_fn_ptr_t)(const sycl::queue &, sycl::usm::alloc);

/*! @brief Function which returns a constant value for atomic support */
template <bool return_value>
bool fixed_decision(const sycl::queue &, sycl::usm::alloc)
{
    return return_value;
}

/*! @brief Template for querying atomic support for a type on a device */
template <typename T>
bool check_atomic_support(const sycl::queue &exec_q,
                          sycl::usm::alloc usm_alloc_type)
{
    static constexpr bool atomic32 = (sizeof(T) == 4);
    static constexpr bool atomic64 = (sizeof(T) == 8);
    using dpctl::tensor::type_utils::is_complex;
    if constexpr ((!atomic32 && !atomic64) || is_complex<T>::value) {
        return fixed_decision<false>(exec_q, usm_alloc_type);
    }
    else {
        bool supports_atomics = false;
        const sycl::device &dev = exec_q.get_device();
        if constexpr (atomic64) {
            if (!dev.has(sycl::aspect::atomic64)) {
                return false;
            }
        }
        switch (usm_alloc_type) {
        case sycl::usm::alloc::shared:
            supports_atomics =
                dev.has(sycl::aspect::usm_atomic_shared_allocations);
            break;
        case sycl::usm::alloc::host:
            supports_atomics =
                dev.has(sycl::aspect::usm_atomic_host_allocations);
            break;
        case sycl::usm::alloc::device:
            supports_atomics = true;
            break;
        default:
            supports_atomics = false;
        }
        return supports_atomics;
    }
}

template <typename fnT, typename T>
struct ArithmeticAtomicSupportFactory
{
    fnT get()
    {
        using dpctl::tensor::type_utils::is_complex;
        if constexpr (std::is_floating_point_v<T> ||
                      std::is_same_v<T, sycl::half> || is_complex<T>::value) {
            // for real- and complex- floating point types, tree reduction has
            // better round-off accumulation properties (round-off error is
            // proportional to the log2(reduction_size), while naive elementwise
            // summation used by atomic implementation has round-off error
            // growing proportional to the reduction_size.), hence reduction
            // over floating point types should always use tree_reduction
            // algorithm, even though atomic implementation may be applicable
            return fixed_decision<false>;
        }
        else {
            return check_atomic_support<T>;
        }
    }
};

template <typename fnT, typename T>
struct MinMaxAtomicSupportFactory
{
    fnT get() { return check_atomic_support<T>; }
};

template <typename fnT, typename T>
struct MaxAtomicSupportFactory : public MinMaxAtomicSupportFactory<fnT, T>
{
};

template <typename fnT, typename T>
struct MinAtomicSupportFactory : public MinMaxAtomicSupportFactory<fnT, T>
{
};

template <typename fnT, typename T>
struct SumAtomicSupportFactory : public ArithmeticAtomicSupportFactory<fnT, T>
{
};

template <typename fnT, typename T>
struct ProductAtomicSupportFactory
    : public ArithmeticAtomicSupportFactory<fnT, T>
{
};

} // namespace dpctl::tensor::py_internal::atomic_support
