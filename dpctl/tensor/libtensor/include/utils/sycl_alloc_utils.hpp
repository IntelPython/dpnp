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
/// This file defines CIndexer_array, and CIndexer_vector classes, as well
/// iteration space simplifiers.
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

namespace dpctl::tensor::alloc_utils
{
template <typename T>
class usm_host_allocator : public sycl::usm_allocator<T, sycl::usm::alloc::host>
{
public:
    using baseT = sycl::usm_allocator<T, sycl::usm::alloc::host>;
    using baseT::baseT;

    template <typename U>
    struct rebind
    {
        typedef usm_host_allocator<U> other;
    };

    void deallocate(T *ptr, std::size_t n)
    {
        try {
            baseT::deallocate(ptr, n);
        } catch (const std::exception &e) {
            std::cerr
                << "Exception caught in `usm_host_allocator::deallocate`: "
                << e.what() << std::endl;
        }
    }
};

template <typename T>
void sycl_free_noexcept(T *ptr, const sycl::context &ctx) noexcept
{
    try {
        sycl::free(ptr, ctx);
    } catch (const std::exception &e) {
        std::cerr << "Call to sycl::free caught exception: " << e.what()
                  << std::endl;
    }
}

template <typename T>
void sycl_free_noexcept(T *ptr, const sycl::queue &q) noexcept
{
    sycl_free_noexcept(ptr, q.get_context());
}

class USMDeleter
{
private:
    sycl::context ctx_;

public:
    USMDeleter(const sycl::queue &q) : ctx_(q.get_context()) {}
    USMDeleter(const sycl::context &ctx) : ctx_(ctx) {}

    template <typename T>
    void operator()(T *ptr) const
    {
        sycl_free_noexcept(ptr, ctx_);
    }
};

template <typename T>
std::unique_ptr<T, USMDeleter>
    smart_malloc(std::size_t count,
                 const sycl::queue &q,
                 sycl::usm::alloc kind,
                 const sycl::property_list &propList = {})
{
    T *ptr = sycl::malloc<T>(count, q, kind, propList);
    if (nullptr == ptr) {
        throw std::runtime_error("Unable to allocate device_memory");
    }

    auto usm_deleter = USMDeleter(q);
    return std::unique_ptr<T, USMDeleter>(ptr, usm_deleter);
}

template <typename T>
std::unique_ptr<T, USMDeleter>
    smart_malloc_device(std::size_t count,
                        const sycl::queue &q,
                        const sycl::property_list &propList = {})
{
    return smart_malloc<T>(count, q, sycl::usm::alloc::device, propList);
}

template <typename T>
std::unique_ptr<T, USMDeleter>
    smart_malloc_shared(std::size_t count,
                        const sycl::queue &q,
                        const sycl::property_list &propList = {})
{
    return smart_malloc<T>(count, q, sycl::usm::alloc::shared, propList);
}

template <typename T>
std::unique_ptr<T, USMDeleter>
    smart_malloc_host(std::size_t count,
                      const sycl::queue &q,
                      const sycl::property_list &propList = {})
{
    return smart_malloc<T>(count, q, sycl::usm::alloc::host, propList);
}

namespace detail
{
template <typename T>
struct valid_smart_ptr : public std::false_type
{
};

template <typename ValT, typename DeleterT>
struct valid_smart_ptr<std::unique_ptr<ValT, DeleterT> &>
    : public std::is_same<DeleterT, USMDeleter>
{
};

template <typename ValT, typename DeleterT>
struct valid_smart_ptr<std::unique_ptr<ValT, DeleterT>>
    : public std::is_same<DeleterT, USMDeleter>
{
};

// base case
template <typename... Rest>
struct all_valid_smart_ptrs
{
    static constexpr bool value = true;
};

template <typename Arg, typename... RestArgs>
struct all_valid_smart_ptrs<Arg, RestArgs...>
{
    static constexpr bool value = valid_smart_ptr<Arg>::value &&
                                  (all_valid_smart_ptrs<RestArgs...>::value);
};
} // end of namespace detail

/*! @brief Submit host_task and transfer ownership from smart pointers to it */
template <typename... UniquePtrTs>
sycl::event async_smart_free(sycl::queue &exec_q,
                             const std::vector<sycl::event> &depends,
                             UniquePtrTs &&...unique_pointers)
{
    static constexpr std::size_t n = sizeof...(UniquePtrTs);
    static_assert(
        n > 0, "async_smart_free requires at least one smart pointer argument");

    static_assert(
        detail::all_valid_smart_ptrs<UniquePtrTs...>::value,
        "async_smart_free requires unique_ptr created with smart_malloc");

    std::vector<void *> ptrs;
    ptrs.reserve(n);
    (ptrs.push_back(reinterpret_cast<void *>(unique_pointers.get())), ...);

    std::vector<USMDeleter> dels;
    dels.reserve(n);
    (dels.emplace_back(unique_pointers.get_deleter()), ...);

    sycl::event ht_e = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        cgh.host_task([ptrs = std::move(ptrs), dels = std::move(dels)]() {
            for (std::size_t i = 0; i < ptrs.size(); ++i) {
                dels[i](ptrs[i]);
            }
        });
    });

    // Upon successful submission of host_task, USM allocations are owned
    // by the host_task. Release smart pointer ownership to avoid double
    // deallocation
    (unique_pointers.release(), ...);

    return ht_e;
}
} // namespace dpctl::tensor::alloc_utils
