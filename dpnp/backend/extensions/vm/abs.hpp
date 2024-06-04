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

#include <CL/sycl.hpp>

#include "common.hpp"
#include "types_matrix.hpp"

// dpctl tensor headers
#include "utils/type_dispatch.hpp"

namespace td_ns = dpctl::tensor::type_dispatch;

namespace dpnp::backend::ext::vm
{
template <typename T>
sycl::event abs_contig_impl(sycl::queue &exec_q,
                            std::size_t in_n,
                            const char *in_a,
                            char *out_y,
                            const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<T>(exec_q);

    std::int64_t n = static_cast<std::int64_t>(in_n);
    const T *a = reinterpret_cast<const T *>(in_a);
    using resTy = typename types::AbsOutputType<T>::value_type;
    resTy *y = reinterpret_cast<resTy *>(out_y);

    return mkl_vm::abs(exec_q,
                       n, // number of elements to be calculated
                       a, // pointer `a` containing input vector of size n
                       y, // pointer `y` to the output vector of size n
                       depends);
}

template <typename fnT, typename T>
struct AbsContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename types::AbsOutputType<T>::value_type, void>)
        {
            return nullptr;
        }
        else {
            return abs_contig_impl<T>;
        }
    }
};

template <typename fnT, typename T>
struct AbsStridedFactory
{
    fnT get()
    {
        return nullptr;
    }
};

template <typename fnT, typename T>
struct AbsTypeMapFactory
{
    /*! @brief get typeid for output type of abs(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename types::AbsOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};
} // namespace dpnp::backend::ext::vm
