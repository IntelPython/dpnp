//*****************************************************************************
// Copyright (c) 2023, Intel Corporation
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

namespace dpnp
{
namespace backend
{
namespace ext
{
namespace vm
{
template <typename T>
sycl::event atan2_contig_impl(sycl::queue exec_q,
                              const std::int64_t n,
                              const char *in_a,
                              const char *in_b,
                              char *out_y,
                              const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<T>(exec_q);

    const T *a = reinterpret_cast<const T *>(in_a);
    const T *b = reinterpret_cast<const T *>(in_b);
    using resTy = typename types::Atan2OutputType<T>::value_type;
    resTy *y = reinterpret_cast<resTy *>(out_y);

    return mkl_vm::atan2(exec_q,
                         n, // number of elements to be calculated
                         a, // pointer `a` containing 1st input vector of size n
                         b, // pointer `b` containing 2nd input vector of size n
                         y, // pointer `y` to the output vector of size n
                         depends);
}

template <typename fnT, typename T>
struct Atan2ContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename types::Atan2OutputType<T>::value_type, void>)
        {
            return nullptr;
        }
        else {
            return atan2_contig_impl<T>;
        }
    }
};
} // namespace vm
} // namespace ext
} // namespace backend
} // namespace dpnp
