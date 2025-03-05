//*****************************************************************************
// Copyright (c) 2025, Intel Corporation
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

#include "common.hpp"
#include <sycl/sycl.hpp>

namespace dpnp::extensions::window::kernels
{

template <typename T>
class BartlettFunctor
{
private:
    T *data = nullptr;
    const std::size_t N;

public:
    BartlettFunctor(T *data, const std::size_t N) : data(data), N(N) {}

    void operator()(sycl::id<1> id) const
    {
        const auto i = id.get(0);

        data[i] =
            T(2) / (N - 1) * ((N - 1) / T(2) - sycl::fabs(i - (N - 1) / T(2)));
    }
};

template <typename fnT, typename T>
struct BartlettFactory
{
    fnT get()
    {
        if constexpr (std::is_floating_point_v<T>) {
            return window_impl<T, BartlettFunctor>;
        }
        else {
            return nullptr;
        }
    }
};

} // namespace dpnp::extensions::window::kernels
