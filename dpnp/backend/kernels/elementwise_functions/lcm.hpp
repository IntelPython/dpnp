//*****************************************************************************
// Copyright (c) 2024-2025, Intel Corporation
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

#include <oneapi/dpl/numeric>
#include <sycl/sycl.hpp>

namespace dpnp::kernels::lcm
{
template <typename argT1, typename argT2, typename resT>
struct LcmFunctor
{
    using supports_sg_loadstore = typename std::true_type;
    using supports_vec = typename std::false_type;

    resT operator()(const argT1 &in1, const argT2 &in2) const
    {
        static_assert(std::is_same_v<argT1, argT2>,
                      "Input types are expected to be the same");

        if (in1 == 0 || in2 == 0)
            return 0;

        resT res = in1 / oneapi::dpl::gcd(in1, in2) * in2;
        if constexpr (std::is_signed_v<argT1>) {
            if (res < 0) {
                return -res;
            }
        }
        return res;

        // TODO: undo the w/a once ONEDPL-1320 is resolved
        // return oneapi::dpl::lcm(in1, in2);
    }
};
} // namespace dpnp::kernels::lcm
