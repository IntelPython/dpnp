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

#include <sycl/sycl.hpp>

namespace dpnp::kernels::fmod
{
template <typename argT1, typename argT2, typename resT>
struct FmodFunctor
{
    // using supports_sg_loadstore = std::negation<
    //     std::disjunction<tu_ns::is_complex<argT1>,
    //     tu_ns::is_complex<argT2>>>;
    // using supports_vec = std::negation<
    //     std::disjunction<tu_ns::is_complex<argT1>,
    //     tu_ns::is_complex<argT2>>>;

    using supports_sg_loadstore = typename std::false_type;
    using supports_vec = typename std::false_type;

    resT operator()(const argT1 &in1, const argT2 &in2) const
    {
        if constexpr (std::is_integral<argT1>::value &&
                      std::is_integral<argT2>::value) {
            if (in2 == argT2(0)) {
                return resT(0);
            }
            return in1 % in2;
        }
        else if constexpr (std::is_integral<argT1>::value) {
            return sycl::fmod(argT2(in1), in2);
        }
        else if constexpr (std::is_integral<argT2>::value) {
            return sycl::fmod(in1, argT1(in2));
        }
        else {
            return sycl::fmod(in1, in2);
        }
    }
};
} // namespace dpnp::kernels::fmod
