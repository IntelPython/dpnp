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

#include <sycl/sycl.hpp>

namespace dpnp::kernels::divmod
{
template <typename argT1, typename argT2, typename divT, typename modT>
struct DivmodFunctor
{
    using argT = argT1;

    static_assert(std::is_same_v<argT, argT2>,
                  "Input types are expected to be the same");
    static_assert(std::is_integral_v<argT> || std::is_floating_point_v<argT> ||
                      std::is_same_v<argT, sycl::half>,
                  "Input types are expected to be integral or floating");

    using supports_vec = typename std::false_type;
    using supports_sg_loadstore = typename std::true_type;

    divT operator()(const argT &in1, const argT &in2, modT &mod) const
    {
        if constexpr (std::is_integral_v<argT>) {
            if (in2 == argT(0)) {
                mod = modT(0);
                return divT(0);
            }

            if constexpr (std::is_signed_v<argT>) {
                if ((in1 == std::numeric_limits<argT>::min()) &&
                    (in2 == argT(-1))) {
                    mod = modT(0);
                    return std::numeric_limits<argT>::min();
                }
            }

            divT div = in1 / in2;
            mod = in1 % in2;

            if constexpr (std::is_signed_v<argT>) {
                if (l_xor(in1 > 0, in2 > 0) && (mod != 0)) {
                    div -= divT(1);
                    mod += in2;
                }
            }
            return div;
        }
        else {
            mod = sycl::fmod(in1, in2);
            if (!in2) {
                // in2 == 0 (not NaN): return result of fmod (for IEEE is nan)
                return in1 / in2;
            }

            // (in1 - mod) should be very nearly an integer multiple of in2
            auto div = (in1 - mod) / in2;

            // adjust fmod result to conform to Python convention of remainder
            if (mod) {
                if (l_xor(in2 < 0, mod < 0)) {
                    mod += in2;
                    div -= divT(1.0);
                }
            }
            else {
                // if mod is zero ensure correct sign
                mod = sycl::copysign(modT(0), in2);
            }

            // snap quotient to nearest integral value
            if (div) {
                auto floordiv = sycl::floor(div);
                if (div - floordiv > divT(0.5)) {
                    floordiv += divT(1.0);
                }
                div = floordiv;
            }
            else {
                // if div is zero ensure correct sign
                div = sycl::copysign(divT(0), in1 / in2);
            }
            return div;
        }
    }

private:
    bool l_xor(bool b1, bool b2) const
    {
        return (b1 != b2);
    }
};
} // namespace dpnp::kernels::divmod
