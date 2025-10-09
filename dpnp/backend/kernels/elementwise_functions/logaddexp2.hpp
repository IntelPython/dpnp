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

#include <cmath>
#include <sycl/sycl.hpp>

namespace dpnp::kernels::logaddexp2
{
constexpr double log2e = 1.442695040888963407359924681001892137;

template <typename T>
inline T log2_1p(T x)
{
    return T(log2e) * sycl::log1p(x);
}

template <typename T>
inline T logaddexp2(T x, T y)
{
    if (x == y) {
        // handles infinities of the same sign
        return x + 1;
    }

    const T tmp = x - y;
    if (tmp > 0) {
        return x + log2_1p(sycl::exp2(-tmp));
    }
    else if (tmp <= 0) {
        return y + log2_1p(sycl::exp2(tmp));
    }
    return std::numeric_limits<T>::quiet_NaN();
}

template <typename argT1, typename argT2, typename resT>
struct Logaddexp2Functor
{
    using supports_sg_loadstore = std::true_type;
    using supports_vec = std::false_type;

    resT operator()(const argT1 &in1, const argT2 &in2) const
    {
        return logaddexp2<resT>(in1, in2);
    }
};
} // namespace dpnp::kernels::logaddexp2
