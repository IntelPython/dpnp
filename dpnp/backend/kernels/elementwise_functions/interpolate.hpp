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

#include <sycl/sycl.hpp>
#include <vector>

#include "ext/common.hpp"

using ext::common::IsNan;

namespace dpnp::kernels::interpolate
{
template <typename TCoord, typename TValue, typename TIdx = std::int64_t>
sycl::event interpolate_impl(sycl::queue &q,
                             const TCoord *x,
                             const TIdx *idx,
                             const TCoord *xp,
                             const TValue *fp,
                             const TValue *left,
                             const TValue *right,
                             TValue *out,
                             const std::size_t n,
                             const std::size_t xp_size,
                             const std::vector<sycl::event> &depends)
{
    return q.submit([&](sycl::handler &h) {
        h.depends_on(depends);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            TValue left_val = left ? *left : fp[0];
            TValue right_val = right ? *right : fp[xp_size - 1];

            TCoord x_val = x[i];
            std::int64_t x_idx = idx[i] - 1;

            if (IsNan<TCoord>::isnan(x_val)) {
                out[i] = x_val;
            }
            else if (x_idx < 0) {
                out[i] = left_val;
            }
            else if (x_val == xp[xp_size - 1]) {
                out[i] = fp[xp_size - 1];
            }
            else if (x_idx >= static_cast<std::int64_t>(xp_size - 1)) {
                out[i] = right_val;
            }
            else {
                TValue slope =
                    (fp[x_idx + 1] - fp[x_idx]) / (xp[x_idx + 1] - xp[x_idx]);
                TValue res = slope * (x_val - xp[x_idx]) + fp[x_idx];

                if (IsNan<TValue>::isnan(res)) {
                    res = slope * (x_val - xp[x_idx + 1]) + fp[x_idx + 1];
                    if (IsNan<TValue>::isnan(res) &&
                        (fp[x_idx] == fp[x_idx + 1])) {
                        res = fp[x_idx];
                    }
                }
                out[i] = res;
            }
        });
    });
}

} // namespace dpnp::kernels::interpolate
