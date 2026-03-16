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

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "ext/common.hpp"

namespace dpnp::kernels::interpolate
{
using ext::common::IsNan;

template <typename TCoord, typename TValue, typename TIdx = std::int64_t>
class InterpolateFunctor
{
private:
    const TCoord *x = nullptr;
    const TIdx *idx = nullptr;
    const TCoord *xp = nullptr;
    const TValue *fp = nullptr;
    const TValue *left = nullptr;
    const TValue *right = nullptr;
    TValue *out = nullptr;
    const std::size_t xp_size;

public:
    InterpolateFunctor(const TCoord *x_,
                       const TIdx *idx_,
                       const TCoord *xp_,
                       const TValue *fp_,
                       const TValue *left_,
                       const TValue *right_,
                       TValue *out_,
                       const std::size_t xp_size_)
        : x(x_), idx(idx_), xp(xp_), fp(fp_), left(left_), right(right_),
          out(out_), xp_size(xp_size_)
    {
    }

    // Selected over the work-group version
    // due to simpler execution and slightly better performance.
    void operator()(sycl::id<1> id) const
    {
        TValue left_val = left ? *left : fp[0];
        TValue right_val = right ? *right : fp[xp_size - 1];

        TCoord x_val = x[id];
        TIdx x_idx = idx[id] - 1;

        if (IsNan<TCoord>::isnan(x_val)) {
            out[id] = x_val;
        }
        else if (x_idx < 0) {
            out[id] = left_val;
        }
        else if (x_val == xp[xp_size - 1]) {
            out[id] = fp[xp_size - 1];
        }
        else if (x_idx >= static_cast<TIdx>(xp_size - 1)) {
            out[id] = right_val;
        }
        else {
            TValue slope =
                (fp[x_idx + 1] - fp[x_idx]) / (xp[x_idx + 1] - xp[x_idx]);
            TValue res = slope * (x_val - xp[x_idx]) + fp[x_idx];

            if (IsNan<TValue>::isnan(res)) {
                res = slope * (x_val - xp[x_idx + 1]) + fp[x_idx + 1];
                if (IsNan<TValue>::isnan(res) && (fp[x_idx] == fp[x_idx + 1])) {
                    res = fp[x_idx];
                }
            }
            out[id] = res;
        }
    }
};
} // namespace dpnp::kernels::interpolate
