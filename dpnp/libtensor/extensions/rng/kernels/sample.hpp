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

#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "engines.hpp"

namespace dpnp::kernels::rng
{
// The number of output elements each work-item produces. It fixes the
// mapping of elements to engine streams, and so is part of the stream
// definition: changing it changes every generated value.
inline constexpr std::size_t items_per_work_item = 8;

// The number of work-items, and so of counter blocks, a launch producing
// `n` elements uses.
inline std::size_t num_work_items(std::size_t n)
{
    return (n + items_per_work_item - 1) / items_per_work_item;
}

// Fills out[0:n] with samples. Work-item `w` builds its own engine on the
// counter block `w` and draws elements w, w + G, w + 2G, ..., where G is the
// number of work-items, so neighbouring work-items write neighbouring
// elements. The values depend only on the engine state and `n`, never on
// the device or on how the work-items are scheduled.
template <typename EngineTraitsT, typename T, typename SamplerT>
class SampleFunctor
{
private:
    EngineState state_;
    SamplerT sampler_;
    T *out_ = nullptr;
    std::size_t n_ = 0;
    std::size_t n_wi_ = 0;

public:
    SampleFunctor(const EngineState &state,
                  const SamplerT &sampler,
                  T *out,
                  std::size_t n,
                  std::size_t n_wi)
        : state_(state), sampler_(sampler), out_(out), n_(n), n_wi_(n_wi)
    {
    }

    void operator()(sycl::id<1> id) const
    {
        const std::size_t w = id[0];

        BitGen bg(EngineTraitsT::make(state_, w));
        SamplerT sampler = sampler_;

        for (std::size_t i = w; i < n_; i += n_wi_) {
            out_[i] = sampler(bg, i);
        }
    }
};
} // namespace dpnp::kernels::rng
