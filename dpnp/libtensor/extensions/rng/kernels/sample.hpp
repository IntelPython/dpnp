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
// The number of output elements each work-item produces in the strided and
// blocked layouts. It fixes the mapping of elements to engine streams, and
// so is part of the stream definition: changing it changes every generated
// value.
inline constexpr std::size_t items_per_work_item = 8;

// The samplers caching the second normal of a pair take whole pairs
static_assert(items_per_work_item % 2 == 0);

// The number of output elements each work-item produces in the sequential
// layout, where it only sets how much of the stream every work-item skips
// to, not the values.
inline constexpr std::size_t sequential_items_per_work_item = 64;

// How the elements of a launch map to engine streams:
// - strided: work-item `w` builds its own engine on the counter block `w`
//   and draws elements w, w + G, w + 2G, ..., where G is the number of
//   work-items, so neighbouring work-items write neighbouring elements;
// - blocked: the strided layout, with all the words of a work-item
//   generated together, which is faster on GPUs. It needs a sampler taking
//   a fixed number of words per element, and gives the strided values;
// - sequential: for an ordered engine and a sampler taking a fixed number
//   of draws per element, the elements are the sequential stream of the
//   engine, and work-item `w` draws elements w * S to (w + 1) * S - 1,
//   where S is sequential_items_per_work_item, from the engine advanced
//   past the earlier ones.
// The values depend only on the engine state and the number of elements,
// never on the device, the layout chosen for it or how the work-items are
// scheduled.
enum class Layout
{
    strided,
    blocked,
    sequential,
};

template <typename EngineTraitsT, typename SamplerT>
inline constexpr bool can_draw_sequential =
    EngineTraitsT::ordered && SamplerT::halves_per_item != 0;

template <typename EngineTraitsT, typename SamplerT>
inline constexpr bool can_draw_blocked =
    SamplerT::words_per_item != 0 &&
    SamplerT::words_per_item * items_per_work_item <=
        EngineTraitsT::max_block_words;

// The number of work-items a launch producing `n` elements uses
inline std::size_t num_work_items(std::size_t n, Layout layout)
{
    const std::size_t m = (layout == Layout::sequential)
                              ? sequential_items_per_work_item
                              : items_per_work_item;
    return (n + m - 1) / m;
}

// Fills out[0:n] with samples, in the layout L
template <typename EngineTraitsT, typename SamplerT, Layout L>
class SampleFunctor
{
private:
    using T = typename SamplerT::result_type;

    static_assert(L != Layout::sequential ||
                  can_draw_sequential<EngineTraitsT, SamplerT>);
    static_assert(L != Layout::blocked ||
                  can_draw_blocked<EngineTraitsT, SamplerT>);

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
        SamplerT sampler = sampler_;

        if constexpr (L == Layout::sequential) {
            constexpr std::size_t m = sequential_items_per_work_item;
            const std::size_t first = w * m;

            auto bg = EngineTraitsT::make_at(state_, first,
                                             SamplerT::halves_per_item);
            if (first + m <= n_) {
#pragma unroll
                for (std::size_t k = 0; k < m; ++k) {
                    out_[first + k] = sampler(bg, first + k);
                }
            }
            else {
                for (std::size_t i = first; i < n_; ++i) {
                    out_[i] = sampler(bg, i);
                }
            }
        }
        else if constexpr (L == Layout::blocked) {
            constexpr unsigned words =
                SamplerT::words_per_item * items_per_work_item;

            auto bg = EngineTraitsT::template make_block<words>(state_, w);
            // Sample every element, even past the end, so the positions of
            // all the draws are known at compile time. Element w always
            // exists, and stands in for the parameters of those past the
            // end.
#pragma unroll
            for (std::size_t k = 0; k < items_per_work_item; ++k) {
                const std::size_t i = w + k * n_wi_;
                const bool valid = i < n_;
                const T v = sampler(bg, valid ? i : w);
                if (valid) {
                    out_[i] = v;
                }
            }
        }
        else {
            auto bg = EngineTraitsT::make(state_, w);
            for (std::size_t i = w; i < n_; i += n_wi_) {
                out_[i] = sampler(bg, i);
            }
        }
    }
};
} // namespace dpnp::kernels::rng
