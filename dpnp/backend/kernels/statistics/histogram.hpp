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
#include <cstdint>

#include <sycl/sycl.hpp>

namespace dpnp::kernels::histogram
{
template <typename T, typename HistImpl, typename Edges, typename Weights>
class HistogramFunctor
{
private:
    const T *in = nullptr;
    const std::size_t size;
    const std::size_t dims;
    const std::uint32_t WorkPI;
    const HistImpl hist;
    const Edges edges;
    const Weights weights;

public:
    HistogramFunctor(const T *in_,
                     const std::size_t size_,
                     const std::size_t dims_,
                     const std::uint32_t WorkPI_,
                     const HistImpl &hist_,
                     const Edges &edges_,
                     const Weights &weights_)
        : in(in_), size(size_), dims(dims_), WorkPI(WorkPI_), hist(hist_),
          edges(edges_), weights(weights_)
    {
    }

    void operator()(sycl::nd_item<1> item) const
    {
        auto id = item.get_group_linear_id();
        auto lid = item.get_local_linear_id();
        auto group = item.get_group();
        auto local_size = item.get_local_range(0);

        hist.init(item);
        edges.init(item);

        if constexpr (HistImpl::sync_after_init || Edges::sync_after_init) {
            sycl::group_barrier(group, sycl::memory_scope::work_group);
        }

        auto bounds = edges.get_bounds();

        for (uint32_t i = 0; i < WorkPI; ++i) {
            auto data_idx = id * WorkPI * local_size + i * local_size + lid;
            if (data_idx < size) {
                auto *d = &in[data_idx * dims];

                if (edges.in_bounds(d, bounds)) {
                    auto bin = edges.get_bin(item, d, bounds);
                    auto weight = weights.get(data_idx);
                    hist.add(item, bin, weight);
                }
            }
        }

        if constexpr (HistImpl::sync_before_finalize) {
            sycl::group_barrier(group, sycl::memory_scope::work_group);
        }

        hist.finalize(item);
    }
};
} // namespace dpnp::kernels::histogram
