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

#include "kernels/dpctl_tensor_types.hpp"
#include "utils/strided_iters.hpp"

namespace dpnp::kernels::choose
{
using dpctl::tensor::ssize_t;

template <typename ProjectorT,
          typename IndOutIndexerT,
          typename ChoicesIndexerT,
          typename IndT,
          typename T>
class ChooseFunctor
{
private:
    const IndT *ind = nullptr;
    T *dst = nullptr;
    char **chcs = nullptr;
    ssize_t n_chcs;
    const IndOutIndexerT ind_out_indexer;
    const ChoicesIndexerT chcs_indexer;

public:
    ChooseFunctor(const IndT *ind_,
                  T *dst_,
                  char **chcs_,
                  ssize_t n_chcs_,
                  const IndOutIndexerT &ind_out_indexer_,
                  const ChoicesIndexerT &chcs_indexer_)
        : ind(ind_), dst(dst_), chcs(chcs_), n_chcs(n_chcs_),
          ind_out_indexer(ind_out_indexer_), chcs_indexer(chcs_indexer_)
    {
    }

    void operator()(sycl::id<1> id) const
    {
        const ProjectorT proj{};

        ssize_t i = id[0];

        auto ind_dst_offsets = ind_out_indexer(i);
        ssize_t ind_offset = ind_dst_offsets.get_first_offset();
        ssize_t dst_offset = ind_dst_offsets.get_second_offset();

        IndT chc_idx = ind[ind_offset];
        // proj produces an index in the range of n_chcs
        ssize_t projected_idx = proj(n_chcs, chc_idx);

        ssize_t chc_offset = chcs_indexer(i, projected_idx);

        T *chc = reinterpret_cast<T *>(chcs[projected_idx]);

        dst[dst_offset] = chc[chc_offset];
    }
};

namespace strides
{
using dpctl::tensor::strides::CIndexer_vector;

struct NthStrideOffsetUnpacked
{
    NthStrideOffsetUnpacked(int common_nd,
                            ssize_t const *_offsets,
                            ssize_t const *_shape,
                            ssize_t const *_strides)
        : _ind(common_nd), nd(common_nd), offsets(_offsets), shape(_shape),
          strides(_strides)
    {
    }

    template <typename nT>
    size_t operator()(ssize_t gid, nT n) const
    {
        ssize_t relative_offset(0);
        _ind.get_displacement<const ssize_t *, const ssize_t *>(
            gid, shape, strides + (n * nd), relative_offset);

        return relative_offset + offsets[n];
    }

private:
    CIndexer_vector<ssize_t> _ind;

    int nd;
    ssize_t const *offsets;
    ssize_t const *shape;
    ssize_t const *strides;
};

static_assert(sycl::is_device_copyable_v<NthStrideOffsetUnpacked>);

} // namespace strides
} // namespace dpnp::kernels::choose
