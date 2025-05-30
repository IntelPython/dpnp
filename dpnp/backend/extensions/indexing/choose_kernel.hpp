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

#include <algorithm>
#include <complex>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <sycl/sycl.hpp>

#include "kernels/dpctl_tensor_types.hpp"
#include "utils/indexing_utils.hpp"
#include "utils/offset_utils.hpp"
#include "utils/strided_iters.hpp"
#include "utils/type_utils.hpp"

namespace dpnp::extensions::indexing::strides_detail
{

struct NthStrideOffsetUnpacked
{
    NthStrideOffsetUnpacked(int common_nd,
                            dpctl::tensor::ssize_t const *_offsets,
                            dpctl::tensor::ssize_t const *_shape,
                            dpctl::tensor::ssize_t const *_strides)
        : _ind(common_nd), nd(common_nd), offsets(_offsets), shape(_shape),
          strides(_strides)
    {
    }

    template <typename nT>
    size_t operator()(dpctl::tensor::ssize_t gid, nT n) const
    {
        dpctl::tensor::ssize_t relative_offset(0);
        _ind.get_displacement<const dpctl::tensor::ssize_t *,
                              const dpctl::tensor::ssize_t *>(
            gid, shape, strides + (n * nd), relative_offset);

        return relative_offset + offsets[n];
    }

private:
    dpctl::tensor::strides::CIndexer_vector<dpctl::tensor::ssize_t> _ind;

    int nd;
    dpctl::tensor::ssize_t const *offsets;
    dpctl::tensor::ssize_t const *shape;
    dpctl::tensor::ssize_t const *strides;
};

static_assert(sycl::is_device_copyable_v<NthStrideOffsetUnpacked>);

} // namespace dpnp::extensions::indexing::strides_detail

namespace dpnp::extensions::indexing::kernels
{

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
    dpctl::tensor::ssize_t n_chcs;
    const IndOutIndexerT ind_out_indexer;
    const ChoicesIndexerT chcs_indexer;

public:
    ChooseFunctor(const IndT *ind_,
                  T *dst_,
                  char **chcs_,
                  dpctl::tensor::ssize_t n_chcs_,
                  const IndOutIndexerT &ind_out_indexer_,
                  const ChoicesIndexerT &chcs_indexer_)
        : ind(ind_), dst(dst_), chcs(chcs_), n_chcs(n_chcs_),
          ind_out_indexer(ind_out_indexer_), chcs_indexer(chcs_indexer_)
    {
    }

    void operator()(sycl::id<1> id) const
    {
        const ProjectorT proj{};

        dpctl::tensor::ssize_t i = id[0];

        auto ind_dst_offsets = ind_out_indexer(i);
        dpctl::tensor::ssize_t ind_offset = ind_dst_offsets.get_first_offset();
        dpctl::tensor::ssize_t dst_offset = ind_dst_offsets.get_second_offset();

        IndT chc_idx = ind[ind_offset];
        // proj produces an index in the range of n_chcs
        dpctl::tensor::ssize_t projected_idx = proj(n_chcs, chc_idx);

        dpctl::tensor::ssize_t chc_offset = chcs_indexer(i, projected_idx);

        T *chc = reinterpret_cast<T *>(chcs[projected_idx]);

        dst[dst_offset] = chc[chc_offset];
    }
};

typedef sycl::event (*choose_fn_ptr_t)(sycl::queue &,
                                       size_t,
                                       dpctl::tensor::ssize_t,
                                       int,
                                       const dpctl::tensor::ssize_t *,
                                       const char *,
                                       char *,
                                       char **,
                                       dpctl::tensor::ssize_t,
                                       dpctl::tensor::ssize_t,
                                       const dpctl::tensor::ssize_t *,
                                       const std::vector<sycl::event> &);

template <typename ProjectorT, typename indTy, typename Ty>
sycl::event choose_impl(sycl::queue &q,
                        size_t nelems,
                        dpctl::tensor::ssize_t n_chcs,
                        int nd,
                        const dpctl::tensor::ssize_t *shape_and_strides,
                        const char *ind_cp,
                        char *dst_cp,
                        char **chcs_cp,
                        dpctl::tensor::ssize_t ind_offset,
                        dpctl::tensor::ssize_t dst_offset,
                        const dpctl::tensor::ssize_t *chc_offsets,
                        const std::vector<sycl::event> &depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<Ty>(q);

    const indTy *ind_tp = reinterpret_cast<const indTy *>(ind_cp);
    Ty *dst_tp = reinterpret_cast<Ty *>(dst_cp);

    sycl::event choose_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using InOutIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;
        const InOutIndexerT ind_out_indexer{nd, ind_offset, dst_offset,
                                            shape_and_strides};

        using NthChoiceIndexerT = strides_detail::NthStrideOffsetUnpacked;
        const NthChoiceIndexerT choices_indexer{
            nd, chc_offsets, shape_and_strides, shape_and_strides + 3 * nd};

        using ChooseFunc = ChooseFunctor<ProjectorT, InOutIndexerT,
                                         NthChoiceIndexerT, indTy, Ty>;

        cgh.parallel_for<ChooseFunc>(sycl::range<1>(nelems),
                                     ChooseFunc(ind_tp, dst_tp, chcs_cp, n_chcs,
                                                ind_out_indexer,
                                                choices_indexer));
    });

    return choose_ev;
}

} // namespace dpnp::extensions::indexing::kernels
