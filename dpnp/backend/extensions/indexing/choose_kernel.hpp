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

#include "kernels/indexing/choose.hpp"

namespace dpnp::extensions::indexing
{
namespace strides_detail
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

} // namespace strides_detail

namespace kernels
{

using dpnp::kernels::choose::ChooseFunctor;

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

} // namespace kernels
} // namespace dpnp::extensions::indexing
