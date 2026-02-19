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
//
//===---------------------------------------------------------------------===//
///
/// \file
/// This file defines kernels for advanced tensor index operations.
//===---------------------------------------------------------------------===//

#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpctl_tensor_types.hpp"
#include "utils/offset_utils.hpp"
#include "utils/type_dispatch_building.hpp"

namespace dpctl::tensor::kernels::indexing
{

using dpctl::tensor::ssize_t;
using namespace dpctl::tensor::offset_utils;

template <typename OrthogIndexerT,
          typename MaskedSrcIndexerT,
          typename MaskedDstIndexerT,
          typename dataT,
          typename indT,
          typename LocalAccessorT>
struct MaskedExtractStridedFunctor
{
    MaskedExtractStridedFunctor(const dataT *src_data_p,
                                const indT *cumsum_data_p,
                                dataT *dst_data_p,
                                std::size_t masked_iter_size,
                                const OrthogIndexerT &orthog_src_dst_indexer_,
                                const MaskedSrcIndexerT &masked_src_indexer_,
                                const MaskedDstIndexerT &masked_dst_indexer_,
                                const LocalAccessorT &lacc_)
        : src(src_data_p), cumsum(cumsum_data_p), dst(dst_data_p),
          masked_nelems(masked_iter_size),
          orthog_src_dst_indexer(orthog_src_dst_indexer_),
          masked_src_indexer(masked_src_indexer_),
          masked_dst_indexer(masked_dst_indexer_), lacc(lacc_)
    {
        static_assert(
            std::is_same_v<indT, typename LocalAccessorT::value_type>);
    }

    void operator()(sycl::nd_item<2> ndit) const
    {
        const std::size_t orthog_i = ndit.get_global_id(0);
        const std::uint32_t l_i = ndit.get_local_id(1);
        const std::uint32_t lws = ndit.get_local_range(1);

        const std::size_t masked_i = ndit.get_global_id(1);
        const std::size_t masked_block_start = masked_i - l_i;

        const std::size_t max_offset = masked_nelems + 1;
        for (std::uint32_t i = l_i; i < lacc.size(); i += lws) {
            const std::size_t offset = masked_block_start + i;
            lacc[i] = (offset == 0)           ? indT(0)
                      : (offset < max_offset) ? cumsum[offset - 1]
                                              : cumsum[masked_nelems - 1] + 1;
        }

        sycl::group_barrier(ndit.get_group());

        const indT current_running_count = lacc[l_i + 1];
        const bool mask_set = (masked_i == 0)
                                  ? (current_running_count == 1)
                                  : (current_running_count == lacc[l_i] + 1);

        // dst[cumsum[i] - 1, j] = src[i, j]
        //     if cumsum[i] == ((i > 0) ? cumsum[i-1] + 1 : 1)
        if (mask_set && (masked_i < masked_nelems)) {
            const auto &orthog_offsets = orthog_src_dst_indexer(orthog_i);

            const std::size_t total_src_offset =
                masked_src_indexer(masked_i) +
                orthog_offsets.get_first_offset();
            const std::size_t total_dst_offset =
                masked_dst_indexer(current_running_count - 1) +
                orthog_offsets.get_second_offset();

            dst[total_dst_offset] = src[total_src_offset];
        }
    }

private:
    const dataT *src = nullptr;
    const indT *cumsum = nullptr;
    dataT *dst = nullptr;
    std::size_t masked_nelems = 0;
    // has nd, shape, src_strides, dst_strides for
    // dimensions that ARE NOT masked
    OrthogIndexerT orthog_src_dst_indexer;
    // has nd, shape, src_strides for
    // dimensions that ARE masked
    MaskedSrcIndexerT masked_src_indexer;
    // has 1, dst_strides for dimensions that ARE masked
    MaskedDstIndexerT masked_dst_indexer;
    LocalAccessorT lacc;
};

template <typename OrthogIndexerT,
          typename MaskedDstIndexerT,
          typename MaskedRhsIndexerT,
          typename dataT,
          typename indT,
          typename LocalAccessorT>
struct MaskedPlaceStridedFunctor
{
    MaskedPlaceStridedFunctor(dataT *dst_data_p,
                              const indT *cumsum_data_p,
                              const dataT *rhs_data_p,
                              std::size_t masked_iter_size,
                              const OrthogIndexerT &orthog_dst_rhs_indexer_,
                              const MaskedDstIndexerT &masked_dst_indexer_,
                              const MaskedRhsIndexerT &masked_rhs_indexer_,
                              const LocalAccessorT &lacc_)
        : dst(dst_data_p), cumsum(cumsum_data_p), rhs(rhs_data_p),
          masked_nelems(masked_iter_size),
          orthog_dst_rhs_indexer(orthog_dst_rhs_indexer_),
          masked_dst_indexer(masked_dst_indexer_),
          masked_rhs_indexer(masked_rhs_indexer_), lacc(lacc_)
    {
        static_assert(
            std::is_same_v<indT, typename LocalAccessorT::value_type>);
    }

    void operator()(sycl::nd_item<2> ndit) const
    {
        const std::size_t orthog_i = ndit.get_global_id(0);
        const std::uint32_t l_i = ndit.get_local_id(1);
        const std::uint32_t lws = ndit.get_local_range(1);

        const std::size_t masked_i = ndit.get_global_id(1);
        const std::size_t masked_block_start = masked_i - l_i;

        const std::size_t max_offset = masked_nelems + 1;
        for (std::uint32_t i = l_i; i < lacc.size(); i += lws) {
            const std::size_t offset = masked_block_start + i;
            lacc[i] = (offset == 0)           ? indT(0)
                      : (offset < max_offset) ? cumsum[offset - 1]
                                              : cumsum[masked_nelems - 1] + 1;
        }

        sycl::group_barrier(ndit.get_group());

        const indT current_running_count = lacc[l_i + 1];
        const bool mask_set = (masked_i == 0)
                                  ? (current_running_count == 1)
                                  : (current_running_count == lacc[l_i] + 1);

        // src[i, j] = rhs[cumsum[i] - 1, j]
        // if cumsum[i] == ((i > 0) ? cumsum[i-1] + 1 : 1)
        if (mask_set && (masked_i < masked_nelems)) {
            const auto &orthog_offsets = orthog_dst_rhs_indexer(orthog_i);

            const std::size_t total_dst_offset =
                masked_dst_indexer(masked_i) +
                orthog_offsets.get_first_offset();
            const std::size_t total_rhs_offset =
                masked_rhs_indexer(current_running_count - 1) +
                orthog_offsets.get_second_offset();

            dst[total_dst_offset] = rhs[total_rhs_offset];
        }
    }

private:
    dataT *dst = nullptr;
    const indT *cumsum = nullptr;
    const dataT *rhs = nullptr;
    std::size_t masked_nelems = 0;
    // has nd, shape, dst_strides, rhs_strides for
    // dimensions that ARE NOT masked
    OrthogIndexerT orthog_dst_rhs_indexer;
    // has nd, shape, dst_strides for
    // dimensions that ARE masked
    MaskedDstIndexerT masked_dst_indexer;
    // has 1, rhs_strides for dimensions that ARE masked
    MaskedRhsIndexerT masked_rhs_indexer;
    LocalAccessorT lacc;
};

// ======= Masked extraction ================================

namespace detail
{

template <std::size_t I, std::size_t... IR>
std::size_t _get_lws_impl(std::size_t n)
{
    if constexpr (sizeof...(IR) == 0) {
        return I;
    }
    else {
        return (n < I) ? _get_lws_impl<IR...>(n) : I;
    }
}

inline std::size_t get_lws(std::size_t n)
{
    static constexpr std::size_t lws0 = 256u;
    static constexpr std::size_t lws1 = 128u;
    static constexpr std::size_t lws2 = 64u;
    return _get_lws_impl<lws0, lws1, lws2>(n);
}

} // end of namespace detail

template <typename MaskedDstIndexerT, typename dataT, typename indT>
class masked_extract_all_slices_contig_impl_krn;

typedef sycl::event (*masked_extract_all_slices_contig_impl_fn_ptr_t)(
    sycl::queue &,
    ssize_t,
    const char *,
    const char *,
    char *,
    ssize_t,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename dataT, typename indT>
sycl::event masked_extract_all_slices_contig_impl(
    sycl::queue &exec_q,
    ssize_t iteration_size,
    const char *src_p,
    const char *cumsum_p,
    char *dst_p,
    ssize_t dst_size, // dst is 1D
    ssize_t dst_stride,
    const std::vector<sycl::event> &depends = {})
{
    static constexpr TwoZeroOffsets_Indexer orthog_src_dst_indexer{};

    static constexpr NoOpIndexer masked_src_indexer{};
    const Strided1DIndexer masked_dst_indexer(/* size */ dst_size,
                                              /* step */ dst_stride);

    using KernelName =
        class masked_extract_all_slices_contig_impl_krn<Strided1DIndexer, dataT,
                                                        indT>;

    using LocalAccessorT = sycl::local_accessor<indT, 1>;
    using Impl =
        struct MaskedExtractStridedFunctor<TwoZeroOffsets_Indexer, NoOpIndexer,
                                           Strided1DIndexer, dataT, indT,
                                           LocalAccessorT>;

    const std::size_t masked_extent = iteration_size;

    const std::size_t lws = detail::get_lws(masked_extent);

    const std::size_t n_groups = (iteration_size + lws - 1) / lws;

    sycl::range<2> gRange{1, n_groups * lws};
    sycl::range<2> lRange{1, lws};

    sycl::nd_range<2> ndRange(gRange, lRange);

    const dataT *src_tp = reinterpret_cast<const dataT *>(src_p);
    const indT *cumsum_tp = reinterpret_cast<const indT *>(cumsum_p);
    dataT *dst_tp = reinterpret_cast<dataT *>(dst_p);

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        const std::size_t lacc_size = std::min(lws, masked_extent) + 1;
        LocalAccessorT lacc(lacc_size, cgh);

        cgh.parallel_for<KernelName>(
            ndRange, Impl(src_tp, cumsum_tp, dst_tp, masked_extent,
                          orthog_src_dst_indexer, masked_src_indexer,
                          masked_dst_indexer, lacc));
    });

    return comp_ev;
}

template <typename MaskedSrcIndexerT,
          typename MaskedDstIndexerT,
          typename dataT,
          typename indT>
class masked_extract_all_slices_strided_impl_krn;

typedef sycl::event (*masked_extract_all_slices_strided_impl_fn_ptr_t)(
    sycl::queue &,
    ssize_t,
    const char *,
    const char *,
    char *,
    int,
    ssize_t const *,
    ssize_t,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename dataT, typename indT>
sycl::event masked_extract_all_slices_strided_impl(
    sycl::queue &exec_q,
    ssize_t iteration_size,
    const char *src_p,
    const char *cumsum_p,
    char *dst_p,
    int nd,
    const ssize_t
        *packed_src_shape_strides, // [src_shape, src_strides], length 2*nd
    ssize_t dst_size,              // dst is 1D
    ssize_t dst_stride,
    const std::vector<sycl::event> &depends = {})
{
    static constexpr TwoZeroOffsets_Indexer orthog_src_dst_indexer{};

    /* StridedIndexer(int _nd, ssize_t _offset, ssize_t const
     * *_packed_shape_strides) */
    const StridedIndexer masked_src_indexer(nd, 0, packed_src_shape_strides);
    const Strided1DIndexer masked_dst_indexer(/* size */ dst_size,
                                              /* step */ dst_stride);

    using KernelName = class masked_extract_all_slices_strided_impl_krn<
        StridedIndexer, Strided1DIndexer, dataT, indT>;

    using LocalAccessorT = sycl::local_accessor<indT, 1>;
    using Impl =
        struct MaskedExtractStridedFunctor<TwoZeroOffsets_Indexer,
                                           StridedIndexer, Strided1DIndexer,
                                           dataT, indT, LocalAccessorT>;

    const std::size_t masked_nelems = iteration_size;

    const std::size_t lws = detail::get_lws(masked_nelems);

    const std::size_t n_groups = (masked_nelems + lws - 1) / lws;

    sycl::range<2> gRange{1, n_groups * lws};
    sycl::range<2> lRange{1, lws};

    sycl::nd_range<2> ndRange(gRange, lRange);

    const dataT *src_tp = reinterpret_cast<const dataT *>(src_p);
    const indT *cumsum_tp = reinterpret_cast<const indT *>(cumsum_p);
    dataT *dst_tp = reinterpret_cast<dataT *>(dst_p);

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        const std::size_t lacc_size = std::min(lws, masked_nelems) + 1;
        LocalAccessorT lacc(lacc_size, cgh);

        cgh.parallel_for<KernelName>(
            ndRange, Impl(src_tp, cumsum_tp, dst_tp, iteration_size,
                          orthog_src_dst_indexer, masked_src_indexer,
                          masked_dst_indexer, lacc));
    });

    return comp_ev;
}

typedef sycl::event (*masked_extract_some_slices_strided_impl_fn_ptr_t)(
    sycl::queue &,
    ssize_t,
    ssize_t,
    const char *,
    const char *,
    char *,
    int,
    ssize_t const *,
    ssize_t,
    ssize_t,
    int,
    ssize_t const *,
    ssize_t,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename OrthoIndexerT,
          typename MaskedSrcIndexerT,
          typename MaskedDstIndexerT,
          typename dataT,
          typename indT>
class masked_extract_some_slices_strided_impl_krn;

template <typename dataT, typename indT>
sycl::event masked_extract_some_slices_strided_impl(
    sycl::queue &exec_q,
    ssize_t orthog_nelems,
    ssize_t masked_nelems,
    const char *src_p,
    const char *cumsum_p,
    char *dst_p,
    int orthog_nd,
    // [ortho_shape, ortho_src_strides, // ortho_dst_strides],
    // length 3*ortho_nd
    const ssize_t *packed_ortho_src_dst_shape_strides,
    ssize_t ortho_src_offset,
    ssize_t ortho_dst_offset,
    int masked_nd,
    // [masked_src_shape, masked_src_strides],
    // length 2*masked_nd, mask_dst is 1D
    const ssize_t *packed_masked_src_shape_strides,
    ssize_t masked_dst_size,
    ssize_t masked_dst_stride,
    const std::vector<sycl::event> &depends = {})
{
    const TwoOffsets_StridedIndexer orthog_src_dst_indexer{
        orthog_nd, ortho_src_offset, ortho_dst_offset,
        packed_ortho_src_dst_shape_strides};

    const StridedIndexer masked_src_indexer{masked_nd, 0,
                                            packed_masked_src_shape_strides};
    const Strided1DIndexer masked_dst_indexer{/* size */ masked_dst_size,
                                              /* step */ masked_dst_stride};

    using KernelName = class masked_extract_some_slices_strided_impl_krn<
        TwoOffsets_StridedIndexer, StridedIndexer, Strided1DIndexer, dataT,
        indT>;

    using LocalAccessorT = sycl::local_accessor<indT, 1>;
    using Impl =
        struct MaskedExtractStridedFunctor<TwoOffsets_StridedIndexer,
                                           StridedIndexer, Strided1DIndexer,
                                           dataT, indT, LocalAccessorT>;

    const std::size_t masked_extent = masked_nelems;

    const std::size_t lws = detail::get_lws(masked_extent);

    const std::size_t n_groups = ((masked_extent + lws - 1) / lws);
    const std::size_t orthog_extent = static_cast<std::size_t>(orthog_nelems);

    sycl::range<2> gRange{orthog_extent, n_groups * lws};
    sycl::range<2> lRange{1, lws};

    sycl::nd_range<2> ndRange(gRange, lRange);

    const dataT *src_tp = reinterpret_cast<const dataT *>(src_p);
    const indT *cumsum_tp = reinterpret_cast<const indT *>(cumsum_p);
    dataT *dst_tp = reinterpret_cast<dataT *>(dst_p);

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        const std::size_t lacc_size =
            std::min<std::size_t>(lws, masked_extent) + 1;
        LocalAccessorT lacc(lacc_size, cgh);

        cgh.parallel_for<KernelName>(
            ndRange, Impl(src_tp, cumsum_tp, dst_tp, masked_nelems,
                          orthog_src_dst_indexer, masked_src_indexer,
                          masked_dst_indexer, lacc));
    });

    return comp_ev;
}

template <typename fnT, typename T>
struct MaskExtractAllSlicesContigFactoryForInt32
{
    fnT get()
    {
        fnT fn = masked_extract_all_slices_contig_impl<T, std::int32_t>;
        return fn;
    }
};

template <typename fnT, typename T>
struct MaskExtractAllSlicesContigFactoryForInt64
{
    fnT get()
    {
        fnT fn = masked_extract_all_slices_contig_impl<T, std::int64_t>;
        return fn;
    }
};

template <typename fnT, typename T>
struct MaskExtractAllSlicesStridedFactoryForInt32
{
    fnT get()
    {
        fnT fn = masked_extract_all_slices_strided_impl<T, std::int32_t>;
        return fn;
    }
};

template <typename fnT, typename T>
struct MaskExtractAllSlicesStridedFactoryForInt64
{
    fnT get()
    {
        fnT fn = masked_extract_all_slices_strided_impl<T, std::int64_t>;
        return fn;
    }
};

template <typename fnT, typename T>
struct MaskExtractSomeSlicesStridedFactoryForInt32
{
    fnT get()
    {
        fnT fn = masked_extract_some_slices_strided_impl<T, std::int32_t>;
        return fn;
    }
};

template <typename fnT, typename T>
struct MaskExtractSomeSlicesStridedFactoryForInt64
{
    fnT get()
    {
        fnT fn = masked_extract_some_slices_strided_impl<T, std::int64_t>;
        return fn;
    }
};

// Masked placement

template <typename OrthoIndexerT,
          typename MaskedDstIndexerT,
          typename MaskedRhsIndexerT,
          typename dataT,
          typename indT>
class masked_place_all_slices_strided_impl_krn;

typedef sycl::event (*masked_place_all_slices_strided_impl_fn_ptr_t)(
    sycl::queue &,
    ssize_t,
    char *,
    const char *,
    const char *,
    int,
    ssize_t const *,
    ssize_t,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename dataT, typename indT>
sycl::event masked_place_all_slices_strided_impl(
    sycl::queue &exec_q,
    ssize_t iteration_size,
    char *dst_p,
    const char *cumsum_p,
    const char *rhs_p,
    int nd,
    const ssize_t
        *packed_dst_shape_strides, // [dst_shape, dst_strides], length 2*nd
    ssize_t rhs_size,              // rhs is 1D
    ssize_t rhs_stride,
    const std::vector<sycl::event> &depends = {})
{
    static constexpr TwoZeroOffsets_Indexer orthog_dst_rhs_indexer{};

    /* StridedIndexer(int _nd, ssize_t _offset, ssize_t const
     * *_packed_shape_strides) */
    const StridedIndexer masked_dst_indexer(nd, 0, packed_dst_shape_strides);
    const Strided1DCyclicIndexer masked_rhs_indexer(0, rhs_size, rhs_stride);

    using KernelName = class masked_place_all_slices_strided_impl_krn<
        TwoZeroOffsets_Indexer, StridedIndexer, Strided1DCyclicIndexer, dataT,
        indT>;

    static constexpr std::size_t nominal_lws = 256;
    const std::size_t masked_extent = iteration_size;
    const std::size_t lws = std::min(masked_extent, nominal_lws);

    const std::size_t n_groups = (masked_extent + lws - 1) / lws;

    sycl::range<2> gRange{1, n_groups * lws};
    sycl::range<2> lRange{1, lws};
    sycl::nd_range<2> ndRange{gRange, lRange};

    using LocalAccessorT = sycl::local_accessor<indT, 1>;
    using Impl =
        MaskedPlaceStridedFunctor<TwoZeroOffsets_Indexer, StridedIndexer,
                                  Strided1DCyclicIndexer, dataT, indT,
                                  LocalAccessorT>;

    dataT *dst_tp = reinterpret_cast<dataT *>(dst_p);
    const dataT *rhs_tp = reinterpret_cast<const dataT *>(rhs_p);
    const indT *cumsum_tp = reinterpret_cast<const indT *>(cumsum_p);

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        const std::size_t lacc_size = std::min(masked_extent, lws) + 1;
        LocalAccessorT lacc(lacc_size, cgh);

        cgh.parallel_for<KernelName>(
            ndRange, Impl(dst_tp, cumsum_tp, rhs_tp, iteration_size,
                          orthog_dst_rhs_indexer, masked_dst_indexer,
                          masked_rhs_indexer, lacc));
    });

    return comp_ev;
}

typedef sycl::event (*masked_place_some_slices_strided_impl_fn_ptr_t)(
    sycl::queue &,
    ssize_t,
    ssize_t,
    char *,
    const char *,
    const char *,
    int,
    ssize_t const *,
    ssize_t,
    ssize_t,
    int,
    ssize_t const *,
    ssize_t,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename OrthoIndexerT,
          typename MaskedSrcIndexerT,
          typename MaskedDstIndexerT,
          typename dataT,
          typename indT>
class masked_place_some_slices_strided_impl_krn;

template <typename dataT, typename indT>
sycl::event masked_place_some_slices_strided_impl(
    sycl::queue &exec_q,
    ssize_t orthog_nelems,
    ssize_t masked_nelems,
    char *dst_p,
    const char *cumsum_p,
    const char *rhs_p,
    int orthog_nd,
    // [ortho_shape, ortho_dst_strides, ortho_rhs_strides],
    // length 3*ortho_nd
    const ssize_t *packed_ortho_dst_rhs_shape_strides,
    ssize_t ortho_dst_offset,
    ssize_t ortho_rhs_offset,
    int masked_nd,
    // [masked_dst_shape, masked_dst_strides],
    // length 2*masked_nd, mask_dst is 1D
    const ssize_t *packed_masked_dst_shape_strides,
    ssize_t masked_rhs_size,
    ssize_t masked_rhs_stride,
    const std::vector<sycl::event> &depends = {})
{
    const TwoOffsets_StridedIndexer orthog_dst_rhs_indexer{
        orthog_nd, ortho_dst_offset, ortho_rhs_offset,
        packed_ortho_dst_rhs_shape_strides};

    /* StridedIndexer(int _nd, ssize_t _offset, ssize_t const
     * *_packed_shape_strides) */
    const StridedIndexer masked_dst_indexer{masked_nd, 0,
                                            packed_masked_dst_shape_strides};
    const Strided1DCyclicIndexer masked_rhs_indexer{0, masked_rhs_size,
                                                    masked_rhs_stride};

    using KernelName = class masked_place_some_slices_strided_impl_krn<
        TwoOffsets_StridedIndexer, StridedIndexer, Strided1DCyclicIndexer,
        dataT, indT>;

    static constexpr std::size_t nominal_lws = 256;
    const std::size_t orthog_extent = orthog_nelems;
    const std::size_t masked_extent = masked_nelems;
    const std::size_t lws = std::min(masked_extent, nominal_lws);

    const std::size_t n_groups = (masked_extent + lws - 1) / lws;

    sycl::range<2> gRange{orthog_extent, n_groups * lws};
    sycl::range<2> lRange{1, lws};
    sycl::nd_range<2> ndRange{gRange, lRange};

    using LocalAccessorT = sycl::local_accessor<indT, 1>;
    using Impl =
        MaskedPlaceStridedFunctor<TwoOffsets_StridedIndexer, StridedIndexer,
                                  Strided1DCyclicIndexer, dataT, indT,
                                  LocalAccessorT>;

    dataT *dst_tp = reinterpret_cast<dataT *>(dst_p);
    const dataT *rhs_tp = reinterpret_cast<const dataT *>(rhs_p);
    const indT *cumsum_tp = reinterpret_cast<const indT *>(cumsum_p);

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        const std::size_t lacc_size = std::min(masked_extent, lws) + 1;
        LocalAccessorT lacc(lacc_size, cgh);

        cgh.parallel_for<KernelName>(
            ndRange, Impl(dst_tp, cumsum_tp, rhs_tp, masked_nelems,
                          orthog_dst_rhs_indexer, masked_dst_indexer,
                          masked_rhs_indexer, lacc));
    });

    return comp_ev;
}

template <typename fnT, typename T>
struct MaskPlaceAllSlicesStridedFactoryForInt32
{
    fnT get()
    {
        fnT fn = masked_place_all_slices_strided_impl<T, std::int32_t>;
        return fn;
    }
};

template <typename fnT, typename T>
struct MaskPlaceAllSlicesStridedFactoryForInt64
{
    fnT get()
    {
        fnT fn = masked_place_all_slices_strided_impl<T, std::int64_t>;
        return fn;
    }
};

template <typename fnT, typename T>
struct MaskPlaceSomeSlicesStridedFactoryForInt32
{
    fnT get()
    {
        fnT fn = masked_place_some_slices_strided_impl<T, std::int32_t>;
        return fn;
    }
};

template <typename fnT, typename T>
struct MaskPlaceSomeSlicesStridedFactoryForInt64
{
    fnT get()
    {
        fnT fn = masked_place_some_slices_strided_impl<T, std::int64_t>;
        return fn;
    }
};

// Non-zero

template <typename T1, typename T2>
class non_zero_indexes_krn;

typedef sycl::event (*non_zero_indexes_fn_ptr_t)(
    sycl::queue &,
    ssize_t,
    ssize_t,
    int,
    const char *,
    char *,
    const ssize_t *,
    std::vector<sycl::event> const &);

template <typename indT1, typename indT2>
sycl::event non_zero_indexes_impl(sycl::queue &exec_q,
                                  ssize_t iter_size,
                                  ssize_t nz_elems,
                                  int nd,
                                  const char *cumsum_cp,
                                  char *indexes_cp,
                                  const ssize_t *mask_shape,
                                  std::vector<sycl::event> const &depends)
{
    const indT1 *cumsum_data = reinterpret_cast<const indT1 *>(cumsum_cp);
    indT2 *indexes_data = reinterpret_cast<indT2 *>(indexes_cp);

    static constexpr std::size_t nominal_lws = 256u;
    const std::size_t masked_extent = iter_size;
    const std::size_t lws = std::min(masked_extent, nominal_lws);

    const std::size_t n_groups = (masked_extent + lws - 1) / lws;
    sycl::range<1> gRange{n_groups * lws};
    sycl::range<1> lRange{lws};

    sycl::nd_range<1> ndRange{gRange, lRange};

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        const std::size_t lacc_size = std::min(lws, masked_extent) + 1;
        sycl::local_accessor<indT1, 1> lacc(lacc_size, cgh);

        using KernelName = class non_zero_indexes_krn<indT1, indT2>;

        cgh.parallel_for<KernelName>(ndRange, [=](sycl::nd_item<1> ndit) {
            const std::size_t group_i = ndit.get_group(0);
            const std::uint32_t l_i = ndit.get_local_id(0);
            const std::uint32_t lws = ndit.get_local_range(0);

            const std::size_t masked_block_start = group_i * lws;

            for (std::uint32_t i = l_i; i < lacc.size(); i += lws) {
                const std::size_t offset = masked_block_start + i;
                lacc[i] = (offset == 0) ? indT1(0)
                          : (offset - 1 < masked_extent)
                              ? cumsum_data[offset - 1]
                              : cumsum_data[masked_extent - 1] + 1;
            }

            sycl::group_barrier(ndit.get_group());

            const std::size_t i = masked_block_start + l_i;
            const auto cs_val = lacc[l_i];
            const bool cond = (lacc[l_i + 1] == cs_val + 1);

            if (cond && (i < masked_extent)) {
                ssize_t i_ = static_cast<ssize_t>(i);
                for (int dim = nd; --dim > 0;) {
                    const auto sd = mask_shape[dim];
                    const ssize_t q = i_ / sd;
                    const ssize_t r = (i_ - q * sd);
                    indexes_data[cs_val + dim * nz_elems] =
                        static_cast<indT2>(r);
                    i_ = q;
                }
                indexes_data[cs_val] = static_cast<indT2>(i_);
            }
        });
    });

    return comp_ev;
}

} // namespace dpctl::tensor::kernels::indexing
