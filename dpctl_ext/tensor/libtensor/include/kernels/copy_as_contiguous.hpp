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
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines kernels for tensor copying and value casting.
//===----------------------------------------------------------------------===//

#pragma once
#include <complex>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "dpctl_tensor_types.hpp"
#include "kernels/alignment.hpp"
#include "utils/offset_utils.hpp"
#include "utils/sycl_utils.hpp"
#include "utils/type_utils.hpp"

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace copy_as_contig
{

using dpctl::tensor::ssize_t;
using dpctl::tensor::sycl_utils::sub_group_store;

template <typename T,
          typename IndexerT,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
class CopyAsCContigFunctor
{
private:
    std::size_t nelems;
    const T *src_p = nullptr;
    T *dst_p = nullptr;
    IndexerT src_indexer;

public:
    CopyAsCContigFunctor(std::size_t n,
                         const T *src_,
                         T *dst_,
                         const IndexerT &src_indexer_)
        : nelems(n), src_p(src_), dst_p(dst_), src_indexer(src_indexer_)
    {
    }

    void operator()(sycl::nd_item<1> ndit) const
    {
        static_assert(vec_sz > 0);
        static_assert(n_vecs > 0);

        static constexpr std::uint8_t elems_per_wi = vec_sz * n_vecs;

        using dpctl::tensor::type_utils::is_complex;
        if constexpr (!enable_sg_loadstore || is_complex<T>::value) {
            const std::uint16_t sgSize =
                ndit.get_sub_group().get_max_local_range()[0];
            const std::size_t gid = ndit.get_global_linear_id();

            // start = (gid / sgSize) * sgSize * elems_per_wi + (gid % sgSize)
            // gid % sgSize == gid - (gid / sgSize) * sgSize
            const std::uint16_t elems_per_sg = sgSize * elems_per_wi;
            const std::size_t start =
                (gid / sgSize) * (elems_per_sg - sgSize) + gid;
            const std::size_t end = std::min(nelems, start + elems_per_sg);

            for (std::size_t offset = start; offset < end; offset += sgSize) {
                auto src_offset = src_indexer(offset);
                dst_p[offset] = src_p[src_offset];
            }
        }
        else {
            auto sg = ndit.get_sub_group();
            const std::uint16_t sgSize = sg.get_max_local_range()[0];
            const std::size_t base =
                elems_per_wi * (ndit.get_group(0) * ndit.get_local_range(0) +
                                sg.get_group_id()[0] * sgSize);
            const std::uint16_t elems_per_sg = elems_per_wi * sgSize;

            if (base + elems_per_sg < nelems) {
#pragma unroll
                for (std::uint8_t it = 0; it < elems_per_wi; it += vec_sz) {
                    // it == vec_id * vec_sz, for  0 <= vec_id < n_vecs
                    const std::size_t block_start_id = base + it * sgSize;
                    auto dst_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&dst_p[block_start_id]);

                    const std::size_t elem_id0 =
                        block_start_id + sg.get_local_id();
                    sycl::vec<T, vec_sz> dst_vec;
#pragma unroll
                    for (std::uint8_t k = 0; k < vec_sz; ++k) {
                        const std::size_t elem_id = elem_id0 + k * sgSize;
                        const ssize_t src_offset = src_indexer(elem_id);
                        dst_vec[k] = src_p[src_offset];
                    }
                    sub_group_store<vec_sz>(sg, dst_vec, dst_multi_ptr);
                }
            }
            else {
                const std::size_t lane_id = sg.get_local_id()[0];
                const std::size_t k0 = base + lane_id;
                for (std::size_t k = k0; k < nelems; k += sgSize) {
                    const ssize_t src_offset = src_indexer(k);
                    dst_p[k] = src_p[src_offset];
                }
            }
        }
    }
};

template <typename T,
          typename IndexerT,
          std::uint8_t vec_sz,
          std::uint8_t n_vecs,
          bool enable_sg_load,
          typename KernelName>
sycl::event submit_c_contiguous_copy(sycl::queue &exec_q,
                                     std::size_t nelems,
                                     const T *src,
                                     T *dst,
                                     const IndexerT &src_indexer,
                                     const std::vector<sycl::event> &depends)
{
    static_assert(vec_sz > 0);
    static_assert(n_vecs > 0);

    static constexpr std::size_t preferred_lws = 256;

    const auto &kernel_id = sycl::get_kernel_id<KernelName>();

    auto const &ctx = exec_q.get_context();
    auto const &dev = exec_q.get_device();
    auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
        ctx, {dev}, {kernel_id});

    auto krn = kb.get_kernel(kernel_id);

    const std::uint32_t max_sg_size = krn.template get_info<
        sycl::info::kernel_device_specific::max_sub_group_size>(dev);

    const std::size_t lws =
        ((preferred_lws + max_sg_size - 1) / max_sg_size) * max_sg_size;

    static constexpr std::uint8_t nelems_per_wi = n_vecs * vec_sz;

    const std::size_t nelems_per_group = nelems_per_wi * lws;
    const std::size_t n_groups =
        (nelems + nelems_per_group - 1) / (nelems_per_group);

    sycl::event copy_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.use_kernel_bundle(kb);

        const sycl::range<1> gRange{n_groups * lws};
        const sycl::range<1> lRange{lws};

        cgh.parallel_for<KernelName>(
            sycl::nd_range<1>(gRange, lRange),
            CopyAsCContigFunctor<T, IndexerT, vec_sz, n_vecs, enable_sg_load>(
                nelems, src, dst, src_indexer));
    });
    return copy_ev;
}

template <typename T,
          typename IndexT,
          std::uint8_t vec_sz,
          std::uint8_t n_vecs,
          bool enable_sgload>
class as_contig_krn;

template <typename T>
sycl::event
    as_c_contiguous_array_generic_impl(sycl::queue &exec_q,
                                       std::size_t nelems,
                                       int nd,
                                       const ssize_t *shape_and_strides,
                                       const char *src_p,
                                       char *dst_p,
                                       const std::vector<sycl::event> &depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<T>(exec_q);

    const T *src_tp = reinterpret_cast<const T *>(src_p);
    T *dst_tp = reinterpret_cast<T *>(dst_p);

    using IndexerT = dpctl::tensor::offset_utils::StridedIndexer;
    const IndexerT src_indexer(nd, ssize_t(0), shape_and_strides);

    static constexpr std::uint8_t vec_sz = 4u;
    static constexpr std::uint8_t n_vecs = 2u;

    using dpctl::tensor::kernels::alignment_utils::
        disabled_sg_loadstore_wrapper_krn;
    using dpctl::tensor::kernels::alignment_utils::is_aligned;
    using dpctl::tensor::kernels::alignment_utils::required_alignment;

    sycl::event copy_ev;
    if (is_aligned<required_alignment>(dst_p)) {
        static constexpr bool enable_sg_load = true;
        using KernelName =
            as_contig_krn<T, IndexerT, vec_sz, n_vecs, enable_sg_load>;
        copy_ev = submit_c_contiguous_copy<T, IndexerT, vec_sz, n_vecs,
                                           enable_sg_load, KernelName>(
            exec_q, nelems, src_tp, dst_tp, src_indexer, depends);
    }
    else {
        static constexpr bool disable_sg_load = false;
        using InnerKernelName =
            as_contig_krn<T, IndexerT, vec_sz, n_vecs, disable_sg_load>;
        using KernelName = disabled_sg_loadstore_wrapper_krn<InnerKernelName>;
        copy_ev = submit_c_contiguous_copy<T, IndexerT, vec_sz, n_vecs,
                                           disable_sg_load, KernelName>(
            exec_q, nelems, src_tp, dst_tp, src_indexer, depends);
    }

    return copy_ev;
}

typedef sycl::event (*as_c_contiguous_array_impl_fn_ptr_t)(
    sycl::queue &,
    std::size_t,
    int,
    const ssize_t *,
    const char *,
    char *,
    const std::vector<sycl::event> &);

template <typename fnT, typename T>
struct AsCContigFactory
{
    fnT get()
    {
        return as_c_contiguous_array_generic_impl<T>;
    }
};

template <typename T,
          typename IndexerT,
          std::uint16_t tile_size,
          std::uint16_t n_lines>
class as_contig_batch_of_square_matrices_krn;

namespace detail
{
/*! @brief batch of matrices (n, n), source strides (1, src_ld), destination
   strides (dst_ld, 1) src and destination arrays must be disjoint memory blocks
   to avoid race condition
 */
template <typename T, typename BatchIndexerT>
sycl::event as_c_contiguous_batch_of_square_matrices_impl(
    sycl::queue &exec_q,
    std::size_t batch_nelems,
    const BatchIndexerT &batch_two_offsets_indexer,
    std::size_t n,
    const char *src_p,
    ssize_t src_ld,
    char *dst_p,
    ssize_t dst_ld,
    const std::vector<sycl::event> &depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<T>(exec_q);

    const T *src_tp = reinterpret_cast<const T *>(src_p);
    T *dst_tp = reinterpret_cast<T *>(dst_p);

    static constexpr std::uint16_t private_tile_size = 4;
    static constexpr std::uint16_t n_lines = 2;
    static constexpr std::uint16_t block_size =
        n_lines * private_tile_size * private_tile_size;

    static constexpr std::uint16_t lws0 = block_size;
    static constexpr std::uint16_t lws1 = n_lines;
    static constexpr std::uint16_t nelems_per_wi = (block_size / lws1);

    static_assert(nelems_per_wi * lws1 == block_size);
    static_assert(nelems_per_wi == private_tile_size * private_tile_size);

    static constexpr std::uint32_t lws = lws0 * lws1;

    const std::size_t n_tiles = (n + block_size - 1) / block_size;

    const ssize_t src_stride = src_ld;
    const ssize_t dst_stride = dst_ld;

    sycl::range<1> lRange{lws};
    sycl::range<1> gRange{batch_nelems * n_tiles * n_tiles * lws};

    sycl::nd_range<1> ndRange{gRange, lRange};

    using KernelName =
        as_contig_batch_of_square_matrices_krn<T, BatchIndexerT,
                                               private_tile_size, lws1>;

    sycl::event e = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        sycl::local_accessor<T, 1> local_block(block_size * block_size, cgh);

        cgh.parallel_for<KernelName>(ndRange, [=](sycl::nd_item<1> nd_it) {
            // 1. Read block from source array into SLM
            const std::uint32_t lid_lin = nd_it.get_local_linear_id();
            const std::size_t gr_id_lin = nd_it.get_group_linear_id();

            const std::size_t batch_id = gr_id_lin / (n_tiles * n_tiles);
            const std::size_t rem = gr_id_lin - batch_id * (n_tiles * n_tiles);

            const auto &batch_two_offsets = batch_two_offsets_indexer(batch_id);
            const auto &src_batch_offset = batch_two_offsets.get_first_offset();
            const auto &dst_batch_offset =
                batch_two_offsets.get_second_offset();

            // Block id
            /* 0 <= src_gr_i1 < n_groups_n1 */
            const std::size_t src_tile_i1 = rem / n_tiles;
            /* 0 <= src_gr_i0 < n_groups_n0 */
            const std::size_t src_tile_i0 = rem - src_tile_i1 * n_tiles;

            // ID of element within the block
            /* 0 <= src_i1 < lws1 */
            const std::uint32_t src_i1 = lid_lin / lws0;
            /* 0 <= src_i0 < lws0 */
            const std::uint32_t src_i0 = lid_lin - src_i1 * lws0;

            // Matrix element ID
            const std::size_t src_tile_start0 = src_tile_i0 * block_size;
            const std::size_t src_tile_start1 = src_tile_i1 * block_size;
            const std::size_t src_gid0 = (src_tile_start0 + src_i0);
            const std::size_t src_gid1 = (src_tile_start1 + src_i1);

            // src_offset = src_gid0 * 1 + (src_gid1 + pr_id * lws1) *
            // src_stride
            const std::size_t src_offset0 =
                src_batch_offset + src_gid0 * 1 + src_gid1 * src_stride;
            const std::size_t pr_step_src = lws1 * src_stride;

            const std::uint32_t local_offset0 = src_i0 + src_i1 * block_size;
            const std::uint32_t pr_step_local = lws1 * block_size;

            for (std::uint32_t pr_id = 0; pr_id < nelems_per_wi; ++pr_id) {
                local_block[local_offset0 + pr_step_local * pr_id] =
                    (src_gid0 < n && src_gid1 + pr_id * lws1 < n)
                        ? src_tp[src_offset0 + pr_step_src * pr_id]
                        : T(0);
            }

            const std::uint32_t local_dim0 = static_cast<std::uint32_t>(
                std::min<std::size_t>(src_tile_start0 + block_size, n) -
                src_tile_start0);
            const std::uint32_t local_dim1 = static_cast<std::uint32_t>(
                std::min<std::size_t>(src_tile_start1 + block_size, n) -
                src_tile_start1);

            sycl::group_barrier(nd_it.get_group(),
                                sycl::memory_scope::work_group);

            // 2. Permute the block matrix in SLM using two private arrays
            std::array<T, nelems_per_wi> private_block_01 = {T(0)};
            std::array<T, nelems_per_wi> private_block_10 = {T(0)};

            // 0 <= lid_lin < lws0 * lws1 ==
            //       (block_size * block_size / nelems_per_wi) ==
            //       (block_size/private_tile_size)**2
            static constexpr std::uint16_t n_private_tiles_per_axis =
                block_size / private_tile_size;
            const std::uint16_t local_tile_id0 =
                lid_lin / n_private_tiles_per_axis;
            const std::uint16_t local_tile_id1 =
                lid_lin - local_tile_id0 * n_private_tiles_per_axis;

            if (local_tile_id0 <= local_tile_id1) {
                for (std::uint16_t pr_i0 = 0; pr_i0 < private_tile_size;
                     ++pr_i0) {
                    for (std::uint16_t pr_i1 = 0; pr_i1 < private_tile_size;
                         ++pr_i1) {
                        const std::uint16_t t0_offset =
                            local_tile_id0 * private_tile_size;
                        const std::uint16_t t1_offset =
                            local_tile_id1 * private_tile_size;

                        const std::uint16_t pr_offset =
                            pr_i1 * private_tile_size + pr_i0;
                        const std::uint16_t rel_offset =
                            pr_i0 + pr_i1 * block_size;

                        // read (local_tile_id0, local_tile_id1)
                        const std::uint16_t local_01_offset =
                            (t0_offset + t1_offset * block_size) + rel_offset;
                        private_block_01[pr_offset] =
                            local_block[local_01_offset];

                        // read (local_tile_id1, local_tile_id0)
                        const std::uint16_t local_10_offset =
                            (t1_offset + t0_offset * block_size) + rel_offset;
                        private_block_10[pr_offset] =
                            local_block[local_10_offset];
                    }
                }
            }

            sycl::group_barrier(nd_it.get_group(),
                                sycl::memory_scope::work_group);

            if (local_tile_id0 <= local_tile_id1) {
                for (std::uint16_t pr_i0 = 0; pr_i0 < private_tile_size;
                     ++pr_i0) {
                    for (std::uint16_t pr_i1 = 0; pr_i1 < private_tile_size;
                         ++pr_i1) {
                        const std::uint16_t t0_offset =
                            local_tile_id0 * private_tile_size;
                        const std::uint16_t t1_offset =
                            local_tile_id1 * private_tile_size;
                        const std::uint16_t pr_offset =
                            pr_i0 * private_tile_size + pr_i1;

                        const std::uint16_t rel_offset =
                            pr_i0 + pr_i1 * block_size;

                        // write back permuted private blocks
                        const std::uint32_t local_01_offset =
                            (t0_offset + t1_offset * block_size) + rel_offset;
                        local_block[local_01_offset] =
                            private_block_10[pr_offset];

                        const std::uint16_t local_10_offset =
                            (t1_offset + t0_offset * block_size) + rel_offset;
                        local_block[local_10_offset] =
                            private_block_01[pr_offset];
                    }
                }
            }

            sycl::group_barrier(nd_it.get_group(),
                                sycl::memory_scope::work_group);

            // 3. Write out permuted SLM to destination array

            const std::size_t dst_tile_start0 = src_tile_start0;
            const std::size_t dst_tile_start1 = src_tile_start1;

            if (local_dim0 == block_size && local_dim1 == block_size) {
                const std::uint16_t dst_i0 = src_i1;
                const std::uint16_t dst_i1 = src_i0;

                const std::size_t dst_gid0 = (dst_tile_start0 + dst_i0);
                const std::size_t dst_gid1 = (dst_tile_start1 + dst_i1);

                const std::size_t dst_offset0 =
                    dst_batch_offset + dst_gid0 * dst_stride + dst_gid1 * 1;
                const std::size_t pr_step_dst = lws1 * dst_stride;

                const std::uint16_t _local_offset0 =
                    dst_i0 * block_size + dst_i1;
                const std::uint16_t _pr_step_local = lws1 * block_size;

                for (std::uint16_t pr_id = 0; pr_id < nelems_per_wi; ++pr_id) {
                    if ((dst_gid1 < n) && ((dst_gid0 + pr_id * lws1) < n)) {
                        dst_tp[dst_offset0 + pr_step_dst * pr_id] =
                            local_block[_local_offset0 +
                                        _pr_step_local * pr_id];
                    }
                }
            }
            else {
                // map local_linear_id into (local_dim0, local_dim1)
                for (std::uint16_t el_id = lid_lin;
                     el_id < local_dim0 * local_dim1; el_id += lws0 * lws1)
                {

                    // 0 <= local_i0 < local_dim0
                    const std::uint16_t loc_i0 = el_id / local_dim1;
                    // 0 <= local_i1 < local_dim1
                    const std::uint16_t loc_i1 = el_id - loc_i0 * local_dim1;

                    const std::uint16_t dst_i0 = loc_i0;
                    const std::uint16_t dst_i1 = loc_i1;

                    const std::size_t dst_gid0 = (dst_tile_start0 + dst_i0);
                    const std::size_t dst_gid1 = (dst_tile_start1 + dst_i1);

                    const std::size_t dst_offset =
                        dst_batch_offset + dst_gid0 * dst_stride + dst_gid1 * 1;
                    const std::uint16_t local_offset =
                        loc_i0 * block_size + loc_i1;

                    if ((dst_gid1 < n) && (dst_gid0 < n)) {
                        dst_tp[dst_offset] = local_block[local_offset];
                    }
                }
            }
        });
    });

    return e;
}

} // end of namespace detail

template <typename T>
sycl::event as_c_contiguous_1d_batch_of_square_matrices_impl(
    sycl::queue &exec_q,
    std::size_t batch_nelems,
    ssize_t src_batch_step,
    ssize_t dst_batch_step,
    std::size_t n,
    const char *src_p,
    ssize_t src_ld,
    char *dst_p,
    ssize_t dst_ld,
    const std::vector<sycl::event> &depends)
{
    using dpctl::tensor::offset_utils::Strided1DIndexer;
    using dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer;
    using BatchIndexerT =
        TwoOffsets_CombinedIndexer<Strided1DIndexer, Strided1DIndexer>;

    const auto &src_batch_indexer =
        Strided1DIndexer(batch_nelems, src_batch_step);
    const auto &dst_batch_indexer =
        Strided1DIndexer(batch_nelems, dst_batch_step);

    const BatchIndexerT batch_two_indexer{src_batch_indexer, dst_batch_indexer};

    return detail::as_c_contiguous_batch_of_square_matrices_impl<T,
                                                                 BatchIndexerT>(
        exec_q, batch_nelems, batch_two_indexer, n, src_p, src_ld, dst_p,
        dst_ld, depends);
}

typedef sycl::event (
    *as_c_contiguous_1d_batch_of_square_matrices_impl_fn_ptr_t)(
    sycl::queue &, /* execution queue */
    std::size_t,   /* number of batch elements */
    ssize_t,       /* distance between batches in source array */
    ssize_t,       /* distance between batches in destination array */
    std::size_t,   /* size of square matrices in the batch */
    const char *,
    ssize_t, /* untyped pointer to F-contig source array, and matrix leading
                dimension */
    char *,
    ssize_t, /* untyped pointer to C-contig destination array, and matrix
                leading dimension */
    const std::vector<sycl::event> &);

template <typename fnT, typename T>
struct AsCContig1DBatchOfSquareMatricesFactory
{
    fnT get()
    {
        return as_c_contiguous_1d_batch_of_square_matrices_impl<T>;
    }
};

template <typename T>
sycl::event as_c_contiguous_nd_batch_of_square_matrices_impl(
    sycl::queue &exec_q,
    std::size_t batch_nelems,
    int batch_nd,
    const ssize_t *src_batch_shape_strides,
    const ssize_t dst_batch_step,
    std::size_t n,
    const char *src_p,
    ssize_t src_ld,
    char *dst_p,
    ssize_t dst_ld,
    const std::vector<sycl::event> &depends)
{
    using SrcIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
    using DstIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
    using dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer;
    using BatchIndexerT = TwoOffsets_CombinedIndexer<SrcIndexerT, DstIndexerT>;

    static constexpr ssize_t zero_offset{0};

    const SrcIndexerT src_batch_indexer{batch_nd, zero_offset,
                                        src_batch_shape_strides};
    const DstIndexerT dst_batch_indexer{/* size */ batch_nelems,
                                        /* step */ dst_batch_step};

    const BatchIndexerT batch_two_offsets_indexer{src_batch_indexer,
                                                  dst_batch_indexer};

    return detail::as_c_contiguous_batch_of_square_matrices_impl<T,
                                                                 BatchIndexerT>(
        exec_q, batch_nelems, batch_two_offsets_indexer, n, src_p, src_ld,
        dst_p, dst_ld, depends);
}

typedef sycl::event (
    *as_c_contiguous_nd_batch_of_square_matrices_impl_fn_ptr_t)(
    sycl::queue &, /* execution queue */
    std::size_t,   /* number of matrices in the batch */
    int,
    const ssize_t *, /* dimensionality, and packed [shape, src_strides]
                        describing iteration over batch in source array */
    ssize_t,         /* distance between batches in destination array */
    std::size_t,     /* matrix size */
    const char *,
    ssize_t, /* untyped pointer to source array of F-contig matrices, and
                leading dimension of the matrix */
    char *,
    ssize_t, /* untyped pointer to destination array of F-contig matrices, and
                leading dimension of the matrix */
    const std::vector<sycl::event> &);

template <typename fnT, typename T>
struct AsCContigNDBatchOfSquareMatricesFactory
{
    fnT get()
    {
        return as_c_contiguous_nd_batch_of_square_matrices_impl<T>;
    }
};

} // namespace copy_as_contig
} // namespace kernels
} // namespace tensor
} // namespace dpctl
