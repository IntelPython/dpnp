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
/// This file defines kernels for tensor topk operation.
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

#include <sycl/sycl.hpp>

#include "kernels/sorting/merge_sort.hpp"
#include "kernels/sorting/radix_sort.hpp"
#include "kernels/sorting/search_sorted_detail.hpp"
#include "kernels/sorting/sort_utils.hpp"
#include "utils/sycl_alloc_utils.hpp"

namespace dpctl::tensor::kernels
{

namespace topk_detail
{

void scale_topk_params(const std::uint64_t nelems_per_slm,
                       const std::size_t sub_groups_per_work_group,
                       const std::uint32_t elems_per_wi,
                       const std::vector<std::size_t> &sg_sizes,
                       std::size_t &lws,
                       std::size_t &nelems_wg_sorts)
{
    for (auto it = sg_sizes.rbegin(); it != sg_sizes.rend(); ++it) {
        auto sg_size = *it;
        lws = sub_groups_per_work_group * sg_size;
        nelems_wg_sorts = elems_per_wi * lws;
        if (nelems_wg_sorts < nelems_per_slm) {
            return;
        }
    }
    // should never reach
    throw std::runtime_error("Could not construct top k kernel parameters");
}

template <class KernelName, typename argTy, typename IndexTy>
sycl::event write_out_impl(sycl::queue &exec_q,
                           std::size_t iter_nelems,
                           std::size_t k,
                           const argTy *arg_tp,
                           const IndexTy *index_data,
                           std::size_t iter_index_stride,
                           std::size_t axis_nelems,
                           argTy *vals_tp,
                           IndexTy *inds_tp,
                           const std::vector<sycl::event> &depends)
{
    static constexpr std::uint32_t lws = 64;
    static constexpr std::uint32_t n_wi = 4;
    const std::size_t nelems = iter_nelems * k;
    const std::size_t n_groups = (nelems + lws * n_wi - 1) / (n_wi * lws);

    sycl::range<1> lRange{lws};
    sycl::range<1> gRange{n_groups * lws};
    sycl::nd_range<1> ndRange{gRange, lRange};

    sycl::event write_out_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        cgh.parallel_for<KernelName>(ndRange, [=](sycl::nd_item<1> it) {
            const std::size_t gid = it.get_global_linear_id();
            const auto &sg = it.get_sub_group();
            const std::uint32_t lane_id = sg.get_local_id()[0];
            const std::uint32_t sg_size = sg.get_max_local_range()[0];

            const std::size_t start_id = (gid - lane_id) * n_wi + lane_id;

#pragma unroll
            for (std::uint32_t i = 0; i < n_wi; ++i) {
                const std::size_t data_id = start_id + i * sg_size;

                if (data_id < nelems) {
                    const std::size_t iter_id = data_id / k;

                    /*
                    const std::size_t axis_gid = data_id - (iter_gid * k);
                    const std::size_t src_idx = iter_gid * iter_index_stride +
                    axis_gid;
                    */
                    const std::size_t src_idx =
                        data_id + iter_id * (iter_index_stride - k);

                    const IndexTy res_ind = index_data[src_idx];
                    const argTy v = arg_tp[res_ind];

                    const std::size_t dst_idx = data_id;
                    vals_tp[dst_idx] = v;
                    inds_tp[dst_idx] = (res_ind % axis_nelems);
                }
            }
        });
    });

    return write_out_ev;
}

} // namespace topk_detail

template <typename T1, typename T2>
class topk_populate_index_data_krn;

template <typename T1, typename T2>
class topk_full_merge_map_back_krn;

template <typename argTy, typename IndexTy, typename CompT>
sycl::event
    topk_full_merge_sort_impl(sycl::queue &exec_q,
                              std::size_t iter_nelems, // number of sub-arrays
                              std::size_t axis_nelems, // size of each sub-array
                              std::size_t k,
                              const argTy *arg_tp,
                              argTy *vals_tp,
                              IndexTy *inds_tp,
                              const CompT &comp,
                              const std::vector<sycl::event> &depends)
{
    auto index_data_owner =
        dpctl::tensor::alloc_utils::smart_malloc_device<IndexTy>(
            iter_nelems * axis_nelems, exec_q);
    // extract USM pointer
    IndexTy *index_data = index_data_owner.get();

    using IotaKernelName = topk_populate_index_data_krn<argTy, IndexTy>;

    using dpctl::tensor::kernels::sort_utils_detail::iota_impl;

    sycl::event populate_indexed_data_ev = iota_impl<IotaKernelName, IndexTy>(
        exec_q, index_data, iter_nelems * axis_nelems, depends);

    std::size_t sorted_block_size;
    // Sort segments of the array
    sycl::event base_sort_ev =
        merge_sort_detail::sort_over_work_group_contig_impl(
            exec_q, iter_nelems, axis_nelems, index_data, index_data, comp,
            sorted_block_size, // modified in place with size of sorted block
                               // size
            {populate_indexed_data_ev});

    // Merge segments in parallel until all elements are sorted
    sycl::event merges_ev = merge_sort_detail::merge_sorted_block_contig_impl(
        exec_q, iter_nelems, axis_nelems, index_data, comp, sorted_block_size,
        {base_sort_ev});

    using WriteOutKernelName = topk_full_merge_map_back_krn<argTy, IndexTy>;

    sycl::event write_out_ev =
        topk_detail::write_out_impl<WriteOutKernelName, argTy, IndexTy>(
            exec_q, iter_nelems, k, arg_tp, index_data, axis_nelems,
            axis_nelems, vals_tp, inds_tp, {merges_ev});

    sycl::event cleanup_host_task_event =
        dpctl::tensor::alloc_utils::async_smart_free(exec_q, {write_out_ev},
                                                     index_data_owner);

    return cleanup_host_task_event;
};

template <typename T1, typename T2>
class topk_partial_merge_map_back_krn;

template <typename T1, typename T2, typename Comp>
class topk_over_work_group_krn;

template <typename argTy,
          typename IndexTy,
          typename ValueComp = std::less<argTy>>
sycl::event topk_merge_impl(
    sycl::queue &exec_q,
    std::size_t iter_nelems, // number of sub-arrays to sort (num. of rows
                             // in a matrix when sorting over rows)
    std::size_t axis_nelems, // size of each array to sort  (length of
                             // rows, i.e. number of columns)
    std::size_t k,
    const char *arg_cp,
    char *vals_cp,
    char *inds_cp,
    const std::vector<sycl::event> &depends)
{
    if (axis_nelems < k) {
        throw std::runtime_error("Invalid sort axis size for value of k");
    }

    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp);
    argTy *vals_tp = reinterpret_cast<argTy *>(vals_cp);
    IndexTy *inds_tp = reinterpret_cast<IndexTy *>(inds_cp);

    using dpctl::tensor::kernels::IndexComp;
    const IndexComp<IndexTy, argTy, ValueComp> index_comp{arg_tp, ValueComp{}};

    if (axis_nelems <= 512 || k >= 1024 || k > axis_nelems / 2) {
        return topk_full_merge_sort_impl(exec_q, iter_nelems, axis_nelems, k,
                                         arg_tp, vals_tp, inds_tp, index_comp,
                                         depends);
    }
    else {
        using PartialKernelName =
            topk_over_work_group_krn<IndexTy, IndexTy, ValueComp>;

        const auto &kernel_id = sycl::get_kernel_id<PartialKernelName>();

        auto const &ctx = exec_q.get_context();
        auto const &dev = exec_q.get_device();

        auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            ctx, {dev}, {kernel_id});

        auto krn = kb.get_kernel(kernel_id);

        const std::uint32_t max_sg_size = krn.template get_info<
            sycl::info::kernel_device_specific::max_sub_group_size>(dev);
        const std::uint64_t device_local_memory_size =
            dev.get_info<sycl::info::device::local_mem_size>();

        //  leave 512 bytes of local memory for RT
        const std::uint64_t safety_margin = 512;

        const std::uint64_t nelems_per_slm =
            (device_local_memory_size - safety_margin) / (2 * sizeof(IndexTy));

        static constexpr std::uint32_t sub_groups_per_work_group = 4;
        const std::uint32_t elems_per_wi = dev.has(sycl::aspect::cpu) ? 8 : 2;

        std::size_t lws = sub_groups_per_work_group * max_sg_size;

        std::size_t sorted_block_size = elems_per_wi * lws;
        if (sorted_block_size > nelems_per_slm) {
            const std::vector<std::size_t> sg_sizes =
                dev.get_info<sycl::info::device::sub_group_sizes>();
            topk_detail::scale_topk_params(
                nelems_per_slm, sub_groups_per_work_group, elems_per_wi,
                sg_sizes,
                lws,              // modified by reference
                sorted_block_size // modified by reference
            );
        }

        // This assumption permits doing away with using a loop
        assert(sorted_block_size % lws == 0);

        using search_sorted_detail::quotient_ceil;
        const std::size_t n_segments =
            quotient_ceil<std::size_t>(axis_nelems, sorted_block_size);

        // round k up for the later merge kernel if necessary
        const std::size_t round_k_to = dev.has(sycl::aspect::cpu) ? 32 : 4;
        std::size_t k_rounded =
            (k < round_k_to)
                ? k
                : quotient_ceil<std::size_t>(k, round_k_to) * round_k_to;

        // get length of tail for alloc size
        auto rem = axis_nelems % sorted_block_size;
        auto alloc_len = (rem && rem < k_rounded)
                             ? rem + k_rounded * (n_segments - 1)
                             : k_rounded * n_segments;

        // if allocation would be sufficiently large or k is larger than
        // elements processed, use full sort
        if (k_rounded >= axis_nelems || k_rounded >= sorted_block_size ||
            alloc_len >= axis_nelems / 2) {
            return topk_full_merge_sort_impl(exec_q, iter_nelems, axis_nelems,
                                             k, arg_tp, vals_tp, inds_tp,
                                             index_comp, depends);
        }

        auto index_data_owner =
            dpctl::tensor::alloc_utils::smart_malloc_device<IndexTy>(
                iter_nelems * alloc_len, exec_q);
        // get raw USM pointer
        IndexTy *index_data = index_data_owner.get();

        // no need to populate index data: SLM will be populated with default
        // values

        sycl::event base_sort_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            cgh.use_kernel_bundle(kb);

            sycl::range<1> global_range{iter_nelems * n_segments * lws};
            sycl::range<1> local_range{lws};

            sycl::range<1> slm_range{sorted_block_size};
            sycl::local_accessor<IndexTy, 1> work_space(slm_range, cgh);
            sycl::local_accessor<IndexTy, 1> scratch_space(slm_range, cgh);

            sycl::nd_range<1> ndRange(global_range, local_range);

            cgh.parallel_for<PartialKernelName>(
                ndRange, [=](sycl::nd_item<1> it) {
                    const std::size_t group_id = it.get_group_linear_id();
                    const std::size_t iter_id = group_id / n_segments;
                    const std::size_t segment_id =
                        group_id - iter_id * n_segments;
                    const std::size_t lid = it.get_local_linear_id();

                    const std::size_t segment_start_idx =
                        segment_id * sorted_block_size;
                    const std::size_t segment_end_idx = std::min<std::size_t>(
                        segment_start_idx + sorted_block_size, axis_nelems);
                    const std::size_t wg_chunk_size =
                        segment_end_idx - segment_start_idx;

                    // load input into SLM
                    for (std::size_t array_id = segment_start_idx + lid;
                         array_id < segment_end_idx; array_id += lws) {
                        IndexTy v = (array_id < axis_nelems)
                                        ? iter_id * axis_nelems + array_id
                                        : IndexTy{};
                        work_space[array_id - segment_start_idx] = v;
                    }
                    sycl::group_barrier(it.get_group());

                    const std::size_t chunk =
                        quotient_ceil<std::size_t>(sorted_block_size, lws);

                    const std::size_t chunk_start_idx = lid * chunk;
                    const std::size_t chunk_end_idx =
                        sycl::min(chunk_start_idx + chunk, wg_chunk_size);

                    merge_sort_detail::leaf_sort_impl(
                        work_space, chunk_start_idx, chunk_end_idx, index_comp);

                    sycl::group_barrier(it.get_group());

                    bool data_in_temp = false;
                    std::size_t n_chunks_merged = 1;

                    // merge chunk while n_chunks_merged * chunk < wg_chunk_size
                    const std::size_t max_chunks_merged =
                        1 + ((wg_chunk_size - 1) / chunk);
                    for (; n_chunks_merged < max_chunks_merged;
                         data_in_temp = !data_in_temp, n_chunks_merged *= 2) {
                        const std::size_t nelems_sorted_so_far =
                            n_chunks_merged * chunk;
                        const std::size_t q = (lid / n_chunks_merged);
                        const std::size_t start_1 = sycl::min(
                            2 * nelems_sorted_so_far * q, wg_chunk_size);
                        const std::size_t end_1 = sycl::min(
                            start_1 + nelems_sorted_so_far, wg_chunk_size);
                        const std::size_t end_2 = sycl::min(
                            end_1 + nelems_sorted_so_far, wg_chunk_size);
                        const std::size_t offset =
                            chunk * (lid - q * n_chunks_merged);

                        if (data_in_temp) {
                            merge_sort_detail::merge_impl(
                                offset, scratch_space, work_space, start_1,
                                end_1, end_2, start_1, index_comp, chunk);
                        }
                        else {
                            merge_sort_detail::merge_impl(
                                offset, work_space, scratch_space, start_1,
                                end_1, end_2, start_1, index_comp, chunk);
                        }
                        sycl::group_barrier(it.get_group());
                    }

                    // output assumed to be structured as (iter_nelems,
                    // alloc_len)
                    const std::size_t k_segment_start_idx =
                        segment_id * k_rounded;
                    const std::size_t k_segment_end_idx = std::min<std::size_t>(
                        k_segment_start_idx + k_rounded, alloc_len);
                    const auto &out_src =
                        (data_in_temp) ? scratch_space : work_space;
                    for (std::size_t array_id = k_segment_start_idx + lid;
                         array_id < k_segment_end_idx; array_id += lws) {
                        if (lid < k_rounded) {
                            index_data[iter_id * alloc_len + array_id] =
                                out_src[array_id - k_segment_start_idx];
                        }
                    }
                });
        });

        // Merge segments in parallel until all elements are sorted
        sycl::event merges_ev =
            merge_sort_detail::merge_sorted_block_contig_impl(
                exec_q, iter_nelems, alloc_len, index_data, index_comp,
                k_rounded, {base_sort_ev});

        // Write out top k of the merge-sorted memory
        using WriteOutKernelName =
            topk_partial_merge_map_back_krn<argTy, IndexTy>;

        sycl::event write_topk_ev =
            topk_detail::write_out_impl<WriteOutKernelName, argTy, IndexTy>(
                exec_q, iter_nelems, k, arg_tp, index_data, alloc_len,
                axis_nelems, vals_tp, inds_tp, {merges_ev});

        sycl::event cleanup_host_task_event =
            dpctl::tensor::alloc_utils::async_smart_free(
                exec_q, {write_topk_ev}, index_data_owner);

        return cleanup_host_task_event;
    }
}

template <typename T1, typename T2>
class topk_iota_krn;

template <typename T1, typename T2>
class topk_radix_map_back_krn;

template <typename argTy, typename IndexTy>
sycl::event topk_radix_impl(sycl::queue &exec_q,
                            std::size_t iter_nelems, // number of sub-arrays
                            std::size_t axis_nelems, // size of each sub-array
                            std::size_t k,
                            bool ascending,
                            const char *arg_cp,
                            char *vals_cp,
                            char *inds_cp,
                            const std::vector<sycl::event> &depends)
{
    if (axis_nelems < k) {
        throw std::runtime_error("Invalid sort axis size for value of k");
    }

    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp);
    argTy *vals_tp = reinterpret_cast<argTy *>(vals_cp);
    IndexTy *inds_tp = reinterpret_cast<IndexTy *>(inds_cp);

    const std::size_t total_nelems = iter_nelems * axis_nelems;
    const std::size_t padded_total_nelems = ((total_nelems + 63) / 64) * 64;
    auto workspace_owner =
        dpctl::tensor::alloc_utils::smart_malloc_device<IndexTy>(
            padded_total_nelems + total_nelems, exec_q);

    // get raw USM pointer
    IndexTy *workspace = workspace_owner.get();
    IndexTy *tmp_tp = workspace + padded_total_nelems;

    using IdentityProjT = radix_sort_details::IdentityProj;
    using IndexedProjT =
        radix_sort_details::IndexedProj<IndexTy, argTy, IdentityProjT>;
    const IndexedProjT proj_op{arg_tp};

    using IotaKernelName = topk_iota_krn<argTy, IndexTy>;

    using dpctl::tensor::kernels::sort_utils_detail::iota_impl;

    sycl::event iota_ev = iota_impl<IotaKernelName, IndexTy>(
        exec_q, workspace, total_nelems, depends);

    sycl::event radix_sort_ev =
        radix_sort_details::parallel_radix_sort_impl<IndexTy, IndexedProjT>(
            exec_q, iter_nelems, axis_nelems, workspace, tmp_tp, proj_op,
            ascending, {iota_ev});

    // Write out top k of the temporary
    using WriteOutKernelName = topk_radix_map_back_krn<argTy, IndexTy>;

    sycl::event write_topk_ev =
        topk_detail::write_out_impl<WriteOutKernelName, argTy, IndexTy>(
            exec_q, iter_nelems, k, arg_tp, tmp_tp, axis_nelems, axis_nelems,
            vals_tp, inds_tp, {radix_sort_ev});

    sycl::event cleanup_ev = dpctl::tensor::alloc_utils::async_smart_free(
        exec_q, {write_topk_ev}, workspace_owner);

    return cleanup_ev;
}

} // namespace dpctl::tensor::kernels
