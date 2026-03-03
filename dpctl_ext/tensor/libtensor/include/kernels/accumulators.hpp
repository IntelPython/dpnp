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
/// This file defines kernels for accumulators (cumulative sum, prod, etc.).
//===---------------------------------------------------------------------===//

#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpctl_tensor_types.hpp"
#include "utils/offset_utils.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/sycl_utils.hpp"
#include "utils/type_dispatch_building.hpp"
#include "utils/type_utils.hpp"

namespace dpctl::tensor::kernels::accumulators
{

namespace su_ns = dpctl::tensor::sycl_utils;

using dpctl::tensor::ssize_t;
using namespace dpctl::tensor::offset_utils;

template <typename T>
T ceiling_quotient(T n, T m)
{
    return (n + m - 1) / m;
}

template <typename inputT, typename outputT>
struct NonZeroIndicator
{
    constexpr NonZeroIndicator() {}

    outputT operator()(const inputT &val) const
    {
        static constexpr outputT out_one(1);
        static constexpr outputT out_zero(0);
        static constexpr inputT val_zero(0);

        return (val == val_zero) ? out_zero : out_one;
    }
};

template <typename T>
struct NoOpTransformer
{
    constexpr NoOpTransformer() {}

    T operator()(const T &val) const
    {
        return val;
    }
};

template <typename srcTy, typename dstTy>
struct CastTransformer
{
    constexpr CastTransformer() {}

    dstTy operator()(const srcTy &val) const
    {
        using dpctl::tensor::type_utils::convert_impl;
        return convert_impl<dstTy, srcTy>(val);
    }
};

template <typename ScanOpT, typename T>
struct needs_workaround
{
    // workaround needed due to crash in JITing on CPU
    // remove when CMPLRLLVM-65813 is resolved
    static constexpr bool value = su_ns::IsSyclLogicalAnd<T, ScanOpT>::value ||
                                  su_ns::IsSyclLogicalOr<T, ScanOpT>::value;
};

template <typename BinOpT, typename T>
struct can_use_inclusive_scan_over_group
{
    static constexpr bool value = sycl::has_known_identity<BinOpT, T>::value &&
                                  !needs_workaround<BinOpT, T>::value;
};

namespace detail
{
template <typename T>
class stack_t
{
    T *src_;
    std::size_t size_;
    T *local_scans_;

public:
    stack_t() : src_{}, size_{}, local_scans_{} {}
    stack_t(T *src, std::size_t sz, T *local_scans)
        : src_(src), size_(sz), local_scans_(local_scans)
    {
    }
    ~stack_t(){};

    T *get_src_ptr() const
    {
        return src_;
    }

    std::size_t get_size() const
    {
        return size_;
    }

    T *get_local_scans_ptr() const
    {
        return local_scans_;
    }
};

template <typename T>
class stack_strided_t
{
    T *src_;
    std::size_t size_;
    T *local_scans_;
    std::size_t local_stride_;

public:
    stack_strided_t() : src_{}, size_{}, local_scans_{}, local_stride_{} {}
    stack_strided_t(T *src,
                    std::size_t sz,
                    T *local_scans,
                    std::size_t local_stride)
        : src_(src), size_(sz), local_scans_(local_scans),
          local_stride_(local_stride)
    {
    }
    ~stack_strided_t(){};

    T *get_src_ptr() const
    {
        return src_;
    }

    std::size_t get_size() const
    {
        return size_;
    }

    T *get_local_scans_ptr() const
    {
        return local_scans_;
    }

    std::size_t get_local_stride() const
    {
        return local_stride_;
    }
};

} // end of namespace detail

// Iterative cumulative summation

using nwiT = std::uint32_t;

template <typename inputT,
          typename outputT,
          nwiT n_wi,
          typename IterIndexerT,
          typename InpIndexerT,
          typename OutIndexerT,
          typename TransformerT,
          typename ScanOpT,
          bool include_initial>
class inclusive_scan_iter_local_scan_blocked_krn;

template <typename inputT,
          typename outputT,
          nwiT n_wi,
          typename IterIndexerT,
          typename InpIndexerT,
          typename OutIndexerT,
          typename TransformerT,
          typename ScanOpT,
          bool include_initial>
class inclusive_scan_iter_local_scan_striped_krn;

template <typename inputT,
          typename outputT,
          nwiT n_wi,
          typename IterIndexerT,
          typename InpIndexerT,
          typename OutIndexerT,
          typename TransformerT,
          typename ScanOpT,
          bool include_initial = false>
sycl::event inclusive_scan_base_step_blocked(
    sycl::queue &exec_q,
    const std::uint32_t wg_size,
    const std::size_t iter_nelems,
    const std::size_t acc_nelems,
    const inputT *input,
    outputT *output,
    const std::size_t s0,
    const std::size_t s1,
    const IterIndexerT &iter_indexer,
    const InpIndexerT &inp_indexer,
    const OutIndexerT &out_indexer,
    TransformerT transformer,
    const ScanOpT &scan_op,
    outputT identity,
    std::size_t &acc_groups,
    const std::vector<sycl::event> &depends = {})
{
    acc_groups = ceiling_quotient<std::size_t>(acc_nelems, n_wi * wg_size);

    sycl::event inc_scan_phase1_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using slmT = sycl::local_accessor<outputT, 1>;

        auto gws = sycl::range<1>(iter_nelems * acc_groups * wg_size);
        auto lws = sycl::range<1>(wg_size);

        auto ndRange = sycl::nd_range<1>(gws, lws);

        slmT slm_iscan_tmp(lws, cgh);

        using KernelName = inclusive_scan_iter_local_scan_blocked_krn<
            inputT, outputT, n_wi, IterIndexerT, InpIndexerT, OutIndexerT,
            TransformerT, ScanOpT, include_initial>;

        cgh.parallel_for<KernelName>(ndRange, [=, slm_iscan_tmp =
                                                      std::move(slm_iscan_tmp)](
                                                  sycl::nd_item<1> it) {
            const std::size_t gid = it.get_global_id(0);
            const std::size_t lid = it.get_local_id(0);

            const std::uint32_t wg_size = it.get_local_range(0);
            const std::size_t reduce_chunks = acc_groups * wg_size;
            const std::size_t iter_gid = gid / reduce_chunks;
            const std::size_t chunk_gid = gid - (iter_gid * reduce_chunks);

            const std::size_t i = chunk_gid * n_wi;
            const auto &iter_offsets = iter_indexer(iter_gid);
            const auto &inp_iter_offset = iter_offsets.get_first_offset();
            const auto &out_iter_offset = iter_offsets.get_second_offset();

            std::array<outputT, n_wi> local_iscan;

#pragma unroll
            for (nwiT m_wi = 0; m_wi < n_wi; ++m_wi) {
                const std::size_t i_m_wi = i + m_wi;
                if constexpr (!include_initial) {
                    local_iscan[m_wi] =
                        (i_m_wi < acc_nelems)
                            ? transformer(input[inp_iter_offset +
                                                inp_indexer(s0 + s1 * i_m_wi)])
                            : identity;
                }
                else {
                    // shift input to the left by a single element relative to
                    // output
                    local_iscan[m_wi] =
                        (i_m_wi < acc_nelems && i_m_wi > 0)
                            ? transformer(
                                  input[inp_iter_offset +
                                        inp_indexer((s0 + s1 * i_m_wi) - 1)])
                            : identity;
                }
            }

#pragma unroll
            for (nwiT m_wi = 1; m_wi < n_wi; ++m_wi) {
                local_iscan[m_wi] =
                    scan_op(local_iscan[m_wi], local_iscan[m_wi - 1]);
            }
            // local_iscan is now result of
            // inclusive scan of locally stored inputs

            outputT wg_iscan_val;
            if constexpr (can_use_inclusive_scan_over_group<ScanOpT,
                                                            outputT>::value) {
                wg_iscan_val = sycl::inclusive_scan_over_group(
                    it.get_group(), local_iscan.back(), scan_op, identity);
            }
            else {
                wg_iscan_val = su_ns::custom_inclusive_scan_over_group(
                    it.get_group(), it.get_sub_group(), slm_iscan_tmp,
                    local_iscan.back(), identity, scan_op);
                // ensure all finished reading from SLM, to avoid race condition
                // with subsequent writes into SLM
                it.barrier(sycl::access::fence_space::local_space);
            }

            slm_iscan_tmp[(lid + 1) % wg_size] = wg_iscan_val;
            it.barrier(sycl::access::fence_space::local_space);
            const outputT modifier = (lid == 0) ? identity : slm_iscan_tmp[lid];

#pragma unroll
            for (nwiT m_wi = 0; m_wi < n_wi; ++m_wi) {
                local_iscan[m_wi] = scan_op(local_iscan[m_wi], modifier);
            }

            const std::size_t start = std::min(i, acc_nelems);
            const std::size_t end = std::min(i + n_wi, acc_nelems);
            const nwiT m_max = static_cast<nwiT>(end - start);
            for (nwiT m_wi = 0; m_wi < m_max; ++m_wi) {
                output[out_iter_offset + out_indexer(i + m_wi)] =
                    local_iscan[m_wi];
            }
        });
    });

    return inc_scan_phase1_ev;
}

template <typename inputT,
          typename outputT,
          nwiT n_wi,
          typename IterIndexerT,
          typename InpIndexerT,
          typename OutIndexerT,
          typename TransformerT,
          typename ScanOpT,
          bool include_initial = false>
sycl::event inclusive_scan_base_step_striped(
    sycl::queue &exec_q,
    const std::uint32_t wg_size,
    const std::size_t iter_nelems,
    const std::size_t acc_nelems,
    const inputT *input,
    outputT *output,
    const std::size_t s0,
    const std::size_t s1,
    const IterIndexerT &iter_indexer,
    const InpIndexerT &inp_indexer,
    const OutIndexerT &out_indexer,
    TransformerT transformer,
    const ScanOpT &scan_op,
    outputT identity,
    std::size_t &acc_groups,
    const std::vector<sycl::event> &depends = {})
{
    const std::uint32_t reduce_nelems_per_wg = n_wi * wg_size;
    acc_groups =
        ceiling_quotient<std::size_t>(acc_nelems, reduce_nelems_per_wg);

    sycl::event inc_scan_phase1_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using slmT = sycl::local_accessor<outputT, 1>;

        const auto &gRange = sycl::range<1>{iter_nelems * acc_groups * wg_size};
        const auto &lRange = sycl::range<1>{wg_size};

        const auto &ndRange = sycl::nd_range<1>{gRange, lRange};

        slmT slm_iscan_tmp(reduce_nelems_per_wg, cgh);

        using KernelName = inclusive_scan_iter_local_scan_striped_krn<
            inputT, outputT, n_wi, IterIndexerT, InpIndexerT, OutIndexerT,
            TransformerT, ScanOpT, include_initial>;

        cgh.parallel_for<KernelName>(ndRange, [=, slm_iscan_tmp =
                                                      std::move(slm_iscan_tmp)](
                                                  sycl::nd_item<1> it) {
            const std::uint32_t lid = it.get_local_linear_id();
            const std::uint32_t wg_size = it.get_local_range(0);

            const auto &sg = it.get_sub_group();
            const std::uint32_t sgSize = sg.get_max_local_range()[0];
            const std::size_t sgroup_id = sg.get_group_id()[0];
            const std::uint32_t lane_id = sg.get_local_id()[0];

            const std::size_t flat_group_id = it.get_group(0);
            const std::size_t iter_gid = flat_group_id / acc_groups;
            const std::size_t acc_group_id =
                flat_group_id - (iter_gid * acc_groups);

            const auto &iter_offsets = iter_indexer(iter_gid);
            const auto &inp_iter_offset = iter_offsets.get_first_offset();
            const auto &out_iter_offset = iter_offsets.get_second_offset();

            std::array<outputT, n_wi> local_iscan{};

            const std::size_t inp_id0 = acc_group_id * n_wi * wg_size +
                                        sgroup_id * n_wi * sgSize + lane_id;

#pragma unroll
            for (nwiT m_wi = 0; m_wi < n_wi; ++m_wi) {
                const std::size_t inp_id = inp_id0 + m_wi * sgSize;
                if constexpr (!include_initial) {
                    local_iscan[m_wi] =
                        (inp_id < acc_nelems)
                            ? transformer(input[inp_iter_offset +
                                                inp_indexer(s0 + s1 * inp_id)])
                            : identity;
                }
                else {
                    // shift input to the left by a single element relative to
                    // output
                    local_iscan[m_wi] =
                        (inp_id < acc_nelems && inp_id > 0)
                            ? transformer(
                                  input[inp_iter_offset +
                                        inp_indexer((s0 + s1 * inp_id) - 1)])
                            : identity;
                }
            }

            // change layout from striped to blocked
            {
                {
                    const std::uint32_t local_offset0 = lid * n_wi;
#pragma unroll
                    for (std::uint32_t i = 0; i < n_wi; ++i) {
                        slm_iscan_tmp[local_offset0 + i] = local_iscan[i];
                    }

                    it.barrier(sycl::access::fence_space::local_space);
                }

                {
                    const std::uint32_t block_offset =
                        sgroup_id * sgSize * n_wi;
                    const std::uint32_t disp0 = lane_id * n_wi;
#pragma unroll
                    for (nwiT i = 0; i < n_wi; ++i) {
                        const std::uint32_t disp = disp0 + i;

                        // disp == lane_id1 + i1 * sgSize;
                        const std::uint32_t i1 = disp / sgSize;
                        const std::uint32_t lane_id1 = disp - i1 * sgSize;

                        const std::uint32_t disp_exchanged =
                            (lane_id1 * n_wi + i1);

                        local_iscan[i] =
                            slm_iscan_tmp[block_offset + disp_exchanged];
                    }

                    it.barrier(sycl::access::fence_space::local_space);
                }
            }

#pragma unroll
            for (nwiT m_wi = 1; m_wi < n_wi; ++m_wi) {
                local_iscan[m_wi] =
                    scan_op(local_iscan[m_wi], local_iscan[m_wi - 1]);
            }
            // local_iscan is now result of
            // inclusive scan of locally stored inputs

            outputT wg_iscan_val;
            if constexpr (can_use_inclusive_scan_over_group<ScanOpT,
                                                            outputT>::value) {
                wg_iscan_val = sycl::inclusive_scan_over_group(
                    it.get_group(), local_iscan.back(), scan_op, identity);
            }
            else {
                wg_iscan_val = su_ns::custom_inclusive_scan_over_group(
                    it.get_group(), sg, slm_iscan_tmp, local_iscan.back(),
                    identity, scan_op);
                // ensure all finished reading from SLM, to avoid race condition
                // with subsequent writes into SLM
                it.barrier(sycl::access::fence_space::local_space);
            }

            slm_iscan_tmp[(lid + 1) % wg_size] = wg_iscan_val;
            it.barrier(sycl::access::fence_space::local_space);
            const outputT modifier = (lid == 0) ? identity : slm_iscan_tmp[lid];

#pragma unroll
            for (nwiT m_wi = 0; m_wi < n_wi; ++m_wi) {
                local_iscan[m_wi] = scan_op(local_iscan[m_wi], modifier);
            }

            it.barrier(sycl::access::fence_space::local_space);

            // convert back to blocked layout
            {{const std::uint32_t local_offset0 = lid * n_wi;
#pragma unroll
            for (nwiT m_wi = 0; m_wi < n_wi; ++m_wi) {
                slm_iscan_tmp[local_offset0 + m_wi] = local_iscan[m_wi];
            }

            it.barrier(sycl::access::fence_space::local_space);
                }
            }

            {
        const std::uint32_t block_offset = sgroup_id * sgSize * n_wi + lane_id;
#pragma unroll
        for (nwiT m_wi = 0; m_wi < n_wi; ++m_wi) {
            const std::uint32_t m_wi_scaled = m_wi * sgSize;
            const std::size_t out_id = inp_id0 + m_wi_scaled;
            if (out_id < acc_nelems) {
                output[out_iter_offset + out_indexer(out_id)] =
                    slm_iscan_tmp[block_offset + m_wi_scaled];
            }
        }
            }
});
});

return inc_scan_phase1_ev;
}

template <typename inputT,
          typename outputT,
          nwiT n_wi,
          typename IterIndexerT,
          typename InpIndexerT,
          typename OutIndexerT,
          typename TransformerT,
          typename ScanOpT,
          bool include_initial = false>
sycl::event
    inclusive_scan_base_step(sycl::queue &exec_q,
                             const std::uint32_t wg_size,
                             const std::size_t iter_nelems,
                             const std::size_t acc_nelems,
                             const inputT *input,
                             outputT *output,
                             const std::size_t s0,
                             const std::size_t s1,
                             const IterIndexerT &iter_indexer,
                             const InpIndexerT &inp_indexer,
                             const OutIndexerT &out_indexer,
                             TransformerT transformer,
                             const ScanOpT &scan_op,
                             outputT identity,
                             std::size_t &acc_groups,
                             const std::vector<sycl::event> &depends = {})
{
    // For small stride use striped load/store.
    // Threshold value chosen experimentally.
    if (s1 <= 16) {
        return inclusive_scan_base_step_striped<
            inputT, outputT, n_wi, IterIndexerT, InpIndexerT, OutIndexerT,
            TransformerT, ScanOpT, include_initial>(
            exec_q, wg_size, iter_nelems, acc_nelems, input, output, s0, s1,
            iter_indexer, inp_indexer, out_indexer, transformer, scan_op,
            identity, acc_groups, depends);
    }
    else {
        return inclusive_scan_base_step_blocked<
            inputT, outputT, n_wi, IterIndexerT, InpIndexerT, OutIndexerT,
            TransformerT, ScanOpT, include_initial>(
            exec_q, wg_size, iter_nelems, acc_nelems, input, output, s0, s1,
            iter_indexer, inp_indexer, out_indexer, transformer, scan_op,
            identity, acc_groups, depends);
    }
}

template <typename outputT, nwiT n_wi, typename ScanOpT>
class inclusive_scan_1d_iter_chunk_update_krn;

template <typename UpdateKernelName,
          typename outputT,
          nwiT n_wi,
          typename ScanOpT>
sycl::event update_local_chunks_1d(sycl::queue &exec_q,
                                   outputT *src,
                                   std::size_t src_size,
                                   const outputT *local_scans,
                                   std::size_t chunk_size,
                                   const sycl::event &dependent_event)
{
    const auto &ctx = exec_q.get_context();
    const auto &dev = exec_q.get_device();

    const auto &kernel_id = sycl::get_kernel_id<UpdateKernelName>();
    auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
        ctx, {dev}, {kernel_id});
    auto krn = kb.get_kernel(kernel_id);

    const std::uint32_t sg_size = krn.template get_info<
        sycl::info::kernel_device_specific::max_sub_group_size>(dev);

    // output[ chunk_size * (i + 1) + j] += temp[i]
    sycl::event update_event = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependent_event);
        cgh.use_kernel_bundle(kb);

        static constexpr nwiT updates_per_wi = n_wi;
        const std::size_t n_items =
            ceiling_quotient<std::size_t>(src_size, sg_size * n_wi) * sg_size;

        sycl::range<1> gRange{n_items};
        sycl::range<1> lRange{sg_size};
        sycl::nd_range<1> ndRange{gRange, lRange};

        cgh.parallel_for<UpdateKernelName>(
            ndRange,
            [chunk_size, src, src_size, local_scans](sycl::nd_item<1> ndit) {
                static constexpr ScanOpT scan_op{};
                static constexpr outputT identity =
                    su_ns::Identity<ScanOpT, outputT>::value;

                const std::uint32_t lws = ndit.get_local_range(0);
                const std::size_t block_offset = ndit.get_group(0) * n_wi * lws;
#pragma unroll
                for (std::size_t i = 0; i < updates_per_wi; ++i) {
                    const std::size_t src_id =
                        block_offset + ndit.get_local_id(0) + i * lws;
                    if (src_id < src_size) {
                        const std::size_t scan_id = (src_id / chunk_size);
                        const outputT modifier =
                            (scan_id > 0) ? local_scans[scan_id - 1] : identity;
                        src[src_id] = scan_op(src[src_id], modifier);
                    }
                }
            });
    });

    return update_event;
}

/*
 * output[j] = sum( input[s0 + i * s1], 0 <= i <= j)
 * for 0 <= j < n_elems
 */
template <typename inputT,
          typename outputT,
          nwiT n_wi,
          typename IndexerT,
          typename TransformerT,
          typename ScanOpT,
          bool include_initial>
sycl::event inclusive_scan_iter_1d(sycl::queue &exec_q,
                                   const std::uint32_t wg_size,
                                   const std::size_t n_elems,
                                   const inputT *input,
                                   outputT *output,
                                   const std::size_t s0,
                                   const std::size_t s1,
                                   const IndexerT &indexer,
                                   const TransformerT &transformer,
                                   std::vector<sycl::event> &host_tasks,
                                   const std::vector<sycl::event> &depends = {})
{
    static constexpr ScanOpT scan_op{};
    static constexpr outputT identity =
        su_ns::Identity<ScanOpT, outputT>::value;

    static constexpr std::size_t _iter_nelems = 1;

    using IterIndexerT = dpctl::tensor::offset_utils::TwoZeroOffsets_Indexer;
    static constexpr IterIndexerT _no_op_iter_indexer{};

    using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
    static constexpr NoOpIndexerT _no_op_indexer{};

    std::size_t n_groups;
    sycl::event inc_scan_phase1_ev =
        inclusive_scan_base_step<inputT, outputT, n_wi, IterIndexerT, IndexerT,
                                 NoOpIndexerT, TransformerT, ScanOpT,
                                 include_initial>(
            exec_q, wg_size, _iter_nelems, n_elems, input, output, s0, s1,
            _no_op_iter_indexer, indexer, _no_op_indexer, transformer, scan_op,
            identity, n_groups, depends);

    sycl::event dependent_event = inc_scan_phase1_ev;
    if (n_groups > 1) {
        const std::size_t chunk_size = wg_size * n_wi;

        // how much of temporary allocation do we need
        std::size_t n_groups_ = n_groups;
        std::size_t temp_size = 0;
        while (n_groups_ > 1) {
            const std::size_t this_size = (n_groups_ - 1);
            temp_size += this_size;
            n_groups_ = ceiling_quotient(this_size, chunk_size);
        }

        // allocate
        auto temp_owner =
            dpctl::tensor::alloc_utils::smart_malloc_device<outputT>(temp_size,
                                                                     exec_q);
        outputT *temp = temp_owner.get();

        std::vector<detail::stack_t<outputT>> stack{};

        // inclusive scans over blocks
        n_groups_ = n_groups;
        outputT *src = output;
        outputT *local_scans = temp;

        using NoOpTransformerT = NoOpTransformer<outputT>;
        static constexpr NoOpTransformerT _no_op_transformer{};
        std::size_t size_to_update = n_elems;
        while (n_groups_ > 1) {

            const std::size_t src_size = n_groups_ - 1;
            dependent_event =
                inclusive_scan_base_step<outputT, outputT, n_wi, IterIndexerT,
                                         NoOpIndexerT, NoOpIndexerT,
                                         NoOpTransformerT, ScanOpT>(
                    exec_q, wg_size, _iter_nelems, src_size, src, local_scans,
                    chunk_size - 1, chunk_size, _no_op_iter_indexer,
                    _no_op_indexer, _no_op_indexer, _no_op_transformer, scan_op,
                    identity, n_groups_, // n_groups_ is modified in place
                    {dependent_event});
            stack.push_back({src, size_to_update, local_scans});
            src = local_scans;
            local_scans += src_size;
            size_to_update = src_size;
        }

        for (std::size_t reverse_stack_id = 0; reverse_stack_id < stack.size();
             ++reverse_stack_id)
        {
            const std::size_t stack_id = stack.size() - 1 - reverse_stack_id;

            const auto &stack_elem = stack[stack_id];
            outputT *src = stack_elem.get_src_ptr();
            const std::size_t src_size = stack_elem.get_size();
            const outputT *local_scans = stack_elem.get_local_scans_ptr();

            using UpdateKernelName =
                class inclusive_scan_1d_iter_chunk_update_krn<outputT, n_wi,
                                                              ScanOpT>;

            dependent_event = update_local_chunks_1d<UpdateKernelName, outputT,
                                                     n_wi, ScanOpT>(
                exec_q, src, src_size, local_scans, chunk_size,
                dependent_event);
        }

        sycl::event free_ev = dpctl::tensor::alloc_utils::async_smart_free(
            exec_q, {dependent_event}, temp_owner);

        host_tasks.push_back(free_ev);
    }

    return dependent_event;
}

typedef sycl::event (*accumulate_1d_contig_impl_fn_ptr_t)(
    sycl::queue &,
    std::size_t,
    const char *,
    char *,
    std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

template <typename srcT,
          typename dstT,
          typename transformerT,
          typename AccumulateOpT,
          bool include_initial>
sycl::event
    accumulate_1d_contig_impl(sycl::queue &q,
                              std::size_t n_elems,
                              const char *src,
                              char *dst,
                              std::vector<sycl::event> &host_tasks,
                              const std::vector<sycl::event> &depends = {})
{
    const srcT *src_data_ptr = reinterpret_cast<const srcT *>(src);
    dstT *dst_data_ptr = reinterpret_cast<dstT *>(dst);

    using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
    static constexpr NoOpIndexerT flat_indexer{};
    static constexpr transformerT transformer{};

    static constexpr std::size_t s0 = 0;
    static constexpr std::size_t s1 = 1;

    sycl::event comp_ev;
    const sycl::device &dev = q.get_device();
    if (dev.has(sycl::aspect::cpu)) {
        static constexpr nwiT n_wi_for_cpu = 8;
        const std::uint32_t wg_size = 256;
        comp_ev = inclusive_scan_iter_1d<srcT, dstT, n_wi_for_cpu, NoOpIndexerT,
                                         transformerT, AccumulateOpT,
                                         include_initial>(
            q, wg_size, n_elems, src_data_ptr, dst_data_ptr, s0, s1,
            flat_indexer, transformer, host_tasks, depends);
    }
    else {
        static constexpr nwiT n_wi_for_gpu = 4;
        // base_scan_striped algorithm does not execute correctly
        // on HIP device with wg_size > 64
        const std::uint32_t wg_size =
            (q.get_backend() == sycl::backend::ext_oneapi_hip) ? 64 : 256;
        comp_ev = inclusive_scan_iter_1d<srcT, dstT, n_wi_for_gpu, NoOpIndexerT,
                                         transformerT, AccumulateOpT,
                                         include_initial>(
            q, wg_size, n_elems, src_data_ptr, dst_data_ptr, s0, s1,
            flat_indexer, transformer, host_tasks, depends);
    }
    return comp_ev;
}

template <typename outputT,
          nwiT n_wi,
          typename IterIndexerT,
          typename IndexerT,
          typename ScanOpT>
class inclusive_scan_final_chunk_update_krn;

template <typename UpdateKernelName,
          typename outputT,
          nwiT n_wi,
          typename OutIterIndexerT,
          typename OutIndexerT,
          typename ScanOpT>
sycl::event final_update_local_chunks(sycl::queue &exec_q,
                                      std::size_t iter_nelems,
                                      outputT *src,
                                      std::size_t src_size,
                                      const outputT *local_scans,
                                      std::size_t chunk_size,
                                      std::size_t local_stride,
                                      const OutIterIndexerT &out_iter_indexer,
                                      const OutIndexerT &out_indexer,
                                      sycl::event dependent_event)
{
    const auto &kernel_id = sycl::get_kernel_id<UpdateKernelName>();

    auto const &ctx = exec_q.get_context();
    auto const &dev = exec_q.get_device();
    auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
        ctx, {dev}, {kernel_id});

    auto krn = kb.get_kernel(kernel_id);

    const std::uint32_t sg_size = krn.template get_info<
        sycl::info::kernel_device_specific::max_sub_group_size>(dev);

    static constexpr nwiT updates_per_wi = n_wi;
    const std::size_t updates_per_sg = sg_size * updates_per_wi;
    const std::size_t update_nelems =
        ceiling_quotient(src_size, updates_per_sg) * sg_size;

    sycl::range<2> gRange{iter_nelems, update_nelems};
    sycl::range<2> lRange{1, sg_size};

    sycl::nd_range<2> ndRange{gRange, lRange};

    sycl::event update_event = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependent_event);

        cgh.parallel_for<UpdateKernelName>(
            ndRange, [chunk_size, src_size, local_stride, src, local_scans,
                      out_iter_indexer, out_indexer](sycl::nd_item<2> ndit) {
                static constexpr ScanOpT scan_op{};
                static constexpr outputT identity =
                    su_ns::Identity<ScanOpT, outputT>::value;

                const std::uint32_t lws = ndit.get_local_range(1);

                const std::size_t iter_gid = ndit.get_group(0);

                const std::size_t src_axis_id0 =
                    ndit.get_group(1) * updates_per_wi * lws +
                    ndit.get_local_id(1);
                const std::size_t src_iter_id = out_iter_indexer(iter_gid);
#pragma unroll
                for (nwiT i = 0; i < updates_per_wi; ++i) {
                    const std::size_t src_axis_id = src_axis_id0 + i * lws;
                    const std::size_t src_id =
                        out_indexer(src_axis_id) + src_iter_id;

                    if (src_axis_id < src_size) {
                        const std::size_t scan_axis_id =
                            src_axis_id / chunk_size;
                        const std::size_t scan_id =
                            scan_axis_id + iter_gid * local_stride;

                        const outputT modifier = (scan_axis_id > 0)
                                                     ? local_scans[scan_id - 1]
                                                     : identity;

                        src[src_id] = scan_op(src[src_id], modifier);
                    }
                }
            });
    });

    return update_event;
}

template <typename outputT, nwiT n_wi, typename ScanOpT>
class inclusive_scan_iter_chunk_update_krn;

template <typename UpdateKernelName,
          typename outputT,
          nwiT n_wi,
          typename ScanOpT>
sycl::event update_local_chunks(sycl::queue &exec_q,
                                std::size_t iter_nelems,
                                outputT *src,
                                std::size_t src_size,
                                const outputT *local_scans,
                                std::size_t chunk_size,
                                std::size_t local_stride,
                                sycl::event dependent_event)
{
    static constexpr NoOpIndexer out_indexer{};
    static constexpr NoOpIndexer iter_out_indexer{};

    return final_update_local_chunks<UpdateKernelName, outputT, n_wi,
                                     NoOpIndexer, NoOpIndexer, ScanOpT>(
        exec_q, iter_nelems, src, src_size, local_scans, chunk_size,
        local_stride, iter_out_indexer, out_indexer, dependent_event);
}

template <typename inputT,
          typename outputT,
          nwiT n_wi,
          typename InpIterIndexerT,
          typename OutIterIndexerT,
          typename InpIndexerT,
          typename OutIndexerT,
          typename TransformerT,
          typename ScanOpT,
          bool include_initial>
sycl::event inclusive_scan_iter(sycl::queue &exec_q,
                                const std::uint32_t wg_size,
                                const std::size_t iter_nelems,
                                const std::size_t acc_nelems,
                                const inputT *input,
                                outputT *output,
                                const std::size_t s0,
                                const std::size_t s1,
                                const InpIterIndexerT &inp_iter_indexer,
                                const OutIterIndexerT &out_iter_indexer,
                                const InpIndexerT &inp_indexer,
                                const OutIndexerT &out_indexer,
                                const TransformerT &transformer,
                                std::vector<sycl::event> &host_tasks,
                                const std::vector<sycl::event> &depends = {})
{
    static constexpr ScanOpT scan_op{};
    static constexpr outputT identity =
        su_ns::Identity<ScanOpT, outputT>::value;

    using IterIndexerT =
        dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
            InpIterIndexerT, OutIterIndexerT>;
    const IterIndexerT iter_indexer{inp_iter_indexer, out_iter_indexer};

    std::size_t acc_groups;
    sycl::event inc_scan_phase1_ev =
        inclusive_scan_base_step<inputT, outputT, n_wi, IterIndexerT,
                                 InpIndexerT, OutIndexerT, TransformerT,
                                 ScanOpT, include_initial>(
            exec_q, wg_size, iter_nelems, acc_nelems, input, output, s0, s1,
            iter_indexer, inp_indexer, out_indexer, transformer, scan_op,
            identity, acc_groups, depends);

    sycl::event dependent_event = inc_scan_phase1_ev;
    if (acc_groups > 1) {
        const std::size_t chunk_size = wg_size * n_wi;

        // how much of temporary allocation do we need
        std::size_t acc_groups_ = acc_groups;
        std::size_t temp_size = 0;
        while (acc_groups_ > 1) {
            const std::size_t this_size = (acc_groups_ - 1);
            temp_size += this_size;
            acc_groups_ = ceiling_quotient<std::size_t>(this_size, chunk_size);
        }

        // allocate
        auto temp_owner =
            dpctl::tensor::alloc_utils::smart_malloc_device<outputT>(
                iter_nelems * temp_size, exec_q);
        outputT *temp = temp_owner.get();

        std::vector<detail::stack_strided_t<outputT>> stack{};

        // inclusive scans over blocks
        acc_groups_ = acc_groups;
        outputT *src = output;
        outputT *local_scans = temp;

        using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        static constexpr NoOpIndexerT _no_op_indexer{};
        using NoOpTransformerT = NoOpTransformer<outputT>;
        static constexpr NoOpTransformerT _no_op_transformer{};
        std::size_t size_to_update = acc_nelems;

        {
            std::size_t src_size = acc_groups - 1;
            using LocalScanIndexerT =
                dpctl::tensor::offset_utils::Strided1DIndexer;
            const LocalScanIndexerT scan_iter_indexer{/* size */ iter_nelems,
                                                      /* step */ src_size};

            using IterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    OutIterIndexerT, LocalScanIndexerT>;
            const IterIndexerT iter_indexer_{out_iter_indexer,
                                             scan_iter_indexer};

            dependent_event =
                inclusive_scan_base_step<outputT, outputT, n_wi, IterIndexerT,
                                         OutIndexerT, NoOpIndexerT,
                                         NoOpTransformerT, ScanOpT>(
                    exec_q, wg_size, iter_nelems, src_size, src, local_scans,
                    chunk_size - 1, chunk_size, iter_indexer_, out_indexer,
                    _no_op_indexer, _no_op_transformer, scan_op, identity,
                    acc_groups_, // acc_groups_ is modified in place
                    {dependent_event});
            stack.push_back({src, size_to_update, local_scans, src_size});
            src = local_scans;
            local_scans += src_size * iter_nelems;
            size_to_update = src_size;
        }

        while (acc_groups_ > 1) {
            std::size_t src_size = acc_groups_ - 1;

            using LocalScanIndexerT =
                dpctl::tensor::offset_utils::Strided1DIndexer;
            const LocalScanIndexerT scan1_iter_indexer{
                /* size */ iter_nelems,
                /* step */ size_to_update};
            const LocalScanIndexerT scan2_iter_indexer{/* size */ iter_nelems,
                                                       /* step */ src_size};

            using IterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    LocalScanIndexerT, LocalScanIndexerT>;
            const IterIndexerT iter_indexer_{scan1_iter_indexer,
                                             scan2_iter_indexer};

            dependent_event =
                inclusive_scan_base_step<outputT, outputT, n_wi, IterIndexerT,
                                         NoOpIndexerT, NoOpIndexerT,
                                         NoOpTransformerT, ScanOpT>(
                    exec_q, wg_size, iter_nelems, src_size, src, local_scans,
                    chunk_size - 1, chunk_size, iter_indexer_, _no_op_indexer,
                    _no_op_indexer, _no_op_transformer, scan_op, identity,
                    acc_groups_, // acc_groups_ is modified in place
                    {dependent_event});
            stack.push_back({src, size_to_update, local_scans, src_size});
            src = local_scans;
            local_scans += src_size * iter_nelems;
            size_to_update = src_size;
        }

        for (std::size_t reverse_stack_id = 0;
             reverse_stack_id < stack.size() - 1; ++reverse_stack_id)
        {
            const std::size_t stack_id = stack.size() - 1 - reverse_stack_id;

            const auto &stack_elem = stack[stack_id];
            outputT *src = stack_elem.get_src_ptr();
            std::size_t src_size = stack_elem.get_size();
            outputT *local_scans = stack_elem.get_local_scans_ptr();
            std::size_t local_stride = stack_elem.get_local_stride();

            using UpdateKernelName =
                class inclusive_scan_iter_chunk_update_krn<outputT, n_wi,
                                                           ScanOpT>;

            dependent_event =
                update_local_chunks<UpdateKernelName, outputT, n_wi, ScanOpT>(
                    exec_q, iter_nelems, src, src_size, local_scans, chunk_size,
                    local_stride, dependent_event);
        }

        // last stack element is always directly to output
        {
            const auto &stack_elem = stack[0];
            outputT *src = stack_elem.get_src_ptr();
            const std::size_t src_size = stack_elem.get_size();
            outputT *local_scans = stack_elem.get_local_scans_ptr();
            const std::size_t local_stride = stack_elem.get_local_stride();

            using UpdateKernelName =
                class inclusive_scan_final_chunk_update_krn<
                    outputT, n_wi, OutIterIndexerT, OutIndexerT, ScanOpT>;

            dependent_event =
                final_update_local_chunks<UpdateKernelName, outputT, n_wi,
                                          OutIterIndexerT, OutIndexerT,
                                          ScanOpT>(
                    exec_q, iter_nelems, src, src_size, local_scans, chunk_size,
                    local_stride, out_iter_indexer, out_indexer,
                    dependent_event);
        }

        sycl::event free_ev = dpctl::tensor::alloc_utils::async_smart_free(
            exec_q, {dependent_event}, temp_owner);
        host_tasks.push_back(free_ev);
    }

    return dependent_event;
}

typedef sycl::event (*accumulate_strided_impl_fn_ptr_t)(
    sycl::queue &,
    std::size_t,
    std::size_t,
    const char *,
    int,
    const ssize_t *,
    ssize_t,
    ssize_t,
    int,
    const ssize_t *,
    char *,
    std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

template <typename srcT,
          typename dstT,
          typename transformerT,
          typename AccumulateOpT,
          bool include_initial>
sycl::event
    accumulate_strided_impl(sycl::queue &q,
                            std::size_t iter_nelems,
                            std::size_t acc_nelems,
                            const char *src,
                            int iter_nd,
                            const ssize_t *iter_shape_strides,
                            ssize_t inp_iter_offset,
                            ssize_t out_iter_offset,
                            int acc_nd,
                            const ssize_t *acc_shape_strides,
                            char *dst,
                            std::vector<sycl::event> &host_tasks,
                            const std::vector<sycl::event> &depends = {})
{
    const srcT *src_data_ptr = reinterpret_cast<const srcT *>(src);
    dstT *dst_data_ptr = reinterpret_cast<dstT *>(dst);

    using InpIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
    const InpIndexerT inp_axis_indexer{acc_nd, 0, acc_shape_strides};
    const InpIndexerT inp_iter_indexer{iter_nd, inp_iter_offset,
                                       iter_shape_strides};

    using OutIndexerT = dpctl::tensor::offset_utils::UnpackedStridedIndexer;
    const OutIndexerT out_axis_indexer{acc_nd, 0, acc_shape_strides,
                                       acc_shape_strides + 2 * acc_nd};
    const OutIndexerT out_iter_indexer{iter_nd, out_iter_offset,
                                       iter_shape_strides,
                                       iter_shape_strides + 2 * iter_nd};

    static constexpr transformerT transformer{};

    static constexpr std::size_t s0 = 0;
    static constexpr std::size_t s1 = 1;

    const sycl::device &dev = q.get_device();
    sycl::event comp_ev;
    if (dev.has(sycl::aspect::cpu)) {
        static constexpr nwiT n_wi_for_cpu = 8;
        const std::uint32_t wg_size = 256;
        comp_ev =
            inclusive_scan_iter<srcT, dstT, n_wi_for_cpu, InpIndexerT,
                                OutIndexerT, InpIndexerT, OutIndexerT,
                                transformerT, AccumulateOpT, include_initial>(
                q, wg_size, iter_nelems, acc_nelems, src_data_ptr, dst_data_ptr,
                s0, s1, inp_iter_indexer, out_iter_indexer, inp_axis_indexer,
                out_axis_indexer, transformer, host_tasks, depends);
    }
    else {
        static constexpr nwiT n_wi_for_gpu = 4;
        // base_scan_striped algorithm does not execute correctly
        // on HIP device with wg_size > 64
        const std::uint32_t wg_size =
            (q.get_backend() == sycl::backend::ext_oneapi_hip) ? 64 : 256;
        comp_ev =
            inclusive_scan_iter<srcT, dstT, n_wi_for_gpu, InpIndexerT,
                                OutIndexerT, InpIndexerT, OutIndexerT,
                                transformerT, AccumulateOpT, include_initial>(
                q, wg_size, iter_nelems, acc_nelems, src_data_ptr, dst_data_ptr,
                s0, s1, inp_iter_indexer, out_iter_indexer, inp_axis_indexer,
                out_axis_indexer, transformer, host_tasks, depends);
    }

    return comp_ev;
}

typedef std::size_t (*cumsum_val_contig_impl_fn_ptr_t)(
    sycl::queue &,
    std::size_t,
    const char *,
    char *,
    std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

template <typename maskT, typename cumsumT, typename transformerT>
std::size_t cumsum_val_contig_impl(sycl::queue &q,
                                   std::size_t n_elems,
                                   const char *mask,
                                   char *cumsum,
                                   std::vector<sycl::event> &host_tasks,
                                   const std::vector<sycl::event> &depends = {})
{
    const maskT *mask_data_ptr = reinterpret_cast<const maskT *>(mask);
    cumsumT *cumsum_data_ptr = reinterpret_cast<cumsumT *>(cumsum);

    using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
    static constexpr NoOpIndexerT flat_indexer{};
    static constexpr transformerT transformer{};

    static constexpr std::size_t s0 = 0;
    static constexpr std::size_t s1 = 1;
    static constexpr bool include_initial = false;
    using AccumulateOpT = sycl::plus<cumsumT>;

    sycl::event comp_ev;
    const sycl::device &dev = q.get_device();
    if (dev.has(sycl::aspect::cpu)) {
        static constexpr nwiT n_wi_for_cpu = 8;
        const std::uint32_t wg_size = 256;
        comp_ev = inclusive_scan_iter_1d<maskT, cumsumT, n_wi_for_cpu,
                                         NoOpIndexerT, transformerT,
                                         AccumulateOpT, include_initial>(
            q, wg_size, n_elems, mask_data_ptr, cumsum_data_ptr, s0, s1,
            flat_indexer, transformer, host_tasks, depends);
    }
    else {
        static constexpr nwiT n_wi_for_gpu = 4;
        // base_scan_striped algorithm does not execute correctly
        // on HIP device with wg_size > 64
        const std::uint32_t wg_size =
            (q.get_backend() == sycl::backend::ext_oneapi_hip) ? 64 : 256;
        comp_ev = inclusive_scan_iter_1d<maskT, cumsumT, n_wi_for_gpu,
                                         NoOpIndexerT, transformerT,
                                         AccumulateOpT, include_initial>(
            q, wg_size, n_elems, mask_data_ptr, cumsum_data_ptr, s0, s1,
            flat_indexer, transformer, host_tasks, depends);
    }
    cumsumT *last_elem = cumsum_data_ptr + (n_elems - 1);

    auto host_usm_owner =
        dpctl::tensor::alloc_utils::smart_malloc_host<cumsumT>(1, q);
    cumsumT *last_elem_host_usm = host_usm_owner.get();

    sycl::event copy_e = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(comp_ev);
        cgh.copy<cumsumT>(last_elem, last_elem_host_usm, 1);
    });
    copy_e.wait();
    std::size_t return_val = static_cast<std::size_t>(*last_elem_host_usm);

    // explicitly free USM host allocation, by invoking deleter of
    // the unique_ptr
    host_usm_owner.reset(nullptr);

    return return_val;
}

template <typename fnT, typename T>
struct MaskPositionsContigFactoryForInt32
{
    fnT get()
    {
        using cumsumT = std::int32_t;
        fnT fn =
            cumsum_val_contig_impl<T, cumsumT, NonZeroIndicator<T, cumsumT>>;
        return fn;
    }
};

template <typename fnT, typename T>
struct MaskPositionsContigFactoryForInt64
{
    fnT get()
    {
        using cumsumT = std::int64_t;
        fnT fn =
            cumsum_val_contig_impl<T, cumsumT, NonZeroIndicator<T, cumsumT>>;
        return fn;
    }
};

template <typename fnT, typename T>
struct Cumsum1DContigFactory
{
    fnT get()
    {
        if constexpr (std::is_integral_v<T>) {
            using cumsumT = std::int64_t;
            fnT fn =
                cumsum_val_contig_impl<T, cumsumT, NoOpTransformer<cumsumT>>;
            return fn;
        }
        else {
            return nullptr;
        }
    }
};

typedef std::size_t (*cumsum_val_strided_impl_fn_ptr_t)(
    sycl::queue &,
    std::size_t,
    const char *,
    int,
    const ssize_t *,
    char *,
    std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

template <typename maskT, typename cumsumT, typename transformerT>
std::size_t
    cumsum_val_strided_impl(sycl::queue &q,
                            std::size_t n_elems,
                            const char *mask,
                            int nd,
                            const ssize_t *shape_strides,
                            char *cumsum,
                            std::vector<sycl::event> &host_tasks,
                            const std::vector<sycl::event> &depends = {})
{
    const maskT *mask_data_ptr = reinterpret_cast<const maskT *>(mask);
    cumsumT *cumsum_data_ptr = reinterpret_cast<cumsumT *>(cumsum);

    using StridedIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
    const StridedIndexerT strided_indexer{nd, 0, shape_strides};
    static constexpr transformerT transformer{};

    static constexpr std::size_t s0 = 0;
    static constexpr std::size_t s1 = 1;
    static constexpr bool include_initial = false;
    using AccumulateOpT = sycl::plus<cumsumT>;

    const sycl::device &dev = q.get_device();
    sycl::event comp_ev;
    if (dev.has(sycl::aspect::cpu)) {
        static constexpr nwiT n_wi_for_cpu = 8;
        const std::uint32_t wg_size = 256;
        comp_ev = inclusive_scan_iter_1d<maskT, cumsumT, n_wi_for_cpu,
                                         StridedIndexerT, transformerT,
                                         AccumulateOpT, include_initial>(
            q, wg_size, n_elems, mask_data_ptr, cumsum_data_ptr, s0, s1,
            strided_indexer, transformer, host_tasks, depends);
    }
    else {
        static constexpr nwiT n_wi_for_gpu = 4;
        // base_scan_striped algorithm does not execute correctly
        // on HIP device with wg_size > 64
        const std::uint32_t wg_size =
            (q.get_backend() == sycl::backend::ext_oneapi_hip) ? 64 : 256;
        comp_ev = inclusive_scan_iter_1d<maskT, cumsumT, n_wi_for_gpu,
                                         StridedIndexerT, transformerT,
                                         AccumulateOpT, include_initial>(
            q, wg_size, n_elems, mask_data_ptr, cumsum_data_ptr, s0, s1,
            strided_indexer, transformer, host_tasks, depends);
    }

    cumsumT *last_elem = cumsum_data_ptr + (n_elems - 1);

    auto host_usm_owner =
        dpctl::tensor::alloc_utils::smart_malloc_host<cumsumT>(1, q);
    cumsumT *last_elem_host_usm = host_usm_owner.get();

    sycl::event copy_e = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(comp_ev);
        cgh.copy<cumsumT>(last_elem, last_elem_host_usm, 1);
    });
    copy_e.wait();
    std::size_t return_val = static_cast<std::size_t>(*last_elem_host_usm);

    // explicitly free USM-host temporary, by invoking deleter of
    // the unique_ptr
    host_usm_owner.reset(nullptr);

    return return_val;
}

template <typename fnT, typename T>
struct MaskPositionsStridedFactoryForInt32
{
    fnT get()
    {
        using cumsumT = std::int32_t;
        fnT fn =
            cumsum_val_strided_impl<T, cumsumT, NonZeroIndicator<T, cumsumT>>;
        return fn;
    }
};

template <typename fnT, typename T>
struct MaskPositionsStridedFactoryForInt64
{
    fnT get()
    {
        using cumsumT = std::int64_t;
        fnT fn =
            cumsum_val_strided_impl<T, cumsumT, NonZeroIndicator<T, cumsumT>>;
        return fn;
    }
};

template <typename fnT, typename T>
struct Cumsum1DStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_integral_v<T>) {
            using cumsumT = std::int64_t;
            fnT fn =
                cumsum_val_strided_impl<T, cumsumT, NoOpTransformer<cumsumT>>;
            return fn;
        }
        else {
            return nullptr;
        }
    }
};

} // namespace dpctl::tensor::kernels::accumulators
