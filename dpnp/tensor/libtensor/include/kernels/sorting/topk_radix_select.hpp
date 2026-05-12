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
/// This file defines kernels for tensor topk using radix select algorithm.
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <vector>

#include "kernels/dpnp_tensor_types.hpp"
#include "radix_utils.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include <sycl/ext/oneapi/sub_group_mask.hpp>

namespace dpnp::tensor::kernels
{
namespace topk_detail
{

template <typename T>
T quotient_ceil(T n, T m)
{
    return (n + m - 1) / m;
}

template <typename AccT>
auto get_accessor_pointer(const AccT &acc)
{
    return acc.template get_multi_ptr<sycl::access::decorated::no>().get();
}

std::uint32_t mask_from_subgroup_ballot(sycl::sub_group sg, bool vote)
{
    auto sg_sz = sg.get_local_range()[0];
    if (sg_sz == 1) {
        return static_cast<std::uint32_t>(vote);
    }
    std::uint32_t mask;
    sycl::ext::oneapi::group_ballot(sg, vote).extract_bits(mask, 0);
    return mask;
}

template <typename T,
          typename BitwiseT,
          typename CountT,
          std::uint32_t radix_states,
          std::uint32_t radix_mask>
void count_radix(sycl::nd_item<1> &it,
                 const T *data,
                 CountT counts[radix_states],
                 CountT *slm,
                 BitwiseT desired,
                 BitwiseT desired_mask,
                 std::uint32_t radix_pos,
                 std::size_t axis_nelems)
{
#pragma unroll
    for (int i = 0; i < radix_states; ++i) {
        counts[i] = 0;
    }

    auto idx = it.get_local_id(0);
    if (idx < radix_states) {
        slm[idx] = 0;
    }

    sycl::group_barrier(it.get_group());

    auto g = it.get_sub_group();
    for (std::size_t i = idx; i < axis_nelems;) {
        BitwiseT val = radix_utils::RadixTypeConfig<true, T>::encode(data[i]);
        bool has_val = ((val & desired_mask) == desired);
        BitwiseT digit_in_radix =
            radix_utils::get_bucket_id<radix_mask>(val, radix_pos);
#pragma unroll
        for (std::size_t j = 0; j < radix_states; ++j) {
            bool vote = has_val && (digit_in_radix == j);
            std::uint32_t inner_mask = mask_from_subgroup_ballot(g, vote);
            counts[j] += sycl::popcount(inner_mask);
        }
        i += it.get_local_range(0);
    }
    // now sum values in each sub-group
    if (g.leader()) {
#pragma unroll
        for (std::size_t i = 0; i < radix_states; ++i) {
            sycl::atomic_ref<CountT, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::local_space>
                slm_ref(slm[i]);
            slm_ref += counts[i];
        }
    }

    sycl::group_barrier(it.get_group());

#pragma unroll
    for (std::size_t i = 0; i < radix_states; ++i) {
        counts[i] = slm[i];
    }

    sycl::group_barrier(it.get_group());
}

template <typename T, typename BitwiseT>
T find_pattern(sycl::nd_item<1> &it,
               const T *data,
               std::size_t axis_nelems,
               T *slm,
               BitwiseT desired,
               BitwiseT desired_mask)
{
    auto lid = it.get_local_id(0);
    if (lid == 0) {
        slm[0] = T(0);
    }

    sycl::group_barrier(it.get_group());

    std::size_t lws = static_cast<std::size_t>(it.get_local_range(0));
    std::size_t n_iters =
        topk_detail::quotient_ceil<std::size_t>(axis_nelems, lws) * lws;
    bool found = false;
    for (std::size_t i = lid; i < n_iters; i += lws) {
        bool in_range = (i < axis_nelems);
        T v = in_range ? data[i] : T(0);

        if (in_range && ((radix_utils::RadixTypeConfig<true, T>::encode(v) &
                          desired_mask) == desired)) {
            found = true;
            slm[0] = v;
        }

        if (sycl::any_of_group(it.get_group(), found)) {
            return slm[0];
        }
    }

    return T(0);
}

template <typename T,
          typename BitwiseT,
          typename CountT,
          std::uint32_t radix_bits,
          std::uint32_t radix_states,
          std::uint32_t radix_mask>
void radix_select(sycl::nd_item<1> &it,
                  const T *data,
                  dpnp::tensor::ssize_t k,
                  bool largest,
                  std::size_t axis_nelems,
                  CountT *radix_count_slm,
                  T *top_k_val_slm,
                  T &top_k)
{
    // private array for count values
    std::array<CountT, radix_states> counts;

    BitwiseT desired(0);
    BitwiseT desired_mask(0);

    int k_to_find = k;

    if constexpr (std::is_same_v<BitwiseT, bool>) {
        constexpr int radix_offset(0);

        count_radix<T, BitwiseT, CountT, radix_states, radix_mask>(
            it, data, counts.data(), radix_count_slm, desired, desired_mask,
            radix_offset, axis_nelems);

        auto found_unique = [&](int i, CountT count) -> bool {
            if (count == 1 && k_to_find == 1) {
                desired = radix_utils::set_bucket_id<radix_mask, BitwiseT>(
                    desired, i, radix_offset);
                desired_mask = radix_utils::set_bucket_id<radix_mask, BitwiseT>(
                    desired_mask, radix_mask, radix_offset);

                top_k = find_pattern<T, BitwiseT>(it, data, axis_nelems,
                                                  top_k_val_slm, desired,
                                                  desired_mask);
                return true;
            }
            return false;
        };

        auto found_non_unique = [&](int i, CountT count) -> bool {
            if (count >= k_to_find) {
                desired = radix_utils::set_bucket_id<radix_mask, BitwiseT>(
                    desired, i, radix_offset);
                desired_mask = radix_utils::set_bucket_id<radix_mask, BitwiseT>(
                    desired_mask, radix_mask, radix_offset);

                return true;
            }
            k_to_find -= count;
            return false;
        };

        if (largest) {
#pragma unroll
            for (int i = radix_states - 1; i >= 0; --i) {
                CountT count = counts[i];
                if (found_unique(i, count)) {
                    return;
                }
                if (found_non_unique(i, count)) {
                    break;
                }
            }
        }
        else {
#pragma unroll
            for (int i = 0; i < radix_states; ++i) {
                int count = counts[i];
                if (found_unique(i, count)) {
                    return;
                }
                if (found_non_unique(i, count)) {
                    break;
                }
            }
        }
    }
    else {
        // signed int avoids overflow, radix_offset implicitly cast to uint32_t
        for (int radix_offset =
                 radix_utils::number_of_bits_in_type<T>() - radix_bits;
             radix_offset >= 0; radix_offset -= radix_bits) {
            count_radix<T, BitwiseT, CountT, radix_states, radix_mask>(
                it, data, counts.data(), radix_count_slm, desired, desired_mask,
                radix_offset, axis_nelems);

            auto found_unique = [&](int i, CountT count) -> bool {
                if (count == 1 && k_to_find == 1) {
                    desired = radix_utils::set_bucket_id<radix_mask, BitwiseT>(
                        desired, i, radix_offset);
                    desired_mask =
                        radix_utils::set_bucket_id<radix_mask, BitwiseT>(
                            desired_mask, radix_mask, radix_offset);

                    top_k = find_pattern<T, BitwiseT>(it, data, axis_nelems,
                                                      top_k_val_slm, desired,
                                                      desired_mask);
                    return true;
                }
                return false;
            };

            auto found_non_unique = [&](int i, CountT count) -> bool {
                if (count >= k_to_find) {
                    desired = radix_utils::set_bucket_id<radix_mask, BitwiseT>(
                        desired, i, radix_offset);
                    desired_mask =
                        radix_utils::set_bucket_id<radix_mask, BitwiseT>(
                            desired_mask, radix_mask, radix_offset);

                    return true;
                }
                k_to_find -= count;
                return false;
            };

            if (largest) {
#pragma unroll
                for (int i = radix_states - 1; i >= 0; --i) {
                    CountT count = counts[i];
                    if (found_unique(i, count)) {
                        return;
                    }
                    if (found_non_unique(i, count)) {
                        break;
                    }
                }
            }
            else {
#pragma unroll
                for (int i = 0; i < radix_states; ++i) {
                    int count = counts[i];
                    if (found_unique(i, count)) {
                        return;
                    }
                    if (found_non_unique(i, count)) {
                        break;
                    }
                }
            }
        } // end radix_offset loop
    }
    top_k = radix_utils::RadixTypeConfig<true, T>::decode(desired);
}

} // namespace topk_detail

namespace sg_topk
{

template <typename T, bool HaveKthVals = false>
class topk_krn;

template <typename T, typename IndexT, bool HaveKthVals = false>
struct TopK
{
    static constexpr std::uint32_t radix_bits = 2;
    static constexpr std::uint32_t radix_states = 1 << radix_bits;
    static constexpr std::uint32_t radix_mask = (radix_states - 1);

    using CountT = std::uint32_t;

    template <typename LocAccT1, typename LocAccT2>
    class TopKFunctor
    {
    private:
        const T *inp = nullptr;
        std::size_t axis_nelems;
        dpnp::tensor::ssize_t k;
        bool largest;
        std::size_t iter_nelems;
        T *top_k = nullptr;
        IndexT *indices = nullptr;
        LocAccT1 radix_count_slm;
        LocAccT2 top_k_val_slm;
        T *kth_vals = nullptr;

    public:
        TopKFunctor(const T *inp_,
                    std::size_t axis_nelems_,
                    dpnp::tensor::ssize_t k_,
                    bool largest_,
                    std::size_t iter_nelems_,
                    T *top_k_,
                    IndexT *indices_,
                    LocAccT1 radix_count_slm_,
                    LocAccT2 top_k_val_slm_,
                    T *kth_vals_)
            : inp(inp_), axis_nelems(axis_nelems_), k(k_), largest(largest_),
              iter_nelems(iter_nelems_), top_k(top_k_), indices(indices_),
              radix_count_slm(radix_count_slm_), top_k_val_slm(top_k_val_slm_),
              kth_vals(kth_vals_)
        {
        }

        void operator()(sycl::nd_item<1> it) const
        {
            std::size_t iter_idx = it.get_group_linear_id();

            const T *inp_start = inp + axis_nelems * iter_idx;
            T *top_k_start = (top_k) ? top_k + k * iter_idx : top_k;
            IndexT *indices_start =
                (indices) ? indices + k * iter_idx : indices;

            T top_k_val;
            if constexpr (HaveKthVals) {
                top_k_val = kth_vals[iter_idx];
            }
            else {
                top_k_val = T(0);
                topk_detail::radix_select<
                    T,
                    typename radix_utils::RadixTypeConfig<true, T>::RadixType,
                    CountT, radix_bits, radix_states, radix_mask>(
                    it, inp_start, k, largest, axis_nelems,
                    topk_detail::get_accessor_pointer(radix_count_slm),
                    topk_detail::get_accessor_pointer(top_k_val_slm),
                    top_k_val);
            }
            auto top_k_converted =
                radix_utils::RadixTypeConfig<true, T>::encode(top_k_val);

            std::size_t lws = it.get_local_range(0);
            std::size_t n_iters =
                topk_detail::quotient_ceil<std::size_t>(axis_nelems, lws) * lws;
            std::size_t write_idx_start(0);

            for (std::size_t i = it.get_local_id(0); i < n_iters; i += lws) {
                bool in_range = (i < axis_nelems);
                T v = in_range ? inp_start[i] : T(0);

                auto v_converted =
                    radix_utils::RadixTypeConfig<true, T>::encode(v);
                bool has_top_k;
                if (largest) {
                    has_top_k = in_range && (v_converted > top_k_converted);
                }
                else {
                    has_top_k = in_range && (v_converted < top_k_converted);
                }
                auto wg = it.get_group();
                int index = sycl::inclusive_scan_over_group(
                    wg, static_cast<int>(has_top_k), sycl::plus<int>{});
                int carry = sycl::group_broadcast(
                    wg, index, wg.get_local_linear_range() - 1);
                index -= static_cast<int>(
                    has_top_k); // turn inclusive scan exclusive

                if (has_top_k) {
                    int write_idx = write_idx_start + index;

                    top_k_start[write_idx] = v;
                    indices_start[write_idx] = i;
                }
                write_idx_start += carry;
            }

            std::size_t top_k_remaining = k - write_idx_start;

            for (std::size_t i = it.get_local_id(0); i < n_iters; i += lws) {
                bool in_range = (i < axis_nelems);
                T v = in_range ? inp_start[i] : T(0);
                auto v_converted =
                    radix_utils::RadixTypeConfig<true, T>::encode(v);
                bool has_top_k = in_range && (v_converted == top_k_converted);

                auto wg = it.get_group();
                int index = sycl::inclusive_scan_over_group(
                    wg, static_cast<int>(has_top_k), sycl::plus<int>{});
                int carry = sycl::group_broadcast(
                    wg, index, wg.get_local_linear_range() - 1);
                index -= static_cast<int>(
                    has_top_k); // turn inclusive scan exclusive

                if (has_top_k && index < top_k_remaining) {
                    int write_idx = write_idx_start + index;

                    top_k_start[write_idx] = v;
                    indices_start[write_idx] = i;
                }

                if (carry >= top_k_remaining) {
                    break;
                }

                top_k_remaining -= carry;
                write_idx_start += carry;
            }
        }
    };

    static sycl::event
        submit_top_k(sycl::queue &exec_q,
                     const T *arg,
                     std::size_t axis_nelems,
                     std::size_t k,
                     bool largest,
                     std::size_t iter_nelems,
                     T *top_k,
                     IndexT *indices,
                     const std::vector<sycl::event> &depends = {})
    {
        using KernelName = topk_krn<T>;
        const auto &kernel_id = sycl::get_kernel_id<KernelName>();

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

        // require at least radix_states shared memory, atomically modified
        // by each sub-group
        static constexpr std::size_t radix_counts_slm_sz = radix_states;
        static constexpr std::size_t top_k_val_slm_sz = 1;
        if (device_local_memory_size - safety_margin <
            sizeof(CountT) * radix_counts_slm_sz +
                sizeof(T) * top_k_val_slm_sz) {
            throw std::runtime_error("Insufficient resources");
        }

        // Adaptive sub-groups-per-work-group: cover axis_nelems with as many
        // sub-groups as needed, capped at 8.  Avoids over-padding tiny rows
        // (e.g. N=9 gets lws=16 instead of 64) while still saturating the GPU
        // for large N.
        const std::size_t sgs_needed =
            topk_detail::quotient_ceil<std::size_t>(axis_nelems, max_sg_size);
        const std::size_t preferred_sg_per_wg =
            std::min<std::size_t>(sgs_needed, std::size_t(8));
        const std::size_t preferred_wg_sz = max_sg_size * preferred_sg_per_wg;

        sycl::event topk_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            std::size_t lws = std::min(
                topk_detail::quotient_ceil<std::size_t>(axis_nelems,
                                                        preferred_wg_sz) *
                    preferred_wg_sz,
                dev.get_info<sycl::info::device::max_work_group_size>());

            auto gRange = sycl::range<1>(iter_nelems * lws);
            auto lRange = sycl::range<1>(lws);
            auto ndRange = sycl::nd_range<1>{gRange, lRange};

            using LocAccT1 = sycl::local_accessor<CountT, 1>;
            LocAccT1 radix_count_slm(radix_counts_slm_sz, cgh);

            using LocAccT2 = sycl::local_accessor<T, 1>;
            LocAccT2 top_k_val_slm(top_k_val_slm_sz, cgh);

            cgh.parallel_for<KernelName>(
                ndRange,
                TopKFunctor(arg, axis_nelems, k, largest, iter_nelems, top_k,
                            indices, radix_count_slm, top_k_val_slm, nullptr));
        });

        return topk_ev;
    }

    template <bool b = HaveKthVals, std::enable_if_t<b, bool> = true>
    static sycl::event
        submit_top_k(sycl::queue &exec_q,
                     const T *arg,
                     std::size_t axis_nelems,
                     std::size_t k,
                     bool largest,
                     std::size_t iter_nelems,
                     T *top_k,
                     IndexT *indices,
                     T *kth_vals,
                     const std::vector<sycl::event> &depends = {})
    {
        using KernelName = topk_krn<T, HaveKthVals>;
        const auto &kernel_id = sycl::get_kernel_id<KernelName>();

        auto const &ctx = exec_q.get_context();
        auto const &dev = exec_q.get_device();
        auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            ctx, {dev}, {kernel_id});

        auto krn = kb.get_kernel(kernel_id);

        const std::uint32_t max_sg_size = krn.template get_info<
            sycl::info::kernel_device_specific::max_sub_group_size>(dev);

        // Adaptive sub-groups-per-work-group (see kth-less submit_top_k).
        const std::size_t sgs_needed =
            topk_detail::quotient_ceil<std::size_t>(axis_nelems, max_sg_size);
        const std::size_t preferred_sg_per_wg =
            std::min<std::size_t>(sgs_needed, std::size_t(8));
        const std::size_t preferred_wg_sz = max_sg_size * preferred_sg_per_wg;

        sycl::event topk_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            std::size_t lws = std::min(
                topk_detail::quotient_ceil<std::size_t>(axis_nelems,
                                                        preferred_wg_sz) *
                    preferred_wg_sz,
                dev.get_info<sycl::info::device::max_work_group_size>());

            auto gRange = sycl::range<1>(iter_nelems * lws);
            auto lRange = sycl::range<1>(lws);
            auto ndRange = sycl::nd_range<1>{gRange, lRange};

            cgh.parallel_for<KernelName>(
                ndRange,
                TopKFunctor(arg, axis_nelems, k, largest, iter_nelems, top_k,
                            indices, nullptr, nullptr, kth_vals));
        });

        return topk_ev;
    }
};

} // end of namespace sg_topk

namespace mg_topk
{

// TODO: declare these either in a separate struct or in the bodies of impl
// funcs

constexpr int wi_per_group = 256;

constexpr std::uint32_t radix_bits = 8;
constexpr std::uint32_t radix_states = 1 << radix_bits; // 2 ^ radix_bits
constexpr std::uint32_t radix_mask = (radix_states - 1);
static_assert(
    radix_states <= wi_per_group,
    "radix_find_kth_values kernel requires radix_states <= wi_per_group");
constexpr int MIN_ITEMS_PER_THREAD = 4;
constexpr int MAX_ITEMS_PER_THREAD = 64;

static_assert(radix_states <= wi_per_group);

template <typename T, typename BitwiseT, typename LocAccT>
class RadixFindKthValuesFunctor
{
    static_assert(MAX_ITEMS_PER_THREAD * wi_per_group <
                  std::numeric_limits<std::int16_t>::max());
    static_assert(radix_states <= wi_per_group);

private:
    // inputs
    const T *inp = nullptr;
    std::size_t axis_nelems;
    std::uint32_t *ks_to_find = nullptr; // iter_nelems array
    std::size_t iter_nelems;
    int current_bit;
    int base_items_per_thread;
    std::uint32_t axis_groups;
    BitwiseT desired_mask;
    // outputs
    std::uint32_t *axis_groups_done = nullptr; // iter_nelems array
    BitwiseT *desires = nullptr;               // iter_nelems array
    std::int16_t *counts = nullptr; // iter_nelems * axis_groups * radix_digits
    T *kth_values =
        nullptr; // size: iter_nelems, written to when current_bit is 0
    // local memory
    LocAccT slm_counters;
    LocAccT slm_cumsum;

public:
    RadixFindKthValuesFunctor(const T *inp_,
                              std::size_t axis_nelems_,
                              std::uint32_t *ks_to_find_,
                              std::size_t iter_nelems_,
                              int current_bit_,
                              int base_items_per_thread_,
                              std::uint32_t axis_groups_,
                              BitwiseT desired_mask_,
                              std::uint32_t *axis_groups_done_,
                              BitwiseT *desires_,
                              std::int16_t *counts_,
                              T *kth_values_,
                              LocAccT slm_counters_,
                              LocAccT slm_cumsum_)
        : inp(inp_), axis_nelems(axis_nelems_), ks_to_find(ks_to_find_),
          iter_nelems(iter_nelems_), current_bit(current_bit_),
          base_items_per_thread(base_items_per_thread_),
          axis_groups(axis_groups_), desired_mask(desired_mask_),
          axis_groups_done(axis_groups_done_), desires(desires_),
          counts(counts_), kth_values(kth_values_), slm_counters(slm_counters_),
          slm_cumsum(slm_cumsum_)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        std::size_t items_per_wg = base_items_per_thread * wi_per_group;

        std::size_t lid = it.get_local_id(0);
        std::uint32_t wg_idx = it.get_group_linear_id();
        std::uint32_t iter_idx = wg_idx / axis_groups;
        std::uint32_t wg_idx_in_axis = wg_idx % axis_groups;

        BitwiseT desired = desires[iter_idx];
        std::uint32_t k_to_find = ks_to_find[iter_idx];
        const T *inp_start = inp + axis_nelems * iter_idx;

        if (lid < radix_states) {
            slm_counters[lid] = 0;
        }

        sycl::group_barrier(it.get_group());

        int elems_per_wi =
            (wg_idx_in_axis + 1 < axis_groups)
                ? base_items_per_thread
                : topk_detail::quotient_ceil<std::size_t>(
                      axis_nelems - wg_idx_in_axis * items_per_wg,
                      wi_per_group);

        for (std::size_t i = 0; i < elems_per_wi; ++i) {
            // find start offset for this index along iteration axis
            std::size_t idx =
                wg_idx_in_axis * items_per_wg + i * wi_per_group + lid;
            if (idx < axis_nelems) {
                BitwiseT val = radix_utils::RadixTypeConfig<true, T>::encode(
                    inp_start[idx]);
                bool has_val =
                    ((val & desired_mask) == (desired & desired_mask));

                BitwiseT digit =
                    radix_utils::get_bucket_id<radix_mask>(val, current_bit);
                if (has_val) {
                    sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::local_space>
                        loc_counter_ref(slm_counters[digit]);
                    loc_counter_ref += std::uint32_t(1);
                }
            }
        }

        sycl::group_barrier(it.get_group());

        std::uint32_t digit_count(0);
        if (lid < radix_states) {
            digit_count = slm_counters[lid];
            counts[wg_idx * radix_states + lid] = digit_count;
        }

        if (axis_groups > 1) {
            sycl::atomic_fence(sycl::memory_order::acq_rel,
                               sycl::memory_scope::device);
            sycl::group_barrier(it.get_group());
        }

        bool last_group_done = false;
        if (lid == 0) {
            if (axis_groups == 1) {
                last_group_done = true;
            }
            else {
                // TODO: factor this atomic fetch add out, maybe put
                // axis_groups_done into a class or something
                sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    loc_sem_ref(axis_groups_done[iter_idx]);
                std::uint32_t axis_groups_done_old = loc_sem_ref.fetch_add(
                    std::uint32_t(1), sycl::memory_order::acq_rel);
                last_group_done = (axis_groups_done_old == axis_groups - 1);
            }
        }

        // broadcasts from smallest (i.e., 0th) item by default
        last_group_done =
            sycl::group_broadcast(it.get_group(), last_group_done);

        if (!last_group_done) {
            return;
        }

        if (lid < radix_states && axis_groups > 1) {
            digit_count = 0;
            for (int gr = 0; gr < axis_groups; ++gr) {
                digit_count +=
                    counts[(iter_idx * axis_groups + gr) * radix_states + lid];
            }
        }

        std::uint32_t digit_count_cumsum = sycl::inclusive_scan_over_group(
            it.get_group(), digit_count, sycl::plus<std::uint32_t>{});
        // need value to the left, so move into SLM
        if (lid < radix_states) {
            slm_cumsum[lid] = digit_count_cumsum;
        }

        sycl::group_barrier(it.get_group());

        if (lid < radix_states) {
            std::uint32_t digit_count_cumsum_left = (lid == std::uint32_t(0))
                                                        ? std::uint32_t(0)
                                                        : slm_cumsum[lid - 1];

            if (digit_count_cumsum_left < k_to_find &&
                k_to_find <= digit_count_cumsum) {
                desired = radix_utils::set_bucket_id<radix_mask, BitwiseT>(
                    desired, lid, current_bit);
                desires[iter_idx] = desired;
                if (current_bit > 0) {
                    // if not last pass, update ks_to_find
                    ks_to_find[iter_idx] = k_to_find - digit_count_cumsum_left;
                }
                else {
                    // if last pass, update kth value
                    kth_values[iter_idx] =
                        radix_utils::RadixTypeConfig<true, T>::decode(desired);
                }
            }
        }

        // reset axis_groups_done for possible next kernel launch in loop
        if (lid == 0) {
            axis_groups_done[iter_idx] = 0;
        }
    }
};

int get_items_per_thread(std::uint64_t iter_nelems,
                         std::uint64_t axis_nelems,
                         sycl::device dev)
{
    // occupancy of this kernel is limited by registers per threads
    constexpr int REGS_PER_THREAD = 128; // from GPU opt guide
    constexpr int REGS_PER_BLOCK = REGS_PER_THREAD * wi_per_group;
    int mpc = dev.get_info<sycl::info::device::max_compute_units>();

    int regs_per_mp = 65536;
    int max_groups_per_mp = 32;

    // ensure every EU has work
    int groups_per_mp =
        std::min(regs_per_mp / REGS_PER_BLOCK, max_groups_per_mp);
    std::int64_t elems_per_wi = topk_detail::quotient_ceil<std::int64_t>(
        static_cast<std::int64_t>(axis_nelems * iter_nelems),
        static_cast<std::int64_t>(mpc * groups_per_mp * wi_per_group));

    // enforce a ceiling on axis_groups to bound the histogram reduction
    constexpr int TARGET_MAX_GROUPS = 32;
    std::int64_t elems_from_group_limit =
        topk_detail::quotient_ceil<std::int64_t>(
            static_cast<std::int64_t>(axis_nelems),
            static_cast<std::int64_t>(TARGET_MAX_GROUPS * wi_per_group));
    elems_per_wi = std::max(elems_per_wi, elems_from_group_limit);

    elems_per_wi = std::clamp(static_cast<int>(elems_per_wi),
                              MIN_ITEMS_PER_THREAD, MAX_ITEMS_PER_THREAD);
    return elems_per_wi;
}

template <typename T>
class radix_select_find_kth_vals_krn;

template <typename T, typename BitwiseT>
sycl::event submit_radix_find_kth_vals(
    sycl::queue &exec_q,
    sycl::nd_range<1> range,
    std::size_t iter_nelems,
    std::size_t axis_nelems,
    std::size_t elems_per_wi,
    std::size_t axis_groups,
    // input array memory
    const T *arg_tp,
    // output
    T *kth_values,
    // temporary memory
    std::uint32_t *ks_to_find,
    std::uint32_t *axis_groups_done,
    BitwiseT *desired,
    std::int16_t *counts,
    const std::vector<sycl::event> &depends,
    const std::vector<sycl::event> &additional_depends)
{
    using KernelName = radix_select_find_kth_vals_krn<T>;

    sycl::event last_launch_ev;
    if constexpr (std::is_same_v<BitwiseT, bool>) {
        constexpr int current_bit = 0;

        sycl::event launch_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);
            cgh.depends_on(additional_depends);

            using LocAccT = sycl::local_accessor<std::uint32_t, 1>;
            LocAccT slm_counters(radix_states, cgh);
            LocAccT slm_cumsum(radix_states, cgh);

            cgh.parallel_for<KernelName>(
                range,
                RadixFindKthValuesFunctor<T, BitwiseT, LocAccT>(
                    arg_tp, axis_nelems, ks_to_find, iter_nelems, current_bit,
                    elems_per_wi, axis_groups, BitwiseT(0), axis_groups_done,
                    desired, counts, kth_values, slm_counters, slm_cumsum));
        });
        last_launch_ev = launch_ev;
    }
    else {
        bool used_deps = false;
        BitwiseT desired_mask = 0;
        for (int current_bit =
                 radix_utils::number_of_bits_in_type<T>() - radix_bits;
             current_bit >= 0; current_bit -= radix_bits) {
            sycl::event launch_ev = exec_q.submit([&](sycl::handler &cgh) {
                if (!used_deps) {
                    cgh.depends_on(depends);
                    cgh.depends_on(additional_depends);
                    used_deps = true;
                }
                else {
                    cgh.depends_on(last_launch_ev);
                }

                using LocAccT = sycl::local_accessor<std::uint32_t, 1>;
                LocAccT slm_counters(radix_states, cgh);
                LocAccT slm_cumsum(radix_states, cgh);

                cgh.parallel_for<KernelName>(
                    range, RadixFindKthValuesFunctor<T, BitwiseT, LocAccT>(
                               arg_tp, axis_nelems, ks_to_find, iter_nelems,
                               current_bit, elems_per_wi, axis_groups,
                               desired_mask, axis_groups_done, desired, counts,
                               kth_values, slm_counters, slm_cumsum));
            });
            last_launch_ev = launch_ev;
            desired_mask = radix_utils::set_bucket_id<radix_mask, BitwiseT>(
                desired_mask, radix_mask, current_bit);
        }
    }
    return last_launch_ev;
}

template <typename argTy, typename IndexTy>
sycl::event submit_top_k_radix_select_multi_group(
    sycl::queue &exec_q,
    std::size_t iter_nelems, // number of sub-arrays to search
    std::size_t axis_nelems, // size of axis to find top k over
    std::size_t k,
    bool largest,
    const argTy *arg_tp,
    argTy *vals_tp,
    IndexTy *inds_tp,
    const std::vector<sycl::event> &depends)
{
    static_assert(std::is_same_v<IndexTy, std::int64_t>);

    auto const &dev = exec_q.get_device();
    if (dev.get_info<sycl::info::device::max_work_group_size>() <
        wi_per_group) {
        throw std::runtime_error("maximum work group size insufficient for "
                                 "submit_top_k_radix_select_multi_group");
    }

    // 512 bytes of local memory reserved for RT
    const std::uint64_t safety_margin = 512;

    const std::uint64_t device_local_memory_size =
        dev.get_info<sycl::info::device::local_mem_size>();

    auto const &ctx = exec_q.get_context();

    int elems_per_wi = get_items_per_thread(iter_nelems, axis_nelems, dev);
    int elems_per_group = elems_per_wi * wi_per_group;

    using BitwiseT =
        typename radix_utils::RadixTypeConfig<true, argTy>::RadixType;
    std::uint32_t axis_groups =
        topk_detail::quotient_ceil<std::uint32_t>(axis_nelems, elems_per_group);
    std::uint32_t n_groups = iter_nelems * axis_groups;

    // find kth vals kernel required SLM
    const std::size_t find_kth_vals_slm_size =
        sizeof(std::uint32_t) * radix_states * 2;

    if (device_local_memory_size - safety_margin < find_kth_vals_slm_size) {
        throw std::runtime_error("Insufficient resources");
    }

    // allocate temp for kth values
    auto kth_values_owner =
        dpnp::tensor::alloc_utils::smart_malloc_device<argTy>(iter_nelems,
                                                              exec_q);
    argTy *kth_values = kth_values_owner.get();

    // allocate temp for axis_groups_done
    auto axis_groups_done_owner =
        dpnp::tensor::alloc_utils::smart_malloc_device<std::uint32_t>(
            iter_nelems, exec_q);
    std::uint32_t *axis_groups_done = axis_groups_done_owner.get();

    // allocate temp for ks_to_find
    auto ks_to_find_owner =
        dpnp::tensor::alloc_utils::smart_malloc_device<std::uint32_t>(
            iter_nelems, exec_q);
    std::uint32_t *ks_to_find = ks_to_find_owner.get();

    // allocate temp for desired
    auto desired_owner =
        dpnp::tensor::alloc_utils::smart_malloc_device<BitwiseT>(iter_nelems,
                                                                 exec_q);
    BitwiseT *desired = desired_owner.get();

    // allocate temp for counts
    auto counts_owner =
        dpnp::tensor::alloc_utils::smart_malloc_device<std::int16_t>(
            n_groups * radix_states, exec_q);
    std::int16_t *counts = counts_owner.get();

    sycl::event populate_axis_groups_done_ev = exec_q.fill<std::uint32_t>(
        axis_groups_done, std::uint32_t(0), iter_nelems, depends);

    std::uint32_t k_to_find =
        largest ? static_cast<std::uint32_t>(axis_nelems - k + 1)
                : static_cast<std::uint32_t>(k);
    sycl::event fill_ks_to_find_ev =
        exec_q.fill<std::uint32_t>(ks_to_find, k_to_find, iter_nelems, depends);

    sycl::event desired_init_ev =
        exec_q.fill<BitwiseT>(desired, BitwiseT(0), iter_nelems, depends);

    const std::vector<sycl::event> other_depends = {
        populate_axis_groups_done_ev, fill_ks_to_find_ev, desired_init_ev};

    auto gRange = sycl::range<1>(n_groups * wi_per_group);
    auto lRange = sycl::range<1>(wi_per_group);
    auto ndRange = sycl::nd_range<1>{gRange, lRange};

    sycl::event find_kth_vals_ev = submit_radix_find_kth_vals(
        exec_q, ndRange, iter_nelems, axis_nelems, elems_per_wi, axis_groups,
        arg_tp, kth_values, ks_to_find, axis_groups_done, desired, counts,
        depends, other_depends);

    sycl::event topk_with_kth_vals_ev =
        sg_topk::TopK<argTy, IndexTy, true>::submit_top_k(
            exec_q, arg_tp, axis_nelems, k, largest, iter_nelems, vals_tp,
            inds_tp, kth_values, {find_kth_vals_ev});

    sycl::event temp_free_event = dpnp::tensor::alloc_utils::async_smart_free(
        exec_q, {topk_with_kth_vals_ev}, kth_values_owner,
        axis_groups_done_owner, ks_to_find_owner, desired_owner, counts_owner);

    return temp_free_event;
}

} // end of namespace mg_topk

template <typename T1, typename T2>
class topk_radix_select_single_group_krn;

template <typename argTy, typename IndexTy>
sycl::event topk_radix_select_single_group_impl(
    sycl::queue &exec_q,
    std::size_t iter_nelems, // number of sub-arrays to search
    std::size_t axis_nelems, // size of axis to find top k over
    std::size_t k,
    bool largest,
    const char *arg_cp,
    char *vals_cp,
    char *inds_cp,
    dpnp::tensor::ssize_t iter_arg_offset,
    dpnp::tensor::ssize_t iter_vals_offset,
    dpnp::tensor::ssize_t iter_inds_offset,
    dpnp::tensor::ssize_t axis_arg_offset,
    dpnp::tensor::ssize_t axis_vals_offset,
    dpnp::tensor::ssize_t axis_inds_offset,
    const std::vector<sycl::event> &depends)
{
    static_assert(std::is_same_v<IndexTy, std::int64_t>);

    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp) +
                          iter_arg_offset + axis_arg_offset;
    argTy *vals_tp = reinterpret_cast<argTy *>(vals_cp) + iter_vals_offset +
                     axis_vals_offset;
    IndexTy *inds_tp = reinterpret_cast<IndexTy *>(inds_cp) + iter_inds_offset +
                       axis_inds_offset;

    return sg_topk::TopK<argTy, IndexTy, false>::submit_top_k(
        exec_q, arg_tp, axis_nelems, k, largest, iter_nelems, vals_tp, inds_tp,
        depends);
}

template <typename argTy, typename IndexTy>
sycl::event topk_radix_select_multi_group_impl(
    sycl::queue &exec_q,
    std::size_t iter_nelems, // number of sub-arrays to search
    std::size_t axis_nelems, // size of axis to find top k over
    std::size_t k,
    bool largest,
    const char *arg_cp,
    char *vals_cp,
    char *inds_cp,
    dpnp::tensor::ssize_t iter_arg_offset,
    dpnp::tensor::ssize_t iter_vals_offset,
    dpnp::tensor::ssize_t iter_inds_offset,
    dpnp::tensor::ssize_t axis_arg_offset,
    dpnp::tensor::ssize_t axis_vals_offset,
    dpnp::tensor::ssize_t axis_inds_offset,
    const std::vector<sycl::event> &depends)
{
    static_assert(std::is_same_v<IndexTy, std::int64_t>);

    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp) +
                          iter_arg_offset + axis_arg_offset;
    argTy *vals_tp = reinterpret_cast<argTy *>(vals_cp) + iter_vals_offset +
                     axis_vals_offset;
    IndexTy *inds_tp = reinterpret_cast<IndexTy *>(inds_cp) + iter_inds_offset +
                       axis_inds_offset;

    return mg_topk::submit_top_k_radix_select_multi_group<argTy, IndexTy>(
        exec_q, iter_nelems, axis_nelems, k, largest, arg_tp, vals_tp, inds_tp,
        depends);
}

} // end of namespace dpnp::tensor::kernels
