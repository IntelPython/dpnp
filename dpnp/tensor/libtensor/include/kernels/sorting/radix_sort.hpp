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
/// This file defines functions of dpnp.tensor._tensor_sorting_impl
/// extension.
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "kernels/dpctl_tensor_types.hpp"
#include "kernels/sorting/sort_utils.hpp"
#include "utils/sycl_alloc_utils.hpp"

namespace dpctl::tensor::kernels
{

namespace radix_sort_details
{

template <std::uint32_t, bool, typename... TrailingNames>
class radix_sort_count_kernel;

template <std::uint32_t, typename... TrailingNames>
class radix_sort_scan_kernel;

template <std::uint32_t, bool, typename... TrailingNames>
class radix_sort_reorder_peer_kernel;

template <std::uint32_t, bool, typename... TrailingNames>
class radix_sort_reorder_kernel;

/*! @brief Computes smallest exponent such that `n <= (1 << exponent)` */
template <typename SizeT,
          std::enable_if_t<std::is_unsigned_v<SizeT> &&
                               sizeof(SizeT) == sizeof(std::uint64_t),
                           int> = 0>
std::uint32_t ceil_log2(SizeT n)
{
    // if n > 2^b, n = q * 2^b + r for q > 0 and 0 <= r < 2^b
    // floor_log2(q * 2^b + r) == floor_log2(q * 2^b) == q + floor_log2(n1)
    // ceil_log2(n) == 1 + floor_log2(n-1)
    if (n <= 1)
        return std::uint32_t{1};

    std::uint32_t exp{1};
    --n;
    if (n >= (SizeT{1} << 32)) {
        n >>= 32;
        exp += 32;
    }
    if (n >= (SizeT{1} << 16)) {
        n >>= 16;
        exp += 16;
    }
    if (n >= (SizeT{1} << 8)) {
        n >>= 8;
        exp += 8;
    }
    if (n >= (SizeT{1} << 4)) {
        n >>= 4;
        exp += 4;
    }
    if (n >= (SizeT{1} << 2)) {
        n >>= 2;
        exp += 2;
    }
    if (n >= (SizeT{1} << 1)) {
        n >>= 1;
        ++exp;
    }
    return exp;
}

//----------------------------------------------------------
// bitwise order-preserving conversions to unsigned integers
//----------------------------------------------------------

template <bool is_ascending>
bool order_preserving_cast(bool val)
{
    if constexpr (is_ascending)
        return val;
    else
        return !val;
}

template <bool is_ascending,
          typename UIntT,
          std::enable_if_t<std::is_unsigned_v<UIntT>, int> = 0>
UIntT order_preserving_cast(UIntT val)
{
    if constexpr (is_ascending) {
        return val;
    }
    else {
        // bitwise invert
        return (~val);
    }
}

template <bool is_ascending,
          typename IntT,
          std::enable_if_t<std::is_integral_v<IntT> && std::is_signed_v<IntT>,
                           int> = 0>
std::make_unsigned_t<IntT> order_preserving_cast(IntT val)
{
    using UIntT = std::make_unsigned_t<IntT>;
    const UIntT uint_val = sycl::bit_cast<UIntT>(val);

    if constexpr (is_ascending) {
        // ascending_mask: 100..0
        static constexpr UIntT ascending_mask =
            (UIntT(1) << std::numeric_limits<IntT>::digits);
        return (uint_val ^ ascending_mask);
    }
    else {
        // descending_mask: 011..1
        static constexpr UIntT descending_mask =
            (std::numeric_limits<UIntT>::max() >> 1);
        return (uint_val ^ descending_mask);
    }
}

template <bool is_ascending>
std::uint16_t order_preserving_cast(sycl::half val)
{
    using UIntT = std::uint16_t;

    const UIntT uint_val = sycl::bit_cast<UIntT>(
        (sycl::isnan(val)) ? std::numeric_limits<sycl::half>::quiet_NaN()
                           : val);
    UIntT mask;

    // test the sign bit of the original value
    const bool zero_fp_sign_bit = (UIntT(0) == (uint_val >> 15));

    static constexpr UIntT zero_mask = UIntT(0x8000u);
    static constexpr UIntT nonzero_mask = UIntT(0xFFFFu);

    static constexpr UIntT inv_zero_mask = static_cast<UIntT>(~zero_mask);
    static constexpr UIntT inv_nonzero_mask = static_cast<UIntT>(~nonzero_mask);

    if constexpr (is_ascending) {
        mask = (zero_fp_sign_bit) ? zero_mask : nonzero_mask;
    }
    else {
        mask = (zero_fp_sign_bit) ? (inv_zero_mask) : (inv_nonzero_mask);
    }

    return (uint_val ^ mask);
}

template <bool is_ascending,
          typename FloatT,
          std::enable_if_t<std::is_floating_point_v<FloatT> &&
                               sizeof(FloatT) == sizeof(std::uint32_t),
                           int> = 0>
std::uint32_t order_preserving_cast(FloatT val)
{
    using UIntT = std::uint32_t;

    UIntT uint_val = sycl::bit_cast<UIntT>(
        (sycl::isnan(val)) ? std::numeric_limits<FloatT>::quiet_NaN() : val);

    UIntT mask;

    // test the sign bit of the original value
    const bool zero_fp_sign_bit = (UIntT(0) == (uint_val >> 31));

    static constexpr UIntT zero_mask = UIntT(0x80000000u);
    static constexpr UIntT nonzero_mask = UIntT(0xFFFFFFFFu);

    if constexpr (is_ascending)
        mask = (zero_fp_sign_bit) ? zero_mask : nonzero_mask;
    else
        mask = (zero_fp_sign_bit) ? (~zero_mask) : (~nonzero_mask);

    return (uint_val ^ mask);
}

template <bool is_ascending,
          typename FloatT,
          std::enable_if_t<std::is_floating_point_v<FloatT> &&
                               sizeof(FloatT) == sizeof(std::uint64_t),
                           int> = 0>
std::uint64_t order_preserving_cast(FloatT val)
{
    using UIntT = std::uint64_t;

    UIntT uint_val = sycl::bit_cast<UIntT>(
        (sycl::isnan(val)) ? std::numeric_limits<FloatT>::quiet_NaN() : val);
    UIntT mask;

    // test the sign bit of the original value
    const bool zero_fp_sign_bit = (UIntT(0) == (uint_val >> 63));

    static constexpr UIntT zero_mask = UIntT(0x8000000000000000u);
    static constexpr UIntT nonzero_mask = UIntT(0xFFFFFFFFFFFFFFFFu);

    if constexpr (is_ascending)
        mask = (zero_fp_sign_bit) ? zero_mask : nonzero_mask;
    else
        mask = (zero_fp_sign_bit) ? (~zero_mask) : (~nonzero_mask);

    return (uint_val ^ mask);
}

//-----------------
// bucket functions
//-----------------

template <typename T>
constexpr std::size_t number_of_bits_in_type()
{
    constexpr std::size_t type_bits =
        (sizeof(T) * std::numeric_limits<unsigned char>::digits);
    return type_bits;
}

// the number of buckets (size of radix bits) in T
template <typename T>
constexpr std::uint32_t number_of_buckets_in_type(std::uint32_t radix_bits)
{
    constexpr std::size_t type_bits = number_of_bits_in_type<T>();
    return (type_bits + radix_bits - 1) / radix_bits;
}

// get bits value (bucket) in a certain radix position
template <std::uint32_t radix_mask, typename T>
std::uint32_t get_bucket_id(T val, std::uint32_t radix_offset)
{
    static_assert(std::is_unsigned_v<T>);

    return (val >> radix_offset) & T(radix_mask);
}

//--------------------------------
// count kernel (single iteration)
//--------------------------------

template <typename KernelName,
          std::uint32_t radix_bits,
          typename ValueT,
          typename CountT,
          typename Proj>
sycl::event
    radix_sort_count_submit(sycl::queue &exec_q,
                            std::size_t n_iters,
                            std::size_t n_segments,
                            std::size_t wg_size,
                            std::uint32_t radix_offset,
                            std::size_t n_values,
                            ValueT *vals_ptr,
                            std::size_t n_counts,
                            CountT *counts_ptr,
                            const Proj &proj_op,
                            const bool is_ascending,
                            const std::vector<sycl::event> &dependency_events)
{
    // bin_count = radix_states used for an array storing bucket state counters
    static constexpr std::uint32_t radix_states =
        (std::uint32_t(1) << radix_bits);
    static constexpr std::uint32_t radix_mask = radix_states - 1;

    // iteration space info
    const std::size_t n = n_values;
    // each segment is processed by a work-group
    const std::size_t elems_per_segment = (n + n_segments - 1) / n_segments;
    const std::size_t no_op_flag_id = n_counts - 1;

    assert(n_counts == (n_segments + 1) * radix_states + 1);

    sycl::event local_count_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependency_events);

        sycl::local_accessor<CountT, 1> counts_lacc(wg_size * radix_states,
                                                    cgh);

        sycl::nd_range<1> ndRange(n_iters * n_segments * wg_size, wg_size);

        cgh.parallel_for<KernelName>(ndRange, [=](sycl::nd_item<1> ndit) {
            // 0 <= lid < wg_size
            const std::size_t lid = ndit.get_local_id(0);
            // 0 <= group_id < n_segments * n_iters
            const std::size_t group_id = ndit.get_group(0);
            const std::size_t iter_id = group_id / n_segments;
            const std::size_t val_iter_offset = iter_id * n;
            // 0 <= wgr_id < n_segments
            const std::size_t wgr_id = group_id - iter_id * n_segments;

            const std::size_t seg_start = elems_per_segment * wgr_id;

            // count per work-item: create a private array for storing count
            // values here bin_count = radix_states
            std::array<CountT, radix_states> counts_arr = {CountT{0}};

            // count per work-item: count values and write result to private
            // count array
            const std::size_t seg_end =
                sycl::min(seg_start + elems_per_segment, n);
            if (is_ascending) {
                for (std::size_t val_id = seg_start + lid; val_id < seg_end;
                     val_id += wg_size)
                {
                    // get the bucket for the bit-ordered input value,
                    // applying the offset and mask for radix bits
                    const auto val =
                        order_preserving_cast</*is_ascending*/ true>(
                            proj_op(vals_ptr[val_iter_offset + val_id]));
                    const std::uint32_t bucket_id =
                        get_bucket_id<radix_mask>(val, radix_offset);

                    // increment counter for this bit bucket
                    ++counts_arr[bucket_id];
                }
            }
            else {
                for (std::size_t val_id = seg_start + lid; val_id < seg_end;
                     val_id += wg_size)
                {
                    // get the bucket for the bit-ordered input value,
                    // applying the offset and mask for radix bits
                    const auto val =
                        order_preserving_cast</*is_ascending*/ false>(
                            proj_op(vals_ptr[val_iter_offset + val_id]));
                    const std::uint32_t bucket_id =
                        get_bucket_id<radix_mask>(val, radix_offset);

                    // increment counter for this bit bucket
                    ++counts_arr[bucket_id];
                }
            }

            // count per work-item: write private count array to local count
            // array counts_lacc is concatenation of private count arrays from
            // each work-item in the order of their local ids
            const std::uint32_t count_start_id = radix_states * lid;
            for (std::uint32_t radix_state_id = 0;
                 radix_state_id < radix_states; ++radix_state_id)
            {
                counts_lacc[count_start_id + radix_state_id] =
                    counts_arr[radix_state_id];
            }

            sycl::group_barrier(ndit.get_group());

            // count per work-group: reduce till count_lacc[] size > wg_size
            // all work-items in the work-group do the work.
            for (std::uint32_t i = 1; i < radix_states; ++i) {
                // Since we interested in computing total count over work-group
                // for each radix state, the correct result is only assured if
                // wg_size >= radix_states
                counts_lacc[lid] += counts_lacc[wg_size * i + lid];
            }

            sycl::group_barrier(ndit.get_group());

            // count per work-group: reduce until count_lacc[] size >
            // radix_states (n_witems /= 2 per iteration)
            for (std::uint32_t n_witems = (wg_size >> 1);
                 n_witems >= radix_states; n_witems >>= 1)
            {
                if (lid < n_witems)
                    counts_lacc[lid] += counts_lacc[n_witems + lid];

                sycl::group_barrier(ndit.get_group());
            }

            const std::size_t iter_counter_offset = iter_id * n_counts;

            // count per work-group: write local count array to global count
            // array
            if (lid < radix_states) {
                // move buckets with the same id to adjacent positions,
                // thus splitting count array into radix_states regions
                counts_ptr[iter_counter_offset + (n_segments + 1) * lid +
                           wgr_id] = counts_lacc[lid];
            }

            // side work: reset 'no-operation-flag', signaling to skip re-order
            // phase
            if (wgr_id == 0 && lid == 0) {
                CountT &no_op_flag =
                    counts_ptr[iter_counter_offset + no_op_flag_id];
                no_op_flag = 0;
            }
        });
    });

    return local_count_ev;
}

//-----------------------------------------------------------------------
// radix sort: scan kernel (single iteration)
//-----------------------------------------------------------------------

template <typename KernelName, std::uint32_t radix_bits, typename CountT>
sycl::event radix_sort_scan_submit(sycl::queue &exec_q,
                                   std::size_t n_iters,
                                   std::size_t n_segments,
                                   std::size_t wg_size,
                                   std::size_t n_values,
                                   std::size_t n_counts,
                                   CountT *counts_ptr,
                                   const std::vector<sycl::event> depends)
{
    const std::size_t no_op_flag_id = n_counts - 1;

    // Scan produces local offsets using count values.
    // There are no local offsets for the first segment, but the rest segments
    // should be scanned with respect to the count value in the first segment
    // what requires n + 1 positions
    const std::size_t scan_size = n_segments + 1;
    wg_size = std::min(scan_size, wg_size);

    static constexpr std::uint32_t radix_states = std::uint32_t(1)
                                                  << radix_bits;

    // compilation of the kernel prevents out of resources issue, which may
    // occur due to usage of collective algorithms such as joint_exclusive_scan
    // even if local memory is not explicitly requested
    sycl::event scan_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        sycl::nd_range<1> ndRange(n_iters * radix_states * wg_size, wg_size);

        cgh.parallel_for<KernelName>(ndRange, [=](sycl::nd_item<1> ndit) {
            const std::size_t group_id = ndit.get_group(0);
            const std::size_t iter_id = group_id / radix_states;
            const std::size_t wgr_id = group_id - iter_id * radix_states;
            // find borders of a region with a specific bucket id
            auto begin_ptr =
                counts_ptr + scan_size * wgr_id + iter_id * n_counts;

            sycl::joint_exclusive_scan(ndit.get_group(), begin_ptr,
                                       begin_ptr + scan_size, begin_ptr,
                                       CountT(0), sycl::plus<CountT>{});

            const auto lid = ndit.get_local_linear_id();

            // NB: No race condition here, because the condition may ever be
            // true for only on one WG, one WI.
            if ((lid == wg_size - 1) && (begin_ptr[scan_size - 1] == n_values))
            {
                // set flag, since all the values got into one
                // this is optimization, may happy often for
                // higher radix offsets (all zeros)
                auto &no_op_flag =
                    counts_ptr[iter_id * n_counts + no_op_flag_id];
                no_op_flag = 1;
            }
        });
    });

    return scan_ev;
}

//-----------------------------------------------------------------------
// radix sort: group level reorder algorithms
//-----------------------------------------------------------------------

struct empty_storage
{
    template <typename... T>
    empty_storage(T &&...)
    {
    }
};

// Number with `n` least significant bits of uint32_t
inline std::uint32_t n_ls_bits_set(std::uint32_t n) noexcept
{
    static constexpr std::uint32_t zero{};
    static constexpr std::uint32_t all_bits_set = ~zero;

    return ~(all_bits_set << n);
}

enum class peer_prefix_algo
{
    subgroup_ballot,
    atomic_fetch_or,
    scan_then_broadcast
};

template <typename OffsetT, peer_prefix_algo Algo>
struct peer_prefix_helper;

template <typename AccT>
auto get_accessor_pointer(const AccT &acc)
{
    return acc.template get_multi_ptr<sycl::access::decorated::no>().get();
}

template <typename OffsetT>
struct peer_prefix_helper<OffsetT, peer_prefix_algo::atomic_fetch_or>
{
    using AtomicT = sycl::atomic_ref<std::uint32_t,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space>;
    using TempStorageT = sycl::local_accessor<std::uint32_t, 1>;

private:
    sycl::sub_group sgroup;
    std::uint32_t lid;
    std::uint32_t item_mask;
    AtomicT atomic_peer_mask;

public:
    peer_prefix_helper(sycl::nd_item<1> ndit, TempStorageT lacc)
        : sgroup(ndit.get_sub_group()), lid(ndit.get_local_linear_id()),
          item_mask(n_ls_bits_set(lid)), atomic_peer_mask(lacc[0])
    {
    }

    std::uint32_t peer_contribution(OffsetT &new_offset_id,
                                    OffsetT offset_prefix,
                                    bool wi_bit_set) const
    {
        // reset mask for each radix state
        if (lid == 0)
            atomic_peer_mask.store(std::uint32_t{0});
        sycl::group_barrier(sgroup);

        const std::uint32_t uint_contrib{wi_bit_set ? std::uint32_t{1}
                                                    : std::uint32_t{0}};

        // set local id's bit to 1 if the bucket value matches the radix state
        atomic_peer_mask.fetch_or(uint_contrib << lid);
        sycl::group_barrier(sgroup);
        std::uint32_t peer_mask_bits = atomic_peer_mask.load();
        std::uint32_t sg_total_offset = sycl::popcount(peer_mask_bits);

        // get the local offset index from the bits set in the peer mask with
        // index less than the work item ID
        peer_mask_bits &= item_mask;
        new_offset_id |= wi_bit_set
                             ? (offset_prefix + sycl::popcount(peer_mask_bits))
                             : OffsetT{0};
        return sg_total_offset;
    }
};

template <typename OffsetT>
struct peer_prefix_helper<OffsetT, peer_prefix_algo::scan_then_broadcast>
{
    using TempStorageT = empty_storage;
    using ItemType = sycl::nd_item<1>;
    using SubGroupType = sycl::sub_group;

private:
    SubGroupType sgroup;
    std::uint32_t sg_size;

public:
    peer_prefix_helper(sycl::nd_item<1> ndit, TempStorageT)
        : sgroup(ndit.get_sub_group()), sg_size(sgroup.get_local_range()[0])
    {
    }

    std::uint32_t peer_contribution(OffsetT &new_offset_id,
                                    OffsetT offset_prefix,
                                    bool wi_bit_set) const
    {
        const std::uint32_t contrib{wi_bit_set ? std::uint32_t{1}
                                               : std::uint32_t{0}};

        std::uint32_t sg_item_offset = sycl::exclusive_scan_over_group(
            sgroup, contrib, sycl::plus<std::uint32_t>{});

        new_offset_id |=
            (wi_bit_set ? (offset_prefix + sg_item_offset) : OffsetT(0));

        // the last scanned value does not contain number of all copies, thus
        // adding contribution
        std::uint32_t sg_total_offset = sycl::group_broadcast(
            sgroup, sg_item_offset + contrib, sg_size - 1);

        return sg_total_offset;
    }
};

template <typename OffsetT>
struct peer_prefix_helper<OffsetT, peer_prefix_algo::subgroup_ballot>
{
private:
    sycl::sub_group sgroup;
    std::uint32_t lid;
    sycl::ext::oneapi::sub_group_mask item_sg_mask;

    sycl::ext::oneapi::sub_group_mask mask_builder(std::uint32_t mask,
                                                   std::uint32_t sg_size)
    {
        return sycl::detail::Builder::createSubGroupMask<
            sycl::ext::oneapi::sub_group_mask>(mask, sg_size);
    }

public:
    using TempStorageT = empty_storage;

    peer_prefix_helper(sycl::nd_item<1> ndit, TempStorageT)
        : sgroup(ndit.get_sub_group()), lid(ndit.get_local_linear_id()),
          item_sg_mask(
              mask_builder(n_ls_bits_set(lid), sgroup.get_local_linear_range()))
    {
    }

    std::uint32_t peer_contribution(OffsetT &new_offset_id,
                                    OffsetT offset_prefix,
                                    bool wi_bit_set) const
    {
        // set local id's bit to 1 if the bucket value matches the radix state
        auto peer_mask = sycl::ext::oneapi::group_ballot(sgroup, wi_bit_set);
        std::uint32_t peer_mask_bits{};

        peer_mask.extract_bits(peer_mask_bits);
        std::uint32_t sg_total_offset = sycl::popcount(peer_mask_bits);

        // get the local offset index from the bits set in the peer mask with
        // index less than the work item ID
        peer_mask &= item_sg_mask;
        peer_mask.extract_bits(peer_mask_bits);

        new_offset_id |= wi_bit_set
                             ? (offset_prefix + sycl::popcount(peer_mask_bits))
                             : OffsetT(0);

        return sg_total_offset;
    }
};

template <typename InputT, typename OutputT>
void copy_func_for_radix_sort(const std::size_t n_segments,
                              const std::size_t elems_per_segment,
                              const std::size_t sg_size,
                              const std::uint32_t lid,
                              const std::size_t wgr_id,
                              const InputT *input_ptr,
                              const std::size_t n_values,
                              OutputT *output_ptr)
{
    // item info
    const std::size_t seg_start = elems_per_segment * wgr_id;

    std::size_t seg_end = sycl::min(seg_start + elems_per_segment, n_values);

    // ensure that each work item in a subgroup does the same number of loop
    // iterations
    const std::uint16_t tail_size = (seg_end - seg_start) % sg_size;
    seg_end -= tail_size;

    // find offsets for the same values within a segment and fill the resulting
    // buffer
    for (std::size_t val_id = seg_start + lid; val_id < seg_end;
         val_id += sg_size)
    {
        output_ptr[val_id] = std::move(input_ptr[val_id]);
    }

    if (tail_size > 0 && lid < tail_size) {
        const std::size_t val_id = seg_end + lid;
        output_ptr[val_id] = std::move(input_ptr[val_id]);
    }
}

//-----------------------------------------------------------------------
// radix sort: reorder kernel (per iteration)
//-----------------------------------------------------------------------
template <typename KernelName,
          std::uint32_t radix_bits,
          peer_prefix_algo PeerAlgo,
          typename InputT,
          typename OutputT,
          typename OffsetT,
          typename ProjT>
sycl::event
    radix_sort_reorder_submit(sycl::queue &exec_q,
                              std::size_t n_iters,
                              std::size_t n_segments,
                              std::uint32_t radix_offset,
                              std::size_t n_values,
                              const InputT *input_ptr,
                              OutputT *output_ptr,
                              std::size_t n_offsets,
                              OffsetT *offset_ptr,
                              const ProjT &proj_op,
                              const bool is_ascending,
                              const std::vector<sycl::event> dependency_events)
{
    using ValueT = InputT;
    using PeerHelper = peer_prefix_helper<OffsetT, PeerAlgo>;

    static constexpr std::uint32_t radix_states = std::uint32_t{1}
                                                  << radix_bits;
    static constexpr std::uint32_t radix_mask = radix_states - 1;
    const std::size_t elems_per_segment =
        (n_values + n_segments - 1) / n_segments;

    const std::size_t no_op_flag_id = n_offsets - 1;

    const auto &kernel_id = sycl::get_kernel_id<KernelName>();

    auto const &ctx = exec_q.get_context();
    auto const &dev = exec_q.get_device();
    auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
        ctx, {dev}, {kernel_id});

    auto krn = kb.get_kernel(kernel_id);

    const std::uint32_t sg_size = krn.template get_info<
        sycl::info::kernel_device_specific::max_sub_group_size>(dev);

    sycl::event reorder_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependency_events);
        cgh.use_kernel_bundle(kb);

        using StorageT = typename PeerHelper::TempStorageT;

        StorageT peer_temp(1, cgh);

        sycl::range<1> lRange{sg_size};
        sycl::range<1> gRange{n_iters * n_segments * sg_size};

        sycl::nd_range<1> ndRange{gRange, lRange};

        cgh.parallel_for<KernelName>(ndRange, [=](sycl::nd_item<1> ndit) {
            const std::size_t group_id = ndit.get_group(0);
            const std::size_t iter_id = group_id / n_segments;
            const std::size_t segment_id = group_id - iter_id * n_segments;

            auto b_offset_ptr = offset_ptr + iter_id * n_offsets;
            auto b_input_ptr = input_ptr + iter_id * n_values;
            auto b_output_ptr = output_ptr + iter_id * n_values;

            const std::uint32_t lid = ndit.get_local_id(0);

            auto &no_op_flag = b_offset_ptr[no_op_flag_id];
            if (no_op_flag) {
                // no reordering necessary, simply copy
                copy_func_for_radix_sort<InputT, OutputT>(
                    n_segments, elems_per_segment, sg_size, lid, segment_id,
                    b_input_ptr, n_values, b_output_ptr);
                return;
            }

            // create a private array for storing offset values
            // and add total offset and offset for compute unit
            // for a certain radix state
            std::array<OffsetT, radix_states> offset_arr{};
            const std::size_t scan_size = n_segments + 1;

            OffsetT scanned_bin = 0;

            /* find cumulative offset */
            static constexpr std::uint32_t zero_radix_state_id = 0;
            offset_arr[zero_radix_state_id] = b_offset_ptr[segment_id];

            for (std::uint32_t radix_state_id = 1;
                 radix_state_id < radix_states; ++radix_state_id)
            {
                const std::uint32_t local_offset_id =
                    segment_id + scan_size * radix_state_id;

                // scan bins serially
                const std::size_t last_segment_bucket_id =
                    radix_state_id * scan_size - 1;
                scanned_bin += b_offset_ptr[last_segment_bucket_id];

                offset_arr[radix_state_id] =
                    scanned_bin + b_offset_ptr[local_offset_id];
            }

            const std::size_t seg_start = elems_per_segment * segment_id;
            std::size_t seg_end =
                sycl::min(seg_start + elems_per_segment, n_values);
            // ensure that each work item in a subgroup does the same number of
            // loop iterations
            const std::uint32_t tail_size = (seg_end - seg_start) % sg_size;
            seg_end -= tail_size;

            const PeerHelper peer_prefix_hlp(ndit, peer_temp);

            // find offsets for the same values within a segment and fill the
            // resulting buffer
            if (is_ascending) {
                for (std::size_t val_id = seg_start + lid; val_id < seg_end;
                     val_id += sg_size)
                {
                    ValueT in_val = std::move(b_input_ptr[val_id]);

                    // get the bucket for the bit-ordered input value, applying
                    // the offset and mask for radix bits
                    const auto mapped_val =
                        order_preserving_cast</*is_ascending*/ true>(
                            proj_op(in_val));
                    std::uint32_t bucket_id =
                        get_bucket_id<radix_mask>(mapped_val, radix_offset);

                    OffsetT new_offset_id = 0;
                    for (std::uint32_t radix_state_id = 0;
                         radix_state_id < radix_states; ++radix_state_id)
                    {
                        bool is_current_bucket = (bucket_id == radix_state_id);
                        std::uint32_t sg_total_offset =
                            peer_prefix_hlp.peer_contribution(
                                /* modified by reference */ new_offset_id,
                                offset_arr[radix_state_id],
                                /* bit contribution from this work-item */
                                is_current_bucket);
                        offset_arr[radix_state_id] += sg_total_offset;
                    }
                    b_output_ptr[new_offset_id] = std::move(in_val);
                }
            }
            else {
                for (std::size_t val_id = seg_start + lid; val_id < seg_end;
                     val_id += sg_size)
                {
                    ValueT in_val = std::move(b_input_ptr[val_id]);

                    // get the bucket for the bit-ordered input value, applying
                    // the offset and mask for radix bits
                    const auto mapped_val =
                        order_preserving_cast</*is_ascending*/ false>(
                            proj_op(in_val));
                    std::uint32_t bucket_id =
                        get_bucket_id<radix_mask>(mapped_val, radix_offset);

                    OffsetT new_offset_id = 0;
                    for (std::uint32_t radix_state_id = 0;
                         radix_state_id < radix_states; ++radix_state_id)
                    {
                        bool is_current_bucket = (bucket_id == radix_state_id);
                        std::uint32_t sg_total_offset =
                            peer_prefix_hlp.peer_contribution(
                                /* modified by reference */ new_offset_id,
                                offset_arr[radix_state_id],
                                /* bit contribution from this work-item */
                                is_current_bucket);
                        offset_arr[radix_state_id] += sg_total_offset;
                    }
                    b_output_ptr[new_offset_id] = std::move(in_val);
                }
            }
            if (tail_size > 0) {
                ValueT in_val;

                // default: is greater than any actual radix state
                std::uint32_t bucket_id = radix_states;
                if (lid < tail_size) {
                    in_val = std::move(b_input_ptr[seg_end + lid]);

                    const auto proj_val = proj_op(in_val);
                    const auto mapped_val =
                        (is_ascending)
                            ? order_preserving_cast</*is_ascending*/ true>(
                                  proj_val)
                            : order_preserving_cast</*is_ascending*/ false>(
                                  proj_val);
                    bucket_id =
                        get_bucket_id<radix_mask>(mapped_val, radix_offset);
                }

                OffsetT new_offset_id = 0;
                for (std::uint32_t radix_state_id = 0;
                     radix_state_id < radix_states; ++radix_state_id)
                {
                    bool is_current_bucket = (bucket_id == radix_state_id);
                    std::uint32_t sg_total_offset =
                        peer_prefix_hlp.peer_contribution(
                            new_offset_id, offset_arr[radix_state_id],
                            is_current_bucket);

                    offset_arr[radix_state_id] += sg_total_offset;
                }

                if (lid < tail_size) {
                    b_output_ptr[new_offset_id] = std::move(in_val);
                }
            }
        });
    });

    return reorder_ev;
}

template <typename sizeT>
sizeT _slm_adjusted_work_group_size(sycl::queue &exec_q,
                                    sizeT required_slm_bytes_per_wg,
                                    sizeT wg_size)
{
    const auto &dev = exec_q.get_device();

    if (wg_size == 0)
        wg_size =
            dev.template get_info<sycl::info::device::max_work_group_size>();

    const auto local_mem_sz =
        dev.template get_info<sycl::info::device::local_mem_size>();

    return sycl::min(local_mem_sz / required_slm_bytes_per_wg, wg_size);
}

//-----------------------------------------------------------------------
// radix sort: one iteration
//-----------------------------------------------------------------------

template <std::uint32_t radix_bits, bool even>
struct parallel_radix_sort_iteration_step
{
    template <typename... Name>
    using count_phase = radix_sort_count_kernel<radix_bits, even, Name...>;
    template <typename... Name>
    using local_scan_phase = radix_sort_scan_kernel<radix_bits, Name...>;
    template <typename... Name>
    using reorder_peer_phase =
        radix_sort_reorder_peer_kernel<radix_bits, even, Name...>;
    template <typename... Name>
    using reorder_phase = radix_sort_reorder_kernel<radix_bits, even, Name...>;

    template <typename InputT,
              typename OutputT,
              typename CountT,
              typename ProjT>
    static sycl::event submit(sycl::queue &exec_q,
                              std::size_t n_iters,
                              std::size_t n_segments,
                              std::uint32_t radix_iter,
                              std::size_t n_values,
                              const InputT *in_ptr,
                              OutputT *out_ptr,
                              std::size_t n_counts,
                              CountT *counts_ptr,
                              const ProjT &proj_op,
                              const bool is_ascending,
                              const std::vector<sycl::event> &dependency_events)
    {
        using _RadixCountKernel = count_phase<InputT, OutputT, CountT, ProjT>;
        using _RadixLocalScanKernel =
            local_scan_phase<InputT, OutputT, CountT, ProjT>;
        using _RadixReorderPeerKernel =
            reorder_peer_phase<InputT, OutputT, CountT, ProjT>;
        using _RadixReorderKernel =
            reorder_phase<InputT, OutputT, CountT, ProjT>;

        const auto &supported_sub_group_sizes =
            exec_q.get_device()
                .template get_info<sycl::info::device::sub_group_sizes>();
        const std::size_t max_sg_size =
            (supported_sub_group_sizes.empty()
                 ? 0
                 : supported_sub_group_sizes.back());
        const std::size_t reorder_sg_size = max_sg_size;
        const std::size_t scan_wg_size =
            exec_q.get_device()
                .template get_info<sycl::info::device::max_work_group_size>();

        static constexpr std::size_t two_mils = (std::size_t(1) << 21);
        std::size_t count_wg_size =
            ((max_sg_size > 0) && (n_values > two_mils) ? 128 : max_sg_size);

        static constexpr std::uint32_t radix_states = std::uint32_t(1)
                                                      << radix_bits;

        // correct count_wg_size according to local memory limit in count phase
        const auto max_count_wg_size = _slm_adjusted_work_group_size(
            exec_q, sizeof(CountT) * radix_states, count_wg_size);
        count_wg_size =
            static_cast<::std::size_t>((max_count_wg_size / radix_states)) *
            radix_states;

        // work-group size must be a power of 2 and not less than the number of
        // states, for scanning to work correctly

        const std::size_t rounded_down_count_wg_size =
            std::size_t{1} << (number_of_bits_in_type<std::size_t>() -
                               sycl::clz(count_wg_size) - 1);
        count_wg_size =
            sycl::max(rounded_down_count_wg_size, std::size_t(radix_states));

        // Compute the radix position for the given iteration
        std::uint32_t radix_offset = radix_iter * radix_bits;

        // 1. Count Phase
        sycl::event count_ev =
            radix_sort_count_submit<_RadixCountKernel, radix_bits>(
                exec_q, n_iters, n_segments, count_wg_size, radix_offset,
                n_values, in_ptr, n_counts, counts_ptr, proj_op, is_ascending,
                dependency_events);

        // 2. Scan Phase
        sycl::event scan_ev =
            radix_sort_scan_submit<_RadixLocalScanKernel, radix_bits>(
                exec_q, n_iters, n_segments, scan_wg_size, n_values, n_counts,
                counts_ptr, {count_ev});

        // 3. Reorder Phase
        sycl::event reorder_ev{};
        // subgroup_ballot-based peer algo uses extract_bits to populate
        // uint32_t mask and hence relies on sub-group to be 32 or narrower
        static constexpr std::size_t sg32_v = 32u;
        static constexpr std::size_t sg16_v = 16u;
        static constexpr std::size_t sg08_v = 8u;
        if (sg32_v == reorder_sg_size || sg16_v == reorder_sg_size ||
            sg08_v == reorder_sg_size)
        {
            static constexpr auto peer_algorithm =
                peer_prefix_algo::subgroup_ballot;

            reorder_ev = radix_sort_reorder_submit<_RadixReorderPeerKernel,
                                                   radix_bits, peer_algorithm>(
                exec_q, n_iters, n_segments, radix_offset, n_values, in_ptr,
                out_ptr, n_counts, counts_ptr, proj_op, is_ascending,
                {scan_ev});
        }
        else {
            static constexpr auto peer_algorithm =
                peer_prefix_algo::scan_then_broadcast;

            reorder_ev = radix_sort_reorder_submit<_RadixReorderKernel,
                                                   radix_bits, peer_algorithm>(
                exec_q, n_iters, n_segments, radix_offset, n_values, in_ptr,
                out_ptr, n_counts, counts_ptr, proj_op, is_ascending,
                {scan_ev});
        }

        return reorder_ev;
    }
}; // struct parallel_radix_sort_iteration

template <typename Names, std::uint16_t... Constants>
class radix_sort_one_wg_krn;

template <typename KernelNameBase,
          std::uint16_t wg_size = 256,
          std::uint16_t block_size = 16,
          std::uint32_t radix = 4,
          std::uint16_t req_sub_group_size = (block_size < 4 ? 32 : 16)>
struct subgroup_radix_sort
{
private:
    class use_slm_tag
    {
    };
    class use_global_mem_tag
    {
    };

public:
    template <typename ValueT, typename OutputT, typename ProjT>
    sycl::event operator()(sycl::queue &exec_q,
                           std::size_t n_iters,
                           std::size_t n_to_sort,
                           ValueT *input_ptr,
                           OutputT *output_ptr,
                           ProjT proj_op,
                           const bool is_ascending,
                           const std::vector<sycl::event> &depends)
    {
        static_assert(std::is_same_v<std::remove_cv_t<ValueT>, OutputT>);

        using _SortKernelLoc =
            radix_sort_one_wg_krn<KernelNameBase, wg_size, block_size, 0>;
        using _SortKernelPartGlob =
            radix_sort_one_wg_krn<KernelNameBase, wg_size, block_size, 1>;
        using _SortKernelGlob =
            radix_sort_one_wg_krn<KernelNameBase, wg_size, block_size, 2>;

        static constexpr std::size_t max_concurrent_work_groups = 128U;

        // Choose this to occupy the entire accelerator
        const std::size_t n_work_groups =
            std::min<std::size_t>(n_iters, max_concurrent_work_groups);

        // determine which temporary allocation can be accommodated in SLM
        const auto &SLM_availability =
            check_slm_size<ValueT>(exec_q, n_to_sort);

        const std::size_t n_batch_size = n_work_groups;

        switch (SLM_availability) {
        case temp_allocations::both_in_slm:
        {
            static constexpr auto storage_for_values = use_slm_tag{};
            static constexpr auto storage_for_counters = use_slm_tag{};

            return one_group_submitter<_SortKernelLoc>()(
                exec_q, n_iters, n_iters, n_to_sort, input_ptr, output_ptr,
                proj_op, is_ascending, storage_for_values, storage_for_counters,
                depends);
        }
        case temp_allocations::counters_in_slm:
        {
            static constexpr auto storage_for_values = use_global_mem_tag{};
            static constexpr auto storage_for_counters = use_slm_tag{};

            return one_group_submitter<_SortKernelPartGlob>()(
                exec_q, n_iters, n_batch_size, n_to_sort, input_ptr, output_ptr,
                proj_op, is_ascending, storage_for_values, storage_for_counters,
                depends);
        }
        default:
        {
            static constexpr auto storage_for_values = use_global_mem_tag{};
            static constexpr auto storage_for_counters = use_global_mem_tag{};

            return one_group_submitter<_SortKernelGlob>()(
                exec_q, n_iters, n_batch_size, n_to_sort, input_ptr, output_ptr,
                proj_op, is_ascending, storage_for_values, storage_for_counters,
                depends);
        }
        }
    }

private:
    template <typename KeyT, typename>
    class TempBuf;

    template <typename KeyT>
    class TempBuf<KeyT, use_slm_tag>
    {
        std::size_t buf_size;

    public:
        TempBuf(std::size_t, std::size_t n) : buf_size(n) {}
        auto get_acc(sycl::handler &cgh)
        {
            return sycl::local_accessor<KeyT>(buf_size, cgh);
        }

        std::size_t get_iter_stride() const { return std::size_t{0}; }
    };

    template <typename KeyT>
    class TempBuf<KeyT, use_global_mem_tag>
    {
        sycl::buffer<KeyT> buf;
        std::size_t iter_stride;

    public:
        TempBuf(std::size_t n_iters, std::size_t n)
            : buf(n_iters * n), iter_stride(n)
        {
        }
        auto get_acc(sycl::handler &cgh)
        {
            return sycl::accessor(buf, cgh, sycl::read_write, sycl::no_init);
        }
        std::size_t get_iter_stride() const { return iter_stride; }
    };

    static_assert(wg_size <= 1024);
    static constexpr std::uint16_t bin_count = (1 << radix);
    static constexpr std::uint16_t counter_buf_sz = wg_size * bin_count + 1;

    enum class temp_allocations
    {
        both_in_slm,
        counters_in_slm,
        both_in_global_mem
    };

    template <typename T, typename SizeT>
    temp_allocations check_slm_size(const sycl::queue &exec_q, SizeT n)
    {
        // the kernel is designed for data size <= 64K
        assert(n <= (SizeT(1) << 16));

        static constexpr auto req_slm_size_counters =
            counter_buf_sz * sizeof(std::uint16_t);

        const auto &dev = exec_q.get_device();

        // Pessimistically only use half of the memory to take into account
        // a SYCL group algorithm might use a portion of SLM
        const std::size_t max_slm_size =
            dev.template get_info<sycl::info::device::local_mem_size>() / 2;

        const auto n_uniform = 1 << ceil_log2(n);
        const auto req_slm_size_val = sizeof(T) * n_uniform;

        return ((req_slm_size_val + req_slm_size_counters) <= max_slm_size)
                   ?
                   // the values and the counters are placed in SLM
                   temp_allocations::both_in_slm
                   : (req_slm_size_counters <= max_slm_size)
                         ?
                         // the counters are placed in SLM, the values - in the
                         // global memory
                         temp_allocations::counters_in_slm
                         :
                         // the values and the counters are placed in the global
                         // memory
                         temp_allocations::both_in_global_mem;
    }

    template <typename KernelName>
    struct one_group_submitter
    {
        template <typename InputT,
                  typename OutputT,
                  typename ProjT,
                  typename SLM_value_tag,
                  typename SLM_counter_tag>
        sycl::event operator()(sycl::queue &exec_q,
                               std::size_t n_iters,
                               std::size_t n_batch_size,
                               std::size_t n_values,
                               InputT *input_arr,
                               OutputT *output_arr,
                               const ProjT &proj_op,
                               const bool is_ascending,
                               SLM_value_tag,
                               SLM_counter_tag,
                               const std::vector<sycl::event> &depends)
        {
            assert(!(n_values >> 16));

            assert(n_values <= static_cast<std::size_t>(block_size) *
                                   static_cast<std::size_t>(wg_size));

            const std::uint16_t n = static_cast<std::uint16_t>(n_values);
            static_assert(std::is_same_v<std::remove_cv_t<InputT>, OutputT>);

            using ValueT = OutputT;

            using KeyT = std::invoke_result_t<ProjT, ValueT>;

            TempBuf<ValueT, SLM_value_tag> buf_val(
                n_batch_size, static_cast<std::size_t>(block_size * wg_size));
            TempBuf<std::uint16_t, SLM_counter_tag> buf_count(
                n_batch_size, static_cast<std::size_t>(counter_buf_sz));

            sycl::range<1> lRange{wg_size};

            sycl::event sort_ev;
            std::vector<sycl::event> deps{depends};

            const std::size_t n_batches =
                (n_iters + n_batch_size - 1) / n_batch_size;

            const auto &kernel_id = sycl::get_kernel_id<KernelName>();

            auto const &ctx = exec_q.get_context();
            auto const &dev = exec_q.get_device();
            auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
                ctx, {dev}, {kernel_id});

            const auto &krn = kb.get_kernel(kernel_id);

            const std::uint32_t krn_sg_size = krn.template get_info<
                sycl::info::kernel_device_specific::max_sub_group_size>(dev);

            // due to a bug in CPU device implementation, an additional
            // synchronization is necessary for short sub-group sizes
            const bool work_around_needed =
                exec_q.get_device().has(sycl::aspect::cpu) &&
                (krn_sg_size < 16);

            for (std::size_t batch_id = 0; batch_id < n_batches; ++batch_id) {

                const std::size_t block_start = batch_id * n_batch_size;

                // input_arr/output_arr each has shape (n_iters, n)
                InputT *this_input_arr = input_arr + block_start * n_values;
                OutputT *this_output_arr = output_arr + block_start * n_values;

                const std::size_t block_end =
                    std::min<std::size_t>(block_start + n_batch_size, n_iters);

                sycl::range<1> gRange{(block_end - block_start) * wg_size};
                sycl::nd_range ndRange{gRange, lRange};

                sort_ev = exec_q.submit([&](sycl::handler &cgh) {
                    cgh.depends_on(deps);
                    cgh.use_kernel_bundle(kb);

                    // allocation to use for value exchanges
                    auto exchange_acc = buf_val.get_acc(cgh);
                    const std::size_t exchange_acc_iter_stride =
                        buf_val.get_iter_stride();

                    // allocation for counters
                    auto counter_acc = buf_count.get_acc(cgh);
                    const std::size_t counter_acc_iter_stride =
                        buf_count.get_iter_stride();

                    cgh.parallel_for<KernelName>(ndRange, [=](sycl::nd_item<1>
                                                                  ndit) {
                        ValueT values[block_size];

                        const std::size_t iter_id = ndit.get_group(0);
                        const std::size_t iter_val_offset =
                            iter_id * static_cast<std::size_t>(n);
                        const std::size_t iter_counter_offset =
                            iter_id * counter_acc_iter_stride;
                        const std::size_t iter_exchange_offset =
                            iter_id * exchange_acc_iter_stride;

                        std::uint16_t wi = ndit.get_local_linear_id();
                        std::uint16_t begin_bit = 0;

                        static constexpr std::uint16_t end_bit =
                            number_of_bits_in_type<KeyT>();

                        // copy from input array into values
#pragma unroll
                        for (std::uint16_t i = 0; i < block_size; ++i) {
                            const std::uint16_t id = wi * block_size + i;
                            values[i] =
                                (id < n) ? this_input_arr[iter_val_offset + id]
                                         : ValueT{};
                        }

                        while (true) {
                            // indices for indirect access in the "re-order"
                            // phase
                            std::uint16_t indices[block_size];
                            {
                                // pointers to bucket's counters
                                std::uint16_t *counters[block_size];

                                // counting phase
                                auto pcounter =
                                    get_accessor_pointer(counter_acc) +
                                    (wi + iter_counter_offset);

                                // initialize counters
#pragma unroll
                                for (std::uint16_t i = 0; i < bin_count; ++i)
                                    pcounter[i * wg_size] = std::uint16_t{0};

                                sycl::group_barrier(ndit.get_group());

                                if (is_ascending) {
#pragma unroll
                                    for (std::uint16_t i = 0; i < block_size;
                                         ++i)
                                    {
                                        const std::uint16_t id =
                                            wi * block_size + i;
                                        static constexpr std::uint16_t
                                            bin_mask = bin_count - 1;

                                        // points to the padded element, i.e. id
                                        // is in-range
                                        static constexpr std::uint16_t
                                            default_out_of_range_bin_id =
                                                bin_mask;

                                        const std::uint16_t bin =
                                            (id < n)
                                                ? get_bucket_id<bin_mask>(
                                                      order_preserving_cast<
                                                          /* is_ascending */
                                                          true>(
                                                          proj_op(values[i])),
                                                      begin_bit)
                                                : default_out_of_range_bin_id;

                                        // counting and local offset calculation
                                        counters[i] = &pcounter[bin * wg_size];
                                        indices[i] = *counters[i];
                                        *counters[i] = indices[i] + 1;

                                        if (work_around_needed) {
                                            sycl::group_barrier(
                                                ndit.get_group());
                                        }
                                    }
                                }
                                else {
#pragma unroll
                                    for (std::uint16_t i = 0; i < block_size;
                                         ++i)
                                    {
                                        const std::uint16_t id =
                                            wi * block_size + i;
                                        static constexpr std::uint16_t
                                            bin_mask = bin_count - 1;

                                        // points to the padded element, i.e. id
                                        // is in-range
                                        static constexpr std::uint16_t
                                            default_out_of_range_bin_id =
                                                bin_mask;

                                        const std::uint16_t bin =
                                            (id < n)
                                                ? get_bucket_id<bin_mask>(
                                                      order_preserving_cast<
                                                          /* is_ascending */
                                                          false>(
                                                          proj_op(values[i])),
                                                      begin_bit)
                                                : default_out_of_range_bin_id;

                                        // counting and local offset calculation
                                        counters[i] = &pcounter[bin * wg_size];
                                        indices[i] = *counters[i];
                                        *counters[i] = indices[i] + 1;

                                        if (work_around_needed) {
                                            sycl::group_barrier(
                                                ndit.get_group());
                                        }
                                    }
                                }

                                sycl::group_barrier(ndit.get_group());

                                // exclusive scan phase
                                {

                                    // scan contiguous numbers
                                    std::uint16_t bin_sum[bin_count];
                                    const std::size_t counter_offset0 =
                                        iter_counter_offset + wi * bin_count;
                                    bin_sum[0] = counter_acc[counter_offset0];

#pragma unroll
                                    for (std::uint16_t i = 1; i < bin_count;
                                         ++i)
                                        bin_sum[i] =
                                            bin_sum[i - 1] +
                                            counter_acc[counter_offset0 + i];

                                    sycl::group_barrier(ndit.get_group());

                                    // exclusive scan local sum
                                    std::uint16_t sum_scan =
                                        sycl::exclusive_scan_over_group(
                                            ndit.get_group(),
                                            bin_sum[bin_count - 1],
                                            sycl::plus<std::uint16_t>());

// add to local sum, generate exclusive scan result
#pragma unroll
                                    for (std::uint16_t i = 0; i < bin_count;
                                         ++i)
                                        counter_acc[counter_offset0 + i + 1] =
                                            sum_scan + bin_sum[i];

                                    if (wi == 0)
                                        counter_acc[iter_counter_offset + 0] =
                                            std::uint32_t{0};

                                    sycl::group_barrier(ndit.get_group());
                                }

#pragma unroll
                                for (std::uint16_t i = 0; i < block_size; ++i) {
                                    // a global index is a local offset plus a
                                    // global base index
                                    indices[i] += *counters[i];
                                }

                                sycl::group_barrier(ndit.get_group());
                            }

                            begin_bit += radix;

                            // "re-order" phase
                            sycl::group_barrier(ndit.get_group());
                            if (begin_bit >= end_bit) {
                                // the last iteration - writing out the result
#pragma unroll
                                for (std::uint16_t i = 0; i < block_size; ++i) {
                                    const std::uint16_t r = indices[i];
                                    if (r < n) {
                                        this_output_arr[iter_val_offset + r] =
                                            values[i];
                                    }
                                }

                                return;
                            }

                            // data exchange
#pragma unroll
                            for (std::uint16_t i = 0; i < block_size; ++i) {
                                const std::uint16_t r = indices[i];
                                if (r < n)
                                    exchange_acc[iter_exchange_offset + r] =
                                        values[i];
                            }

                            sycl::group_barrier(ndit.get_group());

#pragma unroll
                            for (std::uint16_t i = 0; i < block_size; ++i) {
                                const std::uint16_t id = wi * block_size + i;
                                if (id < n)
                                    values[i] =
                                        exchange_acc[iter_exchange_offset + id];
                            }

                            sycl::group_barrier(ndit.get_group());
                        }
                    });
                });

                deps = {sort_ev};
            }

            return sort_ev;
        }
    };
};

template <typename ValueT, typename ProjT>
struct OneWorkGroupRadixSortKernel;

//-----------------------------------------------------------------------
// radix sort: main function
//-----------------------------------------------------------------------
template <typename ValueT, typename ProjT>
sycl::event parallel_radix_sort_impl(sycl::queue &exec_q,
                                     std::size_t n_iters,
                                     std::size_t n_to_sort,
                                     const ValueT *input_arr,
                                     ValueT *output_arr,
                                     const ProjT &proj_op,
                                     const bool is_ascending,
                                     const std::vector<sycl::event> &depends)
{
    assert(n_to_sort > 1);

    using KeyT = std::remove_cv_t<
        std::remove_reference_t<std::invoke_result_t<ProjT, ValueT>>>;

    // radix bits represent number of processed bits in each value during one
    // iteration
    static constexpr std::uint32_t radix_bits = 4;

    sycl::event sort_ev{};

    const auto &dev = exec_q.get_device();
    const auto max_wg_size =
        dev.template get_info<sycl::info::device::max_work_group_size>();

    static constexpr std::uint16_t ref_wg_size = 64;
    if (n_to_sort <= 16384 && ref_wg_size * 8 <= max_wg_size) {
        using _RadixSortKernel = OneWorkGroupRadixSortKernel<ValueT, ProjT>;

        if (n_to_sort <= 64 && ref_wg_size <= max_wg_size) {
            // wg_size * block_size == 64 * 1 * 1 == 64
            static constexpr std::uint16_t wg_size = ref_wg_size;
            static constexpr std::uint16_t block_size = 1;

            sort_ev = subgroup_radix_sort<_RadixSortKernel, wg_size, block_size,
                                          radix_bits>{}(
                exec_q, n_iters, n_to_sort, input_arr, output_arr, proj_op,
                is_ascending, depends);
        }
        else if (n_to_sort <= 128 && ref_wg_size * 2 <= max_wg_size) {
            // wg_size * block_size == 64 * 2 * 1 == 128
            static constexpr std::uint16_t wg_size = ref_wg_size * 2;
            static constexpr std::uint16_t block_size = 1;

            sort_ev = subgroup_radix_sort<_RadixSortKernel, wg_size, block_size,
                                          radix_bits>{}(
                exec_q, n_iters, n_to_sort, input_arr, output_arr, proj_op,
                is_ascending, depends);
        }
        else if (n_to_sort <= 256 && ref_wg_size * 2 <= max_wg_size) {
            // wg_size * block_size == 64 * 2 * 2 == 256
            static constexpr std::uint16_t wg_size = ref_wg_size * 2;
            static constexpr std::uint16_t block_size = 2;

            sort_ev = subgroup_radix_sort<_RadixSortKernel, wg_size, block_size,
                                          radix_bits>{}(
                exec_q, n_iters, n_to_sort, input_arr, output_arr, proj_op,
                is_ascending, depends);
        }
        else if (n_to_sort <= 512 && ref_wg_size * 2 <= max_wg_size) {
            // wg_size * block_size == 64 * 2 * 4 == 512
            static constexpr std::uint16_t wg_size = ref_wg_size * 2;
            static constexpr std::uint16_t block_size = 4;

            sort_ev = subgroup_radix_sort<_RadixSortKernel, wg_size, block_size,
                                          radix_bits>{}(
                exec_q, n_iters, n_to_sort, input_arr, output_arr, proj_op,
                is_ascending, depends);
        }
        else if (n_to_sort <= 1024 && ref_wg_size * 2 <= max_wg_size) {
            // wg_size * block_size == 64 * 2 * 8 == 1024
            static constexpr std::uint16_t wg_size = ref_wg_size * 2;
            static constexpr std::uint16_t block_size = 8;

            sort_ev = subgroup_radix_sort<_RadixSortKernel, wg_size, block_size,
                                          radix_bits>{}(
                exec_q, n_iters, n_to_sort, input_arr, output_arr, proj_op,
                is_ascending, depends);
        }
        else if (n_to_sort <= 2048 && ref_wg_size * 4 <= max_wg_size) {
            // wg_size * block_size == 64 * 4 * 8 == 2048
            static constexpr std::uint16_t wg_size = ref_wg_size * 4;
            static constexpr std::uint16_t block_size = 8;

            sort_ev = subgroup_radix_sort<_RadixSortKernel, wg_size, block_size,
                                          radix_bits>{}(
                exec_q, n_iters, n_to_sort, input_arr, output_arr, proj_op,
                is_ascending, depends);
        }
        else if (n_to_sort <= 4096 && ref_wg_size * 4 <= max_wg_size) {
            // wg_size * block_size == 64 * 4 * 16 == 4096
            static constexpr std::uint16_t wg_size = ref_wg_size * 4;
            static constexpr std::uint16_t block_size = 16;

            sort_ev = subgroup_radix_sort<_RadixSortKernel, wg_size, block_size,
                                          radix_bits>{}(
                exec_q, n_iters, n_to_sort, input_arr, output_arr, proj_op,
                is_ascending, depends);
        }
        else if (n_to_sort <= 8192 && ref_wg_size * 8 <= max_wg_size) {
            // wg_size * block_size == 64 * 8 * 16 == 8192
            static constexpr std::uint16_t wg_size = ref_wg_size * 8;
            static constexpr std::uint16_t block_size = 16;

            sort_ev = subgroup_radix_sort<_RadixSortKernel, wg_size, block_size,
                                          radix_bits>{}(
                exec_q, n_iters, n_to_sort, input_arr, output_arr, proj_op,
                is_ascending, depends);
        }
        else {
            // wg_size * block_size == 64 * 8 * 32 == 16384
            static constexpr std::uint16_t wg_size = ref_wg_size * 8;
            static constexpr std::uint16_t block_size = 32;

            sort_ev = subgroup_radix_sort<_RadixSortKernel, wg_size, block_size,
                                          radix_bits>{}(
                exec_q, n_iters, n_to_sort, input_arr, output_arr, proj_op,
                is_ascending, depends);
        }
    }
    else {
        static constexpr std::uint32_t radix_iters =
            number_of_buckets_in_type<KeyT>(radix_bits);
        static constexpr std::uint32_t radix_states = std::uint32_t(1)
                                                      << radix_bits;

        static constexpr std::size_t bound_512k = (std::size_t(1) << 19);
        static constexpr std::size_t bound_2m = (std::size_t(1) << 21);

        const auto wg_sz_k = (n_to_sort < bound_512k)  ? 8
                             : (n_to_sort <= bound_2m) ? 4
                                                       : 1;
        const std::size_t wg_size = max_wg_size / wg_sz_k;

        const std::size_t n_segments = (n_to_sort + wg_size - 1) / wg_size;

        // Additional radix_states elements are used for getting local offsets
        // from count values + no_op flag; 'No operation' flag specifies whether
        // to skip re-order phase if the all keys are the same (lie in one bin)
        const std::size_t n_counts =
            (n_segments + 1) * radix_states + 1 /*no_op flag*/;

        using CountT = std::uint32_t;

        // memory for storing count and offset values
        auto count_owner =
            dpctl::tensor::alloc_utils::smart_malloc_device<CountT>(
                n_iters * n_counts, exec_q);

        CountT *count_ptr = count_owner.get();

        static constexpr std::uint32_t zero_radix_iter{0};

        if constexpr (std::is_same_v<KeyT, bool>) {

            sort_ev = parallel_radix_sort_iteration_step<
                radix_bits, /*even=*/true>::submit(exec_q, n_iters, n_segments,
                                                   zero_radix_iter, n_to_sort,
                                                   input_arr, output_arr,
                                                   n_counts, count_ptr, proj_op,
                                                   is_ascending, depends);

            sort_ev = dpctl::tensor::alloc_utils::async_smart_free(
                exec_q, {sort_ev}, count_owner);

            return sort_ev;
        }

        auto tmp_arr_owner =
            dpctl::tensor::alloc_utils::smart_malloc_device<ValueT>(
                n_iters * n_to_sort, exec_q);

        ValueT *tmp_arr = tmp_arr_owner.get();

        // iterations per each bucket
        assert("Number of iterations must be even" && radix_iters % 2 == 0);
        assert(radix_iters > 0);

        sort_ev = parallel_radix_sort_iteration_step<
            radix_bits, /*even=*/true>::submit(exec_q, n_iters, n_segments,
                                               zero_radix_iter, n_to_sort,
                                               input_arr, tmp_arr, n_counts,
                                               count_ptr, proj_op, is_ascending,
                                               depends);

        for (std::uint32_t radix_iter = 1; radix_iter < radix_iters;
             ++radix_iter)
        {
            if (radix_iter % 2 == 0) {
                sort_ev = parallel_radix_sort_iteration_step<
                    radix_bits,
                    /*even=*/true>::submit(exec_q, n_iters, n_segments,
                                           radix_iter, n_to_sort, output_arr,
                                           tmp_arr, n_counts, count_ptr,
                                           proj_op, is_ascending, {sort_ev});
            }
            else {
                sort_ev = parallel_radix_sort_iteration_step<
                    radix_bits,
                    /*even=*/false>::submit(exec_q, n_iters, n_segments,
                                            radix_iter, n_to_sort, tmp_arr,
                                            output_arr, n_counts, count_ptr,
                                            proj_op, is_ascending, {sort_ev});
            }
        }

        sort_ev = dpctl::tensor::alloc_utils::async_smart_free(
            exec_q, {sort_ev}, tmp_arr_owner, count_owner);
    }

    return sort_ev;
}

struct IdentityProj
{
    constexpr IdentityProj() {}

    template <typename T>
    constexpr T operator()(T val) const
    {
        return val;
    }
};

template <typename ValueT, typename IndexT>
struct ValueProj
{
    constexpr ValueProj() {}

    constexpr ValueT operator()(const std::pair<ValueT, IndexT> &pair) const
    {
        return pair.first;
    }
};

template <typename IndexT, typename ValueT, typename ProjT>
struct IndexedProj
{
    IndexedProj(const ValueT *arg_ptr) : ptr(arg_ptr), value_projector{} {}

    IndexedProj(const ValueT *arg_ptr, const ProjT &proj_op)
        : ptr(arg_ptr), value_projector(proj_op)
    {
    }

    auto operator()(IndexT i) const { return value_projector(ptr[i]); }

private:
    const ValueT *ptr;
    ProjT value_projector;
};

} // namespace radix_sort_details

using dpctl::tensor::ssize_t;

template <typename argTy>
sycl::event
    radix_sort_axis1_contig_impl(sycl::queue &exec_q,
                                 const bool sort_ascending,
                                 // number of sub-arrays to sort (num. of rows
                                 // in a matrix when sorting over rows)
                                 std::size_t iter_nelems,
                                 // size of each array to sort  (length of rows,
                                 // i.e. number of columns)
                                 std::size_t sort_nelems,
                                 const char *arg_cp,
                                 char *res_cp,
                                 ssize_t iter_arg_offset,
                                 ssize_t iter_res_offset,
                                 ssize_t sort_arg_offset,
                                 ssize_t sort_res_offset,
                                 const std::vector<sycl::event> &depends)
{
    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp) +
                          iter_arg_offset + sort_arg_offset;
    argTy *res_tp =
        reinterpret_cast<argTy *>(res_cp) + iter_res_offset + sort_res_offset;

    using Proj = radix_sort_details::IdentityProj;
    static constexpr Proj proj_op{};

    sycl::event radix_sort_ev =
        radix_sort_details::parallel_radix_sort_impl<argTy, Proj>(
            exec_q, iter_nelems, sort_nelems, arg_tp, res_tp, proj_op,
            sort_ascending, depends);

    return radix_sort_ev;
}

template <typename ValueT, typename IndexT>
class radix_argsort_index_write_out_krn;

template <typename ValueT, typename IndexT>
class radix_argsort_iota_krn;

template <typename argTy, typename IndexTy>
sycl::event
    radix_argsort_axis1_contig_impl(sycl::queue &exec_q,
                                    const bool sort_ascending,
                                    // number of sub-arrays to sort (num. of
                                    // rows in a matrix when sorting over rows)
                                    std::size_t iter_nelems,
                                    // size of each array to sort  (length of
                                    // rows, i.e. number of columns)
                                    std::size_t sort_nelems,
                                    const char *arg_cp,
                                    char *res_cp,
                                    ssize_t iter_arg_offset,
                                    ssize_t iter_res_offset,
                                    ssize_t sort_arg_offset,
                                    ssize_t sort_res_offset,
                                    const std::vector<sycl::event> &depends)
{
    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp) +
                          iter_arg_offset + sort_arg_offset;
    IndexTy *res_tp =
        reinterpret_cast<IndexTy *>(res_cp) + iter_res_offset + sort_res_offset;

    const std::size_t total_nelems = iter_nelems * sort_nelems;
    auto workspace_owner =
        dpctl::tensor::alloc_utils::smart_malloc_device<IndexTy>(total_nelems,
                                                                 exec_q);

    // get raw USM pointer
    IndexTy *workspace = workspace_owner.get();

    using IdentityProjT = radix_sort_details::IdentityProj;
    using IndexedProjT =
        radix_sort_details::IndexedProj<IndexTy, argTy, IdentityProjT>;
    const IndexedProjT proj_op{arg_tp};

    using IotaKernelName = radix_argsort_iota_krn<argTy, IndexTy>;

    using dpctl::tensor::kernels::sort_utils_detail::iota_impl;

    sycl::event iota_ev = iota_impl<IotaKernelName, IndexTy>(
        exec_q, workspace, total_nelems, depends);

    sycl::event radix_sort_ev =
        radix_sort_details::parallel_radix_sort_impl<IndexTy, IndexedProjT>(
            exec_q, iter_nelems, sort_nelems, workspace, res_tp, proj_op,
            sort_ascending, {iota_ev});

    using MapBackKernelName = radix_argsort_index_write_out_krn<argTy, IndexTy>;
    using dpctl::tensor::kernels::sort_utils_detail::map_back_impl;

    sycl::event dep = radix_sort_ev;

    // no need to perform map_back ( id % sort_nelems)
    //   if total_nelems == sort_nelems
    if (iter_nelems > 1u) {
        dep = map_back_impl<MapBackKernelName, IndexTy>(
            exec_q, total_nelems, res_tp, res_tp, sort_nelems, {dep});
    }

    sycl::event cleanup_ev = dpctl::tensor::alloc_utils::async_smart_free(
        exec_q, {dep}, workspace_owner);

    return cleanup_ev;
}

} // namespace dpctl::tensor::kernels
