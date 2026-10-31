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
/// This file defines radix select kernels for tensor topk, partition,
/// argpartition and median operations.
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "kernels/sorting/radix_utils.hpp"
#include "utils/sycl_alloc_utils.hpp"

namespace dpnp::tensor::kernels::radix_select_details
{

inline constexpr std::uint32_t radix_bits = 8;
inline constexpr std::uint32_t radix_states = std::uint32_t(1) << radix_bits;
inline constexpr std::uint32_t radix_mask = radix_states - 1;

/*! @brief Number of radix passes which resolve a key of type `KeyT` */
template <typename KeyT>
constexpr std::uint32_t n_radix_passes()
{
    return radix_utils::number_of_buckets_in_type<KeyT>(radix_bits);
}

/*! @brief Bucket holding the element of a given rank */
struct bucket_info
{
    std::uint32_t bucket_id;
    // number of elements in buckets preceding `bucket_id`
    std::uint64_t before;
    // number of elements in `bucket_id`
    std::uint64_t count;
};

/*! @brief Executable bundle of kernels, and the largest work-group size all
 * of them may be run with */
struct kernels_bundle
{
    sycl::kernel_bundle<sycl::bundle_state::executable> kb;
    std::size_t max_wg_size;
};

/*! @brief Bundle of the kernels `KernelNames` for the device of `exec_q` */
template <typename... KernelNames>
kernels_bundle get_kernels_bundle(sycl::queue &exec_q)
{
    const auto &ctx = exec_q.get_context();
    const auto &dev = exec_q.get_device();

    const std::vector<sycl::kernel_id> kernel_ids{
        sycl::get_kernel_id<KernelNames>()...};
    auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
        ctx, {dev}, kernel_ids);

    std::size_t max_wg_size = std::numeric_limits<std::size_t>::max();
    for (const auto &kernel_id : kernel_ids) {
        const auto krn = kb.get_kernel(kernel_id);
        max_wg_size = std::min(
            max_wg_size,
            krn.template get_info<
                sycl::info::kernel_device_specific::work_group_size>(dev));
    }
    return kernels_bundle{std::move(kb), max_wg_size};
}

//-----------------------------------------------------------------------
// radix select: what is written for each row
//-----------------------------------------------------------------------

enum class select_output
{
    // the k smallest elements, in unspecified order
    selected,
    // the whole row, the elements ordering before the k-th smallest ahead of
    // those equal to it, ahead of the others
    whole_row,
    // the lower and the upper median
    median,
};

/*! @brief Sink which discards the elements passed to it */
struct NoOut
{
    void operator()(std::uint64_t, std::size_t) const {}
};

/*! @brief Stores the value and the index of element `i` of a row as the
 * `j`-th selected element of the row */
template <typename ValueT, typename IndexT>
struct SelectedOut
{
    const ValueT *arg;
    ValueT *vals;
    IndexT *inds;

    void operator()(std::uint64_t j, std::size_t i) const
    {
        vals[j] = arg[i];
        inds[j] = static_cast<IndexT>(i);
    }
};

/*! @brief Stores the value of element `i` of a row at position `j` */
template <typename ValueT>
struct ValueOut
{
    const ValueT *arg;
    ValueT *vals;

    void operator()(std::uint64_t j, std::size_t i) const { vals[j] = arg[i]; }
};

/*! @brief Stores the index of element `i` of a row at position `j` */
template <typename IndexT>
struct IndexOut
{
    IndexT *inds;

    void operator()(std::uint64_t j, std::size_t i) const
    {
        inds[j] = static_cast<IndexT>(i);
    }
};

/*! @brief Passes element `i` to `out` at position `offset + j` */
template <typename OutT>
struct ShiftedOut
{
    OutT out;
    std::uint64_t offset;

    void operator()(std::uint64_t j, std::size_t i) const
    {
        out(offset + j, i);
    }
};

template <typename OutT>
ShiftedOut<OutT> shifted(const OutT &out, std::uint64_t offset)
{
    return {out, offset};
}

inline NoOut shifted(const NoOut &, std::uint64_t) { return {}; }

/*! @brief Passes candidate `i` of a row to `out` as the element of the row it
 * lists */
template <typename OutT>
struct CandidateOut
{
    const std::uint32_t *cand;
    OutT out;

    void operator()(std::uint64_t j, std::size_t i) const { out(j, cand[i]); }
};

template <typename OutT>
CandidateOut<OutT> via_candidates(const std::uint32_t *cand, const OutT &out)
{
    return {cand, out};
}

inline NoOut via_candidates(const std::uint32_t *, const NoOut &) { return {}; }

/*! @brief Lists element `i` of a row as its `j`-th candidate */
struct CandidateListOut
{
    std::uint32_t *cand;

    void operator()(std::uint64_t j, std::size_t i) const
    {
        cand[j] = static_cast<std::uint32_t>(i);
    }
};

/*! @brief The `k` smallest elements of each row and their indices, written
 * into `(n_iters, k)` arrays */
template <typename ValueT, typename IndexT>
struct TopKOp
{
    using value_type = ValueT;
    static constexpr select_output output = select_output::selected;

    ValueT *vals;
    IndexT *inds;
    std::size_t k;

    std::uint64_t rank(std::uint64_t) const { return k; }

    SelectedOut<ValueT, IndexT> row_out(std::size_t iter_id,
                                        const ValueT *row_arg) const
    {
        return {row_arg, vals + iter_id * k, inds + iter_id * k};
    }
};

/*! @brief Each row partitioned about its `k`-th smallest element, written
 * into an `(n_iters, n_values)` array */
template <typename ValueT>
struct PartitionOp
{
    using value_type = ValueT;
    static constexpr select_output output = select_output::whole_row;

    ValueT *vals;
    std::size_t n_values;
    std::size_t k;

    std::uint64_t rank(std::uint64_t) const { return k; }

    ValueOut<ValueT> row_out(std::size_t iter_id, const ValueT *row_arg) const
    {
        return {row_arg, vals + iter_id * n_values};
    }
};

/*! @brief The indices which partition each row about its `k`-th smallest
 * element, written into an `(n_iters, n_values)` array */
template <typename ValueT, typename IndexT>
struct ArgPartitionOp
{
    using value_type = ValueT;
    static constexpr select_output output = select_output::whole_row;

    IndexT *inds;
    std::size_t n_values;
    std::size_t k;

    std::uint64_t rank(std::uint64_t) const { return k; }

    IndexOut<IndexT> row_out(std::size_t iter_id, const ValueT *) const
    {
        return {inds + iter_id * n_values};
    }
};

/*! @brief The median of each row, the mean of its lower and upper medians,
 * written as `ResT` into an `(n_iters,)` array; it is NaN for a row with NaNs
 * unless `ignore_nan`, and for a row of NaNs only, which then sets
 * `*all_nan` if it is not null */
template <typename ValueT, typename ResT>
struct MedianOp
{
    using value_type = ValueT;
    static constexpr select_output output = select_output::median;

    ResT *out;
    std::uint32_t *all_nan;
    std::size_t n_values;
    bool ignore_nan;

    std::uint64_t n_valid(std::uint64_t n_nan) const
    {
        return n_values - n_nan;
    }

    /*! @brief 1-based rank of the upper median, 0 if the median is NaN */
    std::uint64_t rank(std::uint64_t n_nan) const
    {
        if (n_nan > 0 && !ignore_nan) {
            return 0;
        }
        const std::uint64_t n = n_valid(n_nan);
        return (n == 0) ? 0 : n / 2 + 1;
    }

    /*! @brief 0-based rank of the lower median */
    std::uint64_t lo_rank(std::uint64_t n_nan) const
    {
        return (n_valid(n_nan) - 1) / 2;
    }

    /*! @brief Whether the lower median is the largest element ordering
     * before the upper one, of 1-based rank `k`, rather than equal to it */
    bool lo_before(std::uint64_t k,
                   std::uint64_t n_less,
                   std::uint64_t n_nan) const
    {
        return (lo_rank(n_nan) + 1 != k) && (n_less + 1 == k);
    }

    NoOut row_out(std::size_t, const ValueT *) const { return {}; }

    void write(std::size_t iter_id, ValueT lo, ValueT hi) const
    {
        // a single median is written as is, its mean with itself could
        // overflow
        if (lo == hi) {
            out[iter_id] = static_cast<ResT>(hi);
        }
        else if constexpr (std::is_same_v<ResT, sycl::half>) {
            // as numpy, the mean of half floats is taken in single precision
            out[iter_id] = static_cast<ResT>(
                (static_cast<float>(lo) + static_cast<float>(hi)) / 2);
        }
        else {
            out[iter_id] =
                (static_cast<ResT>(lo) + static_cast<ResT>(hi)) / ResT(2);
        }
    }

    void write_nan(std::size_t iter_id) const
    {
        if constexpr (!std::is_integral_v<ValueT>) {
            out[iter_id] = std::numeric_limits<ResT>::quiet_NaN();
            // with NaNs ignored, the median is NaN for a row of NaNs only
            if (ignore_nan && all_nan != nullptr) {
                sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>
                    flag(*all_nan);
                flag.store(1);
            }
        }
    }

    /*! @brief Writes the median of a row given the keys of its lower and
     * upper medians */
    template <typename WideKeyT>
    void write_median_keys(std::size_t iter_id,
                           WideKeyT lo_key,
                           WideKeyT hi_key) const
    {
        using KeyT = radix_utils::radix_key_t<ValueT>;
        write(iter_id,
              radix_utils::ordered_radix_value<true, ValueT>(KeyT(lo_key)),
              radix_utils::ordered_radix_value<true, ValueT>(KeyT(hi_key)));
    }

    /*! @brief Writes the median of a row given the largest keys of its
     * elements ordering before the upper median, and of those equal to it */
    template <typename WideKeyT>
    void write_keys(std::size_t iter_id,
                    std::uint64_t k,
                    std::uint64_t n_less,
                    std::uint64_t n_nan,
                    WideKeyT less_max,
                    WideKeyT eq_max) const
    {
        write_median_keys(
            iter_id, (lo_before(k, n_less, n_nan)) ? less_max : eq_max, eq_max);
    }
};

/*! @brief Whether the first pass counts the NaNs of the rows */
template <typename OpT>
constexpr bool counts_nan()
{
    return OpT::output == select_output::median &&
           !std::is_integral_v<typename OpT::value_type>;
}

/*! @brief Key shared by the NaNs, 0 if the rows' NaNs are not counted */
template <bool is_ascending, typename OpT>
radix_utils::radix_key_t<typename OpT::value_type> nan_key()
{
    using ValueT = typename OpT::value_type;
    if constexpr (counts_nan<OpT>()) {
        return radix_utils::ordered_radix_key<is_ascending>(
            std::numeric_limits<ValueT>::quiet_NaN());
    }
    else {
        return 0;
    }
}

/*! @brief Whether the selection of a row ends with a bucket of `count`
 * elements holding the sought one at rank `k_rem` */
template <typename OpT>
bool bucket_resolves(std::uint64_t count, std::uint64_t k_rem)
{
    if constexpr (OpT::output == select_output::selected) {
        // the whole bucket is among the k smallest
        return count == k_rem;
    }
    else {
        // the sought element itself is needed, the bucket must hold it only
        // or be resolved to the last digit
        return count == 1;
    }
}

/*! @brief Whether the selection stops once the keys sharing the prefix are
 * all equal; the k smallest elements are resolved by counts alone, while the
 * sought element otherwise needs a bucket of one, which runs of equal values
 * never reach before the last pass. Equal keys never split, so whether they
 * are is only checked in a pass whose bucket keeps all of the keys which had
 * the prefix, which spread out keys rarely do */
template <typename OpT>
constexpr bool stops_on_equal()
{
    return OpT::output != select_output::selected;
}

/*! @brief Sink for the elements ordering after the sought one, which only a
 * partition of the whole row writes */
template <typename OpT, typename OutT>
auto greater_out(const OutT &out)
{
    if constexpr (OpT::output == select_output::whole_row) {
        return out;
    }
    else {
        return NoOut{};
    }
}

/*! @brief Unsigned type holding a key, at least 32 bits wide for reductions */
template <typename KeyT>
using wide_key_t = std::
    conditional_t<(sizeof(KeyT) < sizeof(std::uint32_t)), std::uint32_t, KeyT>;

//-----------------------------------------------------------------------
// radix select: work-group level building blocks
//-----------------------------------------------------------------------

/*! @brief What a work-item saw of the keys whose digits it counted */
template <typename KeyT>
struct digit_stats
{
    // number of the keys equal to the NaN key, if counted
    std::uint32_t n_nan;
    // bitwise or and and of the keys, if tracked, else no bits and all of
    // them, so the keys are all equal if and only if these are
    wide_key_t<KeyT> key_or;
    wide_key_t<KeyT> key_and;
};

/*! @brief Adds to the zeroed local histogram `hist` the digits at `shift` of
 * the keys in `[begin, end)` whose prefix under `mask` is `desired`, and
 * returns how many of the work-item's keys, if `count_nan`, equal `nan_key`,
 * and, if `track_bits`, their bitwise or and and */
template <bool count_nan = false,
          bool track_bits = false,
          typename KeyT,
          typename KeyFnT,
          typename HistAccT>
digit_stats<KeyT> count_digits(const sycl::nd_item<1> &ndit,
                               const KeyFnT &key_fn,
                               std::size_t begin,
                               std::size_t end,
                               KeyT desired,
                               KeyT mask,
                               std::uint32_t shift,
                               const HistAccT &hist,
                               KeyT nan_key = KeyT{0})
{
    using AtomicT = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space>;

    const auto &sg = ndit.get_sub_group();
    const std::uint32_t sg_size = sg.get_local_linear_range();
    const std::size_t lid = ndit.get_local_linear_id();
    const std::size_t wg_size = ndit.get_local_range(0);

    using WideKeyT = wide_key_t<KeyT>;
    digit_stats<KeyT> stats{0, WideKeyT(0),
                            WideKeyT(std::numeric_limits<KeyT>::max())};
    for (std::size_t i0 = begin; i0 < end; i0 += wg_size) {
        const std::size_t i = i0 + lid;

        bool match = false;
        std::uint32_t bucket_id = 0;
        if (i < end) {
            const KeyT key = key_fn(i);
            match = ((key & mask) == desired);
            bucket_id = radix_utils::get_bucket_id<radix_mask>(key, shift);
            if constexpr (count_nan) {
                stats.n_nan += (match && key == nan_key);
            }
            if constexpr (track_bits) {
                if (match) {
                    stats.key_or |= WideKeyT(key);
                    stats.key_and &= WideKeyT(key);
                }
            }
        }

        // a sub-group whose keys all land in one bucket, as for runs of
        // equal values, makes one update instead of contending sg_size times
        const std::uint32_t leader_bucket_id =
            sycl::group_broadcast(sg, bucket_id);
        if (sycl::all_of_group(sg, match && (bucket_id == leader_bucket_id))) {
            if (sg.leader()) {
                AtomicT(hist[bucket_id]).fetch_add(sg_size);
            }
        }
        else if (match) {
            AtomicT(hist[bucket_id]).fetch_add(std::uint32_t(1));
        }
    }
    return stats;
}

/*! @brief Finds the bucket of the complete local histogram `hist` holding
 * the element of 1-based rank `rank` */
template <typename HistAccT>
bucket_info find_bucket(const sycl::nd_item<1> &ndit,
                        const HistAccT &hist,
                        std::uint64_t rank)
{
    const auto &wg = ndit.get_group();
    const std::uint32_t lid = ndit.get_local_linear_id();
    const std::uint32_t wg_size = ndit.get_local_range(0);

    const std::uint32_t buckets_per_wi = (radix_states + wg_size - 1) / wg_size;
    const std::uint32_t bucket_begin =
        std::min(lid * buckets_per_wi, radix_states);
    const std::uint32_t bucket_end =
        std::min(bucket_begin + buckets_per_wi, radix_states);

    std::uint64_t wi_count = 0;
    for (std::uint32_t b = bucket_begin; b < bucket_end; ++b) {
        wi_count += hist[b];
    }

    std::uint64_t prefix = sycl::exclusive_scan_over_group(
        wg, wi_count, sycl::plus<std::uint64_t>());

    // exactly one work-item owns the bucket in which the prefix crosses
    // rank, the others contribute zeros to the result
    bucket_info owned{0, 0, 0};
    for (std::uint32_t b = bucket_begin; b < bucket_end; ++b) {
        const std::uint64_t c = hist[b];
        if (prefix < rank && rank <= prefix + c) {
            owned = bucket_info{b, prefix, c};
        }
        prefix += c;
    }

    const bucket_info info{
        sycl::reduce_over_group(wg, owned.bucket_id,
                                sycl::plus<std::uint32_t>()),
        sycl::reduce_over_group(wg, owned.before, sycl::plus<std::uint64_t>()),
        sycl::reduce_over_group(wg, owned.count, sycl::plus<std::uint64_t>())};
    // the bucket is not found only if the histogram counts fewer than rank
    // elements
    assert(info.count > 0);

    // the caller may overwrite the histogram next
    sycl::group_barrier(wg);

    return info;
}

/*! @brief Largest tile size `gather_selected` can rank within 16 bits */
inline constexpr std::size_t max_tile_size = (std::size_t(1) << 16) - 1;

/*! @brief Passes the elements `i` of `[begin, end)` whose prefix under `mask`
 * is less than `desired` to `less_out(j, i)`, the first `n_ties` of the row
 * equal to it to `tie_out(j, i)`, and the others to `greater_out(j, i)`, `j`
 * being their rank in index order among the row's elements of their kind;
 * sinks of type `NoOut` are skipped */
template <std::uint32_t elems_per_wi,
          typename KeyT,
          typename KeyFnT,
          typename LessOutT,
          typename TieOutT,
          typename GreaterOutT = NoOut>
void gather_selected(const sycl::nd_item<1> &ndit,
                     const KeyFnT &key_fn,
                     std::size_t begin,
                     std::size_t end,
                     KeyT desired,
                     KeyT mask,
                     std::uint64_t less_offset,   // rank of first less
                     std::uint64_t n_less,        // less in range
                     std::uint64_t n_ties_before, // ties before begin
                     std::uint64_t n_ties,
                     const LessOutT &less_out,
                     const TieOutT &tie_out,
                     const GreaterOutT &greater_out = {})
{
    static constexpr bool keeps_less = !std::is_same_v<LessOutT, NoOut>;
    static constexpr bool keeps_greater = !std::is_same_v<GreaterOutT, NoOut>;

    // an element's kind and its rank within a tile are counted in the low
    // (less) and high (tie) halves of one integer
    static constexpr std::uint32_t less_flag = 1;
    static constexpr std::uint32_t tie_flag = std::uint32_t(1) << 16;
    static constexpr std::uint32_t half_mask = tie_flag - 1;

    const auto &wg = ndit.get_group();
    const auto &sg = ndit.get_sub_group();
    const std::uint32_t lane_id = sg.get_local_linear_id();
    const std::uint32_t sg_size = sg.get_local_linear_range();
    const std::size_t lid = ndit.get_local_linear_id();
    const std::size_t wg_size = ndit.get_local_range(0);

    // each sub-group processes a contiguous piece of the tile, so that
    // elements are ranked in index order
    const std::size_t tile_size = wg_size * elems_per_wi;
    assert(tile_size <= max_tile_size);
    const std::size_t sg_tile_offset = (lid - lane_id) * elems_per_wi;

    std::uint64_t less_done = 0;
    std::uint64_t ties_done = 0;
    for (std::size_t t0 = begin; t0 < end; t0 += tile_size) {
        if constexpr (!keeps_greater) {
            if ((!keeps_less || less_done >= n_less) &&
                n_ties_before + ties_done >= n_ties) {
                break;
            }
        }

        const std::size_t sg_begin = t0 + sg_tile_offset;

        std::uint32_t flags[elems_per_wi];
        std::uint32_t ranks[elems_per_wi];
        std::uint32_t sg_count = 0;
#pragma unroll
        for (std::uint32_t j = 0; j < elems_per_wi; ++j) {
            const std::size_t i = sg_begin + j * sg_size + lane_id;

            std::uint32_t f = 0;
            if (i < end) {
                const KeyT prefix = key_fn(i) & mask;
                f = (prefix < desired)    ? less_flag
                    : (prefix == desired) ? tie_flag
                                          : 0;
            }
            const std::uint32_t pre = sycl::exclusive_scan_over_group(
                sg, f, sycl::plus<std::uint32_t>());
            flags[j] = f;
            ranks[j] = sg_count + pre;
            sg_count +=
                sycl::reduce_over_group(sg, f, sycl::plus<std::uint32_t>());
        }

        // offset of the sub-group's piece within the tile
        const std::uint32_t contrib = (lane_id == 0) ? sg_count : 0;
        const std::uint32_t sg_offset = sycl::group_broadcast(
            sg, sycl::exclusive_scan_over_group(wg, contrib,
                                                sycl::plus<std::uint32_t>()));
        const std::uint32_t tile_count =
            sycl::reduce_over_group(wg, contrib, sycl::plus<std::uint32_t>());

#pragma unroll
        for (std::uint32_t j = 0; j < elems_per_wi; ++j) {
            const std::size_t i = sg_begin + j * sg_size + lane_id;
            const std::uint32_t r = sg_offset + ranks[j];
            // numbers of the row's less and tie elements before i
            const std::uint64_t less_rank =
                less_offset + less_done + (r & half_mask);
            const std::uint64_t tie_rank =
                n_ties_before + ties_done + (r >> 16);
            if (flags[j] == less_flag) {
                if constexpr (keeps_less) {
                    less_out(less_rank, i);
                }
            }
            else if (flags[j] == tie_flag) {
                if (tie_rank < n_ties) {
                    tie_out(tie_rank, i);
                }
            }
            else if constexpr (keeps_greater) {
                if (i < end) {
                    greater_out(i - less_rank - tie_rank, i);
                }
            }
        }

        less_done += (tile_count & half_mask);
        ties_done += (tile_count >> 16);
    }
}

/*! @brief Largest keys, over the work-group, of the elements in `[begin, end)`
 * whose prefix under `mask` is less than `desired`, and of those for which it
 * equals it; 0 when there are none */
template <typename KeyT, typename KeyFnT>
std::pair<wide_key_t<KeyT>, wide_key_t<KeyT>>
    max_keys_about(const sycl::nd_item<1> &ndit,
                   const KeyFnT &key_fn,
                   std::size_t begin,
                   std::size_t end,
                   KeyT desired,
                   KeyT mask)
{
    using WideKeyT = wide_key_t<KeyT>;

    const std::size_t lid = ndit.get_local_linear_id();
    const std::size_t wg_size = ndit.get_local_range(0);

    WideKeyT less_max = 0;
    WideKeyT eq_max = 0;
    for (std::size_t i = begin + lid; i < end; i += wg_size) {
        const KeyT key = key_fn(i);
        const KeyT prefix = key & mask;
        if (prefix < desired) {
            less_max = std::max(less_max, WideKeyT(key));
        }
        else if (prefix == desired) {
            eq_max = std::max(eq_max, WideKeyT(key));
        }
    }

    const auto &wg = ndit.get_group();
    return {sycl::reduce_over_group(wg, less_max, sycl::maximum<WideKeyT>()),
            sycl::reduce_over_group(wg, eq_max, sycl::maximum<WideKeyT>())};
}

template <typename KeyT, typename ValueT, bool is_ascending>
struct RowKey
{
    const ValueT *arg;

    KeyT operator()(std::size_t i) const
    {
        return radix_utils::ordered_radix_key<is_ascending>(arg[i]);
    }
};

/*! @brief Key of candidate `i` of a row, candidates listed by row index */
template <typename KeyT, typename ValueT, bool is_ascending>
struct CandidateKey
{
    const ValueT *arg;
    const std::uint32_t *cand;

    KeyT operator()(std::size_t i) const
    {
        return radix_utils::ordered_radix_key<is_ascending>(arg[cand[i]]);
    }
};

/*! @brief Outcome of selecting the element of rank `k` of a row */
template <typename KeyT>
struct row_selection
{
    // the elements whose key has the prefix `desired` under `mask`, the
    // sought one among them, number `n_eq`; when the selection resolves the
    // sought element they all compare equal to it, or it is the only one
    KeyT desired;
    KeyT mask;
    // 1-based rank of the sought element, 0 if there is none
    std::uint64_t k;
    // number of elements whose prefix orders before `desired`
    std::uint64_t n_less;
    std::uint64_t n_eq;
    std::uint64_t n_nan;
};

/*! @brief Selects, by the work-group, the element of rank `op.rank(n_nan)`
 * of a row of `n_values` elements */
template <bool is_ascending,
          typename KeyT,
          typename OpT,
          typename KeyFnT,
          typename HistAccT>
row_selection<KeyT> select_over_group(const sycl::nd_item<1> &ndit,
                                      const OpT &op,
                                      const KeyFnT &key_fn,
                                      std::size_t n_values,
                                      const HistAccT &hist)
{
    static constexpr std::uint32_t n_passes = n_radix_passes<KeyT>();
    static constexpr std::uint32_t key_bits =
        radix_utils::number_of_bits_in_type<KeyT>();
    static constexpr bool count_nan = counts_nan<OpT>();
    static constexpr bool stop_on_equal = stops_on_equal<OpT>();
    using WideKeyT = wide_key_t<KeyT>;

    const auto &wg = ndit.get_group();
    const std::size_t lid = ndit.get_local_linear_id();
    const std::size_t wg_size = ndit.get_local_range(0);

    row_selection<KeyT> sel{KeyT{0}, KeyT{0}, 0, 0, 0, 0};
    std::uint64_t k_rem = 0;
    for (std::uint32_t pass = 0; pass < n_passes; ++pass) {
        const std::uint32_t shift = key_bits - (pass + 1) * radix_bits;

        for (std::size_t b = lid; b < radix_states; b += wg_size) {
            hist[b] = 0;
        }
        sycl::group_barrier(wg);

        const digit_stats<KeyT> wi_stats =
            count_digits<count_nan, stop_on_equal>(
                ndit, key_fn, 0, n_values, sel.desired, sel.mask, shift, hist,
                nan_key<is_ascending, OpT>());
        sycl::group_barrier(wg);

        if (pass == 0) {
            if constexpr (count_nan) {
                sel.n_nan =
                    sycl::reduce_over_group(wg, std::uint64_t(wi_stats.n_nan),
                                            sycl::plus<std::uint64_t>());
            }
            sel.k = op.rank(sel.n_nan);
            if (sel.k == 0) {
                return sel;
            }
            k_rem = sel.k;
        }

        const bucket_info info = find_bucket(ndit, hist, k_rem);

        // the keys having the prefix, among which the sought one is, are all
        // equal, so the bucket of this pass resolves it
        bool all_equal = false;
        if constexpr (stop_on_equal) {
            if (info.count == ((pass == 0) ? n_values : sel.n_eq)) {
                all_equal =
                    (sycl::reduce_over_group(wg, wi_stats.key_or,
                                             sycl::bit_or<WideKeyT>()) ==
                     sycl::reduce_over_group(wg, wi_stats.key_and,
                                             sycl::bit_and<WideKeyT>()));
            }
        }
        sel.desired |= static_cast<KeyT>(KeyT(info.bucket_id) << shift);
        sel.mask |= static_cast<KeyT>(KeyT(radix_mask) << shift);
        k_rem -= info.before;
        sel.n_eq = info.count;

        if (all_equal || bucket_resolves<OpT>(info.count, k_rem)) {
            break;
        }
    }
    sel.n_less = sel.k - k_rem;
    return sel;
}

//-----------------------------------------------------------------------
// radix select: one work-group per row
//-----------------------------------------------------------------------

template <typename OpT, bool is_ascending, std::uint32_t elems_per_wi>
class radix_select_one_group_krn;

template <bool is_ascending,
          std::uint32_t elems_per_wi,
          typename ValueT,
          typename OpT>
sycl::event
    radix_select_one_group_submit(sycl::queue &exec_q,
                                  std::size_t n_iters,
                                  std::size_t n_values,
                                  const ValueT *arg_ptr,
                                  const OpT &op,
                                  std::size_t wg_size,
                                  const std::vector<sycl::event> &depends)
{
    using KeyT = radix_utils::radix_key_t<ValueT>;
    using KernelName =
        radix_select_one_group_krn<OpT, is_ascending, elems_per_wi>;

    const kernels_bundle bundle = get_kernels_bundle<KernelName>(exec_q);
    wg_size = std::min(wg_size, bundle.max_wg_size);

    if (n_values > std::numeric_limits<std::uint32_t>::max() ||
        wg_size * elems_per_wi > max_tile_size) {
        throw std::runtime_error("Invalid parameters for radix select");
    }

    return exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.use_kernel_bundle(bundle.kb);

        sycl::local_accessor<std::uint32_t, 1> hist(radix_states, cgh);

        sycl::nd_range<1> ndRange(n_iters * wg_size, wg_size);

        cgh.parallel_for<KernelName>(ndRange, [=](sycl::nd_item<1> ndit) {
            const std::size_t iter_id = ndit.get_group(0);

            const ValueT *row_arg = arg_ptr + iter_id * n_values;
            const RowKey<KeyT, ValueT, is_ascending> key_fn{row_arg};

            const row_selection<KeyT> sel =
                select_over_group<is_ascending, KeyT>(ndit, op, key_fn,
                                                      n_values, hist);

            if constexpr (OpT::output == select_output::median) {
                if (sel.k == 0) {
                    if (ndit.get_local_linear_id() == 0) {
                        op.write_nan(iter_id);
                    }
                    return;
                }
                const auto [less_max, eq_max] = max_keys_about(
                    ndit, key_fn, 0, n_values, sel.desired, sel.mask);
                if (ndit.get_local_linear_id() == 0) {
                    op.write_keys(iter_id, sel.k, sel.n_less, sel.n_nan,
                                  less_max, eq_max);
                }
            }
            else {
                const auto row_out = op.row_out(iter_id, row_arg);
                const std::uint64_t n_ties =
                    (OpT::output == select_output::whole_row)
                        ? sel.n_eq
                        : sel.k - sel.n_less;
                gather_selected<elems_per_wi>(
                    ndit, key_fn, 0, n_values, sel.desired, sel.mask, 0,
                    sel.n_less, 0, n_ties, row_out,
                    shifted(row_out, sel.n_less),
                    greater_out<OpT>(shifted(row_out, sel.n_less + sel.n_eq)));
            }
        });
    });
}

//-----------------------------------------------------------------------
// radix select: one sub-group per row
//-----------------------------------------------------------------------

template <typename OpT, bool is_ascending, std::uint32_t max_chunks>
class radix_select_sub_group_krn;

/*! @brief Selection of a row of `n` elements by the sub-group `sg`, ranking
 * each element by counting from memory the elements ordering before it; for a
 * sub-group too small to hold the row in registers */
template <bool is_ascending, typename ValueT, typename OpT>
void select_sub_group_row_from_memory(const sycl::sub_group &sg,
                                      const OpT &op,
                                      std::size_t iter_id,
                                      const ValueT *row_arg,
                                      std::uint32_t n)
{
    using KeyT = radix_utils::radix_key_t<ValueT>;

    const std::uint32_t lane_id = sg.get_local_linear_id();
    const std::uint32_t sg_size = sg.get_local_linear_range();

    const auto key_at = [=](std::uint32_t i) {
        return radix_utils::ordered_radix_key<is_ascending>(row_arg[i]);
    };
    const auto rank_of = [=](std::uint32_t i, KeyT ki) {
        std::uint32_t rank = 0;
        for (std::uint32_t j = 0; j < n; ++j) {
            const KeyT kj = key_at(j);
            rank += (kj < ki) || (kj == ki && j < i);
        }
        return rank;
    };

    if constexpr (OpT::output == select_output::median) {
        std::uint32_t n_nan = 0;
        if constexpr (counts_nan<OpT>()) {
            const KeyT nan = nan_key<is_ascending, OpT>();
            std::uint32_t wi_nan = 0;
            for (std::uint32_t i = lane_id; i < n; i += sg_size) {
                wi_nan += (key_at(i) == nan);
            }
            n_nan = sycl::reduce_over_group(sg, wi_nan,
                                            sycl::plus<std::uint32_t>());
        }
        const std::uint64_t k = op.rank(n_nan);
        if (k == 0) {
            if (lane_id == 0) {
                op.write_nan(iter_id);
            }
            return;
        }
        // the ranks are distinct, so each median is held by a single lane
        using WideKeyT = wide_key_t<KeyT>;
        const std::uint64_t lo = op.lo_rank(n_nan);
        WideKeyT lo_key = 0;
        WideKeyT hi_key = 0;
        for (std::uint32_t i = lane_id; i < n; i += sg_size) {
            const KeyT ki = key_at(i);
            const std::uint32_t rank = rank_of(i, ki);
            if (rank == lo) {
                lo_key = ki;
            }
            if (rank + 1 == k) {
                hi_key = ki;
            }
        }
        lo_key = sycl::reduce_over_group(sg, lo_key, sycl::maximum<WideKeyT>());
        hi_key = sycl::reduce_over_group(sg, hi_key, sycl::maximum<WideKeyT>());
        if (lane_id == 0) {
            op.write_median_keys(iter_id, lo_key, hi_key);
        }
    }
    else {
        const std::uint64_t n_out =
            (OpT::output == select_output::whole_row) ? n : op.rank(0);
        const auto row_out = op.row_out(iter_id, row_arg);
        for (std::uint32_t i = lane_id; i < n; i += sg_size) {
            const std::uint32_t rank = rank_of(i, key_at(i));
            if (rank < n_out) {
                row_out(rank, i);
            }
        }
    }
}

/*! @brief Selection with a sub-group per row of at most `max_chunks` times the
 * sub-group size, each element is ranked by counting the elements ordering
 * before it and written at its rank, so the selection comes out sorted */
template <bool is_ascending,
          std::uint32_t max_chunks,
          typename ValueT,
          typename OpT>
sycl::event
    radix_select_sub_group_submit(sycl::queue &exec_q,
                                  std::size_t n_iters,
                                  std::size_t n_values,
                                  const ValueT *arg_ptr,
                                  const OpT &op,
                                  std::size_t min_sg_size,
                                  std::size_t wg_size,
                                  const std::vector<sycl::event> &depends)
{
    using KeyT = radix_utils::radix_key_t<ValueT>;
    using KernelName =
        radix_select_sub_group_krn<OpT, is_ascending, max_chunks>;

    const kernels_bundle bundle = get_kernels_bundle<KernelName>(exec_q);
    wg_size = std::min(wg_size, bundle.max_wg_size);

    const std::size_t rows_per_group = std::max<std::size_t>(
        1, wg_size / std::max<std::size_t>(1, min_sg_size));
    const std::size_t n_groups =
        (n_iters + rows_per_group - 1) / rows_per_group;

    return exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.use_kernel_bundle(bundle.kb);

        sycl::nd_range<1> ndRange(n_groups * wg_size, wg_size);

        cgh.parallel_for<KernelName>(ndRange, [=](sycl::nd_item<1> ndit) {
            const auto &sg = ndit.get_sub_group();
            const std::uint32_t lane_id = sg.get_local_linear_id();
            const std::uint32_t sg_size = sg.get_local_linear_range();
            const std::size_t sgs_per_group = sg.get_group_linear_range();
            const std::size_t n_sub_groups =
                ndit.get_group_range(0) * sgs_per_group;
            const std::size_t sg_id =
                ndit.get_group(0) * sgs_per_group + sg.get_group_linear_id();

            const std::uint32_t n = static_cast<std::uint32_t>(n_values);
            const std::uint32_t n_chunks = (n + sg_size - 1) / sg_size;

            // rows are strided over sub-groups, so any number of sub-groups
            // covers all of them
            for (std::size_t iter_id = sg_id; iter_id < n_iters;
                 iter_id += n_sub_groups) {
                const ValueT *row_arg = arg_ptr + iter_id * n_values;

                // a sub-group smaller than the dispatch assumed, such as a
                // partial one, cannot hold the row in registers and ranks it
                // from memory instead
                if (n_chunks > max_chunks) {
                    select_sub_group_row_from_memory<is_ascending>(
                        sg, op, iter_id, row_arg, n);
                    continue;
                }

                // element c * sg_size + lane_id of the row
                KeyT keys[max_chunks];
                std::uint32_t ranks[max_chunks];
#pragma unroll
                for (std::uint32_t c = 0; c < max_chunks; ++c) {
                    const std::uint32_t i = c * sg_size + lane_id;
                    keys[c] =
                        (i < n) ? radix_utils::ordered_radix_key<is_ascending>(
                                      row_arg[i])
                                : KeyT{0};
                    ranks[c] = 0;
                }

                // the chunks are unrolled so that the keys stay in registers,
                // the loops over chunks past the row's end are skipped by the
                // whole sub-group
#pragma unroll
                for (std::uint32_t cj = 0; cj < max_chunks; ++cj) {
                    if (cj >= n_chunks) {
                        break;
                    }
                    const std::uint32_t j0 = cj * sg_size;
                    const std::uint32_t l_end = std::min(sg_size, n - j0);
                    for (std::uint32_t l = 0; l < l_end; ++l) {
                        const KeyT kj = sycl::group_broadcast(sg, keys[cj], l);
                        const std::uint32_t j = j0 + l;
#pragma unroll
                        for (std::uint32_t c = 0; c < max_chunks; ++c) {
                            if (c < n_chunks) {
                                const std::uint32_t i = c * sg_size + lane_id;
                                ranks[c] +=
                                    (kj < keys[c]) || (kj == keys[c] && j < i);
                            }
                        }
                    }
                }

                if constexpr (OpT::output == select_output::median) {
                    std::uint32_t n_nan = 0;
                    if constexpr (counts_nan<OpT>()) {
                        const KeyT nan = nan_key<is_ascending, OpT>();
                        std::uint32_t wi_nan = 0;
#pragma unroll
                        for (std::uint32_t c = 0; c < max_chunks; ++c) {
                            const std::uint32_t i = c * sg_size + lane_id;
                            wi_nan += (i < n && keys[c] == nan);
                        }
                        n_nan = sycl::reduce_over_group(
                            sg, wi_nan, sycl::plus<std::uint32_t>());
                    }
                    const std::uint64_t k = op.rank(n_nan);
                    if (k == 0) {
                        if (lane_id == 0) {
                            op.write_nan(iter_id);
                        }
                        continue;
                    }
                    // NaNs rank last, and the ranks are distinct, so
                    // each median is held by a single lane
                    using WideKeyT = wide_key_t<KeyT>;
                    const std::uint64_t lo = op.lo_rank(n_nan);
                    WideKeyT lo_key = 0;
                    WideKeyT hi_key = 0;
#pragma unroll
                    for (std::uint32_t c = 0; c < max_chunks; ++c) {
                        const std::uint32_t i = c * sg_size + lane_id;
                        if (i < n) {
                            if (ranks[c] == lo) {
                                lo_key = keys[c];
                            }
                            if (ranks[c] + 1 == k) {
                                hi_key = keys[c];
                            }
                        }
                    }
                    lo_key = sycl::reduce_over_group(sg, lo_key,
                                                     sycl::maximum<WideKeyT>());
                    hi_key = sycl::reduce_over_group(sg, hi_key,
                                                     sycl::maximum<WideKeyT>());
                    if (lane_id == 0) {
                        op.write_median_keys(iter_id, lo_key, hi_key);
                    }
                }
                else {
                    const std::uint64_t n_out =
                        (OpT::output == select_output::whole_row) ? n
                                                                  : op.rank(0);
                    const auto row_out = op.row_out(iter_id, row_arg);
#pragma unroll
                    for (std::uint32_t c = 0; c < max_chunks; ++c) {
                        const std::uint32_t i = c * sg_size + lane_id;
                        if (i < n && ranks[c] < n_out) {
                            row_out(ranks[c], i);
                        }
                    }
                }
            }
        });
    });
}

//-----------------------------------------------------------------------
// radix select: several work-groups per row
//-----------------------------------------------------------------------

/*! @brief Selection state of a row, carried from pass to pass */
template <typename KeyT>
struct row_state
{
    KeyT desired;
    KeyT mask;
    // 1-based rank of the sought element, set by the first pass, 0 if there
    // is none
    std::uint64_t k;
    // rank of the sought element among the keys having the prefix
    std::uint64_t k_rem;
    // number of keys having the prefix
    std::uint64_t n_eq;
    std::uint64_t n_nan;
    // when nonzero, the passes after `filter_pass` go over this many
    // candidates, the elements which share the prefix of that pass
    std::uint64_t n_cand;
    // number of elements ordering before the candidates
    std::uint64_t n_less_first;
    std::uint32_t filter_pass;
    std::uint32_t done;
    // number of work-groups of the row which completed the current pass
    std::uint32_t n_arrived;
};

template <typename OpT, bool is_ascending>
class radix_select_init_krn;

template <typename OpT, bool is_ascending>
class radix_select_count_krn;

template <typename OpT, bool is_ascending, std::uint32_t elems_per_wi>
class radix_select_filter_krn;

template <typename OpT, bool is_ascending, std::uint32_t elems_per_wi>
class radix_select_gather_krn;

template <typename OpT, bool is_ascending>
class radix_select_median_segment_krn;

template <typename OpT, bool is_ascending>
class radix_select_median_row_krn;

/*! @brief Range `[begin, end)` of segment `segment_id` of `n_segments`
 * covering `[0, nelems)` */
inline std::pair<std::size_t, std::size_t> segment_range(std::size_t segment_id,
                                                         std::size_t n_segments,
                                                         std::size_t nelems)
{
    const std::size_t elems_per_segment =
        (nelems + n_segments - 1) / n_segments;
    const std::size_t begin = std::min(segment_id * elems_per_segment, nelems);
    return {begin, std::min(begin + elems_per_segment, nelems)};
}

// candidates are listed only when they are at most this fraction of their
// row, a larger buffer costs more to allocate than it saves
inline constexpr std::size_t filter_cap_divisor = 16;

/*! @brief Waits for the event `ev` refers to when destroyed by a thrown
 * exception, so that the kernels it stands for are done with the temporaries
 * freed next */
class wait_on_throw
{
public:
    explicit wait_on_throw(sycl::event &ev)
        : ev_(ev), n_uncaught_(std::uncaught_exceptions())
    {
    }
    wait_on_throw(const wait_on_throw &) = delete;
    wait_on_throw &operator=(const wait_on_throw &) = delete;

    ~wait_on_throw()
    {
        if (std::uncaught_exceptions() > n_uncaught_) {
            try {
                ev_.wait();
            } catch (...) {
            }
        }
    }

private:
    sycl::event &ev_;
    int n_uncaught_;
};

/*! @brief Radix select with `n_segments` work-groups per row and a kernel per
 * pass, the last work-group of a row to finish a pass resolves its digit */
template <bool is_ascending,
          std::uint32_t elems_per_wi,
          typename ValueT,
          typename OpT>
sycl::event
    radix_select_multi_group_impl(sycl::queue &exec_q,
                                  std::size_t n_iters,
                                  std::size_t n_values,
                                  const ValueT *arg_ptr,
                                  const OpT &op,
                                  std::size_t n_segments,
                                  std::size_t wg_size,
                                  bool filter,
                                  const std::vector<sycl::event> &depends)
{
    using KeyT = radix_utils::radix_key_t<ValueT>;
    using WideKeyT = wide_key_t<KeyT>;
    using StateT = row_state<KeyT>;
    using RowKeyT = RowKey<KeyT, ValueT, is_ascending>;
    using CandKeyT = CandidateKey<KeyT, ValueT, is_ascending>;

    static constexpr std::uint32_t n_passes = n_radix_passes<KeyT>();
    static constexpr std::uint32_t key_bits =
        radix_utils::number_of_bits_in_type<KeyT>();
    static constexpr bool count_nan = counts_nan<OpT>();
    static constexpr bool stop_on_equal = stops_on_equal<OpT>();
    static constexpr bool is_median = (OpT::output == select_output::median);

    using CountKernelName = radix_select_count_krn<OpT, is_ascending>;
    using FilterKernelName =
        radix_select_filter_krn<OpT, is_ascending, elems_per_wi>;
    using GatherKernelName =
        radix_select_gather_krn<OpT, is_ascending, elems_per_wi>;
    using MedianSegmentKernelName =
        radix_select_median_segment_krn<OpT, is_ascending>;
    // the kernel ending the selection over the rows' segments
    using LastKernelName =
        std::conditional_t<is_median, MedianSegmentKernelName,
                           GatherKernelName>;

    // the kernels share the work-groups of the rows' segments
    const kernels_bundle bundle =
        get_kernels_bundle<CountKernelName, FilterKernelName, LastKernelName>(
            exec_q);
    wg_size = std::min(wg_size, bundle.max_wg_size);

    const std::size_t elems_per_segment =
        (n_values + n_segments - 1) / n_segments;
    if (elems_per_segment > std::numeric_limits<std::uint32_t>::max() ||
        n_segments > std::numeric_limits<std::uint32_t>::max() ||
        wg_size * elems_per_wi > max_tile_size) {
        throw std::runtime_error("Invalid parameters for radix select");
    }
    // when filtering, the elements sharing the prefix of the first pass to
    // leave few enough of them are listed, by 32-bit indices within their
    // row, for the later passes to go over
    filter = filter && (n_passes > 1) &&
             (n_values <= std::numeric_limits<std::uint32_t>::max());
    // rows with more candidates than this are not filtered
    const std::size_t cand_cap =
        (n_values + filter_cap_divisor - 1) / filter_cap_divisor;

    const std::size_t n_all_segments = n_iters * n_segments;

    auto state_owner =
        dpnp::tensor::alloc_utils::smart_malloc_device<StateT>(n_iters, exec_q);
    StateT *state_ptr = state_owner.get();

    // per work-group digit histograms of the current pass
    auto hist_owner =
        dpnp::tensor::alloc_utils::smart_malloc_device<std::uint32_t>(
            n_all_segments * radix_states, exec_q);
    std::uint32_t *segment_hist_ptr = hist_owner.get();

    // per work-group counts of less and tie elements, and their exclusive
    // scans over the work-groups of a row, followed by the per work-group
    // counts of NaNs if they are counted, and the bitwise or and and of the
    // keys having the prefix if equal keys stop the selection
    static constexpr std::size_t n_counts =
        4 + (count_nan ? 1 : 0) + (stop_on_equal ? 2 : 0);
    auto counts_owner =
        dpnp::tensor::alloc_utils::smart_malloc_device<std::uint64_t>(
            n_counts * n_all_segments, exec_q);
    std::uint64_t *less_count_ptr = counts_owner.get();
    std::uint64_t *less_offset_ptr = less_count_ptr + n_all_segments;
    std::uint64_t *tie_count_ptr = less_offset_ptr + n_all_segments;
    std::uint64_t *tie_offset_ptr = tie_count_ptr + n_all_segments;
    std::uint64_t *nan_count_ptr = tie_offset_ptr + n_all_segments;
    std::uint64_t *key_or_ptr =
        (stop_on_equal) ? nan_count_ptr + (count_nan ? n_all_segments : 0)
                        : nullptr;
    std::uint64_t *key_and_ptr =
        (stop_on_equal) ? key_or_ptr + n_all_segments : nullptr;

    // candidates of each row, in index order
    auto cand_owner =
        dpnp::tensor::alloc_utils::smart_malloc_device<std::uint32_t>(
            (filter) ? n_iters * cand_cap : 1, exec_q);
    std::uint32_t *cand_ptr = cand_owner.get();

    // the largest keys of the elements ordering before the upper median and
    // of those equal to it, for each work-group
    auto partial_owner =
        dpnp::tensor::alloc_utils::smart_malloc_device<WideKeyT>(
            (is_median) ? 2 * n_all_segments : 1, exec_q);
    WideKeyT *partial_ptr = partial_owner.get();

    // the last kernel submitted, each one depends on the one before; if a
    // submission throws, it is waited for before the temporaries are freed
    sycl::event last_ev{};
    const wait_on_throw last_ev_guard(last_ev);

    last_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using KernelName = radix_select_init_krn<OpT, is_ascending>;
        cgh.parallel_for<KernelName>(
            sycl::range<1>(n_iters), [=](sycl::id<1> id) {
                state_ptr[id[0]] =
                    StateT{KeyT{0}, KeyT{0}, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            });
    });

    const sycl::nd_range<1> ndRange(n_all_segments * wg_size, wg_size);

    for (std::uint32_t pass = 0; pass < n_passes; ++pass) {
        const std::uint32_t shift = key_bits - (pass + 1) * radix_bits;
        const bool last_pass = (pass + 1 == n_passes);
        // the last pass leaves nothing to filter for
        const bool filter_after = filter && !last_pass;

        last_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(last_ev);
            cgh.use_kernel_bundle(bundle.kb);

            sycl::local_accessor<std::uint32_t, 1> hist(radix_states, cgh);
            sycl::local_accessor<std::uint64_t, 1> row_hist(radix_states, cgh);
            sycl::local_accessor<std::uint32_t, 1> is_last(1, cgh);

            using KernelName = CountKernelName;
            cgh.parallel_for<KernelName>(ndRange, [=](sycl::nd_item<1> ndit) {
                const auto &wg = ndit.get_group();
                const std::size_t group_id = ndit.get_group(0);
                const std::size_t iter_id = group_id / n_segments;
                const std::size_t segment_id = group_id - iter_id * n_segments;
                const std::size_t lid = ndit.get_local_linear_id();

                StateT &state = state_ptr[iter_id];
                if (state.done) {
                    return;
                }
                const KeyT desired = state.desired;
                const KeyT mask = state.mask;
                const std::uint64_t n_cand = state.n_cand;

                for (std::size_t b = lid; b < radix_states; b += wg_size) {
                    hist[b] = 0;
                }
                sycl::group_barrier(wg);

                const ValueT *row_arg = arg_ptr + iter_id * n_values;
                digit_stats<KeyT> wi_stats;
                if (n_cand) {
                    const auto [begin, end] =
                        segment_range(segment_id, n_segments, n_cand);
                    wi_stats = count_digits<false, stop_on_equal>(
                        ndit, CandKeyT{row_arg, cand_ptr + iter_id * cand_cap},
                        begin, end, desired, mask, shift, hist);
                }
                else {
                    const auto [begin, end] =
                        segment_range(segment_id, n_segments, n_values);
                    wi_stats = count_digits<count_nan, stop_on_equal>(
                        ndit, RowKeyT{row_arg}, begin, end, desired, mask,
                        shift, hist, nan_key<is_ascending, OpT>());
                }
                if constexpr (count_nan) {
                    // the first pass goes over the row
                    if (pass == 0) {
                        const std::uint64_t n_nan = sycl::reduce_over_group(
                            wg, std::uint64_t(wi_stats.n_nan),
                            sycl::plus<std::uint64_t>());
                        if (lid == 0) {
                            nan_count_ptr[group_id] = n_nan;
                        }
                    }
                }
                if constexpr (stop_on_equal) {
                    const WideKeyT key_or = sycl::reduce_over_group(
                        wg, wi_stats.key_or, sycl::bit_or<WideKeyT>());
                    const WideKeyT key_and = sycl::reduce_over_group(
                        wg, wi_stats.key_and, sycl::bit_and<WideKeyT>());
                    if (lid == 0) {
                        key_or_ptr[group_id] = key_or;
                        key_and_ptr[group_id] = key_and;
                    }
                }
                sycl::group_barrier(wg);

                std::uint32_t *row_segment_hist =
                    segment_hist_ptr + iter_id * n_segments * radix_states;
                for (std::size_t b = lid; b < radix_states; b += wg_size) {
                    row_segment_hist[segment_id * radix_states + b] = hist[b];
                }

                // publish the histogram, then count this work-group in
                sycl::atomic_fence(sycl::memory_order::release,
                                   sycl::memory_scope::device);
                sycl::group_barrier(wg, sycl::memory_scope::device);
                if (lid == 0) {
                    sycl::atomic_ref<std::uint32_t, sycl::memory_order::acq_rel,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        n_arrived(state.n_arrived);
                    is_last[0] = (n_arrived.fetch_add(std::uint32_t(1)) + 1 ==
                                  n_segments);
                }
                sycl::group_barrier(wg, sycl::memory_scope::device);
                if (!is_last[0]) {
                    return;
                }
                sycl::atomic_fence(sycl::memory_order::acquire,
                                   sycl::memory_scope::device);

                // the last work-group of the row resolves this pass' digit;
                // the state is read ahead of the barriers below, as work-item
                // 0 updates it after them
                std::uint64_t k = state.k;
                std::uint64_t n_nan = state.n_nan;
                std::uint64_t k_rem = state.k_rem;
                const std::uint64_t n_eq = state.n_eq;
                const std::uint32_t filter_pass = state.filter_pass;
                if (pass == 0) {
                    if constexpr (count_nan) {
                        const std::uint64_t *row_nan_count =
                            nan_count_ptr + iter_id * n_segments;
                        n_nan = sycl::joint_reduce(
                            wg, row_nan_count, row_nan_count + n_segments,
                            std::uint64_t(0), sycl::plus<std::uint64_t>());
                    }
                    k = op.rank(n_nan);
                    k_rem = k;
                    if (k == 0) {
                        if (lid == 0) {
                            state.n_nan = n_nan;
                            state.done = 1;
                            state.n_arrived = 0;
                        }
                        return;
                    }
                }

                for (std::size_t b = lid; b < radix_states; b += wg_size) {
                    std::uint64_t s = 0;
                    for (std::size_t j = 0; j < n_segments; ++j) {
                        s += row_segment_hist[j * radix_states + b];
                    }
                    row_hist[b] = s;
                }
                sycl::group_barrier(wg);

                const bucket_info info = find_bucket(ndit, row_hist, k_rem);
                const std::uint64_t new_k_rem = k_rem - info.before;

                // the keys having the prefix, among which the sought one is,
                // are all equal, so the bucket of this pass resolves it
                bool all_equal = false;
                if constexpr (stop_on_equal) {
                    if (info.count == ((pass == 0) ? n_values : n_eq)) {
                        const std::uint64_t *row_key_or =
                            key_or_ptr + iter_id * n_segments;
                        const std::uint64_t *row_key_and =
                            key_and_ptr + iter_id * n_segments;
                        all_equal =
                            (sycl::joint_reduce(
                                 wg, row_key_or, row_key_or + n_segments,
                                 std::uint64_t(0),
                                 sycl::bit_or<std::uint64_t>()) ==
                             sycl::joint_reduce(
                                 wg, row_key_and, row_key_and + n_segments,
                                 ~std::uint64_t(0),
                                 sycl::bit_and<std::uint64_t>()));
                    }
                }
                const bool done = last_pass || all_equal ||
                                  bucket_resolves<OpT>(info.count, new_k_rem);
                // the elements of this pass' bucket become the candidates
                const bool to_filter = filter_after && !done && (n_cand == 0) &&
                                       (info.count <= cand_cap);

                // the counts restart when the segments switch to candidates
                const bool restart =
                    (pass == 0) || (n_cand && pass == filter_pass + 1);
                std::uint64_t *row_less_count =
                    less_count_ptr + iter_id * n_segments;
                std::uint64_t *row_less_offset =
                    less_offset_ptr + iter_id * n_segments;
                std::uint64_t *row_tie_count =
                    tie_count_ptr + iter_id * n_segments;
                std::uint64_t *row_tie_offset =
                    tie_offset_ptr + iter_id * n_segments;
                for (std::size_t j = lid; j < n_segments; j += wg_size) {
                    const std::uint32_t *h =
                        row_segment_hist + j * radix_states;
                    std::uint64_t s = 0;
                    for (std::uint32_t b = 0; b < info.bucket_id; ++b) {
                        s += h[b];
                    }
                    row_less_count[j] = ((restart) ? 0 : row_less_count[j]) + s;
                    if (done || to_filter) {
                        row_tie_count[j] = h[info.bucket_id];
                    }
                }

                if (done || to_filter) {
                    sycl::group_barrier(wg);
                    sycl::joint_exclusive_scan(
                        wg, row_less_count, row_less_count + n_segments,
                        row_less_offset, std::uint64_t(0),
                        sycl::plus<std::uint64_t>());
                    sycl::joint_exclusive_scan(wg, row_tie_count,
                                               row_tie_count + n_segments,
                                               row_tie_offset, std::uint64_t(0),
                                               sycl::plus<std::uint64_t>());
                }

                if (lid == 0) {
                    state.desired =
                        desired |
                        static_cast<KeyT>(KeyT(info.bucket_id) << shift);
                    state.mask =
                        mask | static_cast<KeyT>(KeyT(radix_mask) << shift);
                    state.k = k;
                    state.k_rem = new_k_rem;
                    state.n_eq = info.count;
                    state.n_nan = n_nan;
                    state.done = done;
                    state.n_arrived = 0;
                    if (to_filter) {
                        state.n_cand = info.count;
                        state.n_less_first = k - new_k_rem;
                        state.filter_pass = pass;
                    }
                }
            });
        });

        if (filter_after) {
            // writes out the elements resolved by the rows filtered after this
            // pass and lists their candidates
            last_ev = exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(last_ev);
                cgh.use_kernel_bundle(bundle.kb);

                cgh.parallel_for<FilterKernelName>(
                    ndRange, [=](sycl::nd_item<1> ndit) {
                        const std::size_t group_id = ndit.get_group(0);
                        const std::size_t iter_id = group_id / n_segments;
                        const std::size_t segment_id =
                            group_id - iter_id * n_segments;

                        const StateT state = state_ptr[iter_id];
                        if (state.n_cand == 0 || state.filter_pass != pass) {
                            return;
                        }

                        const ValueT *row_arg = arg_ptr + iter_id * n_values;
                        const auto row_out = op.row_out(iter_id, row_arg);
                        const auto [begin, end] =
                            segment_range(segment_id, n_segments, n_values);
                        gather_selected<elems_per_wi>(
                            ndit, RowKeyT{row_arg}, begin, end, state.desired,
                            state.mask, less_offset_ptr[group_id],
                            less_count_ptr[group_id], tie_offset_ptr[group_id],
                            state.n_cand, row_out,
                            CandidateListOut{cand_ptr + iter_id * cand_cap},
                            greater_out<OpT>(shifted(
                                row_out, state.n_less_first + state.n_cand)));
                    });
            });
        }
    }

    if constexpr (is_median) {
        last_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(last_ev);
            cgh.use_kernel_bundle(bundle.kb);

            cgh.parallel_for<MedianSegmentKernelName>(
                ndRange, [=](sycl::nd_item<1> ndit) {
                    const std::size_t group_id = ndit.get_group(0);
                    const std::size_t iter_id = group_id / n_segments;
                    const std::size_t segment_id =
                        group_id - iter_id * n_segments;

                    const StateT state = state_ptr[iter_id];
                    if (state.k == 0) {
                        return;
                    }

                    // the elements ordering before the candidates are gone over
                    // only if the lower median is among them
                    const std::uint64_t n_less = state.k - state.k_rem;
                    const bool in_cand =
                        state.n_cand &&
                        (n_less > state.n_less_first ||
                         !op.lo_before(state.k, n_less, state.n_nan));

                    const ValueT *row_arg = arg_ptr + iter_id * n_values;
                    std::pair<WideKeyT, WideKeyT> maxima;
                    if (in_cand) {
                        const auto [begin, end] =
                            segment_range(segment_id, n_segments, state.n_cand);
                        maxima = max_keys_about(
                            ndit,
                            CandKeyT{row_arg, cand_ptr + iter_id * cand_cap},
                            begin, end, state.desired, state.mask);
                    }
                    else {
                        const auto [begin, end] =
                            segment_range(segment_id, n_segments, n_values);
                        maxima = max_keys_about(ndit, RowKeyT{row_arg}, begin,
                                                end, state.desired, state.mask);
                    }
                    if (ndit.get_local_linear_id() == 0) {
                        partial_ptr[2 * group_id] = maxima.first;
                        partial_ptr[2 * group_id + 1] = maxima.second;
                    }
                });
        });

        last_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(last_ev);

            using KernelName = radix_select_median_row_krn<OpT, is_ascending>;
            cgh.parallel_for<KernelName>(
                sycl::range<1>(n_iters), [=](sycl::id<1> id) {
                    const std::size_t iter_id = id[0];
                    const StateT state = state_ptr[iter_id];
                    if (state.k == 0) {
                        op.write_nan(iter_id);
                        return;
                    }

                    const WideKeyT *row_partial =
                        partial_ptr + 2 * iter_id * n_segments;
                    WideKeyT less_max = 0;
                    WideKeyT eq_max = 0;
                    for (std::size_t j = 0; j < n_segments; ++j) {
                        less_max = std::max(less_max, row_partial[2 * j]);
                        eq_max = std::max(eq_max, row_partial[2 * j + 1]);
                    }
                    op.write_keys(iter_id, state.k, state.k - state.k_rem,
                                  state.n_nan, less_max, eq_max);
                });
        });
    }
    else {
        last_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(last_ev);
            cgh.use_kernel_bundle(bundle.kb);

            cgh.parallel_for<GatherKernelName>(
                ndRange, [=](sycl::nd_item<1> ndit) {
                    const std::size_t group_id = ndit.get_group(0);
                    const std::size_t iter_id = group_id / n_segments;
                    const std::size_t segment_id =
                        group_id - iter_id * n_segments;

                    const StateT state = state_ptr[iter_id];

                    const ValueT *row_arg = arg_ptr + iter_id * n_values;
                    // the filter wrote out the elements before the candidates
                    // already
                    const auto row_out = shifted(op.row_out(iter_id, row_arg),
                                                 state.n_less_first);
                    const std::uint64_t n_less =
                        state.k - state.n_less_first - state.k_rem;
                    const std::uint64_t n_ties =
                        (OpT::output == select_output::whole_row) ? state.n_eq
                                                                  : state.k_rem;
                    const auto tie_out = shifted(row_out, n_less);
                    const auto greater =
                        greater_out<OpT>(shifted(row_out, n_less + state.n_eq));

                    if (state.n_cand) {
                        const std::uint32_t *row_cand =
                            cand_ptr + iter_id * cand_cap;
                        const auto [begin, end] =
                            segment_range(segment_id, n_segments, state.n_cand);
                        gather_selected<elems_per_wi>(
                            ndit, CandKeyT{row_arg, row_cand}, begin, end,
                            state.desired, state.mask,
                            less_offset_ptr[group_id], less_count_ptr[group_id],
                            tie_offset_ptr[group_id], n_ties,
                            via_candidates(row_cand, row_out),
                            via_candidates(row_cand, tie_out),
                            via_candidates(row_cand, greater));
                    }
                    else {
                        const auto [begin, end] =
                            segment_range(segment_id, n_segments, n_values);
                        gather_selected<elems_per_wi>(
                            ndit, RowKeyT{row_arg}, begin, end, state.desired,
                            state.mask, less_offset_ptr[group_id],
                            less_count_ptr[group_id], tie_offset_ptr[group_id],
                            n_ties, row_out, tie_out, greater);
                    }
                });
        });
    }

    return dpnp::tensor::alloc_utils::async_smart_free(
        exec_q, {last_ev}, state_owner, hist_owner, counts_owner, cand_owner,
        partial_owner);
}

//-----------------------------------------------------------------------
// radix select: main function
//-----------------------------------------------------------------------

inline constexpr std::uint32_t gather_elems_per_wi = 4;

// longest rows ranked by a sub-group each, and the most sub-group sizes they
// may span; the keys are held in registers, so shorter rows get a kernel with
// fewer of them for better occupancy
inline constexpr std::size_t sub_group_max_n = 64;
inline constexpr std::uint32_t sub_group_max_chunks = 8;
inline constexpr std::uint32_t sub_group_few_chunks = 4;

template <bool is_ascending, typename ValueT, typename OpT>
sycl::event radix_select_dispatch(sycl::queue &exec_q,
                                  std::size_t n_iters,
                                  std::size_t n_values,
                                  const ValueT *arg_ptr,
                                  const OpT &op,
                                  const std::vector<sycl::event> &depends)
{
    const auto &dev = exec_q.get_device();
    const std::size_t max_wg_size =
        dev.get_info<sycl::info::device::max_work_group_size>();
    const std::size_t n_cus =
        dev.get_info<sycl::info::device::max_compute_units>();

    const std::size_t wg_size = std::min<std::size_t>(256, max_wg_size);

    // short rows spend their time scheduling work-groups, so they are ranked
    // by a sub-group each; the kernel may be compiled for any of the device's
    // sub-group sizes, so the smallest one bounds the rows it takes
    const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    const std::size_t min_sg_size =
        (sg_sizes.empty())
            ? 1
            : *std::min_element(sg_sizes.begin(), sg_sizes.end());
    if (n_values <=
            std::min(sub_group_max_n, sub_group_max_chunks * min_sg_size) &&
        min_sg_size <= wg_size) {
        if (n_values <= sub_group_few_chunks * min_sg_size) {
            return radix_select_sub_group_submit<is_ascending,
                                                 sub_group_few_chunks>(
                exec_q, n_iters, n_values, arg_ptr, op, min_sg_size, wg_size,
                depends);
        }
        return radix_select_sub_group_submit<is_ascending,
                                             sub_group_max_chunks>(
            exec_q, n_iters, n_values, arg_ptr, op, min_sg_size, wg_size,
            depends);
    }

    // enough work-groups to occupy the device
    const std::size_t target_groups = 4 * n_cus;
    const std::size_t min_segment_size = 8 * wg_size * gather_elems_per_wi;
    // below this row size a kernel per pass costs more than it saves
    static constexpr std::size_t multi_group_min_n = std::size_t(1) << 16;

    std::size_t n_segments = 1;
    if (n_iters < target_groups && n_values >= multi_group_min_n) {
        n_segments =
            std::min((target_groups + n_iters - 1) / n_iters,
                     (n_values + min_segment_size - 1) / min_segment_size);
    }
    // counts of a work-group's elements are 32-bit
    static constexpr std::size_t max_segment_size =
        std::numeric_limits<std::uint32_t>::max();
    n_segments = std::max(n_segments,
                          (n_values + max_segment_size - 1) / max_segment_size);

    if (n_segments > 1) {
        using KeyT = radix_utils::radix_key_t<ValueT>;
        // with fewer passes re-reading the rows costs less than listing the
        // candidates
        static constexpr bool filter = (sizeof(KeyT) >= 4);
        return radix_select_multi_group_impl<is_ascending, gather_elems_per_wi>(
            exec_q, n_iters, n_values, arg_ptr, op, n_segments, wg_size, filter,
            depends);
    }
    // short rows leave most of a large work-group idle
    std::size_t row_wg_size = 64;
    while (row_wg_size < wg_size && row_wg_size * 8 < n_values) {
        row_wg_size *= 2;
    }
    row_wg_size = std::min(row_wg_size, wg_size);
    return radix_select_one_group_submit<is_ascending, gather_elems_per_wi>(
        exec_q, n_iters, n_values, arg_ptr, op, row_wg_size, depends);
}

/*! @brief Writes the `k` smallest (largest if not `is_ascending`) elements of
 * each row of C-contiguous `(n_iters, n_values)` array and their indices into
 * `(n_iters, k)` arrays, in unspecified order; ties are resolved in favor of
 * smaller indices, NaNs order last, and -0.0 and +0.0 compare equal */
template <typename ValueT, typename IndexT>
sycl::event radix_select_impl(sycl::queue &exec_q,
                              std::size_t n_iters,
                              std::size_t n_values,
                              std::size_t k,
                              bool is_ascending,
                              const ValueT *arg_ptr,
                              ValueT *vals_ptr,
                              IndexT *inds_ptr,
                              const std::vector<sycl::event> &depends)
{
    if (k == 0 || k > n_values) {
        throw std::runtime_error("Invalid value of k for radix select");
    }

    const TopKOp<ValueT, IndexT> op{vals_ptr, inds_ptr, k};
    if (is_ascending) {
        return radix_select_dispatch</*is_ascending*/ true>(
            exec_q, n_iters, n_values, arg_ptr, op, depends);
    }
    return radix_select_dispatch</*is_ascending*/ false>(
        exec_q, n_iters, n_values, arg_ptr, op, depends);
}

/*! @brief Writes each row of C-contiguous `(n_iters, n_values)` array into
 * `(n_iters, n_values)` array rearranged so that its element at `kth` is the
 * one a sort would put there, preceded by those ordering before it and
 * followed by the others; each of these three groups keeps the order of the
 * row, NaNs order last, and -0.0 and +0.0 compare equal */
template <typename ValueT>
sycl::event radix_partition_impl(sycl::queue &exec_q,
                                 std::size_t n_iters,
                                 std::size_t n_values,
                                 std::size_t kth,
                                 const ValueT *arg_ptr,
                                 ValueT *vals_ptr,
                                 const std::vector<sycl::event> &depends)
{
    if (kth >= n_values) {
        throw std::runtime_error("Invalid value of kth for radix partition");
    }

    const PartitionOp<ValueT> op{vals_ptr, n_values, kth + 1};
    return radix_select_dispatch</*is_ascending*/ true>(
        exec_q, n_iters, n_values, arg_ptr, op, depends);
}

/*! @brief Writes the indices by which `radix_partition_impl` rearranges each
 * row of C-contiguous `(n_iters, n_values)` array into `(n_iters, n_values)`
 * array */
template <typename ValueT, typename IndexT>
sycl::event radix_argpartition_impl(sycl::queue &exec_q,
                                    std::size_t n_iters,
                                    std::size_t n_values,
                                    std::size_t kth,
                                    const ValueT *arg_ptr,
                                    IndexT *inds_ptr,
                                    const std::vector<sycl::event> &depends)
{
    if (kth >= n_values) {
        throw std::runtime_error("Invalid value of kth for radix partition");
    }

    const ArgPartitionOp<ValueT, IndexT> op{inds_ptr, n_values, kth + 1};
    return radix_select_dispatch</*is_ascending*/ true>(
        exec_q, n_iters, n_values, arg_ptr, op, depends);
}

/*! @brief Writes the median of each row of C-contiguous `(n_iters,
 * n_values)` array, as `ResT`, into `(n_iters,)` array; it is NaN for a row
 * with NaNs, or with NaNs only if `ignore_nan`, which then sets `*all_nan_ptr`
 * unless it is null */
template <typename ValueT, typename ResT>
sycl::event radix_median_impl(sycl::queue &exec_q,
                              std::size_t n_iters,
                              std::size_t n_values,
                              bool ignore_nan,
                              const ValueT *arg_ptr,
                              ResT *out_ptr,
                              std::uint32_t *all_nan_ptr,
                              const std::vector<sycl::event> &depends)
{
    if (n_values == 0) {
        throw std::runtime_error("Median of an empty row is undefined");
    }

    const MedianOp<ValueT, ResT> op{out_ptr, all_nan_ptr, n_values, ignore_nan};
    return radix_select_dispatch</*is_ascending*/ true>(
        exec_q, n_iters, n_values, arg_ptr, op, depends);
}

} // namespace dpnp::tensor::kernels::radix_select_details
