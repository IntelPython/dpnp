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
/// This file defines kernels selecting the k smallest elements of each row of
/// a C-contiguous array with an MSD radix select.
///
/// Each pass histograms one 8-bit digit of the keys which share the prefix
/// resolved so far and picks the bucket holding the k-th smallest key. Once
/// the prefix of the k-th key is known, the elements with a smaller prefix
/// and the first elements (in index order) sharing it are gathered. Rows are
/// processed either by a single work-group running all passes in one kernel,
/// or, for few long rows, by several work-groups per row with one kernel per
/// pass: the last work-group of a row to finish a pass reduces the
/// work-groups' histograms and publishes the state for the next pass.
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
    std::uint32_t bin;
    // number of elements in buckets preceding `bin`
    std::uint64_t before;
    // number of elements in `bin`
    std::uint64_t count;
};

//-----------------------------------------------------------------------
// work-group level building blocks
//-----------------------------------------------------------------------

/*! @brief Adds to the local histogram `hist` the digits at `shift` of the
 * keys in `[begin, end)` whose prefix selected by `mask` is `desired`.
 *
 * `hist` must be zeroed by the caller; its work-group sees the complete counts
 * only after a barrier.
 */
template <typename KeyT, typename KeyFnT, typename HistAccT>
void count_digits(const sycl::nd_item<1> &ndit,
                  const KeyFnT &key_fn,
                  std::size_t begin,
                  std::size_t end,
                  KeyT desired,
                  KeyT mask,
                  std::uint32_t shift,
                  const HistAccT &hist)
{
    using AtomicT = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space>;

    const auto &sg = ndit.get_sub_group();
    const std::uint32_t sg_size = sg.get_local_linear_range();
    const std::size_t lid = ndit.get_local_linear_id();
    const std::size_t wg_size = ndit.get_local_range(0);

    for (std::size_t i0 = begin; i0 < end; i0 += wg_size) {
        const std::size_t i = i0 + lid;

        bool match = false;
        std::uint32_t bin = 0;
        if (i < end) {
            const KeyT key = key_fn(i);
            match = ((key & mask) == desired);
            bin = radix_utils::get_bucket_id<radix_mask>(key, shift);
        }

        // a sub-group whose keys all land in one bucket, as for runs of
        // equal values, makes one update instead of contending sg_size times
        const std::uint32_t leader_bin = sycl::group_broadcast(sg, bin);
        if (sycl::all_of_group(sg, match && (bin == leader_bin))) {
            if (sg.leader()) {
                AtomicT(hist[bin]).fetch_add(sg_size);
            }
        }
        else if (match) {
            AtomicT(hist[bin]).fetch_add(std::uint32_t(1));
        }
    }
}

/*! @brief Finds the bucket of the histogram `hist` which holds the element
 * of (1-based) rank `rank`.
 *
 * `hist` must be complete in all work-items. `result` is local scratch
 * memory for a single `bucket_info`.
 */
template <typename HistAccT, typename ResultAccT>
bucket_info find_bucket(const sycl::nd_item<1> &ndit,
                        const HistAccT &hist,
                        std::uint64_t rank,
                        const ResultAccT &result)
{
    const std::uint32_t lid = ndit.get_local_linear_id();
    const std::uint32_t wg_size = ndit.get_local_range(0);

    const std::uint32_t bins_per_wi = (radix_states + wg_size - 1) / wg_size;
    const std::uint32_t bin_begin = std::min(lid * bins_per_wi, radix_states);
    const std::uint32_t bin_end =
        std::min(bin_begin + bins_per_wi, radix_states);

    std::uint64_t wi_count = 0;
    for (std::uint32_t b = bin_begin; b < bin_end; ++b) {
        wi_count += hist[b];
    }

    std::uint64_t prefix = sycl::exclusive_scan_over_group(
        ndit.get_group(), wi_count, sycl::plus<std::uint64_t>());

    // exactly one work-item owns the bucket in which the prefix crosses rank
    for (std::uint32_t b = bin_begin; b < bin_end; ++b) {
        const std::uint64_t c = hist[b];
        if (prefix < rank && rank <= prefix + c) {
            result[0] = bucket_info{b, prefix, c};
        }
        prefix += c;
    }

    sycl::group_barrier(ndit.get_group());

    return result[0];
}

/*! @brief Passes the selected elements `i` of `[begin, end)` to
 * `less_out(j, i)` and `tie_out(j, i)`.
 *
 * An element is selected when its prefix selected by `mask` is smaller than
 * `desired` ("less"), or equal to it ("tie") and fewer than `n_ties` ties
 * precede it in the row. `j` is `less_offset` plus the rank of a less element
 * in index order among the less elements of `[begin, end)`, or the rank of a
 * tie among the ties of the row, so ties are resolved the way a stable sort
 * resolves them.
 *
 * @param n_less   number of less elements in `[begin, end)`
 * @param n_ties_before  number of ties in the row preceding `begin`
 */
template <std::uint32_t elems_per_wi,
          typename KeyT,
          typename KeyFnT,
          typename LessOutT,
          typename TieOutT>
void gather_selected(const sycl::nd_item<1> &ndit,
                     const KeyFnT &key_fn,
                     std::size_t begin,
                     std::size_t end,
                     KeyT desired,
                     KeyT mask,
                     std::uint64_t less_offset,
                     std::uint64_t n_less,
                     std::uint64_t n_ties_before,
                     std::uint64_t n_ties,
                     const LessOutT &less_out,
                     const TieOutT &tie_out)
{
    // an element's kind and its rank within a tile are counted in the low
    // (less) and high (tie) halves of one integer
    static constexpr std::uint32_t less_flag = 1;
    static constexpr std::uint32_t tie_flag = std::uint32_t(1) << 16;
    static constexpr std::uint32_t half_mask = tie_flag - 1;

    const auto &wg = ndit.get_group();
    const auto &sg = ndit.get_sub_group();
    const std::uint32_t lane = sg.get_local_linear_id();
    const std::uint32_t sg_size = sg.get_local_linear_range();
    const std::size_t lid = ndit.get_local_linear_id();
    const std::size_t wg_size = ndit.get_local_range(0);

    // each sub-group processes a contiguous piece of the tile, so that
    // elements are ranked in index order
    const std::size_t tile_size = wg_size * elems_per_wi;
    const std::size_t sg_tile_offset = (lid - lane) * elems_per_wi;

    std::uint64_t less_done = 0;
    std::uint64_t ties_done = 0;
    for (std::size_t t0 = begin; t0 < end; t0 += tile_size) {
        if (less_done >= n_less && n_ties_before + ties_done >= n_ties) {
            break;
        }

        const std::size_t sg_begin = t0 + sg_tile_offset;

        std::uint32_t flags[elems_per_wi];
        std::uint32_t ranks[elems_per_wi];
        std::uint32_t sg_count = 0;
#pragma unroll
        for (std::uint32_t j = 0; j < elems_per_wi; ++j) {
            const std::size_t i = sg_begin + j * sg_size + lane;

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
        const std::uint32_t contrib = (lane == 0) ? sg_count : 0;
        const std::uint32_t sg_offset = sycl::group_broadcast(
            sg, sycl::exclusive_scan_over_group(wg, contrib,
                                                sycl::plus<std::uint32_t>()));
        const std::uint32_t tile_count =
            sycl::reduce_over_group(wg, contrib, sycl::plus<std::uint32_t>());

#pragma unroll
        for (std::uint32_t j = 0; j < elems_per_wi; ++j) {
            if (flags[j] == 0) {
                continue;
            }
            const std::size_t i = sg_begin + j * sg_size + lane;
            const std::uint32_t r = sg_offset + ranks[j];
            if (flags[j] == less_flag) {
                less_out(less_offset + less_done + (r & half_mask), i);
            }
            else {
                const std::uint64_t tie_rank =
                    n_ties_before + ties_done + (r >> 16);
                if (tie_rank < n_ties) {
                    tie_out(tie_rank, i);
                }
            }
        }

        less_done += (tile_count & half_mask);
        ties_done += (tile_count >> 16);
    }
}

template <typename KeyT, typename ValueT, bool is_ascending>
struct RowKey
{
    const ValueT *row;

    KeyT operator()(std::size_t i) const
    {
        return radix_utils::ordered_radix_key<is_ascending>(row[i]);
    }
};

/*! @brief Key of the `i`-th of the candidates of a row, listed by their
 * indices within the row */
template <typename KeyT, typename ValueT, bool is_ascending>
struct CandidateKey
{
    const ValueT *row;
    const std::uint32_t *cand;

    KeyT operator()(std::size_t i) const
    {
        return radix_utils::ordered_radix_key<is_ascending>(row[cand[i]]);
    }
};

/*! @brief Stores the value and the index of element `i` of a row as the
 * `j`-th selected element of the row */
template <typename ValueT, typename IndexT>
struct SelectedOut
{
    const ValueT *row;
    ValueT *vals;
    IndexT *inds;

    void operator()(std::uint64_t j, std::size_t i) const
    {
        vals[j] = row[i];
        inds[j] = static_cast<IndexT>(i);
    }
};

/*! @brief Stores the value and the index of candidate `i` of a row as the
 * `j`-th selected element of the row */
template <typename ValueT, typename IndexT>
struct CandidateSelectedOut
{
    const ValueT *row;
    const std::uint32_t *cand;
    ValueT *vals;
    IndexT *inds;

    void operator()(std::uint64_t j, std::size_t i) const
    {
        const std::uint32_t c = cand[i];
        vals[j] = row[c];
        inds[j] = static_cast<IndexT>(c);
    }
};

/*! @brief Lists element `i` of a row as its `j`-th candidate */
struct CandidateListOut
{
    std::uint32_t *cand;

    void operator()(std::uint64_t j, std::size_t i) const
    {
        cand[j] = static_cast<std::uint32_t>(i);
    }
};

//-----------------------------------------------------------------------
// one work-group per row
//-----------------------------------------------------------------------

template <typename ValueT,
          typename IndexT,
          bool is_ascending,
          std::uint32_t elems_per_wi>
class radix_select_one_group_krn;

/*! @brief Largest tile size `gather_selected` can rank within 16 bits */
inline constexpr std::size_t max_tile_size = (std::size_t(1) << 16) - 1;

template <bool is_ascending,
          std::uint32_t elems_per_wi,
          typename ValueT,
          typename IndexT>
sycl::event
    radix_select_one_group_submit(sycl::queue &exec_q,
                                  std::size_t n_iters,
                                  std::size_t n,
                                  std::size_t k,
                                  const ValueT *vals_ptr,
                                  ValueT *dst_vals_ptr,
                                  IndexT *dst_inds_ptr,
                                  std::size_t wg_size,
                                  const std::vector<sycl::event> &depends)
{
    using KeyT = radix_utils::radix_key_t<ValueT>;
    using KernelName =
        radix_select_one_group_krn<ValueT, IndexT, is_ascending, elems_per_wi>;

    static constexpr std::uint32_t n_passes = n_radix_passes<KeyT>();
    static constexpr std::uint32_t key_bits =
        radix_utils::number_of_bits_in_type<KeyT>();

    if (n > std::numeric_limits<std::uint32_t>::max() ||
        wg_size * elems_per_wi > max_tile_size) {
        throw std::runtime_error("Invalid parameters for radix select");
    }

    return exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        sycl::local_accessor<std::uint32_t, 1> hist(radix_states, cgh);
        sycl::local_accessor<bucket_info, 1> bucket(1, cgh);

        sycl::nd_range<1> ndRange(n_iters * wg_size, wg_size);

        cgh.parallel_for<KernelName>(ndRange, [=](sycl::nd_item<1> ndit) {
            const std::size_t row = ndit.get_group(0);
            const std::size_t lid = ndit.get_local_linear_id();

            const RowKey<KeyT, ValueT, is_ascending> key_fn{vals_ptr + row * n};

            KeyT desired{0};
            KeyT mask{0};
            std::uint64_t k_rem = k;
            for (std::uint32_t pass = 0; pass < n_passes; ++pass) {
                const std::uint32_t shift = key_bits - (pass + 1) * radix_bits;

                for (std::size_t b = lid; b < radix_states; b += wg_size) {
                    hist[b] = 0;
                }
                sycl::group_barrier(ndit.get_group());

                count_digits(ndit, key_fn, 0, n, desired, mask, shift, hist);
                sycl::group_barrier(ndit.get_group());

                const bucket_info info = find_bucket(ndit, hist, k_rem, bucket);

                desired |= static_cast<KeyT>(KeyT(info.bin) << shift);
                mask |= static_cast<KeyT>(KeyT(radix_mask) << shift);
                k_rem -= info.before;

                // all keys with the prefix are selected, the rest of the
                // digits does not matter
                if (info.count == k_rem) {
                    break;
                }
            }

            const std::uint64_t n_less = k - k_rem;
            using OutT = SelectedOut<ValueT, IndexT>;
            const ValueT *row_vals = vals_ptr + row * n;
            ValueT *row_dst_vals = dst_vals_ptr + row * k;
            IndexT *row_dst_inds = dst_inds_ptr + row * k;
            gather_selected<elems_per_wi>(
                ndit, key_fn, 0, n, desired, mask, 0, n_less, 0, k_rem,
                OutT{row_vals, row_dst_vals, row_dst_inds},
                OutT{row_vals, row_dst_vals + n_less, row_dst_inds + n_less});
        });
    });
}

//-----------------------------------------------------------------------
// several work-groups per row
//-----------------------------------------------------------------------

/*! @brief Selection state of a row, carried from pass to pass */
template <typename KeyT>
struct row_state
{
    KeyT desired;
    KeyT mask;
    // rank of the sought element among the keys having the prefix
    std::uint64_t k_rem;
    // when nonzero, the passes after the first one go over this many
    // candidates, the elements which share the prefix of the first pass
    std::uint64_t n_cand;
    // number of elements selected by the first pass
    std::uint64_t n_less_first;
    std::uint32_t done;
    // number of work-groups of the row which completed the current pass
    std::uint32_t n_arrived;
};

template <typename ValueT, typename IndexT, bool is_ascending>
class radix_select_init_krn;

template <typename ValueT, typename IndexT, bool is_ascending>
class radix_select_count_krn;

template <typename ValueT,
          typename IndexT,
          bool is_ascending,
          std::uint32_t elems_per_wi>
class radix_select_filter_krn;

template <typename ValueT,
          typename IndexT,
          bool is_ascending,
          std::uint32_t elems_per_wi>
class radix_select_gather_krn;

/*! @brief Range `[begin, end)` of block `blk` of `n_blocks` covering `[0, n)`
 */
inline std::pair<std::size_t, std::size_t>
    block_range(std::size_t blk, std::size_t n_blocks, std::size_t n)
{
    const std::size_t block_size = (n + n_blocks - 1) / n_blocks;
    const std::size_t begin = std::min(blk * block_size, n);
    return {begin, std::min(begin + block_size, n)};
}

/*! @brief Candidates are listed only when they are at most this fraction of
 * their row, since a large buffer costs more to allocate than it saves */
inline constexpr std::size_t filter_cap_divisor = 16;

/*! @brief Radix select with `n_blocks` work-groups per row and a kernel per
 * pass.
 *
 * When `filter`, the elements sharing the prefix of the first pass, if few,
 * are listed so that the later passes only go over them.
 */
template <bool is_ascending,
          std::uint32_t elems_per_wi,
          typename ValueT,
          typename IndexT>
sycl::event
    radix_select_multi_group_impl(sycl::queue &exec_q,
                                  std::size_t n_iters,
                                  std::size_t n,
                                  std::size_t k,
                                  const ValueT *vals_ptr,
                                  ValueT *dst_vals_ptr,
                                  IndexT *dst_inds_ptr,
                                  std::size_t n_blocks,
                                  std::size_t wg_size,
                                  bool filter,
                                  const std::vector<sycl::event> &depends)
{
    using KeyT = radix_utils::radix_key_t<ValueT>;
    using StateT = row_state<KeyT>;
    using RowKeyT = RowKey<KeyT, ValueT, is_ascending>;
    using CandKeyT = CandidateKey<KeyT, ValueT, is_ascending>;

    static constexpr std::uint32_t n_passes = n_radix_passes<KeyT>();
    static constexpr std::uint32_t key_bits =
        radix_utils::number_of_bits_in_type<KeyT>();

    const std::size_t block_size = (n + n_blocks - 1) / n_blocks;
    if (block_size > std::numeric_limits<std::uint32_t>::max() ||
        n_blocks > std::numeric_limits<std::uint32_t>::max() ||
        wg_size * elems_per_wi > max_tile_size) {
        throw std::runtime_error("Invalid parameters for radix select");
    }
    // candidates are listed by 32-bit indices within their row
    filter = filter && (n_passes > 1) &&
             (n <= std::numeric_limits<std::uint32_t>::max());
    // rows with more candidates than this are not filtered
    const std::size_t cand_cap =
        (n + filter_cap_divisor - 1) / filter_cap_divisor;

    const std::size_t n_row_blocks = n_iters * n_blocks;

    auto state_owner =
        dpnp::tensor::alloc_utils::smart_malloc_device<StateT>(n_iters, exec_q);
    StateT *state_ptr = state_owner.get();

    // per work-group digit histograms of the current pass
    auto hist_owner =
        dpnp::tensor::alloc_utils::smart_malloc_device<std::uint32_t>(
            n_row_blocks * radix_states, exec_q);
    std::uint32_t *block_hist_ptr = hist_owner.get();

    // per work-group counts of less and tie elements, and their exclusive
    // scans over the work-groups of a row
    auto counts_owner =
        dpnp::tensor::alloc_utils::smart_malloc_device<std::uint64_t>(
            4 * n_row_blocks, exec_q);
    std::uint64_t *less_count_ptr = counts_owner.get();
    std::uint64_t *less_offset_ptr = less_count_ptr + n_row_blocks;
    std::uint64_t *tie_count_ptr = less_offset_ptr + n_row_blocks;
    std::uint64_t *tie_offset_ptr = tie_count_ptr + n_row_blocks;

    // candidates of each row, in index order
    auto cand_owner =
        dpnp::tensor::alloc_utils::smart_malloc_device<std::uint32_t>(
            (filter) ? n_iters * cand_cap : 1, exec_q);
    std::uint32_t *cand_ptr = cand_owner.get();

    sycl::event init_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using KernelName = radix_select_init_krn<ValueT, IndexT, is_ascending>;
        cgh.parallel_for<KernelName>(
            sycl::range<1>(n_iters), [=](sycl::id<1> id) {
                state_ptr[id[0]] = StateT{KeyT{0}, KeyT{0}, k, 0, 0, 0, 0};
            });
    });

    const sycl::nd_range<1> ndRange(n_row_blocks * wg_size, wg_size);

    sycl::event pass_ev = init_ev;
    for (std::uint32_t pass = 0; pass < n_passes; ++pass) {
        const std::uint32_t shift = key_bits - (pass + 1) * radix_bits;
        const bool last_pass = (pass + 1 == n_passes);
        const bool filter_after = filter && (pass == 0);

        pass_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(pass_ev);

            sycl::local_accessor<std::uint32_t, 1> hist(radix_states, cgh);
            sycl::local_accessor<std::uint64_t, 1> row_hist(radix_states, cgh);
            sycl::local_accessor<bucket_info, 1> bucket(1, cgh);
            sycl::local_accessor<std::uint32_t, 1> is_last(1, cgh);

            using KernelName =
                radix_select_count_krn<ValueT, IndexT, is_ascending>;
            cgh.parallel_for<KernelName>(ndRange, [=](sycl::nd_item<1> ndit) {
                const auto &wg = ndit.get_group();
                const std::size_t group_id = ndit.get_group(0);
                const std::size_t row = group_id / n_blocks;
                const std::size_t blk = group_id - row * n_blocks;
                const std::size_t lid = ndit.get_local_linear_id();

                StateT &st = state_ptr[row];
                if (st.done) {
                    return;
                }
                const KeyT desired = st.desired;
                const KeyT mask = st.mask;
                const std::uint64_t n_cand = st.n_cand;

                for (std::size_t b = lid; b < radix_states; b += wg_size) {
                    hist[b] = 0;
                }
                sycl::group_barrier(wg);

                const ValueT *row_vals = vals_ptr + row * n;
                if (n_cand) {
                    const auto [begin, end] =
                        block_range(blk, n_blocks, n_cand);
                    count_digits(ndit,
                                 CandKeyT{row_vals, cand_ptr + row * cand_cap},
                                 begin, end, desired, mask, shift, hist);
                }
                else {
                    const auto [begin, end] = block_range(blk, n_blocks, n);
                    count_digits(ndit, RowKeyT{row_vals}, begin, end, desired,
                                 mask, shift, hist);
                }
                sycl::group_barrier(wg);

                std::uint32_t *row_block_hist =
                    block_hist_ptr + row * n_blocks * radix_states;
                for (std::size_t b = lid; b < radix_states; b += wg_size) {
                    row_block_hist[blk * radix_states + b] = hist[b];
                }

                // publish the histogram, then count this work-group in
                sycl::atomic_fence(sycl::memory_order::release,
                                   sycl::memory_scope::device);
                sycl::group_barrier(wg, sycl::memory_scope::device);
                if (lid == 0) {
                    sycl::atomic_ref<std::uint32_t, sycl::memory_order::acq_rel,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        n_arrived(st.n_arrived);
                    is_last[0] =
                        (n_arrived.fetch_add(std::uint32_t(1)) + 1 == n_blocks);
                }
                sycl::group_barrier(wg, sycl::memory_scope::device);
                if (!is_last[0]) {
                    return;
                }
                sycl::atomic_fence(sycl::memory_order::acquire,
                                   sycl::memory_scope::device);

                // the last work-group of the row resolves this pass' digit
                for (std::size_t b = lid; b < radix_states; b += wg_size) {
                    std::uint64_t s = 0;
                    for (std::size_t j = 0; j < n_blocks; ++j) {
                        s += row_block_hist[j * radix_states + b];
                    }
                    row_hist[b] = s;
                }
                sycl::group_barrier(wg);

                const std::uint64_t k_rem = st.k_rem;
                const bucket_info info =
                    find_bucket(ndit, row_hist, k_rem, bucket);
                const std::uint64_t new_k_rem = k_rem - info.before;
                const bool done = last_pass || (info.count == new_k_rem);
                // the elements of this pass' bucket become the candidates
                const bool to_filter =
                    filter_after && !done && (info.count <= cand_cap);

                // the counts restart when the blocks switch to candidates
                const bool restart = (pass == 0) || (n_cand && pass == 1);
                std::uint64_t *row_less_count = less_count_ptr + row * n_blocks;
                std::uint64_t *row_less_offset =
                    less_offset_ptr + row * n_blocks;
                std::uint64_t *row_tie_count = tie_count_ptr + row * n_blocks;
                std::uint64_t *row_tie_offset = tie_offset_ptr + row * n_blocks;
                for (std::size_t j = lid; j < n_blocks; j += wg_size) {
                    const std::uint32_t *h = row_block_hist + j * radix_states;
                    std::uint64_t s = 0;
                    for (std::uint32_t b = 0; b < info.bin; ++b) {
                        s += h[b];
                    }
                    row_less_count[j] = ((restart) ? 0 : row_less_count[j]) + s;
                    if (done || to_filter) {
                        row_tie_count[j] = h[info.bin];
                    }
                }

                if (done || to_filter) {
                    sycl::group_barrier(wg);
                    sycl::joint_exclusive_scan(
                        wg, row_less_count, row_less_count + n_blocks,
                        row_less_offset, std::uint64_t(0),
                        sycl::plus<std::uint64_t>());
                    sycl::joint_exclusive_scan(wg, row_tie_count,
                                               row_tie_count + n_blocks,
                                               row_tie_offset, std::uint64_t(0),
                                               sycl::plus<std::uint64_t>());
                }

                if (lid == 0) {
                    st.desired =
                        desired | static_cast<KeyT>(KeyT(info.bin) << shift);
                    st.mask =
                        mask | static_cast<KeyT>(KeyT(radix_mask) << shift);
                    st.k_rem = new_k_rem;
                    st.done = done;
                    st.n_arrived = 0;
                    if (to_filter) {
                        st.n_cand = info.count;
                        st.n_less_first = info.before;
                    }
                }
            });
        });

        if (filter_after) {
            // writes out the less elements of the first pass and lists its
            // candidates
            pass_ev = exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(pass_ev);

                using KernelName =
                    radix_select_filter_krn<ValueT, IndexT, is_ascending,
                                            elems_per_wi>;
                cgh.parallel_for<KernelName>(
                    ndRange, [=](sycl::nd_item<1> ndit) {
                        const std::size_t group_id = ndit.get_group(0);
                        const std::size_t row = group_id / n_blocks;
                        const std::size_t blk = group_id - row * n_blocks;

                        const StateT st = state_ptr[row];
                        if (st.n_cand == 0) {
                            return;
                        }

                        const auto [begin, end] = block_range(blk, n_blocks, n);
                        gather_selected<elems_per_wi>(
                            ndit, RowKeyT{vals_ptr + row * n}, begin, end,
                            st.desired, st.mask, less_offset_ptr[group_id],
                            less_count_ptr[group_id], tie_offset_ptr[group_id],
                            st.n_cand,
                            SelectedOut<ValueT, IndexT>{vals_ptr + row * n,
                                                        dst_vals_ptr + row * k,
                                                        dst_inds_ptr + row * k},
                            CandidateListOut{cand_ptr + row * cand_cap});
                    });
            });
        }
    }

    sycl::event gather_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(pass_ev);

        using KernelName =
            radix_select_gather_krn<ValueT, IndexT, is_ascending, elems_per_wi>;
        cgh.parallel_for<KernelName>(ndRange, [=](sycl::nd_item<1> ndit) {
            const std::size_t group_id = ndit.get_group(0);
            const std::size_t row = group_id / n_blocks;
            const std::size_t blk = group_id - row * n_blocks;

            const StateT st = state_ptr[row];

            const ValueT *row_vals = vals_ptr + row * n;
            // the first pass wrote out its less elements already
            ValueT *row_dst_vals = dst_vals_ptr + row * k + st.n_less_first;
            IndexT *row_dst_inds = dst_inds_ptr + row * k + st.n_less_first;
            const std::uint64_t n_less = k - st.n_less_first - st.k_rem;

            if (st.n_cand) {
                using OutT = CandidateSelectedOut<ValueT, IndexT>;
                const std::uint32_t *row_cand = cand_ptr + row * cand_cap;
                const auto [begin, end] = block_range(blk, n_blocks, st.n_cand);
                gather_selected<elems_per_wi>(
                    ndit, CandKeyT{row_vals, row_cand}, begin, end, st.desired,
                    st.mask, less_offset_ptr[group_id],
                    less_count_ptr[group_id], tie_offset_ptr[group_id],
                    st.k_rem,
                    OutT{row_vals, row_cand, row_dst_vals, row_dst_inds},
                    OutT{row_vals, row_cand, row_dst_vals + n_less,
                         row_dst_inds + n_less});
            }
            else {
                using OutT = SelectedOut<ValueT, IndexT>;
                const auto [begin, end] = block_range(blk, n_blocks, n);
                gather_selected<elems_per_wi>(
                    ndit, RowKeyT{row_vals}, begin, end, st.desired, st.mask,
                    less_offset_ptr[group_id], less_count_ptr[group_id],
                    tie_offset_ptr[group_id], st.k_rem,
                    OutT{row_vals, row_dst_vals, row_dst_inds},
                    OutT{row_vals, row_dst_vals + n_less,
                         row_dst_inds + n_less});
            }
        });
    });

    return dpnp::tensor::alloc_utils::async_smart_free(
        exec_q, {gather_ev}, state_owner, hist_owner, counts_owner, cand_owner);
}

//-----------------------------------------------------------------------
// radix select: main function
//-----------------------------------------------------------------------

inline constexpr std::uint32_t gather_elems_per_wi = 4;

template <bool is_ascending, typename ValueT, typename IndexT>
sycl::event radix_select_dispatch(sycl::queue &exec_q,
                                  std::size_t n_iters,
                                  std::size_t n,
                                  std::size_t k,
                                  const ValueT *vals_ptr,
                                  ValueT *dst_vals_ptr,
                                  IndexT *dst_inds_ptr,
                                  const std::vector<sycl::event> &depends)
{
    const auto &dev = exec_q.get_device();
    const std::size_t max_wg_size =
        dev.get_info<sycl::info::device::max_work_group_size>();
    const std::size_t n_cus =
        dev.get_info<sycl::info::device::max_compute_units>();

    static constexpr std::uint32_t epw = gather_elems_per_wi;

    const std::size_t wg_size = std::min<std::size_t>(256, max_wg_size);

    // enough work-groups to occupy the device
    const std::size_t target_groups = 4 * n_cus;
    const std::size_t min_block_size = 8 * wg_size * epw;
    // below this row size a kernel per pass costs more than it saves
    static constexpr std::size_t multi_group_min_n = std::size_t(1) << 16;

    std::size_t n_blocks = 1;
    if (n_iters < target_groups && n >= multi_group_min_n) {
        n_blocks = std::min((target_groups + n_iters - 1) / n_iters,
                            (n + min_block_size - 1) / min_block_size);
    }
    // counts of a work-group's elements are 32-bit
    static constexpr std::size_t max_block_size =
        std::numeric_limits<std::uint32_t>::max();
    n_blocks = std::max(n_blocks, (n + max_block_size - 1) / max_block_size);

    if (n_blocks > 1) {
        using KeyT = radix_utils::radix_key_t<ValueT>;
        // with fewer passes re-reading the rows costs less than listing the
        // candidates
        static constexpr bool filter = (sizeof(KeyT) >= 4);
        return radix_select_multi_group_impl<is_ascending, epw>(
            exec_q, n_iters, n, k, vals_ptr, dst_vals_ptr, dst_inds_ptr,
            n_blocks, wg_size, filter, depends);
    }
    // short rows leave most of a large work-group idle
    std::size_t row_wg_size = 64;
    while (row_wg_size < wg_size && row_wg_size * 8 < n) {
        row_wg_size *= 2;
    }
    row_wg_size = std::min(row_wg_size, wg_size);
    return radix_select_one_group_submit<is_ascending, epw>(
        exec_q, n_iters, n, k, vals_ptr, dst_vals_ptr, dst_inds_ptr,
        row_wg_size, depends);
}

/*! @brief Writes the `k` smallest (largest when `is_ascending` is false)
 * elements of each row of the C-contiguous `(n_iters, n)` array `vals_ptr`
 * into the rows of the `(n_iters, k)` array `dst_vals_ptr`, and their indices
 * within the row into those of `dst_inds_ptr`.
 *
 * Ties are resolved in favor of smaller indices, NaNs order after any other
 * value, and -0.0 and +0.0 compare equal. The selection is not sorted and
 * its order is unspecified.
 */
template <typename ValueT, typename IndexT>
sycl::event radix_select_impl(sycl::queue &exec_q,
                              std::size_t n_iters,
                              std::size_t n,
                              std::size_t k,
                              bool is_ascending,
                              const ValueT *vals_ptr,
                              ValueT *dst_vals_ptr,
                              IndexT *dst_inds_ptr,
                              const std::vector<sycl::event> &depends)
{
    if (k == 0 || k > n) {
        throw std::runtime_error("Invalid value of k for radix select");
    }

    if (is_ascending) {
        return radix_select_dispatch</*is_ascending*/ true>(
            exec_q, n_iters, n, k, vals_ptr, dst_vals_ptr, dst_inds_ptr,
            depends);
    }
    return radix_select_dispatch</*is_ascending*/ false>(
        exec_q, n_iters, n, k, vals_ptr, dst_vals_ptr, dst_inds_ptr, depends);
}

} // namespace dpnp::tensor::kernels::radix_select_details
