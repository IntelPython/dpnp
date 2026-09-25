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
/// This file defines MSD radix select kernels for tensor topk operation.
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
    std::uint32_t bucket_id;
    // number of elements in buckets preceding `bucket_id`
    std::uint64_t before;
    // number of elements in `bucket_id`
    std::uint64_t count;
};

//-----------------------------------------------------------------------
// radix select: work-group level building blocks
//-----------------------------------------------------------------------

/*! @brief Adds to the zeroed local histogram `hist` the digits at `shift` of
 * the keys in `[begin, end)` whose prefix under `mask` is `desired` */
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
        std::uint32_t bucket_id = 0;
        if (i < end) {
            const KeyT key = key_fn(i);
            match = ((key & mask) == desired);
            bucket_id = radix_utils::get_bucket_id<radix_mask>(key, shift);
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
}

/*! @brief Finds the bucket of the complete local histogram `hist` holding
 * the element of 1-based rank `rank` */
template <typename HistAccT, typename ResultAccT>
bucket_info find_bucket(const sycl::nd_item<1> &ndit,
                        const HistAccT &hist,
                        std::uint64_t rank,
                        const ResultAccT &result)
{
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
        ndit.get_group(), wi_count, sycl::plus<std::uint64_t>());

    // exactly one work-item owns the bucket in which the prefix crosses rank
    for (std::uint32_t b = bucket_begin; b < bucket_end; ++b) {
        const std::uint64_t c = hist[b];
        if (prefix < rank && rank <= prefix + c) {
            result[0] = bucket_info{b, prefix, c};
        }
        prefix += c;
    }

    sycl::group_barrier(ndit.get_group());

    return result[0];
}

/*! @brief Passes the elements `i` of `[begin, end)` whose prefix under `mask`
 * is less than `desired` to `less_out(j, i)`, and the first `n_ties` of the
 * row equal to it to `tie_out(j, i)`, `j` being their rank in index order */
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
                     std::uint64_t less_offset,   // rank of first less
                     std::uint64_t n_less,        // less in range
                     std::uint64_t n_ties_before, // ties before begin
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
    const std::uint32_t lane_id = sg.get_local_linear_id();
    const std::uint32_t sg_size = sg.get_local_linear_range();
    const std::size_t lid = ndit.get_local_linear_id();
    const std::size_t wg_size = ndit.get_local_range(0);

    // each sub-group processes a contiguous piece of the tile, so that
    // elements are ranked in index order
    const std::size_t tile_size = wg_size * elems_per_wi;
    const std::size_t sg_tile_offset = (lid - lane_id) * elems_per_wi;

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
            if (flags[j] == 0) {
                continue;
            }
            const std::size_t i = sg_begin + j * sg_size + lane_id;
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

/*! @brief Stores the value and the index of candidate `i` of a row as the
 * `j`-th selected element of the row */
template <typename ValueT, typename IndexT>
struct CandidateSelectedOut
{
    const ValueT *arg;
    const std::uint32_t *cand;
    ValueT *vals;
    IndexT *inds;

    void operator()(std::uint64_t j, std::size_t i) const
    {
        const std::uint32_t c = cand[i];
        vals[j] = arg[c];
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
// radix select: one work-group per row
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
                                  std::size_t n_values,
                                  std::size_t k,
                                  const ValueT *arg_ptr,
                                  ValueT *vals_ptr,
                                  IndexT *inds_ptr,
                                  std::size_t wg_size,
                                  const std::vector<sycl::event> &depends)
{
    using KeyT = radix_utils::radix_key_t<ValueT>;
    using KernelName =
        radix_select_one_group_krn<ValueT, IndexT, is_ascending, elems_per_wi>;

    static constexpr std::uint32_t n_passes = n_radix_passes<KeyT>();
    static constexpr std::uint32_t key_bits =
        radix_utils::number_of_bits_in_type<KeyT>();

    if (n_values > std::numeric_limits<std::uint32_t>::max() ||
        wg_size * elems_per_wi > max_tile_size) {
        throw std::runtime_error("Invalid parameters for radix select");
    }

    return exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        sycl::local_accessor<std::uint32_t, 1> hist(radix_states, cgh);
        sycl::local_accessor<bucket_info, 1> bucket(1, cgh);

        sycl::nd_range<1> ndRange(n_iters * wg_size, wg_size);

        cgh.parallel_for<KernelName>(ndRange, [=](sycl::nd_item<1> ndit) {
            const std::size_t iter_id = ndit.get_group(0);
            const std::size_t lid = ndit.get_local_linear_id();

            const RowKey<KeyT, ValueT, is_ascending> key_fn{arg_ptr +
                                                            iter_id * n_values};

            KeyT desired{0};
            KeyT mask{0};
            std::uint64_t k_rem = k;
            for (std::uint32_t pass = 0; pass < n_passes; ++pass) {
                const std::uint32_t shift = key_bits - (pass + 1) * radix_bits;

                for (std::size_t b = lid; b < radix_states; b += wg_size) {
                    hist[b] = 0;
                }
                sycl::group_barrier(ndit.get_group());

                count_digits(ndit, key_fn, 0, n_values, desired, mask, shift,
                             hist);
                sycl::group_barrier(ndit.get_group());

                const bucket_info info = find_bucket(ndit, hist, k_rem, bucket);

                desired |= static_cast<KeyT>(KeyT(info.bucket_id) << shift);
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
            const ValueT *row_arg = arg_ptr + iter_id * n_values;
            ValueT *row_vals = vals_ptr + iter_id * k;
            IndexT *row_inds = inds_ptr + iter_id * k;
            gather_selected<elems_per_wi>(
                ndit, key_fn, 0, n_values, desired, mask, 0, n_less, 0, k_rem,
                OutT{row_arg, row_vals, row_inds},
                OutT{row_arg, row_vals + n_less, row_inds + n_less});
        });
    });
}

//-----------------------------------------------------------------------
// radix select: one sub-group per row
//-----------------------------------------------------------------------

template <typename ValueT,
          typename IndexT,
          bool is_ascending,
          std::uint32_t max_chunks>
class radix_select_sub_group_krn;

/*! @brief Selection with a sub-group per row of at most `max_chunks` times the
 * sub-group size, each element is ranked by counting the elements ordering
 * before it and written at its rank, so the selection comes out sorted */
template <bool is_ascending,
          std::uint32_t max_chunks,
          typename ValueT,
          typename IndexT>
sycl::event
    radix_select_sub_group_submit(sycl::queue &exec_q,
                                  std::size_t n_iters,
                                  std::size_t n_values,
                                  std::size_t k,
                                  const ValueT *arg_ptr,
                                  ValueT *vals_ptr,
                                  IndexT *inds_ptr,
                                  std::size_t n_groups,
                                  std::size_t wg_size,
                                  const std::vector<sycl::event> &depends)
{
    using KeyT = radix_utils::radix_key_t<ValueT>;
    using KernelName =
        radix_select_sub_group_krn<ValueT, IndexT, is_ascending, max_chunks>;

    return exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

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

            // rows are strided over sub-groups, so any sub-group size of at
            // least n_values / max_chunks covers all of them
            for (std::size_t iter_id = sg_id; iter_id < n_iters;
                 iter_id += n_sub_groups) {
                const ValueT *row_arg = arg_ptr + iter_id * n_values;

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

                ValueT *row_vals = vals_ptr + iter_id * k;
                IndexT *row_inds = inds_ptr + iter_id * k;
#pragma unroll
                for (std::uint32_t c = 0; c < max_chunks; ++c) {
                    const std::uint32_t i = c * sg_size + lane_id;
                    if (i < n && ranks[c] < k) {
                        row_vals[ranks[c]] = row_arg[i];
                        row_inds[ranks[c]] = static_cast<IndexT>(i);
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

/*! @brief Radix select with `n_segments` work-groups per row and a kernel per
 * pass, the last work-group of a row to finish a pass resolves its digit */
template <bool is_ascending,
          std::uint32_t elems_per_wi,
          typename ValueT,
          typename IndexT>
sycl::event
    radix_select_multi_group_impl(sycl::queue &exec_q,
                                  std::size_t n_iters,
                                  std::size_t n_values,
                                  std::size_t k,
                                  const ValueT *arg_ptr,
                                  ValueT *vals_ptr,
                                  IndexT *inds_ptr,
                                  std::size_t n_segments,
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

    const std::size_t elems_per_segment =
        (n_values + n_segments - 1) / n_segments;
    if (elems_per_segment > std::numeric_limits<std::uint32_t>::max() ||
        n_segments > std::numeric_limits<std::uint32_t>::max() ||
        wg_size * elems_per_wi > max_tile_size) {
        throw std::runtime_error("Invalid parameters for radix select");
    }
    // when filtering, the elements sharing the prefix of the first pass are
    // listed, by 32-bit indices within their row, for the later passes to go
    // over
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
    // scans over the work-groups of a row
    auto counts_owner =
        dpnp::tensor::alloc_utils::smart_malloc_device<std::uint64_t>(
            4 * n_all_segments, exec_q);
    std::uint64_t *less_count_ptr = counts_owner.get();
    std::uint64_t *less_offset_ptr = less_count_ptr + n_all_segments;
    std::uint64_t *tie_count_ptr = less_offset_ptr + n_all_segments;
    std::uint64_t *tie_offset_ptr = tie_count_ptr + n_all_segments;

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

    const sycl::nd_range<1> ndRange(n_all_segments * wg_size, wg_size);

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
                if (n_cand) {
                    const auto [begin, end] =
                        segment_range(segment_id, n_segments, n_cand);
                    count_digits(
                        ndit, CandKeyT{row_arg, cand_ptr + iter_id * cand_cap},
                        begin, end, desired, mask, shift, hist);
                }
                else {
                    const auto [begin, end] =
                        segment_range(segment_id, n_segments, n_values);
                    count_digits(ndit, RowKeyT{row_arg}, begin, end, desired,
                                 mask, shift, hist);
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

                // the last work-group of the row resolves this pass' digit
                for (std::size_t b = lid; b < radix_states; b += wg_size) {
                    std::uint64_t s = 0;
                    for (std::size_t j = 0; j < n_segments; ++j) {
                        s += row_segment_hist[j * radix_states + b];
                    }
                    row_hist[b] = s;
                }
                sycl::group_barrier(wg);

                const std::uint64_t k_rem = state.k_rem;
                const bucket_info info =
                    find_bucket(ndit, row_hist, k_rem, bucket);
                const std::uint64_t new_k_rem = k_rem - info.before;
                const bool done = last_pass || (info.count == new_k_rem);
                // the elements of this pass' bucket become the candidates
                const bool to_filter =
                    filter_after && !done && (info.count <= cand_cap);

                // the counts restart when the segments switch to candidates
                const bool restart = (pass == 0) || (n_cand && pass == 1);
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
                    state.k_rem = new_k_rem;
                    state.done = done;
                    state.n_arrived = 0;
                    if (to_filter) {
                        state.n_cand = info.count;
                        state.n_less_first = info.before;
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
                        const std::size_t iter_id = group_id / n_segments;
                        const std::size_t segment_id =
                            group_id - iter_id * n_segments;

                        const StateT state = state_ptr[iter_id];
                        if (state.n_cand == 0) {
                            return;
                        }

                        const auto [begin, end] =
                            segment_range(segment_id, n_segments, n_values);
                        gather_selected<elems_per_wi>(
                            ndit, RowKeyT{arg_ptr + iter_id * n_values}, begin,
                            end, state.desired, state.mask,
                            less_offset_ptr[group_id], less_count_ptr[group_id],
                            tie_offset_ptr[group_id], state.n_cand,
                            SelectedOut<ValueT, IndexT>{
                                arg_ptr + iter_id * n_values,
                                vals_ptr + iter_id * k, inds_ptr + iter_id * k},
                            CandidateListOut{cand_ptr + iter_id * cand_cap});
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
            const std::size_t iter_id = group_id / n_segments;
            const std::size_t segment_id = group_id - iter_id * n_segments;

            const StateT state = state_ptr[iter_id];

            const ValueT *row_arg = arg_ptr + iter_id * n_values;
            // the first pass wrote out its less elements already
            ValueT *row_vals = vals_ptr + iter_id * k + state.n_less_first;
            IndexT *row_inds = inds_ptr + iter_id * k + state.n_less_first;
            const std::uint64_t n_less = k - state.n_less_first - state.k_rem;

            if (state.n_cand) {
                using OutT = CandidateSelectedOut<ValueT, IndexT>;
                const std::uint32_t *row_cand = cand_ptr + iter_id * cand_cap;
                const auto [begin, end] =
                    segment_range(segment_id, n_segments, state.n_cand);
                gather_selected<elems_per_wi>(
                    ndit, CandKeyT{row_arg, row_cand}, begin, end,
                    state.desired, state.mask, less_offset_ptr[group_id],
                    less_count_ptr[group_id], tie_offset_ptr[group_id],
                    state.k_rem, OutT{row_arg, row_cand, row_vals, row_inds},
                    OutT{row_arg, row_cand, row_vals + n_less,
                         row_inds + n_less});
            }
            else {
                using OutT = SelectedOut<ValueT, IndexT>;
                const auto [begin, end] =
                    segment_range(segment_id, n_segments, n_values);
                gather_selected<elems_per_wi>(
                    ndit, RowKeyT{row_arg}, begin, end, state.desired,
                    state.mask, less_offset_ptr[group_id],
                    less_count_ptr[group_id], tie_offset_ptr[group_id],
                    state.k_rem, OutT{row_arg, row_vals, row_inds},
                    OutT{row_arg, row_vals + n_less, row_inds + n_less});
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

// longest rows ranked by a sub-group each, and the most sub-group sizes they
// may span; the keys are held in registers, so shorter rows get a kernel with
// fewer of them for better occupancy
inline constexpr std::size_t sub_group_max_n = 64;
inline constexpr std::uint32_t sub_group_max_chunks = 8;
inline constexpr std::uint32_t sub_group_few_chunks = 4;

template <bool is_ascending, typename ValueT, typename IndexT>
sycl::event radix_select_dispatch(sycl::queue &exec_q,
                                  std::size_t n_iters,
                                  std::size_t n_values,
                                  std::size_t k,
                                  const ValueT *arg_ptr,
                                  ValueT *vals_ptr,
                                  IndexT *inds_ptr,
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
        const std::size_t rows_per_group = wg_size / min_sg_size;
        const std::size_t n_groups =
            (n_iters + rows_per_group - 1) / rows_per_group;
        if (n_values <= sub_group_few_chunks * min_sg_size) {
            return radix_select_sub_group_submit<is_ascending,
                                                 sub_group_few_chunks>(
                exec_q, n_iters, n_values, k, arg_ptr, vals_ptr, inds_ptr,
                n_groups, wg_size, depends);
        }
        return radix_select_sub_group_submit<is_ascending,
                                             sub_group_max_chunks>(
            exec_q, n_iters, n_values, k, arg_ptr, vals_ptr, inds_ptr, n_groups,
            wg_size, depends);
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
            exec_q, n_iters, n_values, k, arg_ptr, vals_ptr, inds_ptr,
            n_segments, wg_size, filter, depends);
    }
    // short rows leave most of a large work-group idle
    std::size_t row_wg_size = 64;
    while (row_wg_size < wg_size && row_wg_size * 8 < n_values) {
        row_wg_size *= 2;
    }
    row_wg_size = std::min(row_wg_size, wg_size);
    return radix_select_one_group_submit<is_ascending, gather_elems_per_wi>(
        exec_q, n_iters, n_values, k, arg_ptr, vals_ptr, inds_ptr, row_wg_size,
        depends);
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

    if (is_ascending) {
        return radix_select_dispatch</*is_ascending*/ true>(
            exec_q, n_iters, n_values, k, arg_ptr, vals_ptr, inds_ptr, depends);
    }
    return radix_select_dispatch</*is_ascending*/ false>(
        exec_q, n_iters, n_values, k, arg_ptr, vals_ptr, inds_ptr, depends);
}

} // namespace dpnp::tensor::kernels::radix_select_details
