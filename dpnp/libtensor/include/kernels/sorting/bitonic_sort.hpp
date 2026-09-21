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
/// This file defines kernels for tensor sort/argsort operations implemented as
/// Batcher's bitonic sorting network.
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>

#include "kernels/dpnp_tensor_types.hpp"
#include "kernels/sorting/search_sorted_detail.hpp"
#include "utils/sycl_alloc_utils.hpp"

namespace dpnp::tensor::kernels
{

namespace bitonic_sort_detail
{

using dpnp::tensor::ssize_t;
using dpnp::tensor::kernels::search_sorted_detail::quotient_ceil;

/*! @brief Smallest power of two no less than `n`, `n` no greater than 2^62 */
inline std::size_t ceil_pow2(const std::size_t n)
{
    std::size_t p{1};
    while (p < n) {
        p <<= 1;
    }
    return p;
}

/*! @brief Largest power of two no greater than `n`, `n` positive */
inline std::size_t floor_pow2(const std::size_t n)
{
    std::size_t p{1};
    while ((p << 1) <= n) {
        p <<= 1;
    }
    return p;
}

/*! @brief Base two logarithm of a power of two `p` */
inline std::uint32_t log2_pow2(std::size_t p)
{
    std::uint32_t l{0};
    for (; p > 1; p >>= 1) {
        ++l;
    }
    return l;
}

template <typename T>
inline constexpr bool is_real_fp_v =
    (std::is_same_v<T, sycl::half> || std::is_same_v<T, float> ||
     std::is_same_v<T, double>);

/*! @brief Value of type `T` which the sorting order places after every value an
 *  input array may hold
 *
 *  A bitonic network only sorts sequences whose length is a power of two, so
 *  rows are padded up to the next power of two with this value. The padding
 *  ends up in the tail of the sorted row, leaving the leading `sort_nelems`
 *  entries to be the sorted row. Input values which compare equal to the
 *  padding are interchangeable with it, so the result is correct even when the
 *  input holds the padding value itself.
 */
template <typename T, bool is_ascending>
struct PadValue
{
    static T get()
    {
        if constexpr (std::is_same_v<T, bool>) {
            return is_ascending;
        }
        else if constexpr (is_real_fp_v<T>) {
            // NaN is ordered after every other value when ascending, while
            // -infinity is ordered after every other value, NaN included,
            // when descending
            return (is_ascending)
                       ? static_cast<T>(std::numeric_limits<float>::quiet_NaN())
                       : static_cast<T>(
                             -std::numeric_limits<float>::infinity());
        }
        else {
            return (is_ascending) ? std::numeric_limits<T>::max()
                                  : std::numeric_limits<T>::lowest();
        }
    }
};

template <typename T, bool is_ascending>
struct PadValue<std::complex<T>, is_ascending>
{
    static std::complex<T> get()
    {
        // complex values are ordered by the pair of NaN-ness of their parts
        // first, so (NaN, NaN) is ordered after every other value when
        // ascending, while (-infinity, -infinity) is the smallest value of the
        // leading (not NaN, not NaN) group and hence is ordered after every
        // other value when descending
        const T nan_v = static_cast<T>(std::numeric_limits<float>::quiet_NaN());
        const T inf_v = static_cast<T>(std::numeric_limits<float>::infinity());

        return (is_ascending) ? std::complex<T>{nan_v, nan_v}
                              : std::complex<T>{-inf_v, -inf_v};
    }
};

/*! @brief Element of the sorting network when sorting values */
template <typename argTy>
struct ValueNetwork
{
    using input_type = argTy;
    using element_type = argTy;
    using result_type = argTy;

    static element_type make_element(const argTy &v, const std::size_t)
    {
        return v;
    }

    static result_type result_of(const element_type &e) { return e; }
};

template <typename T, typename IndexTy>
struct KeyIndexPair
{
    T key;
    IndexTy idx;
};

/*! @brief Order key-index pairs by key, breaking ties by index
 *
 *  A bitonic network is not a stable sort, but ordering equal keys by their
 *  position makes the permutation it produces the stable one.
 */
template <typename T, typename IndexTy, typename ValueCompare>
struct KeyIndexCompare
{
    KeyIndexCompare(const ValueCompare &comp_op) : value_comp(comp_op) {}

    bool operator()(const KeyIndexPair<T, IndexTy> &lhs,
                    const KeyIndexPair<T, IndexTy> &rhs) const
    {
        if (value_comp(lhs.key, rhs.key)) {
            return true;
        }
        if (value_comp(rhs.key, lhs.key)) {
            return false;
        }
        return lhs.idx < rhs.idx;
    }

private:
    ValueCompare value_comp;
};

/*! @brief Element of the sorting network when sorting indices */
template <typename argTy, typename IndexTy>
struct KeyIndexNetwork
{
    using input_type = argTy;
    using element_type = KeyIndexPair<argTy, IndexTy>;
    using result_type = IndexTy;

    static element_type make_element(const argTy &v, const std::size_t i)
    {
        return element_type{v, static_cast<IndexTy>(i)};
    }

    static result_type result_of(const element_type &e) { return e.idx; }
};

/*! @brief Order `lhs` before `rhs` when `in_order` is set, and after it
 *  otherwise */
template <typename RefT, typename Compare>
void compare_exchange(RefT &&lhs,
                      RefT &&rhs,
                      const Compare &comp,
                      const bool in_order)
{
    using ElemT = std::remove_reference_t<RefT>;

    const ElemT lhs_v = lhs;
    const ElemT rhs_v = rhs;

    if ((in_order) ? comp(rhs_v, lhs_v) : comp(lhs_v, rhs_v)) {
        lhs = rhs_v;
        rhs = lhs_v;
    }
}

/*! @brief Run the compare-exchange steps of the bitonic network stages
 *  `k_first, 2 * k_first, ..., k_last` over blocks resident in local memory
 *
 *  Stage `k` of the network applied to a padded row compare-exchanges, for
 *  every stride `j` in `k/2, k/4, ..., 1`, the pair of elements at positions
 *  `i` and `i + j` of the row, for every `i` whose bit `log2(j)` is not set.
 *  The pair is ordered ascendingly when bit `log2(k)` of `i` is not set, and
 *  descendingly otherwise. Both positions of a pair belong to the same
 *  `block_size`-aligned block as long as `j < block_size`, which is what lets
 *  these steps run on a block held in local memory.
 *
 *  `work` holds `rows_per_wg` blocks of `block_size` elements, each starting at
 *  position `index_offset` of its (padded) row. Work-items of the group
 *  cooperate on all of the blocks, and the group is synchronized after every
 *  step.
 */
template <typename AccT, typename Compare>
void local_network_steps(const sycl::nd_item<1> &it,
                         AccT work,
                         const std::size_t rows_per_wg,
                         const std::size_t block_size,
                         const std::size_t index_offset,
                         const std::size_t k_first,
                         const std::size_t k_last,
                         const std::size_t j_first,
                         const Compare &comp)
{
    const std::size_t lws = it.get_local_range(0);
    const std::size_t lid = it.get_local_linear_id();

    const std::size_t half_block = block_size / 2;
    const std::uint32_t half_block_log2 = log2_pow2(half_block);
    const std::size_t n_pairs = rows_per_wg * half_block;

    std::size_t j_first_of_stage = j_first;
    for (std::size_t k = k_first; k <= k_last; k *= 2) {
        for (std::size_t j = j_first_of_stage; j > 0; j >>= 1) {
            for (std::size_t p = lid; p < n_pairs; p += lws) {
                const std::size_t block_id = p >> half_block_log2;
                const std::size_t pair_id = p & (half_block - 1);

                // position of the pair within the block, obtained by inserting
                // an unset bit at position log2(j) into the pair number
                const std::size_t low_bits = pair_id & (j - 1);
                const std::size_t i = ((pair_id - low_bits) << 1) + low_bits;

                const std::size_t base = block_id * block_size;
                const bool in_order = (((index_offset + i) & k) == 0);

                compare_exchange(work[base + i], work[base + i + j], comp,
                                 in_order);
            }
            sycl::group_barrier(it.get_group());
        }
        // the next stage is 2 * k, and its first stride is k
        j_first_of_stage = k;
    }
}

/*! @brief Parameters of the kernels running network steps in local memory */
struct LocalNetworkParams
{
    std::size_t lws;         //!< work-group size
    std::size_t block_size;  //!< row elements held in local memory at a time
    std::size_t rows_per_wg; //!< rows a work-group is responsible for
};

/*! @brief Number of work-items a kernel wants to have to keep the device busy
 *
 *  Enough of them to fill every compute unit several times over, so that the
 *  memory latency of one work-item is hidden behind the work of another.
 */
inline std::size_t work_items_target(const sycl::device &dev)
{
    static constexpr std::size_t work_items_per_compute_unit = 4;

    const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    const std::size_t max_sg_size =
        *std::max_element(sg_sizes.begin(), sg_sizes.end());

    return work_items_per_compute_unit *
           dev.get_info<sycl::info::device::max_compute_units>() * max_sg_size;
}

/*! @brief Number of elements of type `ElemT` local memory has room for */
template <typename ElemT>
std::size_t local_memory_nelems(const sycl::device &dev)
{
    //  leave 512 bytes of local memory for RT
    static constexpr std::uint64_t safety_margin = 512;
    const std::uint64_t device_local_memory_size =
        dev.get_info<sycl::info::device::local_mem_size>();

    return (device_local_memory_size - safety_margin) / sizeof(ElemT);
}

/*! @brief Size the local memory blocks and the work-groups operating on them
 *
 *  When a whole padded row fits in local memory the block covers it entirely,
 *  and short rows are batched into a work-group so that it has enough pairs to
 *  keep its work-items busy. Otherwise the block is the largest power-of-two
 *  number of elements which fits, and the network stages with a stride no
 *  smaller than the block size have to be run through global memory.
 *
 *  A work-item is given several elements of the block to work on, rather than
 *  the single pair a compare-exchange step needs, because the group barrier
 *  which separates the steps costs about as much as a handful of them. The
 *  work-items are given fewer elements each when a large share of them would
 *  otherwise sit idle for the want of work-groups to run.
 */
template <typename ElemT>
LocalNetworkParams get_local_network_params(const sycl::device &dev,
                                            const std::size_t iter_nelems,
                                            const std::size_t padded_nelems)
{
    const std::size_t nelems_per_slm = local_memory_nelems<ElemT>(dev);

    if (nelems_per_slm < 2) {
        throw std::runtime_error(
            "Bitonic sort requires more local memory than the device provides");
    }

    const std::size_t max_lws =
        dev.get_info<sycl::info::device::max_work_group_size>();
    const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    const std::size_t max_sg_size =
        *std::max_element(sg_sizes.begin(), sg_sizes.end());

    // elements of a block a work-item is responsible for, chosen experimentally
    // over a sweep of row lengths, row counts and data types
    static constexpr std::size_t nelems_per_work_item = 8;
    const std::size_t items_target = work_items_target(dev);

    LocalNetworkParams params{};
    params.block_size = std::min(padded_nelems, floor_pow2(nelems_per_slm));

    // batch rows until a work-group has a sub-group's worth of work, which a
    // block covering only a part of a row has on its own
    const std::size_t min_slm_nelems = nelems_per_work_item * max_sg_size;
    std::size_t rows_per_wg = 1;
    if (params.block_size == padded_nelems) {
        while (rows_per_wg * params.block_size < min_slm_nelems &&
               2 * rows_per_wg <= iter_nelems &&
               2 * rows_per_wg * params.block_size <= nelems_per_slm) {
            rows_per_wg *= 2;
        }
    }
    params.rows_per_wg = rows_per_wg;

    // hand the work-items fewer elements each when there would not be enough of
    // them to keep the device busy otherwise
    std::size_t nelems_per_item = nelems_per_work_item;
    while (nelems_per_item > 1 &&
           iter_nelems * padded_nelems < nelems_per_item * items_target) {
        nelems_per_item /= 2;
    }

    const std::size_t slm_nelems = rows_per_wg * params.block_size;
    params.lws =
        std::min({max_lws, std::max(std::size_t(1), slm_nelems / 2),
                  std::max(std::size_t(1), slm_nelems / nelems_per_item)});

    return params;
}

template <typename ElemT, typename Compare>
class bitonic_sort_rows_krn;

/*! @brief Sort every row with a single work-group pass over local memory
 *
 *  Applicable when a padded row fits in local memory, in which case the row is
 *  read and written exactly once.
 */
template <typename Network, typename Compare>
sycl::event sort_rows_in_slm(sycl::queue &exec_q,
                             const std::size_t iter_nelems,
                             const std::size_t sort_nelems,
                             const typename Network::input_type *src,
                             typename Network::result_type *dst,
                             const Compare &comp,
                             const typename Network::input_type &pad,
                             const LocalNetworkParams &params,
                             const std::vector<sycl::event> &depends)
{
    using ElemT = typename Network::element_type;
    using KernelName = bitonic_sort_rows_krn<ElemT, Compare>;

    const std::size_t block_size = params.block_size;
    const std::size_t rows_per_wg = params.rows_per_wg;
    const std::size_t lws = params.lws;

    const std::uint32_t block_log2 = log2_pow2(block_size);
    const std::size_t slm_nelems = rows_per_wg * block_size;
    const std::size_t n_groups = quotient_ceil(iter_nelems, rows_per_wg);

    return exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        sycl::local_accessor<ElemT, 1> work(sycl::range<1>{slm_nelems}, cgh);

        sycl::nd_range<1> ndRange(sycl::range<1>{n_groups * lws},
                                  sycl::range<1>{lws});

        cgh.parallel_for<KernelName>(ndRange, [=](sycl::nd_item<1> it) {
            const std::size_t lid = it.get_local_linear_id();
            const std::size_t row_base = it.get_group_linear_id() * rows_per_wg;

            for (std::size_t p = lid; p < slm_nelems; p += lws) {
                const std::size_t row = row_base + (p >> block_log2);
                const std::size_t i = p & (block_size - 1);

                const bool is_valid = (row < iter_nelems) && (i < sort_nelems);
                work[p] = Network::make_element(
                    (is_valid) ? src[row * sort_nelems + i] : pad, i);
            }
            sycl::group_barrier(it.get_group());

            local_network_steps(it, work, rows_per_wg, block_size,
                                /* index_offset */ 0,
                                /* k_first */ 2, /* k_last */ block_size,
                                /* j_first */ 1, comp);

            for (std::size_t p = lid; p < slm_nelems; p += lws) {
                const std::size_t row = row_base + (p >> block_log2);
                const std::size_t i = p & (block_size - 1);

                if ((row < iter_nelems) && (i < sort_nelems)) {
                    dst[row * sort_nelems + i] = Network::result_of(work[p]);
                }
            }
        });
    });
}

template <typename ElemT, typename Compare>
class bitonic_sort_blocks_krn;

/*! @brief Sort the `block_size`-long blocks of every padded row in local memory
 *
 *  These are the network stages up to and including `k == block_size`, so
 *  consecutive blocks come out sorted in alternating directions, as the
 *  `k == 2 * block_size` stage requires.
 */
template <typename Network, typename Compare>
sycl::event sort_blocks_in_slm(sycl::queue &exec_q,
                               const std::size_t iter_nelems,
                               const std::size_t sort_nelems,
                               const std::size_t padded_nelems,
                               const typename Network::input_type *src,
                               typename Network::element_type *work_tp,
                               const Compare &comp,
                               const typename Network::input_type &pad,
                               const LocalNetworkParams &params,
                               const std::vector<sycl::event> &depends)
{
    using ElemT = typename Network::element_type;
    using KernelName = bitonic_sort_blocks_krn<ElemT, Compare>;

    const std::size_t block_size = params.block_size;
    const std::size_t lws = params.lws;

    const std::size_t n_blocks = padded_nelems / block_size;
    const std::size_t n_groups = iter_nelems * n_blocks;

    return exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        sycl::local_accessor<ElemT, 1> work(sycl::range<1>{block_size}, cgh);

        sycl::nd_range<1> ndRange(sycl::range<1>{n_groups * lws},
                                  sycl::range<1>{lws});

        cgh.parallel_for<KernelName>(ndRange, [=](sycl::nd_item<1> it) {
            const std::size_t lid = it.get_local_linear_id();
            const std::size_t group_id = it.get_group_linear_id();

            const std::size_t row = group_id / n_blocks;
            const std::size_t index_offset =
                (group_id - row * n_blocks) * block_size;

            for (std::size_t i = lid; i < block_size; i += lws) {
                const std::size_t row_i = index_offset + i;

                work[i] = Network::make_element(
                    (row_i < sort_nelems) ? src[row * sort_nelems + row_i]
                                          : pad,
                    row_i);
            }
            sycl::group_barrier(it.get_group());

            local_network_steps(it, work, /* rows_per_wg */ 1, block_size,
                                index_offset,
                                /* k_first */ 2, /* k_last */ block_size,
                                /* j_first */ 1, comp);

            for (std::size_t i = lid; i < block_size; i += lws) {
                work_tp[row * padded_nelems + index_offset + i] = work[i];
            }
        });
    });
}

template <typename ElemT, typename Compare, bool store_result>
class bitonic_merge_blocks_krn;

/*! @brief Run the strides of stage `k` which are smaller than the block size in
 *  local memory
 *
 *  When `store_result` is set, the sorted block is written to `dst` rather than
 *  back to `work_tp`, which saves a pass over global memory for the last stage
 *  of the network.
 */
template <typename Network, typename Compare, bool store_result>
sycl::event merge_blocks_in_slm(sycl::queue &exec_q,
                                const std::size_t iter_nelems,
                                const std::size_t sort_nelems,
                                const std::size_t padded_nelems,
                                typename Network::element_type *work_tp,
                                typename Network::result_type *dst,
                                const Compare &comp,
                                const std::size_t k,
                                const LocalNetworkParams &params,
                                const std::vector<sycl::event> &depends)
{
    using ElemT = typename Network::element_type;
    using KernelName = bitonic_merge_blocks_krn<ElemT, Compare, store_result>;

    const std::size_t block_size = params.block_size;
    const std::size_t lws = params.lws;

    const std::size_t n_blocks = padded_nelems / block_size;
    const std::size_t n_groups = iter_nelems * n_blocks;

    return exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        sycl::local_accessor<ElemT, 1> work(sycl::range<1>{block_size}, cgh);

        sycl::nd_range<1> ndRange(sycl::range<1>{n_groups * lws},
                                  sycl::range<1>{lws});

        cgh.parallel_for<KernelName>(ndRange, [=](sycl::nd_item<1> it) {
            const std::size_t lid = it.get_local_linear_id();
            const std::size_t group_id = it.get_group_linear_id();

            const std::size_t row = group_id / n_blocks;
            const std::size_t index_offset =
                (group_id - row * n_blocks) * block_size;
            ElemT *block_tp = work_tp + row * padded_nelems + index_offset;

            for (std::size_t i = lid; i < block_size; i += lws) {
                work[i] = block_tp[i];
            }
            sycl::group_barrier(it.get_group());

            local_network_steps(it, work, /* rows_per_wg */ 1, block_size,
                                index_offset,
                                /* k_first */ k, /* k_last */ k,
                                /* j_first */ block_size / 2, comp);

            if constexpr (store_result) {
                typename Network::result_type *dst_tp =
                    dst + row * sort_nelems + index_offset;
                const std::size_t n_to_store =
                    (index_offset < sort_nelems)
                        ? std::min<std::size_t>(block_size,
                                                sort_nelems - index_offset)
                        : std::size_t(0);

                for (std::size_t i = lid; i < n_to_store; i += lws) {
                    dst_tp[i] = Network::result_of(work[i]);
                }
            }
            else {
                for (std::size_t i = lid; i < block_size; i += lws) {
                    block_tp[i] = work[i];
                }
            }
        });
    });
}

template <typename ElemT, typename Compare, std::uint32_t n_strides>
class bitonic_merge_global_krn;

/*! @brief Compare-exchange steps of stage `k` for the `n_strides` strides
 *  `j_low * 2^(n_strides - 1), ..., 2 * j_low, j_low`, which pair up elements
 *  of different local memory blocks
 *
 *  Those strides only ever pair up the `2^n_strides` elements
 *  `i, i + j_low, ..., i + (2^n_strides - 1) * j_low` among themselves, for an
 *  `i` whose bits `log2(j_low), ..., log2(j_low) + n_strides - 1` are unset. A
 *  work-item takes all of them, so the whole run of strides costs one pass over
 *  global memory rather than one pass each.
 */
template <typename ElemT, typename Compare, std::uint32_t n_strides>
sycl::event global_merge_steps(sycl::queue &exec_q,
                               const std::size_t iter_nelems,
                               const std::size_t padded_nelems,
                               ElemT *work_tp,
                               const Compare &comp,
                               const std::size_t k,
                               const std::size_t j_low,
                               const std::vector<sycl::event> &depends)
{
    using KernelName = bitonic_merge_global_krn<ElemT, Compare, n_strides>;

    static constexpr std::size_t nelems_per_item = std::size_t(1) << n_strides;

    const std::size_t items_per_row = padded_nelems / nelems_per_item;
    const std::uint32_t items_per_row_log2 = log2_pow2(items_per_row);
    const std::uint32_t j_low_log2 = log2_pow2(j_low);

    return exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        cgh.parallel_for<KernelName>(
            sycl::range<1>{iter_nelems * items_per_row}, [=](sycl::id<1> id) {
                const std::size_t p = id[0];
                const std::size_t row = p >> items_per_row_log2;
                const std::size_t item_id = p & (items_per_row - 1);

                // position of the first of the elements, obtained by inserting
                // `n_strides` unset bits at position log2(j_low) into the
                // work-item number
                const std::size_t low_bits = item_id & (j_low - 1);
                const std::size_t i =
                    ((item_id - low_bits) << n_strides) + low_bits;

                ElemT *row_tp = work_tp + row * padded_nelems;

                ElemT elems[nelems_per_item];
                for (std::size_t u = 0; u < nelems_per_item; ++u) {
                    elems[u] = row_tp[i + (u << j_low_log2)];
                }

                for (std::uint32_t s = n_strides; s-- > 0;) {
                    const std::size_t stride_u = std::size_t(1) << s;
                    for (std::size_t u = 0; u < nelems_per_item; ++u) {
                        if ((u & stride_u) != 0) {
                            continue;
                        }
                        const bool in_order =
                            (((i + (u << j_low_log2)) & k) == 0);

                        compare_exchange(elems[u], elems[u + stride_u], comp,
                                         in_order);
                    }
                }

                for (std::size_t u = 0; u < nelems_per_item; ++u) {
                    row_tp[i + (u << j_low_log2)] = elems[u];
                }
            });
    });
}

/*! @brief Number of strides of stage `k` which a single pass over global memory
 *  runs, at most `max_strides_per_pass`
 *
 *  Taking more strides per pass saves passes, but leaves fewer work-items to
 *  run, which is why it is not done when the device would be left short of
 *  work.
 */
inline std::uint32_t strides_per_global_pass(const std::size_t items_target,
                                             const std::size_t iter_nelems,
                                             const std::size_t padded_nelems,
                                             const std::size_t j,
                                             const std::size_t block_size)
{
    static constexpr std::uint32_t max_strides_per_pass = 3;

    std::uint32_t n_strides = 1;
    while (n_strides < max_strides_per_pass && (j >> n_strides) >= block_size &&
           ((iter_nelems * padded_nelems) >> (n_strides + 1)) >= items_target) {
        ++n_strides;
    }

    return n_strides;
}

/*! @brief Run the network on padded rows held in `work_tp`, for a block size
 *  which leaves some of the stages to global memory
 *
 *  `work_tp` holds `iter_nelems` rows of `padded_nelems` elements, and may be
 *  `dst` itself when the two hold the same type and the rows need no padding.
 */
template <typename Network, typename Compare>
sycl::event
    sort_rows_through_global_memory(sycl::queue &exec_q,
                                    const std::size_t iter_nelems,
                                    const std::size_t sort_nelems,
                                    const std::size_t padded_nelems,
                                    const typename Network::input_type *src,
                                    typename Network::element_type *work_tp,
                                    typename Network::result_type *dst,
                                    const Compare &comp,
                                    const typename Network::input_type &pad,
                                    const LocalNetworkParams &params,
                                    const std::vector<sycl::event> &depends)
{
    using ElemT = typename Network::element_type;

    const std::size_t block_size = params.block_size;
    const std::size_t items_target = work_items_target(exec_q.get_device());

    sycl::event dep_ev = sort_blocks_in_slm<Network, Compare>(
        exec_q, iter_nelems, sort_nelems, padded_nelems, src, work_tp, comp,
        pad, params, depends);

    for (std::size_t k = 2 * block_size; k <= padded_nelems; k *= 2) {
        std::size_t j = k / 2;
        while (j >= block_size) {
            const std::uint32_t n_strides = strides_per_global_pass(
                items_target, iter_nelems, padded_nelems, j, block_size);
            const std::size_t j_low = j >> (n_strides - 1);

            switch (n_strides) {
            case 3:
                dep_ev = global_merge_steps<ElemT, Compare, 3>(
                    exec_q, iter_nelems, padded_nelems, work_tp, comp, k, j_low,
                    {dep_ev});
                break;
            case 2:
                dep_ev = global_merge_steps<ElemT, Compare, 2>(
                    exec_q, iter_nelems, padded_nelems, work_tp, comp, k, j_low,
                    {dep_ev});
                break;
            default:
                dep_ev = global_merge_steps<ElemT, Compare, 1>(
                    exec_q, iter_nelems, padded_nelems, work_tp, comp, k, j_low,
                    {dep_ev});
                break;
            }

            j = j_low / 2;
        }

        if (k == padded_nelems) {
            // the network is done, so the sorted blocks can go straight out
            dep_ev = merge_blocks_in_slm<Network, Compare, /* store */ true>(
                exec_q, iter_nelems, sort_nelems, padded_nelems, work_tp, dst,
                comp, k, params, {dep_ev});
        }
        else {
            dep_ev = merge_blocks_in_slm<Network, Compare, /* store */ false>(
                exec_q, iter_nelems, sort_nelems, padded_nelems, work_tp, dst,
                comp, k, params, {dep_ev});
        }
    }

    return dep_ev;
}

/*! @brief Sort rows of a contiguous array with a bitonic sorting network
 *
 *  Rows are padded up to a power of two length, sorted by the network, and the
 *  leading `sort_nelems` elements of every padded row are written out. The
 *  stages of the network are run in local memory as long as their strides let
 *  the pairs they compare-exchange stay within a block held there, and through
 *  global memory otherwise.
 *
 *  The source and the destination must not overlap, since the destination is
 *  written to while the source is still being read.
 */
template <typename Network, typename Compare>
sycl::event bitonic_sort_impl(sycl::queue &exec_q,
                              const std::size_t iter_nelems,
                              const std::size_t sort_nelems,
                              const typename Network::input_type *src,
                              typename Network::result_type *dst,
                              const Compare &comp,
                              const typename Network::input_type &pad,
                              const std::vector<sycl::event> &depends)
{
    using ElemT = typename Network::element_type;

    const std::size_t padded_nelems = ceil_pow2(sort_nelems);
    const auto params = get_local_network_params<ElemT>(
        exec_q.get_device(), iter_nelems, padded_nelems);

    if (params.block_size == padded_nelems) {
        // a padded row fits in local memory, so a single kernel sorts it
        return sort_rows_in_slm<Network, Compare>(exec_q, iter_nelems,
                                                  sort_nelems, src, dst, comp,
                                                  pad, params, depends);
    }

    // rows which are already of a power of two length need no padding, so when
    // the destination holds the type the network sorts it can hold the network
    // as it runs, sparing a temporary the size of the array
    if constexpr (std::is_same_v<ElemT, typename Network::result_type>) {
        if (padded_nelems == sort_nelems) {
            return sort_rows_through_global_memory<Network, Compare>(
                exec_q, iter_nelems, sort_nelems, padded_nelems, src, dst, dst,
                comp, pad, params, depends);
        }
    }

    auto work_owner = dpnp::tensor::alloc_utils::smart_malloc_device<ElemT>(
        iter_nelems * padded_nelems, exec_q);
    // extract USM pointer
    ElemT *work_tp = work_owner.get();

    sycl::event dep_ev = sort_rows_through_global_memory<Network, Compare>(
        exec_q, iter_nelems, sort_nelems, padded_nelems, src, work_tp, dst,
        comp, pad, params, depends);

    return dpnp::tensor::alloc_utils::async_smart_free(exec_q, {dep_ev},
                                                       work_owner);
}

} // namespace bitonic_sort_detail

template <typename argTy, typename Comp, bool is_ascending>
sycl::event bitonic_sort_axis1_contig_impl(
    sycl::queue &exec_q,
    std::size_t iter_nelems, // number of sub-arrays to sort (num. of rows in a
                             // matrix when sorting over rows)
    std::size_t sort_nelems, // size of each array to sort  (length of rows,
                             // i.e. number of columns)
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

    if (sort_nelems < 2) {
        return exec_q.copy<argTy>(arg_tp, res_tp, iter_nelems * sort_nelems,
                                  depends);
    }

    using bitonic_sort_detail::PadValue;
    using Network = bitonic_sort_detail::ValueNetwork<argTy>;

    return bitonic_sort_detail::bitonic_sort_impl<Network, Comp>(
        exec_q, iter_nelems, sort_nelems, arg_tp, res_tp, Comp{},
        PadValue<argTy, is_ascending>::get(), depends);
}

template <typename argTy,
          typename IndexTy,
          typename ValueComp,
          bool is_ascending>
sycl::event bitonic_argsort_axis1_contig_impl(
    sycl::queue &exec_q,
    std::size_t iter_nelems, // number of sub-arrays to sort (num. of rows in a
                             // matrix when sorting over rows)
    std::size_t sort_nelems, // size of each array to sort  (length of rows,
                             // i.e. number of columns)
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

    if (sort_nelems < 2) {
        return exec_q.fill<IndexTy>(res_tp, IndexTy{0},
                                    iter_nelems * sort_nelems, depends);
    }

    using bitonic_sort_detail::KeyIndexCompare;
    using bitonic_sort_detail::PadValue;
    using Network = bitonic_sort_detail::KeyIndexNetwork<argTy, IndexTy>;
    using Comp = KeyIndexCompare<argTy, IndexTy, ValueComp>;

    return bitonic_sort_detail::bitonic_sort_impl<Network, Comp>(
        exec_q, iter_nelems, sort_nelems, arg_tp, res_tp, Comp{ValueComp{}},
        PadValue<argTy, is_ascending>::get(), depends);
}

/*! @brief Longest row `bitonic_sort_axis1_contig_impl` sorts without resorting
 *  to global memory
 *
 *  Rows longer than this are sorted through a sequence of kernels which pass
 *  over global memory several times, rather than by a single kernel which
 *  reads the row, sorts it in local memory, and writes it out. Note that a row
 *  is padded up to a power of two length before it is sorted, so it is the
 *  padded length which has to fit.
 */
template <typename argTy>
std::size_t bitonic_sort_local_memory_nelems(const sycl::device &dev)
{
    using Network = bitonic_sort_detail::ValueNetwork<argTy>;
    using ElemT = typename Network::element_type;

    return bitonic_sort_detail::floor_pow2(
        bitonic_sort_detail::local_memory_nelems<ElemT>(dev));
}

/*! @brief Longest row `bitonic_argsort_axis1_contig_impl` sorts without
 *  resorting to global memory
 *
 *  @see bitonic_sort_local_memory_nelems
 */
template <typename argTy, typename IndexTy>
std::size_t bitonic_argsort_local_memory_nelems(const sycl::device &dev)
{
    using Network = bitonic_sort_detail::KeyIndexNetwork<argTy, IndexTy>;
    using ElemT = typename Network::element_type;

    return bitonic_sort_detail::floor_pow2(
        bitonic_sort_detail::local_memory_nelems<ElemT>(dev));
}

} // namespace dpnp::tensor::kernels
