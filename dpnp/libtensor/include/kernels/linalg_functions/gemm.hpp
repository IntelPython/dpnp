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
/// This file defines kernels for general matrix multiplication (GEMM).
///
/// Every entry point computes, for each batch element b,
///
///     res[b](i, j) = sum_s lhs[b](i, s) * rhs[b](s, j)
///
/// with i < n, s < k, j < m, accumulating in the result type. Matrices are
/// addressed through (row, col) indexers, so C- and F-contiguous, transposed
/// and 2D-strided operands all take the same code path.
///
/// Three kernels do the work, selected by `gemm_detail::gemm_driver`:
///
///   - GemvFunctor: n == 1 or m == 1 (matrix-vector / vector-matrix);
///   - GemmSmallFunctor: one work-item per output element, for matrices too
///     small to fill a register tile;
///   - GemmTileFunctor: register-blocked kernel staging tiles of both
///     operands in SLM.
///
/// When the number of output elements is too small to occupy the device, the
/// reduction over k is split into chunks computed by independent work-groups.
/// If the result type supports atomics on the result allocation, the result
/// is zeroed and every chunk atomically adds its partial sums to it. Otherwise
/// the partial results are written to a temporary buffer and summed by a
/// single GemmPartialsReduceFunctor pass.
///
/// When lhs[b] is the transpose of rhs[b] (x.T @ x, x @ x.T), the result is
/// symmetric: the small and tile kernels then only compute the upper triangle
/// and mirror it.
//===---------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "kernels/dpnp_tensor_types.hpp"
#include "utils/offset_utils.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/sycl_utils.hpp"
#include "utils/type_utils.hpp"

namespace dpnp::tensor::kernels
{

using dpnp::tensor::ssize_t;

namespace gemm_detail
{

/*! @brief Maps (row, col) of a single matrix to the element displacement
 * `row * row_stride + col * col_stride`. */
struct MatrixStridedIndexer
{
    constexpr MatrixStridedIndexer(ssize_t row_stride, ssize_t col_stride)
        : rs(row_stride), cs(col_stride)
    {
    }

    constexpr ssize_t operator()(std::size_t row, std::size_t col) const
    {
        return static_cast<ssize_t>(row) * rs + static_cast<ssize_t>(col) * cs;
    }

    /*! @brief Whether elements of a row are closer in memory than elements of
     * a column, i.e. whether loads should be coalesced along columns. */
    constexpr bool row_major() const
    {
        return (cs < 0 ? -cs : cs) <= (rs < 0 ? -rs : rs);
    }

private:
    ssize_t rs;
    ssize_t cs;
};

static_assert(sycl::is_device_copyable_v<MatrixStridedIndexer>);

/*! @brief Maps (row, col) of a single matrix, whose rows and columns
 * each flatten several array dimensions that cannot be collapsed into one,
 * to the element displacement. Used by tensordot on non-contiguous views. */
struct MatrixFlatIndexer
{
    MatrixFlatIndexer(int nd,
                      const ssize_t *packed_shape_strides,
                      std::size_t ncols)
        : indexer(nd, 0, packed_shape_strides), cols(ncols)
    {
    }

    ssize_t operator()(std::size_t row, std::size_t col) const
    {
        return indexer(row * cols + col);
    }

    constexpr bool row_major() const { return true; }

private:
    dpnp::tensor::offset_utils::StridedIndexer indexer;
    std::size_t cols;
};

static_assert(sycl::is_device_copyable_v<MatrixFlatIndexer>);

/*! @brief Batch indexer returning offsets of lhs, rhs and res matrices of
 * the given batch element.
 *
 * When `partials_step` is non-zero, the res offset instead points to the
 * batch element's slot in the contiguous buffer of partial results used by
 * the k-split, `batch_id * partials_step`.
 */
template <typename BaseIndexerT>
struct GemmBatchIndexer
{
    constexpr GemmBatchIndexer(const BaseIndexerT &base_indexer,
                               ssize_t partials_step_ = 0)
        : base(base_indexer), partials_step(partials_step_)
    {
    }

    dpnp::tensor::offset_utils::ThreeOffsets<ssize_t>
        operator()(std::size_t batch_id) const
    {
        const auto &offsets = base(static_cast<ssize_t>(batch_id));
        return dpnp::tensor::offset_utils::ThreeOffsets<ssize_t>(
            offsets.get_first_offset(), offsets.get_second_offset(),
            (partials_step == 0)
                ? offsets.get_third_offset()
                : static_cast<ssize_t>(batch_id) * partials_step);
    }

    constexpr GemmBatchIndexer for_partials(ssize_t step) const
    {
        return GemmBatchIndexer(base, step);
    }

private:
    BaseIndexerT base;
    ssize_t partials_step;
};

template <typename T>
constexpr T ceil_div(T a, T b)
{
    return (a + b - 1) / b;
}

inline std::size_t round_up_pow2(std::size_t v)
{
    std::size_t p = 1;
    while (p < v) {
        p <<= 1;
    }
    return p;
}

/*! @brief Number of work-items the device needs in flight to be kept busy.
 * Kernels whose launch is smaller than this split the k-reduction. */
inline std::size_t target_work_items(const sycl::device &dev)
{
    const std::size_t cu =
        dev.get_info<sycl::info::device::max_compute_units>();
    // Intel GPUs report EUs as compute units, each running several threads of
    // SIMD16/32 work-items. CPU devices report cores.
    return (dev.is_gpu() ? cu * 256 : cu * 64);
}

/*! @brief Choose the number of chunks to split the reduction over `k` into.
 *
 * @param launched_wis  Work-items launched without splitting.
 * @param k             Reduction size.
 * @param min_chunk     Minimum number of k-iterations per chunk.
 * @param chunk_align   Chunks sizes are multiples of this value.
 * @return split with k_split * k_chunk >= k
 */
struct KSplit
{
    std::size_t k_split;
    std::size_t k_chunk;
};

inline KSplit choose_k_split(const sycl::device &dev,
                             std::size_t launched_wis,
                             std::size_t k,
                             std::size_t min_chunk,
                             std::size_t chunk_align = 1)
{
    static constexpr std::size_t max_k_split = 256;

    const std::size_t target = target_work_items(dev);
    std::size_t k_split = 1;
    if (launched_wis < target) {
        k_split = std::min({ceil_div(target, launched_wis),
                            ceil_div(k, min_chunk), max_k_split});
        k_split = std::max<std::size_t>(k_split, 1);
    }
    std::size_t k_chunk = ceil_div(k, k_split);
    k_chunk = ceil_div(k_chunk, chunk_align) * chunk_align;
    k_split = ceil_div(k, k_chunk);
    return KSplit{k_split, k_chunk};
}

/*! @brief Whether the k-chunks of a split product can accumulate into the
 * result with atomic adds, given that the device supports atomics of this
 * type on the result's allocation. Matches the types accepted by
 * atomic_support::check_atomic_support. */
template <typename T>
inline constexpr bool gemm_atomic_accumulate_v =
    std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::uint32_t> ||
    std::is_same_v<T, std::int64_t> || std::is_same_v<T, std::uint64_t> ||
    std::is_same_v<T, float> || std::is_same_v<T, double>;

/*! @brief Write the partial result `val` of k-chunk `block_s` to element
 * `off` of a result matrix.
 *
 * With `accumulate`, all chunks share the zero-initialized result and add to
 * it atomically. Otherwise each chunk owns a copy of the result,
 * `split_stride` elements apart (a single copy with no split).
 */
template <typename T>
[[gnu::always_inline]] inline void store_result(T *res,
                                                std::size_t block_s,
                                                std::size_t split_stride,
                                                bool accumulate,
                                                ssize_t off,
                                                T val)
{
    if constexpr (gemm_atomic_accumulate_v<T>) {
        if (accumulate) {
            sycl::atomic_ref<T, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
                res[off])
                .fetch_add(val);
            return;
        }
    }
    res[block_s * split_stride + off] = val;
}

} // namespace gemm_detail

/*! @brief Matrix-vector product: n == 1 or m == 1.
 *
 * Computes y(r) = sum_s M(r, s) * v(s), where for `transposed == false`
 * M = lhs, v = rhs(:, 0), y = res(:, 0), and for `transposed == true`
 * M = rhs^T, v = lhs(0, :), y = res(0, :).
 *
 * A work-group covers `wg_rows` values of r, with `wg_lanes` work-items per
 * r striding over s. When M is row-major the lanes along s are adjacent
 * work-items, otherwise the rows are, so that loads of M are coalesced. The
 * lanes are then reduced in SLM.
 */
template <typename lhsT,
          typename rhsT,
          typename resT,
          typename LocAccT,
          typename BatchIndexerT,
          typename LhsIndexerT,
          typename RhsIndexerT,
          typename ResIndexerT,
          bool transposed>
class GemvFunctor
{
private:
    const lhsT *lhs = nullptr;
    const rhsT *rhs = nullptr;
    resT *res = nullptr;
    LocAccT workspace;
    std::size_t rows = 0;
    std::size_t k = 0;
    std::size_t row_groups = 0;
    std::size_t k_split = 0;
    std::size_t k_chunk = 0;
    std::size_t res_split_stride = 0;
    bool accumulate = false;
    std::uint32_t wg_rows = 0;
    std::uint32_t wg_lanes = 0;
    bool lanes_along_k = true;
    BatchIndexerT batch_indexer;
    LhsIndexerT lhs_indexer;
    RhsIndexerT rhs_indexer;
    ResIndexerT res_indexer;

public:
    GemvFunctor(const lhsT *lhs_,
                const rhsT *rhs_,
                resT *res_,
                LocAccT workspace_,
                std::size_t rows_,
                std::size_t k_,
                std::size_t row_groups_,
                std::size_t k_split_,
                std::size_t k_chunk_,
                std::size_t res_split_stride_,
                bool accumulate_,
                std::uint32_t wg_rows_,
                std::uint32_t wg_lanes_,
                bool lanes_along_k_,
                const BatchIndexerT &batch_indexer_,
                const LhsIndexerT &lhs_indexer_,
                const RhsIndexerT &rhs_indexer_,
                const ResIndexerT &res_indexer_)
        : lhs(lhs_), rhs(rhs_), res(res_), workspace(workspace_), rows(rows_),
          k(k_), row_groups(row_groups_), k_split(k_split_), k_chunk(k_chunk_),
          res_split_stride(res_split_stride_), accumulate(accumulate_),
          wg_rows(wg_rows_), wg_lanes(wg_lanes_), lanes_along_k(lanes_along_k_),
          batch_indexer(batch_indexer_), lhs_indexer(lhs_indexer_),
          rhs_indexer(rhs_indexer_), res_indexer(res_indexer_)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        const std::size_t groups_per_batch = row_groups * k_split;
        const std::size_t batch_id =
            it.get_group_linear_id() / groups_per_batch;
        const std::size_t gr_id =
            it.get_group_linear_id() - batch_id * groups_per_batch;
        // block_r moves fastest, then block_s
        const std::size_t block_s = gr_id / row_groups;
        const std::size_t block_r = gr_id - block_s * row_groups;

        const std::uint32_t lid = it.get_local_linear_id();
        std::uint32_t r_l, s_l, lane_step;
        if (lanes_along_k) {
            r_l = lid / wg_lanes;
            s_l = lid - r_l * wg_lanes;
            lane_step = 1;
        }
        else {
            s_l = lid / wg_rows;
            r_l = lid - s_l * wg_rows;
            lane_step = wg_rows;
        }

        const std::size_t r = block_r * wg_rows + r_l;
        const std::size_t s_begin = block_s * k_chunk;
        const std::size_t s_end = std::min(k, s_begin + k_chunk);

        const auto &offsets = batch_indexer(batch_id);
        const lhsT *lhs_p = lhs + offsets.get_first_offset();
        const rhsT *rhs_p = rhs + offsets.get_second_offset();

        resT acc(0);
        if (r < rows) {
            for (std::size_t s = s_begin + s_l; s < s_end; s += wg_lanes) {
                if constexpr (transposed) {
                    acc += static_cast<resT>(lhs_p[lhs_indexer(0, s)]) *
                           static_cast<resT>(rhs_p[rhs_indexer(s, r)]);
                }
                else {
                    acc += static_cast<resT>(lhs_p[lhs_indexer(r, s)]) *
                           static_cast<resT>(rhs_p[rhs_indexer(s, 0)]);
                }
            }
        }

        // tree-reduce the lanes of each row; lid indexes the workspace
        workspace[lid] = acc;
        for (std::uint32_t h = wg_lanes / 2; h > 0; h /= 2) {
            sycl::group_barrier(it.get_group(), sycl::memory_scope::work_group);
            if (s_l < h) {
                workspace[lid] =
                    workspace[lid] + workspace[lid + h * lane_step];
            }
        }

        if (s_l == 0 && r < rows) {
            resT *res_p = res + offsets.get_third_offset();
            const ssize_t off =
                transposed ? res_indexer(0, r) : res_indexer(r, 0);
            gemm_detail::store_result(res_p, block_s, res_split_stride,
                                      accumulate, off, workspace[lid]);
        }
    }
};

/*! @brief One work-item per output element, reading operands directly from
 * global memory. Suited to matrices too small to fill a register tile, e.g.
 * large batches of tiny matrices. */
template <typename lhsT,
          typename rhsT,
          typename resT,
          typename BatchIndexerT,
          typename LhsIndexerT,
          typename RhsIndexerT,
          typename ResIndexerT>
class GemmSmallFunctor
{
private:
    const lhsT *lhs = nullptr;
    const rhsT *rhs = nullptr;
    resT *res = nullptr;
    std::size_t batch_nelems = 0;
    std::size_t n = 0;
    std::size_t k = 0;
    std::size_t m = 0;
    std::size_t k_split = 0;
    std::size_t k_chunk = 0;
    std::size_t res_split_stride = 0;
    bool accumulate = false;
    bool symmetric = false;
    BatchIndexerT batch_indexer;
    LhsIndexerT lhs_indexer;
    RhsIndexerT rhs_indexer;
    ResIndexerT res_indexer;

public:
    GemmSmallFunctor(const lhsT *lhs_,
                     const rhsT *rhs_,
                     resT *res_,
                     std::size_t batch_nelems_,
                     std::size_t n_,
                     std::size_t k_,
                     std::size_t m_,
                     std::size_t k_split_,
                     std::size_t k_chunk_,
                     std::size_t res_split_stride_,
                     bool accumulate_,
                     bool symmetric_,
                     const BatchIndexerT &batch_indexer_,
                     const LhsIndexerT &lhs_indexer_,
                     const RhsIndexerT &rhs_indexer_,
                     const ResIndexerT &res_indexer_)
        : lhs(lhs_), rhs(rhs_), res(res_), batch_nelems(batch_nelems_), n(n_),
          k(k_), m(m_), k_split(k_split_), k_chunk(k_chunk_),
          res_split_stride(res_split_stride_), accumulate(accumulate_),
          symmetric(symmetric_), batch_indexer(batch_indexer_),
          lhs_indexer(lhs_indexer_), rhs_indexer(rhs_indexer_),
          res_indexer(res_indexer_)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        const std::size_t gid = it.get_global_linear_id();
        const std::size_t nm = n * m;
        const std::size_t per_batch = nm * k_split;
        const std::size_t batch_id = gid / per_batch;
        if (batch_id >= batch_nelems) {
            return;
        }
        std::size_t rem = gid - batch_id * per_batch;
        const std::size_t block_s = rem / nm;
        rem -= block_s * nm;
        // j moves fastest so that neighbouring work-items share lhs(i, s)
        const std::size_t i = rem / m;
        const std::size_t j = rem - i * m;

        if (symmetric && i > j) {
            return;
        }

        const auto &offsets = batch_indexer(batch_id);
        const lhsT *lhs_p = lhs + offsets.get_first_offset();
        const rhsT *rhs_p = rhs + offsets.get_second_offset();

        const std::size_t s_begin = block_s * k_chunk;
        const std::size_t s_end = std::min(k, s_begin + k_chunk);

        resT acc(0);
        for (std::size_t s = s_begin; s < s_end; ++s) {
            acc += static_cast<resT>(lhs_p[lhs_indexer(i, s)]) *
                   static_cast<resT>(rhs_p[rhs_indexer(s, j)]);
        }

        resT *res_p = res + offsets.get_third_offset();
        gemm_detail::store_result(res_p, block_s, res_split_stride, accumulate,
                                  res_indexer(i, j), acc);
        if (symmetric && i < j) {
            gemm_detail::store_result(res_p, block_s, res_split_stride,
                                      accumulate, res_indexer(j, i), acc);
        }
    }
};

/*! @brief Register-tiled GEMM.
 *
 * A work-group computes a tile of (wg_delta_n * wi_delta_n) rows and
 * (wg_delta_m * wi_delta_m_vecs * m_vec_size) columns of the result over one
 * chunk of k. Tiles of lhs and rhs, `wi_delta_k` deep in k, are staged in
 * SLM. Work-items of a sub-group map to adjacent rows of the tile, so that
 * reads of the lhs tile from SLM are conflict-free and reads of the rhs tile
 * are broadcasts.
 *
 * SLM tiles are populated with loads coalesced along whichever dimension of
 * the operand has the smaller stride (`lhs_row_major`, `rhs_row_major`).
 *
 * With `symmetric`, tiles strictly below the diagonal exit immediately, and
 * the remaining ones write res(i, j) for i <= j and mirror it to res(j, i).
 */
template <typename lhsT,
          typename rhsT,
          typename resT,
          typename LocAccT1,
          typename LocAccT2,
          typename BatchIndexerT,
          typename LhsIndexerT,
          typename RhsIndexerT,
          typename ResIndexerT,
          std::uint32_t wi_delta_n,
          std::uint32_t wi_delta_m_vecs,
          std::uint32_t m_vec_size>
class GemmTileFunctor
{
private:
    const lhsT *lhs = nullptr;
    const rhsT *rhs = nullptr;
    resT *res = nullptr;
    LocAccT1 local_lhs_block;
    LocAccT2 local_rhs_block;
    std::size_t n = 0;
    std::size_t k = 0;
    std::size_t m = 0;
    std::size_t n_groups = 0;
    std::size_t m_groups = 0;
    std::size_t k_split = 0;
    std::size_t k_chunk = 0;
    std::size_t res_split_stride = 0;
    bool accumulate = false;
    std::uint32_t wg_delta_n = 0;
    std::uint32_t wg_delta_m = 0;
    std::uint32_t wi_delta_k = 0;
    bool lhs_row_major = true;
    bool rhs_row_major = true;
    bool symmetric = false;
    BatchIndexerT batch_indexer;
    LhsIndexerT lhs_indexer;
    RhsIndexerT rhs_indexer;
    ResIndexerT res_indexer;

public:
    GemmTileFunctor(const lhsT *lhs_,
                    const rhsT *rhs_,
                    resT *res_,
                    LocAccT1 local_lhs_block_,
                    LocAccT2 local_rhs_block_,
                    std::size_t n_,
                    std::size_t k_,
                    std::size_t m_,
                    std::size_t n_groups_,
                    std::size_t m_groups_,
                    std::size_t k_split_,
                    std::size_t k_chunk_,
                    std::size_t res_split_stride_,
                    bool accumulate_,
                    std::uint32_t wg_delta_n_,
                    std::uint32_t wg_delta_m_,
                    std::uint32_t wi_delta_k_,
                    bool lhs_row_major_,
                    bool rhs_row_major_,
                    bool symmetric_,
                    const BatchIndexerT &batch_indexer_,
                    const LhsIndexerT &lhs_indexer_,
                    const RhsIndexerT &rhs_indexer_,
                    const ResIndexerT &res_indexer_)
        : lhs(lhs_), rhs(rhs_), res(res_), local_lhs_block(local_lhs_block_),
          local_rhs_block(local_rhs_block_), n(n_), k(k_), m(m_),
          n_groups(n_groups_), m_groups(m_groups_), k_split(k_split_),
          k_chunk(k_chunk_), res_split_stride(res_split_stride_),
          accumulate(accumulate_), wg_delta_n(wg_delta_n_),
          wg_delta_m(wg_delta_m_), wi_delta_k(wi_delta_k_),
          lhs_row_major(lhs_row_major_), rhs_row_major(rhs_row_major_),
          symmetric(symmetric_), batch_indexer(batch_indexer_),
          lhs_indexer(lhs_indexer_), rhs_indexer(rhs_indexer_),
          res_indexer(res_indexer_)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        static constexpr resT zero_(0);
        static constexpr std::uint32_t wi_total_delta_m =
            wi_delta_m_vecs * m_vec_size;

        const std::size_t tiles = n_groups * m_groups;
        const std::size_t groups_per_batch = tiles * k_split;
        const std::size_t batch_id =
            it.get_group_linear_id() / groups_per_batch;
        std::size_t gr_id =
            it.get_group_linear_id() - batch_id * groups_per_batch;
        // block_i moves fastest, then block_j, then block_s
        const std::size_t block_s = gr_id / tiles;
        gr_id -= block_s * tiles;
        const std::size_t block_j = gr_id / n_groups;
        const std::size_t block_i = gr_id - block_j * n_groups;

        const std::uint32_t tile_n = wg_delta_n * wi_delta_n;
        // padding the rows of the lhs tile avoids SLM bank conflicts when it
        // is populated along k
        const std::uint32_t lhs_pitch = tile_n + 1;
        const std::uint32_t tile_m = wg_delta_m * wi_total_delta_m;
        const std::uint32_t tile_m_vecs = wg_delta_m * wi_delta_m_vecs;

        // Coordinates of the block of C the work-group works on
        const std::size_t i = block_i * tile_n;
        const std::size_t j = block_j * tile_m;

        if (symmetric && i >= j + tile_m) {
            // every element of the tile is below the diagonal
            return;
        }

        // Assumption: lws == wg_delta_n * wg_delta_m
        const std::uint32_t lid = it.get_local_linear_id();
        const std::uint32_t lws = it.get_local_range(0);
        // 0 <= local_j < wg_delta_m
        const std::uint32_t local_j = lid / wg_delta_n;
        // sub-group lanes map to adjacent local_i
        const std::uint32_t local_i = lid - local_j * wg_delta_n;

        const auto &offsets = batch_indexer(batch_id);
        const lhsT *lhs_p = lhs + offsets.get_first_offset();
        const rhsT *rhs_p = rhs + offsets.get_second_offset();

        using slmA_t = typename LocAccT1::value_type;
        using slmB_t = typename LocAccT2::value_type;

        static constexpr std::uint32_t C_size = wi_delta_n * wi_delta_m_vecs;
        std::array<slmB_t, C_size> private_C;
#pragma unroll
        for (std::uint32_t idx = 0; idx < C_size; ++idx) {
            private_C[idx] = slmB_t(zero_);
        }

        const std::size_t s_begin = block_s * k_chunk;
        const std::size_t s_end = std::min(k, s_begin + k_chunk);

        for (std::size_t s = s_begin; s < s_end; s += wi_delta_k) {
            // populate local_lhs_block<resT> (wi_delta_k, lhs_pitch)
            for (std::uint32_t vid = lid; vid < wi_delta_k * tile_n;
                 vid += lws) {
                std::uint32_t v_i, v_s;
                if (lhs_row_major) {
                    v_i = vid / wi_delta_k;
                    v_s = vid - v_i * wi_delta_k;
                }
                else {
                    v_s = vid / tile_n;
                    v_i = vid - v_s * tile_n;
                }

                const std::size_t g_i = i + v_i;
                const std::size_t g_s = s + v_s;

                local_lhs_block[v_s * lhs_pitch + v_i] =
                    (g_i < n && g_s < s_end)
                        ? static_cast<resT>(lhs_p[lhs_indexer(g_i, g_s)])
                        : zero_;
            }

            // populate local_rhs_block<vec<resT, m_vec_size>>
            // (wi_delta_k, tile_m_vecs)
            for (std::uint32_t vid = lid; vid < local_rhs_block.size();
                 vid += lws) {
                std::uint32_t v_j, v_s;
                if (rhs_row_major) {
                    v_s = vid / tile_m_vecs;
                    v_j = vid - v_s * tile_m_vecs;
                }
                else {
                    v_j = vid / wi_delta_k;
                    v_s = vid - v_j * wi_delta_k;
                }

                const std::size_t g_j = j + v_j * m_vec_size;
                const std::size_t g_s = s + v_s;
                const std::uint32_t mapped_vid = v_s * tile_m_vecs + v_j;

                if constexpr (m_vec_size == 1) {
                    local_rhs_block[mapped_vid] =
                        (g_j < m && g_s < s_end)
                            ? static_cast<resT>(rhs_p[rhs_indexer(g_s, g_j)])
                            : zero_;
                }
                else {
                    slmB_t vec{};
#pragma unroll
                    for (std::uint32_t lane_id = 0; lane_id < m_vec_size;
                         ++lane_id) {
                        const std::size_t g_j1 = g_j + lane_id;
                        vec[lane_id] = (g_j1 < m && g_s < s_end)
                                           ? static_cast<resT>(
                                                 rhs_p[rhs_indexer(g_s, g_j1)])
                                           : zero_;
                    }
                    local_rhs_block[mapped_vid] = vec;
                }
            }

            sycl::group_barrier(it.get_group(), sycl::memory_scope::work_group);

            for (std::uint32_t pr_k = 0; pr_k < wi_delta_k; ++pr_k) {
                std::array<slmA_t, wi_delta_n> pr_lhs{};
#pragma unroll
                for (std::uint32_t pr_i = 0; pr_i < wi_delta_n; ++pr_i) {
                    pr_lhs[pr_i] = local_lhs_block[pr_k * lhs_pitch + local_i +
                                                   pr_i * wg_delta_n];
                }

                std::array<slmB_t, wi_delta_m_vecs> pr_rhs{};
#pragma unroll
                for (std::uint32_t pr_j = 0; pr_j < wi_delta_m_vecs; ++pr_j) {
                    pr_rhs[pr_j] = local_rhs_block[pr_k * tile_m_vecs +
                                                   local_j + pr_j * wg_delta_m];
                }

#pragma unroll
                for (std::uint32_t pr_i = 0; pr_i < wi_delta_n; ++pr_i) {
#pragma unroll
                    for (std::uint32_t pr_j = 0; pr_j < wi_delta_m_vecs;
                         ++pr_j) {
                        private_C[pr_i * wi_delta_m_vecs + pr_j] +=
                            pr_lhs[pr_i] * pr_rhs[pr_j];
                    }
                }
            }

            if constexpr (std::is_same_v<resT, bool>) {
                // bool tiles hold 0/1 bytes: saturate once per k-slab, which
                // cannot overflow since wi_delta_k < 255
#pragma unroll
                for (std::uint32_t idx = 0; idx < C_size; ++idx) {
                    private_C[idx] = sycl::min(private_C[idx], slmB_t(1));
                }
            }

            sycl::group_barrier(it.get_group(), sycl::memory_scope::work_group);
        }

        resT *res_p = res + offsets.get_third_offset();

#pragma unroll
        for (std::uint32_t pr_i = 0; pr_i < wi_delta_n; ++pr_i) {
            const std::size_t out_i = i + local_i + pr_i * wg_delta_n;
            if (out_i >= n) {
                continue;
            }
#pragma unroll
            for (std::uint32_t pr_j = 0; pr_j < wi_delta_m_vecs; ++pr_j) {
                const std::size_t out_j =
                    j + (local_j + pr_j * wg_delta_m) * m_vec_size;
#pragma unroll
                for (std::uint32_t lane_id = 0; lane_id < m_vec_size;
                     ++lane_id) {
                    const std::size_t out_j1 = out_j + lane_id;
                    if (out_j1 >= m || (symmetric && out_i > out_j1)) {
                        continue;
                    }
                    resT val;
                    if constexpr (m_vec_size == 1) {
                        val = private_C[pr_i * wi_delta_m_vecs + pr_j];
                    }
                    else {
                        val = private_C[pr_i * wi_delta_m_vecs + pr_j][lane_id];
                    }
                    gemm_detail::store_result(res_p, block_s, res_split_stride,
                                              accumulate,
                                              res_indexer(out_i, out_j1), val);
                    if (symmetric && out_i < out_j1) {
                        gemm_detail::store_result(
                            res_p, block_s, res_split_stride, accumulate,
                            res_indexer(out_j1, out_i), val);
                    }
                }
            }
        }
    }
};

struct GemmTileHyperParameters
{
private:
    std::uint32_t wi_delta_n = 2;
    std::uint32_t wi_delta_m_vecs = 4;
    std::uint32_t m_vec_size = 1;

public:
    constexpr GemmTileHyperParameters(std::uint32_t wi_delta_n_,
                                      std::uint32_t wi_delta_m_vecs_,
                                      std::uint32_t m_vec_size_)
        : wi_delta_n(wi_delta_n_), wi_delta_m_vecs(wi_delta_m_vecs_),
          m_vec_size(m_vec_size_)
    {
    }

    constexpr std::uint32_t get_wi_delta_n() const { return wi_delta_n; }
    constexpr std::uint32_t get_wi_delta_m_vecs() const
    {
        return wi_delta_m_vecs;
    }
    constexpr std::uint32_t get_m_vec_size() const { return m_vec_size; }
};

/*! @brief Per work-item register tile, chosen so that each work-item holds
 * 64 bytes of the result. */
template <typename resT>
struct GemmTileHyperParametersSelector
{
    constexpr GemmTileHyperParametersSelector() {}

    constexpr GemmTileHyperParameters get() const
    {
        if constexpr (sizeof(resT) == 1) {
            // 1 * 8 * 2 * 4 == 64
            return GemmTileHyperParameters(8, 2, 4);
        }
        else if constexpr (sizeof(resT) == 2) {
            // 2 * 4 * 2 * 4 == 64
            return GemmTileHyperParameters(4, 2, 4);
        }
        else if constexpr (sizeof(resT) == 4) {
            // 4 * 4 * 1 * 4 == 64
            return GemmTileHyperParameters(4, 1, 4);
        }
        else if constexpr (sizeof(resT) == 8) {
            // 8 * 2 * 1 * 4 == 64
            if constexpr (std::is_same_v<resT, std::complex<float>>) {
                return GemmTileHyperParameters(2, 4, 1);
            }
            else {
                return GemmTileHyperParameters(2, 1, 4);
            }
        }
        else {
            // 16 * 2 * 2 * 1 == 64
            return GemmTileHyperParameters(2, 2, 1);
        }
    }
};

/*! @brief Sums k_split partial results stored contiguously, as
 * [k_split][batch][n][m], and writes them to the result. */
template <typename resT, typename BatchIndexerT, typename ResIndexerT>
class GemmPartialsReduceFunctor
{
private:
    const resT *partials = nullptr;
    resT *res = nullptr;
    std::size_t nelems = 0;
    std::size_t n = 0;
    std::size_t m = 0;
    std::size_t k_split = 0;
    BatchIndexerT batch_indexer;
    ResIndexerT res_indexer;

public:
    GemmPartialsReduceFunctor(const resT *partials_,
                              resT *res_,
                              std::size_t nelems_,
                              std::size_t n_,
                              std::size_t m_,
                              std::size_t k_split_,
                              const BatchIndexerT &batch_indexer_,
                              const ResIndexerT &res_indexer_)
        : partials(partials_), res(res_), nelems(nelems_), n(n_), m(m_),
          k_split(k_split_), batch_indexer(batch_indexer_),
          res_indexer(res_indexer_)
    {
    }

    void operator()(sycl::id<1> id) const
    {
        const std::size_t gid = id[0];
        resT acc = partials[gid];
        for (std::size_t c = 1; c < k_split; ++c) {
            acc += partials[c * nelems + gid];
        }

        const std::size_t nm = n * m;
        const std::size_t batch_id = gid / nm;
        const std::size_t rem = gid - batch_id * nm;
        const std::size_t i = rem / m;
        const std::size_t j = rem - i * m;
        res[batch_indexer(batch_id).get_third_offset() + res_indexer(i, j)] =
            acc;
    }
};

template <typename T1, typename T2, typename T3>
class gemm_zero_krn;

template <typename T1, typename T2, typename T3>
class gemm_partials_reduce_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          typename T7,
          bool>
class gemv_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          typename T7>
class gemm_small_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          typename T7,
          std::uint32_t p1,
          std::uint32_t p2,
          std::uint32_t p3>
class gemm_tile_krn;

namespace gemm_detail
{

/*! @brief res = 0, used when k == 0 */
template <typename resTy, typename BatchIndexerT, typename ResIndexerT>
sycl::event gemm_zero(sycl::queue &exec_q,
                      resTy *res_tp,
                      std::size_t batch_nelems,
                      std::size_t n,
                      std::size_t m,
                      const BatchIndexerT &batch_indexer,
                      const ResIndexerT &res_indexer,
                      const std::vector<sycl::event> &depends)
{
    return exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        using KernelName =
            class gemm_zero_krn<resTy, BatchIndexerT, ResIndexerT>;
        cgh.parallel_for<KernelName>(
            sycl::range<1>(batch_nelems * n * m), [=](sycl::id<1> id) {
                const std::size_t nm = n * m;
                const std::size_t batch_id = id[0] / nm;
                const std::size_t rem = id[0] - batch_id * nm;
                const std::size_t i = rem / m;
                const std::size_t j = rem - i * m;
                res_tp[batch_indexer(batch_id).get_third_offset() +
                       res_indexer(i, j)] = resTy(0);
            });
    });
}

/*! @brief Run `launch` over the full k range if `k_split == 1`, or over
 * `k_split` chunks of k whose partial results are combined.
 *
 * `launch(res_ptr, batch_indexer, res_indexer, res_split_stride, accumulate,
 * depends)` must submit a kernel passing the partial result of chunk
 * `block_s` to gemm_detail::store_result at displacement
 * `batch_indexer(b).third + res_indexer(i, j)` from `res_ptr`.
 *
 * With `use_atomics`, which the caller sets if the device supports atomics of
 * resTy on the allocation of the result, the chunks atomically add to the
 * zeroed result. Otherwise they write to a temporary buffer, which a second
 * kernel reduces into the result.
 */
template <typename resTy,
          typename BatchIndexerT,
          typename ResIndexerT,
          typename LaunchFnT>
sycl::event gemm_with_k_split(sycl::queue &exec_q,
                              resTy *res_tp,
                              std::size_t batch_nelems,
                              std::size_t n,
                              std::size_t m,
                              std::size_t k_split,
                              const BatchIndexerT &batch_indexer,
                              const ResIndexerT &res_indexer,
                              bool use_atomics,
                              LaunchFnT &&launch,
                              const std::vector<sycl::event> &depends)
{
    if (k_split == 1) {
        return launch(res_tp, batch_indexer, res_indexer, std::size_t(0), false,
                      depends);
    }

    if constexpr (gemm_atomic_accumulate_v<resTy>) {
        if (use_atomics) {
            // cheaper than a temporary and a reduction kernel
            const sycl::event zero_ev =
                gemm_zero(exec_q, res_tp, batch_nelems, n, m, batch_indexer,
                          res_indexer, depends);
            return launch(res_tp, batch_indexer, res_indexer, std::size_t(0),
                          true, {zero_ev});
        }
    }

    const std::size_t nelems = batch_nelems * n * m;
    auto partials_owner = dpnp::tensor::alloc_utils::smart_malloc_device<resTy>(
        k_split * nelems, exec_q);
    resTy *partials = partials_owner.get();

    const MatrixStridedIndexer partials_indexer(static_cast<ssize_t>(m), 1);
    sycl::event partials_ev = launch(
        partials, batch_indexer.for_partials(static_cast<ssize_t>(n * m)),
        partials_indexer, nelems, false, depends);

    sycl::event red_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(partials_ev);
        using KernelName =
            class gemm_partials_reduce_krn<resTy, BatchIndexerT, ResIndexerT>;
        cgh.parallel_for<KernelName>(
            sycl::range<1>(nelems),
            GemmPartialsReduceFunctor<resTy, BatchIndexerT, ResIndexerT>(
                partials, res_tp, nelems, n, m, k_split, batch_indexer,
                res_indexer));
    });

    return dpnp::tensor::alloc_utils::async_smart_free(exec_q, {red_ev},
                                                       partials_owner);
}

template <typename lhsTy,
          typename rhsTy,
          typename resTy,
          typename BatchIndexerT,
          typename LhsIndexerT,
          typename RhsIndexerT,
          typename ResIndexerT>
sycl::event gemv_impl(sycl::queue &exec_q,
                      const lhsTy *lhs_tp,
                      const rhsTy *rhs_tp,
                      resTy *res_tp,
                      std::size_t batch_nelems,
                      std::size_t n,
                      std::size_t k,
                      std::size_t m,
                      const BatchIndexerT &batch_indexer,
                      const LhsIndexerT &lhs_indexer,
                      const RhsIndexerT &rhs_indexer,
                      const ResIndexerT &res_indexer,
                      bool lanes_along_k,
                      bool use_atomics,
                      const std::vector<sycl::event> &depends)
{
    const sycl::device &dev = exec_q.get_device();

    // y(r) = sum_s M(r, s) v(s), with M = rhs^T if n == 1
    const bool transposed = (m != 1);
    const std::size_t rows = transposed ? m : n;

    const std::size_t max_wg =
        dev.get_info<sycl::info::device::max_work_group_size>();
    const std::uint32_t lws =
        static_cast<std::uint32_t>(std::min<std::size_t>(max_wg, 256));

    std::uint32_t wg_lanes, wg_rows;
    if (lanes_along_k) {
        wg_lanes = static_cast<std::uint32_t>(
            std::min<std::size_t>(round_up_pow2(k), std::min(lws, 64u)));
        wg_rows = lws / wg_lanes;
    }
    else {
        wg_rows = static_cast<std::uint32_t>(
            std::min<std::size_t>(round_up_pow2(rows), std::min(lws, 64u)));
        wg_lanes = lws / wg_rows;
    }

    const std::size_t row_groups = ceil_div<std::size_t>(rows, wg_rows);
    const gemm_detail::KSplit split = gemm_detail::choose_k_split(
        dev, batch_nelems * row_groups * lws, k, std::size_t(wg_lanes) * 16);
    const std::size_t k_split = split.k_split;
    const std::size_t k_chunk = split.k_chunk;

    auto launch = [&](resTy *out_tp, const auto &out_batch_indexer,
                      const auto &out_indexer, std::size_t out_split_stride,
                      bool accumulate, const std::vector<sycl::event> &deps) {
        using OutBatchIndexerT = std::decay_t<decltype(out_batch_indexer)>;
        using OutIndexerT = std::decay_t<decltype(out_indexer)>;

        return exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(deps);

            using LocAccT = sycl::local_accessor<resTy, 1>;
            LocAccT workspace(lws, cgh);

            const sycl::nd_range<1> ndRange(
                sycl::range<1>(batch_nelems * row_groups * k_split * lws),
                sycl::range<1>(lws));

            auto submit_for = [&](auto transposed_tag) {
                static constexpr bool transposed_v =
                    decltype(transposed_tag)::value;
                using KernelName =
                    class gemv_krn<lhsTy, rhsTy, resTy, OutBatchIndexerT,
                                   LhsIndexerT, RhsIndexerT, OutIndexerT,
                                   transposed_v>;
                using FunctorT =
                    GemvFunctor<lhsTy, rhsTy, resTy, LocAccT, OutBatchIndexerT,
                                LhsIndexerT, RhsIndexerT, OutIndexerT,
                                transposed_v>;
                cgh.parallel_for<KernelName>(
                    ndRange,
                    FunctorT(lhs_tp, rhs_tp, out_tp, workspace, rows, k,
                             row_groups, k_split, k_chunk, out_split_stride,
                             accumulate, wg_rows, wg_lanes, lanes_along_k,
                             out_batch_indexer, lhs_indexer, rhs_indexer,
                             out_indexer));
            };
            if (transposed) {
                submit_for(std::true_type{});
            }
            else {
                submit_for(std::false_type{});
            }
        });
    };

    return gemm_with_k_split(exec_q, res_tp, batch_nelems, n, m, k_split,
                             batch_indexer, res_indexer, use_atomics, launch,
                             depends);
}

template <typename lhsTy,
          typename rhsTy,
          typename resTy,
          typename BatchIndexerT,
          typename LhsIndexerT,
          typename RhsIndexerT,
          typename ResIndexerT>
sycl::event gemm_small_impl(sycl::queue &exec_q,
                            const lhsTy *lhs_tp,
                            const rhsTy *rhs_tp,
                            resTy *res_tp,
                            std::size_t batch_nelems,
                            std::size_t n,
                            std::size_t k,
                            std::size_t m,
                            const BatchIndexerT &batch_indexer,
                            const LhsIndexerT &lhs_indexer,
                            const RhsIndexerT &rhs_indexer,
                            const ResIndexerT &res_indexer,
                            bool symmetric,
                            bool use_atomics,
                            const std::vector<sycl::event> &depends)
{
    const sycl::device &dev = exec_q.get_device();
    const gemm_detail::KSplit split =
        gemm_detail::choose_k_split(dev, batch_nelems * n * m, k, 64);
    const std::size_t k_split = split.k_split;
    const std::size_t k_chunk = split.k_chunk;

    auto launch = [&](resTy *out_tp, const auto &out_batch_indexer,
                      const auto &out_indexer, std::size_t out_split_stride,
                      bool accumulate, const std::vector<sycl::event> &deps) {
        using OutBatchIndexerT = std::decay_t<decltype(out_batch_indexer)>;
        using OutIndexerT = std::decay_t<decltype(out_indexer)>;

        return exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(deps);
            using KernelName =
                class gemm_small_krn<lhsTy, rhsTy, resTy, OutBatchIndexerT,
                                     LhsIndexerT, RhsIndexerT, OutIndexerT>;
            using FunctorT =
                GemmSmallFunctor<lhsTy, rhsTy, resTy, OutBatchIndexerT,
                                 LhsIndexerT, RhsIndexerT, OutIndexerT>;
            static constexpr std::size_t lws = 128;
            const std::size_t gws =
                ceil_div(batch_nelems * k_split * n * m, lws) * lws;
            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(sycl::range<1>(gws), sycl::range<1>(lws)),
                FunctorT(lhs_tp, rhs_tp, out_tp, batch_nelems, n, k, m, k_split,
                         k_chunk, out_split_stride, accumulate, symmetric,
                         out_batch_indexer, lhs_indexer, rhs_indexer,
                         out_indexer));
        });
    };

    return gemm_with_k_split(exec_q, res_tp, batch_nelems, n, m, k_split,
                             batch_indexer, res_indexer, use_atomics, launch,
                             depends);
}

/*! @brief Largest wg_delta_m (number of sub-groups) and wi_delta_k for which
 * both SLM tiles fit in `slm_byte_size`. */
template <typename T, std::uint32_t wi_delta_n, std::uint32_t wi_delta_m>
std::tuple<std::uint32_t, std::uint32_t>
    get_wg_delta_m_and_wi_delta_k(const std::size_t slm_byte_size,
                                  const std::uint32_t wg_delta_n,
                                  const std::uint32_t suggested_wg_delta_m)
{
    std::uint32_t wg_delta_m = suggested_wg_delta_m;
    std::uint32_t wi_delta_k = 0;

    for (std::uint32_t it = 0; !wi_delta_k && (it < 5); ++it) {
        const std::size_t slm_max_rows =
            slm_byte_size /
            ((wg_delta_n * wi_delta_n + 1 + wg_delta_m * wi_delta_m) *
             sizeof(T));

        wi_delta_k =
            (slm_max_rows >= 64)
                ? 64
                : ((slm_max_rows >= 32)
                       ? 32
                       : (slm_max_rows >= 16 ? 16
                                             : 8 * static_cast<std::uint32_t>(
                                                       slm_max_rows / 8)));
        if (!wi_delta_k) {
            wg_delta_m = std::max<std::uint32_t>(wg_delta_m / 2, 1);
        }
    }

    if (!wi_delta_k) {
        throw std::runtime_error("Insufficient resources");
    }

    return std::make_tuple(wg_delta_m, wi_delta_k);
}

template <typename lhsTy, typename rhsTy, typename resTy>
struct GemmTileConfig
{
    // bool tiles are held as 0/1 bytes and accumulated as integers, which
    // avoids the costly arithmetic on sycl::vec<bool>
    using value_type =
        std::conditional_t<std::is_same_v<resTy, bool>, std::uint8_t, resTy>;

    static constexpr GemmTileHyperParametersSelector<resTy> selector{};
    static constexpr auto hyper_params = selector.get();

    static constexpr std::uint32_t wi_delta_n = hyper_params.get_wi_delta_n();
    static constexpr std::uint32_t wi_delta_m_vecs =
        hyper_params.get_wi_delta_m_vecs();
    static constexpr std::uint32_t m_vec_size = hyper_params.get_m_vec_size();
    static constexpr std::uint32_t wi_total_delta_m =
        wi_delta_m_vecs * m_vec_size;

    using slmB_t =
        typename std::conditional<m_vec_size == 1,
                                  value_type,
                                  sycl::vec<value_type, m_vec_size>>::type;
    using LocAccT1 = sycl::local_accessor<value_type, 1>;
    using LocAccT2 = sycl::local_accessor<slmB_t, 1>;

    template <typename BatchIndexerT,
              typename LhsIndexerT,
              typename RhsIndexerT,
              typename ResIndexerT>
    using KernelName = class gemm_tile_krn<lhsTy,
                                           rhsTy,
                                           resTy,
                                           BatchIndexerT,
                                           LhsIndexerT,
                                           RhsIndexerT,
                                           ResIndexerT,
                                           wi_delta_n,
                                           wi_delta_m_vecs,
                                           m_vec_size>;

    template <typename BatchIndexerT,
              typename LhsIndexerT,
              typename RhsIndexerT,
              typename ResIndexerT>
    using FunctorT = GemmTileFunctor<lhsTy,
                                     rhsTy,
                                     resTy,
                                     LocAccT1,
                                     LocAccT2,
                                     BatchIndexerT,
                                     LhsIndexerT,
                                     RhsIndexerT,
                                     ResIndexerT,
                                     wi_delta_n,
                                     wi_delta_m_vecs,
                                     m_vec_size>;
};

/*! @brief Work-group geometry of the tile kernel */
struct GemmTileGeometry
{
    std::uint32_t wg_delta_n;
    std::uint32_t wg_delta_m;
    std::uint32_t wi_delta_k;
    std::uint32_t tile_n;
    std::uint32_t tile_m;
};

/*! @brief Work-group geometry of the tile kernel for the given device,
 * before it is adjusted to the size of the result.
 *
 * Querying the kernel bundle costs tens of microseconds, comparable to the
 * run time of small products, so the result is cached per device.
 */
template <typename ConfigT, typename KernelNameT>
GemmTileGeometry gemm_tile_device_geometry(sycl::queue &exec_q)
{
    static std::mutex cache_mutex;
    static std::vector<std::pair<sycl::device, GemmTileGeometry>> cache;

    const sycl::device &dev = exec_q.get_device();
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        for (const auto &entry : cache) {
            if (entry.first == dev) {
                return entry.second;
            }
        }
    }

    const auto &kernel_id = sycl::get_kernel_id<KernelNameT>();

    const auto &ctx = exec_q.get_context();
    auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
        ctx, {dev}, {kernel_id});
    auto krn = kb.get_kernel(kernel_id);

    const std::uint32_t max_sg_size = krn.template get_info<
        sycl::info::kernel_device_specific::max_sub_group_size>(dev);
    const std::size_t k_wg_sz = krn.template get_info<
        sycl::info::kernel_device_specific::work_group_size>(dev);

    // Limit work-group size
    static constexpr std::size_t wg_sz_limit(2048);
    const std::size_t max_wg_sz = std::min(wg_sz_limit, k_wg_sz);
    const std::uint32_t max_subgroups_per_wg =
        static_cast<std::uint32_t>(max_wg_sz / max_sg_size);

    const std::size_t reserved_slm_byte_size = 512;
    const std::size_t slm_byte_size =
        dev.get_info<sycl::info::device::local_mem_size>();

    const std::uint32_t wg_delta_n = max_sg_size;
    auto [wg_delta_m, wi_delta_k] =
        get_wg_delta_m_and_wi_delta_k<typename ConfigT::value_type,
                                      ConfigT::wi_delta_n,
                                      ConfigT::wi_total_delta_m>(
            slm_byte_size - reserved_slm_byte_size, wg_delta_n,
            max_subgroups_per_wg);

    // Cap the work-group at 512 work-items: larger groups leave too few groups
    // in flight and measured ~20% slower for large products on Intel GPUs.
    // The cap is applied after sizing wi_delta_k, so that the smaller group
    // uses proportionally less SLM rather than a deeper k-slab, which lowers
    // occupancy and measured up to 2x slower.
    static constexpr std::uint32_t max_tile_wg_size = 512;
    while (wg_delta_m > 1 && wg_delta_n * wg_delta_m > max_tile_wg_size) {
        wg_delta_m /= 2;
    }

    const GemmTileGeometry geom{wg_delta_n, wg_delta_m, wi_delta_k,
                                wg_delta_n * ConfigT::wi_delta_n,
                                wg_delta_m * ConfigT::wi_total_delta_m};

    std::lock_guard<std::mutex> lock(cache_mutex);
    cache.emplace_back(dev, geom);
    return geom;
}

template <typename ConfigT, typename KernelNameT>
GemmTileGeometry gemm_tile_geometry(sycl::queue &exec_q, std::size_t m)
{
    GemmTileGeometry geom =
        gemm_tile_device_geometry<ConfigT, KernelNameT>(exec_q);

    // do not make the tile much wider than the result
    while (geom.wg_delta_m > 1 &&
           std::size_t(geom.wg_delta_m / 2) * ConfigT::wi_total_delta_m >= m) {
        geom.wg_delta_m /= 2;
    }
    geom.tile_m = geom.wg_delta_m * ConfigT::wi_total_delta_m;

    return geom;
}

template <typename lhsTy,
          typename rhsTy,
          typename resTy,
          typename BatchIndexerT,
          typename LhsIndexerT,
          typename RhsIndexerT,
          typename ResIndexerT>
sycl::event gemm_tile_impl(sycl::queue &exec_q,
                           const lhsTy *lhs_tp,
                           const rhsTy *rhs_tp,
                           resTy *res_tp,
                           std::size_t batch_nelems,
                           std::size_t n,
                           std::size_t k,
                           std::size_t m,
                           const BatchIndexerT &batch_indexer,
                           const LhsIndexerT &lhs_indexer,
                           const RhsIndexerT &rhs_indexer,
                           const ResIndexerT &res_indexer,
                           const GemmTileGeometry &geom,
                           bool lhs_row_major,
                           bool rhs_row_major,
                           bool symmetric,
                           bool use_atomics,
                           const std::vector<sycl::event> &depends)
{
    using ConfigT = GemmTileConfig<lhsTy, rhsTy, resTy>;

    const sycl::device &dev = exec_q.get_device();

    const std::uint32_t lws = geom.wg_delta_n * geom.wg_delta_m;
    const std::size_t n_groups = ceil_div<std::size_t>(n, geom.tile_n);
    const std::size_t m_groups = ceil_div<std::size_t>(m, geom.tile_m);

    std::size_t tiles = n_groups * m_groups;
    if (symmetric) {
        // about half the tiles exit immediately
        tiles = std::max<std::size_t>(tiles / 2, 1);
    }
    // A split costs a second kernel, and without atomics also a temporary
    // and an asynchronous free; on Intel GPUs it did not pay off for chunks
    // shorter than this
    static constexpr std::size_t min_k_chunk = 256;
    const gemm_detail::KSplit split = gemm_detail::choose_k_split(
        dev, batch_nelems * tiles * lws, k, min_k_chunk, geom.wi_delta_k);
    const std::size_t k_split = split.k_split;
    const std::size_t k_chunk = split.k_chunk;

    auto launch = [&](resTy *out_tp, const auto &out_batch_indexer,
                      const auto &out_indexer, std::size_t out_split_stride,
                      bool accumulate, const std::vector<sycl::event> &deps) {
        using OutBatchIndexerT = std::decay_t<decltype(out_batch_indexer)>;
        using OutIndexerT = std::decay_t<decltype(out_indexer)>;
        using KernelName =
            typename ConfigT::template KernelName<OutBatchIndexerT, LhsIndexerT,
                                                  RhsIndexerT, OutIndexerT>;
        using FunctorT =
            typename ConfigT::template FunctorT<OutBatchIndexerT, LhsIndexerT,
                                                RhsIndexerT, OutIndexerT>;

        return exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(deps);

            typename ConfigT::LocAccT1 local_A_block(
                (std::size_t(geom.tile_n) + 1) * geom.wi_delta_k, cgh);
            typename ConfigT::LocAccT2 local_B_block(
                std::size_t(geom.wg_delta_m) * ConfigT::wi_delta_m_vecs *
                    geom.wi_delta_k,
                cgh);

            const sycl::nd_range<1> ndRange(
                sycl::range<1>(batch_nelems * n_groups * m_groups * k_split *
                               lws),
                sycl::range<1>(lws));

            cgh.parallel_for<KernelName>(
                ndRange,
                FunctorT(lhs_tp, rhs_tp, out_tp, local_A_block, local_B_block,
                         n, k, m, n_groups, m_groups, k_split, k_chunk,
                         out_split_stride, accumulate, geom.wg_delta_n,
                         geom.wg_delta_m, geom.wi_delta_k, lhs_row_major,
                         rhs_row_major, symmetric, out_batch_indexer,
                         lhs_indexer, rhs_indexer, out_indexer));
        });
    };

    return gemm_with_k_split(exec_q, res_tp, batch_nelems, n, m, k_split,
                             batch_indexer, res_indexer, use_atomics, launch,
                             depends);
}

/*! @brief Select and launch the kernel computing a batch of matrix products.
 *
 * @param symmetric    lhs[b](i, s) and rhs[b](s, i) refer to the same
 *                     element, so that each res[b] is symmetric.
 * @param use_atomics  the device supports atomics of resTy on the allocation
 *                     of the result, see gemm_with_k_split.
 */
template <typename lhsTy,
          typename rhsTy,
          typename resTy,
          typename BatchIndexerT,
          typename LhsIndexerT,
          typename RhsIndexerT,
          typename ResIndexerT>
sycl::event gemm_driver(sycl::queue &exec_q,
                        const lhsTy *lhs_tp,
                        const rhsTy *rhs_tp,
                        resTy *res_tp,
                        std::size_t batch_nelems,
                        std::size_t n,
                        std::size_t k,
                        std::size_t m,
                        const BatchIndexerT &batch_indexer,
                        const LhsIndexerT &lhs_indexer,
                        const RhsIndexerT &rhs_indexer,
                        const ResIndexerT &res_indexer,
                        bool symmetric,
                        bool use_atomics,
                        const std::vector<sycl::event> &depends)
{
    if (k == 0) {
        return gemm_zero(exec_q, res_tp, batch_nelems, n, m, batch_indexer,
                         res_indexer, depends);
    }

    if (n == 1 || m == 1) {
        // coalesce loads of the matrix operand along k if it is contiguous
        // along k
        const bool lanes_along_k =
            (m == 1) ? lhs_indexer.row_major() : !rhs_indexer.row_major();
        return gemv_impl(exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
                         batch_indexer, lhs_indexer, rhs_indexer, res_indexer,
                         lanes_along_k, use_atomics, depends);
    }

    using ConfigT = GemmTileConfig<lhsTy, rhsTy, resTy>;
    using KernelName =
        typename ConfigT::template KernelName<BatchIndexerT, LhsIndexerT,
                                              RhsIndexerT, ResIndexerT>;
    const GemmTileGeometry geom =
        gemm_tile_geometry<ConfigT, KernelName>(exec_q, m);

    // Fraction of the tile kernel's work spent on elements of the result
    const double tile_efficiency =
        double(n * m) /
        double(ceil_div<std::size_t>(n, geom.tile_n) * geom.tile_n *
               ceil_div<std::size_t>(m, geom.tile_m) * geom.tile_m);

    // Below this the tile kernel mostly computes padding; the untiled kernel
    // wins (e.g. 16x16 results against a 64x16 tile, or 32x32 against 64x32)
    static constexpr double min_tile_efficiency = 0.3;
    if (tile_efficiency < min_tile_efficiency) {
        return gemm_small_impl(exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n,
                               k, m, batch_indexer, lhs_indexer, rhs_indexer,
                               res_indexer, symmetric, use_atomics, depends);
    }

    return gemm_tile_impl(
        exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m, batch_indexer,
        lhs_indexer, rhs_indexer, res_indexer, geom, lhs_indexer.row_major(),
        rhs_indexer.row_major(), symmetric, use_atomics, depends);
}

} // namespace gemm_detail

/*! @brief Layout of the batch dimensions of a batched matrix product.
 *
 * If `nd <= 1`, batch element b of lhs is at `lhs_offset + b * lhs_stride`,
 * and similarly for rhs and res. Otherwise the batch dimensions are
 * described by `packed_shape_strides`, a device-accessible array of
 * `4 * nd` elements: the shape followed by the strides of lhs, rhs and res.
 */
struct GemmBatchLayout
{
    int nd = 0;
    const ssize_t *packed_shape_strides = nullptr;
    ssize_t lhs_offset = 0;
    ssize_t rhs_offset = 0;
    ssize_t res_offset = 0;
    ssize_t lhs_stride = 0;
    ssize_t rhs_stride = 0;
    ssize_t res_stride = 0;
};

/*! @brief Row and column strides of a matrix, in elements */
struct GemmMatrixStrides
{
    ssize_t row_stride;
    ssize_t col_stride;
};

typedef sycl::event (*gemm_impl_fn_ptr_t)(
    sycl::queue &,
    const char *,              // lhs
    const char *,              // rhs
    char *,                    // res
    std::size_t,               // batch_nelems
    std::size_t,               // n
    std::size_t,               // k
    std::size_t,               // m
    const GemmBatchLayout &,   // batch dimensions
    const GemmMatrixStrides &, // lhs (n, k) strides
    const GemmMatrixStrides &, // rhs (k, m) strides
    const GemmMatrixStrides &, // res (n, m) strides
    bool,                      // res is symmetric, lhs == rhs^T
    bool,                      // atomics on res are supported
    const std::vector<sycl::event> &);

/*! @brief Batched matrix product of matrices with arbitrary row and column
 * strides. */
template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event gemm_impl(sycl::queue &exec_q,
                      const char *lhs_cp,
                      const char *rhs_cp,
                      char *res_cp,
                      std::size_t batch_nelems,
                      std::size_t n,
                      std::size_t k,
                      std::size_t m,
                      const GemmBatchLayout &batch,
                      const GemmMatrixStrides &lhs_strides,
                      const GemmMatrixStrides &rhs_strides,
                      const GemmMatrixStrides &res_strides,
                      bool symmetric,
                      bool use_atomics,
                      const std::vector<sycl::event> &depends = {})
{
    const lhsTy *lhs_tp = reinterpret_cast<const lhsTy *>(lhs_cp);
    const rhsTy *rhs_tp = reinterpret_cast<const rhsTy *>(rhs_cp);
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp);

    using gemm_detail::MatrixStridedIndexer;
    const MatrixStridedIndexer lhs_indexer(lhs_strides.row_stride,
                                           lhs_strides.col_stride);
    const MatrixStridedIndexer rhs_indexer(rhs_strides.row_stride,
                                           rhs_strides.col_stride);
    const MatrixStridedIndexer res_indexer(res_strides.row_stride,
                                           res_strides.col_stride);

    using dpnp::tensor::offset_utils::Strided1DIndexer;
    using dpnp::tensor::offset_utils::ThreeOffsets_CombinedIndexer;
    using dpnp::tensor::offset_utils::ThreeOffsets_StridedIndexer;

    if (batch.nd <= 1) {
        using BaseIndexerT =
            ThreeOffsets_CombinedIndexer<Strided1DIndexer, Strided1DIndexer,
                                         Strided1DIndexer>;
        const BaseIndexerT base_indexer(
            Strided1DIndexer(batch.lhs_offset, batch_nelems, batch.lhs_stride),
            Strided1DIndexer(batch.rhs_offset, batch_nelems, batch.rhs_stride),
            Strided1DIndexer(batch.res_offset, batch_nelems, batch.res_stride));
        const gemm_detail::GemmBatchIndexer<BaseIndexerT> batch_indexer(
            base_indexer);

        return gemm_detail::gemm_driver(exec_q, lhs_tp, rhs_tp, res_tp,
                                        batch_nelems, n, k, m, batch_indexer,
                                        lhs_indexer, rhs_indexer, res_indexer,
                                        symmetric, use_atomics, depends);
    }
    else {
        using BaseIndexerT = ThreeOffsets_StridedIndexer;
        const BaseIndexerT base_indexer(batch.nd, batch.lhs_offset,
                                        batch.rhs_offset, batch.res_offset,
                                        batch.packed_shape_strides);
        const gemm_detail::GemmBatchIndexer<BaseIndexerT> batch_indexer(
            base_indexer);

        return gemm_detail::gemm_driver(exec_q, lhs_tp, rhs_tp, res_tp,
                                        batch_nelems, n, k, m, batch_indexer,
                                        lhs_indexer, rhs_indexer, res_indexer,
                                        symmetric, use_atomics, depends);
    }
}

typedef sycl::event (*gemm_nd_impl_fn_ptr_t)(
    sycl::queue &,
    const char *,    // lhs
    const char *,    // rhs
    char *,          // res
    std::size_t,     // lhs_outer_nelems (n)
    std::size_t,     // inner_nelems (k)
    std::size_t,     // rhs_outer_nelems (m)
    int,             // inner nd
    int,             // lhs outer nd
    const ssize_t *, // lhs shape and strides
    int,             // rhs outer nd
    const ssize_t *, // rhs shape and strides
    int,             // res outer nd
    const ssize_t *, // res shape and strides
    bool,            // atomics on res are supported
    const std::vector<sycl::event> &);

/*! @brief Matrix product of operands whose rows or columns span several
 * dimensions that can not be collapsed into one, as for tensordot of
 * non-contiguous views. */
template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event gemm_nd_impl(sycl::queue &exec_q,
                         const char *lhs_cp,
                         const char *rhs_cp,
                         char *res_cp,
                         std::size_t n,
                         std::size_t k,
                         std::size_t m,
                         int inner_nd,
                         int lhs_outer_nd,
                         const ssize_t *lhs_shape_strides,
                         int rhs_outer_nd,
                         const ssize_t *rhs_shape_strides,
                         int res_outer_nd,
                         const ssize_t *res_shape_strides,
                         bool use_atomics,
                         const std::vector<sycl::event> &depends = {})
{
    const lhsTy *lhs_tp = reinterpret_cast<const lhsTy *>(lhs_cp);
    const rhsTy *rhs_tp = reinterpret_cast<const rhsTy *>(rhs_cp);
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp);

    using gemm_detail::MatrixFlatIndexer;
    const MatrixFlatIndexer lhs_indexer(inner_nd + lhs_outer_nd,
                                        lhs_shape_strides, k);
    const MatrixFlatIndexer rhs_indexer(inner_nd + rhs_outer_nd,
                                        rhs_shape_strides, m);
    const MatrixFlatIndexer res_indexer(res_outer_nd, res_shape_strides, m);

    using BaseIndexerT = dpnp::tensor::offset_utils::ThreeZeroOffsets_Indexer;
    const gemm_detail::GemmBatchIndexer<BaseIndexerT> batch_indexer(
        BaseIndexerT{});

    static constexpr std::size_t single_batch_nelems = 1;
    static constexpr bool symmetric = false;

    return gemm_detail::gemm_driver(exec_q, lhs_tp, rhs_tp, res_tp,
                                    single_batch_nelems, n, k, m, batch_indexer,
                                    lhs_indexer, rhs_indexer, res_indexer,
                                    symmetric, use_atomics, depends);
}

} // namespace dpnp::tensor::kernels
