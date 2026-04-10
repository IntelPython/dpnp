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
//===---------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "kernels/dpctl_tensor_types.hpp"
#include "kernels/reductions.hpp"
#include "utils/offset_utils.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/sycl_utils.hpp"
#include "utils/type_utils.hpp"

namespace dpctl::tensor::kernels
{

using dpctl::tensor::ssize_t;

namespace gemm_detail
{

template <typename T, std::size_t m_groups>
void scale_gemm_k_parameters(const std::size_t &local_mem_size,
                             const std::size_t &reserved_slm_size,
                             const std::size_t delta_k,
                             std::size_t &n_wi,
                             std::size_t &delta_n)
{
    static constexpr std::size_t slm_elem_size = sizeof(T) * m_groups;

    while (slm_elem_size * (n_wi + delta_n) * delta_k + reserved_slm_size >=
           local_mem_size)
    {
        n_wi = n_wi / 2;
        delta_n = delta_n / 2;
        if (delta_n == 0)
            throw std::runtime_error("Insufficient resources");
    }
}

template <typename T, int wi_delta_m>
void scale_gemm_nm_parameters(const std::size_t &local_mem_size,
                              const std::size_t &reserved_slm_size,
                              const std::size_t &wi_delta_n,
                              std::size_t &wi_delta_k,
                              std::size_t &wg_delta_n,
                              std::size_t &wg_delta_m)
{
    static constexpr std::size_t slm_A_elem_size = sizeof(T);
    static constexpr std::size_t slm_B_elem_size = sizeof(T) * wi_delta_m;

    while ((wi_delta_n * wg_delta_n * wi_delta_k * slm_A_elem_size) +
               (wi_delta_k * wg_delta_m * slm_B_elem_size) +
               reserved_slm_size >=
           local_mem_size)
    {
        wg_delta_n /= 2;
        wg_delta_m /= 2;
        wi_delta_k /= 2;
        if (wg_delta_n == 0)
            throw std::runtime_error("Insufficient resources");
    }
}
} // namespace gemm_detail

using dpctl::tensor::sycl_utils::choose_workgroup_size;

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class gemm_seq_reduction_krn;

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class gemm_tree_reduction_krn;

template <typename T, typename ReductionOpT>
sycl::event single_reduction_for_gemm(sycl::queue &exec_q,
                                      T *tmp_tp,
                                      T *res_tp,
                                      T identity_val,
                                      std::size_t iter_nelems,
                                      std::size_t reduction_nelems,
                                      std::size_t reduction_groups,
                                      std::size_t wg,
                                      std::size_t max_wg,
                                      std::size_t preferred_reductions_per_wi,
                                      std::size_t reductions_per_wi,
                                      int res_nd,
                                      ssize_t res_offset,
                                      const ssize_t *res_shapes_strides,
                                      const std::vector<sycl::event> &depends)
{
    sycl::event red_ev;
    if (reduction_nelems < wg) {
        using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using ResIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                NoOpIndexerT, ResIndexerT>;
        using ReductionIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;

        const ResIndexerT res_iter_indexer{res_nd, 0, res_shapes_strides};
        const InputOutputIterIndexerT in_out_iter_indexer{NoOpIndexerT{},
                                                          res_iter_indexer};
        const ReductionIndexerT reduction_indexer{/* size   */ reduction_nelems,
                                                  /* step   */ iter_nelems};

        red_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            sycl::range<1> iter_range{iter_nelems};

            cgh.parallel_for<class gemm_seq_reduction_krn<
                T, T, ReductionOpT, InputOutputIterIndexerT,
                ReductionIndexerT>>(
                iter_range,
                SequentialReduction<T, T, ReductionOpT, InputOutputIterIndexerT,
                                    ReductionIndexerT>(
                    tmp_tp, res_tp, ReductionOpT(), identity_val,
                    in_out_iter_indexer, reduction_indexer, reduction_nelems));
        });
    }
    else {
        using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using ResIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                NoOpIndexerT, ResIndexerT>;
        using ReductionIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;

        const ResIndexerT res_iter_indexer{res_nd, 0, res_shapes_strides};
        const InputOutputIterIndexerT in_out_iter_indexer{NoOpIndexerT{},
                                                          res_iter_indexer};
        const ReductionIndexerT reduction_indexer{/* size */ reduction_nelems,
                                                  /* step */ iter_nelems};

        if (iter_nelems == 1) {
            // increase GPU occupancy
            wg = max_wg;
        }
        reductions_per_wi =
            std::max<std::size_t>(1, (reduction_nelems + wg - 1) / wg);

        std::size_t reduction_groups =
            (reduction_nelems + reductions_per_wi * wg - 1) /
            (reductions_per_wi * wg);
        assert(reduction_groups == 1);

        red_ev = dpctl::tensor::kernels::submit_no_atomic_reduction<
            T, T, ReductionOpT, InputOutputIterIndexerT, ReductionIndexerT,
            gemm_tree_reduction_krn>(
            exec_q, tmp_tp, res_tp, identity_val, wg, iter_nelems,
            reduction_nelems, reductions_per_wi, reduction_groups,
            in_out_iter_indexer, reduction_indexer, depends);
    }
    return red_ev;
}

template <typename T, typename ReductionOpT>
sycl::event
    single_reduction_for_gemm_contig(sycl::queue &exec_q,
                                     T *tmp_tp,
                                     T *res_tp,
                                     T identity_val,
                                     std::size_t iter_nelems,
                                     std::size_t reduction_nelems,
                                     std::size_t reduction_groups,
                                     std::size_t wg,
                                     std::size_t max_wg,
                                     std::size_t preferred_reductions_per_wi,
                                     std::size_t reductions_per_wi,
                                     const std::vector<sycl::event> &depends)
{
    sycl::event red_ev;
    if (reduction_nelems < wg) {
        using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                NoOpIndexerT, NoOpIndexerT>;
        using ReductionIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;

        static constexpr InputOutputIterIndexerT in_out_iter_indexer{
            NoOpIndexerT{}, NoOpIndexerT{}};
        // tmp allocation is a C-contiguous matrix (reduction_nelems,
        // iter_nelems) and we are reducing by axis 0
        const ReductionIndexerT reduction_indexer{/* size */ reduction_nelems,
                                                  /* step */ iter_nelems};

        red_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            sycl::range<1> iter_range{iter_nelems};

            cgh.parallel_for<class gemm_seq_reduction_krn<
                T, T, ReductionOpT, InputOutputIterIndexerT,
                ReductionIndexerT>>(
                iter_range,
                SequentialReduction<T, T, ReductionOpT, InputOutputIterIndexerT,
                                    ReductionIndexerT>(
                    tmp_tp, res_tp, ReductionOpT(), identity_val,
                    in_out_iter_indexer, reduction_indexer, reduction_nelems));
        });
    }
    else {
        using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                NoOpIndexerT, NoOpIndexerT>;
        using ReductionIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;

        static constexpr InputOutputIterIndexerT in_out_iter_indexer{
            NoOpIndexerT{}, NoOpIndexerT{}};
        // tmp allocation is a C-contiguous matrix
        // (reduction_nelems, iter_nelems). Reducing along axis 0
        const ReductionIndexerT reduction_indexer{/* size */ reduction_nelems,
                                                  /* step */ iter_nelems};

        if (iter_nelems == 1) {
            // increase GPU occupancy
            wg = max_wg;
        }
        reductions_per_wi =
            std::max<std::size_t>(1, (reduction_nelems + wg - 1) / wg);

        std::size_t reduction_groups =
            (reduction_nelems + reductions_per_wi * wg - 1) /
            (reductions_per_wi * wg);
        assert(reduction_groups == 1);

        red_ev = dpctl::tensor::kernels::submit_no_atomic_reduction<
            T, T, ReductionOpT, InputOutputIterIndexerT, ReductionIndexerT,
            gemm_tree_reduction_krn>(
            exec_q, tmp_tp, res_tp, identity_val, wg, iter_nelems,
            reduction_nelems, reductions_per_wi, reduction_groups,
            in_out_iter_indexer, reduction_indexer, depends);
    }
    return red_ev;
}

template <typename T, typename ReductionOpT>
sycl::event tree_reduction_for_gemm(sycl::queue &exec_q,
                                    T *partially_reduced_tmp,
                                    T *partially_reduced_tmp2,
                                    T *res_tp,
                                    T identity_val,
                                    std::size_t iter_nelems,
                                    std::size_t reduction_nelems,
                                    std::size_t reduction_groups,
                                    std::size_t wg,
                                    std::size_t max_wg,
                                    std::size_t preferred_reductions_per_wi,
                                    std::size_t reductions_per_wi,
                                    int res_nd,
                                    ssize_t res_offset,
                                    const ssize_t *res_shape_strides,
                                    const std::vector<sycl::event> &depends)
{
    sycl::event first_reduction_ev;
    {
        using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                NoOpIndexerT, NoOpIndexerT>;
        using ReductionIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;

        static constexpr InputOutputIterIndexerT in_out_iter_indexer{
            NoOpIndexerT{}, NoOpIndexerT{}};
        // partially_reduced_tmp is C-contig matrix with shape
        // (reduction_nelems, iter_nelems). Reducing along axis 0.
        const ReductionIndexerT reduction_indexer{/* size */ reduction_nelems,
                                                  /* step */ iter_nelems};

        first_reduction_ev = dpctl::tensor::kernels::submit_no_atomic_reduction<
            T, T, ReductionOpT, InputOutputIterIndexerT, ReductionIndexerT,
            gemm_tree_reduction_krn>(
            exec_q, partially_reduced_tmp, partially_reduced_tmp2, identity_val,
            wg, iter_nelems, reduction_nelems, reductions_per_wi,
            reduction_groups, in_out_iter_indexer, reduction_indexer, depends);
    }

    std::size_t remaining_reduction_nelems = reduction_groups;

    T *temp_arg = partially_reduced_tmp2;
    T *temp2_arg = partially_reduced_tmp;
    sycl::event dependent_ev = first_reduction_ev;

    while (remaining_reduction_nelems > preferred_reductions_per_wi * max_wg) {
        std::size_t reduction_groups_ = (remaining_reduction_nelems +
                                         preferred_reductions_per_wi * wg - 1) /
                                        (preferred_reductions_per_wi * wg);
        assert(reduction_groups_ > 1);

        // keep reducing
        using InputIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
        using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                InputIndexerT, ResIndexerT>;
        using ReductionIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;

        const InputIndexerT inp_indexer{/* size */ iter_nelems,
                                        /* step */ reduction_groups_};
        static constexpr ResIndexerT res_iter_indexer{};

        const InputOutputIterIndexerT in_out_iter_indexer{inp_indexer,
                                                          res_iter_indexer};

        static constexpr ReductionIndexerT reduction_indexer{};

        sycl::event partial_reduction_ev =
            dpctl::tensor::kernels::submit_no_atomic_reduction<
                T, T, ReductionOpT, InputOutputIterIndexerT, ReductionIndexerT,
                gemm_tree_reduction_krn>(
                exec_q, temp_arg, temp2_arg, identity_val, wg, iter_nelems,
                remaining_reduction_nelems, reductions_per_wi,
                reduction_groups_, in_out_iter_indexer, reduction_indexer,
                {dependent_ev});

        remaining_reduction_nelems = reduction_groups_;
        std::swap(temp_arg, temp2_arg);
        dependent_ev = std::move(partial_reduction_ev);
    }

    // final reduction to res
    using InputIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
    using ResIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
    using InputOutputIterIndexerT =
        dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<InputIndexerT,
                                                                ResIndexerT>;
    using ReductionIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;

    const InputIndexerT inp_indexer{/* size */ iter_nelems,
                                    /* step */ remaining_reduction_nelems};
    const ResIndexerT res_iter_indexer{
        /* ndim                */ res_nd,
        /* offset              */ static_cast<ssize_t>(res_offset),
        /* packed shape_strides*/ res_shape_strides};

    const InputOutputIterIndexerT in_out_iter_indexer{inp_indexer,
                                                      res_iter_indexer};
    static constexpr ReductionIndexerT reduction_indexer{};

    wg = max_wg;
    reductions_per_wi =
        std::max<std::size_t>(1, (remaining_reduction_nelems + wg - 1) / wg);

    reduction_groups =
        (remaining_reduction_nelems + reductions_per_wi * wg - 1) /
        (reductions_per_wi * wg);
    assert(reduction_groups == 1);

    sycl::event final_reduction_ev =
        dpctl::tensor::kernels::submit_no_atomic_reduction<
            T, T, ReductionOpT, InputOutputIterIndexerT, ReductionIndexerT,
            gemm_tree_reduction_krn>(
            exec_q, temp_arg, res_tp, identity_val, wg, iter_nelems,
            remaining_reduction_nelems, reductions_per_wi, reduction_groups,
            in_out_iter_indexer, reduction_indexer, {dependent_ev});

    return final_reduction_ev;
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class gemm_reduction_over_group_temps_contig_krn;

template <typename T, typename ReductionOpT>
sycl::event
    tree_reduction_for_gemm_contig(sycl::queue &exec_q,
                                   T *partially_reduced_tmp,
                                   T *partially_reduced_tmp2,
                                   T *res_tp,
                                   T identity_val,
                                   std::size_t iter_nelems,
                                   std::size_t reduction_nelems,
                                   std::size_t reduction_groups,
                                   std::size_t wg,
                                   std::size_t max_wg,
                                   std::size_t preferred_reductions_per_wi,
                                   std::size_t reductions_per_wi,
                                   const std::vector<sycl::event> &depends)
{
    using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
    using InputOutputIterIndexerT =
        dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<NoOpIndexerT,
                                                                NoOpIndexerT>;
    using ReductionIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;

    static constexpr InputOutputIterIndexerT in_out_iter_indexer{
        NoOpIndexerT{}, NoOpIndexerT{}};
    const ReductionIndexerT reduction_indexer{/* size */ reduction_nelems,
                                              /* step */ iter_nelems};

    const sycl::event &first_reduction_ev =
        dpctl::tensor::kernels::submit_no_atomic_reduction<
            T, T, ReductionOpT, InputOutputIterIndexerT, ReductionIndexerT,
            gemm_reduction_over_group_temps_contig_krn>(
            exec_q, partially_reduced_tmp, partially_reduced_tmp2, identity_val,
            wg, iter_nelems, reduction_nelems, reductions_per_wi,
            reduction_groups, in_out_iter_indexer, reduction_indexer, depends);

    std::size_t remaining_reduction_nelems = reduction_groups;

    T *temp_arg = partially_reduced_tmp2;
    T *temp2_arg = partially_reduced_tmp;
    sycl::event dependent_ev = first_reduction_ev;

    while (remaining_reduction_nelems > preferred_reductions_per_wi * max_wg) {
        std::size_t reduction_groups_ = (remaining_reduction_nelems +
                                         preferred_reductions_per_wi * wg - 1) /
                                        (preferred_reductions_per_wi * wg);
        assert(reduction_groups_ > 1);

        // keep reducing
        using InputIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
        using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                InputIndexerT, ResIndexerT>;
        using ReductionIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;

        // n * m = iter_nelems because essentially, this process
        // creates a stack of reduction_nelems 2D matrices and we reduce
        // along the stack axis
        const InputIndexerT inp_indexer{/* size */ iter_nelems,
                                        /* step */ reduction_groups_};
        static constexpr ResIndexerT res_iter_indexer{};

        const InputOutputIterIndexerT in_out_iter_indexer{inp_indexer,
                                                          res_iter_indexer};

        static constexpr ReductionIndexerT reduction_indexer{};

        sycl::event partial_reduction_ev =
            dpctl::tensor::kernels::submit_no_atomic_reduction<
                T, T, ReductionOpT, InputOutputIterIndexerT, ReductionIndexerT,
                gemm_reduction_over_group_temps_contig_krn>(
                exec_q, temp_arg, temp2_arg, identity_val, wg, iter_nelems,
                remaining_reduction_nelems, reductions_per_wi,
                reduction_groups_, in_out_iter_indexer, reduction_indexer,
                {dependent_ev});

        remaining_reduction_nelems = reduction_groups_;
        std::swap(temp_arg, temp2_arg);
        dependent_ev = std::move(partial_reduction_ev);
    }

    // final reduction to res
    {
        using InputIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
        using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                InputIndexerT, ResIndexerT>;
        using ReductionIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;

        const InputIndexerT inp_indexer{
            /* size   */ iter_nelems,
            /* step   */ remaining_reduction_nelems};
        static constexpr ResIndexerT res_iter_indexer{};

        const InputOutputIterIndexerT in_out_iter_indexer{inp_indexer,
                                                          res_iter_indexer};
        static constexpr ReductionIndexerT reduction_indexer{};

        wg = max_wg;
        reductions_per_wi = std::max<std::size_t>(
            1, (remaining_reduction_nelems + wg - 1) / wg);

        std::size_t reduction_groups =
            (remaining_reduction_nelems + reductions_per_wi * wg - 1) /
            (reductions_per_wi * wg);
        assert(reduction_groups == 1);

        sycl::event final_reduction_ev =
            dpctl::tensor::kernels::submit_no_atomic_reduction<
                T, T, ReductionOpT, InputOutputIterIndexerT, ReductionIndexerT,
                gemm_reduction_over_group_temps_contig_krn>(
                exec_q, temp_arg, res_tp, identity_val, wg, iter_nelems,
                remaining_reduction_nelems, reductions_per_wi, reduction_groups,
                in_out_iter_indexer, reduction_indexer, {dependent_ev});

        return final_reduction_ev;
    }
}

template <typename lhsT,
          typename rhsT,
          typename resT,
          typename LocAccT,
          typename OuterInnerDimsIndexerT,
          typename BatchDimsIndexerT,
          std::size_t m_groups>
class GemmBatchFunctorThreadK
{
private:
    const lhsT *lhs = nullptr;
    const rhsT *rhs = nullptr;
    resT *res = nullptr;
    LocAccT workspace;
    LocAccT local_B_block;
    std::size_t n = 0;
    std::size_t n_blocks = 0;
    std::size_t delta_n = 0;
    std::size_t k = 0;
    std::size_t k_blocks = 0;
    std::size_t delta_k = 0;
    std::size_t n_wi = 0;
    std::size_t m = 0;
    std::size_t batch_nelems = 0;
    BatchDimsIndexerT batch_indexer;
    OuterInnerDimsIndexerT lhs_indexer;
    OuterInnerDimsIndexerT rhs_indexer;
    OuterInnerDimsIndexerT res_indexer;

public:
    GemmBatchFunctorThreadK(const lhsT *lhs_,
                            const rhsT *rhs_,
                            resT *res_,
                            LocAccT workspace_,
                            LocAccT local_B_block_,
                            std::size_t n_,
                            std::size_t n_blocks_,
                            std::size_t delta_n_,
                            std::size_t k_,
                            std::size_t k_blocks_,
                            std::size_t delta_k_,
                            std::size_t n_wi_,
                            std::size_t m_,
                            std::size_t batch_nelems_,
                            const BatchDimsIndexerT &batch_indexer_,
                            const OuterInnerDimsIndexerT &lhs_indexer_,
                            const OuterInnerDimsIndexerT &rhs_indexer_,
                            const OuterInnerDimsIndexerT &res_indexer_)
        : lhs(lhs_), rhs(rhs_), res(res_), workspace(workspace_),
          local_B_block(local_B_block_), n(n_), n_blocks(n_blocks_),
          delta_n(delta_n_), k(k_), k_blocks(k_blocks_), delta_k(delta_k_),
          n_wi(n_wi_), m(m_), batch_nelems(batch_nelems_),
          batch_indexer(batch_indexer_), lhs_indexer(lhs_indexer_),
          rhs_indexer(rhs_indexer_), res_indexer(res_indexer_)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        // for batching:
        // (current matrix in batch) m_id = global_id / (global_range /
        // batch_nelems) for lhs, offset = m_id * (n * k) for rhs, offset =
        // m_id
        // * (k * m) for res, offset = m_id * (n * m)
        const std::size_t n_groups_per_batch =
            it.get_group_range(0) / batch_nelems;
        const std::size_t m_id = it.get_group_linear_id() / n_groups_per_batch;
        const std::size_t gr_id =
            it.get_group_linear_id() - m_id * n_groups_per_batch;
        const std::size_t lid = it.get_local_linear_id();

        const auto &three_offsets_ = batch_indexer(static_cast<ssize_t>(m_id));

        const auto &lhs_offset = three_offsets_.get_first_offset();
        const auto &rhs_offset = three_offsets_.get_second_offset();
        const auto &res_offset = three_offsets_.get_third_offset();

        // lift gr_id -> (block_i, block_j, block_s)
        //   block_i moves fastest, then block_s, then block_j

        const std::size_t r_size = (n_blocks * k_blocks);
        // 0 <= block_j < m_blocks,
        const std::size_t block_j = gr_id / r_size;
        // 0 <= block_r < n_blocks * k_blocks
        const std::size_t block_r = gr_id - block_j * r_size;
        // 0 <= block_s < k_blocks
        const std::size_t block_s = block_r / n_blocks;
        // 0 <= block_i < n_blocks
        const std::size_t block_i = block_r - block_s * n_blocks;

        // 0 <= local_i < delta_n
        const std::size_t local_i = lid / (delta_k);
        // 0 <= local_s < delta_k
        const std::size_t local_s = lid - local_i * (delta_k);

        std::size_t i = block_i * delta_n + local_i;
        std::size_t j = m_groups * block_j;
        std::size_t s = block_s * delta_k * n_wi + local_s;

        using accV_t = typename LocAccT::value_type;

        static constexpr resT identity_ = resT(0);
        if (local_i == 0) {
            for (std::size_t q = 0; q < n_wi * delta_k; q += delta_k) {
                const std::size_t sq = s + q;
                const std::size_t sqmj = sq * m + j;

                if constexpr (m_groups == 1 && std::is_same_v<accV_t, resT>) {
                    local_B_block[local_s + q] =
                        (sq < k && j < m)
                            ? static_cast<resT>(
                                  rhs[rhs_offset + rhs_indexer(sqmj)])
                            : identity_;
                }
                else {
                    accV_t local_B_vec;
#pragma unroll
                    for (std::size_t vec_idx = 0; vec_idx < m_groups; ++vec_idx)
                    {
                        local_B_vec[vec_idx] =
                            (sq < k && j + vec_idx < m)
                                ? static_cast<resT>(
                                      rhs[rhs_offset +
                                          rhs_indexer(sqmj + vec_idx)])
                                : identity_;
                    }
                    local_B_block[local_s + q] = local_B_vec;
                }
            }
        }

        it.barrier(sycl::access::fence_space::local_space);

        std::size_t t_shift = block_s * delta_k * n_wi;
        std::size_t global_s_offset = i * k + t_shift;

        accV_t private_sum(identity_);
        static constexpr accV_t vec_identity_(identity_);
        for (std::size_t t = local_s; t < local_B_block.size(); t += delta_k) {
            private_sum +=
                ((i < n) && (t + t_shift < k))
                    ? (static_cast<resT>(
                           lhs[lhs_offset + lhs_indexer(global_s_offset + t)]) *
                       local_B_block[t])
                    : vec_identity_;
        }

        std::size_t workspace_i_shift = local_i * delta_k;
        workspace[workspace_i_shift + local_s] = private_sum;

        it.barrier(sycl::access::fence_space::local_space);

        if (local_s == 0 && i < n) {
            accV_t local_sum(workspace[workspace_i_shift]);
            for (std::size_t t = 1; t < delta_k; ++t) {
                local_sum += workspace[workspace_i_shift + t];
            }

            sycl::atomic_ref<resT, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                aout0(res[res_offset + res_indexer(i * m + j)]);

            if constexpr (m_groups == 1 && std::is_same_v<accV_t, resT>) {
                aout0 += local_sum;
            }
            else {
                aout0 += local_sum[0];

#pragma unroll
                for (std::size_t vec_id = 1; vec_id < m_groups; ++vec_id) {
                    if (j + vec_id < m) {
                        sycl::atomic_ref<
                            resT, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            aout1(res[res_offset +
                                      res_indexer(i * m + j + vec_id)]);

                        aout1 += local_sum[vec_id];
                    }
                }
            }
        }
    }
};

template <typename T1, typename T2, typename T3>
class gemm_init_krn;

template <typename T1, typename T2, typename T3, typename T4, std::size_t>
class gemm_k_krn;

template <typename T1, typename T2, typename T3, typename T4, std::size_t>
class gemm_nm_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          std::size_t>
class gemm_batch_k_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          std::size_t>
class gemm_batch_nm_krn;

namespace gemm_detail
{

template <typename lhsTy,
          typename rhsTy,
          typename resTy,
          typename BatchIndexerT,
          typename LhsIndexerT,
          typename RhsIndexerT,
          typename ResIndexerT>
sycl::event _gemm_k_impl(sycl::queue &exec_q,
                         const lhsTy *lhs_tp,
                         const rhsTy *rhs_tp,
                         resTy *res_tp,
                         const std::size_t batch_nelems,
                         const std::size_t n,
                         const std::size_t k,
                         const std::size_t m,
                         const BatchIndexerT &batch_indexer,
                         const LhsIndexerT &lhs_indexer,
                         const RhsIndexerT &rhs_indexer,
                         const ResIndexerT &res_indexer,
                         const std::vector<sycl::event> &depends)
{
    static constexpr std::size_t m_groups = 4;
    const std::size_t delta_k(4);
    std::size_t n_wi(64);
    std::size_t delta_n(32);

    static_assert(std::is_same_v<LhsIndexerT, RhsIndexerT>);
    static_assert(std::is_same_v<LhsIndexerT, ResIndexerT>);

    const sycl::device &dev = exec_q.get_device();
    const std::size_t local_mem_size =
        dev.get_info<sycl::info::device::local_mem_size>();
    const std::size_t reserved_slm_size = 512;

    gemm_detail::scale_gemm_k_parameters<resTy, m_groups>(
        local_mem_size, reserved_slm_size, delta_k,
        n_wi,   // modified by reference
        delta_n // modified by reference
    );

    std::size_t n_blocks = (n + delta_n - 1) / delta_n;
    std::size_t m_blocks = (m + m_groups - 1) / m_groups;
    std::size_t k_blocks = (k + n_wi * delta_k - 1) / (n_wi * delta_k);

    std::size_t lws = delta_n * delta_k;

    auto gRange =
        sycl::range<1>(batch_nelems * n_blocks * m_blocks * k_blocks * lws);
    auto lRange = sycl::range<1>(lws);

    auto ndRange = sycl::nd_range<1>(gRange, lRange);

    sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using LocAccT = sycl::local_accessor<sycl::vec<resTy, m_groups>, 1>;
        LocAccT local_B_block(n_wi * delta_k, cgh);
        LocAccT workspace(delta_n * delta_k, cgh);

        using KernelName =
            class gemm_batch_k_krn<lhsTy, rhsTy, resTy, LhsIndexerT,
                                   BatchIndexerT, m_groups>;
        cgh.parallel_for<KernelName>(
            ndRange,
            GemmBatchFunctorThreadK<lhsTy, rhsTy, resTy, LocAccT, LhsIndexerT,
                                    BatchIndexerT, m_groups>(
                lhs_tp, rhs_tp, res_tp, std::move(workspace),
                std::move(local_B_block), n, n_blocks, delta_n, k, k_blocks,
                delta_k, n_wi, m, batch_nelems, batch_indexer, lhs_indexer,
                rhs_indexer, res_indexer));
    });
    return gemm_ev;
}

template <typename lhsTy,
          typename rhsTy,
          typename resTy,
          typename BatchIndexerT,
          typename LhsIndexerT,
          typename RhsIndexerT,
          typename ResIndexerT>
sycl::event _gemm_small_m_impl(sycl::queue &exec_q,
                               const lhsTy *lhs_tp,
                               const rhsTy *rhs_tp,
                               resTy *res_tp,
                               const std::size_t batch_nelems,
                               const std::size_t n,
                               const std::size_t k,
                               const std::size_t m,
                               const BatchIndexerT &batch_indexer,
                               const LhsIndexerT &lhs_indexer,
                               const RhsIndexerT &rhs_indexer,
                               const ResIndexerT &res_indexer,
                               const std::vector<sycl::event> &depends)
{
    static constexpr std::size_t m_groups = 1;
    const std::size_t delta_k(4);
    std::size_t n_wi(64);
    std::size_t delta_n(32);

    static_assert(std::is_same_v<LhsIndexerT, RhsIndexerT>);
    static_assert(std::is_same_v<LhsIndexerT, ResIndexerT>);

    const sycl::device &dev = exec_q.get_device();
    const std::size_t local_mem_size =
        dev.get_info<sycl::info::device::local_mem_size>();
    const std::size_t reserved_slm_size = 512;

    gemm_detail::scale_gemm_k_parameters<resTy, m_groups>(
        local_mem_size, reserved_slm_size, delta_k,
        n_wi,   // modified by reference
        delta_n // modified by reference
    );

    std::size_t n_blocks = (n + delta_n - 1) / delta_n;
    std::size_t m_blocks = (m + m_groups - 1) / m_groups;
    std::size_t k_blocks = (k + n_wi * delta_k - 1) / (n_wi * delta_k);

    std::size_t lws = delta_n * delta_k;

    auto gRange =
        sycl::range<1>(batch_nelems * n_blocks * m_blocks * k_blocks * lws);
    auto lRange = sycl::range<1>(lws);

    auto ndRange = sycl::nd_range<1>(gRange, lRange);

    sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using LocAccT = sycl::local_accessor<resTy, 1>;
        LocAccT local_B_block(n_wi * delta_k, cgh);
        LocAccT workspace(delta_n * delta_k, cgh);

        using KernelName =
            class gemm_batch_k_krn<lhsTy, rhsTy, resTy, LhsIndexerT,
                                   BatchIndexerT, m_groups>;
        cgh.parallel_for<KernelName>(
            ndRange,
            GemmBatchFunctorThreadK<lhsTy, rhsTy, resTy, LocAccT, LhsIndexerT,
                                    BatchIndexerT, m_groups>(
                lhs_tp, rhs_tp, res_tp, std::move(workspace),
                std::move(local_B_block), n, n_blocks, delta_n, k, k_blocks,
                delta_k, n_wi, m, batch_nelems, batch_indexer, lhs_indexer,
                rhs_indexer, res_indexer));
    });

    return gemm_ev;
}

} // end of namespace gemm_detail

template <typename lhsT,
          typename rhsT,
          typename resT,
          typename LocAccT1,
          typename LocAccT2,
          typename BatchDimsIndexerT,
          typename LhsIndexerT,
          typename RhsIndexerT,
          typename ResIndexerT,
          std::uint32_t wi_delta_n,
          std::uint32_t wi_delta_m_vecs,
          std::uint32_t m_vec_size>
class GemmBatchFunctorThreadNM_vecm
{
private:
    const lhsT *lhs = nullptr;
    const rhsT *rhs = nullptr;
    resT *res = nullptr;
    LocAccT1 local_lhs_block;
    LocAccT2 local_rhs_block;
    std::size_t batch_nelems;
    std::size_t n = 0;
    std::size_t k = 0;
    std::size_t m = 0;
    std::size_t n_groups = 0;
    std::uint32_t wg_delta_n = 0;
    std::uint32_t wg_delta_m = 0;
    std::uint32_t wi_delta_k = 0;
    BatchDimsIndexerT batch_indexer;
    LhsIndexerT lhs_indexer;
    RhsIndexerT rhs_indexer;
    ResIndexerT res_indexer;

public:
    /*! @brief */
    GemmBatchFunctorThreadNM_vecm(const lhsT *lhs_,
                                  const rhsT *rhs_,
                                  resT *res_,
                                  LocAccT1 local_lhs_block_,
                                  LocAccT2 local_rhs_block_,
                                  std::size_t batch_nelems_,
                                  std::size_t n_,
                                  std::size_t k_,
                                  std::size_t m_,
                                  std::size_t n_groups_,
                                  std::size_t wg_delta_n_,
                                  std::size_t wg_delta_m_,
                                  std::size_t wi_delta_k_,
                                  const BatchDimsIndexerT &batch_indexer_,
                                  const LhsIndexerT &lhs_indexer_,
                                  const RhsIndexerT &rhs_indexer_,
                                  const ResIndexerT &res_indexer_)
        : lhs(lhs_), rhs(rhs_), res(res_), local_lhs_block(local_lhs_block_),
          local_rhs_block(local_rhs_block_), batch_nelems(batch_nelems_), n(n_),
          k(k_), m(m_), n_groups(n_groups_), wg_delta_n(wg_delta_n_),
          wg_delta_m(wg_delta_m_), wi_delta_k(wi_delta_k_),
          batch_indexer(batch_indexer_), lhs_indexer(lhs_indexer_),
          rhs_indexer(rhs_indexer_), res_indexer(res_indexer_)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        static constexpr resT zero_(0);
        static constexpr std::uint32_t wi_total_delta_m =
            wi_delta_m_vecs * m_vec_size;

        const std::size_t gws_per_batch = it.get_group_range(0) / batch_nelems;
        const std::size_t batch_id = it.get_group_linear_id() / gws_per_batch;
        const std::size_t gr_id =
            it.get_group_linear_id() - batch_id * gws_per_batch;

        const auto &three_offsets_ =
            batch_indexer(static_cast<ssize_t>(batch_id));

        const auto &lhs_offset = three_offsets_.get_first_offset();
        const auto &rhs_offset = three_offsets_.get_second_offset();
        const auto &res_offset = three_offsets_.get_third_offset();

        // 0 <= block_j < m_groups
        const std::size_t block_j = gr_id / n_groups;
        // 0 <= block_i < n_groups
        const std::size_t block_i = gr_id - block_j * n_groups;

        // Assumption: lws == wg_delta_n * wg_delta_m
        const std::uint32_t lid = it.get_local_linear_id();
        // 0 <= local_j < (lws / wg_delta_n == wg_delta_m)
        const std::uint32_t local_j = lid / wg_delta_n;
        // sub-group lanes map to adjacent local_i
        const std::uint32_t local_i = lid - local_j * wg_delta_n;

        // Coordinates of the block of C the work-group works on
        std::size_t i = block_i * wg_delta_n * wi_delta_n;
        std::size_t j = block_j * wg_delta_m * wi_total_delta_m;

        using slmA_t = typename LocAccT1::value_type;
        using slmB_t = typename LocAccT2::value_type;

        const std::size_t a_st0 = k;
        const std::size_t a_st1 = 1;

        const std::size_t b_st0 = m;
        const std::size_t b_st1 = 1;

        const std::size_t c_st0 = m;
        const std::size_t c_st1 = 1;

        // allocate/initialize private matrix C
        // size ( wi_total_delta_n, wi_total_delta_m )
        static constexpr std::uint32_t C_size = wi_delta_n * wi_delta_m_vecs;
        std::array<slmB_t, C_size> private_C{slmB_t{zero_}};

        for (std::size_t s = 0; s < k; s += wi_delta_k) {
            // populate local_lhs_block<resT> ( wg_delta_n * wi_delta_n,
            // wi_delta_k)
            for (std::uint32_t vid = lid; vid < local_lhs_block.size();
                 vid += it.get_local_range()[0])
            {
                // 0 <= v_i < wg_delta_n * wi_delta_n
                const std::uint32_t v_i = vid / wi_delta_k;
                // 0 <= v_s < wi_delta_k
                const std::uint32_t v_s = vid - v_i * wi_delta_k;

                const std::size_t g_i = i + v_i;
                const std::size_t g_s = s + v_s;

                const std::uint32_t mapped_vid =
                    wg_delta_n * wi_delta_n * v_s + v_i;
                local_lhs_block[mapped_vid] =
                    (g_i < n && g_s < k)
                        ? static_cast<resT>(
                              lhs[lhs_offset +
                                  lhs_indexer(g_i * a_st0 + g_s * a_st1)])
                        : zero_;
            }

            // populate local_rhs_block<vec<resT, m_vec_size>> ( wg_delta_m *
            // wi_delta_m_vecs, wi_delta_k )
            for (std::uint32_t vid = lid; vid < local_rhs_block.size();
                 vid += it.get_local_range()[0])
            {
                // 0 <= v_j < wg_delta_m * wi_delta_m_vecs
                const std::uint32_t v_j = vid / wi_delta_k;
                // 0 <= v_s < wi_delta_k
                const std::uint32_t v_s = vid - v_j * wi_delta_k;

                const std::size_t g_j = j + v_j * m_vec_size;
                const std::size_t g_s = s + v_s;
                const std::uint32_t mapped_vid =
                    wg_delta_m * wi_delta_m_vecs * v_s + v_j;

                if constexpr (m_vec_size == 1) {
                    local_rhs_block[mapped_vid] =
                        (g_j < m && g_s < k)
                            ? static_cast<resT>(
                                  rhs[rhs_offset +
                                      rhs_indexer(g_s * b_st0 + g_j * b_st1)])
                            : zero_;
                }
                else {
                    slmB_t vec{};
#pragma unroll
                    for (std::uint32_t lane_id = 0; lane_id < m_vec_size;
                         ++lane_id)
                    {
                        const std::size_t g_j1 = g_j + lane_id;
                        vec[lane_id] = (g_j1 < m && g_s < k)
                                           ? static_cast<resT>(
                                                 rhs[rhs_offset +
                                                     rhs_indexer(g_s * b_st0 +
                                                                 g_j1 * b_st1)])
                                           : zero_;
                    };

                    local_rhs_block[mapped_vid] = vec;
                }
            }

            it.barrier(sycl::access::fence_space::local_space);

            const std::uint32_t lo_lhs_st_k = (wg_delta_n * wi_delta_n);
            const std::uint32_t lo_rhs_rk_k = (wg_delta_m * wi_delta_m_vecs);
            for (std::uint32_t pr_k = 0; pr_k < wi_delta_k; ++pr_k) {
                std::array<slmA_t, wi_delta_n> pr_lhs{};
#pragma unroll
                for (std::uint32_t pr_i = 0; pr_i < wi_delta_n; ++pr_i) {
                    pr_lhs[pr_i] =
                        local_lhs_block[pr_k * lo_lhs_st_k +
                                        (local_i + pr_i * wg_delta_n)];
                }

                std::array<slmB_t, wi_delta_m_vecs> pr_rhs{};
#pragma unroll
                for (std::uint32_t pr_j = 0; pr_j < wi_delta_m_vecs; ++pr_j) {
                    pr_rhs[pr_j] =
                        local_rhs_block[pr_k * lo_rhs_rk_k +
                                        (local_j + pr_j * wg_delta_m)];
                }

#pragma unroll
                for (std::uint32_t pr_i = 0; pr_i < wi_delta_n; ++pr_i) {
#pragma unroll
                    for (std::uint32_t pr_j = 0; pr_j < wi_delta_m_vecs; ++pr_j)
                    {
                        private_C[pr_i * wi_delta_m_vecs + pr_j] +=
                            pr_lhs[pr_i] * pr_rhs[pr_j];
                    }
                }
            }

            it.barrier(sycl::access::fence_space::local_space);
        }

        if constexpr (m_vec_size == 1) {
#pragma unroll
            for (std::uint32_t pr_i = 0; pr_i < wi_delta_n; ++pr_i) {
                std::size_t out_i = i + local_i + pr_i * wg_delta_n;
                if (out_i < n) {
#pragma unroll
                    for (std::uint32_t pr_j = 0; pr_j < wi_delta_m_vecs; ++pr_j)
                    {
                        const std::size_t out_j =
                            j + (local_j + pr_j * wg_delta_m) * m_vec_size;
                        const std::size_t out_flat_id =
                            out_i * c_st0 + out_j * c_st1;
                        if (out_j < m) {
                            res[res_offset + res_indexer(out_flat_id)] =
                                private_C[pr_i * wi_delta_m_vecs + pr_j];
                        }
                    }
                }
            }
        }
        else {
#pragma unroll
            for (std::uint32_t pr_i = 0; pr_i < wi_delta_n; ++pr_i) {
                std::size_t out_i = i + local_i + pr_i * wg_delta_n;
                if (out_i < n) {
                    // could be unrolled
                    for (std::uint32_t pr_j = 0; pr_j < wi_delta_m_vecs; ++pr_j)
                    {
                        std::size_t out_j =
                            j + (local_j + pr_j * wg_delta_m) * m_vec_size;
#pragma unroll
                        for (std::uint32_t lane_id = 0; lane_id < m_vec_size;
                             ++lane_id)
                        {
                            const std::size_t out_flat_id =
                                out_i * c_st0 + (out_j + lane_id) * c_st1;
                            if (out_j + lane_id < m) {
                                res[res_offset + res_indexer(out_flat_id)] =
                                    private_C[pr_i * wi_delta_m_vecs + pr_j]
                                             [lane_id];
                            }
                        }
                    }
                }
            }
        }
    }
};

struct GemmBatchFunctorThreadNM_vecm_HyperParameters
{
private:
    std::uint32_t wi_delta_n = 2;
    std::uint32_t wi_delta_m_vecs = 4;
    std::uint32_t m_vec_size = 1;

public:
    constexpr GemmBatchFunctorThreadNM_vecm_HyperParameters();
    constexpr GemmBatchFunctorThreadNM_vecm_HyperParameters(
        std::uint32_t wi_delta_n_,
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

template <typename resT>
struct GemmBatchFunctorThreadNM_vecm_HyperParametersSelector
{
    constexpr GemmBatchFunctorThreadNM_vecm_HyperParametersSelector() {}

    constexpr GemmBatchFunctorThreadNM_vecm_HyperParameters get() const
    {
        if constexpr (sizeof(resT) == 1) {
            // 1 * 8 * 2 * 4 == 64
            return GemmBatchFunctorThreadNM_vecm_HyperParameters(8, 2, 4);
        }
        else if constexpr (sizeof(resT) == 2) {
            // 2 * 4 * 2 * 4 == 64
            return GemmBatchFunctorThreadNM_vecm_HyperParameters(4, 2, 4);
        }
        else if constexpr (sizeof(resT) == 4) {
            // 4 * 4 * 1 * 4 == 64
            return GemmBatchFunctorThreadNM_vecm_HyperParameters(4, 1, 4);
        }
        else if constexpr (sizeof(resT) == 8) {
            // 8 * 2 * 1 * 4 == 64
            if constexpr (std::is_same_v<resT, std::complex<float>>) {
                return GemmBatchFunctorThreadNM_vecm_HyperParameters(2, 4, 1);
            }
            else {
                return GemmBatchFunctorThreadNM_vecm_HyperParameters(2, 1, 4);
            }
        }
        else if constexpr (std::is_same_v<resT, std::complex<double>>) {
            // 16 * 2 * 2 * 1 == 64
            return GemmBatchFunctorThreadNM_vecm_HyperParameters(2, 2, 1);
        }
        else {
            return GemmBatchFunctorThreadNM_vecm_HyperParameters(2, 2, 1);
        }
    }
};

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
class gemm_batch_nm_vecm_krn;

namespace gemm_detail
{

template <typename T, std::uint32_t wi_delta_n, std::uint32_t wi_delta_m>
std::tuple<std::uint32_t, std::uint32_t>
    get_wg_delta_m_and_wi_delta_k(const std::size_t slm_byte_size,
                                  const std::uint32_t wg_delta_n,
                                  const std::uint32_t suggested_wg_delta_m)
{
    std::uint32_t wg_delta_m = suggested_wg_delta_m;

    const std::size_t slm_max_rows =
        slm_byte_size /
        ((wg_delta_n * wi_delta_n + wg_delta_m * wi_delta_m) * sizeof(T));

    std::uint32_t wi_delta_k =
        (slm_max_rows >= 64)
            ? 64
            : 32 * static_cast<std::uint32_t>(slm_max_rows / 32);

    for (std::uint32_t it = 0; !wi_delta_k && (it < 4); ++it) {
        wg_delta_m /= 2;

        const std::size_t slm_max_rows =
            slm_byte_size /
            ((wg_delta_n * wi_delta_n + wg_delta_m * wi_delta_m) * sizeof(T));

        wi_delta_k =
            (slm_max_rows >= 64)
                ? 64
                : ((slm_max_rows >= 32)
                       ? 32
                       : (slm_max_rows >= 16 ? 16
                                             : 8 * static_cast<std::uint32_t>(
                                                       slm_max_rows / 8)));
    }

    if (!wi_delta_k) {
        throw std::runtime_error("Insufficient resources");
    }

    return std::make_tuple(wg_delta_m, wi_delta_k);
}

template <typename lhsTy,
          typename rhsTy,
          typename resTy,
          typename BatchIndexerT,
          typename LhsIndexerT,
          typename RhsIndexerT,
          typename ResIndexerT>
sycl::event _gemm_batch_nm_impl(sycl::queue &exec_q,
                                const lhsTy *lhs_tp,
                                const rhsTy *rhs_tp,
                                resTy *res_tp,
                                const std::size_t batch_nelems,
                                const std::size_t n,
                                const std::size_t k,
                                const std::size_t m,
                                const BatchIndexerT &batch_indexer,
                                const LhsIndexerT &lhs_indexer,
                                const RhsIndexerT &rhs_indexer,
                                const ResIndexerT &res_indexer,
                                std::vector<sycl::event> const &depends)
{
    static constexpr GemmBatchFunctorThreadNM_vecm_HyperParametersSelector<
        resTy>
        selector{};
    static constexpr auto hyper_params = selector.get();

    static constexpr std::uint32_t wi_delta_n = hyper_params.get_wi_delta_n();
    static constexpr std::uint32_t wi_delta_m_vecs =
        hyper_params.get_wi_delta_m_vecs();
    static constexpr std::uint32_t m_vec_size = hyper_params.get_m_vec_size();

    static constexpr std::uint32_t wi_total_delta_m =
        wi_delta_m_vecs * m_vec_size;

    using KernelName =
        class gemm_batch_nm_vecm_krn<lhsTy, rhsTy, resTy, BatchIndexerT,
                                     LhsIndexerT, RhsIndexerT, ResIndexerT,
                                     wi_delta_n, wi_delta_m_vecs, m_vec_size>;

    const auto &kernel_id = sycl::get_kernel_id<KernelName>();

    auto const &ctx = exec_q.get_context();
    auto const &dev = exec_q.get_device();
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
    std::uint32_t wg_delta_m = 0;
    std::uint32_t wi_delta_k = 0;

    std::tie(wg_delta_m, wi_delta_k) =
        get_wg_delta_m_and_wi_delta_k<resTy, wi_delta_n, wi_total_delta_m>(
            slm_byte_size - reserved_slm_byte_size, wg_delta_n,
            max_subgroups_per_wg);

    const std::uint32_t lws = wg_delta_n * wg_delta_m;

    const std::size_t n_groups =
        (n + wg_delta_n * wi_delta_n - 1) / (wg_delta_n * wi_delta_n);
    const std::size_t m_groups = (m + wg_delta_m * wi_total_delta_m - 1) /
                                 (wg_delta_m * wi_total_delta_m);

    const std::size_t gws = lws * batch_nelems * n_groups * m_groups;

    sycl::range<1> lRange(lws);
    sycl::range<1> gRange(gws);
    sycl::nd_range<1> ndRange(gRange, lRange);

    using slmB_t =
        typename std::conditional<m_vec_size == 1, resTy,
                                  sycl::vec<resTy, m_vec_size>>::type;

    sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        cgh.use_kernel_bundle(kb);

        using LocAccT1 = sycl::local_accessor<resTy, 1>;
        LocAccT1 local_A_block(wg_delta_n * wi_delta_n * wi_delta_k, cgh);

        using LocAccT2 = sycl::local_accessor<slmB_t, 1>;
        LocAccT2 local_B_block(wg_delta_m * wi_delta_m_vecs * wi_delta_k, cgh);

        using Impl_FunctorT = GemmBatchFunctorThreadNM_vecm<
            lhsTy, rhsTy, resTy, LocAccT1, LocAccT2, BatchIndexerT, LhsIndexerT,
            RhsIndexerT, ResIndexerT, wi_delta_n, wi_delta_m_vecs, m_vec_size>;

        cgh.parallel_for<KernelName>(
            ndRange, Impl_FunctorT(
                         lhs_tp, rhs_tp, res_tp, std::move(local_A_block),
                         std::move(local_B_block), batch_nelems, n, k, m,
                         n_groups, wg_delta_n, wg_delta_m, wi_delta_k,
                         batch_indexer, lhs_indexer, rhs_indexer, res_indexer));
    });
    return gemm_ev;
}

} // namespace gemm_detail

typedef sycl::event (*gemm_impl_fn_ptr_t)(
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
    std::vector<sycl::event> const &);

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event gemm_impl(sycl::queue &exec_q,
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
                      std::vector<sycl::event> const &depends = {})
{
    const lhsTy *lhs_tp = reinterpret_cast<const lhsTy *>(lhs_cp);
    const rhsTy *rhs_tp = reinterpret_cast<const rhsTy *>(rhs_cp);
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp);

    using OuterInnerIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
    const OuterInnerIndexerT lhs_indexer(inner_nd + lhs_outer_nd, 0,
                                         lhs_shape_strides);
    const OuterInnerIndexerT rhs_indexer(inner_nd + rhs_outer_nd, 0,
                                         rhs_shape_strides);
    const OuterInnerIndexerT res_indexer(res_outer_nd, 0, res_shape_strides);

    using BatchIndexerT = dpctl::tensor::offset_utils::ThreeZeroOffsets_Indexer;
    static constexpr BatchIndexerT batch_indexer{};

    static constexpr std::size_t single_batch_nelems = 1;

    const std::size_t min_nm = std::min(n, m);
    const std::size_t max_nm = std::max(n, m);

    if (min_nm > 0 && (max_nm >= ((64 * 1024) / min_nm))) {
        return gemm_detail::_gemm_batch_nm_impl<
            lhsTy, rhsTy, resTy, BatchIndexerT, OuterInnerIndexerT,
            OuterInnerIndexerT, OuterInnerIndexerT>(
            exec_q, lhs_tp, rhs_tp, res_tp, single_batch_nelems, n, k, m,
            batch_indexer, lhs_indexer, rhs_indexer, res_indexer, depends);
    }

    sycl::event res_init_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using IndexerT = dpctl::tensor::offset_utils::StridedIndexer;
        const IndexerT res_indexer(res_outer_nd, 0, res_shape_strides);
        using InitKernelName = class gemm_init_krn<lhsTy, rhsTy, resTy>;
        cgh.parallel_for<InitKernelName>(
            sycl::range<1>(n * m), [=](sycl::id<1> id) {
                auto res_offset = res_indexer(id[0]);
                res_tp[res_offset] = resTy(0);
            });
    });

    if (k == 0) {
        return res_init_ev;
    }

    if ((max_nm < 64)) {
        if (m < 4) {
            return gemm_detail::_gemm_small_m_impl<
                lhsTy, rhsTy, resTy, BatchIndexerT, OuterInnerIndexerT,
                OuterInnerIndexerT, OuterInnerIndexerT>(
                exec_q, lhs_tp, rhs_tp, res_tp, single_batch_nelems, n, k, m,
                batch_indexer, lhs_indexer, rhs_indexer, res_indexer,
                {res_init_ev});
        }
        return gemm_detail::_gemm_k_impl<lhsTy, rhsTy, resTy, BatchIndexerT,
                                         OuterInnerIndexerT, OuterInnerIndexerT,
                                         OuterInnerIndexerT>(
            exec_q, lhs_tp, rhs_tp, res_tp, single_batch_nelems, n, k, m,
            batch_indexer, lhs_indexer, rhs_indexer, res_indexer,
            {res_init_ev});
    }

    return gemm_detail::_gemm_batch_nm_impl<
        lhsTy, rhsTy, resTy, BatchIndexerT, OuterInnerIndexerT,
        OuterInnerIndexerT, OuterInnerIndexerT>(
        exec_q, lhs_tp, rhs_tp, res_tp, single_batch_nelems, n, k, m,
        batch_indexer, lhs_indexer, rhs_indexer, res_indexer, {res_init_ev});
}

typedef sycl::event (*gemm_contig_impl_fn_ptr_t)(
    sycl::queue &,
    const char *, // lhs
    const char *, // rhs
    char *,       // res
    std::size_t,  // n
    std::size_t,  // k
    std::size_t,  // m
    std::vector<sycl::event> const &);

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event gemm_contig_impl(sycl::queue &exec_q,
                             const char *lhs_cp,
                             const char *rhs_cp,
                             char *res_cp,
                             std::size_t n,
                             std::size_t k,
                             std::size_t m,
                             std::vector<sycl::event> const &depends = {})
{
    const lhsTy *lhs_tp = reinterpret_cast<const lhsTy *>(lhs_cp);
    const rhsTy *rhs_tp = reinterpret_cast<const rhsTy *>(rhs_cp);
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp);

    using OuterInnerIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
    static constexpr OuterInnerIndexerT lhs_indexer{};
    static constexpr OuterInnerIndexerT rhs_indexer{};
    static constexpr OuterInnerIndexerT res_indexer{};

    using BatchIndexerT = dpctl::tensor::offset_utils::ThreeZeroOffsets_Indexer;
    static constexpr BatchIndexerT batch_indexer{};

    static constexpr std::size_t single_batch_nelems = 1;

    const std::size_t min_nm = std::min(n, m);
    const std::size_t max_nm = std::max(n, m);
    if (min_nm > 0 && (max_nm >= ((64 * 1024) / min_nm))) {
        return gemm_detail::_gemm_batch_nm_impl<
            lhsTy, rhsTy, resTy, BatchIndexerT, OuterInnerIndexerT,
            OuterInnerIndexerT, OuterInnerIndexerT>(
            exec_q, lhs_tp, rhs_tp, res_tp, single_batch_nelems, n, k, m,
            batch_indexer, lhs_indexer, rhs_indexer, res_indexer, depends);
    }

    sycl::event res_init_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.fill<resTy>(res_tp, resTy(0), n * m);
    });

    if (k == 0) {
        return res_init_ev;
    }

    if (max_nm < 64) {
        if (m < 4) {
            return gemm_detail::_gemm_small_m_impl<
                lhsTy, rhsTy, resTy, BatchIndexerT, OuterInnerIndexerT,
                OuterInnerIndexerT, OuterInnerIndexerT>(
                exec_q, lhs_tp, rhs_tp, res_tp, single_batch_nelems, n, k, m,
                batch_indexer, lhs_indexer, rhs_indexer, res_indexer,
                {res_init_ev});
        }
        return gemm_detail::_gemm_k_impl<lhsTy, rhsTy, resTy, BatchIndexerT,
                                         OuterInnerIndexerT, OuterInnerIndexerT,
                                         OuterInnerIndexerT>(
            exec_q, lhs_tp, rhs_tp, res_tp, single_batch_nelems, n, k, m,
            batch_indexer, lhs_indexer, rhs_indexer, res_indexer,
            {res_init_ev});
    }

    return gemm_detail::_gemm_batch_nm_impl<
        lhsTy, rhsTy, resTy, BatchIndexerT, OuterInnerIndexerT,
        OuterInnerIndexerT, OuterInnerIndexerT>(
        exec_q, lhs_tp, rhs_tp, res_tp, single_batch_nelems, n, k, m,
        batch_indexer, lhs_indexer, rhs_indexer, res_indexer, {res_init_ev});
}

template <typename T1, typename T2, typename T3>
class gemm_batch_init_krn;

typedef sycl::event (*gemm_batch_impl_fn_ptr_t)(
    sycl::queue &,
    const char *,    // lhs
    const char *,    // rhs
    char *,          // res
    std::size_t,     // batch nelems
    std::size_t,     // lhs outer nelems (n)
    std::size_t,     // inner nelems (k)
    std::size_t,     // rhs outer nelems (m)
    int,             // batching nd
    const ssize_t *, // batch shape strides
    ssize_t,         // lhs batch offset
    ssize_t,         // rhs batch offset
    ssize_t,         // res batch offset
    int,             // inner dims
    int,             // lhs outer dims
    const ssize_t *, // lhs outer and inner shape and strides
    int,             // rhs outer dims
    const ssize_t *, // rhs outer and inner shape and strides
    int,             // res outer dims
    const ssize_t *, // res outer and inner shape and strides
    const ssize_t *, // res full shape and strides
    std::vector<sycl::event> const &);

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event gemm_batch_impl(sycl::queue &exec_q,
                            const char *lhs_cp,
                            const char *rhs_cp,
                            char *res_cp,
                            std::size_t batch_nelems,
                            std::size_t n,
                            std::size_t k,
                            std::size_t m,
                            int batch_nd,
                            const ssize_t *batch_shape_strides,
                            ssize_t lhs_batch_offset,
                            ssize_t rhs_batch_offset,
                            ssize_t res_batch_offset,
                            int inner_nd,
                            int lhs_outer_nd,
                            const ssize_t *lhs_outer_inner_shapes_strides,
                            int rhs_outer_nd,
                            const ssize_t *rhs_outer_inner_shapes_strides,
                            int res_outer_nd,
                            const ssize_t *res_outer_shapes_strides,
                            const ssize_t *res_shape_strides,
                            std::vector<sycl::event> const &depends = {})
{
    const lhsTy *lhs_tp = reinterpret_cast<const lhsTy *>(lhs_cp);
    const rhsTy *rhs_tp = reinterpret_cast<const rhsTy *>(rhs_cp);
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp);

    using OuterInnerDimsIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
    const OuterInnerDimsIndexerT lhs_indexer(inner_nd + lhs_outer_nd, 0,
                                             lhs_outer_inner_shapes_strides);
    const OuterInnerDimsIndexerT rhs_indexer(inner_nd + rhs_outer_nd, 0,
                                             rhs_outer_inner_shapes_strides);
    const OuterInnerDimsIndexerT res_indexer(res_outer_nd, 0,
                                             res_outer_shapes_strides);
    using BatchDimsIndexerT =
        dpctl::tensor::offset_utils::ThreeOffsets_StridedIndexer;
    const BatchDimsIndexerT batch_indexer(batch_nd, lhs_batch_offset,
                                          rhs_batch_offset, res_batch_offset,
                                          batch_shape_strides);

    const std::size_t min_nm = std::min(n, m);
    const std::size_t max_nm = std::max(n, m);

    if (min_nm > 0 && (max_nm >= ((64 * 1024) / min_nm))) {
        return gemm_detail::_gemm_batch_nm_impl<
            lhsTy, rhsTy, resTy, BatchDimsIndexerT, OuterInnerDimsIndexerT,
            OuterInnerDimsIndexerT, OuterInnerDimsIndexerT>(
            exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
            batch_indexer, lhs_indexer, rhs_indexer, res_indexer, depends);
    }

    sycl::event res_init_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using IndexerT = dpctl::tensor::offset_utils::StridedIndexer;
        const IndexerT res_indexer(batch_nd + res_outer_nd, res_batch_offset,
                                   res_shape_strides);
        using InitKernelName = class gemm_batch_init_krn<lhsTy, rhsTy, resTy>;
        cgh.parallel_for<InitKernelName>(
            sycl::range<1>(n * m * batch_nelems), [=](sycl::id<1> id) {
                auto res_offset = res_indexer(id[0]);
                res_tp[res_offset] = resTy(0);
            });
    });

    if (k == 0) {
        return res_init_ev;
    }

    if (m < 4) {
        return gemm_detail::_gemm_small_m_impl<
            lhsTy, rhsTy, resTy, BatchDimsIndexerT, OuterInnerDimsIndexerT,
            OuterInnerDimsIndexerT, OuterInnerDimsIndexerT>(
            exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
            batch_indexer, lhs_indexer, rhs_indexer, res_indexer,
            {res_init_ev});
    }
    else if (k > n && k > m) {
        return gemm_detail::_gemm_k_impl<
            lhsTy, rhsTy, resTy, BatchDimsIndexerT, OuterInnerDimsIndexerT,
            OuterInnerDimsIndexerT, OuterInnerDimsIndexerT>(
            exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
            batch_indexer, lhs_indexer, rhs_indexer, res_indexer,
            {res_init_ev});
    }
    else {
        return gemm_detail::_gemm_batch_nm_impl<
            lhsTy, rhsTy, resTy, BatchDimsIndexerT, OuterInnerDimsIndexerT,
            OuterInnerDimsIndexerT, OuterInnerDimsIndexerT>(
            exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
            batch_indexer, lhs_indexer, rhs_indexer, res_indexer,
            {res_init_ev});
    }
}

typedef sycl::event (*gemm_batch_contig_impl_fn_ptr_t)(
    sycl::queue &,
    const char *, // lhs
    const char *, // rhs
    char *,       // res
    std::size_t,  // batch nelems
    std::size_t,  // n
    std::size_t,  // k
    std::size_t,  // m
    ssize_t,      // lhs batch offset
    ssize_t,      // rhs batch offset
    ssize_t,      // res batch offset
    std::vector<sycl::event> const &);

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event gemm_batch_contig_impl(sycl::queue &exec_q,
                                   const char *lhs_cp,
                                   const char *rhs_cp,
                                   char *res_cp,
                                   std::size_t batch_nelems,
                                   std::size_t n,
                                   std::size_t k,
                                   std::size_t m,
                                   ssize_t lhs_batch_offset,
                                   ssize_t rhs_batch_offset,
                                   ssize_t res_batch_offset,
                                   std::vector<sycl::event> const &depends = {})
{
    const lhsTy *lhs_tp =
        reinterpret_cast<const lhsTy *>(lhs_cp) + lhs_batch_offset;
    const rhsTy *rhs_tp =
        reinterpret_cast<const rhsTy *>(rhs_cp) + rhs_batch_offset;
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp) + res_batch_offset;

    using OuterInnerDimsIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
    static constexpr OuterInnerDimsIndexerT lhs_indexer{};
    static constexpr OuterInnerDimsIndexerT rhs_indexer{};
    static constexpr OuterInnerDimsIndexerT res_indexer{};

    using dpctl::tensor::offset_utils::Strided1DIndexer;
    using dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer;
    using BatchDimsIndexerT =
        ThreeOffsets_CombinedIndexer<Strided1DIndexer, Strided1DIndexer,
                                     Strided1DIndexer>;

    const BatchDimsIndexerT batch_indexer(
        Strided1DIndexer{/* size */ batch_nelems,
                         /* step */ n * k},
        Strided1DIndexer{/* size */ batch_nelems,
                         /* step */ k * m},
        Strided1DIndexer{/* size */ batch_nelems,
                         /* step */ n * m});

    const std::size_t min_nm = std::min(n, m);
    const std::size_t max_nm = std::max(n, m);

    if (min_nm > 0 && (max_nm >= ((64 * 1024) / min_nm))) {
        return gemm_detail::_gemm_batch_nm_impl<
            lhsTy, rhsTy, resTy, BatchDimsIndexerT, OuterInnerDimsIndexerT,
            OuterInnerDimsIndexerT, OuterInnerDimsIndexerT>(
            exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
            batch_indexer, lhs_indexer, rhs_indexer, res_indexer, depends);
    }

    sycl::event res_init_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.fill<resTy>(res_tp, resTy(0), n * m * batch_nelems);
    });

    if (k == 0) {
        return res_init_ev;
    }

    if (max_nm < 64) {
        if (m < 4) {
            return gemm_detail::_gemm_small_m_impl<
                lhsTy, rhsTy, resTy, BatchDimsIndexerT, OuterInnerDimsIndexerT,
                OuterInnerDimsIndexerT, OuterInnerDimsIndexerT>(
                exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
                batch_indexer, lhs_indexer, rhs_indexer, res_indexer,
                {res_init_ev});
        }
        return gemm_detail::_gemm_k_impl<
            lhsTy, rhsTy, resTy, BatchDimsIndexerT, OuterInnerDimsIndexerT,
            OuterInnerDimsIndexerT, OuterInnerDimsIndexerT>(
            exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
            batch_indexer, lhs_indexer, rhs_indexer, res_indexer,
            {res_init_ev});
    }

    return gemm_detail::_gemm_batch_nm_impl<
        lhsTy, rhsTy, resTy, BatchDimsIndexerT, OuterInnerDimsIndexerT,
        OuterInnerDimsIndexerT, OuterInnerDimsIndexerT>(
        exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m, batch_indexer,
        lhs_indexer, rhs_indexer, res_indexer, {res_init_ev});
}

// ========== Gemm Tree

template <typename lhsT,
          typename rhsT,
          typename resT,
          typename LocAccT1,
          typename LocAccT2,
          typename OuterInnerDimsIndexerT,
          typename ResIndexerT,
          typename BatchDimsIndexerT,
          int wi_delta_n,
          int wi_delta_m>
class GemmBatchNoAtomicFunctorThreadNM
{
private:
    const lhsT *lhs = nullptr;
    const rhsT *rhs = nullptr;
    resT *res = nullptr;
    LocAccT1 local_A_block;
    LocAccT2 local_B_block;
    std::size_t n = 0;
    std::size_t wg_delta_n = 0;
    std::size_t k = 0;
    std::size_t k_blocks = 0;
    std::size_t wi_delta_k = 0;
    std::size_t m = 0;
    std::size_t m_blocks = 0;
    std::size_t wg_delta_m = 0;
    std::size_t batch_nelems;
    BatchDimsIndexerT batch_indexer;
    OuterInnerDimsIndexerT lhs_indexer;
    OuterInnerDimsIndexerT rhs_indexer;
    ResIndexerT res_indexer;

public:
    GemmBatchNoAtomicFunctorThreadNM(const lhsT *lhs_,
                                     const rhsT *rhs_,
                                     resT *res_,
                                     LocAccT1 local_A_block_,
                                     LocAccT2 local_B_block_,
                                     std::size_t n_,
                                     std::size_t wg_delta_n_,
                                     std::size_t k_,
                                     std::size_t k_blocks_,
                                     std::size_t wi_delta_k_,
                                     std::size_t m_,
                                     std::size_t m_blocks_,
                                     std::size_t wg_delta_m_,
                                     std::size_t batch_nelems_,
                                     const BatchDimsIndexerT batch_indexer_,
                                     const OuterInnerDimsIndexerT lhs_indexer_,
                                     const OuterInnerDimsIndexerT rhs_indexer_,
                                     const ResIndexerT res_indexer_)
        : lhs(lhs_), rhs(rhs_), res(res_), local_A_block(local_A_block_),
          local_B_block(local_B_block_), n(n_), wg_delta_n(wg_delta_n_), k(k_),
          k_blocks(k_blocks_), wi_delta_k(wi_delta_k_), m(m_),
          m_blocks(m_blocks_), wg_delta_m(wg_delta_m_),
          batch_nelems(batch_nelems_), batch_indexer(batch_indexer_),
          lhs_indexer(lhs_indexer_), rhs_indexer(rhs_indexer_),
          res_indexer(res_indexer_)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        const std::size_t n_groups_per_batch =
            it.get_group_range(0) / batch_nelems;
        const std::size_t m_id = it.get_group_linear_id() / n_groups_per_batch;
        const std::size_t gr_id =
            it.get_group_linear_id() - m_id * n_groups_per_batch;

        const auto &three_offsets_ = batch_indexer(static_cast<ssize_t>(m_id));

        // lift group_id to (block_i, block_j, block_s),
        //    0 <= block_i < n_blocks, 0 <= block_j < m_blocks, 0 <= block_s
        //    < k_blocks

        const auto &lhs_offset = three_offsets_.get_first_offset();
        const auto &rhs_offset = three_offsets_.get_second_offset();
        const auto &res_offset = three_offsets_.get_third_offset();

        std::size_t block_i = gr_id / (m_blocks * k_blocks);
        std::size_t block_r = gr_id - block_i * (m_blocks * k_blocks);
        std::size_t block_j = block_r / k_blocks;
        std::size_t block_s = block_r - block_j * k_blocks;

        std::size_t lid = it.get_local_linear_id();
        std::size_t local_i = lid / wg_delta_m; // 0<= local_i < wg_delta_n
        std::size_t local_j =
            lid - local_i * wg_delta_m; // 0<= local_j < wg_delta_m

        // load A block and B blocks into SLM

        std::size_t i = block_i * wi_delta_n * wg_delta_n;
        std::size_t j = block_j * wi_delta_m * wg_delta_m;
        std::size_t s = block_s * wi_delta_k;

        const std::int64_t a_st0 = k;
        const std::int64_t a_st1 = 1;

        const std::int64_t b_st0 = m;
        const std::int64_t b_st1 = 1;

        const std::int64_t c_st0 = m;
        const std::int64_t c_st1 = 1;

        std::size_t lws = it.get_local_range(0);

        for (std::size_t vid = lid; vid < local_A_block.size(); vid += lws) {
            std::size_t v_i =
                vid / wi_delta_k; // 0<= v_i < wg_delta_n * wi_delta_n
            std::size_t v_s = vid - v_i * wi_delta_k; // 0<= v_s < wi_delta_k

            std::size_t g_i = i + v_i;
            std::size_t g_s = s + v_s;

            local_A_block[vid] =
                (g_i < n && g_s < k)
                    ? static_cast<resT>(
                          lhs[lhs_offset +
                              lhs_indexer(g_i * a_st0 + g_s * a_st1)])
                    : resT(0);
        }

        using slmB_t = typename LocAccT2::value_type;

        for (std::size_t vid = lid; vid < local_B_block.size(); vid += lws) {
            std::size_t v_j = vid / wi_delta_k;       // 0<= v_i < wg_delta_m
            std::size_t v_s = vid - v_j * wi_delta_k; // 0<= v_s < wi_delta_k

            std::size_t g_j = j + v_j * wi_delta_m;
            std::size_t g_s = s + v_s;

            if constexpr (wi_delta_m == 1 && std::is_same_v<slmB_t, resT>) {
                local_B_block[vid] =
                    (g_j < m && g_s < k)
                        ? static_cast<resT>(
                              rhs[rhs_offset +
                                  rhs_indexer(g_s * b_st0 + g_j * b_st1)])
                        : resT(0);
            }
            else {
                slmB_t vec{};
#pragma unroll
                for (std::uint8_t lane_id = 0; lane_id < wi_delta_m; ++lane_id)
                {
                    std::size_t g_j1 = g_j + lane_id;
                    vec[lane_id] =
                        (g_j1 < m && g_s < k)
                            ? static_cast<resT>(
                                  rhs[rhs_offset +
                                      rhs_indexer(g_s * b_st0 + g_j1 * b_st1)])
                            : resT(0);
                }

                local_B_block[vid] = vec;
            }
        }

        it.barrier(sycl::access::fence_space::local_space);

        i += local_i * wi_delta_n;
        j += local_j * wi_delta_m;

        const std::size_t a_offset = local_i * wi_delta_k * wi_delta_n;
        const std::size_t b_offset = local_j * wi_delta_k;

        static constexpr resT identity_(0);

        for (std::uint8_t private_i = 0; private_i < wi_delta_n; ++private_i) {
            const std::size_t a_pr_offset = private_i * wi_delta_k;

            slmB_t local_sum(identity_);
            for (std::size_t private_s = 0; private_s < wi_delta_k; ++private_s)
            {
                local_sum = local_sum +
                            (local_A_block[a_offset + a_pr_offset + private_s] *
                             local_B_block[b_offset + private_s]);
            }

            const std::size_t gl_i = i + private_i;

            if constexpr (wi_delta_m == 1 && std::is_same_v<slmB_t, resT>) {
                const std::size_t gl_j = j;
                if (gl_i < n && gl_j < m) {
                    res[res_offset + res_indexer(gl_i * c_st0 + gl_j * c_st1) +
                        (block_s * n * m * batch_nelems)] = local_sum;
                }
            }
            else {
#pragma unroll
                for (std::uint8_t lane_id = 0; lane_id < wi_delta_m; ++lane_id)
                {
                    const std::size_t gl_j = j + lane_id;

                    if (gl_i < n && gl_j < m) {
                        res[res_offset +
                            res_indexer(gl_i * c_st0 + gl_j * c_st1) +
                            (block_s * n * m * batch_nelems)] =
                            local_sum[lane_id];
                    }
                }
            }
        }
    }
};

template <typename lhsT,
          typename rhsT,
          typename resT,
          typename LocAccT,
          typename OuterInnerDimsIndexerT,
          typename ResIndexerT,
          typename BatchDimsIndexerT,
          std::size_t m_groups>
class GemmBatchNoAtomicFunctorThreadK
{
private:
    const lhsT *lhs = nullptr;
    const rhsT *rhs = nullptr;
    resT *res = nullptr;
    LocAccT workspace;
    LocAccT local_B_block;
    std::size_t n = 0;
    std::size_t n_blocks = 0;
    std::size_t delta_n = 0;
    std::size_t k = 0;
    std::size_t k_blocks = 0;
    std::size_t delta_k = 0;
    std::size_t n_wi = 0;
    std::size_t m = 0;
    std::size_t batch_nelems = 0;
    BatchDimsIndexerT batch_indexer;
    OuterInnerDimsIndexerT lhs_indexer;
    OuterInnerDimsIndexerT rhs_indexer;
    ResIndexerT res_indexer;

public:
    GemmBatchNoAtomicFunctorThreadK(const lhsT *lhs_,
                                    const rhsT *rhs_,
                                    resT *res_,
                                    LocAccT workspace_,
                                    LocAccT local_B_block_,
                                    std::size_t n_,
                                    std::size_t n_blocks_,
                                    std::size_t delta_n_,
                                    std::size_t k_,
                                    std::size_t k_blocks_,
                                    std::size_t delta_k_,
                                    std::size_t n_wi_,
                                    std::size_t m_,
                                    std::size_t batch_nelems_,
                                    const BatchDimsIndexerT &batch_indexer_,
                                    const OuterInnerDimsIndexerT &lhs_indexer_,
                                    const OuterInnerDimsIndexerT &rhs_indexer_,
                                    const ResIndexerT &res_indexer_)
        : lhs(lhs_), rhs(rhs_), res(res_), workspace(workspace_),
          local_B_block(local_B_block_), n(n_), n_blocks(n_blocks_),
          delta_n(delta_n_), k(k_), k_blocks(k_blocks_), delta_k(delta_k_),
          n_wi(n_wi_), m(m_), batch_nelems(batch_nelems_),
          batch_indexer(batch_indexer_), lhs_indexer(lhs_indexer_),
          rhs_indexer(rhs_indexer_), res_indexer(res_indexer_)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        const std::size_t n_groups_per_batch =
            it.get_group_range(0) / batch_nelems;
        const std::size_t m_id = it.get_group_linear_id() / n_groups_per_batch;
        const std::size_t gr_id =
            it.get_group_linear_id() - m_id * n_groups_per_batch;
        std::size_t lid = it.get_local_linear_id();

        const auto &three_offsets_ = batch_indexer(static_cast<ssize_t>(m_id));
        const auto &lhs_offset = three_offsets_.get_first_offset();
        const auto &rhs_offset = three_offsets_.get_second_offset();
        const auto &res_offset = three_offsets_.get_third_offset();

        // lift gr_id -> (block_i, block_j, block_s)
        //   block_i moves fastest, then block_s, then block_j

        const std::size_t r_size = (n_blocks * k_blocks);
        // 0 <= block_j < m_blocks
        std::size_t block_j = gr_id / r_size;
        // 0 <= block_r < n_blocks * k_blocks
        std::size_t block_r = gr_id - block_j * r_size;
        // 0 <= block_s < k_blocks
        std::size_t block_s = block_r / n_blocks;
        // 0 <= block_i < n_blocks
        std::size_t block_i = block_r - block_s * n_blocks;

        std::size_t local_i = lid / (delta_k); // 0 <= local_i < delta_n
        std::size_t local_s =
            lid - local_i * (delta_k); // 0 <= local_s < delta_k

        std::size_t i = block_i * delta_n + local_i;
        std::size_t j = m_groups * block_j;
        std::size_t s = block_s * delta_k * n_wi + local_s;

        using accV_t = typename LocAccT::value_type;

        static constexpr resT identity_ = resT(0);
        if (local_i == 0) {
            for (std::size_t q = 0; q < n_wi * delta_k; q += delta_k) {
                std::size_t sq = s + q;
                std::size_t sqmj = sq * m + j;

                if constexpr (m_groups == 1 && std::is_same_v<accV_t, resT>) {
                    local_B_block[local_s + q] =
                        (sq < k && j < m)
                            ? static_cast<resT>(
                                  rhs[rhs_offset + rhs_indexer(sqmj)])
                            : identity_;
                }
                else {
                    accV_t local_B_vec;
#pragma unroll
                    for (std::size_t vec_idx = 0; vec_idx < m_groups; ++vec_idx)
                    {
                        local_B_vec[vec_idx] =
                            (sq < k && j + vec_idx < m)
                                ? static_cast<resT>(
                                      rhs[rhs_offset +
                                          rhs_indexer(sqmj + vec_idx)])
                                : identity_;
                    }
                    local_B_block[local_s + q] = local_B_vec;
                }
            }
        }

        it.barrier(sycl::access::fence_space::local_space);

        std::size_t t_shift = block_s * delta_k * n_wi;
        std::size_t global_s_offset = i * k + t_shift;

        accV_t private_sum(identity_);
        static constexpr accV_t vec_identity_(identity_);
        for (std::size_t t = local_s; t < local_B_block.size(); t += delta_k) {
            private_sum +=
                ((i < n) && (t + t_shift < k))
                    ? (static_cast<resT>(
                           lhs[lhs_offset + lhs_indexer(global_s_offset + t)]) *
                       local_B_block[t])
                    : vec_identity_;
        }

        std::size_t workspace_i_shift = local_i * delta_k;
        workspace[workspace_i_shift + local_s] = private_sum;

        it.barrier(sycl::access::fence_space::local_space);

        if (local_s == 0 && i < n) {
            accV_t local_sum(workspace[workspace_i_shift]);
            for (std::size_t t = 1; t < delta_k; ++t) {
                local_sum += workspace[workspace_i_shift + t];
            }

            const std::size_t total_offset =
                res_offset + (block_s * n * m * batch_nelems);

            if constexpr (m_groups == 1 && std::is_same_v<accV_t, resT>) {
                res[total_offset + res_indexer(i * m + j)] = local_sum;
            }
            else {
                res[total_offset + res_indexer(i * m + j)] = local_sum[0];

#pragma unroll
                for (std::size_t vec_id = 1; vec_id < m_groups; ++vec_id) {
                    if (j + vec_id < m) {
                        res[total_offset + res_indexer(i * m + j + vec_id)] =
                            local_sum[vec_id];
                    }
                }
            }
        }
    }
};

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          std::size_t>
class gemm_batch_tree_k_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          std::size_t>
class gemm_batch_tree_nm_krn;

namespace gemm_detail
{

template <typename lhsTy,
          typename rhsTy,
          typename resTy,
          typename BatchIndexerT,
          typename LhsIndexerT,
          typename RhsIndexerT,
          typename ResIndexerT,
          std::uint32_t m_groups>
sycl::event _gemm_tree_k_step(sycl::queue &exec_q,
                              const lhsTy *lhs_tp,
                              const rhsTy *rhs_tp,
                              resTy *res_tp,
                              const std::size_t batch_nelems,
                              const std::size_t n,
                              const std::size_t k,
                              const std::size_t m,
                              const std::size_t delta_n,
                              const std::size_t n_wi,
                              const std::size_t delta_k,
                              const BatchIndexerT &batch_indexer,
                              const LhsIndexerT &lhs_indexer,
                              const RhsIndexerT &rhs_indexer,
                              const ResIndexerT &res_indexer,
                              const std::vector<sycl::event> &depends)
{
    static_assert(std::is_same_v<LhsIndexerT, RhsIndexerT>);

    sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        const std::size_t n_blocks = (n + delta_n - 1) / delta_n;
        const std::size_t k_blocks =
            (k + n_wi * delta_k - 1) / (n_wi * delta_k);
        const std::size_t m_blocks = (m + m_groups - 1) / m_groups;

        const std::size_t lws = delta_n * delta_k;
        const std::size_t gws =
            batch_nelems * n_blocks * m_blocks * k_blocks * lws;

        auto gRange = sycl::range<1>(gws);
        auto lRange = sycl::range<1>(lws);
        auto ndRange = sycl::nd_range<1>(gRange, lRange);

        using slmB_t =
            typename std::conditional<m_groups == 1, resTy,
                                      sycl::vec<resTy, m_groups>>::type;

        using LocAccT = sycl::local_accessor<slmB_t, 1>;
        LocAccT local_B_block(n_wi * delta_k, cgh);
        LocAccT workspace(delta_n * delta_k, cgh);

        using KernelName =
            class gemm_batch_tree_k_krn<lhsTy, rhsTy, resTy, LhsIndexerT,
                                        ResIndexerT, BatchIndexerT, m_groups>;

        cgh.parallel_for<KernelName>(
            ndRange,
            GemmBatchNoAtomicFunctorThreadK<lhsTy, rhsTy, resTy, LocAccT,
                                            LhsIndexerT, ResIndexerT,
                                            BatchIndexerT, m_groups>(
                lhs_tp, rhs_tp, res_tp, std::move(workspace),
                std::move(local_B_block), n, n_blocks, delta_n, k, k_blocks,
                delta_k, n_wi, m, batch_nelems, batch_indexer, lhs_indexer,
                rhs_indexer, res_indexer));
    });
    return gemm_ev;
}

} // end of namespace gemm_detail

template <typename lhsTy,
          typename rhsTy,
          typename resTy,
          std::uint32_t m_groups>
sycl::event
    gemm_batch_tree_k_impl(sycl::queue &exec_q,
                           const lhsTy *lhs_tp,
                           const rhsTy *rhs_tp,
                           resTy *res_tp,
                           std::size_t batch_nelems,
                           std::size_t n,
                           std::size_t k,
                           std::size_t m,
                           int batch_nd,
                           const ssize_t *batch_shape_strides,
                           ssize_t lhs_batch_offset,
                           ssize_t rhs_batch_offset,
                           ssize_t res_batch_offset,
                           int inner_nd,
                           int lhs_outer_nd,
                           const ssize_t *lhs_outer_inner_shapes_strides,
                           int rhs_outer_nd,
                           const ssize_t *rhs_outer_inner_shapes_strides,
                           int res_outer_nd,
                           const ssize_t *res_outer_shapes_strides,
                           const ssize_t *res_shape_strides,
                           std::vector<sycl::event> const &depends)
{
    std::size_t delta_k(4);
    std::size_t n_wi(64);
    std::size_t delta_n(32);

    const sycl::device &dev = exec_q.get_device();
    const std::size_t local_mem_size =
        dev.get_info<sycl::info::device::local_mem_size>();
    const std::size_t reserved_slm_size = 512;

    gemm_detail::scale_gemm_k_parameters<resTy, m_groups>(
        local_mem_size, reserved_slm_size, delta_k,
        n_wi,   // modified by reference
        delta_n // modified by reference
    );

    if (k <= (delta_k * n_wi)) {
        using OuterInnerDimsIndexerT =
            dpctl::tensor::offset_utils::StridedIndexer;
        const OuterInnerDimsIndexerT lhs_indexer(
            inner_nd + lhs_outer_nd, 0, lhs_outer_inner_shapes_strides);
        const OuterInnerDimsIndexerT rhs_indexer(
            inner_nd + rhs_outer_nd, 0, rhs_outer_inner_shapes_strides);
        const OuterInnerDimsIndexerT res_indexer(res_outer_nd, 0,
                                                 res_outer_shapes_strides);
        using BatchDimsIndexerT =
            dpctl::tensor::offset_utils::ThreeOffsets_StridedIndexer;
        const BatchDimsIndexerT batch_indexer(
            batch_nd, lhs_batch_offset, rhs_batch_offset, res_batch_offset,
            batch_shape_strides);

        return gemm_detail::_gemm_tree_k_step<
            lhsTy, rhsTy, resTy, BatchDimsIndexerT, OuterInnerDimsIndexerT,
            OuterInnerDimsIndexerT, OuterInnerDimsIndexerT, m_groups>(
            exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m, delta_n,
            n_wi, delta_k, batch_indexer, lhs_indexer, rhs_indexer, res_indexer,
            depends);
    }
    else {
        using ReductionOpT =
            typename std::conditional<std::is_same_v<resTy, bool>,
                                      sycl::logical_or<resTy>,
                                      sycl::plus<resTy>>::type;
        static constexpr resTy identity_val =
            sycl::known_identity<ReductionOpT, resTy>::value;

        std::size_t iter_nelems = batch_nelems * n * m;
        std::size_t reduction_nelems =
            (k + delta_k * n_wi - 1) / (delta_k * n_wi);

        // more than one work-group is needed, requires a
        // temporary delta_k * n_wi elements processed along k,
        // so if more to process use multiple
        const auto &sg_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();
        std::size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

        static constexpr std::size_t preferred_reductions_per_wi = 4;
        std::size_t reductions_per_wi(preferred_reductions_per_wi);

        std::size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);

        // max_max_wg prevents running out of resources on CPU
        static constexpr std::size_t max_max_wg = 2048;
        std::size_t max_wg = std::min(
            max_max_wg,
            dev.get_info<sycl::info::device::max_work_group_size>() / 2);

        if (reduction_nelems <= preferred_reductions_per_wi * max_wg) {
            auto tmp_owner =
                dpctl::tensor::alloc_utils::smart_malloc_device<resTy>(
                    iter_nelems * reduction_nelems, exec_q);
            resTy *tmp = tmp_owner.get();

            using OuterInnerDimsIndexerT =
                dpctl::tensor::offset_utils::StridedIndexer;
            using TmpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            const OuterInnerDimsIndexerT lhs_indexer(
                inner_nd + lhs_outer_nd, 0, lhs_outer_inner_shapes_strides);
            const OuterInnerDimsIndexerT rhs_indexer(
                inner_nd + rhs_outer_nd, 0, rhs_outer_inner_shapes_strides);
            static constexpr TmpIndexerT res_indexer{};

            using dpctl::tensor::offset_utils::Strided1DIndexer;
            using dpctl::tensor::offset_utils::StridedIndexer;
            using dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer;
            using dpctl::tensor::offset_utils::UnpackedStridedIndexer;
            using BatchDimsIndexerT = ThreeOffsets_CombinedIndexer<
                StridedIndexer, UnpackedStridedIndexer, Strided1DIndexer>;
            const StridedIndexer lhs_batch_indexer(batch_nd, lhs_batch_offset,
                                                   batch_shape_strides);
            const UnpackedStridedIndexer rhs_batch_indexer(
                batch_nd, rhs_batch_offset, batch_shape_strides,
                batch_shape_strides + 2 * batch_nd);
            const Strided1DIndexer tmp_batch_indexer(
                /* size   */ batch_nelems,
                /* step   */ n * m);
            const BatchDimsIndexerT batch_indexer(
                lhs_batch_indexer, rhs_batch_indexer, tmp_batch_indexer);

            sycl::event gemm_ev = gemm_detail::_gemm_tree_k_step<
                lhsTy, rhsTy, resTy, BatchDimsIndexerT, OuterInnerDimsIndexerT,
                OuterInnerDimsIndexerT, TmpIndexerT, m_groups>(
                exec_q, lhs_tp, rhs_tp, tmp, batch_nelems, n, k, m, delta_n,
                n_wi, delta_k, batch_indexer, lhs_indexer, rhs_indexer,
                res_indexer, depends);

            sycl::event red_ev = single_reduction_for_gemm<resTy, ReductionOpT>(
                exec_q, tmp, res_tp, identity_val, iter_nelems,
                reduction_nelems, reduction_groups, wg, max_wg,
                preferred_reductions_per_wi, reductions_per_wi,
                batch_nd + res_outer_nd, res_batch_offset, res_shape_strides,
                {gemm_ev});

            sycl::event cleanup_host_task_event =
                dpctl::tensor::alloc_utils::async_smart_free(exec_q, {red_ev},
                                                             tmp_owner);

            return cleanup_host_task_event;
        }
        else {
            assert(reduction_groups > 1);

            const std::size_t tmp_alloc_size =
                iter_nelems * (
                                  /* temp */ reduction_nelems +
                                  /* first reduction temp */ reduction_groups);

            // get unique_ptr owning the temporary allocation
            auto tmp_owner =
                dpctl::tensor::alloc_utils::smart_malloc_device<resTy>(
                    tmp_alloc_size, exec_q);
            // get raw USM pointer
            resTy *partially_reduced_tmp = tmp_owner.get();
            resTy *partially_reduced_tmp2 =
                partially_reduced_tmp + reduction_nelems * iter_nelems;
            ;

            using OuterInnerDimsIndexerT =
                dpctl::tensor::offset_utils::StridedIndexer;
            using TmpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            const OuterInnerDimsIndexerT lhs_indexer(
                inner_nd + lhs_outer_nd, 0, lhs_outer_inner_shapes_strides);
            const OuterInnerDimsIndexerT rhs_indexer(
                inner_nd + rhs_outer_nd, 0, rhs_outer_inner_shapes_strides);
            static constexpr TmpIndexerT res_indexer{};
            using dpctl::tensor::offset_utils::Strided1DIndexer;
            using dpctl::tensor::offset_utils::StridedIndexer;
            using dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer;
            using BatchDimsIndexerT =
                ThreeOffsets_CombinedIndexer<StridedIndexer, StridedIndexer,
                                             Strided1DIndexer>;
            const StridedIndexer lhs_batch_indexer(batch_nd, lhs_batch_offset,
                                                   batch_shape_strides);
            const StridedIndexer rhs_batch_indexer(
                batch_nd, rhs_batch_offset, batch_shape_strides + 2 * batch_nd);
            const Strided1DIndexer tmp_batch_indexer(
                /* size   */ batch_nelems,
                /* step   */ n * m);
            const BatchDimsIndexerT batch_indexer(
                lhs_batch_indexer, rhs_batch_indexer, tmp_batch_indexer);

            sycl::event gemm_ev = gemm_detail::_gemm_tree_k_step<
                lhsTy, rhsTy, resTy, BatchDimsIndexerT, OuterInnerDimsIndexerT,
                OuterInnerDimsIndexerT, TmpIndexerT, m_groups>(
                exec_q, lhs_tp, rhs_tp, partially_reduced_tmp, batch_nelems, n,
                k, m, delta_n, n_wi, delta_k, batch_indexer, lhs_indexer,
                rhs_indexer, res_indexer, depends);

            sycl::event red_ev = tree_reduction_for_gemm<resTy, ReductionOpT>(
                exec_q, partially_reduced_tmp, partially_reduced_tmp2, res_tp,
                identity_val, iter_nelems, reduction_nelems, reduction_groups,
                wg, max_wg, preferred_reductions_per_wi, reductions_per_wi,
                batch_nd + res_outer_nd, res_batch_offset, res_shape_strides,
                {gemm_ev});

            sycl::event cleanup_host_task_event =
                dpctl::tensor::alloc_utils::async_smart_free(exec_q, {red_ev},
                                                             tmp_owner);

            return cleanup_host_task_event;
        }
    }
}

namespace gemm_detail
{

template <typename lhsTy,
          typename rhsTy,
          typename resTy,
          typename BatchIndexerT,
          typename LhsIndexerT,
          typename RhsIndexerT,
          typename ResIndexerT,
          std::uint32_t wi_delta_n,
          std::uint32_t wi_delta_m>
sycl::event _gemm_tree_nm_step(sycl::queue &exec_q,
                               const lhsTy *lhs_tp,
                               const rhsTy *rhs_tp,
                               resTy *res_tp,
                               const std::size_t batch_nelems,
                               const std::size_t n,
                               const std::size_t k,
                               const std::size_t m,
                               const std::uint32_t wg_delta_n,
                               const std::uint32_t wg_delta_m,
                               const std::uint32_t wi_delta_k,
                               const BatchIndexerT &batch_indexer,
                               const LhsIndexerT &lhs_indexer,
                               const RhsIndexerT &rhs_indexer,
                               const ResIndexerT &res_indexer,
                               const std::vector<sycl::event> &depends)
{
    static_assert(std::is_same_v<LhsIndexerT, RhsIndexerT>);

    sycl::event gemm_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        const std::size_t lws = wg_delta_n * wg_delta_m;

        const std::size_t n_blocks =
            ((n + wi_delta_n * wg_delta_n - 1) / (wi_delta_n * wg_delta_n));
        const std::size_t k_blocks = ((k + wi_delta_k - 1) / wi_delta_k);
        const std::size_t m_blocks =
            ((m + wi_delta_m * wg_delta_m - 1) / (wi_delta_m * wg_delta_m));

        const std::size_t gws =
            batch_nelems * n_blocks * m_blocks * k_blocks * lws;

        auto gwsRange = sycl::range<1>(gws);
        auto lwsRange = sycl::range<1>(lws);
        auto ndRange = sycl::nd_range<1>(gwsRange, lwsRange);

        using slmB_t =
            typename std::conditional<wi_delta_m == 1, resTy,
                                      sycl::vec<resTy, wi_delta_m>>::type;
        using LocAccT1 = sycl::local_accessor<resTy, 1>;
        using LocAccT2 = sycl::local_accessor<slmB_t, 1>;

        const sycl::range<1> local_A_size((wi_delta_n * wg_delta_n) *
                                          wi_delta_k);
        const sycl::range<1> local_B_size(wi_delta_k * wg_delta_m);

        LocAccT1 local_A_block(local_A_size, cgh);
        LocAccT2 local_B_block(local_B_size, cgh);

        using KernelName =
            class gemm_batch_tree_nm_krn<lhsTy, rhsTy, resTy, LhsIndexerT,
                                         ResIndexerT, BatchIndexerT,
                                         wi_delta_m>;
        cgh.parallel_for<KernelName>(
            ndRange, GemmBatchNoAtomicFunctorThreadNM<
                         lhsTy, rhsTy, resTy, LocAccT1, LocAccT2, LhsIndexerT,
                         ResIndexerT, BatchIndexerT, wi_delta_n, wi_delta_m>(
                         lhs_tp, rhs_tp, res_tp, std::move(local_A_block),
                         std::move(local_B_block), n, wg_delta_n, k, k_blocks,
                         wi_delta_k, m, m_blocks, wg_delta_m, batch_nelems,
                         batch_indexer, lhs_indexer, rhs_indexer, res_indexer));
    });
    return gemm_ev;
}

} // end namespace gemm_detail

template <typename lhsTy, typename rhsTy, typename resTy, int wi_delta_m>
sycl::event
    gemm_batch_tree_nm_impl(sycl::queue &exec_q,
                            const lhsTy *lhs_tp,
                            const rhsTy *rhs_tp,
                            resTy *res_tp,
                            std::size_t batch_nelems,
                            std::size_t n,
                            std::size_t k,
                            std::size_t m,
                            int batch_nd,
                            const ssize_t *batch_shape_strides,
                            ssize_t lhs_batch_offset,
                            ssize_t rhs_batch_offset,
                            ssize_t res_batch_offset,
                            int inner_nd,
                            int lhs_outer_nd,
                            const ssize_t *lhs_outer_inner_shapes_strides,
                            int rhs_outer_nd,
                            const ssize_t *rhs_outer_inner_shapes_strides,
                            int res_outer_nd,
                            const ssize_t *res_outer_shapes_strides,
                            const ssize_t *res_shape_strides,
                            std::vector<sycl::event> const &depends)
{
    static constexpr int wi_delta_n = 2;
    std::size_t wg_delta_n(16); // rows of A processed in WG
    std::size_t wg_delta_m(16); // rows of B processed in WG
    std::size_t wi_delta_k(64); // Elements in K dimension processed by WI

    const sycl::device &dev = exec_q.get_device();
    const std::size_t local_mem_size =
        dev.get_info<sycl::info::device::local_mem_size>();
    const std::size_t reserved_slm_size = 512;

    gemm_detail::scale_gemm_nm_parameters<resTy, wi_delta_m>(
        local_mem_size, reserved_slm_size, wi_delta_n,
        wi_delta_k, // modified by reference
        wg_delta_n, // modified by reference
        wg_delta_m  // modified by reference
    );

    // each group processes delta_k * n_wi
    // items in a column, so no need for allocating
    // temp memory if only one group is needed
    if (k <= wi_delta_k) {
        using OuterInnerDimsIndexerT =
            dpctl::tensor::offset_utils::StridedIndexer;
        const OuterInnerDimsIndexerT lhs_indexer(
            inner_nd + lhs_outer_nd, 0, lhs_outer_inner_shapes_strides);
        const OuterInnerDimsIndexerT rhs_indexer(
            inner_nd + rhs_outer_nd, 0, rhs_outer_inner_shapes_strides);
        const OuterInnerDimsIndexerT res_indexer(res_outer_nd, 0,
                                                 res_outer_shapes_strides);
        using BatchDimsIndexerT =
            dpctl::tensor::offset_utils::ThreeOffsets_StridedIndexer;
        const BatchDimsIndexerT batch_indexer(
            batch_nd, lhs_batch_offset, rhs_batch_offset, res_batch_offset,
            batch_shape_strides);

        return gemm_detail::_gemm_tree_nm_step<
            lhsTy, rhsTy, resTy, BatchDimsIndexerT, OuterInnerDimsIndexerT,
            OuterInnerDimsIndexerT, OuterInnerDimsIndexerT, wi_delta_n,
            wi_delta_m>(exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
                        wg_delta_n, wg_delta_m, wi_delta_k, batch_indexer,
                        lhs_indexer, rhs_indexer, res_indexer, depends);
    }
    else {
        using ReductionOpT =
            typename std::conditional<std::is_same_v<resTy, bool>,
                                      sycl::logical_or<resTy>,
                                      sycl::plus<resTy>>::type;
        static constexpr resTy identity_val =
            sycl::known_identity<ReductionOpT, resTy>::value;
        std::size_t iter_nelems = batch_nelems * n * m;
        std::size_t reduction_nelems = (k + wi_delta_k - 1) / wi_delta_k;

        // more than one work-group is needed, requires a temporary
        // delta_k * n_wi elements processed along k, so if more to
        // process use multiple
        const auto &sg_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();
        std::size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

        static constexpr std::size_t preferred_reductions_per_wi = 4;
        std::size_t reductions_per_wi(preferred_reductions_per_wi);

        std::size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);

        std::size_t max_wg = reduction_detail::get_work_group_size(dev);

        if (reduction_nelems <= preferred_reductions_per_wi * max_wg) {
            auto tmp_owner =
                dpctl::tensor::alloc_utils::smart_malloc_device<resTy>(
                    iter_nelems * reduction_nelems, exec_q);
            resTy *tmp = tmp_owner.get();

            using OuterInnerDimsIndexerT =
                dpctl::tensor::offset_utils::StridedIndexer;
            using TmpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            const OuterInnerDimsIndexerT lhs_indexer(
                inner_nd + lhs_outer_nd, 0, lhs_outer_inner_shapes_strides);
            const OuterInnerDimsIndexerT rhs_indexer(
                inner_nd + rhs_outer_nd, 0, rhs_outer_inner_shapes_strides);
            static constexpr TmpIndexerT res_indexer{};

            using dpctl::tensor::offset_utils::Strided1DIndexer;
            using dpctl::tensor::offset_utils::StridedIndexer;
            using dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer;
            using dpctl::tensor::offset_utils::UnpackedStridedIndexer;
            using BatchDimsIndexerT = ThreeOffsets_CombinedIndexer<
                StridedIndexer, UnpackedStridedIndexer, Strided1DIndexer>;
            const StridedIndexer lhs_batch_indexer(batch_nd, lhs_batch_offset,
                                                   batch_shape_strides);
            const UnpackedStridedIndexer rhs_batch_indexer(
                batch_nd, rhs_batch_offset, batch_shape_strides,
                batch_shape_strides + 2 * batch_nd);
            const Strided1DIndexer tmp_batch_indexer(
                /* size   */ batch_nelems,
                /* step   */ n * m);
            const BatchDimsIndexerT batch_indexer(
                lhs_batch_indexer, rhs_batch_indexer, tmp_batch_indexer);

            sycl::event gemm_ev = gemm_detail::_gemm_tree_nm_step<
                lhsTy, rhsTy, resTy, BatchDimsIndexerT, OuterInnerDimsIndexerT,
                OuterInnerDimsIndexerT, TmpIndexerT, wi_delta_n, wi_delta_m>(
                exec_q, lhs_tp, rhs_tp, tmp, batch_nelems, n, k, m, wg_delta_n,
                wg_delta_m, wi_delta_k, batch_indexer, lhs_indexer, rhs_indexer,
                res_indexer, depends);

            sycl::event red_ev = single_reduction_for_gemm<resTy, ReductionOpT>(
                exec_q, tmp, res_tp, identity_val, iter_nelems,
                reduction_nelems, reduction_groups, wg, max_wg,
                preferred_reductions_per_wi, reductions_per_wi,
                batch_nd + res_outer_nd, res_batch_offset, res_shape_strides,
                {gemm_ev});

            sycl::event cleanup_host_task_event =
                dpctl::tensor::alloc_utils::async_smart_free(exec_q, {red_ev},
                                                             tmp_owner);

            return cleanup_host_task_event;
        }
        else {
            assert(reduction_groups > 1);

            const std::size_t tmp_alloc_size =
                iter_nelems * (/* temp */ reduction_nelems +
                               /* first reduction temp */ reduction_groups);

            auto tmp_owner =
                dpctl::tensor::alloc_utils::smart_malloc_device<resTy>(
                    tmp_alloc_size, exec_q);

            resTy *partially_reduced_tmp = tmp_owner.get();
            resTy *partially_reduced_tmp2 =
                partially_reduced_tmp + reduction_nelems * iter_nelems;
            ;

            using OuterInnerDimsIndexerT =
                dpctl::tensor::offset_utils::StridedIndexer;
            using TmpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;

            const OuterInnerDimsIndexerT lhs_indexer(
                inner_nd + lhs_outer_nd, 0, lhs_outer_inner_shapes_strides);
            const OuterInnerDimsIndexerT rhs_indexer(
                inner_nd + rhs_outer_nd, 0, rhs_outer_inner_shapes_strides);
            static constexpr TmpIndexerT res_indexer{};

            using dpctl::tensor::offset_utils::Strided1DIndexer;
            using dpctl::tensor::offset_utils::StridedIndexer;
            using dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer;
            using dpctl::tensor::offset_utils::UnpackedStridedIndexer;
            using BatchDimsIndexerT = ThreeOffsets_CombinedIndexer<
                StridedIndexer, UnpackedStridedIndexer, Strided1DIndexer>;

            const StridedIndexer lhs_batch_indexer(batch_nd, lhs_batch_offset,
                                                   batch_shape_strides);
            const UnpackedStridedIndexer rhs_batch_indexer(
                batch_nd, rhs_batch_offset, batch_shape_strides,
                batch_shape_strides + 2 * batch_nd);
            const Strided1DIndexer tmp_batch_indexer(
                /* size   */ batch_nelems,
                /* step   */ n * m);
            const BatchDimsIndexerT batch_indexer(
                lhs_batch_indexer, rhs_batch_indexer, tmp_batch_indexer);

            sycl::event gemm_ev = gemm_detail::_gemm_tree_nm_step<
                lhsTy, rhsTy, resTy, BatchDimsIndexerT, OuterInnerDimsIndexerT,
                OuterInnerDimsIndexerT, TmpIndexerT, wi_delta_n, wi_delta_m>(
                exec_q, lhs_tp, rhs_tp, partially_reduced_tmp, batch_nelems, n,
                k, m, wg_delta_n, wg_delta_m, wi_delta_k, batch_indexer,
                lhs_indexer, rhs_indexer, res_indexer, depends);

            sycl::event red_ev = tree_reduction_for_gemm<resTy, ReductionOpT>(
                exec_q, partially_reduced_tmp, partially_reduced_tmp2, res_tp,
                identity_val, iter_nelems, reduction_nelems, reduction_groups,
                wg, max_wg, preferred_reductions_per_wi, reductions_per_wi,
                batch_nd + res_outer_nd, res_batch_offset, res_shape_strides,
                {gemm_ev});

            sycl::event cleanup_host_task_event =
                dpctl::tensor::alloc_utils::async_smart_free(exec_q, {red_ev},
                                                             tmp_owner);

            return cleanup_host_task_event;
        }
    }
}

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event gemm_batch_nm_impl(sycl::queue &exec_q,
                               const lhsTy *lhs_tp,
                               const rhsTy *rhs_tp,
                               resTy *res_tp,
                               std::size_t batch_nelems,
                               std::size_t n,
                               std::size_t k,
                               std::size_t m,
                               int batch_nd,
                               const ssize_t *batch_shape_strides,
                               ssize_t lhs_batch_offset,
                               ssize_t rhs_batch_offset,
                               ssize_t res_batch_offset,
                               int inner_nd,
                               int lhs_outer_nd,
                               const ssize_t *lhs_outer_inner_shapes_strides,
                               int rhs_outer_nd,
                               const ssize_t *rhs_outer_inner_shapes_strides,
                               int res_outer_nd,
                               const ssize_t *res_outer_shapes_strides,
                               const ssize_t *res_shape_strides,
                               std::vector<sycl::event> const &depends = {})
{

    using OuterInnerDimsIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
    const OuterInnerDimsIndexerT lhs_indexer(inner_nd + lhs_outer_nd, 0,
                                             lhs_outer_inner_shapes_strides);
    const OuterInnerDimsIndexerT rhs_indexer(inner_nd + rhs_outer_nd, 0,
                                             rhs_outer_inner_shapes_strides);
    const OuterInnerDimsIndexerT res_indexer(res_outer_nd, 0,
                                             res_outer_shapes_strides);

    using BatchDimsIndexerT =
        dpctl::tensor::offset_utils::ThreeOffsets_StridedIndexer;
    const BatchDimsIndexerT batch_indexer(batch_nd, lhs_batch_offset,
                                          rhs_batch_offset, res_batch_offset,
                                          batch_shape_strides);

    sycl::event gemm_ev = gemm_detail::_gemm_batch_nm_impl<
        lhsTy, rhsTy, resTy, BatchDimsIndexerT, OuterInnerDimsIndexerT,
        OuterInnerDimsIndexerT, OuterInnerDimsIndexerT>(
        exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m, batch_indexer,
        lhs_indexer, rhs_indexer, res_indexer, depends);

    return gemm_ev;
}

template <typename T1, typename T2, typename T3>
class gemm_batch_tree_empty_krn;

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event gemm_batch_tree_impl(sycl::queue &exec_q,
                                 const char *lhs_cp,
                                 const char *rhs_cp,
                                 char *res_cp,
                                 std::size_t batch_nelems,
                                 std::size_t n,
                                 std::size_t k,
                                 std::size_t m,
                                 int batch_nd,
                                 const ssize_t *batch_shape_strides,
                                 ssize_t lhs_batch_offset,
                                 ssize_t rhs_batch_offset,
                                 ssize_t res_batch_offset,
                                 int inner_nd,
                                 int lhs_outer_nd,
                                 const ssize_t *lhs_outer_inner_shapes_strides,
                                 int rhs_outer_nd,
                                 const ssize_t *rhs_outer_inner_shapes_strides,
                                 int res_outer_nd,
                                 const ssize_t *res_outer_shapes_strides,
                                 const ssize_t *res_shape_strides,
                                 std::vector<sycl::event> const &depends = {})
{
    const lhsTy *lhs_tp = reinterpret_cast<const lhsTy *>(lhs_cp);
    const rhsTy *rhs_tp = reinterpret_cast<const rhsTy *>(rhs_cp);
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp);

    const std::size_t min_nm = std::min(n, m);
    const std::size_t max_nm = std::max(n, m);

    if (min_nm > 0 && (max_nm >= ((64 * 1024) / min_nm))) {
        return gemm_batch_nm_impl<lhsTy, rhsTy, resTy>(
            exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m, batch_nd,
            batch_shape_strides, lhs_batch_offset, rhs_batch_offset,
            res_batch_offset, inner_nd, lhs_outer_nd,
            lhs_outer_inner_shapes_strides, rhs_outer_nd,
            rhs_outer_inner_shapes_strides, res_outer_nd,
            res_outer_shapes_strides, res_shape_strides, depends);
    }

    if (k == 0) {
        sycl::event gemm_batch_no_reduction_ev =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(depends);

                using IndexerT = dpctl::tensor::offset_utils::StridedIndexer;
                const IndexerT res_indexer(batch_nd + res_outer_nd,
                                           res_batch_offset, res_shape_strides);
                using InitKernelName =
                    class gemm_batch_tree_empty_krn<lhsTy, rhsTy, resTy>;
                cgh.parallel_for<InitKernelName>(
                    sycl::range<1>(n * m * batch_nelems), [=](sycl::id<1> id) {
                        auto res_offset = res_indexer(id[0]);
                        res_tp[res_offset] = resTy(0);
                    });
            });
        return gemm_batch_no_reduction_ev;
    }

    if (max_nm < 64) {
        using dpctl::tensor::type_utils::is_complex;
        if constexpr (!is_complex<resTy>::value) {
            if (m < 4) {
                static constexpr std::uint32_t m_groups_one = 1;
                return gemm_batch_tree_k_impl<lhsTy, rhsTy, resTy,
                                              m_groups_one>(
                    exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
                    batch_nd, batch_shape_strides, lhs_batch_offset,
                    rhs_batch_offset, res_batch_offset, inner_nd, lhs_outer_nd,
                    lhs_outer_inner_shapes_strides, rhs_outer_nd,
                    rhs_outer_inner_shapes_strides, res_outer_nd,
                    res_outer_shapes_strides, res_shape_strides, depends);
            }
            else {
                static constexpr std::uint32_t m_groups_four = 4;
                return gemm_batch_tree_k_impl<lhsTy, rhsTy, resTy,
                                              m_groups_four>(
                    exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
                    batch_nd, batch_shape_strides, lhs_batch_offset,
                    rhs_batch_offset, res_batch_offset, inner_nd, lhs_outer_nd,
                    lhs_outer_inner_shapes_strides, rhs_outer_nd,
                    rhs_outer_inner_shapes_strides, res_outer_nd,
                    res_outer_shapes_strides, res_shape_strides, depends);
            }
        }
        else {
            static constexpr std::uint32_t m_groups_one = 1;
            return gemm_batch_tree_k_impl<lhsTy, rhsTy, resTy, m_groups_one>(
                exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m, batch_nd,
                batch_shape_strides, lhs_batch_offset, rhs_batch_offset,
                res_batch_offset, inner_nd, lhs_outer_nd,
                lhs_outer_inner_shapes_strides, rhs_outer_nd,
                rhs_outer_inner_shapes_strides, res_outer_nd,
                res_outer_shapes_strides, res_shape_strides, depends);
        }
    }
    else { // m > 1, n > k or m > k
        using dpctl::tensor::type_utils::is_complex;
        if constexpr (!is_complex<resTy>::value) {
            static constexpr std::uint32_t m_groups_four = 4;
            return gemm_batch_tree_nm_impl<lhsTy, rhsTy, resTy, m_groups_four>(
                exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m, batch_nd,
                batch_shape_strides, lhs_batch_offset, rhs_batch_offset,
                res_batch_offset, inner_nd, lhs_outer_nd,
                lhs_outer_inner_shapes_strides, rhs_outer_nd,
                rhs_outer_inner_shapes_strides, res_outer_nd,
                res_outer_shapes_strides, res_shape_strides, depends);
        }
        else { // m > 1, n > k or m > k, resTy complex
            static constexpr std::uint32_t m_groups_one = 1;
            return gemm_batch_tree_nm_impl<lhsTy, rhsTy, resTy, m_groups_one>(
                exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m, batch_nd,
                batch_shape_strides, lhs_batch_offset, rhs_batch_offset,
                res_batch_offset, inner_nd, lhs_outer_nd,
                lhs_outer_inner_shapes_strides, rhs_outer_nd,
                rhs_outer_inner_shapes_strides, res_outer_nd,
                res_outer_shapes_strides, res_shape_strides, depends);
        }
    }
}

template <typename lhsTy, typename rhsTy, typename resTy, std::size_t m_groups>
sycl::event
    gemm_batch_contig_tree_k_impl(sycl::queue &exec_q,
                                  const lhsTy *lhs_tp,
                                  const rhsTy *rhs_tp,
                                  resTy *res_tp,
                                  std::size_t batch_nelems,
                                  std::size_t n,
                                  std::size_t k,
                                  std::size_t m,
                                  std::vector<sycl::event> const &depends)
{
    std::size_t delta_k(4);
    std::size_t n_wi(64);
    std::size_t delta_n(32);

    const sycl::device &dev = exec_q.get_device();
    const std::size_t local_mem_size =
        dev.get_info<sycl::info::device::local_mem_size>();
    const std::size_t reserved_slm_size = 512;

    gemm_detail::scale_gemm_k_parameters<resTy, m_groups>(
        local_mem_size, reserved_slm_size, delta_k,
        n_wi,   // modified by reference
        delta_n // modified by reference
    );

    if (k <= (delta_k * n_wi)) {
        using OuterInnerDimsIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        static constexpr OuterInnerDimsIndexerT lhs_indexer{};
        static constexpr OuterInnerDimsIndexerT rhs_indexer{};
        static constexpr OuterInnerDimsIndexerT res_indexer{};

        using dpctl::tensor::offset_utils::Strided1DIndexer;
        using dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer;
        using BatchDimsIndexerT =
            ThreeOffsets_CombinedIndexer<Strided1DIndexer, Strided1DIndexer,
                                         Strided1DIndexer>;

        using dpctl::tensor::offset_utils::Strided1DIndexer;
        const BatchDimsIndexerT batch_indexer(
            Strided1DIndexer{/* size   */ batch_nelems,
                             /* step   */ n * k},
            Strided1DIndexer{/* size   */ batch_nelems,
                             /* step   */ k * m},
            Strided1DIndexer{/* size   */ batch_nelems,
                             /* step   */ n * m});

        return gemm_detail::_gemm_tree_k_step<
            lhsTy, rhsTy, resTy, BatchDimsIndexerT, OuterInnerDimsIndexerT,
            OuterInnerDimsIndexerT, OuterInnerDimsIndexerT, m_groups>(
            exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m, delta_n,
            n_wi, delta_k, batch_indexer, lhs_indexer, rhs_indexer, res_indexer,
            depends);
    }
    else {
        using ReductionOpT =
            typename std::conditional<std::is_same_v<resTy, bool>,
                                      sycl::logical_or<resTy>,
                                      sycl::plus<resTy>>::type;
        static constexpr resTy identity_val =
            sycl::known_identity<ReductionOpT, resTy>::value;

        std::size_t iter_nelems = batch_nelems * n * m;
        std::size_t reduction_nelems =
            (k + delta_k * n_wi - 1) / (delta_k * n_wi);

        // more than one work-group is needed, requires a
        // temporary delta_k * n_wi elements processed along k,
        // so if more to process use multiple
        const auto &sg_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();
        std::size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

        static constexpr std::size_t preferred_reductions_per_wi = 4;
        std::size_t reductions_per_wi(preferred_reductions_per_wi);

        std::size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);

        std::size_t max_wg = reduction_detail::get_work_group_size(dev);

        if (reduction_nelems <= preferred_reductions_per_wi * max_wg) {
            auto tmp_owner =
                dpctl::tensor::alloc_utils::smart_malloc_device<resTy>(
                    iter_nelems * reduction_nelems, exec_q);
            resTy *tmp = tmp_owner.get();

            using OuterInnerDimsIndexerT =
                dpctl::tensor::offset_utils::NoOpIndexer;
            static constexpr OuterInnerDimsIndexerT lhs_indexer{};
            static constexpr OuterInnerDimsIndexerT rhs_indexer{};
            static constexpr OuterInnerDimsIndexerT tmp_indexer{};
            using dpctl::tensor::offset_utils::Strided1DIndexer;
            using dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer;
            using BatchDimsIndexerT =
                ThreeOffsets_CombinedIndexer<Strided1DIndexer, Strided1DIndexer,
                                             Strided1DIndexer>;

            const BatchDimsIndexerT batch_indexer(
                Strided1DIndexer{/* size   */ batch_nelems,
                                 /* step   */ n * k},
                Strided1DIndexer{/* size   */ batch_nelems,
                                 /* step   */ k * m},
                Strided1DIndexer{/* size   */ batch_nelems,
                                 /* step   */ n * m});

            sycl::event gemm_ev = gemm_detail::_gemm_tree_k_step<
                lhsTy, rhsTy, resTy, BatchDimsIndexerT, OuterInnerDimsIndexerT,
                OuterInnerDimsIndexerT, OuterInnerDimsIndexerT, m_groups>(
                exec_q, lhs_tp, rhs_tp, tmp, batch_nelems, n, k, m, delta_n,
                n_wi, delta_k, batch_indexer, lhs_indexer, rhs_indexer,
                tmp_indexer, depends);

            sycl::event red_ev =
                single_reduction_for_gemm_contig<resTy, ReductionOpT>(
                    exec_q, tmp, res_tp, identity_val, iter_nelems,
                    reduction_nelems, reduction_groups, wg, max_wg,
                    preferred_reductions_per_wi, reductions_per_wi, {gemm_ev});

            sycl::event cleanup_host_task_event =
                dpctl::tensor::alloc_utils::async_smart_free(exec_q, {red_ev},
                                                             tmp_owner);

            return cleanup_host_task_event;
        }
        else {
            assert(reduction_groups > 1);

            const std::size_t tmp_alloc_size =
                iter_nelems * (/* temp */ reduction_nelems +
                               /* first reduction temp */ reduction_groups);

            auto tmp_owner =
                dpctl::tensor::alloc_utils::smart_malloc_device<resTy>(
                    tmp_alloc_size, exec_q);

            resTy *partially_reduced_tmp = tmp_owner.get();
            resTy *partially_reduced_tmp2 =
                partially_reduced_tmp + reduction_nelems * iter_nelems;

            using OuterInnerDimsIndexerT =
                dpctl::tensor::offset_utils::NoOpIndexer;
            static constexpr OuterInnerDimsIndexerT lhs_indexer{};
            static constexpr OuterInnerDimsIndexerT rhs_indexer{};
            static constexpr OuterInnerDimsIndexerT tmp_indexer{};
            using dpctl::tensor::offset_utils::Strided1DIndexer;
            using dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer;
            using BatchDimsIndexerT =
                ThreeOffsets_CombinedIndexer<Strided1DIndexer, Strided1DIndexer,
                                             Strided1DIndexer>;

            const BatchDimsIndexerT batch_indexer(
                Strided1DIndexer{/* size   */ batch_nelems,
                                 /* step   */ n * k},
                Strided1DIndexer{/* size   */ batch_nelems,
                                 /* step   */ k * m},
                Strided1DIndexer{/* size   */ batch_nelems,
                                 /* step   */ n * m});

            sycl::event gemm_ev = gemm_detail::_gemm_tree_k_step<
                lhsTy, rhsTy, resTy, BatchDimsIndexerT, OuterInnerDimsIndexerT,
                OuterInnerDimsIndexerT, OuterInnerDimsIndexerT, m_groups>(
                exec_q, lhs_tp, rhs_tp, partially_reduced_tmp, batch_nelems, n,
                k, m, delta_n, n_wi, delta_k, batch_indexer, lhs_indexer,
                rhs_indexer, tmp_indexer, depends);

            sycl::event red_ev =
                tree_reduction_for_gemm_contig<resTy, ReductionOpT>(
                    exec_q, partially_reduced_tmp, partially_reduced_tmp2,
                    res_tp, identity_val, iter_nelems, reduction_nelems,
                    reduction_groups, wg, max_wg, preferred_reductions_per_wi,
                    reductions_per_wi, {gemm_ev});

            sycl::event cleanup_host_task_event =
                dpctl::tensor::alloc_utils::async_smart_free(exec_q, {red_ev},
                                                             tmp_owner);

            return cleanup_host_task_event;
        }
    }
}

template <typename lhsTy, typename rhsTy, typename resTy, int wi_delta_m>
sycl::event
    gemm_batch_contig_tree_nm_impl(sycl::queue &exec_q,
                                   const lhsTy *lhs_tp,
                                   const rhsTy *rhs_tp,
                                   resTy *res_tp,
                                   std::size_t batch_nelems,
                                   std::size_t n,
                                   std::size_t k,
                                   std::size_t m,
                                   std::vector<sycl::event> const &depends)
{
    static constexpr int wi_delta_n = 2;
    std::size_t wg_delta_n(16); // rows of A processed in WG
    std::size_t wg_delta_m(16); // rows of B processed in WG
    std::size_t wi_delta_k(64); // Elements in K dimension processed by WI

    const sycl::device &dev = exec_q.get_device();
    const std::size_t local_mem_size =
        dev.get_info<sycl::info::device::local_mem_size>();
    const std::size_t reserved_slm_size = 512;

    gemm_detail::scale_gemm_nm_parameters<resTy, wi_delta_m>(
        local_mem_size, reserved_slm_size, wi_delta_n,
        wi_delta_k, // modified by reference
        wg_delta_n, // modified by reference
        wg_delta_m  // modified by reference
    );

    // each group processes delta_k * n_wi
    // items in a column, so no need for allocating
    // temp memory if only one group is needed
    if (k <= wi_delta_k) {
        using OuterInnerDimsIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        static constexpr OuterInnerDimsIndexerT lhs_indexer{};
        static constexpr OuterInnerDimsIndexerT rhs_indexer{};
        static constexpr OuterInnerDimsIndexerT res_indexer{};

        using dpctl::tensor::offset_utils::Strided1DIndexer;
        using dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer;
        using BatchDimsIndexerT =
            ThreeOffsets_CombinedIndexer<Strided1DIndexer, Strided1DIndexer,
                                         Strided1DIndexer>;

        const BatchDimsIndexerT batch_indexer(
            Strided1DIndexer{/* size   */ batch_nelems,
                             /* step   */ n * k},
            Strided1DIndexer{/* size   */ batch_nelems,
                             /* step   */ k * m},
            Strided1DIndexer{/* size   */ batch_nelems,
                             /* step   */ n * m});

        return gemm_detail::_gemm_tree_nm_step<
            lhsTy, rhsTy, resTy, BatchDimsIndexerT, OuterInnerDimsIndexerT,
            OuterInnerDimsIndexerT, OuterInnerDimsIndexerT, wi_delta_n,
            wi_delta_m>(exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
                        wg_delta_n, wg_delta_m, wi_delta_k, batch_indexer,
                        lhs_indexer, rhs_indexer, res_indexer, depends);
    }
    else {
        using ReductionOpT =
            typename std::conditional<std::is_same_v<resTy, bool>,
                                      sycl::logical_or<resTy>,
                                      sycl::plus<resTy>>::type;
        static constexpr resTy identity_val =
            sycl::known_identity<ReductionOpT, resTy>::value;
        std::size_t iter_nelems = batch_nelems * n * m;
        std::size_t reduction_nelems = (k + wi_delta_k - 1) / wi_delta_k;

        // more than one work-group is needed, requires a temporary
        // delta_k * n_wi elements processed along k, so if more to
        // process use multiple
        const auto &sg_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();
        std::size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

        static constexpr std::size_t preferred_reductions_per_wi = 4;
        std::size_t reductions_per_wi(preferred_reductions_per_wi);

        std::size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);

        std::size_t max_wg = reduction_detail::get_work_group_size(dev);

        if (reduction_nelems <= preferred_reductions_per_wi * max_wg) {
            auto tmp_owner =
                dpctl::tensor::alloc_utils::smart_malloc_device<resTy>(
                    iter_nelems * reduction_nelems, exec_q);

            resTy *tmp = tmp_owner.get();

            using OuterInnerDimsIndexerT =
                dpctl::tensor::offset_utils::NoOpIndexer;
            static constexpr OuterInnerDimsIndexerT lhs_indexer{};
            static constexpr OuterInnerDimsIndexerT rhs_indexer{};
            static constexpr OuterInnerDimsIndexerT tmp_indexer{};

            using dpctl::tensor::offset_utils::Strided1DIndexer;
            using dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer;
            using BatchDimsIndexerT =
                ThreeOffsets_CombinedIndexer<Strided1DIndexer, Strided1DIndexer,
                                             Strided1DIndexer>;

            const BatchDimsIndexerT batch_indexer(
                Strided1DIndexer{/* size */ batch_nelems,
                                 /* step */ n * k},
                Strided1DIndexer{/* size   */ batch_nelems,
                                 /* step   */ k * m},
                Strided1DIndexer{/* size   */ batch_nelems,
                                 /* step   */ n * m});

            sycl::event gemm_ev = gemm_detail::_gemm_tree_nm_step<
                lhsTy, rhsTy, resTy, BatchDimsIndexerT, OuterInnerDimsIndexerT,
                OuterInnerDimsIndexerT, OuterInnerDimsIndexerT, wi_delta_n,
                wi_delta_m>(exec_q, lhs_tp, rhs_tp, tmp, batch_nelems, n, k, m,
                            wg_delta_n, wg_delta_m, wi_delta_k, batch_indexer,
                            lhs_indexer, rhs_indexer, tmp_indexer, depends);

            sycl::event red_ev =
                single_reduction_for_gemm_contig<resTy, ReductionOpT>(
                    exec_q, tmp, res_tp, identity_val, iter_nelems,
                    reduction_nelems, reduction_groups, wg, max_wg,
                    preferred_reductions_per_wi, reductions_per_wi, {gemm_ev});

            sycl::event cleanup_host_task_event =
                dpctl::tensor::alloc_utils::async_smart_free(exec_q, {red_ev},
                                                             tmp_owner);

            return cleanup_host_task_event;
        }
        else {
            assert(reduction_groups > 1);

            const std::size_t tmp_alloc_size =
                iter_nelems * (/* temp */ reduction_nelems +
                               /* first reduction temp */ reduction_groups);

            auto tmp_owner =
                dpctl::tensor::alloc_utils::smart_malloc_device<resTy>(
                    tmp_alloc_size, exec_q);

            resTy *partially_reduced_tmp = tmp_owner.get();
            resTy *partially_reduced_tmp2 =
                partially_reduced_tmp + reduction_nelems * iter_nelems;

            using OuterInnerDimsIndexerT =
                dpctl::tensor::offset_utils::NoOpIndexer;
            static constexpr OuterInnerDimsIndexerT lhs_indexer{};
            static constexpr OuterInnerDimsIndexerT rhs_indexer{};
            static constexpr OuterInnerDimsIndexerT tmp_indexer{};

            using dpctl::tensor::offset_utils::Strided1DIndexer;
            using dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer;
            using BatchDimsIndexerT =
                ThreeOffsets_CombinedIndexer<Strided1DIndexer, Strided1DIndexer,
                                             Strided1DIndexer>;

            const BatchDimsIndexerT batch_indexer(
                Strided1DIndexer{/* size */ batch_nelems,
                                 /* step */ n * k},
                Strided1DIndexer{/* size   */ batch_nelems,
                                 /* step   */ k * m},
                Strided1DIndexer{/* size   */ batch_nelems,
                                 /* step   */ n * m});

            sycl::event gemm_ev = gemm_detail::_gemm_tree_nm_step<
                lhsTy, rhsTy, resTy, BatchDimsIndexerT, OuterInnerDimsIndexerT,
                OuterInnerDimsIndexerT, OuterInnerDimsIndexerT, wi_delta_n,
                wi_delta_m>(exec_q, lhs_tp, rhs_tp, partially_reduced_tmp,
                            batch_nelems, n, k, m, wg_delta_n, wg_delta_m,
                            wi_delta_k, batch_indexer, lhs_indexer, rhs_indexer,
                            tmp_indexer, depends);

            sycl::event red_ev =
                tree_reduction_for_gemm_contig<resTy, ReductionOpT>(
                    exec_q, partially_reduced_tmp, partially_reduced_tmp2,
                    res_tp, identity_val, iter_nelems, reduction_nelems,
                    reduction_groups, wg, max_wg, preferred_reductions_per_wi,
                    reductions_per_wi, {gemm_ev});

            sycl::event cleanup_host_task_event =
                dpctl::tensor::alloc_utils::async_smart_free(exec_q, {red_ev},
                                                             tmp_owner);

            return cleanup_host_task_event;
        }
    }
}

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event gemm_nm_impl(sycl::queue &exec_q,
                         const lhsTy *lhs_tp,
                         const rhsTy *rhs_tp,
                         resTy *res_tp,
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
                         std::vector<sycl::event> const &depends = {})
{
    using OuterInnerDimsIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
    const OuterInnerDimsIndexerT lhs_indexer(inner_nd + lhs_outer_nd, 0,
                                             lhs_shape_strides);
    const OuterInnerDimsIndexerT rhs_indexer(inner_nd + rhs_outer_nd, 0,
                                             rhs_shape_strides);
    const OuterInnerDimsIndexerT res_indexer(res_outer_nd, 0,
                                             res_shape_strides);

    using BatchDimsIndexerT =
        dpctl::tensor::offset_utils::ThreeZeroOffsets_Indexer;
    static constexpr BatchDimsIndexerT batch_indexer{};

    static constexpr std::size_t single_batch_nelems = 1;

    sycl::event gemm_ev = gemm_detail::_gemm_batch_nm_impl<
        lhsTy, rhsTy, resTy, BatchDimsIndexerT, OuterInnerDimsIndexerT,
        OuterInnerDimsIndexerT, OuterInnerDimsIndexerT>(
        exec_q, lhs_tp, rhs_tp, res_tp, single_batch_nelems, n, k, m,
        batch_indexer, lhs_indexer, rhs_indexer, res_indexer, depends);

    return gemm_ev;
}

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event
    gemm_batch_nm_contig_impl(sycl::queue &exec_q,
                              const lhsTy *lhs_tp,
                              const rhsTy *rhs_tp,
                              resTy *res_tp,
                              std::size_t batch_nelems,
                              std::size_t n,
                              std::size_t k,
                              std::size_t m,
                              std::vector<sycl::event> const &depends = {})
{
    using OuterInnerDimsIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
    static constexpr OuterInnerDimsIndexerT lhs_indexer{};
    static constexpr OuterInnerDimsIndexerT rhs_indexer{};
    static constexpr OuterInnerDimsIndexerT res_indexer{};

    static constexpr std::size_t single_batch_nelems = 1;
    if (batch_nelems == single_batch_nelems) {
        using BatchDimsIndexerT =
            dpctl::tensor::offset_utils::ThreeZeroOffsets_Indexer;
        static constexpr BatchDimsIndexerT batch_indexer{};

        sycl::event gemm_ev = gemm_detail::_gemm_batch_nm_impl<
            lhsTy, rhsTy, resTy, BatchDimsIndexerT, OuterInnerDimsIndexerT,
            OuterInnerDimsIndexerT, OuterInnerDimsIndexerT>(
            exec_q, lhs_tp, rhs_tp, res_tp, single_batch_nelems, n, k, m,
            batch_indexer, lhs_indexer, rhs_indexer, res_indexer, depends);

        return gemm_ev;
    }
    else {
        using dpctl::tensor::offset_utils::Strided1DIndexer;
        using dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer;
        using BatchDimsIndexerT =
            ThreeOffsets_CombinedIndexer<Strided1DIndexer, Strided1DIndexer,
                                         Strided1DIndexer>;

        using dpctl::tensor::offset_utils::Strided1DIndexer;

        const BatchDimsIndexerT batch_indexer(
            Strided1DIndexer{/* size   */ batch_nelems,
                             /* step   */ n * k},
            Strided1DIndexer{/* size   */ batch_nelems,
                             /* step   */ k * m},
            Strided1DIndexer{/* size   */ batch_nelems,
                             /* step   */ n * m});

        sycl::event gemm_ev = gemm_detail::_gemm_batch_nm_impl<
            lhsTy, rhsTy, resTy, BatchDimsIndexerT, OuterInnerDimsIndexerT,
            OuterInnerDimsIndexerT, OuterInnerDimsIndexerT>(
            exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
            batch_indexer, lhs_indexer, rhs_indexer, res_indexer, depends);

        return gemm_ev;
    }
}

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event
    gemm_batch_contig_tree_impl(sycl::queue &exec_q,
                                const char *lhs_cp,
                                const char *rhs_cp,
                                char *res_cp,
                                std::size_t batch_nelems,
                                std::size_t n,
                                std::size_t k,
                                std::size_t m,
                                ssize_t lhs_batch_offset,
                                ssize_t rhs_batch_offset,
                                ssize_t res_batch_offset,
                                std::vector<sycl::event> const &depends = {})
{
    const lhsTy *lhs_tp =
        reinterpret_cast<const lhsTy *>(lhs_cp) + lhs_batch_offset;
    const rhsTy *rhs_tp =
        reinterpret_cast<const rhsTy *>(rhs_cp) + rhs_batch_offset;
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp) + res_batch_offset;

    const std::size_t min_nm = std::min(n, m);
    const std::size_t max_nm = std::max(n, m);

    if (min_nm > 0 && (max_nm >= ((64 * 1024) / min_nm))) {
        return gemm_batch_nm_contig_impl<lhsTy, rhsTy, resTy>(
            exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m, depends);
    }

    if (k == 0) {
        sycl::event gemm_batch_no_reduction_ev =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(depends);
                cgh.fill<resTy>(res_tp, resTy(0), n * m * batch_nelems);
            });
        return gemm_batch_no_reduction_ev;
    }

    if (max_nm < 64) {
        using dpctl::tensor::type_utils::is_complex;
        if constexpr (!is_complex<resTy>::value) {
            if (m < 4) {
                return gemm_batch_contig_tree_k_impl<lhsTy, rhsTy, resTy, 1>(
                    exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
                    depends);
            }
            else {
                return gemm_batch_contig_tree_k_impl<lhsTy, rhsTy, resTy, 4>(
                    exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m,
                    depends);
            }
        }
        else {
            return gemm_batch_contig_tree_k_impl<lhsTy, rhsTy, resTy, 1>(
                exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m, depends);
        }
    }
    else { // m > 1, n > k or m > k
        using dpctl::tensor::type_utils::is_complex;
        if constexpr (!is_complex<resTy>::value) {
            return gemm_batch_contig_tree_nm_impl<lhsTy, rhsTy, resTy, 4>(
                exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m, depends);
        }
        else { // m > 1, n > k or m > k, resTy complex
            return gemm_batch_contig_tree_nm_impl<lhsTy, rhsTy, resTy, 1>(
                exec_q, lhs_tp, rhs_tp, res_tp, batch_nelems, n, k, m, depends);
        }
    }
}

// Gemm tree non-batched

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          std::size_t>
class gemm_tree_nm_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          std::size_t>
class gemm_tree_k_krn;

template <typename lhsTy, typename rhsTy, typename resTy, std::size_t m_groups>
sycl::event gemm_tree_k_impl(sycl::queue &exec_q,
                             const lhsTy *lhs_tp,
                             const rhsTy *rhs_tp,
                             resTy *res_tp,
                             std::size_t n,
                             std::size_t k,
                             std::size_t m,
                             int inner_nd,
                             int lhs_outer_nd,
                             const ssize_t *lhs_outer_inner_shapes_strides,
                             int rhs_outer_nd,
                             const ssize_t *rhs_outer_inner_shapes_strides,
                             int res_nd,
                             const ssize_t *res_shapes_strides,
                             const std::vector<sycl::event> &depends)
{
    std::size_t delta_k(4);
    std::size_t n_wi(64);
    std::size_t delta_n(32);

    const sycl::device &dev = exec_q.get_device();
    const std::size_t local_mem_size =
        dev.get_info<sycl::info::device::local_mem_size>();
    const std::size_t reserved_slm_size = 512;

    gemm_detail::scale_gemm_k_parameters<resTy, m_groups>(
        local_mem_size, reserved_slm_size, delta_k,
        n_wi,   // modified by reference
        delta_n // modified by reference
    );

    using BatchIndexerT = dpctl::tensor::offset_utils::ThreeZeroOffsets_Indexer;
    static constexpr BatchIndexerT batch_indexer{};

    static constexpr std::size_t single_batch_nelems = 1;

    using OuterInnerDimsIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
    const OuterInnerDimsIndexerT lhs_indexer(inner_nd + lhs_outer_nd, 0,
                                             lhs_outer_inner_shapes_strides);
    const OuterInnerDimsIndexerT rhs_indexer(inner_nd + rhs_outer_nd, 0,
                                             rhs_outer_inner_shapes_strides);

    sycl::event gemm_ev;
    if (k <= (delta_k * n_wi)) {
        const OuterInnerDimsIndexerT res_indexer(res_nd, 0, res_shapes_strides);

        return gemm_detail::_gemm_tree_k_step<
            lhsTy, rhsTy, resTy, BatchIndexerT, OuterInnerDimsIndexerT,
            OuterInnerDimsIndexerT, OuterInnerDimsIndexerT, m_groups>(
            exec_q, lhs_tp, rhs_tp, res_tp, single_batch_nelems, n, k, m,
            delta_n, n_wi, delta_k, batch_indexer, lhs_indexer, rhs_indexer,
            res_indexer, depends);
    }
    else {
        using ReductionOpT =
            typename std::conditional<std::is_same_v<resTy, bool>,
                                      sycl::logical_or<resTy>,
                                      sycl::plus<resTy>>::type;
        static constexpr resTy identity_val =
            sycl::known_identity<ReductionOpT, resTy>::value;

        std::size_t iter_nelems = n * m;
        std::size_t reduction_nelems =
            (k + delta_k * n_wi - 1) / (delta_k * n_wi);

        // more than one work-groups is needed, requires a temporary
        // delta_k * n_wi elements processed along k, so if more to
        // process use multiple
        const auto &sg_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();
        std::size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

        static constexpr std::size_t preferred_reductions_per_wi = 8;
        std::size_t reductions_per_wi(preferred_reductions_per_wi);

        std::size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);

        std::size_t max_wg = reduction_detail::get_work_group_size(dev);

        if (reduction_nelems <= preferred_reductions_per_wi * max_wg) {

            auto tmp_owner =
                dpctl::tensor::alloc_utils::smart_malloc_device<resTy>(
                    iter_nelems * reduction_nelems, exec_q);
            resTy *tmp = tmp_owner.get();

            using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            static constexpr ResIndexerT res_indexer{};

            sycl::event gemm_ev = gemm_detail::_gemm_tree_k_step<
                lhsTy, rhsTy, resTy, BatchIndexerT, OuterInnerDimsIndexerT,
                OuterInnerDimsIndexerT, ResIndexerT, m_groups>(
                exec_q, lhs_tp, rhs_tp, tmp, single_batch_nelems, n, k, m,
                delta_n, n_wi, delta_k, batch_indexer, lhs_indexer, rhs_indexer,
                res_indexer, depends);

            sycl::event red_ev = single_reduction_for_gemm<resTy, ReductionOpT>(
                exec_q, tmp, res_tp, identity_val, iter_nelems,
                reduction_nelems, reduction_groups, wg, max_wg,
                preferred_reductions_per_wi, reductions_per_wi, res_nd, 0,
                res_shapes_strides, {gemm_ev});

            sycl::event cleanup_host_task_event =
                dpctl::tensor::alloc_utils::async_smart_free(exec_q, {red_ev},
                                                             tmp_owner);
            return cleanup_host_task_event;
        }
        else {
            assert(reduction_groups > 1);

            const std::size_t tmp_alloc_size =
                iter_nelems * (/* temp */ reduction_nelems +
                               /* first reduction temp */ reduction_groups);

            auto tmp_owner =
                dpctl::tensor::alloc_utils::smart_malloc_device<resTy>(
                    tmp_alloc_size, exec_q);

            resTy *partially_reduced_tmp = tmp_owner.get();
            resTy *partially_reduced_tmp2 =
                partially_reduced_tmp + reduction_nelems * iter_nelems;

            using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            static constexpr ResIndexerT res_indexer{};

            sycl::event gemm_ev = gemm_detail::_gemm_tree_k_step<
                lhsTy, rhsTy, resTy, BatchIndexerT, OuterInnerDimsIndexerT,
                OuterInnerDimsIndexerT, ResIndexerT, m_groups>(
                exec_q, lhs_tp, rhs_tp, partially_reduced_tmp,
                single_batch_nelems, n, k, m, delta_n, n_wi, delta_k,
                batch_indexer, lhs_indexer, rhs_indexer, res_indexer, depends);

            // tree_reduction_for_gemm returns sycl::event for reduction
            sycl::event red_ev = tree_reduction_for_gemm<resTy, ReductionOpT>(
                exec_q, partially_reduced_tmp, partially_reduced_tmp2, res_tp,
                identity_val, iter_nelems, reduction_nelems, reduction_groups,
                wg, max_wg, preferred_reductions_per_wi, reductions_per_wi,
                res_nd, 0, res_shapes_strides, {gemm_ev});

            sycl::event cleanup_host_task_event =
                dpctl::tensor::alloc_utils::async_smart_free(exec_q, {red_ev},
                                                             tmp_owner);

            return cleanup_host_task_event;
        }
    }
}

template <typename lhsTy, typename rhsTy, typename resTy, int wi_delta_m>
sycl::event gemm_tree_nm_impl(sycl::queue &exec_q,
                              const lhsTy *lhs_tp,
                              const rhsTy *rhs_tp,
                              resTy *res_tp,
                              std::size_t n,
                              std::size_t k,
                              std::size_t m,
                              int inner_nd,
                              int lhs_outer_nd,
                              const ssize_t *lhs_outer_inner_shapes_strides,
                              int rhs_outer_nd,
                              const ssize_t *rhs_outer_inner_shapes_strides,
                              int res_nd,
                              const ssize_t *res_shapes_strides,
                              const std::vector<sycl::event> &depends)
{
    static constexpr int wi_delta_n = 2;
    std::size_t wg_delta_n(16); // rows of A processed in WG
    std::size_t wg_delta_m(16); // rows of B processed in WG
    std::size_t wi_delta_k(64); // Elements in K dimension processed by WI

    const sycl::device &dev = exec_q.get_device();
    const std::size_t local_mem_size =
        dev.get_info<sycl::info::device::local_mem_size>();
    const std::size_t reserved_slm_size = 512;

    gemm_detail::scale_gemm_nm_parameters<resTy, wi_delta_m>(
        local_mem_size, reserved_slm_size, wi_delta_n,
        wi_delta_k, // modified by reference
        wg_delta_n, // modified by reference
        wg_delta_m  // modified by reference
    );

    using BatchIndexerT = dpctl::tensor::offset_utils::ThreeZeroOffsets_Indexer;
    static constexpr BatchIndexerT batch_indexer{};

    static constexpr std::size_t single_batch_nelems = 1;

    using OuterInnerDimsIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
    const OuterInnerDimsIndexerT lhs_indexer(inner_nd + lhs_outer_nd, 0,
                                             lhs_outer_inner_shapes_strides);
    const OuterInnerDimsIndexerT rhs_indexer(inner_nd + rhs_outer_nd, 0,
                                             rhs_outer_inner_shapes_strides);

    // each group processes delta_k items in a column,
    // so no need to allocate temp memory if one group needed
    if (k <= wi_delta_k) {
        const OuterInnerDimsIndexerT res_indexer(res_nd, 0, res_shapes_strides);

        return gemm_detail::_gemm_tree_nm_step<
            lhsTy, rhsTy, resTy, BatchIndexerT, OuterInnerDimsIndexerT,
            OuterInnerDimsIndexerT, OuterInnerDimsIndexerT, wi_delta_n,
            wi_delta_m>(exec_q, lhs_tp, rhs_tp, res_tp, single_batch_nelems, n,
                        k, m, wg_delta_n, wg_delta_m, wi_delta_k, batch_indexer,
                        lhs_indexer, rhs_indexer, res_indexer, depends);
    }
    else {
        using ReductionOpT =
            typename std::conditional<std::is_same_v<resTy, bool>,
                                      sycl::logical_or<resTy>,
                                      sycl::plus<resTy>>::type;
        static constexpr resTy identity_val =
            sycl::known_identity<ReductionOpT, resTy>::value;

        std::size_t iter_nelems = n * m;
        std::size_t reduction_nelems = (k + wi_delta_k - 1) / wi_delta_k;

        // more than one work-groups is needed, requires a temporary
        // wi_delta_k elements processed along k, so if more to
        // process use multiple
        const auto &sg_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();
        std::size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

        static constexpr std::size_t preferred_reductions_per_wi = 8;
        std::size_t reductions_per_wi(preferred_reductions_per_wi);

        std::size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);

        std::size_t max_wg = reduction_detail::get_work_group_size(dev);

        if (reduction_nelems <= preferred_reductions_per_wi * max_wg) {
            auto tmp_owner =
                dpctl::tensor::alloc_utils::smart_malloc_device<resTy>(
                    iter_nelems * reduction_nelems, exec_q);
            resTy *tmp = tmp_owner.get();

            using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            static constexpr ResIndexerT res_indexer{};

            sycl::event gemm_ev = gemm_detail::_gemm_tree_nm_step<
                lhsTy, rhsTy, resTy, BatchIndexerT, OuterInnerDimsIndexerT,
                OuterInnerDimsIndexerT, ResIndexerT, wi_delta_n, wi_delta_m>(
                exec_q, lhs_tp, rhs_tp, tmp, single_batch_nelems, n, k, m,
                wg_delta_n, wg_delta_m, wi_delta_k, batch_indexer, lhs_indexer,
                rhs_indexer, res_indexer, depends);

            sycl::event red_ev = single_reduction_for_gemm<resTy, ReductionOpT>(
                exec_q, tmp, res_tp, identity_val, iter_nelems,
                reduction_nelems, reduction_groups, wg, max_wg,
                preferred_reductions_per_wi, reductions_per_wi, res_nd, 0,
                res_shapes_strides, {gemm_ev});

            sycl::event cleanup_host_task_event =
                dpctl::tensor::alloc_utils::async_smart_free(exec_q, {red_ev},
                                                             tmp_owner);

            return cleanup_host_task_event;
        }
        else {
            assert(reduction_groups > 1);

            const std::size_t tmp_alloc_size =
                iter_nelems * (/* temp */ reduction_nelems +
                               /* first reduction temp */ reduction_groups);
            auto tmp_owner =
                dpctl::tensor::alloc_utils::smart_malloc_device<resTy>(
                    tmp_alloc_size, exec_q);

            resTy *partially_reduced_tmp = tmp_owner.get();
            resTy *partially_reduced_tmp2 =
                partially_reduced_tmp + reduction_nelems * iter_nelems;

            using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            static constexpr ResIndexerT res_indexer{};

            sycl::event gemm_ev = gemm_detail::_gemm_tree_nm_step<
                lhsTy, rhsTy, resTy, BatchIndexerT, OuterInnerDimsIndexerT,
                OuterInnerDimsIndexerT, ResIndexerT, wi_delta_n, wi_delta_m>(
                exec_q, lhs_tp, rhs_tp, partially_reduced_tmp,
                single_batch_nelems, n, k, m, wg_delta_n, wg_delta_m,
                wi_delta_k, batch_indexer, lhs_indexer, rhs_indexer,
                res_indexer, depends);

            sycl::event red_ev = tree_reduction_for_gemm<resTy, ReductionOpT>(
                exec_q, partially_reduced_tmp, partially_reduced_tmp2, res_tp,
                identity_val, iter_nelems, reduction_nelems, reduction_groups,
                wg, max_wg, preferred_reductions_per_wi, reductions_per_wi,
                res_nd, 0, res_shapes_strides, {gemm_ev});

            sycl::event cleanup_host_task_event =
                dpctl::tensor::alloc_utils::async_smart_free(exec_q, {red_ev},
                                                             tmp_owner);

            return cleanup_host_task_event;
        }
    }
}

template <typename T1, typename T2, typename T3>
class gemm_tree_empty_krn;

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event gemm_tree_impl(sycl::queue &exec_q,
                           const char *lhs_cp,
                           const char *rhs_cp,
                           char *res_cp,
                           std::size_t n,
                           std::size_t k,
                           std::size_t m,
                           int inner_nd,
                           int lhs_outer_nd,
                           const ssize_t *lhs_outer_inner_shapes_strides,
                           int rhs_outer_nd,
                           const ssize_t *rhs_outer_inner_shapes_strides,
                           int res_nd,
                           const ssize_t *res_shapes_strides,
                           std::vector<sycl::event> const &depends = {})
{
    const lhsTy *lhs_tp = reinterpret_cast<const lhsTy *>(lhs_cp);
    const rhsTy *rhs_tp = reinterpret_cast<const rhsTy *>(rhs_cp);
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp);

    const std::size_t min_nm = std::min(n, m);
    const std::size_t max_nm = std::max(n, m);

    if (min_nm > 0 && (max_nm >= ((64 * 1024) / min_nm))) {
        return gemm_nm_impl<lhsTy, rhsTy, resTy>(
            exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, inner_nd, lhs_outer_nd,
            lhs_outer_inner_shapes_strides, rhs_outer_nd,
            rhs_outer_inner_shapes_strides, res_nd, res_shapes_strides,
            depends);
    }

    if (k == 0) {
        sycl::event gemm_no_reduction_ev =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(depends);

                using IndexerT = dpctl::tensor::offset_utils::StridedIndexer;
                const IndexerT res_indexer(res_nd, 0, res_shapes_strides);
                using InitKernelName =
                    class gemm_tree_empty_krn<lhsTy, rhsTy, resTy>;
                cgh.parallel_for<InitKernelName>(
                    sycl::range<1>(n * m), [=](sycl::id<1> id) {
                        auto res_offset = res_indexer(id[0]);
                        res_tp[res_offset] = resTy(0);
                    });
            });
        return gemm_no_reduction_ev;
    }

    if (max_nm < 64) {
        using dpctl::tensor::type_utils::is_complex;
        if constexpr (!is_complex<resTy>::value) {
            if (m < 4) {
                return gemm_tree_k_impl<lhsTy, rhsTy, resTy, 1>(
                    exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, inner_nd,
                    lhs_outer_nd, lhs_outer_inner_shapes_strides, rhs_outer_nd,
                    rhs_outer_inner_shapes_strides, res_nd, res_shapes_strides,
                    depends);
            }
            else {
                return gemm_tree_k_impl<lhsTy, rhsTy, resTy, 4>(
                    exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, inner_nd,
                    lhs_outer_nd, lhs_outer_inner_shapes_strides, rhs_outer_nd,
                    rhs_outer_inner_shapes_strides, res_nd, res_shapes_strides,
                    depends);
            }
        }
        else {
            return gemm_tree_k_impl<lhsTy, rhsTy, resTy, 1>(
                exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, inner_nd, lhs_outer_nd,
                lhs_outer_inner_shapes_strides, rhs_outer_nd,
                rhs_outer_inner_shapes_strides, res_nd, res_shapes_strides,
                depends);
        }
    }
    else { // m > 1, n > k or m > k
        using dpctl::tensor::type_utils::is_complex;
        if constexpr (!is_complex<resTy>::value) {
            return gemm_tree_nm_impl<lhsTy, rhsTy, resTy, 4>(
                exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, inner_nd, lhs_outer_nd,
                lhs_outer_inner_shapes_strides, rhs_outer_nd,
                rhs_outer_inner_shapes_strides, res_nd, res_shapes_strides,
                depends);
        }
        else {
            return gemm_tree_nm_impl<lhsTy, rhsTy, resTy, 1>(
                exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, inner_nd, lhs_outer_nd,
                lhs_outer_inner_shapes_strides, rhs_outer_nd,
                rhs_outer_inner_shapes_strides, res_nd, res_shapes_strides,
                depends);
        }
    }
}

template <typename lhsTy, typename rhsTy, typename resTy, std::size_t m_groups>
sycl::event gemm_contig_tree_k_impl(sycl::queue &exec_q,
                                    const lhsTy *lhs_tp,
                                    const rhsTy *rhs_tp,
                                    resTy *res_tp,
                                    std::size_t n,
                                    std::size_t k,
                                    std::size_t m,
                                    std::vector<sycl::event> const &depends)
{
    std::size_t delta_k(4);
    std::size_t n_wi(64);
    std::size_t delta_n(32);

    const sycl::device &dev = exec_q.get_device();
    const std::size_t local_mem_size =
        dev.get_info<sycl::info::device::local_mem_size>();
    const std::size_t reserved_slm_size = 512;

    gemm_detail::scale_gemm_k_parameters<resTy, m_groups>(
        local_mem_size, reserved_slm_size, delta_k,
        n_wi,   // modified by reference
        delta_n // modified by reference
    );

    using OuterInnerDimsIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
    static constexpr OuterInnerDimsIndexerT lhs_indexer{};
    static constexpr OuterInnerDimsIndexerT rhs_indexer{};
    static constexpr OuterInnerDimsIndexerT res_indexer{};

    using BatchIndexerT = dpctl::tensor::offset_utils::ThreeZeroOffsets_Indexer;
    static constexpr BatchIndexerT batch_indexer{};

    static constexpr std::size_t single_batch_nelems = 1;

    sycl::event gemm_ev;
    if (k <= (delta_k * n_wi)) {
        return gemm_detail::_gemm_tree_k_step<
            lhsTy, rhsTy, resTy, BatchIndexerT, OuterInnerDimsIndexerT,
            OuterInnerDimsIndexerT, OuterInnerDimsIndexerT, m_groups>(
            exec_q, lhs_tp, rhs_tp, res_tp, single_batch_nelems, n, k, m,
            delta_n, n_wi, delta_k, batch_indexer, lhs_indexer, rhs_indexer,
            res_indexer, depends);
    }
    else {
        using ReductionOpT =
            typename std::conditional<std::is_same_v<resTy, bool>,
                                      sycl::logical_or<resTy>,
                                      sycl::plus<resTy>>::type;
        static constexpr resTy identity_val =
            sycl::known_identity<ReductionOpT, resTy>::value;

        std::size_t iter_nelems = n * m;
        std::size_t reduction_nelems =
            (k + delta_k * n_wi - 1) / (delta_k * n_wi);

        // more than one work-groups is needed, requires a
        // temporary delta_k * n_wi elements processed along k,
        // so if more to process use multiple
        const auto &sg_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();
        std::size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

        static constexpr std::size_t preferred_reductions_per_wi = 8;
        std::size_t reductions_per_wi(preferred_reductions_per_wi);

        std::size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);

        std::size_t max_wg = reduction_detail::get_work_group_size(dev);

        if (reduction_nelems <= preferred_reductions_per_wi * max_wg) {
            auto tmp_owner =
                dpctl::tensor::alloc_utils::smart_malloc_device<resTy>(
                    iter_nelems * reduction_nelems, exec_q);
            resTy *tmp = tmp_owner.get();

            sycl::event gemm_ev = gemm_detail::_gemm_tree_k_step<
                lhsTy, rhsTy, resTy, BatchIndexerT, OuterInnerDimsIndexerT,
                OuterInnerDimsIndexerT, OuterInnerDimsIndexerT, m_groups>(
                exec_q, lhs_tp, rhs_tp, tmp, single_batch_nelems, n, k, m,
                delta_n, n_wi, delta_k, batch_indexer, lhs_indexer, rhs_indexer,
                res_indexer, depends);

            sycl::event red_ev =
                single_reduction_for_gemm_contig<resTy, ReductionOpT>(
                    exec_q, tmp, res_tp, identity_val, iter_nelems,
                    reduction_nelems, reduction_groups, wg, max_wg,
                    preferred_reductions_per_wi, reductions_per_wi, {gemm_ev});

            sycl::event cleanup_host_task_event =
                dpctl::tensor::alloc_utils::async_smart_free(exec_q, {red_ev},
                                                             tmp_owner);
            return cleanup_host_task_event;
        }
        else {
            assert(reduction_groups > 1);

            const std::size_t tmp_alloc_size =
                iter_nelems * (/* temp */ reduction_nelems +
                               /* first reduction temp */ reduction_groups);
            auto tmp_owner =
                dpctl::tensor::alloc_utils::smart_malloc_device<resTy>(
                    tmp_alloc_size, exec_q);

            resTy *partially_reduced_tmp = tmp_owner.get();
            resTy *partially_reduced_tmp2 =
                partially_reduced_tmp + reduction_nelems * iter_nelems;

            sycl::event gemm_ev = gemm_detail::_gemm_tree_k_step<
                lhsTy, rhsTy, resTy, BatchIndexerT, OuterInnerDimsIndexerT,
                OuterInnerDimsIndexerT, OuterInnerDimsIndexerT, m_groups>(
                exec_q, lhs_tp, rhs_tp, partially_reduced_tmp,
                single_batch_nelems, n, k, m, delta_n, n_wi, delta_k,
                batch_indexer, lhs_indexer, rhs_indexer, res_indexer, depends);

            // tree_reduction_for_gemm_contig returns sycl::event
            // for reduction
            sycl::event red_ev =
                tree_reduction_for_gemm_contig<resTy, ReductionOpT>(
                    exec_q, partially_reduced_tmp, partially_reduced_tmp2,
                    res_tp, identity_val, iter_nelems, reduction_nelems,
                    reduction_groups, wg, max_wg, preferred_reductions_per_wi,
                    reductions_per_wi, {gemm_ev});

            sycl::event cleanup_host_task_event =
                dpctl::tensor::alloc_utils::async_smart_free(exec_q, {red_ev},
                                                             tmp_owner);

            return cleanup_host_task_event;
        }
    }
}

template <typename lhsTy, typename rhsTy, typename resTy, int wi_delta_m>
sycl::event gemm_contig_tree_nm_impl(sycl::queue &exec_q,
                                     const lhsTy *lhs_tp,
                                     const rhsTy *rhs_tp,
                                     resTy *res_tp,
                                     std::size_t n,
                                     std::size_t k,
                                     std::size_t m,
                                     std::vector<sycl::event> const &depends)
{
    static constexpr int wi_delta_n = 2;
    std::size_t wg_delta_n(16); // rows of A processed in WG
    std::size_t wg_delta_m(16); // rows of B processed in WG
    std::size_t wi_delta_k(64); // Elements in K dimension processed by WI

    const sycl::device &dev = exec_q.get_device();
    const std::size_t local_mem_size =
        dev.get_info<sycl::info::device::local_mem_size>();
    const std::size_t reserved_slm_size = 512;

    gemm_detail::scale_gemm_nm_parameters<resTy, wi_delta_m>(
        local_mem_size, reserved_slm_size, wi_delta_n,
        wi_delta_k, // modified by reference
        wg_delta_n, // modified by reference
        wg_delta_m  // modified by reference
    );

    using OuterInnerDimsIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
    static constexpr OuterInnerDimsIndexerT lhs_indexer{};
    static constexpr OuterInnerDimsIndexerT rhs_indexer{};
    static constexpr OuterInnerDimsIndexerT res_indexer{};

    using BatchIndexerT = dpctl::tensor::offset_utils::ThreeZeroOffsets_Indexer;
    static constexpr BatchIndexerT batch_indexer{};

    static constexpr std::size_t single_batch_nelems = 1;

    // each group processes delta_k items in a column,
    // so no need to allocate temp memory if one group needed
    if (k <= wi_delta_k) {

        return gemm_detail::_gemm_tree_nm_step<
            lhsTy, rhsTy, resTy, BatchIndexerT, OuterInnerDimsIndexerT,
            OuterInnerDimsIndexerT, OuterInnerDimsIndexerT, wi_delta_n,
            wi_delta_m>(exec_q, lhs_tp, rhs_tp, res_tp, single_batch_nelems, n,
                        k, m, wg_delta_n, wg_delta_m, wi_delta_k, batch_indexer,
                        lhs_indexer, rhs_indexer, res_indexer, depends);
    }
    else {
        using ReductionOpT =
            typename std::conditional<std::is_same_v<resTy, bool>,
                                      sycl::logical_or<resTy>,
                                      sycl::plus<resTy>>::type;
        static constexpr resTy identity_val =
            sycl::known_identity<ReductionOpT, resTy>::value;

        std::size_t iter_nelems = n * m;
        std::size_t reduction_nelems = (k + wi_delta_k - 1) / wi_delta_k;

        // more than one work-groups is needed, requires a temporary
        // wi_delta_k elements processed along k, so if more to
        // process use multiple
        const auto &sg_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();
        std::size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

        static constexpr std::size_t preferred_reductions_per_wi = 8;
        std::size_t reductions_per_wi(preferred_reductions_per_wi);

        std::size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);

        std::size_t max_wg = reduction_detail::get_work_group_size(dev);

        if (reduction_nelems <= preferred_reductions_per_wi * max_wg) {
            auto tmp_owner =
                dpctl::tensor::alloc_utils::smart_malloc_device<resTy>(
                    iter_nelems * reduction_nelems, exec_q);
            resTy *tmp = tmp_owner.get();

            sycl::event gemm_ev = gemm_detail::_gemm_tree_nm_step<
                lhsTy, rhsTy, resTy, BatchIndexerT, OuterInnerDimsIndexerT,
                OuterInnerDimsIndexerT, OuterInnerDimsIndexerT, wi_delta_n,
                wi_delta_m>(exec_q, lhs_tp, rhs_tp, tmp, single_batch_nelems, n,
                            k, m, wg_delta_n, wg_delta_m, wi_delta_k,
                            batch_indexer, lhs_indexer, rhs_indexer,
                            res_indexer, depends);

            sycl::event red_ev =
                single_reduction_for_gemm_contig<resTy, ReductionOpT>(
                    exec_q, tmp, res_tp, identity_val, iter_nelems,
                    reduction_nelems, reduction_groups, wg, max_wg,
                    preferred_reductions_per_wi, reductions_per_wi, {gemm_ev});

            sycl::event cleanup_host_task_event =
                dpctl::tensor::alloc_utils::async_smart_free(exec_q, {red_ev},
                                                             tmp_owner);
            return cleanup_host_task_event;
        }
        else {
            assert(reduction_groups > 1);

            const std::size_t tmp_alloc_size =
                iter_nelems * (/* temp */ reduction_nelems +
                               /* first reduction temp */ reduction_groups);

            auto tmp_owner =
                dpctl::tensor::alloc_utils::smart_malloc_device<resTy>(
                    tmp_alloc_size, exec_q);
            resTy *partially_reduced_tmp = tmp_owner.get();
            resTy *partially_reduced_tmp2 =
                partially_reduced_tmp + reduction_nelems * iter_nelems;

            sycl::event gemm_ev = gemm_detail::_gemm_tree_nm_step<
                lhsTy, rhsTy, resTy, BatchIndexerT, OuterInnerDimsIndexerT,
                OuterInnerDimsIndexerT, OuterInnerDimsIndexerT, wi_delta_n,
                wi_delta_m>(exec_q, lhs_tp, rhs_tp, partially_reduced_tmp,
                            single_batch_nelems, n, k, m, wg_delta_n,
                            wg_delta_m, wi_delta_k, batch_indexer, lhs_indexer,
                            rhs_indexer, res_indexer, depends);

            sycl::event red_ev =
                tree_reduction_for_gemm_contig<resTy, ReductionOpT>(
                    exec_q, partially_reduced_tmp, partially_reduced_tmp2,
                    res_tp, identity_val, iter_nelems, reduction_nelems,
                    reduction_groups, wg, max_wg, preferred_reductions_per_wi,
                    reductions_per_wi, {gemm_ev});

            sycl::event cleanup_host_task_event =
                dpctl::tensor::alloc_utils::async_smart_free(exec_q, {red_ev},
                                                             tmp_owner);

            return cleanup_host_task_event;
        }
    }
}

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event gemm_contig_tree_impl(sycl::queue &exec_q,
                                  const char *lhs_cp,
                                  const char *rhs_cp,
                                  char *res_cp,
                                  std::size_t n,
                                  std::size_t k,
                                  std::size_t m,
                                  std::vector<sycl::event> const &depends = {})
{
    const lhsTy *lhs_tp = reinterpret_cast<const lhsTy *>(lhs_cp);
    const rhsTy *rhs_tp = reinterpret_cast<const rhsTy *>(rhs_cp);
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp);

    const std::size_t min_nm = std::min(n, m);
    const std::size_t max_nm = std::max(n, m);

    if (min_nm > 0 && (max_nm >= ((64 * 1024) / min_nm))) {
        static constexpr std::size_t single_batch_nelems = 1;
        return gemm_batch_nm_contig_impl<lhsTy, rhsTy, resTy>(
            exec_q, lhs_tp, rhs_tp, res_tp, single_batch_nelems, n, k, m,
            depends);
    }

    if (k == 0) {
        sycl::event gemm_no_reduction_ev =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(depends);
                cgh.fill<resTy>(res_tp, resTy(0), n * m);
            });
        return gemm_no_reduction_ev;
    }

    if (max_nm < 64) {
        using dpctl::tensor::type_utils::is_complex;
        if constexpr (!is_complex<resTy>::value) {
            if (m < 4) {
                return gemm_contig_tree_k_impl<lhsTy, rhsTy, resTy, 1>(
                    exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, depends);
            }
            else {
                return gemm_contig_tree_k_impl<lhsTy, rhsTy, resTy, 4>(
                    exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, depends);
            }
        }
        else {
            return gemm_contig_tree_k_impl<lhsTy, rhsTy, resTy, 1>(
                exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, depends);
        }
    }
    else { // m > 1, n > k or m > k
        using dpctl::tensor::type_utils::is_complex;
        if constexpr (!is_complex<resTy>::value) {
            return gemm_contig_tree_nm_impl<lhsTy, rhsTy, resTy, 4>(
                exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, depends);
        }
        else {
            return gemm_contig_tree_nm_impl<lhsTy, rhsTy, resTy, 1>(
                exec_q, lhs_tp, rhs_tp, res_tp, n, k, m, depends);
        }
    }
}

} // namespace dpctl::tensor::kernels
