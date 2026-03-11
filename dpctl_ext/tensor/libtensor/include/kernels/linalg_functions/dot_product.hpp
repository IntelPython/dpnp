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
/// This file defines kernels for the vector dot product.
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <sycl/sycl.hpp>
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
namespace su_ns = dpctl::tensor::sycl_utils;

template <typename lhsT,
          typename rhsT,
          typename outT,
          typename BatchIndexerT,
          typename RedIndexerT>
struct SequentialDotProduct
{
private:
    const lhsT *lhs_ = nullptr;
    const rhsT *rhs_ = nullptr;
    outT *out_ = nullptr;
    BatchIndexerT batch_indexer_;
    RedIndexerT reduced_dims_indexer_;
    std::size_t reduction_max_gid_ = 0;

public:
    SequentialDotProduct(const lhsT *lhs,
                         const rhsT *rhs,
                         outT *out,
                         BatchIndexerT batch_indexer,
                         RedIndexerT reduced_dims_indexer,
                         std::size_t reduction_size)
        : lhs_(lhs), rhs_(rhs), out_(out), batch_indexer_(batch_indexer),
          reduced_dims_indexer_(reduced_dims_indexer),
          reduction_max_gid_(reduction_size)
    {
    }

    void operator()(sycl::id<1> id) const
    {

        auto const &batch_offsets = batch_indexer_(id[0]);
        const ssize_t &lhs_batch_offset = batch_offsets.get_first_offset();
        const ssize_t &rhs_batch_offset = batch_offsets.get_second_offset();
        const ssize_t &out_batch_offset = batch_offsets.get_third_offset();

        outT red_val(0);
        for (std::size_t m = 0; m < reduction_max_gid_; ++m) {
            auto reduction_offsets = reduced_dims_indexer_(m);
            auto lhs_reduction_offset = reduction_offsets.get_first_offset();
            auto rhs_reduction_offset = reduction_offsets.get_second_offset();

            using dpctl::tensor::type_utils::convert_impl;
            red_val += convert_impl<outT, lhsT>(
                           lhs_[lhs_batch_offset + lhs_reduction_offset]) *
                       convert_impl<outT, rhsT>(
                           rhs_[rhs_batch_offset + rhs_reduction_offset]);
        }

        out_[out_batch_offset] = red_val;
    }
};

template <typename lhsT,
          typename rhsT,
          typename outT,
          typename ReductionOpT,
          typename BatchIndexerT,
          typename RedIndexerT>
struct DotProductFunctor
{
private:
    const lhsT *lhs_ = nullptr;
    const rhsT *rhs_ = nullptr;
    outT *out_ = nullptr;
    ReductionOpT reduction_op_;
    BatchIndexerT batch_indexer_;
    RedIndexerT reduced_dims_indexer_;
    std::size_t reduction_max_gid_ = 0;
    std::size_t batches_ = 1;
    std::size_t reductions_per_wi = 16;

public:
    DotProductFunctor(const lhsT *lhs,
                      const rhsT *rhs,
                      outT *res,
                      const ReductionOpT &reduction_op,
                      const BatchIndexerT &batch_indexer,
                      const RedIndexerT &arg_reduced_dims_indexer,
                      std::size_t reduction_size,
                      std::size_t iteration_size,
                      std::size_t reduction_size_per_wi)
        : lhs_(lhs), rhs_(rhs), out_(res), reduction_op_(reduction_op),
          batch_indexer_(batch_indexer),
          reduced_dims_indexer_(arg_reduced_dims_indexer),
          reduction_max_gid_(reduction_size), batches_(iteration_size),
          reductions_per_wi(reduction_size_per_wi)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        const std::size_t batch_id = it.get_group(0) % batches_;
        const std::size_t reduction_batch_id = it.get_group(0) / batches_;

        const std::size_t reduction_lid = it.get_local_id(0);
        const std::size_t wg =
            it.get_local_range(0); //   0 <= reduction_lid < wg

        // work-items operate over input with indices
        //   inp_data_id = reduction_batch_id * wg * reductions_per_wi + m * wg
        //   + reduction_lid
        // for 0 <= m < reductions_per_wi
        // for each input

        const auto &batch_offsets_ = batch_indexer_(batch_id);
        const auto &lhs_batch_offset = batch_offsets_.get_first_offset();
        const auto &rhs_batch_offset = batch_offsets_.get_second_offset();
        const auto &out_batch_offset = batch_offsets_.get_third_offset();

        outT local_red_val(0);
        std::size_t arg_reduce_gid0 =
            reduction_lid + reduction_batch_id * wg * reductions_per_wi;
        std::size_t arg_reduce_gid_max = std::min(
            reduction_max_gid_, arg_reduce_gid0 + reductions_per_wi * wg);

        for (std::size_t arg_reduce_gid = arg_reduce_gid0;
             arg_reduce_gid < arg_reduce_gid_max; arg_reduce_gid += wg)
        {
            auto reduction_offsets_ = reduced_dims_indexer_(arg_reduce_gid);
            const auto &lhs_reduction_offset =
                reduction_offsets_.get_first_offset();
            const auto &rhs_reduction_offset =
                reduction_offsets_.get_second_offset();

            using dpctl::tensor::type_utils::convert_impl;
            outT val = convert_impl<outT, lhsT>(
                           lhs_[lhs_batch_offset + lhs_reduction_offset]) *
                       convert_impl<outT, rhsT>(
                           rhs_[rhs_batch_offset + rhs_reduction_offset]);

            local_red_val += val;
        }

        auto work_group = it.get_group();
        outT red_val_over_wg = sycl::reduce_over_group(
            work_group, local_red_val, outT(0), reduction_op_);

        if (work_group.leader()) {
            sycl::atomic_ref<outT, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                res_ref(out_[out_batch_offset]);
            res_ref += red_val_over_wg;
        }
    }
};

template <typename lhsT,
          typename rhsT,
          typename outT,
          typename ReductionOpT,
          typename BatchIndexerT,
          typename RedIndexerT,
          typename SlmT>
struct DotProductCustomFunctor
{
private:
    const lhsT *lhs_ = nullptr;
    const rhsT *rhs_ = nullptr;
    outT *out_ = nullptr;
    ReductionOpT reduction_op_;
    BatchIndexerT batch_indexer_;
    RedIndexerT reduced_dims_indexer_;
    SlmT local_mem_;
    std::size_t reduction_max_gid_ = 0;
    std::size_t batches_ = 1;
    std::size_t reductions_per_wi = 16;

public:
    DotProductCustomFunctor(const lhsT *lhs,
                            const rhsT *rhs,
                            outT *res,
                            const ReductionOpT &reduction_op,
                            const BatchIndexerT &batch_indexer,
                            const RedIndexerT &arg_reduced_dims_indexer,
                            SlmT local_mem,
                            std::size_t reduction_size,
                            std::size_t iteration_size,
                            std::size_t reduction_size_per_wi)
        : lhs_(lhs), rhs_(rhs), out_(res), reduction_op_(reduction_op),
          batch_indexer_(batch_indexer),
          reduced_dims_indexer_(arg_reduced_dims_indexer),
          local_mem_(local_mem), reduction_max_gid_(reduction_size),
          batches_(iteration_size), reductions_per_wi(reduction_size_per_wi)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        const std::size_t batch_id = it.get_group(0) % batches_;
        const std::size_t reduction_batch_id = it.get_group(0) / batches_;

        const std::size_t reduction_lid = it.get_local_id(0);
        const std::size_t wg =
            it.get_local_range(0); //   0 <= reduction_lid < wg

        // work-items operate over input with indices
        //   inp_data_id = reduction_batch_id * wg * reductions_per_wi + m * wg
        //   + reduction_lid
        // for 0 <= m < reductions_per_wi
        // for each input

        const auto &batch_offsets_ = batch_indexer_(batch_id);
        const auto &lhs_batch_offset = batch_offsets_.get_first_offset();
        const auto &rhs_batch_offset = batch_offsets_.get_second_offset();
        const auto &out_batch_offset = batch_offsets_.get_third_offset();

        outT local_red_val(0);
        std::size_t arg_reduce_gid0 =
            reduction_lid + reduction_batch_id * wg * reductions_per_wi;
        std::size_t arg_reduce_gid_max = std::min(
            reduction_max_gid_, arg_reduce_gid0 + reductions_per_wi * wg);

        for (std::size_t arg_reduce_gid = arg_reduce_gid0;
             arg_reduce_gid < arg_reduce_gid_max; arg_reduce_gid += wg)
        {
            auto reduction_offsets_ = reduced_dims_indexer_(arg_reduce_gid);
            const auto &lhs_reduction_offset =
                reduction_offsets_.get_first_offset();
            const auto &rhs_reduction_offset =
                reduction_offsets_.get_second_offset();

            using dpctl::tensor::type_utils::convert_impl;
            outT val = convert_impl<outT, lhsT>(
                           lhs_[lhs_batch_offset + lhs_reduction_offset]) *
                       convert_impl<outT, rhsT>(
                           rhs_[rhs_batch_offset + rhs_reduction_offset]);

            local_red_val += val;
        }

        auto work_group = it.get_group();
        outT red_val_over_wg = su_ns::custom_reduce_over_group(
            work_group, local_mem_, local_red_val, reduction_op_);

        if (work_group.leader()) {
            sycl::atomic_ref<outT, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                res_ref(out_[out_batch_offset]);
            res_ref += red_val_over_wg;
        }
    }
};

template <
    typename lhsTy,
    typename rhsTy,
    typename resTy,
    typename BatchIndexerT,
    typename RedIndexerT,
    template <typename T1, typename T2, typename T3, typename T4, typename T5>
    class kernel_name_token>
sycl::event sequential_dot_product(sycl::queue &exec_q,
                                   const lhsTy *lhs,
                                   const rhsTy *rhs,
                                   resTy *res,
                                   std::size_t batches,
                                   std::size_t reduction_nelems,
                                   const BatchIndexerT &batch_indexer,
                                   const RedIndexerT &reduction_indexer,
                                   const std::vector<sycl::event> &depends)
{
    sycl::event dot_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        cgh.parallel_for<
            kernel_name_token<lhsTy, rhsTy, resTy, BatchIndexerT, RedIndexerT>>(
            sycl::range<1>(batches),
            SequentialDotProduct<lhsTy, rhsTy, resTy, BatchIndexerT,
                                 RedIndexerT>(lhs, rhs, res, batch_indexer,
                                              reduction_indexer,
                                              reduction_nelems));
    });

    return dot_ev;
}

template <typename lhsTy,
          typename rhsTy,
          typename resTy,
          typename ReductionOpT,
          typename BatchIndexerT,
          typename RedIndexerT,
          template <typename T1,
                    typename T2,
                    typename T3,
                    typename T4,
                    typename T5,
                    typename T6>
          class kernel_name_token>
sycl::event submit_atomic_dot_product(sycl::queue &exec_q,
                                      const lhsTy *lhs,
                                      const rhsTy *rhs,
                                      resTy *res,
                                      std::size_t wg,
                                      std::size_t batches,
                                      std::size_t reduction_nelems,
                                      std::size_t reductions_per_wi,
                                      std::size_t reduction_groups,
                                      const BatchIndexerT &batch_indexer,
                                      const RedIndexerT &reduction_indexer,
                                      const std::vector<sycl::event> &depends)
{
    sycl::event dot_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        auto globalRange = sycl::range<1>{batches * reduction_groups * wg};
        auto localRange = sycl::range<1>{wg};
        auto ndRange = sycl::nd_range<1>(globalRange, localRange);

        if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value) {
            using KernelName =
                class kernel_name_token<lhsTy, rhsTy, resTy, ReductionOpT,
                                        BatchIndexerT, RedIndexerT>;

            cgh.parallel_for<KernelName>(
                ndRange, DotProductFunctor<lhsTy, rhsTy, resTy, ReductionOpT,
                                           BatchIndexerT, RedIndexerT>(
                             lhs, rhs, res, ReductionOpT(), batch_indexer,
                             reduction_indexer, reduction_nelems, batches,
                             reductions_per_wi));
        }
        else {
            using SlmT = sycl::local_accessor<resTy, 1>;
            SlmT local_memory = SlmT(localRange, cgh);

            using KernelName = class custom_reduction_wrapper<kernel_name_token<
                lhsTy, rhsTy, resTy, ReductionOpT, BatchIndexerT, RedIndexerT>>;

            cgh.parallel_for<KernelName>(
                ndRange,
                DotProductCustomFunctor<lhsTy, rhsTy, resTy, ReductionOpT,
                                        BatchIndexerT, RedIndexerT, SlmT>(
                    lhs, rhs, res, ReductionOpT(), batch_indexer,
                    reduction_indexer, local_memory, reduction_nelems, batches,
                    reductions_per_wi));
        }
    });
    return dot_ev;
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class dot_product_seq_krn;

template <typename T1, typename T2, typename T3>
class dot_product_init_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6>
class dot_product_krn;

typedef sycl::event (*dot_product_impl_fn_ptr_t)(
    sycl::queue &,
    std::size_t,
    std::size_t,
    const char *,
    const char *,
    char *,
    int,
    const ssize_t *,
    ssize_t,
    ssize_t,
    ssize_t,
    int,
    const ssize_t *,
    ssize_t,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event dot_product_impl(sycl::queue &exec_q,
                             std::size_t batches,
                             std::size_t reduction_nelems,
                             const char *lhs_cp,
                             const char *rhs_cp,
                             char *res_cp,
                             int batch_nd,
                             const ssize_t *batch_shape_and_strides,
                             ssize_t batch_lhs_offset,
                             ssize_t batch_rhs_offset,
                             ssize_t batch_res_offset,
                             int red_nd,
                             const ssize_t *reduction_shape_stride,
                             ssize_t reduction_lhs_offset,
                             ssize_t reduction_rhs_offset,
                             const std::vector<sycl::event> &depends = {})
{
    const lhsTy *lhs_tp = reinterpret_cast<const lhsTy *>(lhs_cp);
    const rhsTy *rhs_tp = reinterpret_cast<const rhsTy *>(rhs_cp);
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp);

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    std::size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    if (reduction_nelems < wg) {
        using InputOutputBatchIndexerT =
            dpctl::tensor::offset_utils::ThreeOffsets_StridedIndexer;
        using ReductionIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;

        const InputOutputBatchIndexerT inp_out_batch_indexer{
            batch_nd, batch_lhs_offset, batch_rhs_offset, batch_res_offset,
            batch_shape_and_strides};
        const ReductionIndexerT reduction_indexer{red_nd, reduction_lhs_offset,
                                                  reduction_rhs_offset,
                                                  reduction_shape_stride};

        sycl::event dot_ev =
            sequential_dot_product<lhsTy, rhsTy, resTy,
                                   InputOutputBatchIndexerT, ReductionIndexerT,
                                   dot_product_seq_krn>(
                exec_q, lhs_tp, rhs_tp, res_tp, batches, reduction_nelems,
                inp_out_batch_indexer, reduction_indexer, depends);

        return dot_ev;
    }
    else {
        sycl::event res_init_ev = exec_q.submit([&](sycl::handler &cgh) {
            using IndexerT =
                dpctl::tensor::offset_utils::UnpackedStridedIndexer;

            const ssize_t *const &res_shape = batch_shape_and_strides;
            const ssize_t *const &res_strides =
                batch_shape_and_strides + 3 * batch_nd;
            const IndexerT res_indexer(batch_nd, batch_res_offset, res_shape,
                                       res_strides);
            using InitKernelName =
                class dot_product_init_krn<lhsTy, rhsTy, resTy>;
            cgh.depends_on(depends);

            cgh.parallel_for<InitKernelName>(
                sycl::range<1>(batches), [=](sycl::id<1> id) {
                    auto res_offset = res_indexer(id[0]);
                    res_tp[res_offset] = 0;
                });
        });

        using ReductionOpT = sycl::plus<resTy>;

        using BatchIndexerT =
            dpctl::tensor::offset_utils::ThreeOffsets_StridedIndexer;
        using ReductionIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;

        const BatchIndexerT batch_indexer{batch_nd, batch_lhs_offset,
                                          batch_rhs_offset, batch_res_offset,
                                          batch_shape_and_strides};
        const ReductionIndexerT reduction_indexer{red_nd, reduction_lhs_offset,
                                                  reduction_rhs_offset,
                                                  reduction_shape_stride};

        static constexpr std::size_t preferred_reductions_per_wi =
            4; // determined experimentally
        std::size_t reductions_per_wi =
            (reduction_nelems < preferred_reductions_per_wi * wg)
                ? std::max<std::size_t>(1, (reduction_nelems + wg - 1) / wg)
                : preferred_reductions_per_wi;

        std::size_t reduction_groups =
            (reduction_nelems + reductions_per_wi * wg - 1) /
            (reductions_per_wi * wg);

        sycl::event dot_ev =
            submit_atomic_dot_product<lhsTy, rhsTy, resTy, ReductionOpT,
                                      BatchIndexerT, ReductionIndexerT,
                                      dot_product_krn>(
                exec_q, lhs_tp, rhs_tp, res_tp, wg, batches, reduction_nelems,
                reductions_per_wi, reduction_groups, batch_indexer,
                reduction_indexer, {res_init_ev});

        return dot_ev;
    }
}

typedef sycl::event (*dot_product_contig_impl_fn_ptr_t)(
    sycl::queue &,
    std::size_t,
    std::size_t,
    const char *,
    const char *,
    char *,
    ssize_t,
    ssize_t,
    ssize_t,
    ssize_t,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event
    dot_product_contig_impl(sycl::queue &exec_q,
                            std::size_t batches,
                            std::size_t reduction_nelems,
                            const char *lhs_cp,
                            const char *rhs_cp,
                            char *res_cp,
                            ssize_t batch_lhs_offset,
                            ssize_t batch_rhs_offset,
                            ssize_t batch_res_offset,
                            ssize_t reduction_lhs_offset,
                            ssize_t reduction_rhs_offset,
                            const std::vector<sycl::event> &depends = {})
{
    const lhsTy *lhs_tp = reinterpret_cast<const lhsTy *>(lhs_cp) +
                          batch_lhs_offset + reduction_lhs_offset;
    const rhsTy *rhs_tp = reinterpret_cast<const rhsTy *>(rhs_cp) +
                          batch_rhs_offset + reduction_rhs_offset;
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp) + batch_res_offset;

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    std::size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    if (reduction_nelems < wg) {
        using InputBatchIndexerT =
            dpctl::tensor::offset_utils::Strided1DIndexer;
        using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using InputOutputBatchIndexerT =
            dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer<
                InputBatchIndexerT, InputBatchIndexerT, NoOpIndexerT>;
        using ReductionIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                NoOpIndexerT, NoOpIndexerT>;

        const InputBatchIndexerT inp_batch_indexer{/* size */ batches,
                                                   /* step */ reduction_nelems};
        const InputOutputBatchIndexerT inp_out_batch_indexer{
            inp_batch_indexer, inp_batch_indexer, NoOpIndexerT{}};
        static constexpr ReductionIndexerT reduction_indexer{NoOpIndexerT{},
                                                             NoOpIndexerT{}};

        sycl::event dot_ev =
            sequential_dot_product<lhsTy, rhsTy, resTy,
                                   InputOutputBatchIndexerT, ReductionIndexerT,
                                   dot_product_seq_krn>(
                exec_q, lhs_tp, rhs_tp, res_tp, batches, reduction_nelems,
                inp_out_batch_indexer, reduction_indexer, depends);

        return dot_ev;
    }
    else {
        sycl::event res_init_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            cgh.fill<resTy>(res_tp, resTy(0), batches);
        });

        using ReductionOpT = sycl::plus<resTy>;

        using InputBatchIndexerT =
            dpctl::tensor::offset_utils::Strided1DIndexer;
        using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using InputOutputBatchIndexerT =
            dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer<
                InputBatchIndexerT, InputBatchIndexerT, NoOpIndexerT>;
        using ReductionIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                NoOpIndexerT, NoOpIndexerT>;

        const InputBatchIndexerT inp_batch_indexer{/* size */ batches,
                                                   /* step */ reduction_nelems};
        const InputOutputBatchIndexerT inp_out_batch_indexer{
            inp_batch_indexer, inp_batch_indexer, NoOpIndexerT{}};
        static constexpr ReductionIndexerT reduction_indexer{NoOpIndexerT{},
                                                             NoOpIndexerT{}};

        static constexpr std::size_t preferred_reductions_per_wi =
            4; // determined experimentally
        std::size_t reductions_per_wi =
            (reduction_nelems < preferred_reductions_per_wi * wg)
                ? std::max<std::size_t>(1, (reduction_nelems + wg - 1) / wg)
                : preferred_reductions_per_wi;

        std::size_t reduction_groups =
            (reduction_nelems + reductions_per_wi * wg - 1) /
            (reductions_per_wi * wg);

        sycl::event dot_ev =
            submit_atomic_dot_product<lhsTy, rhsTy, resTy, ReductionOpT,
                                      InputOutputBatchIndexerT,
                                      ReductionIndexerT, dot_product_krn>(
                exec_q, lhs_tp, rhs_tp, res_tp, wg, batches, reduction_nelems,
                reductions_per_wi, reduction_groups, inp_out_batch_indexer,
                reduction_indexer, {res_init_ev});

        return dot_ev;
    }
}

template <typename lhsT,
          typename rhsT,
          typename outT,
          typename ReductionOpT,
          typename BatchIndexerT,
          typename RedIndexerT>
struct DotProductNoAtomicFunctor
{
private:
    const lhsT *lhs_ = nullptr;
    const rhsT *rhs_ = nullptr;
    outT *out_ = nullptr;
    ReductionOpT reduction_op_;
    BatchIndexerT batch_indexer_;
    RedIndexerT reduced_dims_indexer_;
    std::size_t reduction_max_gid_ = 0;
    std::size_t batches_ = 1;
    std::size_t reductions_per_wi = 16;

public:
    DotProductNoAtomicFunctor(const lhsT *lhs,
                              const rhsT *rhs,
                              outT *res,
                              const ReductionOpT &reduction_op,
                              const BatchIndexerT &batch_indexer,
                              const RedIndexerT &arg_reduced_dims_indexer,
                              std::size_t reduction_size,
                              std::size_t iteration_size,
                              std::size_t reduction_size_per_wi)
        : lhs_(lhs), rhs_(rhs), out_(res), reduction_op_(reduction_op),
          batch_indexer_(batch_indexer),
          reduced_dims_indexer_(arg_reduced_dims_indexer),
          reduction_max_gid_(reduction_size), batches_(iteration_size),
          reductions_per_wi(reduction_size_per_wi)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        const std::size_t reduction_lid = it.get_local_id(0);
        const std::size_t wg =
            it.get_local_range(0); //   0 <= reduction_lid < wg

        const std::size_t batch_id = it.get_group(0) % batches_;
        const std::size_t reduction_batch_id = it.get_group(0) / batches_;
        const std::size_t n_reduction_groups = it.get_group_range(0) / batches_;

        // work-items operate over input with indices
        //   inp_data_id = reduction_batch_id * wg * reductions_per_wi + m * wg
        //   + reduction_lid
        // for 0 <= m < reductions_per_wi
        // for each input

        const auto &batch_offsets_ = batch_indexer_(batch_id);
        const auto &lhs_batch_offset = batch_offsets_.get_first_offset();
        const auto &rhs_batch_offset = batch_offsets_.get_second_offset();
        const auto &out_batch_offset = batch_offsets_.get_third_offset();

        outT local_red_val(0);
        std::size_t arg_reduce_gid0 =
            reduction_lid + reduction_batch_id * wg * reductions_per_wi;
        std::size_t arg_reduce_gid_max = std::min(
            reduction_max_gid_, arg_reduce_gid0 + reductions_per_wi * wg);

        for (std::size_t arg_reduce_gid = arg_reduce_gid0;
             arg_reduce_gid < arg_reduce_gid_max; arg_reduce_gid += wg)
        {
            auto reduction_offsets_ = reduced_dims_indexer_(arg_reduce_gid);
            const auto &lhs_reduction_offset =
                reduction_offsets_.get_first_offset();
            const auto &rhs_reduction_offset =
                reduction_offsets_.get_second_offset();

            using dpctl::tensor::type_utils::convert_impl;
            outT val = convert_impl<outT, lhsT>(
                           lhs_[lhs_batch_offset + lhs_reduction_offset]) *
                       convert_impl<outT, rhsT>(
                           rhs_[rhs_batch_offset + rhs_reduction_offset]);

            local_red_val += val;
        }

        auto work_group = it.get_group();

        using RedOpT = typename std::conditional<std::is_same_v<outT, bool>,
                                                 sycl::logical_or<outT>,
                                                 sycl::plus<outT>>::type;
        outT red_val_over_wg = sycl::reduce_over_group(
            work_group, local_red_val, outT(0), RedOpT());

        if (work_group.leader()) {
            // each group writes to a different memory location
            out_[out_batch_offset * n_reduction_groups + reduction_batch_id] =
                red_val_over_wg;
        }
    }
};

template <typename lhsT,
          typename rhsT,
          typename outT,
          typename ReductionOpT,
          typename BatchIndexerT,
          typename RedIndexerT,
          typename SlmT>
struct DotProductNoAtomicCustomFunctor
{
private:
    const lhsT *lhs_ = nullptr;
    const rhsT *rhs_ = nullptr;
    outT *out_ = nullptr;
    ReductionOpT reduction_op_;
    BatchIndexerT batch_indexer_;
    RedIndexerT reduced_dims_indexer_;
    SlmT local_mem_;
    std::size_t reduction_max_gid_ = 0;
    std::size_t batches_ = 1;
    std::size_t reductions_per_wi = 16;

public:
    DotProductNoAtomicCustomFunctor(const lhsT *lhs,
                                    const rhsT *rhs,
                                    outT *res,
                                    const ReductionOpT &reduction_op,
                                    const BatchIndexerT &batch_indexer,
                                    const RedIndexerT &arg_reduced_dims_indexer,
                                    SlmT local_mem,
                                    std::size_t reduction_size,
                                    std::size_t iteration_size,
                                    std::size_t reduction_size_per_wi)
        : lhs_(lhs), rhs_(rhs), out_(res), reduction_op_(reduction_op),
          batch_indexer_(batch_indexer),
          reduced_dims_indexer_(arg_reduced_dims_indexer),
          local_mem_(local_mem), reduction_max_gid_(reduction_size),
          batches_(iteration_size), reductions_per_wi(reduction_size_per_wi)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        const std::size_t reduction_lid = it.get_local_id(0);
        const std::size_t wg =
            it.get_local_range(0); //   0 <= reduction_lid < wg

        const std::size_t batch_id = it.get_group(0) % batches_;
        const std::size_t reduction_batch_id = it.get_group(0) / batches_;
        const std::size_t n_reduction_groups = it.get_group_range(0) / batches_;

        // work-items operate over input with indices
        //   inp_data_id = reduction_batch_id * wg * reductions_per_wi + m * wg
        //   + reduction_lid
        // for 0 <= m < reductions_per_wi
        // for each input

        const auto &batch_offsets_ = batch_indexer_(batch_id);
        const auto &lhs_batch_offset = batch_offsets_.get_first_offset();
        const auto &rhs_batch_offset = batch_offsets_.get_second_offset();
        const auto &out_batch_offset = batch_offsets_.get_third_offset();

        outT local_red_val(0);
        std::size_t arg_reduce_gid0 =
            reduction_lid + reduction_batch_id * wg * reductions_per_wi;
        std::size_t arg_reduce_gid_max = std::min(
            reduction_max_gid_, arg_reduce_gid0 + reductions_per_wi * wg);

        for (std::size_t arg_reduce_gid = arg_reduce_gid0;
             arg_reduce_gid < arg_reduce_gid_max; arg_reduce_gid += wg)
        {
            auto reduction_offsets_ = reduced_dims_indexer_(arg_reduce_gid);
            const auto &lhs_reduction_offset =
                reduction_offsets_.get_first_offset();
            const auto &rhs_reduction_offset =
                reduction_offsets_.get_second_offset();

            using dpctl::tensor::type_utils::convert_impl;
            outT val = convert_impl<outT, lhsT>(
                           lhs_[lhs_batch_offset + lhs_reduction_offset]) *
                       convert_impl<outT, rhsT>(
                           rhs_[rhs_batch_offset + rhs_reduction_offset]);

            local_red_val += val;
        }

        auto work_group = it.get_group();

        outT red_val_over_wg = su_ns::custom_reduce_over_group(
            work_group, local_mem_, local_red_val, reduction_op_);

        if (work_group.leader()) {
            // each group writes to a different memory location
            out_[out_batch_offset * n_reduction_groups + reduction_batch_id] =
                red_val_over_wg;
        }
    }
};

template <typename lhsTy,
          typename rhsTy,
          typename resTy,
          typename ReductionOpT,
          typename BatchIndexerT,
          typename RedIndexerT,
          template <typename T1,
                    typename T2,
                    typename T3,
                    typename T4,
                    typename T5,
                    typename T6>
          class kernel_name_token>
sycl::event
    submit_no_atomic_dot_product(sycl::queue &exec_q,
                                 const lhsTy *lhs,
                                 const rhsTy *rhs,
                                 resTy *res,
                                 std::size_t wg,
                                 std::size_t batches,
                                 std::size_t reduction_nelems,
                                 std::size_t reductions_per_wi,
                                 std::size_t reduction_groups,
                                 const BatchIndexerT &batch_indexer,
                                 const RedIndexerT &reduction_indexer,
                                 const std::vector<sycl::event> &depends)
{
    sycl::event dot_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        auto globalRange = sycl::range<1>{batches * reduction_groups * wg};
        auto localRange = sycl::range<1>{wg};
        auto ndRange = sycl::nd_range<1>(globalRange, localRange);

        if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value) {
            using KernelName =
                class kernel_name_token<lhsTy, rhsTy, resTy, ReductionOpT,
                                        BatchIndexerT, RedIndexerT>;

            cgh.parallel_for<KernelName>(
                ndRange,
                DotProductNoAtomicFunctor<lhsTy, rhsTy, resTy, ReductionOpT,
                                          BatchIndexerT, RedIndexerT>(
                    lhs, rhs, res, ReductionOpT(), batch_indexer,
                    reduction_indexer, reduction_nelems, batches,
                    reductions_per_wi));
        }
        else {
            using SlmT = sycl::local_accessor<resTy, 1>;
            SlmT local_memory = SlmT(localRange, cgh);

            using KernelName = class custom_reduction_wrapper<kernel_name_token<
                lhsTy, rhsTy, resTy, ReductionOpT, BatchIndexerT, RedIndexerT>>;

            cgh.parallel_for<KernelName>(
                ndRange,
                DotProductNoAtomicCustomFunctor<lhsTy, rhsTy, resTy,
                                                ReductionOpT, BatchIndexerT,
                                                RedIndexerT, SlmT>(
                    lhs, rhs, res, ReductionOpT(), batch_indexer,
                    reduction_indexer, local_memory, reduction_nelems, batches,
                    reductions_per_wi));
        }
    });
    return dot_ev;
}

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6>
class dot_product_tree_krn;

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class dot_product_tree_reduction_krn;

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event dot_product_tree_impl(sycl::queue &exec_q,
                                  std::size_t batches,
                                  std::size_t reduction_nelems,
                                  const char *lhs_cp,
                                  const char *rhs_cp,
                                  char *res_cp,
                                  int batch_nd,
                                  const ssize_t *batch_shape_and_strides,
                                  ssize_t batch_lhs_offset,
                                  ssize_t batch_rhs_offset,
                                  ssize_t batch_res_offset,
                                  int red_nd,
                                  const ssize_t *reduction_shape_stride,
                                  ssize_t reduction_lhs_offset,
                                  ssize_t reduction_rhs_offset,
                                  const std::vector<sycl::event> &depends = {})
{
    const lhsTy *lhs_tp = reinterpret_cast<const lhsTy *>(lhs_cp);
    const rhsTy *rhs_tp = reinterpret_cast<const rhsTy *>(rhs_cp);
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp);

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    std::size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    if (reduction_nelems < wg) {
        using InputOutputBatchIndexerT =
            dpctl::tensor::offset_utils::ThreeOffsets_StridedIndexer;
        using ReductionIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;

        const InputOutputBatchIndexerT inp_out_batch_indexer{
            batch_nd, batch_lhs_offset, batch_rhs_offset, batch_res_offset,
            batch_shape_and_strides};
        const ReductionIndexerT reduction_indexer{red_nd, reduction_lhs_offset,
                                                  reduction_rhs_offset,
                                                  reduction_shape_stride};

        sycl::event dot_ev =
            sequential_dot_product<lhsTy, rhsTy, resTy,
                                   InputOutputBatchIndexerT, ReductionIndexerT,
                                   dot_product_seq_krn>(
                exec_q, lhs_tp, rhs_tp, res_tp, batches, reduction_nelems,
                inp_out_batch_indexer, reduction_indexer, depends);

        return dot_ev;
    }

    static constexpr std::size_t preferred_reductions_per_wi = 8;
    // prevents running out of resources on CPU
    std::size_t max_wg = reduction_detail::get_work_group_size(d);

    using ReductionOpT = typename std::conditional<std::is_same_v<resTy, bool>,
                                                   sycl::logical_or<resTy>,
                                                   sycl::plus<resTy>>::type;

    std::size_t reductions_per_wi(preferred_reductions_per_wi);
    if (reduction_nelems <= preferred_reductions_per_wi * max_wg) {
        using BatchIndexerT =
            dpctl::tensor::offset_utils::ThreeOffsets_StridedIndexer;
        using ReductionIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;

        const BatchIndexerT batch_indexer{batch_nd, batch_lhs_offset,
                                          batch_rhs_offset, batch_res_offset,
                                          batch_shape_and_strides};
        const ReductionIndexerT reduction_indexer{red_nd, reduction_lhs_offset,
                                                  reduction_rhs_offset,
                                                  reduction_shape_stride};

        if (batches == 1) {
            // increase GPU occupancy
            wg = max_wg;
        }
        reductions_per_wi =
            std::max<std::size_t>(1, (reduction_nelems + wg - 1) / wg);

        std::size_t reduction_groups =
            (reduction_nelems + reductions_per_wi * wg - 1) /
            (reductions_per_wi * wg);
        assert(reduction_groups == 1);

        sycl::event dot_ev =
            submit_no_atomic_dot_product<lhsTy, rhsTy, resTy, ReductionOpT,
                                         BatchIndexerT, ReductionIndexerT,
                                         dot_product_tree_krn>(
                exec_q, lhs_tp, rhs_tp, res_tp, wg, batches, reduction_nelems,
                reductions_per_wi, reduction_groups, batch_indexer,
                reduction_indexer, depends);

        return dot_ev;
    }
    else {
        static constexpr resTy identity_val =
            sycl::known_identity<ReductionOpT, resTy>::value;

        // more than one work-groups is needed, requires a temporary
        std::size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);
        assert(reduction_groups > 1);

        std::size_t second_iter_reduction_groups_ =
            (reduction_groups + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);

        // returns unique_ptr
        auto partially_reduced_tmp_owner =
            dpctl::tensor::alloc_utils::smart_malloc_device<resTy>(
                batches * (reduction_groups + second_iter_reduction_groups_),
                exec_q);

        resTy *partially_reduced_tmp = partially_reduced_tmp_owner.get();
        resTy *partially_reduced_tmp2 =
            partially_reduced_tmp + reduction_groups * batches;

        sycl::event first_reduction_ev;
        {
            using LhsIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
            using RhsIndexerT =
                dpctl::tensor::offset_utils::UnpackedStridedIndexer;
            using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using InputOutputBatchIndexerT =
                dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer<
                    LhsIndexerT, RhsIndexerT, ResIndexerT>;
            using ReductionIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;

            const LhsIndexerT lhs_indexer(batch_nd, batch_lhs_offset,
                                          batch_shape_and_strides);
            const RhsIndexerT rhs_indexer(
                batch_nd, batch_rhs_offset, batch_shape_and_strides,
                batch_shape_and_strides + 2 * batch_nd);
            static constexpr ResIndexerT noop_tmp_indexer{};

            const InputOutputBatchIndexerT in_out_iter_indexer{
                lhs_indexer, rhs_indexer, noop_tmp_indexer};
            const ReductionIndexerT reduction_indexer{
                red_nd, reduction_lhs_offset, reduction_rhs_offset,
                reduction_shape_stride};

            first_reduction_ev = submit_no_atomic_dot_product<
                lhsTy, rhsTy, resTy, ReductionOpT, InputOutputBatchIndexerT,
                ReductionIndexerT, dot_product_tree_krn>(
                exec_q, lhs_tp, rhs_tp, partially_reduced_tmp, wg, batches,
                reduction_nelems, preferred_reductions_per_wi, reduction_groups,
                in_out_iter_indexer, reduction_indexer, depends);
        }

        std::size_t remaining_reduction_nelems = reduction_groups;

        resTy *temp_arg = partially_reduced_tmp;
        resTy *temp2_arg = partially_reduced_tmp2;
        sycl::event dependent_ev = first_reduction_ev;

        while (remaining_reduction_nelems >
               preferred_reductions_per_wi * max_wg) {
            std::size_t reduction_groups_ =
                (remaining_reduction_nelems + preferred_reductions_per_wi * wg -
                 1) /
                (preferred_reductions_per_wi * wg);
            assert(reduction_groups_ > 1);

            using InputIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
            using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    InputIndexerT, ResIndexerT>;
            using ReductionIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;

            const InputIndexerT inp_indexer{/* size */ batches,
                                            /* step */ reduction_groups_};
            static constexpr ResIndexerT res_iter_indexer{};

            const InputOutputIterIndexerT in_out_iter_indexer{inp_indexer,
                                                              res_iter_indexer};
            static constexpr ReductionIndexerT reduction_indexer{};

            sycl::event partial_reduction_ev =
                dpctl::tensor::kernels::submit_no_atomic_reduction<
                    resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                    ReductionIndexerT, dot_product_tree_reduction_krn>(
                    exec_q, temp_arg, temp2_arg, identity_val, wg, batches,
                    remaining_reduction_nelems, preferred_reductions_per_wi,
                    reduction_groups_, in_out_iter_indexer, reduction_indexer,
                    {dependent_ev});

            remaining_reduction_nelems = reduction_groups_;
            std::swap(temp_arg, temp2_arg);
            dependent_ev = std::move(partial_reduction_ev);
        }

        // final reduction to res
        using InputIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
        using ResIndexerT = dpctl::tensor::offset_utils::UnpackedStridedIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                InputIndexerT, ResIndexerT>;
        using ReductionIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;

        const InputIndexerT inp_indexer{/* size */ batches,
                                        /* step */ remaining_reduction_nelems};
        const ResIndexerT res_iter_indexer{
            batch_nd, batch_res_offset,
            /* shape */ batch_shape_and_strides,
            /* strides */ batch_shape_and_strides + 2 * batch_nd};

        const InputOutputIterIndexerT in_out_iter_indexer{inp_indexer,
                                                          res_iter_indexer};
        static constexpr ReductionIndexerT reduction_indexer{};

        wg = max_wg;
        reductions_per_wi = std::max<std::size_t>(
            1, (remaining_reduction_nelems + wg - 1) / wg);

        reduction_groups =
            (remaining_reduction_nelems + reductions_per_wi * wg - 1) /
            (reductions_per_wi * wg);
        assert(reduction_groups == 1);

        sycl::event final_reduction_ev =
            dpctl::tensor::kernels::submit_no_atomic_reduction<
                resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                ReductionIndexerT, dot_product_tree_reduction_krn>(
                exec_q, temp_arg, res_tp, identity_val, wg, batches,
                remaining_reduction_nelems, reductions_per_wi, reduction_groups,
                in_out_iter_indexer, reduction_indexer, {dependent_ev});

        // transfer ownership of USM allocation to host_task
        sycl::event cleanup_host_task_event =
            dpctl::tensor::alloc_utils::async_smart_free(
                exec_q, {final_reduction_ev}, partially_reduced_tmp_owner);

        return cleanup_host_task_event;
    }
}

template <typename lhsTy, typename rhsTy, typename resTy>
sycl::event
    dot_product_contig_tree_impl(sycl::queue &exec_q,
                                 std::size_t batches,
                                 std::size_t reduction_nelems,
                                 const char *lhs_cp,
                                 const char *rhs_cp,
                                 char *res_cp,
                                 ssize_t batch_lhs_offset,
                                 ssize_t batch_rhs_offset,
                                 ssize_t batch_res_offset,
                                 ssize_t reduction_lhs_offset,
                                 ssize_t reduction_rhs_offset,
                                 const std::vector<sycl::event> &depends = {})
{
    const lhsTy *lhs_tp = reinterpret_cast<const lhsTy *>(lhs_cp) +
                          batch_lhs_offset + reduction_lhs_offset;
    const rhsTy *rhs_tp = reinterpret_cast<const rhsTy *>(rhs_cp) +
                          batch_rhs_offset + reduction_rhs_offset;
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp) + batch_res_offset;

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    std::size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    if (reduction_nelems < wg) {
        using InputBatchIndexerT =
            dpctl::tensor::offset_utils::Strided1DIndexer;
        using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using InputOutputBatchIndexerT =
            dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer<
                InputBatchIndexerT, InputBatchIndexerT, NoOpIndexerT>;
        using ReductionIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                NoOpIndexerT, NoOpIndexerT>;

        const InputBatchIndexerT inp_batch_indexer{/* size */ batches,
                                                   /* step */ reduction_nelems};
        const InputOutputBatchIndexerT inp_out_batch_indexer{
            inp_batch_indexer, inp_batch_indexer, NoOpIndexerT{}};
        static constexpr ReductionIndexerT reduction_indexer{NoOpIndexerT{},
                                                             NoOpIndexerT{}};

        sycl::event dot_ev =
            sequential_dot_product<lhsTy, rhsTy, resTy,
                                   InputOutputBatchIndexerT, ReductionIndexerT,
                                   dot_product_seq_krn>(
                exec_q, lhs_tp, rhs_tp, res_tp, batches, reduction_nelems,
                inp_out_batch_indexer, reduction_indexer, depends);

        return dot_ev;
    }

    static constexpr std::size_t preferred_reductions_per_wi = 8;
    // prevents running out of resources on CPU
    std::size_t max_wg = reduction_detail::get_work_group_size(d);

    using ReductionOpT = typename std::conditional<std::is_same_v<resTy, bool>,
                                                   sycl::logical_or<resTy>,
                                                   sycl::plus<resTy>>::type;

    std::size_t reductions_per_wi(preferred_reductions_per_wi);
    if (reduction_nelems <= preferred_reductions_per_wi * max_wg) {
        using InputBatchIndexerT =
            dpctl::tensor::offset_utils::Strided1DIndexer;
        using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using InputOutputBatchIndexerT =
            dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer<
                InputBatchIndexerT, InputBatchIndexerT, NoOpIndexerT>;
        using ReductionIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                NoOpIndexerT, NoOpIndexerT>;

        const InputBatchIndexerT inp_batch_indexer{/* size */ batches,
                                                   /* step */ reduction_nelems};
        const InputOutputBatchIndexerT inp_out_batch_indexer{
            inp_batch_indexer, inp_batch_indexer, NoOpIndexerT{}};
        static constexpr ReductionIndexerT reduction_indexer{NoOpIndexerT{},
                                                             NoOpIndexerT{}};

        if (batches == 1) {
            // increase GPU occupancy
            wg = max_wg;
        }
        reductions_per_wi =
            std::max<std::size_t>(1, (reduction_nelems + wg - 1) / wg);

        std::size_t reduction_groups =
            (reduction_nelems + reductions_per_wi * wg - 1) /
            (reductions_per_wi * wg);
        assert(reduction_groups == 1);

        sycl::event dot_ev = submit_no_atomic_dot_product<
            lhsTy, rhsTy, resTy, ReductionOpT, InputOutputBatchIndexerT,
            ReductionIndexerT, dot_product_tree_krn>(
            exec_q, lhs_tp, rhs_tp, res_tp, wg, batches, reduction_nelems,
            reductions_per_wi, reduction_groups, inp_out_batch_indexer,
            reduction_indexer, depends);

        return dot_ev;
    }
    else {
        static constexpr resTy identity_val =
            sycl::known_identity<ReductionOpT, resTy>::value;

        // more than one work-groups is needed, requires a temporary
        std::size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);
        assert(reduction_groups > 1);

        std::size_t second_iter_reduction_groups_ =
            (reduction_groups + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);

        // unique_ptr that owns temporary allocation for partial reductions
        auto partially_reduced_tmp_owner =
            dpctl::tensor::alloc_utils::smart_malloc_device<resTy>(
                batches * (reduction_groups + second_iter_reduction_groups_),
                exec_q);
        // get raw pointers
        resTy *partially_reduced_tmp = partially_reduced_tmp_owner.get();
        resTy *partially_reduced_tmp2 =
            partially_reduced_tmp + reduction_groups * batches;

        sycl::event first_reduction_ev;
        {
            using InputBatchIndexerT =
                dpctl::tensor::offset_utils::Strided1DIndexer;
            using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using InputOutputBatchIndexerT =
                dpctl::tensor::offset_utils::ThreeOffsets_CombinedIndexer<
                    InputBatchIndexerT, InputBatchIndexerT, NoOpIndexerT>;
            using ReductionIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    NoOpIndexerT, NoOpIndexerT>;

            const InputBatchIndexerT inp_batch_indexer{
                /* size */ batches,
                /* step */ reduction_nelems};
            const InputOutputBatchIndexerT inp_out_batch_indexer{
                inp_batch_indexer, inp_batch_indexer, NoOpIndexerT{}};
            static constexpr ReductionIndexerT reduction_indexer{
                NoOpIndexerT{}, NoOpIndexerT{}};

            first_reduction_ev = submit_no_atomic_dot_product<
                lhsTy, rhsTy, resTy, ReductionOpT, InputOutputBatchIndexerT,
                ReductionIndexerT, dot_product_tree_krn>(
                exec_q, lhs_tp, rhs_tp, partially_reduced_tmp, wg, batches,
                reduction_nelems, preferred_reductions_per_wi, reduction_groups,
                inp_out_batch_indexer, reduction_indexer, depends);
        }

        std::size_t remaining_reduction_nelems = reduction_groups;

        resTy *temp_arg = partially_reduced_tmp;
        resTy *temp2_arg = partially_reduced_tmp2;
        sycl::event dependent_ev = first_reduction_ev;

        while (remaining_reduction_nelems >
               preferred_reductions_per_wi * max_wg) {
            std::size_t reduction_groups_ =
                (remaining_reduction_nelems + preferred_reductions_per_wi * wg -
                 1) /
                (preferred_reductions_per_wi * wg);
            assert(reduction_groups_ > 1);

            using InputIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
            using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    InputIndexerT, ResIndexerT>;
            using ReductionIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;

            const InputIndexerT inp_indexer{/* size */ batches,
                                            /* step */ reduction_groups_};
            static constexpr ResIndexerT res_iter_indexer{};

            const InputOutputIterIndexerT in_out_iter_indexer{inp_indexer,
                                                              res_iter_indexer};
            static constexpr ReductionIndexerT reduction_indexer{};

            sycl::event partial_reduction_ev =
                dpctl::tensor::kernels::submit_no_atomic_reduction<
                    resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                    ReductionIndexerT, dot_product_tree_reduction_krn>(
                    exec_q, temp_arg, temp2_arg, identity_val, wg, batches,
                    remaining_reduction_nelems, preferred_reductions_per_wi,
                    reduction_groups_, in_out_iter_indexer, reduction_indexer,
                    {dependent_ev});

            remaining_reduction_nelems = reduction_groups_;
            std::swap(temp_arg, temp2_arg);
            dependent_ev = std::move(partial_reduction_ev);
        }

        // final reduction to res
        using InputIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
        using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                InputIndexerT, ResIndexerT>;
        using ReductionIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;

        const InputIndexerT inp_indexer{/* size */ batches,
                                        /* step */ remaining_reduction_nelems};
        static constexpr ResIndexerT res_iter_indexer{};

        const InputOutputIterIndexerT in_out_iter_indexer{inp_indexer,
                                                          res_iter_indexer};
        static constexpr ReductionIndexerT reduction_indexer{};

        wg = max_wg;
        reductions_per_wi = std::max<std::size_t>(
            1, (remaining_reduction_nelems + wg - 1) / wg);

        reduction_groups =
            (remaining_reduction_nelems + reductions_per_wi * wg - 1) /
            (reductions_per_wi * wg);
        assert(reduction_groups == 1);

        sycl::event final_reduction_ev =
            dpctl::tensor::kernels::submit_no_atomic_reduction<
                resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                ReductionIndexerT, dot_product_tree_reduction_krn>(
                exec_q, temp_arg, res_tp, identity_val, wg, batches,
                remaining_reduction_nelems, reductions_per_wi, reduction_groups,
                in_out_iter_indexer, reduction_indexer, {dependent_ev});

        sycl::event cleanup_host_task_event =
            dpctl::tensor::alloc_utils::async_smart_free(
                exec_q, {final_reduction_ev}, partially_reduced_tmp_owner);

        return cleanup_host_task_event;
    }
}

} // namespace dpctl::tensor::kernels
