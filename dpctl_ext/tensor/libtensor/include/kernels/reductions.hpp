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
/// This file defines kernels for tensor reduction along axis.
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpctl_tensor_types.hpp"
#include "utils/math_utils.hpp"
#include "utils/offset_utils.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/sycl_utils.hpp"
#include "utils/type_utils.hpp"

namespace dpctl::tensor::kernels
{

using dpctl::tensor::ssize_t;
namespace su_ns = dpctl::tensor::sycl_utils;

namespace reduction_detail
{

inline std::size_t get_work_group_size(const sycl::device &d)
{
    // prevents running out of resources on CPU
    return std::min<std::size_t>(
        2048, d.get_info<sycl::info::device::max_work_group_size>() / 2);
}

} // namespace reduction_detail

template <typename ReductionOpT, typename T>
struct needs_workaround
{
    static constexpr bool value =
        (std::is_same_v<ReductionOpT, sycl::multiplies<T>> &&
         (std::is_same_v<T, std::int64_t> ||
          std::is_same_v<T, std::uint64_t>)) ||
        (__LIBSYCL_MAJOR_VERSION < 7 && std::is_same_v<T, bool> &&
         std::is_same_v<ReductionOpT, sycl::logical_or<T>>);
};

template <typename ReductionOpT, typename T>
struct can_use_reduce_over_group
{
    static constexpr bool value =
        sycl::has_known_identity<ReductionOpT, T>::value &&
        !needs_workaround<ReductionOpT, T>::value;
};

template <typename argT,
          typename outT,
          typename ReductionOp,
          typename InputOutputIterIndexerT,
          typename InputRedIndexerT>
struct SequentialReduction
{
private:
    const argT *inp_ = nullptr;
    outT *out_ = nullptr;
    ReductionOp reduction_op_;
    outT identity_;
    InputOutputIterIndexerT inp_out_iter_indexer_;
    InputRedIndexerT inp_reduced_dims_indexer_;
    std::size_t reduction_max_gid_ = 0;

public:
    SequentialReduction(const argT *inp,
                        outT *res,
                        const ReductionOp &reduction_op,
                        const outT &identity_val,
                        const InputOutputIterIndexerT &arg_res_iter_indexer,
                        const InputRedIndexerT &arg_reduced_dims_indexer,
                        std::size_t reduction_size)
        : inp_(inp), out_(res), reduction_op_(reduction_op),
          identity_(identity_val), inp_out_iter_indexer_(arg_res_iter_indexer),
          inp_reduced_dims_indexer_(arg_reduced_dims_indexer),
          reduction_max_gid_(reduction_size)
    {
    }

    void operator()(sycl::id<1> id) const
    {

        auto const &inp_out_iter_offsets_ = inp_out_iter_indexer_(id[0]);
        const ssize_t &inp_iter_offset =
            inp_out_iter_offsets_.get_first_offset();
        const ssize_t &out_iter_offset =
            inp_out_iter_offsets_.get_second_offset();

        outT red_val(identity_);
        for (std::size_t m = 0; m < reduction_max_gid_; ++m) {
            const ssize_t inp_reduction_offset = inp_reduced_dims_indexer_(m);
            const ssize_t inp_offset = inp_iter_offset + inp_reduction_offset;

            using dpctl::tensor::type_utils::convert_impl;
            outT val;
            if constexpr (su_ns::IsLogicalAnd<outT, ReductionOp>::value ||
                          su_ns::IsLogicalOr<outT, ReductionOp>::value)
            {
                val = convert_impl<bool, argT>(inp_[inp_offset]);
            }
            else {
                val = convert_impl<outT, argT>(inp_[inp_offset]);
            }
            red_val = reduction_op_(red_val, val);
        }

        out_[out_iter_offset] = red_val;
    }
};

/* === Reduction, using sycl::reduce_over_group, and sycl::atomic_ref === */

/*
  This kernel only works for outT with sizeof(outT) == 4, or sizeof(outT) == 8
  if the device has aspect atomic64 and only with those supported by
  sycl::atomic_ref
*/
template <typename argT,
          typename outT,
          typename ReductionOp,
          typename InputOutputIterIndexerT,
          typename InputRedIndexerT>
struct ReductionOverGroupWithAtomicFunctor
{
private:
    const argT *inp_ = nullptr;
    outT *out_ = nullptr;
    ReductionOp reduction_op_;
    outT identity_;
    InputOutputIterIndexerT inp_out_iter_indexer_;
    InputRedIndexerT inp_reduced_dims_indexer_;
    std::size_t reduction_max_gid_ = 0;
    std::size_t iter_gws_ = 1;
    std::size_t reductions_per_wi = 16;

public:
    ReductionOverGroupWithAtomicFunctor(
        const argT *data,
        outT *res,
        const ReductionOp &reduction_op,
        const outT &identity_val,
        const InputOutputIterIndexerT &arg_res_iter_indexer,
        const InputRedIndexerT &arg_reduced_dims_indexer,
        std::size_t reduction_size,
        std::size_t iteration_size,
        std::size_t reduction_size_per_wi)
        : inp_(data), out_(res), reduction_op_(reduction_op),
          identity_(identity_val), inp_out_iter_indexer_(arg_res_iter_indexer),
          inp_reduced_dims_indexer_(arg_reduced_dims_indexer),
          reduction_max_gid_(reduction_size), iter_gws_(iteration_size),
          reductions_per_wi(reduction_size_per_wi)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        const std::size_t iter_gid = it.get_group(0) % iter_gws_;
        const std::size_t reduction_batch_id = it.get_group(0) / iter_gws_;

        const std::size_t reduction_lid = it.get_local_id(0);
        const std::size_t wg =
            it.get_local_range(0); //   0 <= reduction_lid < wg

        // work-items operate over input with indices
        //   inp_data_id = reduction_batch_id * wg * reductions_per_wi + m * wg
        //   + reduction_lid
        // for 0 <= m < reductions_per_wi

        const auto &inp_out_iter_offsets_ = inp_out_iter_indexer_(iter_gid);
        const auto &inp_iter_offset = inp_out_iter_offsets_.get_first_offset();
        const auto &out_iter_offset = inp_out_iter_offsets_.get_second_offset();

        outT local_red_val(identity_);
        std::size_t arg_reduce_gid0 =
            reduction_lid + reduction_batch_id * wg * reductions_per_wi;
        std::size_t arg_reduce_gid_max = std::min(
            reduction_max_gid_, arg_reduce_gid0 + reductions_per_wi * wg);

        for (std::size_t arg_reduce_gid = arg_reduce_gid0;
             arg_reduce_gid < arg_reduce_gid_max; arg_reduce_gid += wg)
        {
            auto inp_reduction_offset =
                inp_reduced_dims_indexer_(arg_reduce_gid);
            auto inp_offset = inp_iter_offset + inp_reduction_offset;

            using dpctl::tensor::type_utils::convert_impl;
            outT val;
            if constexpr (su_ns::IsLogicalAnd<outT, ReductionOp>::value ||
                          su_ns::IsLogicalOr<outT, ReductionOp>::value)
            {
                // handle nans
                val = convert_impl<bool, argT>(inp_[inp_offset]);
            }
            else {
                val = convert_impl<outT, argT>(inp_[inp_offset]);
            }

            local_red_val = reduction_op_(local_red_val, val);
        }

        auto work_group = it.get_group();
        // This only works if reduction_op_ is from small set of operators
        outT red_val_over_wg;
        if constexpr (su_ns::IsLogicalAnd<outT, ReductionOp>::value) {
            red_val_over_wg = static_cast<outT>(
                sycl::all_of_group(work_group, local_red_val));
        }
        else if constexpr (su_ns::IsLogicalOr<outT, ReductionOp>::value) {
            red_val_over_wg = static_cast<outT>(
                sycl::any_of_group(work_group, local_red_val));
        }
        else {
            red_val_over_wg = sycl::reduce_over_group(work_group, local_red_val,
                                                      identity_, reduction_op_);
        }

        if (work_group.leader()) {
            sycl::atomic_ref<outT, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                res_ref(out_[out_iter_offset]);
            if constexpr (su_ns::IsPlus<outT, ReductionOp>::value) {
                res_ref += red_val_over_wg;
            }
            else if constexpr (su_ns::IsMaximum<outT, ReductionOp>::value) {
                res_ref.fetch_max(red_val_over_wg);
            }
            else if constexpr (su_ns::IsMinimum<outT, ReductionOp>::value) {
                res_ref.fetch_min(red_val_over_wg);
            }
            else if constexpr (su_ns::IsLogicalAnd<outT, ReductionOp>::value) {
                res_ref.fetch_and(red_val_over_wg);
            }
            else if constexpr (su_ns::IsLogicalOr<outT, ReductionOp>::value) {
                res_ref.fetch_or(red_val_over_wg);
            }
            else {
                outT read_val = res_ref.load();
                outT new_val{};
                do {
                    new_val = reduction_op_(read_val, red_val_over_wg);
                } while (!res_ref.compare_exchange_strong(read_val, new_val));
            }
        }
    }
};

/* === Reduction, using custom_reduce_over_group, and sycl::atomic_ref === */

template <typename argT,
          typename outT,
          typename ReductionOp,
          typename InputOutputIterIndexerT,
          typename InputRedIndexerT,
          typename SlmT>
struct CustomReductionOverGroupWithAtomicFunctor
{
private:
    const argT *inp_ = nullptr;
    outT *out_ = nullptr;
    ReductionOp reduction_op_;
    outT identity_;
    InputOutputIterIndexerT inp_out_iter_indexer_;
    InputRedIndexerT inp_reduced_dims_indexer_;
    SlmT local_mem_;
    std::size_t reduction_max_gid_ = 0;
    std::size_t iter_gws_ = 1;
    std::size_t reductions_per_wi = 16;

public:
    CustomReductionOverGroupWithAtomicFunctor(
        const argT *data,
        outT *res,
        const ReductionOp &reduction_op,
        const outT &identity_val,
        const InputOutputIterIndexerT &arg_res_iter_indexer,
        const InputRedIndexerT &arg_reduced_dims_indexer,
        SlmT local_mem,
        std::size_t reduction_size,
        std::size_t iteration_size,
        std::size_t reduction_size_per_wi)
        : inp_(data), out_(res), reduction_op_(reduction_op),
          identity_(identity_val), inp_out_iter_indexer_(arg_res_iter_indexer),
          inp_reduced_dims_indexer_(arg_reduced_dims_indexer),
          local_mem_(local_mem), reduction_max_gid_(reduction_size),
          iter_gws_(iteration_size), reductions_per_wi(reduction_size_per_wi)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        const std::size_t iter_gid = it.get_group(0) % iter_gws_;
        const std::size_t reduction_batch_id = it.get_group(0) / iter_gws_;

        const std::size_t reduction_lid = it.get_local_id(0);
        const std::size_t wg =
            it.get_local_range(0); //   0 <= reduction_lid < wg

        // work-items operate over input with indices
        //   inp_data_id = reduction_batch_id * wg * reductions_per_wi + m * wg
        //   + reduction_lid
        // for 0 <= m < reductions_per_wi

        const auto &inp_out_iter_offsets_ = inp_out_iter_indexer_(iter_gid);
        const auto &inp_iter_offset = inp_out_iter_offsets_.get_first_offset();
        const auto &out_iter_offset = inp_out_iter_offsets_.get_second_offset();

        outT local_red_val(identity_);
        std::size_t arg_reduce_gid0 =
            reduction_lid + reduction_batch_id * wg * reductions_per_wi;
        std::size_t arg_reduce_gid_max = std::min(
            reduction_max_gid_, arg_reduce_gid0 + reductions_per_wi * wg);

        for (std::size_t arg_reduce_gid = arg_reduce_gid0;
             arg_reduce_gid < arg_reduce_gid_max; arg_reduce_gid += wg)
        {
            auto inp_reduction_offset =
                inp_reduced_dims_indexer_(arg_reduce_gid);
            auto inp_offset = inp_iter_offset + inp_reduction_offset;

            using dpctl::tensor::type_utils::convert_impl;
            outT val;
            if constexpr (su_ns::IsLogicalAnd<outT, ReductionOp>::value ||
                          su_ns::IsLogicalOr<outT, ReductionOp>::value)
            {
                // handle nans
                val = convert_impl<bool, argT>(inp_[inp_offset]);
            }
            else {
                val = convert_impl<outT, argT>(inp_[inp_offset]);
            }

            local_red_val = reduction_op_(local_red_val, val);
        }

        auto work_group = it.get_group();
        outT red_val_over_wg = su_ns::custom_reduce_over_group(
            work_group, local_mem_, local_red_val, reduction_op_);

        if (work_group.leader()) {
            sycl::atomic_ref<outT, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                res_ref(out_[out_iter_offset]);
            // retain these checks in case a reduce_over_group work-around is
            // needed
            if constexpr (su_ns::IsSyclPlus<outT, ReductionOp>::value) {
                res_ref += red_val_over_wg;
            }
            else if constexpr (su_ns::IsSyclMaximum<outT, ReductionOp>::value) {
                res_ref.fetch_max(red_val_over_wg);
            }
            else if constexpr (su_ns::IsSyclMinimum<outT, ReductionOp>::value) {
                res_ref.fetch_min(red_val_over_wg);
            }
            else if constexpr (su_ns::IsSyclLogicalAnd<outT,
                                                       ReductionOp>::value) {
                res_ref.fetch_and(red_val_over_wg);
            }
            else if constexpr (su_ns::IsSyclLogicalOr<outT, ReductionOp>::value)
            {
                res_ref.fetch_or(red_val_over_wg);
            }
            else {
                outT read_val = res_ref.load();
                outT new_val{};
                do {
                    new_val = reduction_op_(read_val, red_val_over_wg);
                } while (!res_ref.compare_exchange_strong(read_val, new_val));
            }
        }
    }
};

template <typename argT,
          typename outT,
          typename ReductionOp,
          typename InputOutputIterIndexerT,
          typename InputRedIndexerT>
struct ReductionOverGroupNoAtomicFunctor
{
private:
    const argT *inp_ = nullptr;
    outT *out_ = nullptr;
    ReductionOp reduction_op_;
    outT identity_;
    InputOutputIterIndexerT inp_out_iter_indexer_;
    InputRedIndexerT inp_reduced_dims_indexer_;
    std::size_t reduction_max_gid_ = 0;
    std::size_t iter_gws_ = 1;
    std::size_t reductions_per_wi = 16;

public:
    ReductionOverGroupNoAtomicFunctor(
        const argT *data,
        outT *res,
        const ReductionOp &reduction_op,
        const outT &identity_val,
        const InputOutputIterIndexerT &arg_res_iter_indexer,
        const InputRedIndexerT &arg_reduced_dims_indexer,
        std::size_t reduction_size,
        std::size_t iteration_size,
        std::size_t reduction_size_per_wi)
        : inp_(data), out_(res), reduction_op_(reduction_op),
          identity_(identity_val), inp_out_iter_indexer_(arg_res_iter_indexer),
          inp_reduced_dims_indexer_(arg_reduced_dims_indexer),
          reduction_max_gid_(reduction_size), iter_gws_(iteration_size),
          reductions_per_wi(reduction_size_per_wi)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        const std::size_t reduction_lid = it.get_local_id(0);
        const std::size_t wg =
            it.get_local_range(0); //   0 <= reduction_lid < wg

        const std::size_t iter_gid = it.get_group(0) % iter_gws_;
        const std::size_t reduction_batch_id = it.get_group(0) / iter_gws_;
        const std::size_t n_reduction_groups =
            it.get_group_range(0) / iter_gws_;

        // work-items operates over input with indices
        //   inp_data_id = reduction_batch_id * wg * reductions_per_wi + m * wg
        //   + reduction_lid
        // for 0 <= m < reductions_per_wi

        const auto &inp_out_iter_offsets_ = inp_out_iter_indexer_(iter_gid);
        const auto &inp_iter_offset = inp_out_iter_offsets_.get_first_offset();
        const auto &out_iter_offset = inp_out_iter_offsets_.get_second_offset();

        outT local_red_val(identity_);
        std::size_t arg_reduce_gid0 =
            reduction_lid + reduction_batch_id * wg * reductions_per_wi;
        for (std::size_t m = 0; m < reductions_per_wi; ++m) {
            std::size_t arg_reduce_gid = arg_reduce_gid0 + m * wg;

            if (arg_reduce_gid < reduction_max_gid_) {
                auto inp_reduction_offset =
                    inp_reduced_dims_indexer_(arg_reduce_gid);
                auto inp_offset = inp_iter_offset + inp_reduction_offset;

                using dpctl::tensor::type_utils::convert_impl;
                outT val;
                if constexpr (su_ns::IsLogicalAnd<outT, ReductionOp>::value ||
                              su_ns::IsLogicalOr<outT, ReductionOp>::value)
                {
                    // handle nans
                    val = convert_impl<bool, argT>(inp_[inp_offset]);
                }
                else {
                    val = convert_impl<outT, argT>(inp_[inp_offset]);
                }

                local_red_val = reduction_op_(local_red_val, val);
            }
        }

        auto work_group = it.get_group();
        // This only works if reduction_op_ is from small set of operators
        outT red_val_over_wg;
        if constexpr (su_ns::IsLogicalAnd<outT, ReductionOp>::value) {
            red_val_over_wg = sycl::all_of_group(work_group, local_red_val);
        }
        else if constexpr (su_ns::IsLogicalOr<outT, ReductionOp>::value) {
            red_val_over_wg = sycl::any_of_group(work_group, local_red_val);
        }
        else {
            red_val_over_wg = sycl::reduce_over_group(work_group, local_red_val,
                                                      identity_, reduction_op_);
        }

        if (work_group.leader()) {
            // each group writes to a different memory location
            out_[out_iter_offset * n_reduction_groups + reduction_batch_id] =
                red_val_over_wg;
        }
    }
};

/* = Reduction, using custom_reduce_over_group and not using atomic_ref*/

template <typename argT,
          typename outT,
          typename ReductionOp,
          typename InputOutputIterIndexerT,
          typename InputRedIndexerT,
          typename SlmT>
struct CustomReductionOverGroupNoAtomicFunctor
{
private:
    const argT *inp_ = nullptr;
    outT *out_ = nullptr;
    ReductionOp reduction_op_;
    outT identity_;
    InputOutputIterIndexerT inp_out_iter_indexer_;
    InputRedIndexerT inp_reduced_dims_indexer_;
    SlmT local_mem_;
    std::size_t reduction_max_gid_ = 0;
    std::size_t iter_gws_ = 1;
    std::size_t reductions_per_wi = 16;

public:
    CustomReductionOverGroupNoAtomicFunctor(
        const argT *data,
        outT *res,
        const ReductionOp &reduction_op,
        const outT &identity_val,
        const InputOutputIterIndexerT &arg_res_iter_indexer,
        const InputRedIndexerT &arg_reduced_dims_indexer,
        SlmT local_mem,
        std::size_t reduction_size,
        std::size_t iteration_size,
        std::size_t reduction_size_per_wi)
        : inp_(data), out_(res), reduction_op_(reduction_op),
          identity_(identity_val), inp_out_iter_indexer_(arg_res_iter_indexer),
          inp_reduced_dims_indexer_(arg_reduced_dims_indexer),
          local_mem_(local_mem), reduction_max_gid_(reduction_size),
          iter_gws_(iteration_size), reductions_per_wi(reduction_size_per_wi)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        const std::size_t reduction_lid = it.get_local_id(0);
        const std::size_t wg =
            it.get_local_range(0); //   0 <= reduction_lid < wg

        const std::size_t iter_gid = it.get_group(0) % iter_gws_;
        const std::size_t reduction_batch_id = it.get_group(0) / iter_gws_;
        const std::size_t n_reduction_groups =
            it.get_group_range(0) / iter_gws_;

        // work-items operates over input with indices
        //   inp_data_id = reduction_batch_id * wg * reductions_per_wi + m * wg
        //   + reduction_lid
        // for 0 <= m < reductions_per_wi

        auto inp_out_iter_offsets_ = inp_out_iter_indexer_(iter_gid);
        const auto &inp_iter_offset = inp_out_iter_offsets_.get_first_offset();
        const auto &out_iter_offset = inp_out_iter_offsets_.get_second_offset();

        outT local_red_val(identity_);
        std::size_t arg_reduce_gid0 =
            reduction_lid + reduction_batch_id * wg * reductions_per_wi;
        for (std::size_t m = 0; m < reductions_per_wi; ++m) {
            std::size_t arg_reduce_gid = arg_reduce_gid0 + m * wg;

            if (arg_reduce_gid < reduction_max_gid_) {
                auto inp_reduction_offset =
                    inp_reduced_dims_indexer_(arg_reduce_gid);
                auto inp_offset = inp_iter_offset + inp_reduction_offset;

                using dpctl::tensor::type_utils::convert_impl;
                outT val;
                if constexpr (std::is_same_v<ReductionOp,
                                             sycl::logical_and<outT>> ||
                              std::is_same_v<ReductionOp,
                                             sycl::logical_or<outT>>)
                {
                    // handle nans
                    val = convert_impl<bool, argT>(inp_[inp_offset]);
                }
                else {
                    val = convert_impl<outT, argT>(inp_[inp_offset]);
                }

                local_red_val = reduction_op_(local_red_val, val);
            }
        }

        auto work_group = it.get_group();
        // This only works if reduction_op_ is from small set of operators
        outT red_val_over_wg = su_ns::custom_reduce_over_group(
            work_group, local_mem_, local_red_val, reduction_op_);

        if (work_group.leader()) {
            // each group writes to a different memory location
            out_[out_iter_offset * n_reduction_groups + reduction_batch_id] =
                red_val_over_wg;
        }
    }
};

template <
    typename argTy,
    typename resTy,
    typename ReductionOpT,
    typename InputOutputIterIndexerT,
    typename ReductionIndexerT,
    template <typename T1, typename T2, typename T3, typename T4, typename T5>
    class kernel_name_token>
sycl::event
    sequential_reduction(sycl::queue &exec_q,
                         const argTy *arg,
                         resTy *res,
                         resTy identity_val,
                         std::size_t iter_nelems,
                         std::size_t reduction_nelems,
                         const InputOutputIterIndexerT &in_out_iter_indexer,
                         const ReductionIndexerT &reduction_indexer,
                         const std::vector<sycl::event> &depends)
{
    sycl::event red_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using KernelName =
            class kernel_name_token<argTy, resTy, ReductionOpT,
                                    InputOutputIterIndexerT, ReductionIndexerT>;

        cgh.parallel_for<KernelName>(
            sycl::range<1>(iter_nelems),
            SequentialReduction<argTy, resTy, ReductionOpT,
                                InputOutputIterIndexerT, ReductionIndexerT>(
                arg, res, ReductionOpT(), identity_val, in_out_iter_indexer,
                reduction_indexer, reduction_nelems));
    });

    return red_ev;
}

template <typename BasedKernelName>
class custom_reduction_wrapper;

template <
    typename argTy,
    typename resTy,
    typename ReductionOpT,
    typename InputOutputIterIndexerT,
    typename ReductionIndexerT,
    template <typename T1, typename T2, typename T3, typename T4, typename T5>
    class kernel_name_token>
sycl::event
    submit_atomic_reduction(sycl::queue &exec_q,
                            const argTy *arg,
                            resTy *res,
                            resTy identity_val,
                            std::size_t wg,
                            std::size_t iter_nelems,
                            std::size_t reduction_nelems,
                            std::size_t reductions_per_wi,
                            std::size_t reduction_groups,
                            const InputOutputIterIndexerT &in_out_iter_indexer,
                            const ReductionIndexerT &reduction_indexer,
                            const std::vector<sycl::event> &depends)
{
    sycl::event red_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        auto globalRange = sycl::range<1>{iter_nelems * reduction_groups * wg};
        auto localRange = sycl::range<1>{wg};
        auto ndRange = sycl::nd_range<1>(globalRange, localRange);

        if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value) {
            using KernelName =
                class kernel_name_token<argTy, resTy, ReductionOpT,
                                        InputOutputIterIndexerT,
                                        ReductionIndexerT>;

            cgh.parallel_for<KernelName>(
                ndRange,
                ReductionOverGroupWithAtomicFunctor<argTy, resTy, ReductionOpT,
                                                    InputOutputIterIndexerT,
                                                    ReductionIndexerT>(
                    arg, res, ReductionOpT(), identity_val, in_out_iter_indexer,
                    reduction_indexer, reduction_nelems, iter_nelems,
                    reductions_per_wi));
        }
        else {
            using SlmT = sycl::local_accessor<resTy, 1>;
            SlmT local_memory = SlmT(localRange, cgh);

            using KernelName = class custom_reduction_wrapper<
                kernel_name_token<argTy, resTy, ReductionOpT,
                                  InputOutputIterIndexerT, ReductionIndexerT>>;

            cgh.parallel_for<KernelName>(
                ndRange,
                CustomReductionOverGroupWithAtomicFunctor<
                    argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                    ReductionIndexerT, SlmT>(
                    arg, res, ReductionOpT(), identity_val, in_out_iter_indexer,
                    reduction_indexer, local_memory, reduction_nelems,
                    iter_nelems, reductions_per_wi));
        }
    });
    return red_ev;
}

template <typename T1, typename T2, typename T3>
class reduction_over_group_with_atomics_init_krn;

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class reduction_seq_krn;

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class reduction_over_group_with_atomics_krn;

typedef sycl::event (*reduction_strided_impl_fn_ptr)(
    sycl::queue &,
    std::size_t,
    std::size_t,
    const char *,
    char *,
    int,
    const ssize_t *,
    ssize_t,
    ssize_t,
    int,
    const ssize_t *,
    ssize_t,
    const std::vector<sycl::event> &);

using dpctl::tensor::sycl_utils::choose_workgroup_size;

template <typename argTy, typename resTy, typename ReductionOpT>
sycl::event reduction_over_group_with_atomics_strided_impl(
    sycl::queue &exec_q,
    std::size_t iter_nelems,      // number of reductions    (num. of rows in a
                                  // matrix when reducing over rows)
    std::size_t reduction_nelems, // size of each reduction  (length of rows,
                                  // i.e. number of columns)
    const char *arg_cp,
    char *res_cp,
    int iter_nd,
    const ssize_t *iter_shape_and_strides,
    ssize_t iter_arg_offset,
    ssize_t iter_res_offset,
    int red_nd,
    const ssize_t *reduction_shape_stride,
    ssize_t reduction_arg_offset,
    const std::vector<sycl::event> &depends)
{
    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp);
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp);

    static constexpr resTy identity_val =
        su_ns::Identity<ReductionOpT, resTy>::value;

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    std::size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    if (reduction_nelems < wg) {
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;
        using ReductionIndexerT = dpctl::tensor::offset_utils::StridedIndexer;

        const InputOutputIterIndexerT in_out_iter_indexer{
            iter_nd, iter_arg_offset, iter_res_offset, iter_shape_and_strides};
        const ReductionIndexerT reduction_indexer{red_nd, reduction_arg_offset,
                                                  reduction_shape_stride};

        sycl::event comp_ev =
            sequential_reduction<argTy, resTy, ReductionOpT,
                                 InputOutputIterIndexerT, ReductionIndexerT,
                                 reduction_seq_krn>(
                exec_q, arg_tp, res_tp, identity_val, iter_nelems,
                reduction_nelems, in_out_iter_indexer, reduction_indexer,
                depends);

        return comp_ev;
    }
    else {
        sycl::event res_init_ev = exec_q.submit([&](sycl::handler &cgh) {
            using IndexerT =
                dpctl::tensor::offset_utils::UnpackedStridedIndexer;

            const ssize_t *const &res_shape = iter_shape_and_strides;
            const ssize_t *const &res_strides =
                iter_shape_and_strides + 2 * iter_nd;
            const IndexerT res_indexer(iter_nd, iter_res_offset, res_shape,
                                       res_strides);
            using InitKernelName =
                class reduction_over_group_with_atomics_init_krn<resTy, argTy,
                                                                 ReductionOpT>;
            cgh.depends_on(depends);

            cgh.parallel_for<InitKernelName>(
                sycl::range<1>(iter_nelems), [=](sycl::id<1> id) {
                    auto res_offset = res_indexer(id[0]);
                    res_tp[res_offset] = identity_val;
                });
        });

        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;
        using ReductionIndexerT = dpctl::tensor::offset_utils::StridedIndexer;

        const InputOutputIterIndexerT in_out_iter_indexer{
            iter_nd, iter_arg_offset, iter_res_offset, iter_shape_and_strides};
        const ReductionIndexerT reduction_indexer{red_nd, reduction_arg_offset,
                                                  reduction_shape_stride};

        static constexpr std::size_t preferred_reductions_per_wi = 8;
        std::size_t reductions_per_wi =
            (reduction_nelems < preferred_reductions_per_wi * wg)
                ? std::max<std::size_t>(1, (reduction_nelems + wg - 1) / wg)
                : preferred_reductions_per_wi;

        std::size_t reduction_groups =
            (reduction_nelems + reductions_per_wi * wg - 1) /
            (reductions_per_wi * wg);

        sycl::event comp_ev =
            submit_atomic_reduction<argTy, resTy, ReductionOpT,
                                    InputOutputIterIndexerT, ReductionIndexerT,
                                    reduction_over_group_with_atomics_krn>(
                exec_q, arg_tp, res_tp, identity_val, wg, iter_nelems,
                reduction_nelems, reductions_per_wi, reduction_groups,
                in_out_iter_indexer, reduction_indexer, {res_init_ev});

        return comp_ev;
    }
}

// Contig

typedef sycl::event (*reduction_contig_impl_fn_ptr)(
    sycl::queue &,
    std::size_t,
    std::size_t,
    const char *,
    char *,
    ssize_t,
    ssize_t,
    ssize_t,
    const std::vector<sycl::event> &);

/* @brief Reduce rows in a matrix */
template <typename argTy, typename resTy, typename ReductionOpT>
sycl::event reduction_axis1_over_group_with_atomics_contig_impl(
    sycl::queue &exec_q,
    std::size_t iter_nelems,      // number of reductions    (num. of rows in a
                                  // matrix when reducing over rows)
    std::size_t reduction_nelems, // size of each reduction  (length of rows,
                                  // i.e. number of columns)
    const char *arg_cp,
    char *res_cp,
    ssize_t iter_arg_offset,
    ssize_t iter_res_offset,
    ssize_t reduction_arg_offset,
    const std::vector<sycl::event> &depends)
{
    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp) +
                          iter_arg_offset + reduction_arg_offset;
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp) + iter_res_offset;

    static constexpr resTy identity_val =
        su_ns::Identity<ReductionOpT, resTy>::value;

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    std::size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    if (reduction_nelems < wg) {
        using InputIterIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
        using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                InputIterIndexerT, NoOpIndexerT>;
        using ReductionIndexerT = NoOpIndexerT;

        const InputOutputIterIndexerT in_out_iter_indexer{
            InputIterIndexerT{/* size */ iter_nelems,
                              /* step */ reduction_nelems},
            NoOpIndexerT{}};
        static constexpr ReductionIndexerT reduction_indexer{};

        sycl::event comp_ev =
            sequential_reduction<argTy, resTy, ReductionOpT,
                                 InputOutputIterIndexerT, ReductionIndexerT,
                                 reduction_seq_krn>(
                exec_q, arg_tp, res_tp, identity_val, iter_nelems,
                reduction_nelems, in_out_iter_indexer, reduction_indexer,
                depends);

        return comp_ev;
    }
    else {
        sycl::event res_init_ev = exec_q.fill<resTy>(
            res_tp, resTy(identity_val), iter_nelems, depends);

        using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using RowsIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                RowsIndexerT, NoOpIndexerT>;
        using ReductionIndexerT = NoOpIndexerT;

        const RowsIndexerT rows_indexer{/* size */ iter_nelems,
                                        /* step */ reduction_nelems};
        static constexpr NoOpIndexerT result_indexer{};
        const InputOutputIterIndexerT in_out_iter_indexer{rows_indexer,
                                                          result_indexer};
        static constexpr ReductionIndexerT reduction_indexer{};

        static constexpr std::size_t preferred_reductions_per_wi = 8;
        std::size_t reductions_per_wi =
            (reduction_nelems < preferred_reductions_per_wi * wg)
                ? std::max<std::size_t>(1, (reduction_nelems + wg - 1) / wg)
                : preferred_reductions_per_wi;

        std::size_t reduction_groups =
            (reduction_nelems + reductions_per_wi * wg - 1) /
            (reductions_per_wi * wg);

        sycl::event comp_ev =
            submit_atomic_reduction<argTy, resTy, ReductionOpT,
                                    InputOutputIterIndexerT, ReductionIndexerT,
                                    reduction_over_group_with_atomics_krn>(
                exec_q, arg_tp, res_tp, identity_val, wg, iter_nelems,
                reduction_nelems, reductions_per_wi, reduction_groups,
                in_out_iter_indexer, reduction_indexer, {res_init_ev});

        return comp_ev;
    }
}

/* @brief Reduce rows in a matrix */
template <typename argTy, typename resTy, typename ReductionOpT>
sycl::event reduction_axis0_over_group_with_atomics_contig_impl(
    sycl::queue &exec_q,
    std::size_t iter_nelems,      // number of reductions    (num. of cols in a
                                  // matrix when reducing over cols)
    std::size_t reduction_nelems, // size of each reduction  (length of cols,
                                  // i.e. number of rows)
    const char *arg_cp,
    char *res_cp,
    ssize_t iter_arg_offset,
    ssize_t iter_res_offset,
    ssize_t reduction_arg_offset,
    const std::vector<sycl::event> &depends)
{
    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp) +
                          iter_arg_offset + reduction_arg_offset;
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp) + iter_res_offset;

    static constexpr resTy identity_val =
        su_ns::Identity<ReductionOpT, resTy>::value;

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    std::size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    if (reduction_nelems < wg) {
        using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                NoOpIndexerT, NoOpIndexerT>;
        using ReductionIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;

        const InputOutputIterIndexerT in_out_iter_indexer{NoOpIndexerT{},
                                                          NoOpIndexerT{}};
        const ReductionIndexerT reduction_indexer{/* size */ reduction_nelems,
                                                  /* step */ iter_nelems};

        sycl::event comp_ev =
            sequential_reduction<argTy, resTy, ReductionOpT,
                                 InputOutputIterIndexerT, ReductionIndexerT,
                                 reduction_seq_krn>(
                exec_q, arg_tp, res_tp, identity_val, iter_nelems,
                reduction_nelems, in_out_iter_indexer, reduction_indexer,
                depends);

        return comp_ev;
    }
    else {
        sycl::event res_init_ev = exec_q.fill<resTy>(
            res_tp, resTy(identity_val), iter_nelems, depends);

        using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using ColsIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                NoOpIndexerT, NoOpIndexerT>;
        using ReductionIndexerT = ColsIndexerT;

        static constexpr NoOpIndexerT columns_indexer{};
        static constexpr NoOpIndexerT result_indexer{};
        const InputOutputIterIndexerT in_out_iter_indexer{columns_indexer,
                                                          result_indexer};
        const ReductionIndexerT reduction_indexer{/* size */ reduction_nelems,
                                                  /* step */ iter_nelems};

        static constexpr std::size_t preferred_reductions_per_wi = 8;
        std::size_t reductions_per_wi =
            (reduction_nelems < preferred_reductions_per_wi * wg)
                ? std::max<std::size_t>(1, (reduction_nelems + wg - 1) / wg)
                : preferred_reductions_per_wi;

        std::size_t reduction_groups =
            (reduction_nelems + reductions_per_wi * wg - 1) /
            (reductions_per_wi * wg);

        sycl::event comp_ev =
            submit_atomic_reduction<argTy, resTy, ReductionOpT,
                                    InputOutputIterIndexerT, ReductionIndexerT,
                                    reduction_over_group_with_atomics_krn>(
                exec_q, arg_tp, res_tp, identity_val, wg, iter_nelems,
                reduction_nelems, reductions_per_wi, reduction_groups,
                in_out_iter_indexer, reduction_indexer, {res_init_ev});

        return comp_ev;
    }
}

/* = Reduction, using sycl::reduce_over_group, but not using atomic_ref = */

template <
    typename argTy,
    typename resTy,
    typename ReductionOpT,
    typename InputOutputIterIndexerT,
    typename ReductionIndexerT,
    template <typename T1, typename T2, typename T3, typename T4, typename T5>
    class kernel_name_token>
sycl::event submit_no_atomic_reduction(
    sycl::queue &exec_q,
    const argTy *arg,
    resTy *res,
    resTy identity_val,
    std::size_t wg,
    std::size_t iter_nelems,
    std::size_t reduction_nelems,
    std::size_t reductions_per_wi,
    std::size_t reduction_groups,
    const InputOutputIterIndexerT &in_out_iter_indexer,
    const ReductionIndexerT &reduction_indexer,
    const std::vector<sycl::event> &depends)
{
    sycl::event red_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        auto globalRange = sycl::range<1>{iter_nelems * reduction_groups * wg};
        auto localRange = sycl::range<1>{wg};
        auto ndRange = sycl::nd_range<1>(globalRange, localRange);

        if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value) {
            using KernelName =
                class kernel_name_token<argTy, resTy, ReductionOpT,
                                        InputOutputIterIndexerT,
                                        ReductionIndexerT>;

            cgh.parallel_for<KernelName>(
                ndRange,
                ReductionOverGroupNoAtomicFunctor<argTy, resTy, ReductionOpT,
                                                  InputOutputIterIndexerT,
                                                  ReductionIndexerT>(
                    arg, res, ReductionOpT(), identity_val, in_out_iter_indexer,
                    reduction_indexer, reduction_nelems, iter_nelems,
                    reductions_per_wi));
        }
        else {
            using SlmT = sycl::local_accessor<resTy, 1>;
            SlmT local_memory = SlmT(localRange, cgh);
            using KernelName = class custom_reduction_wrapper<
                kernel_name_token<argTy, resTy, ReductionOpT,
                                  InputOutputIterIndexerT, ReductionIndexerT>>;

            cgh.parallel_for<KernelName>(
                ndRange,
                CustomReductionOverGroupNoAtomicFunctor<
                    argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                    ReductionIndexerT, SlmT>(
                    arg, res, ReductionOpT(), identity_val, in_out_iter_indexer,
                    reduction_indexer, local_memory, reduction_nelems,
                    iter_nelems, reductions_per_wi));
        }
    });
    return red_ev;
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class reduction_over_group_temps_krn;

typedef sycl::event (*reduction_strided_impl_fn_ptr)(
    sycl::queue &,
    std::size_t,
    std::size_t,
    const char *,
    char *,
    int,
    const ssize_t *,
    ssize_t,
    ssize_t,
    int,
    const ssize_t *,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename T1, typename T2, typename T3>
class reduction_over_group_temps_empty_krn;

template <typename argTy, typename resTy, typename ReductionOpT>
sycl::event reduction_over_group_temps_strided_impl(
    sycl::queue &exec_q,
    std::size_t iter_nelems,      // number of reductions    (num. of rows in a
                                  // matrix when reducing over rows)
    std::size_t reduction_nelems, // size of each reduction  (length of rows,
                                  // i.e. number of columns)
    const char *arg_cp,
    char *res_cp,
    int iter_nd,
    const ssize_t *iter_shape_and_strides,
    ssize_t iter_arg_offset,
    ssize_t iter_res_offset,
    int red_nd,
    const ssize_t *reduction_shape_stride,
    ssize_t reduction_arg_offset,
    const std::vector<sycl::event> &depends)
{
    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp);
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp);

    static constexpr resTy identity_val =
        su_ns::Identity<ReductionOpT, resTy>::value;

    if (reduction_nelems == 0) {
        sycl::event res_init_ev = exec_q.submit([&](sycl::handler &cgh) {
            using IndexerT =
                dpctl::tensor::offset_utils::UnpackedStridedIndexer;

            const ssize_t *const &res_shape = iter_shape_and_strides;
            const ssize_t *const &res_strides =
                iter_shape_and_strides + 2 * iter_nd;
            const IndexerT res_indexer(iter_nd, iter_res_offset, res_shape,
                                       res_strides);
            using InitKernelName =
                class reduction_over_group_temps_empty_krn<resTy, argTy,
                                                           ReductionOpT>;
            cgh.depends_on(depends);

            cgh.parallel_for<InitKernelName>(
                sycl::range<1>(iter_nelems), [=](sycl::id<1> id) {
                    auto res_offset = res_indexer(id[0]);
                    res_tp[res_offset] = identity_val;
                });
        });

        return res_init_ev;
    }

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    std::size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    if (reduction_nelems < wg) {
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;
        using ReductionIndexerT = dpctl::tensor::offset_utils::StridedIndexer;

        const InputOutputIterIndexerT in_out_iter_indexer{
            iter_nd, iter_arg_offset, iter_res_offset, iter_shape_and_strides};
        const ReductionIndexerT reduction_indexer{red_nd, reduction_arg_offset,
                                                  reduction_shape_stride};

        sycl::event comp_ev =
            sequential_reduction<argTy, resTy, ReductionOpT,
                                 InputOutputIterIndexerT, ReductionIndexerT,
                                 reduction_seq_krn>(
                exec_q, arg_tp, res_tp, identity_val, iter_nelems,
                reduction_nelems, in_out_iter_indexer, reduction_indexer,
                depends);

        return comp_ev;
    }

    static constexpr std::size_t preferred_reductions_per_wi = 8;
    // prevents running out of resources on CPU
    std::size_t max_wg = reduction_detail::get_work_group_size(d);

    std::size_t reductions_per_wi(preferred_reductions_per_wi);
    if (reduction_nelems <= preferred_reductions_per_wi * max_wg) {
        // Perform reduction using one 1 work-group per iteration,
        // can output directly to res

        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;
        using ReductionIndexerT = dpctl::tensor::offset_utils::StridedIndexer;

        const InputOutputIterIndexerT in_out_iter_indexer{
            iter_nd, iter_arg_offset, iter_res_offset, iter_shape_and_strides};
        const ReductionIndexerT reduction_indexer{red_nd, reduction_arg_offset,
                                                  reduction_shape_stride};

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

        sycl::event comp_ev = submit_no_atomic_reduction<
            argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
            ReductionIndexerT, reduction_over_group_temps_krn>(
            exec_q, arg_tp, res_tp, identity_val, wg, iter_nelems,
            reduction_nelems, reductions_per_wi, reduction_groups,
            in_out_iter_indexer, reduction_indexer, depends);

        return comp_ev;
    }
    else {
        // more than one work-groups is needed, requires a temporary
        std::size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);
        assert(reduction_groups > 1);

        std::size_t second_iter_reduction_groups_ =
            (reduction_groups + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);

        const std::size_t tmp_alloc_size =
            iter_nelems * (reduction_groups + second_iter_reduction_groups_);
        auto tmp_owner = dpctl::tensor::alloc_utils::smart_malloc_device<resTy>(
            tmp_alloc_size, exec_q);

        resTy *partially_reduced_tmp = tmp_owner.get();
        resTy *partially_reduced_tmp2 =
            partially_reduced_tmp + reduction_groups * iter_nelems;
        ;

        sycl::event first_reduction_ev;
        {
            using InputIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
            using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    InputIndexerT, ResIndexerT>;
            using ReductionIndexerT =
                dpctl::tensor::offset_utils::StridedIndexer;

            // Only 2*iter_nd entries describing shape and strides of
            // iterated dimensions of input array from
            // iter_shape_and_strides are going to be accessed by
            // inp_indexer
            const InputIndexerT inp_indexer(iter_nd, iter_arg_offset,
                                            iter_shape_and_strides);
            static constexpr ResIndexerT noop_tmp_indexer{};

            const InputOutputIterIndexerT in_out_iter_indexer{inp_indexer,
                                                              noop_tmp_indexer};
            const ReductionIndexerT reduction_indexer{
                red_nd, reduction_arg_offset, reduction_shape_stride};

            first_reduction_ev = submit_no_atomic_reduction<
                argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                ReductionIndexerT, reduction_over_group_temps_krn>(
                exec_q, arg_tp, partially_reduced_tmp, identity_val, wg,
                iter_nelems, reduction_nelems, preferred_reductions_per_wi,
                reduction_groups, in_out_iter_indexer, reduction_indexer,
                depends);
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

            // keep reducing
            sycl::event partial_reduction_ev;
            {
                using InputIndexerT =
                    dpctl::tensor::offset_utils::Strided1DIndexer;
                using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
                using InputOutputIterIndexerT =
                    dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                        InputIndexerT, ResIndexerT>;
                using ReductionIndexerT =
                    dpctl::tensor::offset_utils::NoOpIndexer;

                const InputIndexerT inp_indexer{/* size */ iter_nelems,
                                                /* step */ reduction_groups_};
                static constexpr ResIndexerT res_iter_indexer{};

                const InputOutputIterIndexerT in_out_iter_indexer{
                    inp_indexer, res_iter_indexer};
                static constexpr ReductionIndexerT reduction_indexer{};

                partial_reduction_ev = submit_no_atomic_reduction<
                    resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                    ReductionIndexerT, reduction_over_group_temps_krn>(
                    exec_q, temp_arg, temp2_arg, identity_val, wg, iter_nelems,
                    remaining_reduction_nelems, preferred_reductions_per_wi,
                    reduction_groups_, in_out_iter_indexer, reduction_indexer,
                    {dependent_ev});
            }

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

        const InputIndexerT inp_indexer{/* size */ iter_nelems,
                                        /* step */ remaining_reduction_nelems};
        const ResIndexerT res_iter_indexer{
            iter_nd, iter_res_offset,
            /* shape */ iter_shape_and_strides,
            /* strides */ iter_shape_and_strides + 2 * iter_nd};

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

        sycl::event final_reduction_ev = submit_no_atomic_reduction<
            resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
            ReductionIndexerT, reduction_over_group_temps_krn>(
            exec_q, temp_arg, res_tp, identity_val, wg, iter_nelems,
            remaining_reduction_nelems, reductions_per_wi, reduction_groups,
            in_out_iter_indexer, reduction_indexer, {dependent_ev});

        sycl::event cleanup_host_task_event =
            dpctl::tensor::alloc_utils::async_smart_free(
                exec_q, {final_reduction_ev}, tmp_owner);

        // FIXME: do not return host-task event
        //   Instead collect all host-tasks to a list

        return cleanup_host_task_event;
    }
}

template <typename argTy, typename resTy, typename ReductionOpT>
sycl::event reduction_axis1_over_group_temps_contig_impl(
    sycl::queue &exec_q,
    std::size_t iter_nelems,      // number of reductions    (num. of rows in a
                                  // matrix when reducing over rows)
    std::size_t reduction_nelems, // size of each reduction  (length of rows,
                                  // i.e. number of columns)
    const char *arg_cp,
    char *res_cp,
    ssize_t iter_arg_offset,
    ssize_t iter_res_offset,
    ssize_t reduction_arg_offset,
    const std::vector<sycl::event> &depends)
{
    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp) +
                          iter_arg_offset + reduction_arg_offset;
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp) + iter_res_offset;

    static constexpr resTy identity_val =
        su_ns::Identity<ReductionOpT, resTy>::value;

    if (reduction_nelems == 0) {
        sycl::event res_init_ev = exec_q.fill<resTy>(
            res_tp, resTy(identity_val), iter_nelems, depends);

        return res_init_ev;
    }

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    std::size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    if (reduction_nelems < wg) {
        using InputIterIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
        using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                InputIterIndexerT, NoOpIndexerT>;
        using ReductionIndexerT = NoOpIndexerT;

        const InputOutputIterIndexerT in_out_iter_indexer{
            InputIterIndexerT{/* size */ iter_nelems,
                              /* step */ reduction_nelems},
            NoOpIndexerT{}};
        static constexpr ReductionIndexerT reduction_indexer{};

        sycl::event comp_ev =
            sequential_reduction<argTy, resTy, ReductionOpT,
                                 InputOutputIterIndexerT, ReductionIndexerT,
                                 reduction_seq_krn>(
                exec_q, arg_tp, res_tp, identity_val, iter_nelems,
                reduction_nelems, in_out_iter_indexer, reduction_indexer,
                depends);

        return comp_ev;
    }

    static constexpr std::size_t preferred_reductions_per_wi = 8;
    // prevents running out of resources on CPU
    std::size_t max_wg = reduction_detail::get_work_group_size(d);

    std::size_t reductions_per_wi(preferred_reductions_per_wi);
    if (reduction_nelems <= preferred_reductions_per_wi * max_wg) {
        // Perform reduction using one 1 work-group per iteration,
        // can output directly to res

        using InputIterIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
        using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                InputIterIndexerT, NoOpIndexerT>;
        using ReductionIndexerT = NoOpIndexerT;

        const InputOutputIterIndexerT in_out_iter_indexer{
            InputIterIndexerT{/* size */ iter_nelems,
                              /* step */ reduction_nelems},
            NoOpIndexerT{}};
        static constexpr ReductionIndexerT reduction_indexer{};

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

        sycl::event comp_ev = submit_no_atomic_reduction<
            argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
            ReductionIndexerT, reduction_over_group_temps_krn>(
            exec_q, arg_tp, res_tp, identity_val, wg, iter_nelems,
            reduction_nelems, reductions_per_wi, reduction_groups,
            in_out_iter_indexer, reduction_indexer, depends);

        return comp_ev;
    }
    else {
        // more than one work-groups is needed, requires a temporary
        std::size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);
        assert(reduction_groups > 1);

        std::size_t second_iter_reduction_groups_ =
            (reduction_groups + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);

        const std::size_t tmp_alloc_size =
            iter_nelems * (reduction_groups + second_iter_reduction_groups_);
        auto tmp_owner = dpctl::tensor::alloc_utils::smart_malloc_device<resTy>(
            tmp_alloc_size, exec_q);
        resTy *partially_reduced_tmp = tmp_owner.get();
        resTy *partially_reduced_tmp2 =
            partially_reduced_tmp + reduction_groups * iter_nelems;

        sycl::event first_reduction_ev;
        {
            using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using RowsIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    RowsIndexerT, NoOpIndexerT>;
            using ReductionIndexerT = NoOpIndexerT;

            const RowsIndexerT rows_indexer{/* size */ iter_nelems,
                                            /* step */ reduction_nelems};
            static constexpr NoOpIndexerT noop_tmp_indexer{};
            const InputOutputIterIndexerT in_out_iter_indexer{rows_indexer,
                                                              noop_tmp_indexer};
            static constexpr ReductionIndexerT reduction_indexer{};

            first_reduction_ev = submit_no_atomic_reduction<
                argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                ReductionIndexerT, reduction_over_group_temps_krn>(
                exec_q, arg_tp, partially_reduced_tmp, identity_val, wg,
                iter_nelems, reduction_nelems, preferred_reductions_per_wi,
                reduction_groups, in_out_iter_indexer, reduction_indexer,
                depends);
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

            sycl::event partial_reduction_ev = submit_no_atomic_reduction<
                resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                ReductionIndexerT, reduction_over_group_temps_krn>(
                exec_q, temp_arg, temp2_arg, identity_val, wg, iter_nelems,
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

        const InputIndexerT inp_indexer{/* size */ iter_nelems,
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

        sycl::event final_reduction_ev = submit_no_atomic_reduction<
            resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
            ReductionIndexerT, reduction_over_group_temps_krn>(
            exec_q, temp_arg, res_tp, identity_val, wg, iter_nelems,
            remaining_reduction_nelems, reductions_per_wi, reduction_groups,
            in_out_iter_indexer, reduction_indexer, {dependent_ev});

        sycl::event cleanup_host_task_event =
            dpctl::tensor::alloc_utils::async_smart_free(
                exec_q, {final_reduction_ev}, tmp_owner);

        // FIXME: do not return host-task event
        //   Instead collect all host-tasks to a list

        return cleanup_host_task_event;
    }
}

template <typename argTy, typename resTy, typename ReductionOpT>
sycl::event reduction_axis0_over_group_temps_contig_impl(
    sycl::queue &exec_q,
    std::size_t iter_nelems,      // number of reductions    (num. of rows in a
                                  // matrix when reducing over rows)
    std::size_t reduction_nelems, // size of each reduction  (length of rows,
                                  // i.e. number of columns)
    const char *arg_cp,
    char *res_cp,
    ssize_t iter_arg_offset,
    ssize_t iter_res_offset,
    ssize_t reduction_arg_offset,
    const std::vector<sycl::event> &depends)
{
    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp) +
                          iter_arg_offset + reduction_arg_offset;
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp) + iter_res_offset;

    static constexpr resTy identity_val =
        su_ns::Identity<ReductionOpT, resTy>::value;

    if (reduction_nelems == 0) {
        sycl::event res_init_ev = exec_q.fill<resTy>(
            res_tp, resTy(identity_val), iter_nelems, depends);

        return res_init_ev;
    }

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    std::size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    if (reduction_nelems < wg) {
        using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                NoOpIndexerT, NoOpIndexerT>;
        using ReductionIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;

        const InputOutputIterIndexerT in_out_iter_indexer{NoOpIndexerT{},
                                                          NoOpIndexerT{}};
        const ReductionIndexerT reduction_indexer{/* size */ reduction_nelems,
                                                  /* step */ iter_nelems};

        sycl::event comp_ev =
            sequential_reduction<argTy, resTy, ReductionOpT,
                                 InputOutputIterIndexerT, ReductionIndexerT,
                                 reduction_seq_krn>(
                exec_q, arg_tp, res_tp, identity_val, iter_nelems,
                reduction_nelems, in_out_iter_indexer, reduction_indexer,
                depends);

        return comp_ev;
    }

    static constexpr std::size_t preferred_reductions_per_wi = 8;
    // prevents running out of resources on CPU
    std::size_t max_wg = reduction_detail::get_work_group_size(d);

    std::size_t reductions_per_wi(preferred_reductions_per_wi);
    if (reduction_nelems <= preferred_reductions_per_wi * max_wg) {
        // Perform reduction using one 1 work-group per iteration,
        // can output directly to res

        using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using ColsIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                NoOpIndexerT, NoOpIndexerT>;
        using ReductionIndexerT = ColsIndexerT;

        static constexpr NoOpIndexerT columns_indexer{};
        static constexpr NoOpIndexerT result_indexer{};
        const InputOutputIterIndexerT in_out_iter_indexer{columns_indexer,
                                                          result_indexer};
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

        sycl::event comp_ev = submit_no_atomic_reduction<
            argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
            ReductionIndexerT, reduction_over_group_temps_krn>(
            exec_q, arg_tp, res_tp, identity_val, wg, iter_nelems,
            reduction_nelems, reductions_per_wi, reduction_groups,
            in_out_iter_indexer, reduction_indexer, depends);

        return comp_ev;
    }
    else {
        // more than one work-groups is needed, requires a temporary
        std::size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);
        assert(reduction_groups > 1);

        std::size_t second_iter_reduction_groups_ =
            (reduction_groups + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);

        const std::size_t tmp_alloc_size =
            iter_nelems * (reduction_groups + second_iter_reduction_groups_);

        auto tmp_owner = dpctl::tensor::alloc_utils::smart_malloc_device<resTy>(
            tmp_alloc_size, exec_q);

        resTy *partially_reduced_tmp = tmp_owner.get();
        resTy *partially_reduced_tmp2 =
            partially_reduced_tmp + reduction_groups * iter_nelems;

        sycl::event first_reduction_ev;
        {
            using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using ColsIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    NoOpIndexerT, NoOpIndexerT>;
            using ReductionIndexerT = ColsIndexerT;

            static constexpr NoOpIndexerT columns_indexer{};
            static constexpr NoOpIndexerT noop_tmp_indexer{};
            const InputOutputIterIndexerT in_out_iter_indexer{columns_indexer,
                                                              noop_tmp_indexer};
            const ReductionIndexerT reduction_indexer{
                /* size */ reduction_nelems,
                /* step */ iter_nelems};

            first_reduction_ev = submit_no_atomic_reduction<
                argTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                ReductionIndexerT, reduction_over_group_temps_krn>(
                exec_q, arg_tp, partially_reduced_tmp, identity_val, wg,
                iter_nelems, reduction_nelems, preferred_reductions_per_wi,
                reduction_groups, in_out_iter_indexer, reduction_indexer,
                depends);
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

            sycl::event partial_reduction_ev = submit_no_atomic_reduction<
                resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
                ReductionIndexerT, reduction_over_group_temps_krn>(
                exec_q, temp_arg, temp2_arg, identity_val, wg, iter_nelems,
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

        const InputIndexerT inp_indexer{/* size */ iter_nelems,
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

        sycl::event final_reduction_ev = submit_no_atomic_reduction<
            resTy, resTy, ReductionOpT, InputOutputIterIndexerT,
            ReductionIndexerT, reduction_over_group_temps_krn>(
            exec_q, temp_arg, res_tp, identity_val, wg, iter_nelems,
            remaining_reduction_nelems, reductions_per_wi, reduction_groups,
            in_out_iter_indexer, reduction_indexer, {dependent_ev});

        sycl::event cleanup_host_task_event =
            dpctl::tensor::alloc_utils::async_smart_free(
                exec_q, {final_reduction_ev}, tmp_owner);

        // FIXME: do not return host-task event
        //   Instead collect all host-tasks to a list

        return cleanup_host_task_event;
    }
}

// Argmax and Argmin

/* Sequential search reduction */

template <typename argT,
          typename outT,
          typename ReductionOp,
          typename IdxReductionOp,
          typename InputOutputIterIndexerT,
          typename InputRedIndexerT>
struct SequentialSearchReduction
{
private:
    const argT *inp_ = nullptr;
    outT *out_ = nullptr;
    ReductionOp reduction_op_;
    argT identity_;
    IdxReductionOp idx_reduction_op_;
    outT idx_identity_;
    InputOutputIterIndexerT inp_out_iter_indexer_;
    InputRedIndexerT inp_reduced_dims_indexer_;
    std::size_t reduction_max_gid_ = 0;

public:
    SequentialSearchReduction(
        const argT *inp,
        outT *res,
        const ReductionOp &reduction_op,
        const argT &identity_val,
        const IdxReductionOp &idx_reduction_op,
        const outT &idx_identity_val,
        const InputOutputIterIndexerT &arg_res_iter_indexer,
        const InputRedIndexerT &arg_reduced_dims_indexer,
        std::size_t reduction_size)
        : inp_(inp), out_(res), reduction_op_(reduction_op),
          identity_(identity_val), idx_reduction_op_(idx_reduction_op),
          idx_identity_(idx_identity_val),
          inp_out_iter_indexer_(arg_res_iter_indexer),
          inp_reduced_dims_indexer_(arg_reduced_dims_indexer),
          reduction_max_gid_(reduction_size)
    {
    }

    void operator()(sycl::id<1> id) const
    {

        auto const &inp_out_iter_offsets_ = inp_out_iter_indexer_(id[0]);
        const ssize_t &inp_iter_offset =
            inp_out_iter_offsets_.get_first_offset();
        const ssize_t &out_iter_offset =
            inp_out_iter_offsets_.get_second_offset();

        argT red_val(identity_);
        outT idx_val(idx_identity_);
        for (std::size_t m = 0; m < reduction_max_gid_; ++m) {
            const ssize_t inp_reduction_offset = inp_reduced_dims_indexer_(m);
            const ssize_t inp_offset = inp_iter_offset + inp_reduction_offset;

            argT val = inp_[inp_offset];
            if (val == red_val) {
                idx_val = idx_reduction_op_(idx_val, static_cast<outT>(m));
            }
            else {
                if constexpr (su_ns::IsMinimum<argT, ReductionOp>::value) {
                    using dpctl::tensor::type_utils::is_complex;
                    if constexpr (is_complex<argT>::value) {
                        using dpctl::tensor::math_utils::less_complex;
                        // less_complex always returns false for NaNs, so check
                        if (less_complex<argT>(val, red_val) ||
                            std::isnan(std::real(val)) ||
                            std::isnan(std::imag(val)))
                        {
                            red_val = val;
                            idx_val = static_cast<outT>(m);
                        }
                    }
                    else if constexpr (std::is_floating_point_v<argT> ||
                                       std::is_same_v<argT, sycl::half>)
                    {
                        if (val < red_val || std::isnan(val)) {
                            red_val = val;
                            idx_val = static_cast<outT>(m);
                        }
                    }
                    else {
                        if (val < red_val) {
                            red_val = val;
                            idx_val = static_cast<outT>(m);
                        }
                    }
                }
                else if constexpr (su_ns::IsMaximum<argT, ReductionOp>::value) {
                    using dpctl::tensor::type_utils::is_complex;
                    if constexpr (is_complex<argT>::value) {
                        using dpctl::tensor::math_utils::greater_complex;
                        if (greater_complex<argT>(val, red_val) ||
                            std::isnan(std::real(val)) ||
                            std::isnan(std::imag(val)))
                        {
                            red_val = val;
                            idx_val = static_cast<outT>(m);
                        }
                    }
                    else if constexpr (std::is_floating_point_v<argT> ||
                                       std::is_same_v<argT, sycl::half>)
                    {
                        if (val > red_val || std::isnan(val)) {
                            red_val = val;
                            idx_val = static_cast<outT>(m);
                        }
                    }
                    else {
                        if (val > red_val) {
                            red_val = val;
                            idx_val = static_cast<outT>(m);
                        }
                    }
                }
            }
        }
        out_[out_iter_offset] = idx_val;
    }
};

/* = Search reduction using reduce_over_group*/

template <typename argT,
          typename outT,
          typename ReductionOp,
          typename IdxReductionOp,
          typename InputOutputIterIndexerT,
          typename InputRedIndexerT,
          bool First,
          bool Last>
struct SearchReduction
{
private:
    const argT *inp_ = nullptr;
    argT *vals_ = nullptr;
    const outT *inds_ = nullptr;
    outT *out_ = nullptr;
    ReductionOp reduction_op_;
    argT identity_;
    IdxReductionOp idx_reduction_op_;
    outT idx_identity_;
    InputOutputIterIndexerT inp_out_iter_indexer_;
    InputRedIndexerT inp_reduced_dims_indexer_;
    std::size_t reduction_max_gid_ = 0;
    std::size_t iter_gws_ = 1;
    std::size_t reductions_per_wi = 16;

public:
    SearchReduction(const argT *data,
                    argT *vals,
                    const outT *inds,
                    outT *res,
                    const ReductionOp &reduction_op,
                    const argT &identity_val,
                    const IdxReductionOp &idx_reduction_op,
                    const outT &idx_identity_val,
                    const InputOutputIterIndexerT &arg_res_iter_indexer,
                    const InputRedIndexerT &arg_reduced_dims_indexer,
                    std::size_t reduction_size,
                    std::size_t iteration_size,
                    std::size_t reduction_size_per_wi)
        : inp_(data), vals_(vals), inds_(inds), out_(res),
          reduction_op_(reduction_op), identity_(identity_val),
          idx_reduction_op_(idx_reduction_op), idx_identity_(idx_identity_val),
          inp_out_iter_indexer_(arg_res_iter_indexer),
          inp_reduced_dims_indexer_(arg_reduced_dims_indexer),
          reduction_max_gid_(reduction_size), iter_gws_(iteration_size),
          reductions_per_wi(reduction_size_per_wi)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        const std::size_t reduction_lid = it.get_local_id(0);
        const std::size_t wg =
            it.get_local_range(0); //   0 <= reduction_lid < wg

        const std::size_t iter_gid = it.get_group(0) % iter_gws_;
        const std::size_t reduction_batch_id = it.get_group(0) / iter_gws_;
        const std::size_t n_reduction_groups =
            it.get_group_range(0) / iter_gws_;

        // work-items operates over input with indices
        //   inp_data_id = reduction_batch_id * wg * reductions_per_wi + m * wg
        //   + reduction_lid
        // for 0 <= m < reductions_per_wi

        const auto &inp_out_iter_offsets_ = inp_out_iter_indexer_(iter_gid);
        const auto &inp_iter_offset = inp_out_iter_offsets_.get_first_offset();
        const auto &out_iter_offset = inp_out_iter_offsets_.get_second_offset();

        argT local_red_val(identity_);
        outT local_idx(idx_identity_);
        std::size_t arg_reduce_gid0 =
            reduction_lid + reduction_batch_id * wg * reductions_per_wi;
        for (std::size_t m = 0; m < reductions_per_wi; ++m) {
            std::size_t arg_reduce_gid = arg_reduce_gid0 + m * wg;

            if (arg_reduce_gid < reduction_max_gid_) {
                auto inp_reduction_offset =
                    inp_reduced_dims_indexer_(arg_reduce_gid);
                auto inp_offset = inp_iter_offset + inp_reduction_offset;

                argT val = inp_[inp_offset];
                if (val == local_red_val) {
                    if constexpr (!First) {
                        local_idx =
                            idx_reduction_op_(local_idx, inds_[inp_offset]);
                    }
                    else {
                        local_idx = idx_reduction_op_(
                            local_idx, static_cast<outT>(arg_reduce_gid));
                    }
                }
                else {
                    if constexpr (su_ns::IsMinimum<argT, ReductionOp>::value) {
                        if (val < local_red_val) {
                            local_red_val = val;
                            if constexpr (!First) {
                                local_idx = inds_[inp_offset];
                            }
                            else {
                                local_idx = static_cast<outT>(arg_reduce_gid);
                            }
                        }
                    }
                    else if constexpr (su_ns::IsMaximum<argT,
                                                        ReductionOp>::value) {
                        if (val > local_red_val) {
                            local_red_val = val;
                            if constexpr (!First) {
                                local_idx = inds_[inp_offset];
                            }
                            else {
                                local_idx = static_cast<outT>(arg_reduce_gid);
                            }
                        }
                    }
                }
            }
        }

        auto work_group = it.get_group();
        // This only works if reduction_op_ is from small set of operators
        argT red_val_over_wg = sycl::reduce_over_group(
            work_group, local_red_val, identity_, reduction_op_);

        if constexpr (std::is_integral_v<argT>) {
            local_idx =
                (red_val_over_wg == local_red_val) ? local_idx : idx_identity_;
        }
        else {
            local_idx =
                (red_val_over_wg == local_red_val ||
                 std::isnan(red_val_over_wg) || std::isnan(local_red_val))
                    ? local_idx
                    : idx_identity_;
        }
        outT idx_over_wg = sycl::reduce_over_group(
            work_group, local_idx, idx_identity_, idx_reduction_op_);

        if (work_group.leader()) {
            // each group writes to a different memory location
            if constexpr (!Last) {
                // if not the final reduction, write value corresponding to
                // an index to a temporary
                vals_[out_iter_offset * n_reduction_groups +
                      reduction_batch_id] = red_val_over_wg;
            }
            out_[out_iter_offset * n_reduction_groups + reduction_batch_id] =
                idx_over_wg;
        }
    }
};

/* = Search reduction using custom_reduce_over_group*/

template <typename argT,
          typename outT,
          typename ReductionOp,
          typename IdxReductionOp,
          typename InputOutputIterIndexerT,
          typename InputRedIndexerT,
          typename SlmT,
          bool First,
          bool Last>
struct CustomSearchReduction
{
private:
    const argT *inp_ = nullptr;
    argT *vals_ = nullptr;
    const outT *inds_ = nullptr;
    outT *out_ = nullptr;
    ReductionOp reduction_op_;
    argT identity_;
    IdxReductionOp idx_reduction_op_;
    outT idx_identity_;
    InputOutputIterIndexerT inp_out_iter_indexer_;
    InputRedIndexerT inp_reduced_dims_indexer_;
    SlmT local_mem_;
    std::size_t reduction_max_gid_ = 0;
    std::size_t iter_gws_ = 1;
    std::size_t reductions_per_wi = 16;

public:
    CustomSearchReduction(const argT *data,
                          argT *vals,
                          outT *inds,
                          outT *res,
                          const ReductionOp &reduction_op,
                          const argT &identity_val,
                          const IdxReductionOp &idx_reduction_op,
                          const outT &idx_identity_val,
                          const InputOutputIterIndexerT &arg_res_iter_indexer,
                          const InputRedIndexerT &arg_reduced_dims_indexer,
                          SlmT local_mem,
                          std::size_t reduction_size,
                          std::size_t iteration_size,
                          std::size_t reduction_size_per_wi)
        : inp_(data), vals_(vals), inds_(inds), out_(res),
          reduction_op_(reduction_op), identity_(identity_val),
          idx_reduction_op_(idx_reduction_op), idx_identity_(idx_identity_val),
          inp_out_iter_indexer_(arg_res_iter_indexer),
          inp_reduced_dims_indexer_(arg_reduced_dims_indexer),
          local_mem_(local_mem), reduction_max_gid_(reduction_size),
          iter_gws_(iteration_size), reductions_per_wi(reduction_size_per_wi)
    {
    }

    void operator()(sycl::nd_item<1> it) const
    {
        const std::size_t reduction_lid = it.get_local_id(0);
        const std::size_t wg =
            it.get_local_range(0); //   0 <= reduction_lid < wg

        const std::size_t iter_gid = it.get_group(0) % iter_gws_;
        const std::size_t reduction_batch_id = it.get_group(0) / iter_gws_;
        const std::size_t n_reduction_groups =
            it.get_group_range(0) / iter_gws_;

        // work-items operates over input with indices
        //   inp_data_id = reduction_batch_id * wg * reductions_per_wi + m * wg
        //   + reduction_lid
        // for 0 <= m < reductions_per_wi

        const auto &inp_out_iter_offsets_ = inp_out_iter_indexer_(iter_gid);
        const auto &inp_iter_offset = inp_out_iter_offsets_.get_first_offset();
        const auto &out_iter_offset = inp_out_iter_offsets_.get_second_offset();

        argT local_red_val(identity_);
        outT local_idx(idx_identity_);
        std::size_t arg_reduce_gid0 =
            reduction_lid + reduction_batch_id * wg * reductions_per_wi;
        for (std::size_t m = 0; m < reductions_per_wi; ++m) {
            std::size_t arg_reduce_gid = arg_reduce_gid0 + m * wg;

            if (arg_reduce_gid < reduction_max_gid_) {
                auto inp_reduction_offset =
                    inp_reduced_dims_indexer_(arg_reduce_gid);
                auto inp_offset = inp_iter_offset + inp_reduction_offset;

                argT val = inp_[inp_offset];
                if (val == local_red_val) {
                    if constexpr (!First) {
                        local_idx =
                            idx_reduction_op_(local_idx, inds_[inp_offset]);
                    }
                    else {
                        local_idx = idx_reduction_op_(
                            local_idx, static_cast<outT>(arg_reduce_gid));
                    }
                }
                else {
                    if constexpr (su_ns::IsMinimum<argT, ReductionOp>::value) {
                        using dpctl::tensor::type_utils::is_complex;
                        if constexpr (is_complex<argT>::value) {
                            using dpctl::tensor::math_utils::less_complex;
                            // less_complex always returns false for NaNs, so
                            // check
                            if (less_complex<argT>(val, local_red_val) ||
                                std::isnan(std::real(val)) ||
                                std::isnan(std::imag(val)))
                            {
                                local_red_val = val;
                                if constexpr (!First) {
                                    local_idx = inds_[inp_offset];
                                }
                                else {
                                    local_idx =
                                        static_cast<outT>(arg_reduce_gid);
                                }
                            }
                        }
                        else if constexpr (std::is_floating_point_v<argT> ||
                                           std::is_same_v<argT, sycl::half>)
                        {
                            if (val < local_red_val || std::isnan(val)) {
                                local_red_val = val;
                                if constexpr (!First) {
                                    local_idx = inds_[inp_offset];
                                }
                                else {
                                    local_idx =
                                        static_cast<outT>(arg_reduce_gid);
                                }
                            }
                        }
                        else {
                            if (val < local_red_val) {
                                local_red_val = val;
                                if constexpr (!First) {
                                    local_idx = inds_[inp_offset];
                                }
                                else {
                                    local_idx =
                                        static_cast<outT>(arg_reduce_gid);
                                }
                            }
                        }
                    }
                    else if constexpr (su_ns::IsMaximum<argT,
                                                        ReductionOp>::value) {
                        using dpctl::tensor::type_utils::is_complex;
                        if constexpr (is_complex<argT>::value) {
                            using dpctl::tensor::math_utils::greater_complex;
                            if (greater_complex<argT>(val, local_red_val) ||
                                std::isnan(std::real(val)) ||
                                std::isnan(std::imag(val)))
                            {
                                local_red_val = val;
                                if constexpr (!First) {
                                    local_idx = inds_[inp_offset];
                                }
                                else {
                                    local_idx =
                                        static_cast<outT>(arg_reduce_gid);
                                }
                            }
                        }
                        else if constexpr (std::is_floating_point_v<argT> ||
                                           std::is_same_v<argT, sycl::half>)
                        {
                            if (val > local_red_val || std::isnan(val)) {
                                local_red_val = val;
                                if constexpr (!First) {
                                    local_idx = inds_[inp_offset];
                                }
                                else {
                                    local_idx =
                                        static_cast<outT>(arg_reduce_gid);
                                }
                            }
                        }
                        else {
                            if (val > local_red_val) {
                                local_red_val = val;
                                if constexpr (!First) {
                                    local_idx = inds_[inp_offset];
                                }
                                else {
                                    local_idx =
                                        static_cast<outT>(arg_reduce_gid);
                                }
                            }
                        }
                    }
                }
            }
        }

        auto work_group = it.get_group();
        // This only works if reduction_op_ is from small set of operators
        argT red_val_over_wg = su_ns::custom_reduce_over_group(
            work_group, local_mem_, local_red_val, reduction_op_);

        using dpctl::tensor::type_utils::is_complex;
        if constexpr (is_complex<argT>::value) {
            // equality does not hold for NaNs, so check here
            local_idx = (red_val_over_wg == local_red_val ||
                         std::isnan(std::real(local_red_val)) ||
                         std::isnan(std::imag(local_red_val)))
                            ? local_idx
                            : idx_identity_;
        }
        else if constexpr (std::is_floating_point_v<argT> ||
                           std::is_same_v<argT, sycl::half>)
        {
            // equality does not hold for NaNs, so check here
            local_idx =
                (red_val_over_wg == local_red_val || std::isnan(local_red_val))
                    ? local_idx
                    : idx_identity_;
        }
        else {
            local_idx =
                red_val_over_wg == local_red_val ? local_idx : idx_identity_;
        }
        outT idx_over_wg = sycl::reduce_over_group(
            work_group, local_idx, idx_identity_, idx_reduction_op_);
        if (work_group.leader()) {
            // each group writes to a different memory location
            if constexpr (!Last) {
                // if not the final reduction, write value corresponding to
                // an index to a temporary
                vals_[out_iter_offset * n_reduction_groups +
                      reduction_batch_id] = red_val_over_wg;
            }
            out_[out_iter_offset * n_reduction_groups + reduction_batch_id] =
                idx_over_wg;
        }
    }
};

typedef sycl::event (*search_strided_impl_fn_ptr)(
    sycl::queue &,
    std::size_t,
    std::size_t,
    const char *,
    char *,
    int,
    const ssize_t *,
    ssize_t,
    ssize_t,
    int,
    const ssize_t *,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6>
class search_seq_strided_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6>
class search_seq_contig_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          bool b1,
          bool b2>
class search_over_group_krn;

template <typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          typename T6,
          typename T7,
          bool b1,
          bool b2>
class custom_search_over_group_krn;

template <typename T1, typename T2, typename T3>
class search_empty_krn;

template <typename argTy,
          typename resTy,
          typename ReductionOpT,
          typename IndexOpT,
          typename InputOutputIterIndexerT,
          typename ReductionIndexerT,
          bool First,
          bool Last>
sycl::event
    submit_search_reduction(sycl::queue &exec_q,
                            const argTy *arg,
                            argTy *arg_tmp,
                            resTy *res_tmp,
                            resTy *res,
                            argTy identity_val,
                            resTy idx_identity_val,
                            std::size_t wg,
                            std::size_t iter_nelems,
                            std::size_t reduction_nelems,
                            std::size_t reductions_per_wi,
                            std::size_t reduction_groups,
                            const InputOutputIterIndexerT &in_out_iter_indexer,
                            const ReductionIndexerT &reduction_indexer,
                            const std::vector<sycl::event> &depends)
{
    sycl::event red_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        auto globalRange = sycl::range<1>{iter_nelems * reduction_groups * wg};
        auto localRange = sycl::range<1>{wg};
        auto ndRange = sycl::nd_range<1>(globalRange, localRange);

        if constexpr (can_use_reduce_over_group<ReductionOpT, resTy>::value) {
            using KernelName =
                class search_over_group_krn<argTy, resTy, ReductionOpT,
                                            IndexOpT, InputOutputIterIndexerT,
                                            ReductionIndexerT, First, Last>;
            cgh.parallel_for<KernelName>(
                ndRange, SearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                         InputOutputIterIndexerT,
                                         ReductionIndexerT, First, Last>(
                             arg, arg_tmp, res_tmp, res, ReductionOpT(),
                             identity_val, IndexOpT(), idx_identity_val,
                             in_out_iter_indexer, reduction_indexer,
                             reduction_nelems, iter_nelems, reductions_per_wi));
        }
        else {
            using SlmT = sycl::local_accessor<argTy, 1>;
            SlmT local_memory = SlmT(localRange, cgh);
            using KernelName = class custom_search_over_group_krn<
                argTy, resTy, ReductionOpT, IndexOpT, InputOutputIterIndexerT,
                ReductionIndexerT, SlmT, First, Last>;
            cgh.parallel_for<KernelName>(
                ndRange,
                CustomSearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                      InputOutputIterIndexerT,
                                      ReductionIndexerT, SlmT, First, Last>(
                    arg, arg_tmp, res_tmp, res, ReductionOpT(), identity_val,
                    IndexOpT(), idx_identity_val, in_out_iter_indexer,
                    reduction_indexer, local_memory, reduction_nelems,
                    iter_nelems, reductions_per_wi));
        }
    });
    return red_ev;
}

template <typename argTy,
          typename resTy,
          typename ReductionOpT,
          typename IndexOpT>
sycl::event search_over_group_temps_strided_impl(
    sycl::queue &exec_q,
    std::size_t iter_nelems,      // number of reductions    (num. of rows in a
                                  // matrix when reducing over rows)
    std::size_t reduction_nelems, // size of each reduction  (length of rows,
                                  // i.e. number of columns)
    const char *arg_cp,
    char *res_cp,
    int iter_nd,
    const ssize_t *iter_shape_and_strides,
    ssize_t iter_arg_offset,
    ssize_t iter_res_offset,
    int red_nd,
    const ssize_t *reduction_shape_stride,
    ssize_t reduction_arg_offset,
    const std::vector<sycl::event> &depends)
{
    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp);
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp);

    static constexpr argTy identity_val =
        su_ns::Identity<ReductionOpT, argTy>::value;
    static constexpr resTy idx_identity_val =
        su_ns::Identity<IndexOpT, resTy>::value;

    if (reduction_nelems == 0) {
        sycl::event res_init_ev = exec_q.submit([&](sycl::handler &cgh) {
            using IndexerT =
                dpctl::tensor::offset_utils::UnpackedStridedIndexer;

            const ssize_t *const &res_shape = iter_shape_and_strides;
            const ssize_t *const &res_strides =
                iter_shape_and_strides + 2 * iter_nd;
            const IndexerT res_indexer(iter_nd, iter_res_offset, res_shape,
                                       res_strides);
            using InitKernelName =
                class search_empty_krn<resTy, argTy, ReductionOpT>;
            cgh.depends_on(depends);

            cgh.parallel_for<InitKernelName>(
                sycl::range<1>(iter_nelems), [=](sycl::id<1> id) {
                    auto res_offset = res_indexer(id[0]);
                    res_tp[res_offset] = idx_identity_val;
                });
        });

        return res_init_ev;
    }

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    std::size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    if (reduction_nelems < wg) {
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;
        using ReductionIndexerT = dpctl::tensor::offset_utils::StridedIndexer;

        const InputOutputIterIndexerT in_out_iter_indexer{
            iter_nd, iter_arg_offset, iter_res_offset, iter_shape_and_strides};
        const ReductionIndexerT reduction_indexer{red_nd, reduction_arg_offset,
                                                  reduction_shape_stride};

        sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            cgh.parallel_for<class search_seq_strided_krn<
                argTy, resTy, ReductionOpT, IndexOpT, InputOutputIterIndexerT,
                ReductionIndexerT>>(
                sycl::range<1>(iter_nelems),
                SequentialSearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                          InputOutputIterIndexerT,
                                          ReductionIndexerT>(
                    arg_tp, res_tp, ReductionOpT(), identity_val, IndexOpT(),
                    idx_identity_val, in_out_iter_indexer, reduction_indexer,
                    reduction_nelems));
        });

        return comp_ev;
    }

    static constexpr std::size_t preferred_reductions_per_wi = 4;
    // prevents running out of resources on CPU
    std::size_t max_wg = reduction_detail::get_work_group_size(d);

    std::size_t reductions_per_wi(preferred_reductions_per_wi);
    if (reduction_nelems <= preferred_reductions_per_wi * max_wg) {
        // Perform reduction using one 1 work-group per iteration,
        // can output directly to res

        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;
        using ReductionIndexerT = dpctl::tensor::offset_utils::StridedIndexer;

        const InputOutputIterIndexerT in_out_iter_indexer{
            iter_nd, iter_arg_offset, iter_res_offset, iter_shape_and_strides};
        const ReductionIndexerT reduction_indexer{red_nd, reduction_arg_offset,
                                                  reduction_shape_stride};

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

        sycl::event comp_ev =
            submit_search_reduction<argTy, resTy, ReductionOpT, IndexOpT,
                                    InputOutputIterIndexerT, ReductionIndexerT,
                                    true, true>(
                exec_q, arg_tp, nullptr, nullptr, res_tp, identity_val,
                idx_identity_val, wg, iter_nelems, reduction_nelems,
                reductions_per_wi, reduction_groups, in_out_iter_indexer,
                reduction_indexer, depends);

        return comp_ev;
    }
    else {
        // more than one work-groups is needed, requires a temporary
        std::size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);
        assert(reduction_groups > 1);

        std::size_t second_iter_reduction_groups_ =
            (reduction_groups + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);

        const std::size_t tmp_alloc_size =
            iter_nelems * (reduction_groups + second_iter_reduction_groups_);
        auto tmp_owner = dpctl::tensor::alloc_utils::smart_malloc_device<resTy>(
            tmp_alloc_size, exec_q);

        resTy *partially_reduced_tmp = tmp_owner.get();
        resTy *partially_reduced_tmp2 =
            partially_reduced_tmp + reduction_groups * iter_nelems;

        auto val_tmp_owner =
            dpctl::tensor::alloc_utils::smart_malloc_device<argTy>(
                tmp_alloc_size, exec_q);

        argTy *partially_reduced_vals_tmp = val_tmp_owner.get();
        argTy *partially_reduced_vals_tmp2 =
            partially_reduced_vals_tmp + reduction_groups * iter_nelems;

        sycl::event first_reduction_ev;
        {
            using InputIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
            using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    InputIndexerT, ResIndexerT>;
            using ReductionIndexerT =
                dpctl::tensor::offset_utils::StridedIndexer;

            // Only 2*iter_nd entries describing shape and strides of iterated
            // dimensions of input array from iter_shape_and_strides are going
            // to be accessed by inp_indexer
            const InputIndexerT inp_indexer(iter_nd, iter_arg_offset,
                                            iter_shape_and_strides);
            static constexpr ResIndexerT noop_tmp_indexer{};

            const InputOutputIterIndexerT in_out_iter_indexer{inp_indexer,
                                                              noop_tmp_indexer};
            const ReductionIndexerT reduction_indexer{
                red_nd, reduction_arg_offset, reduction_shape_stride};

            first_reduction_ev =
                submit_search_reduction<argTy, resTy, ReductionOpT, IndexOpT,
                                        InputOutputIterIndexerT,
                                        ReductionIndexerT, true, false>(
                    exec_q, arg_tp, partially_reduced_vals_tmp, nullptr,
                    partially_reduced_tmp, identity_val, idx_identity_val, wg,
                    iter_nelems, reduction_nelems, reductions_per_wi,
                    reduction_groups, in_out_iter_indexer, reduction_indexer,
                    depends);
        }

        std::size_t remaining_reduction_nelems = reduction_groups;

        resTy *temp_arg = partially_reduced_tmp;
        resTy *temp2_arg = partially_reduced_tmp2;

        argTy *vals_temp_arg = partially_reduced_vals_tmp;
        argTy *vals_temp2_arg = partially_reduced_vals_tmp2;

        sycl::event dependent_ev = first_reduction_ev;

        while (remaining_reduction_nelems >
               preferred_reductions_per_wi * max_wg) {
            std::size_t reduction_groups_ =
                (remaining_reduction_nelems + preferred_reductions_per_wi * wg -
                 1) /
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
                submit_search_reduction<argTy, resTy, ReductionOpT, IndexOpT,
                                        InputOutputIterIndexerT,
                                        ReductionIndexerT, false, false>(
                    exec_q, vals_temp_arg, vals_temp2_arg, temp_arg, temp2_arg,
                    identity_val, idx_identity_val, wg, iter_nelems,
                    remaining_reduction_nelems, preferred_reductions_per_wi,
                    reduction_groups_, in_out_iter_indexer, reduction_indexer,
                    {dependent_ev});

            remaining_reduction_nelems = reduction_groups_;
            std::swap(temp_arg, temp2_arg);
            std::swap(vals_temp_arg, vals_temp2_arg);
            dependent_ev = partial_reduction_ev;
        }

        // final reduction to res
        using InputIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
        using ResIndexerT = dpctl::tensor::offset_utils::UnpackedStridedIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                InputIndexerT, ResIndexerT>;
        using ReductionIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;

        const InputIndexerT inp_indexer{/* size */ iter_nelems,
                                        /* step */ remaining_reduction_nelems};
        const ResIndexerT res_iter_indexer{
            iter_nd, iter_res_offset,
            /* shape */ iter_shape_and_strides,
            /* strides */ iter_shape_and_strides + 2 * iter_nd};

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
            submit_search_reduction<argTy, resTy, ReductionOpT, IndexOpT,
                                    InputOutputIterIndexerT, ReductionIndexerT,
                                    false, true>(
                exec_q, vals_temp_arg, nullptr, temp_arg, res_tp, identity_val,
                idx_identity_val, wg, iter_nelems, remaining_reduction_nelems,
                reductions_per_wi, reduction_groups, in_out_iter_indexer,
                reduction_indexer, {dependent_ev});

        sycl::event cleanup_host_task_event =
            dpctl::tensor::alloc_utils::async_smart_free(
                exec_q, {final_reduction_ev}, tmp_owner, val_tmp_owner);

        // FIXME: do not return host-task event
        //   Instead collect all host-tasks to a list

        return cleanup_host_task_event;
    }
}

typedef sycl::event (*search_contig_impl_fn_ptr)(
    sycl::queue &,
    std::size_t,
    std::size_t,
    const char *,
    char *,
    ssize_t,
    ssize_t,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename argTy,
          typename resTy,
          typename ReductionOpT,
          typename IndexOpT>
sycl::event search_axis1_over_group_temps_contig_impl(
    sycl::queue &exec_q,
    std::size_t iter_nelems,      // number of reductions    (num. of rows in a
                                  // matrix when reducing over rows)
    std::size_t reduction_nelems, // size of each reduction  (length of rows,
                                  // i.e. number of columns)
    const char *arg_cp,
    char *res_cp,
    ssize_t iter_arg_offset,
    ssize_t iter_res_offset,
    ssize_t reduction_arg_offset,
    const std::vector<sycl::event> &depends)
{
    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp) +
                          iter_arg_offset + reduction_arg_offset;
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp) + iter_res_offset;

    static constexpr argTy identity_val =
        su_ns::Identity<ReductionOpT, argTy>::value;
    static constexpr resTy idx_identity_val =
        su_ns::Identity<IndexOpT, resTy>::value;

    if (reduction_nelems == 0) {
        sycl::event res_init_ev = exec_q.fill<resTy>(
            res_tp, resTy(idx_identity_val), iter_nelems, depends);

        return res_init_ev;
    }

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    std::size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    if (reduction_nelems < wg) {
        using InputIterIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
        using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                InputIterIndexerT, NoOpIndexerT>;
        using ReductionIndexerT = NoOpIndexerT;

        const InputOutputIterIndexerT in_out_iter_indexer{
            InputIterIndexerT{/* size */ iter_nelems,
                              /* step */ reduction_nelems},
            NoOpIndexerT{}};
        static constexpr ReductionIndexerT reduction_indexer{};

        sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            cgh.parallel_for<class search_seq_contig_krn<
                argTy, resTy, ReductionOpT, IndexOpT, InputOutputIterIndexerT,
                ReductionIndexerT>>(
                sycl::range<1>(iter_nelems),
                SequentialSearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                          InputOutputIterIndexerT,
                                          ReductionIndexerT>(
                    arg_tp, res_tp, ReductionOpT(), identity_val, IndexOpT(),
                    idx_identity_val, in_out_iter_indexer, reduction_indexer,
                    reduction_nelems));
        });

        return comp_ev;
    }

    static constexpr std::size_t preferred_reductions_per_wi = 8;
    // prevents running out of resources on CPU
    std::size_t max_wg = reduction_detail::get_work_group_size(d);

    std::size_t reductions_per_wi(preferred_reductions_per_wi);
    if (reduction_nelems <= preferred_reductions_per_wi * max_wg) {
        // Perform reduction using one 1 work-group per iteration,
        // can output directly to res
        using InputIterIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
        using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                InputIterIndexerT, NoOpIndexerT>;
        using ReductionIndexerT = NoOpIndexerT;

        const InputOutputIterIndexerT in_out_iter_indexer{
            InputIterIndexerT{/* size */ iter_nelems,
                              /* step */ reduction_nelems},
            NoOpIndexerT{}};
        static constexpr ReductionIndexerT reduction_indexer{};

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

        sycl::event comp_ev =
            submit_search_reduction<argTy, resTy, ReductionOpT, IndexOpT,
                                    InputOutputIterIndexerT, ReductionIndexerT,
                                    true, true>(
                exec_q, arg_tp, nullptr, nullptr, res_tp, identity_val,
                idx_identity_val, wg, iter_nelems, reduction_nelems,
                reductions_per_wi, reduction_groups, in_out_iter_indexer,
                reduction_indexer, depends);

        return comp_ev;
    }
    else {
        // more than one work-groups is needed, requires a temporary
        std::size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);
        assert(reduction_groups > 1);

        std::size_t second_iter_reduction_groups_ =
            (reduction_groups + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);

        const std::size_t tmp_alloc_size =
            iter_nelems * (reduction_groups + second_iter_reduction_groups_);
        auto tmp_owner = dpctl::tensor::alloc_utils::smart_malloc_device<resTy>(
            tmp_alloc_size, exec_q);
        resTy *partially_reduced_tmp = tmp_owner.get();
        resTy *partially_reduced_tmp2 =
            partially_reduced_tmp + reduction_groups * iter_nelems;

        auto val_tmp_owner =
            dpctl::tensor::alloc_utils::smart_malloc_device<argTy>(
                tmp_alloc_size, exec_q);
        argTy *partially_reduced_vals_tmp = val_tmp_owner.get();
        argTy *partially_reduced_vals_tmp2 =
            partially_reduced_vals_tmp + reduction_groups * iter_nelems;

        sycl::event first_reduction_ev;
        {
            using InputIterIndexerT =
                dpctl::tensor::offset_utils::Strided1DIndexer;
            using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    InputIterIndexerT, NoOpIndexerT>;
            using ReductionIndexerT = NoOpIndexerT;

            const InputOutputIterIndexerT in_out_iter_indexer{
                InputIterIndexerT{/* size */ iter_nelems,
                                  /* step */ reduction_nelems},
                NoOpIndexerT{}};
            static constexpr ReductionIndexerT reduction_indexer{};

            first_reduction_ev =
                submit_search_reduction<argTy, resTy, ReductionOpT, IndexOpT,
                                        InputOutputIterIndexerT,
                                        ReductionIndexerT, true, false>(
                    exec_q, arg_tp, partially_reduced_vals_tmp, nullptr,
                    partially_reduced_tmp, identity_val, idx_identity_val, wg,
                    iter_nelems, reduction_nelems, preferred_reductions_per_wi,
                    reduction_groups, in_out_iter_indexer, reduction_indexer,
                    depends);
        }

        std::size_t remaining_reduction_nelems = reduction_groups;

        resTy *temp_arg = partially_reduced_tmp;
        resTy *temp2_arg = partially_reduced_tmp2;

        argTy *vals_temp_arg = partially_reduced_vals_tmp;
        argTy *vals_temp2_arg = partially_reduced_vals_tmp2;

        sycl::event dependent_ev = first_reduction_ev;

        while (remaining_reduction_nelems >
               preferred_reductions_per_wi * max_wg) {
            std::size_t reduction_groups_ =
                (remaining_reduction_nelems + preferred_reductions_per_wi * wg -
                 1) /
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
                submit_search_reduction<argTy, resTy, ReductionOpT, IndexOpT,
                                        InputOutputIterIndexerT,
                                        ReductionIndexerT, false, false>(
                    exec_q, vals_temp_arg, vals_temp2_arg, temp_arg, temp2_arg,
                    identity_val, idx_identity_val, wg, iter_nelems,
                    remaining_reduction_nelems, preferred_reductions_per_wi,
                    reduction_groups_, in_out_iter_indexer, reduction_indexer,
                    {dependent_ev});

            remaining_reduction_nelems = reduction_groups_;
            std::swap(temp_arg, temp2_arg);
            std::swap(vals_temp_arg, vals_temp2_arg);
            dependent_ev = partial_reduction_ev;
        }

        // final reduction to res
        using InputIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
        using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                InputIndexerT, ResIndexerT>;
        using ReductionIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;

        const InputIndexerT inp_indexer{/* size */ iter_nelems,
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
            submit_search_reduction<argTy, resTy, ReductionOpT, IndexOpT,
                                    InputOutputIterIndexerT, ReductionIndexerT,
                                    false, true>(
                exec_q, vals_temp_arg, nullptr, temp_arg, res_tp, identity_val,
                idx_identity_val, wg, iter_nelems, remaining_reduction_nelems,
                reductions_per_wi, reduction_groups, in_out_iter_indexer,
                reduction_indexer, {dependent_ev});

        sycl::event cleanup_host_task_event =
            dpctl::tensor::alloc_utils::async_smart_free(
                exec_q, {final_reduction_ev}, tmp_owner, val_tmp_owner);

        // FIXME: do not return host-task event
        //   Instead collect all host-tasks to a list

        return cleanup_host_task_event;
    }
}

template <typename argTy,
          typename resTy,
          typename ReductionOpT,
          typename IndexOpT>
sycl::event search_axis0_over_group_temps_contig_impl(
    sycl::queue &exec_q,
    std::size_t iter_nelems,      // number of reductions    (num. of rows in a
                                  // matrix when reducing over rows)
    std::size_t reduction_nelems, // size of each reduction  (length of rows,
                                  // i.e. number of columns)
    const char *arg_cp,
    char *res_cp,
    ssize_t iter_arg_offset,
    ssize_t iter_res_offset,
    ssize_t reduction_arg_offset,
    const std::vector<sycl::event> &depends)
{
    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_cp) +
                          iter_arg_offset + reduction_arg_offset;
    resTy *res_tp = reinterpret_cast<resTy *>(res_cp) + iter_res_offset;

    static constexpr argTy identity_val =
        su_ns::Identity<ReductionOpT, argTy>::value;
    static constexpr resTy idx_identity_val =
        su_ns::Identity<IndexOpT, resTy>::value;

    if (reduction_nelems == 0) {
        sycl::event res_init_ev = exec_q.fill<resTy>(
            res_tp, resTy(idx_identity_val), iter_nelems, depends);

        return res_init_ev;
    }

    const sycl::device &d = exec_q.get_device();
    const auto &sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    std::size_t wg = choose_workgroup_size<4>(reduction_nelems, sg_sizes);

    if (reduction_nelems < wg) {
        using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                NoOpIndexerT, NoOpIndexerT>;
        using ReductionIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;

        const InputOutputIterIndexerT in_out_iter_indexer{NoOpIndexerT{},
                                                          NoOpIndexerT{}};
        const ReductionIndexerT reduction_indexer{/* size */ reduction_nelems,
                                                  /* step */ iter_nelems};

        sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            using KernelName =
                class search_seq_contig_krn<argTy, resTy, ReductionOpT,
                                            IndexOpT, InputOutputIterIndexerT,
                                            ReductionIndexerT>;

            sycl::range<1> iter_range{iter_nelems};

            cgh.parallel_for<KernelName>(
                iter_range,
                SequentialSearchReduction<argTy, resTy, ReductionOpT, IndexOpT,
                                          InputOutputIterIndexerT,
                                          ReductionIndexerT>(
                    arg_tp, res_tp, ReductionOpT(), identity_val, IndexOpT(),
                    idx_identity_val, in_out_iter_indexer, reduction_indexer,
                    reduction_nelems));
        });

        return comp_ev;
    }

    static constexpr std::size_t preferred_reductions_per_wi = 8;
    // prevents running out of resources on CPU
    std::size_t max_wg = reduction_detail::get_work_group_size(d);

    std::size_t reductions_per_wi(preferred_reductions_per_wi);
    if (reduction_nelems <= preferred_reductions_per_wi * max_wg) {
        // Perform reduction using one 1 work-group per iteration,
        // can output directly to res
        using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using ColsIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                NoOpIndexerT, NoOpIndexerT>;
        using ReductionIndexerT = ColsIndexerT;

        static constexpr NoOpIndexerT columns_indexer{};
        static constexpr NoOpIndexerT result_indexer{};
        const InputOutputIterIndexerT in_out_iter_indexer{columns_indexer,
                                                          result_indexer};
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

        sycl::event comp_ev =
            submit_search_reduction<argTy, resTy, ReductionOpT, IndexOpT,
                                    InputOutputIterIndexerT, ReductionIndexerT,
                                    true, true>(
                exec_q, arg_tp, nullptr, nullptr, res_tp, identity_val,
                idx_identity_val, wg, iter_nelems, reduction_nelems,
                reductions_per_wi, reduction_groups, in_out_iter_indexer,
                reduction_indexer, depends);

        return comp_ev;
    }
    else {
        // more than one work-groups is needed, requires a temporary
        std::size_t reduction_groups =
            (reduction_nelems + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);
        assert(reduction_groups > 1);

        std::size_t second_iter_reduction_groups_ =
            (reduction_groups + preferred_reductions_per_wi * wg - 1) /
            (preferred_reductions_per_wi * wg);

        const std::size_t tmp_alloc_size =
            iter_nelems * (reduction_groups + second_iter_reduction_groups_);
        auto tmp_owner = dpctl::tensor::alloc_utils::smart_malloc_device<resTy>(
            tmp_alloc_size, exec_q);

        resTy *partially_reduced_tmp = tmp_owner.get();
        resTy *partially_reduced_tmp2 =
            partially_reduced_tmp + reduction_groups * iter_nelems;

        auto vals_tmp_owner =
            dpctl::tensor::alloc_utils::smart_malloc_device<argTy>(
                tmp_alloc_size, exec_q);
        argTy *partially_reduced_vals_tmp = vals_tmp_owner.get();
        argTy *partially_reduced_vals_tmp2 =
            partially_reduced_vals_tmp + reduction_groups * iter_nelems;

        sycl::event first_reduction_ev;
        {
            using NoOpIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
            using ColsIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
            using InputOutputIterIndexerT =
                dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                    NoOpIndexerT, NoOpIndexerT>;
            using ReductionIndexerT = ColsIndexerT;

            static constexpr NoOpIndexerT columns_indexer{};
            static constexpr NoOpIndexerT result_indexer{};
            const InputOutputIterIndexerT in_out_iter_indexer{columns_indexer,
                                                              result_indexer};
            const ReductionIndexerT reduction_indexer{
                /* size */ reduction_nelems,
                /* step */ iter_nelems};

            first_reduction_ev =
                submit_search_reduction<argTy, resTy, ReductionOpT, IndexOpT,
                                        InputOutputIterIndexerT,
                                        ReductionIndexerT, true, false>(
                    exec_q, arg_tp, partially_reduced_vals_tmp, nullptr,
                    partially_reduced_tmp, identity_val, idx_identity_val, wg,
                    iter_nelems, reduction_nelems, preferred_reductions_per_wi,
                    reduction_groups, in_out_iter_indexer, reduction_indexer,
                    depends);
        }

        std::size_t remaining_reduction_nelems = reduction_groups;

        resTy *temp_arg = partially_reduced_tmp;
        resTy *temp2_arg = partially_reduced_tmp2;

        argTy *vals_temp_arg = partially_reduced_vals_tmp;
        argTy *vals_temp2_arg = partially_reduced_vals_tmp2;

        sycl::event dependent_ev = first_reduction_ev;

        while (remaining_reduction_nelems >
               preferred_reductions_per_wi * max_wg) {
            std::size_t reduction_groups_ =
                (remaining_reduction_nelems + preferred_reductions_per_wi * wg -
                 1) /
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
                submit_search_reduction<argTy, resTy, ReductionOpT, IndexOpT,
                                        InputOutputIterIndexerT,
                                        ReductionIndexerT, false, false>(
                    exec_q, vals_temp_arg, vals_temp2_arg, temp_arg, temp2_arg,
                    identity_val, idx_identity_val, wg, iter_nelems,
                    remaining_reduction_nelems, preferred_reductions_per_wi,
                    reduction_groups_, in_out_iter_indexer, reduction_indexer,
                    {dependent_ev});

            remaining_reduction_nelems = reduction_groups_;
            std::swap(temp_arg, temp2_arg);
            std::swap(vals_temp_arg, vals_temp2_arg);
            dependent_ev = partial_reduction_ev;
        }

        // final reduction to res
        using InputIndexerT = dpctl::tensor::offset_utils::Strided1DIndexer;
        using ResIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;
        using InputOutputIterIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<
                InputIndexerT, ResIndexerT>;
        using ReductionIndexerT = dpctl::tensor::offset_utils::NoOpIndexer;

        const InputIndexerT inp_indexer{/* size */ iter_nelems,
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
            submit_search_reduction<argTy, resTy, ReductionOpT, IndexOpT,
                                    InputOutputIterIndexerT, ReductionIndexerT,
                                    false, true>(
                exec_q, vals_temp_arg, nullptr, temp_arg, res_tp, identity_val,
                idx_identity_val, wg, iter_nelems, remaining_reduction_nelems,
                reductions_per_wi, reduction_groups, in_out_iter_indexer,
                reduction_indexer, {dependent_ev});

        sycl::event cleanup_host_task_event =
            dpctl::tensor::alloc_utils::async_smart_free(
                exec_q, {final_reduction_ev}, tmp_owner, vals_tmp_owner);

        // FIXME: do not return host-task event
        //   Instead collect all host-tasks to a list

        return cleanup_host_task_event;
    }
}

} // namespace dpctl::tensor::kernels
