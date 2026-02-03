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
///
/// \file
/// This file defines common code for elementwise tensor operations.
//===---------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>

#include "common_detail.hpp"

#include "utils/offset_utils.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/sycl_utils.hpp"

#include "kernels/alignment.hpp"
#include "kernels/dpctl_tensor_types.hpp"

namespace dpctl::tensor::kernels::elementwise_common
{
using dpctl::tensor::ssize_t;
using dpctl::tensor::kernels::alignment_utils::
    disabled_sg_loadstore_wrapper_krn;
using dpctl::tensor::kernels::alignment_utils::is_aligned;
using dpctl::tensor::kernels::alignment_utils::required_alignment;

using dpctl::tensor::sycl_utils::sub_group_load;
using dpctl::tensor::sycl_utils::sub_group_store;

/*! @brief Functor for unary function evaluation on contiguous array */
template <typename argT,
          typename resT,
          typename UnaryOperatorT,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
struct UnaryContigFunctor
{
private:
    const argT *in = nullptr;
    resT *out = nullptr;
    std::size_t nelems_;

public:
    UnaryContigFunctor(const argT *inp, resT *res, const std::size_t n_elems)
        : in(inp), out(res), nelems_(n_elems)
    {
    }

    void operator()(sycl::nd_item<1> ndit) const
    {
        static constexpr std::uint8_t elems_per_wi = n_vecs * vec_sz;
        UnaryOperatorT op{};
        /* Each work-item processes vec_sz elements, contiguous in memory */
        /* NOTE: work-group size must be divisible by sub-group size */

        if constexpr (enable_sg_loadstore && UnaryOperatorT::is_constant::value)
        {
            // value of operator is known to be a known constant
            constexpr resT const_val = UnaryOperatorT::constant_value;

            auto sg = ndit.get_sub_group();
            const std::uint16_t sgSize = sg.get_max_local_range()[0];

            const std::size_t base =
                elems_per_wi * (ndit.get_group(0) * ndit.get_local_range(0) +
                                sg.get_group_id()[0] * sgSize);
            if (base + elems_per_wi * sgSize < nelems_) {
                static constexpr sycl::vec<resT, vec_sz> res_vec(const_val);
#pragma unroll
                for (std::uint8_t it = 0; it < elems_per_wi; it += vec_sz) {
                    const std::size_t offset = base + it * sgSize;
                    auto out_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out[offset]);

                    sub_group_store<vec_sz>(sg, res_vec, out_multi_ptr);
                }
            }
            else {
                const std::size_t lane_id = sg.get_local_id()[0];
                for (std::size_t k = base + lane_id; k < nelems_; k += sgSize) {
                    out[k] = const_val;
                }
            }
        }
        else if constexpr (enable_sg_loadstore &&
                           UnaryOperatorT::supports_sg_loadstore::value &&
                           UnaryOperatorT::supports_vec::value && (vec_sz > 1))
        {
            auto sg = ndit.get_sub_group();
            const std::uint16_t sgSize = sg.get_max_local_range()[0];

            const std::size_t base =
                elems_per_wi * (ndit.get_group(0) * ndit.get_local_range(0) +
                                sg.get_group_id()[0] * sgSize);
            if (base + elems_per_wi * sgSize < nelems_) {
#pragma unroll
                for (std::uint8_t it = 0; it < elems_per_wi; it += vec_sz) {
                    const std::size_t offset = base + it * sgSize;
                    auto in_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&in[offset]);
                    auto out_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out[offset]);

                    const sycl::vec<argT, vec_sz> x =
                        sub_group_load<vec_sz>(sg, in_multi_ptr);
                    const sycl::vec<resT, vec_sz> res_vec = op(x);
                    sub_group_store<vec_sz>(sg, res_vec, out_multi_ptr);
                }
            }
            else {
                const std::size_t lane_id = sg.get_local_id()[0];
                for (std::size_t k = base + lane_id; k < nelems_; k += sgSize) {
                    // scalar call
                    out[k] = op(in[k]);
                }
            }
        }
        else if constexpr (enable_sg_loadstore &&
                           UnaryOperatorT::supports_sg_loadstore::value &&
                           std::is_same_v<resT, argT>)
        {
            // default: use scalar-value function

            auto sg = ndit.get_sub_group();
            const std::uint16_t sgSize = sg.get_max_local_range()[0];
            const std::size_t base =
                elems_per_wi * (ndit.get_group(0) * ndit.get_local_range(0) +
                                sg.get_group_id()[0] * sgSize);

            if (base + elems_per_wi * sgSize < nelems_) {
#pragma unroll
                for (std::uint8_t it = 0; it < elems_per_wi; it += vec_sz) {
                    const std::size_t offset = base + it * sgSize;
                    auto in_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&in[offset]);
                    auto out_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out[offset]);

                    sycl::vec<argT, vec_sz> arg_vec =
                        sub_group_load<vec_sz>(sg, in_multi_ptr);
#pragma unroll
                    for (std::uint32_t k = 0; k < vec_sz; ++k) {
                        arg_vec[k] = op(arg_vec[k]);
                    }
                    sub_group_store<vec_sz>(sg, arg_vec, out_multi_ptr);
                }
            }
            else {
                const std::size_t lane_id = sg.get_local_id()[0];
                for (std::size_t k = base + lane_id; k < nelems_; k += sgSize) {
                    out[k] = op(in[k]);
                }
            }
        }
        else if constexpr (enable_sg_loadstore &&
                           UnaryOperatorT::supports_sg_loadstore::value)
        {
            // default: use scalar-value function

            auto sg = ndit.get_sub_group();
            const std::uint16_t sgSize = sg.get_max_local_range()[0];
            const std::size_t base =
                elems_per_wi * (ndit.get_group(0) * ndit.get_local_range(0) +
                                sg.get_group_id()[0] * sgSize);

            if (base + elems_per_wi * sgSize < nelems_) {
#pragma unroll
                for (std::uint8_t it = 0; it < elems_per_wi; it += vec_sz) {
                    const std::size_t offset = base + it * sgSize;
                    auto in_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&in[offset]);
                    auto out_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out[offset]);

                    const sycl::vec<argT, vec_sz> arg_vec =
                        sub_group_load<vec_sz>(sg, in_multi_ptr);
                    sycl::vec<resT, vec_sz> res_vec;
#pragma unroll
                    for (std::uint8_t k = 0; k < vec_sz; ++k) {
                        res_vec[k] = op(arg_vec[k]);
                    }
                    sub_group_store<vec_sz>(sg, res_vec, out_multi_ptr);
                }
            }
            else {
                const std::size_t lane_id = sg.get_local_id()[0];
                for (std::size_t k = base + lane_id; k < nelems_; k += sgSize) {
                    out[k] = op(in[k]);
                }
            }
        }
        else {
            const std::uint16_t sgSize =
                ndit.get_sub_group().get_local_range()[0];
            const std::size_t gid = ndit.get_global_linear_id();
            const std::uint16_t elems_per_sg = sgSize * elems_per_wi;

            const std::size_t start =
                (gid / sgSize) * (elems_per_sg - sgSize) + gid;
            const std::size_t end = std::min(nelems_, start + elems_per_sg);
            for (std::size_t offset = start; offset < end; offset += sgSize) {
                out[offset] = op(in[offset]);
            }
        }
    }
};

template <typename argT, typename resT, typename IndexerT, typename UnaryOpT>
struct UnaryStridedFunctor
{
private:
    const argT *inp_ = nullptr;
    resT *res_ = nullptr;
    IndexerT inp_out_indexer_;

public:
    UnaryStridedFunctor(const argT *inp_p,
                        resT *res_p,
                        const IndexerT &inp_out_indexer)
        : inp_(inp_p), res_(res_p), inp_out_indexer_(inp_out_indexer)
    {
    }

    void operator()(sycl::id<1> wid) const
    {
        const auto &offsets_ = inp_out_indexer_(wid.get(0));
        const ssize_t &inp_offset = offsets_.get_first_offset();
        const ssize_t &res_offset = offsets_.get_second_offset();

        UnaryOpT op{};

        res_[res_offset] = op(inp_[inp_offset]);
    }
};

template <typename SizeT>
SizeT select_lws(const sycl::device &, SizeT n_work_items_needed)
{
    // TODO: make the decision based on device descriptors

    // constexpr SizeT few_threshold = (SizeT(1) << 17);
    static constexpr SizeT med_threshold = (SizeT(1) << 21);

    const SizeT lws =
        (n_work_items_needed <= med_threshold ? SizeT(128) : SizeT(256));

    return lws;
}

template <typename argTy,
          template <typename T>
          class UnaryOutputType,
          template <typename A,
                    typename R,
                    std::uint8_t vs,
                    std::uint8_t nv,
                    bool enable>
          class ContigFunctorT,
          template <typename A, typename R, std::uint8_t vs, std::uint8_t nv>
          class kernel_name,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u>
sycl::event unary_contig_impl(sycl::queue &exec_q,
                              std::size_t nelems,
                              const char *arg_p,
                              char *res_p,
                              const std::vector<sycl::event> &depends = {})
{
    static constexpr std::uint8_t elems_per_wi = n_vecs * vec_sz;
    const std::size_t n_work_items_needed = nelems / elems_per_wi;
    const std::size_t lws =
        select_lws(exec_q.get_device(), n_work_items_needed);

    const std::size_t n_groups =
        ((nelems + lws * elems_per_wi - 1) / (lws * elems_per_wi));
    const auto gws_range = sycl::range<1>(n_groups * lws);
    const auto lws_range = sycl::range<1>(lws);

    using resTy = typename UnaryOutputType<argTy>::value_type;
    using BaseKernelName = kernel_name<argTy, resTy, vec_sz, n_vecs>;

    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_p);
    resTy *res_tp = reinterpret_cast<resTy *>(res_p);

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        if (is_aligned<required_alignment>(arg_p) &&
            is_aligned<required_alignment>(res_p))
        {
            static constexpr bool enable_sg_loadstore = true;
            using KernelName = BaseKernelName;
            using Impl = ContigFunctorT<argTy, resTy, vec_sz, n_vecs,
                                        enable_sg_loadstore>;

            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(gws_range, lws_range),
                Impl(arg_tp, res_tp, nelems));
        }
        else {
            static constexpr bool disable_sg_loadstore = false;
            using KernelName =
                disabled_sg_loadstore_wrapper_krn<BaseKernelName>;
            using Impl = ContigFunctorT<argTy, resTy, vec_sz, n_vecs,
                                        disable_sg_loadstore>;

            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(gws_range, lws_range),
                Impl(arg_tp, res_tp, nelems));
        }
    });

    return comp_ev;
}

template <typename argTy,
          template <typename T>
          class UnaryOutputType,
          template <typename A, typename R, typename I>
          class StridedFunctorT,
          template <typename A, typename R, typename I>
          class kernel_name>
sycl::event
    unary_strided_impl(sycl::queue &exec_q,
                       std::size_t nelems,
                       int nd,
                       const ssize_t *shape_and_strides,
                       const char *arg_p,
                       ssize_t arg_offset,
                       char *res_p,
                       ssize_t res_offset,
                       const std::vector<sycl::event> &depends,
                       const std::vector<sycl::event> &additional_depends)
{
    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.depends_on(additional_depends);

        using resTy = typename UnaryOutputType<argTy>::value_type;
        using IndexerT =
            typename dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;

        const IndexerT indexer{nd, arg_offset, res_offset, shape_and_strides};

        const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_p);
        resTy *res_tp = reinterpret_cast<resTy *>(res_p);

        using Impl = StridedFunctorT<argTy, resTy, IndexerT>;

        cgh.parallel_for<kernel_name<argTy, resTy, IndexerT>>(
            {nelems}, Impl(arg_tp, res_tp, indexer));
    });
    return comp_ev;
}

template <typename argT1,
          typename argT2,
          typename resT,
          typename BinaryOperatorT,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
struct BinaryContigFunctor
{
private:
    const argT1 *in1 = nullptr;
    const argT2 *in2 = nullptr;
    resT *out = nullptr;
    std::size_t nelems_;

public:
    BinaryContigFunctor(const argT1 *inp1,
                        const argT2 *inp2,
                        resT *res,
                        const std::size_t n_elems)
        : in1(inp1), in2(inp2), out(res), nelems_(n_elems)
    {
    }

    void operator()(sycl::nd_item<1> ndit) const
    {
        static constexpr std::uint8_t elems_per_wi = n_vecs * vec_sz;
        BinaryOperatorT op{};
        /* Each work-item processes vec_sz elements, contiguous in memory */
        /* NOTE: work-group size must be divisible by sub-group size */

        if constexpr (enable_sg_loadstore &&
                      BinaryOperatorT::supports_sg_loadstore::value &&
                      BinaryOperatorT::supports_vec::value && (vec_sz > 1))
        {
            auto sg = ndit.get_sub_group();
            std::uint16_t sgSize = sg.get_max_local_range()[0];

            const std::size_t base =
                elems_per_wi * (ndit.get_group(0) * ndit.get_local_range(0) +
                                sg.get_group_id()[0] * sgSize);

            if (base + elems_per_wi * sgSize < nelems_) {
                sycl::vec<resT, vec_sz> res_vec;

#pragma unroll
                for (std::uint8_t it = 0; it < elems_per_wi; it += vec_sz) {
                    std::size_t offset = base + it * sgSize;
                    auto in1_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&in1[offset]);
                    auto in2_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&in2[offset]);
                    auto out_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out[offset]);

                    const sycl::vec<argT1, vec_sz> arg1_vec =
                        sub_group_load<vec_sz>(sg, in1_multi_ptr);
                    const sycl::vec<argT2, vec_sz> arg2_vec =
                        sub_group_load<vec_sz>(sg, in2_multi_ptr);
                    res_vec = op(arg1_vec, arg2_vec);
                    sub_group_store<vec_sz>(sg, res_vec, out_multi_ptr);
                }
            }
            else {
                const std::size_t lane_id = sg.get_local_id()[0];
                for (std::size_t k = base + lane_id; k < nelems_; k += sgSize) {
                    out[k] = op(in1[k], in2[k]);
                }
            }
        }
        else if constexpr (enable_sg_loadstore &&
                           BinaryOperatorT::supports_sg_loadstore::value)
        {
            auto sg = ndit.get_sub_group();
            const std::uint16_t sgSize = sg.get_max_local_range()[0];

            const std::size_t base =
                elems_per_wi * (ndit.get_group(0) * ndit.get_local_range(0) +
                                sg.get_group_id()[0] * sgSize);

            if (base + elems_per_wi * sgSize < nelems_) {
#pragma unroll
                for (std::uint8_t it = 0; it < elems_per_wi; it += vec_sz) {
                    const std::size_t offset = base + it * sgSize;
                    auto in1_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&in1[offset]);
                    auto in2_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&in2[offset]);
                    auto out_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out[offset]);

                    const sycl::vec<argT1, vec_sz> arg1_vec =
                        sub_group_load<vec_sz>(sg, in1_multi_ptr);
                    const sycl::vec<argT2, vec_sz> arg2_vec =
                        sub_group_load<vec_sz>(sg, in2_multi_ptr);

                    sycl::vec<resT, vec_sz> res_vec;
#pragma unroll
                    for (std::uint8_t vec_id = 0; vec_id < vec_sz; ++vec_id) {
                        res_vec[vec_id] =
                            op(arg1_vec[vec_id], arg2_vec[vec_id]);
                    }
                    sub_group_store<vec_sz>(sg, res_vec, out_multi_ptr);
                }
            }
            else {
                const std::size_t lane_id = sg.get_local_id()[0];
                for (std::size_t k = base + lane_id; k < nelems_; k += sgSize) {
                    out[k] = op(in1[k], in2[k]);
                }
            }
        }
        else {
            const std::size_t sgSize =
                ndit.get_sub_group().get_local_range()[0];
            const std::size_t gid = ndit.get_global_linear_id();
            const std::size_t elems_per_sg = sgSize * elems_per_wi;

            const std::size_t start =
                (gid / sgSize) * (elems_per_sg - sgSize) + gid;
            const std::size_t end = std::min(nelems_, start + elems_per_sg);
            for (std::size_t offset = start; offset < end; offset += sgSize) {
                out[offset] = op(in1[offset], in2[offset]);
            }
        }
    }
};

template <typename argT1,
          typename argT2,
          typename resT,
          typename ThreeOffsets_IndexerT,
          typename BinaryOperatorT>
struct BinaryStridedFunctor
{
private:
    const argT1 *in1 = nullptr;
    const argT2 *in2 = nullptr;
    resT *out = nullptr;
    ThreeOffsets_IndexerT three_offsets_indexer_;

public:
    BinaryStridedFunctor(const argT1 *inp1_tp,
                         const argT2 *inp2_tp,
                         resT *res_tp,
                         const ThreeOffsets_IndexerT &inps_res_indexer)
        : in1(inp1_tp), in2(inp2_tp), out(res_tp),
          three_offsets_indexer_(inps_res_indexer)
    {
    }

    void operator()(sycl::id<1> wid) const
    {
        const auto &three_offsets_ =
            three_offsets_indexer_(static_cast<ssize_t>(wid.get(0)));

        const auto &inp1_offset = three_offsets_.get_first_offset();
        const auto &inp2_offset = three_offsets_.get_second_offset();
        const auto &out_offset = three_offsets_.get_third_offset();

        BinaryOperatorT op{};
        out[out_offset] = op(in1[inp1_offset], in2[inp2_offset]);
    }
};

template <typename argT1,
          typename argT2,
          typename resT,
          typename BinaryOperatorT>
struct BinaryContigMatrixContigRowBroadcastingFunctor
{
private:
    const argT1 *mat;
    const argT2 *padded_vec;
    resT *res;
    std::size_t n_elems;
    std::size_t n1;

public:
    BinaryContigMatrixContigRowBroadcastingFunctor(const argT1 *mat_tp,
                                                   const argT2 *row_tp,
                                                   resT *res_tp,
                                                   std::size_t n_elems_in_mat,
                                                   std::size_t n_elems_in_row)
        : mat(mat_tp), padded_vec(row_tp), res(res_tp), n_elems(n_elems_in_mat),
          n1(n_elems_in_row)
    {
    }

    void operator()(sycl::nd_item<1> ndit) const
    {
        /* NOTE: work-group size must be divisible by sub-group size */

        BinaryOperatorT op{};
        static_assert(BinaryOperatorT::supports_sg_loadstore::value);

        const auto &sg = ndit.get_sub_group();
        const std::size_t gid = ndit.get_global_linear_id();

        const std::size_t sgSize = sg.get_max_local_range()[0];
        const std::size_t base = gid - sg.get_local_id()[0];

        if (base + sgSize < n_elems) {
            auto in1_multi_ptr = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::yes>(&mat[base]);

            auto in2_multi_ptr = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::yes>(&padded_vec[base % n1]);

            auto out_multi_ptr = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::yes>(&res[base]);

            const argT1 mat_el = sub_group_load(sg, in1_multi_ptr);
            const argT2 vec_el = sub_group_load(sg, in2_multi_ptr);

            resT res_el = op(mat_el, vec_el);

            sub_group_store(sg, res_el, out_multi_ptr);
        }
        else {
            const std::size_t lane_id = sg.get_local_id()[0];
            for (std::size_t k = base + lane_id; k < n_elems; k += sgSize) {
                res[k] = op(mat[k], padded_vec[k % n1]);
            }
        }
    }
};

template <typename argT1,
          typename argT2,
          typename resT,
          typename BinaryOperatorT>
struct BinaryContigRowContigMatrixBroadcastingFunctor
{
private:
    const argT1 *padded_vec;
    const argT2 *mat;
    resT *res;
    std::size_t n_elems;
    std::size_t n1;

public:
    BinaryContigRowContigMatrixBroadcastingFunctor(const argT1 *row_tp,
                                                   const argT2 *mat_tp,
                                                   resT *res_tp,
                                                   std::size_t n_elems_in_mat,
                                                   std::size_t n_elems_in_row)
        : padded_vec(row_tp), mat(mat_tp), res(res_tp), n_elems(n_elems_in_mat),
          n1(n_elems_in_row)
    {
    }

    void operator()(sycl::nd_item<1> ndit) const
    {
        /* NOTE: work-group size must be divisible by sub-group size */
        BinaryOperatorT op{};
        static_assert(BinaryOperatorT::supports_sg_loadstore::value);

        const auto &sg = ndit.get_sub_group();
        std::size_t gid = ndit.get_global_linear_id();

        const std::size_t sgSize = sg.get_max_local_range()[0];
        const std::size_t base = gid - sg.get_local_id()[0];

        if (base + sgSize < n_elems) {
            auto in1_multi_ptr = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::yes>(&padded_vec[base % n1]);

            auto in2_multi_ptr = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::yes>(&mat[base]);

            auto out_multi_ptr = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::yes>(&res[base]);

            const argT2 mat_el = sub_group_load(sg, in2_multi_ptr);
            const argT1 vec_el = sub_group_load(sg, in1_multi_ptr);

            resT res_el = op(vec_el, mat_el);

            sub_group_store(sg, res_el, out_multi_ptr);
        }
        else {
            const std::size_t lane_id = sg.get_local_id()[0];
            for (std::size_t k = base + lane_id; k < n_elems; k += sgSize) {
                res[k] = op(padded_vec[k % n1], mat[k]);
            }
        }
    }
};

// Typedefs for function pointers

typedef sycl::event (*unary_contig_impl_fn_ptr_t)(
    sycl::queue &,
    std::size_t,
    const char *,
    char *,
    const std::vector<sycl::event> &);

typedef sycl::event (*unary_strided_impl_fn_ptr_t)(
    sycl::queue &,
    std::size_t,
    int,
    const ssize_t *,
    const char *,
    ssize_t,
    char *,
    ssize_t,
    const std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

typedef sycl::event (*binary_contig_impl_fn_ptr_t)(
    sycl::queue &,
    std::size_t,
    const char *,
    ssize_t,
    const char *,
    ssize_t,
    char *,
    ssize_t,
    const std::vector<sycl::event> &);

typedef sycl::event (*binary_strided_impl_fn_ptr_t)(
    sycl::queue &,
    std::size_t,
    int,
    const ssize_t *,
    const char *,
    ssize_t,
    const char *,
    ssize_t,
    char *,
    ssize_t,
    const std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

typedef sycl::event (*binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t)(
    sycl::queue &,
    std::vector<sycl::event> &,
    std::size_t,
    std::size_t,
    const char *,
    ssize_t,
    const char *,
    ssize_t,
    char *,
    ssize_t,
    const std::vector<sycl::event> &);

typedef sycl::event (*binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t)(
    sycl::queue &,
    std::vector<sycl::event> &,
    std::size_t,
    std::size_t,
    const char *,
    ssize_t,
    const char *,
    ssize_t,
    char *,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename argTy1,
          typename argTy2,
          template <typename T1, typename T2>
          class BinaryOutputType,
          template <typename T1,
                    typename T2,
                    typename T3,
                    std::uint8_t vs,
                    std::uint8_t nv,
                    bool enable_sg_loadstore>
          class BinaryContigFunctorT,
          template <typename T1,
                    typename T2,
                    typename T3,
                    std::uint8_t vs,
                    std::uint8_t nv>
          class kernel_name,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u>
sycl::event binary_contig_impl(sycl::queue &exec_q,
                               std::size_t nelems,
                               const char *arg1_p,
                               ssize_t arg1_offset,
                               const char *arg2_p,
                               ssize_t arg2_offset,
                               char *res_p,
                               ssize_t res_offset,
                               const std::vector<sycl::event> &depends = {})
{
    const std::size_t n_work_items_needed = nelems / (n_vecs * vec_sz);
    const std::size_t lws =
        select_lws(exec_q.get_device(), n_work_items_needed);

    const std::size_t n_groups =
        ((nelems + lws * n_vecs * vec_sz - 1) / (lws * n_vecs * vec_sz));
    const auto gws_range = sycl::range<1>(n_groups * lws);
    const auto lws_range = sycl::range<1>(lws);

    using resTy = typename BinaryOutputType<argTy1, argTy2>::value_type;
    using BaseKernelName = kernel_name<argTy1, argTy2, resTy, vec_sz, n_vecs>;

    const argTy1 *arg1_tp =
        reinterpret_cast<const argTy1 *>(arg1_p) + arg1_offset;
    const argTy2 *arg2_tp =
        reinterpret_cast<const argTy2 *>(arg2_p) + arg2_offset;
    resTy *res_tp = reinterpret_cast<resTy *>(res_p) + res_offset;

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        if (is_aligned<required_alignment>(arg1_tp) &&
            is_aligned<required_alignment>(arg2_tp) &&
            is_aligned<required_alignment>(res_tp))
        {
            static constexpr bool enable_sg_loadstore = true;
            using KernelName = BaseKernelName;
            using Impl = BinaryContigFunctorT<argTy1, argTy2, resTy, vec_sz,
                                              n_vecs, enable_sg_loadstore>;

            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(gws_range, lws_range),
                Impl(arg1_tp, arg2_tp, res_tp, nelems));
        }
        else {
            static constexpr bool disable_sg_loadstore = false;
            using KernelName =
                disabled_sg_loadstore_wrapper_krn<BaseKernelName>;
            using Impl = BinaryContigFunctorT<argTy1, argTy2, resTy, vec_sz,
                                              n_vecs, disable_sg_loadstore>;

            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(gws_range, lws_range),
                Impl(arg1_tp, arg2_tp, res_tp, nelems));
        }
    });
    return comp_ev;
}

template <typename argTy1,
          typename argTy2,
          template <typename T1, typename T2>
          class BinaryOutputType,
          template <typename T1, typename T2, typename T3, typename IndT>
          class BinaryStridedFunctorT,
          template <typename T1, typename T2, typename T3, typename IndT>
          class kernel_name>
sycl::event
    binary_strided_impl(sycl::queue &exec_q,
                        std::size_t nelems,
                        int nd,
                        const ssize_t *shape_and_strides,
                        const char *arg1_p,
                        ssize_t arg1_offset,
                        const char *arg2_p,
                        ssize_t arg2_offset,
                        char *res_p,
                        ssize_t res_offset,
                        const std::vector<sycl::event> &depends,
                        const std::vector<sycl::event> &additional_depends)
{
    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.depends_on(additional_depends);

        using resTy = typename BinaryOutputType<argTy1, argTy2>::value_type;

        using IndexerT =
            typename dpctl::tensor::offset_utils::ThreeOffsets_StridedIndexer;

        const IndexerT indexer{nd, arg1_offset, arg2_offset, res_offset,
                               shape_and_strides};

        const argTy1 *arg1_tp = reinterpret_cast<const argTy1 *>(arg1_p);
        const argTy2 *arg2_tp = reinterpret_cast<const argTy2 *>(arg2_p);
        resTy *res_tp = reinterpret_cast<resTy *>(res_p);

        using Impl = BinaryStridedFunctorT<argTy1, argTy2, resTy, IndexerT>;

        cgh.parallel_for<kernel_name<argTy1, argTy2, resTy, IndexerT>>(
            {nelems}, Impl(arg1_tp, arg2_tp, res_tp, indexer));
    });
    return comp_ev;
}

template <typename argT1,
          typename argT2,
          typename resT,
          template <typename T1, typename T2, typename T3>
          class BinaryContigMatrixContigRowBroadcastFunctorT,
          template <typename T1, typename T2, typename T3>
          class kernel_name>
sycl::event binary_contig_matrix_contig_row_broadcast_impl(
    sycl::queue &exec_q,
    std::vector<sycl::event> &host_tasks,
    std::size_t n0,
    std::size_t n1,
    const char *mat_p, // typeless pointer to (n0, n1) C-contiguous matrix
    ssize_t mat_offset,
    const char *vec_p, // typeless pointer to (n1,) contiguous row
    ssize_t vec_offset,
    char *res_p, // typeless pointer to (n0, n1) result C-contig. matrix,
                 //    res[i,j] = op(mat[i,j], vec[j])
    ssize_t res_offset,
    const std::vector<sycl::event> &depends = {})
{
    const argT1 *mat = reinterpret_cast<const argT1 *>(mat_p) + mat_offset;
    const argT2 *vec = reinterpret_cast<const argT2 *>(vec_p) + vec_offset;
    resT *res = reinterpret_cast<resT *>(res_p) + res_offset;

    const auto &dev = exec_q.get_device();
    const auto &sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    // Get device-specific kernel info max_sub_group_size
    std::size_t max_sgSize =
        *(std::max_element(std::begin(sg_sizes), std::end(sg_sizes)));

    std::size_t n1_padded = n1 + max_sgSize;
    auto padded_vec_owner =
        dpctl::tensor::alloc_utils::smart_malloc_device<argT2>(n1_padded,
                                                               exec_q);
    argT2 *padded_vec = padded_vec_owner.get();

    sycl::event make_padded_vec_ev =
        dpctl::tensor::kernels::elementwise_detail::populate_padded_vector<
            argT2>(exec_q, vec, n1, padded_vec, n1_padded, depends);

    // sub-group spans work-items [I, I + sgSize)
    // base = ndit.get_global_linear_id() - sg.get_local_id()[0]
    // Generically, sub_group_load( &mat[base]) may load arrays from
    // different rows of mat. The start corresponds to row (base / n0)
    // We read sub_group_load(&padded_vec[(base / n0)]).
    // The vector is padded to ensure that reads are accessible

    const std::size_t lws = 128;

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(make_padded_vec_ev);

        auto lwsRange = sycl::range<1>(lws);
        std::size_t n_elems = n0 * n1;
        std::size_t n_groups = (n_elems + lws - 1) / lws;
        auto gwsRange = sycl::range<1>(n_groups * lws);

        using Impl =
            BinaryContigMatrixContigRowBroadcastFunctorT<argT1, argT2, resT>;

        cgh.parallel_for<class kernel_name<argT1, argT2, resT>>(
            sycl::nd_range<1>(gwsRange, lwsRange),
            Impl(mat, padded_vec, res, n_elems, n1));
    });

    sycl::event tmp_cleanup_ev = dpctl::tensor::alloc_utils::async_smart_free(
        exec_q, {comp_ev}, padded_vec_owner);

    host_tasks.push_back(tmp_cleanup_ev);

    return comp_ev;
}

template <typename argT1,
          typename argT2,
          typename resT,
          template <typename T1, typename T2, typename T3>
          class BinaryContigRowContigMatrixBroadcastFunctorT,
          template <typename T1, typename T2, typename T3>
          class kernel_name>
sycl::event binary_contig_row_contig_matrix_broadcast_impl(
    sycl::queue &exec_q,
    std::vector<sycl::event> &host_tasks,
    std::size_t n0,
    std::size_t n1,
    const char *vec_p, // typeless pointer to (n1,) contiguous row
    ssize_t vec_offset,
    const char *mat_p, // typeless pointer to (n0, n1) C-contiguous matrix
    ssize_t mat_offset,
    char *res_p, // typeless pointer to (n0, n1) result C-contig. matrix,
                 //    res[i,j] = op(vec[j], mat[i,j])
    ssize_t res_offset,
    const std::vector<sycl::event> &depends = {})
{
    const argT1 *vec = reinterpret_cast<const argT2 *>(vec_p) + vec_offset;
    const argT2 *mat = reinterpret_cast<const argT1 *>(mat_p) + mat_offset;
    resT *res = reinterpret_cast<resT *>(res_p) + res_offset;

    const auto &dev = exec_q.get_device();
    const auto &sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    // Get device-specific kernel info max_sub_group_size
    std::size_t max_sgSize =
        *(std::max_element(std::begin(sg_sizes), std::end(sg_sizes)));

    std::size_t n1_padded = n1 + max_sgSize;
    auto padded_vec_owner =
        dpctl::tensor::alloc_utils::smart_malloc_device<argT2>(n1_padded,
                                                               exec_q);
    argT2 *padded_vec = padded_vec_owner.get();

    sycl::event make_padded_vec_ev =
        dpctl::tensor::kernels::elementwise_detail::populate_padded_vector<
            argT2>(exec_q, vec, n1, padded_vec, n1_padded, depends);

    // sub-group spans work-items [I, I + sgSize)
    // base = ndit.get_global_linear_id() - sg.get_local_id()[0]
    // Generically, sub_group_load( &mat[base]) may load arrays from
    // different rows of mat. The start corresponds to row (base / n0)
    // We read sub_group_load(&padded_vec[(base / n0)]). The vector is
    // padded to ensure that reads are accessible

    const std::size_t lws = 128;

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(make_padded_vec_ev);

        auto lwsRange = sycl::range<1>(lws);
        std::size_t n_elems = n0 * n1;
        std::size_t n_groups = (n_elems + lws - 1) / lws;
        auto gwsRange = sycl::range<1>(n_groups * lws);

        using Impl =
            BinaryContigRowContigMatrixBroadcastFunctorT<argT1, argT2, resT>;

        cgh.parallel_for<class kernel_name<argT1, argT2, resT>>(
            sycl::nd_range<1>(gwsRange, lwsRange),
            Impl(padded_vec, mat, res, n_elems, n1));
    });

    sycl::event tmp_cleanup_ev = dpctl::tensor::alloc_utils::async_smart_free(
        exec_q, {comp_ev}, padded_vec_owner);

    host_tasks.push_back(tmp_cleanup_ev);

    return comp_ev;
};
} // namespace dpctl::tensor::kernels::elementwise_common
