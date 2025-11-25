//*****************************************************************************
// Copyright (c) 2025, Intel Corporation
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

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

// dpctl tensor headers
#include "kernels/alignment.hpp"
#include "kernels/elementwise_functions/common.hpp"
#include "utils/sycl_utils.hpp"

namespace dpnp::extensions::py_internal::elementwise_common
{
using dpctl::tensor::kernels::alignment_utils::
    disabled_sg_loadstore_wrapper_krn;
using dpctl::tensor::kernels::alignment_utils::is_aligned;
using dpctl::tensor::kernels::alignment_utils::required_alignment;

using dpctl::tensor::kernels::elementwise_common::select_lws;

using dpctl::tensor::sycl_utils::sub_group_load;
using dpctl::tensor::sycl_utils::sub_group_store;

/**
 * @brief Functor for evaluation of a unary function with two output arrays on
 * contiguous arrays.
 *
 * @note It extends UnaryContigFunctor from
 * dpctl::tensor::kernels::elementwise_common namespace.
 */
template <typename argT,
          typename resT1,
          typename resT2,
          typename UnaryTwoOutputsOpT,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
struct UnaryTwoOutputsContigFunctor
{
private:
    const argT *in = nullptr;
    resT1 *out1 = nullptr;
    resT2 *out2 = nullptr;
    std::size_t nelems_;

public:
    UnaryTwoOutputsContigFunctor(const argT *inp,
                                 resT1 *res1,
                                 resT2 *res2,
                                 const std::size_t n_elems)
        : in(inp), out1(res1), out2(res2), nelems_(n_elems)
    {
    }

    void operator()(sycl::nd_item<1> ndit) const
    {
        static constexpr std::uint8_t elems_per_wi = n_vecs * vec_sz;
        UnaryTwoOutputsOpT op{};
        /* Each work-item processes vec_sz elements, contiguous in memory */
        /* NOTE: work-group size must be divisible by sub-group size */

        if constexpr (enable_sg_loadstore &&
                      UnaryTwoOutputsOpT::is_constant::value) {
            // value of operator is known to be a known constant
            constexpr resT1 const_val1 = UnaryTwoOutputsOpT::constant_value1;
            constexpr resT2 const_val2 = UnaryTwoOutputsOpT::constant_value2;

            auto sg = ndit.get_sub_group();
            const std::uint16_t sgSize = sg.get_max_local_range()[0];

            const std::size_t base =
                elems_per_wi * (ndit.get_group(0) * ndit.get_local_range(0) +
                                sg.get_group_id()[0] * sgSize);
            if (base + elems_per_wi * sgSize < nelems_) {
                static constexpr sycl::vec<resT1, vec_sz> res1_vec(const_val1);
                static constexpr sycl::vec<resT2, vec_sz> res2_vec(const_val2);
#pragma unroll
                for (std::uint8_t it = 0; it < elems_per_wi; it += vec_sz) {
                    const std::size_t offset = base + it * sgSize;
                    auto out1_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out1[offset]);
                    auto out2_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out2[offset]);

                    sub_group_store<vec_sz>(sg, res1_vec, out1_multi_ptr);
                    sub_group_store<vec_sz>(sg, res2_vec, out2_multi_ptr);
                }
            }
            else {
                const std::size_t lane_id = sg.get_local_id()[0];
                for (std::size_t k = base + lane_id; k < nelems_; k += sgSize) {
                    out1[k] = const_val1;
                    out2[k] = const_val2;
                }
            }
        }
        else if constexpr (enable_sg_loadstore &&
                           UnaryTwoOutputsOpT::supports_sg_loadstore::value &&
                           UnaryTwoOutputsOpT::supports_vec::value &&
                           (vec_sz > 1))
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
                    auto out1_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out1[offset]);
                    auto out2_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out2[offset]);

                    const sycl::vec<argT, vec_sz> x =
                        sub_group_load<vec_sz>(sg, in_multi_ptr);
                    sycl::vec<resT2, vec_sz> res2_vec = {};
                    const sycl::vec<resT1, vec_sz> res1_vec = op(x, res2_vec);
                    sub_group_store<vec_sz>(sg, res1_vec, out1_multi_ptr);
                    sub_group_store<vec_sz>(sg, res2_vec, out2_multi_ptr);
                }
            }
            else {
                const std::size_t lane_id = sg.get_local_id()[0];
                for (std::size_t k = base + lane_id; k < nelems_; k += sgSize) {
                    // scalar call
                    out1[k] = op(in[k], out2[k]);
                }
            }
        }
        else if constexpr (enable_sg_loadstore &&
                           UnaryTwoOutputsOpT::supports_sg_loadstore::value &&
                           std::is_same_v<resT1, argT>)
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
                    auto out1_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out1[offset]);
                    auto out2_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out2[offset]);

                    sycl::vec<argT, vec_sz> arg_vec =
                        sub_group_load<vec_sz>(sg, in_multi_ptr);
                    sycl::vec<resT2, vec_sz> res2_vec = {};
#pragma unroll
                    for (std::uint32_t k = 0; k < vec_sz; ++k) {
                        arg_vec[k] = op(arg_vec[k], res2_vec[k]);
                    }
                    sub_group_store<vec_sz>(sg, arg_vec, out1_multi_ptr);
                    sub_group_store<vec_sz>(sg, res2_vec, out2_multi_ptr);
                }
            }
            else {
                const std::size_t lane_id = sg.get_local_id()[0];
                for (std::size_t k = base + lane_id; k < nelems_; k += sgSize) {
                    out1[k] = op(in[k], out2[k]);
                }
            }
        }
        else if constexpr (enable_sg_loadstore &&
                           UnaryTwoOutputsOpT::supports_sg_loadstore::value)
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
                    auto out1_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out1[offset]);
                    auto out2_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out2[offset]);

                    const sycl::vec<argT, vec_sz> arg_vec =
                        sub_group_load<vec_sz>(sg, in_multi_ptr);
                    sycl::vec<resT1, vec_sz> res1_vec = {};
                    sycl::vec<resT2, vec_sz> res2_vec = {};
#pragma unroll
                    for (std::uint8_t k = 0; k < vec_sz; ++k) {
                        res1_vec[k] = op(arg_vec[k], res2_vec[k]);
                    }
                    sub_group_store<vec_sz>(sg, res1_vec, out1_multi_ptr);
                    sub_group_store<vec_sz>(sg, res2_vec, out2_multi_ptr);
                }
            }
            else {
                const std::size_t lane_id = sg.get_local_id()[0];
                for (std::size_t k = base + lane_id; k < nelems_; k += sgSize) {
                    out1[k] = op(in[k], out2[k]);
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
                out1[offset] = op(in[offset], out2[offset]);
            }
        }
    }
};

/**
 * @brief Functor for evaluation of a unary function with two output arrays on
 * strided data.
 *
 * @note It extends UnaryStridedFunctor from
 * dpctl::tensor::kernels::elementwise_common namespace.
 */
template <typename argT,
          typename resT1,
          typename resT2,
          typename IndexerT,
          typename UnaryTwoOutputsOpT>
struct UnaryTwoOutputsStridedFunctor
{
private:
    const argT *inp_ = nullptr;
    resT1 *res1_ = nullptr;
    resT2 *res2_ = nullptr;
    IndexerT inp_out_indexer_;

public:
    UnaryTwoOutputsStridedFunctor(const argT *inp_p,
                                  resT1 *res1_p,
                                  resT2 *res2_p,
                                  const IndexerT &inp_out_indexer)
        : inp_(inp_p), res1_(res1_p), res2_(res2_p),
          inp_out_indexer_(inp_out_indexer)
    {
    }

    void operator()(sycl::id<1> wid) const
    {
        const auto &offsets_ = inp_out_indexer_(wid.get(0));
        const ssize_t &inp_offset = offsets_.get_first_offset();
        const ssize_t &res1_offset = offsets_.get_second_offset();
        const ssize_t &res2_offset = offsets_.get_third_offset();

        UnaryTwoOutputsOpT op{};

        res1_[res1_offset] = op(inp_[inp_offset], res2_[res2_offset]);
    }
};

/**
 * @brief Functor for evaluation of a binary function with two output arrays on
 * contiguous arrays.
 *
 * @note It extends BinaryContigFunctor from
 * dpctl::tensor::kernels::elementwise_common namespace.
 */
template <typename argT1,
          typename argT2,
          typename resT1,
          typename resT2,
          typename BinaryOperatorT,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
struct BinaryTwoOutputsContigFunctor
{
private:
    const argT1 *in1 = nullptr;
    const argT2 *in2 = nullptr;
    resT1 *out1 = nullptr;
    resT2 *out2 = nullptr;
    std::size_t nelems_;

public:
    BinaryTwoOutputsContigFunctor(const argT1 *inp1,
                                  const argT2 *inp2,
                                  resT1 *res1,
                                  resT2 *res2,
                                  std::size_t n_elems)
        : in1(inp1), in2(inp2), out1(res1), out2(res2), nelems_(n_elems)
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
                sycl::vec<resT1, vec_sz> res1_vec;
                sycl::vec<resT2, vec_sz> res2_vec;

#pragma unroll
                for (std::uint8_t it = 0; it < elems_per_wi; it += vec_sz) {
                    std::size_t offset = base + it * sgSize;
                    auto in1_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&in1[offset]);
                    auto in2_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&in2[offset]);
                    auto out1_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out1[offset]);
                    auto out2_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out2[offset]);

                    const sycl::vec<argT1, vec_sz> arg1_vec =
                        sub_group_load<vec_sz>(sg, in1_multi_ptr);
                    const sycl::vec<argT2, vec_sz> arg2_vec =
                        sub_group_load<vec_sz>(sg, in2_multi_ptr);
                    res1_vec = op(arg1_vec, arg2_vec, res2_vec);
                    sub_group_store<vec_sz>(sg, res1_vec, out1_multi_ptr);
                    sub_group_store<vec_sz>(sg, res2_vec, out2_multi_ptr);
                }
            }
            else {
                const std::size_t lane_id = sg.get_local_id()[0];
                for (std::size_t k = base + lane_id; k < nelems_; k += sgSize) {
                    out1[k] = op(in1[k], in2[k], out2[k]);
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
                    auto out1_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out1[offset]);
                    auto out2_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out2[offset]);

                    const sycl::vec<argT1, vec_sz> arg1_vec =
                        sub_group_load<vec_sz>(sg, in1_multi_ptr);
                    const sycl::vec<argT2, vec_sz> arg2_vec =
                        sub_group_load<vec_sz>(sg, in2_multi_ptr);

                    sycl::vec<resT1, vec_sz> res1_vec;
                    sycl::vec<resT2, vec_sz> res2_vec;
#pragma unroll
                    for (std::uint8_t vec_id = 0; vec_id < vec_sz; ++vec_id) {
                        res1_vec[vec_id] =
                            op(arg1_vec[vec_id], arg2_vec[vec_id],
                               res2_vec[vec_id]);
                    }
                    sub_group_store<vec_sz>(sg, res1_vec, out1_multi_ptr);
                    sub_group_store<vec_sz>(sg, res2_vec, out2_multi_ptr);
                }
            }
            else {
                const std::size_t lane_id = sg.get_local_id()[0];
                for (std::size_t k = base + lane_id; k < nelems_; k += sgSize) {
                    out1[k] = op(in1[k], in2[k], out2[k]);
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
                out1[offset] = op(in1[offset], in2[offset], out2[offset]);
            }
        }
    }
};

/**
 * @brief Functor for evaluation of a binary function with two output arrays on
 * strided data.
 *
 * @note It extends BinaryStridedFunctor from
 * dpctl::tensor::kernels::elementwise_common namespace.
 */
template <typename argT1,
          typename argT2,
          typename resT1,
          typename resT2,
          typename FourOffsets_IndexerT,
          typename BinaryOperatorT>
struct BinaryTwoOutputsStridedFunctor
{
private:
    const argT1 *in1 = nullptr;
    const argT2 *in2 = nullptr;
    resT1 *out1 = nullptr;
    resT2 *out2 = nullptr;
    FourOffsets_IndexerT four_offsets_indexer_;

public:
    BinaryTwoOutputsStridedFunctor(const argT1 *inp1_tp,
                                   const argT2 *inp2_tp,
                                   resT1 *res1_tp,
                                   resT2 *res2_tp,
                                   const FourOffsets_IndexerT &inps_res_indexer)
        : in1(inp1_tp), in2(inp2_tp), out1(res1_tp), out2(res2_tp),
          four_offsets_indexer_(inps_res_indexer)
    {
    }

    void operator()(sycl::id<1> wid) const
    {
        const auto &four_offsets_ =
            four_offsets_indexer_(static_cast<ssize_t>(wid.get(0)));

        const auto &inp1_offset = four_offsets_.get_first_offset();
        const auto &inp2_offset = four_offsets_.get_second_offset();
        const auto &out1_offset = four_offsets_.get_third_offset();
        const auto &out2_offset = four_offsets_.get_fourth_offset();

        BinaryOperatorT op{};
        out1[out1_offset] =
            op(in1[inp1_offset], in2[inp2_offset], out2[out2_offset]);
    }
};

/**
 * @brief Function to submit a kernel for unary functor with two output arrays
 * on contiguous arrays.
 *
 * @note It extends unary_contig_impl from
 * dpctl::tensor::kernels::elementwise_common namespace.
 */
template <typename argTy,
          template <typename T>
          class UnaryTwoOutputsType,
          template <typename A,
                    typename R1,
                    typename R2,
                    std::uint8_t vs,
                    std::uint8_t nv,
                    bool enable>
          class UnaryTwoOutputsContigFunctorT,
          template <typename A,
                    typename R1,
                    typename R2,
                    std::uint8_t vs,
                    std::uint8_t nv>
          class kernel_name,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u>
sycl::event
    unary_two_outputs_contig_impl(sycl::queue &exec_q,
                                  std::size_t nelems,
                                  const char *arg_p,
                                  char *res1_p,
                                  char *res2_p,
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

    using resTy1 = typename UnaryTwoOutputsType<argTy>::value_type1;
    using resTy2 = typename UnaryTwoOutputsType<argTy>::value_type2;
    using BaseKernelName = kernel_name<argTy, resTy1, resTy2, vec_sz, n_vecs>;

    const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_p);
    resTy1 *res1_tp = reinterpret_cast<resTy1 *>(res1_p);
    resTy2 *res2_tp = reinterpret_cast<resTy2 *>(res2_p);

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        if (is_aligned<required_alignment>(arg_p) &&
            is_aligned<required_alignment>(res1_p) &&
            is_aligned<required_alignment>(res2_p))
        {
            static constexpr bool enable_sg_loadstore = true;
            using KernelName = BaseKernelName;
            using Impl =
                UnaryTwoOutputsContigFunctorT<argTy, resTy1, resTy2, vec_sz,
                                              n_vecs, enable_sg_loadstore>;

            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(gws_range, lws_range),
                Impl(arg_tp, res1_tp, res2_tp, nelems));
        }
        else {
            static constexpr bool disable_sg_loadstore = false;
            using KernelName =
                disabled_sg_loadstore_wrapper_krn<BaseKernelName>;
            using Impl =
                UnaryTwoOutputsContigFunctorT<argTy, resTy1, resTy2, vec_sz,
                                              n_vecs, disable_sg_loadstore>;

            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(gws_range, lws_range),
                Impl(arg_tp, res1_tp, res2_tp, nelems));
        }
    });

    return comp_ev;
}

/**
 * @brief Function to submit a kernel for unary functor with two output arrays
 * on strided data.
 *
 * @note It extends unary_strided_impl from
 * dpctl::tensor::kernels::elementwise_common namespace.
 */
template <typename argTy,
          template <typename T>
          class UnaryTwoOutputsType,
          template <typename A, typename R1, typename R2, typename I>
          class UnaryTwoOutputsStridedFunctorT,
          template <typename A, typename R1, typename R2, typename I>
          class kernel_name>
sycl::event unary_two_outputs_strided_impl(
    sycl::queue &exec_q,
    std::size_t nelems,
    int nd,
    const ssize_t *shape_and_strides,
    const char *arg_p,
    ssize_t arg_offset,
    char *res1_p,
    ssize_t res1_offset,
    char *res2_p,
    ssize_t res2_offset,
    const std::vector<sycl::event> &depends,
    const std::vector<sycl::event> &additional_depends)
{
    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.depends_on(additional_depends);

        using res1Ty = typename UnaryTwoOutputsType<argTy>::value_type1;
        using res2Ty = typename UnaryTwoOutputsType<argTy>::value_type2;
        using IndexerT =
            typename dpctl::tensor::offset_utils::ThreeOffsets_StridedIndexer;

        const IndexerT indexer{nd, arg_offset, res1_offset, res2_offset,
                               shape_and_strides};

        const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_p);
        res1Ty *res1_tp = reinterpret_cast<res1Ty *>(res1_p);
        res2Ty *res2_tp = reinterpret_cast<res2Ty *>(res2_p);

        using Impl =
            UnaryTwoOutputsStridedFunctorT<argTy, res1Ty, res2Ty, IndexerT>;

        cgh.parallel_for<kernel_name<argTy, res1Ty, res2Ty, IndexerT>>(
            {nelems}, Impl(arg_tp, res1_tp, res2_tp, indexer));
    });
    return comp_ev;
}

/**
 * @brief Function to submit a kernel for binary functor with two output arrays
 * on contiguous arrays.
 *
 * @note It extends binary_contig_impl from
 * dpctl::tensor::kernels::elementwise_common namespace.
 */
template <typename argTy1,
          typename argTy2,
          template <typename T1, typename T2>
          class BinaryTwoOutputsType,
          template <typename T1,
                    typename T2,
                    typename T3,
                    typename T4,
                    std::uint8_t vs,
                    std::uint8_t nv,
                    bool enable_sg_loadstore>
          class BinaryTwoOutputsContigFunctorT,
          template <typename T1,
                    typename T2,
                    typename T3,
                    typename T4,
                    std::uint8_t vs,
                    std::uint8_t nv>
          class kernel_name,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u>
sycl::event
    binary_two_outputs_contig_impl(sycl::queue &exec_q,
                                   std::size_t nelems,
                                   const char *arg1_p,
                                   ssize_t arg1_offset,
                                   const char *arg2_p,
                                   ssize_t arg2_offset,
                                   char *res1_p,
                                   ssize_t res1_offset,
                                   char *res2_p,
                                   ssize_t res2_offset,
                                   const std::vector<sycl::event> &depends = {})
{
    const std::size_t n_work_items_needed = nelems / (n_vecs * vec_sz);
    const std::size_t lws =
        select_lws(exec_q.get_device(), n_work_items_needed);

    const std::size_t n_groups =
        ((nelems + lws * n_vecs * vec_sz - 1) / (lws * n_vecs * vec_sz));
    const auto gws_range = sycl::range<1>(n_groups * lws);
    const auto lws_range = sycl::range<1>(lws);

    using resTy1 = typename BinaryTwoOutputsType<argTy1, argTy2>::value_type1;
    using resTy2 = typename BinaryTwoOutputsType<argTy1, argTy2>::value_type2;
    using BaseKernelName =
        kernel_name<argTy1, argTy2, resTy1, resTy2, vec_sz, n_vecs>;

    const argTy1 *arg1_tp =
        reinterpret_cast<const argTy1 *>(arg1_p) + arg1_offset;
    const argTy2 *arg2_tp =
        reinterpret_cast<const argTy2 *>(arg2_p) + arg2_offset;
    resTy1 *res1_tp = reinterpret_cast<resTy1 *>(res1_p) + res1_offset;
    resTy2 *res2_tp = reinterpret_cast<resTy2 *>(res2_p) + res2_offset;

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        if (is_aligned<required_alignment>(arg1_tp) &&
            is_aligned<required_alignment>(arg2_tp) &&
            is_aligned<required_alignment>(res1_tp) &&
            is_aligned<required_alignment>(res2_tp))
        {
            static constexpr bool enable_sg_loadstore = true;
            using KernelName = BaseKernelName;
            using Impl = BinaryTwoOutputsContigFunctorT<argTy1, argTy2, resTy1,
                                                        resTy2, vec_sz, n_vecs,
                                                        enable_sg_loadstore>;

            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(gws_range, lws_range),
                Impl(arg1_tp, arg2_tp, res1_tp, res2_tp, nelems));
        }
        else {
            static constexpr bool disable_sg_loadstore = false;
            using KernelName =
                disabled_sg_loadstore_wrapper_krn<BaseKernelName>;
            using Impl = BinaryTwoOutputsContigFunctorT<argTy1, argTy2, resTy1,
                                                        resTy2, vec_sz, n_vecs,
                                                        disable_sg_loadstore>;

            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(gws_range, lws_range),
                Impl(arg1_tp, arg2_tp, res1_tp, res2_tp, nelems));
        }
    });
    return comp_ev;
}

/**
 * @brief Function to submit a kernel for binary functor with two output arrays
 * on strided data.
 *
 * @note It extends binary_strided_impl from
 * dpctl::tensor::kernels::elementwise_common namespace.
 */
template <
    typename argTy1,
    typename argTy2,
    template <typename T1, typename T2>
    class BinaryTwoOutputsType,
    template <typename T1, typename T2, typename T3, typename T4, typename IndT>
    class BinaryTwoOutputsStridedFunctorT,
    template <typename T1, typename T2, typename T3, typename T4, typename IndT>
    class kernel_name>
sycl::event binary_two_outputs_strided_impl(
    sycl::queue &exec_q,
    std::size_t nelems,
    int nd,
    const ssize_t *shape_and_strides,
    const char *arg1_p,
    ssize_t arg1_offset,
    const char *arg2_p,
    ssize_t arg2_offset,
    char *res1_p,
    ssize_t res1_offset,
    char *res2_p,
    ssize_t res2_offset,
    const std::vector<sycl::event> &depends,
    const std::vector<sycl::event> &additional_depends)
{
    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.depends_on(additional_depends);

        using resTy1 =
            typename BinaryTwoOutputsType<argTy1, argTy2>::value_type1;
        using resTy2 =
            typename BinaryTwoOutputsType<argTy1, argTy2>::value_type2;

        using IndexerT =
            typename dpctl::tensor::offset_utils::FourOffsets_StridedIndexer;

        const IndexerT indexer{nd,          arg1_offset, arg2_offset,
                               res1_offset, res2_offset, shape_and_strides};

        const argTy1 *arg1_tp = reinterpret_cast<const argTy1 *>(arg1_p);
        const argTy2 *arg2_tp = reinterpret_cast<const argTy2 *>(arg2_p);
        resTy1 *res1_tp = reinterpret_cast<resTy1 *>(res1_p);
        resTy2 *res2_tp = reinterpret_cast<resTy2 *>(res2_p);

        using Impl = BinaryTwoOutputsStridedFunctorT<argTy1, argTy2, resTy1,
                                                     resTy2, IndexerT>;

        cgh.parallel_for<kernel_name<argTy1, argTy2, resTy1, resTy2, IndexerT>>(
            {nelems}, Impl(arg1_tp, arg2_tp, res1_tp, res2_tp, indexer));
    });
    return comp_ev;
}

// Typedefs for function pointers

typedef sycl::event (*unary_two_outputs_contig_impl_fn_ptr_t)(
    sycl::queue &,
    std::size_t,
    const char *,
    char *,
    char *,
    const std::vector<sycl::event> &);

typedef sycl::event (*unary_two_outputs_strided_impl_fn_ptr_t)(
    sycl::queue &,
    std::size_t,
    int,
    const ssize_t *,
    const char *,
    ssize_t,
    char *,
    ssize_t,
    char *,
    ssize_t,
    const std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

typedef sycl::event (*binary_two_outputs_contig_impl_fn_ptr_t)(
    sycl::queue &,
    std::size_t,
    const char *,
    ssize_t,
    const char *,
    ssize_t,
    char *,
    ssize_t,
    char *,
    ssize_t,
    const std::vector<sycl::event> &);

typedef sycl::event (*binary_two_outputs_strided_impl_fn_ptr_t)(
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
    char *,
    ssize_t,
    const std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

} // namespace dpnp::extensions::py_internal::elementwise_common
