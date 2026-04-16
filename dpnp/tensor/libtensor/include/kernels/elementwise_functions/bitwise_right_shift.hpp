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
//===---------------------------------------------------------------------===//
///
/// \file
/// This file defines kernels for elementwise bitwise_right_shift(ar1, ar2)
/// operation.
//===---------------------------------------------------------------------===//

#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>

#include "vec_size_util.hpp"

#include "kernels/dpctl_tensor_types.hpp"
#include "kernels/elementwise_functions/common.hpp"
#include "kernels/elementwise_functions/common_inplace.hpp"

#include "utils/type_dispatch_building.hpp"

namespace dpctl::tensor::kernels::bitwise_right_shift
{

using dpctl::tensor::ssize_t;
namespace td_ns = dpctl::tensor::type_dispatch;

template <typename argT1, typename argT2, typename resT>
struct BitwiseRightShiftFunctor
{
    static_assert(std::is_same_v<resT, argT1>);
    static_assert(std::is_integral_v<argT1>);
    static_assert(std::is_integral_v<argT2>);

    using supports_sg_loadstore = typename std::true_type;
    using supports_vec = typename std::true_type;

    resT operator()(const argT1 &in1, const argT2 &in2) const
    {
        return impl(in1, in2);
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz>
        operator()(const sycl::vec<argT1, vec_sz> &in1,
                   const sycl::vec<argT2, vec_sz> &in2) const
    {
        sycl::vec<resT, vec_sz> res;
#pragma unroll
        for (int i = 0; i < vec_sz; ++i) {
            res[i] = impl(in1[i], in2[i]);
        }
        return res;
    }

private:
    resT impl(const argT1 &in1, const argT2 &in2) const
    {
        static constexpr argT2 in1_bitsize =
            static_cast<argT2>(sizeof(argT1) * 8);
        static constexpr resT zero = resT(0);

        // bitshift op with second operand negative, or >= bitwidth(argT1) is UB
        // array API spec mandates 0
        if constexpr (std::is_unsigned_v<argT2>) {
            return (in2 < in1_bitsize) ? (in1 >> in2) : zero;
        }
        else {
            return (in2 < argT2(0))
                       ? zero
                       : ((in2 < in1_bitsize)
                              ? (in1 >> in2)
                              : (in1 < argT1(0) ? resT(-1) : zero));
        }
    }
};

template <typename argT1,
          typename argT2,
          typename resT,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
using BitwiseRightShiftContigFunctor = elementwise_common::BinaryContigFunctor<
    argT1,
    argT2,
    resT,
    BitwiseRightShiftFunctor<argT1, argT2, resT>,
    vec_sz,
    n_vecs,
    enable_sg_loadstore>;

template <typename argT1, typename argT2, typename resT, typename IndexerT>
using BitwiseRightShiftStridedFunctor =
    elementwise_common::BinaryStridedFunctor<
        argT1,
        argT2,
        resT,
        IndexerT,
        BitwiseRightShiftFunctor<argT1, argT2, resT>>;

template <typename T1, typename T2>
struct BitwiseRightShiftOutputType
{
    using ResT = T1;
    using value_type = typename std::disjunction<
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int8_t,
                                        T2,
                                        std::int8_t,
                                        std::int8_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint8_t,
                                        T2,
                                        std::uint8_t,
                                        std::uint8_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int16_t,
                                        T2,
                                        std::int16_t,
                                        std::int16_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint16_t,
                                        T2,
                                        std::uint16_t,
                                        std::uint16_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int32_t,
                                        T2,
                                        std::int32_t,
                                        std::int32_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint32_t,
                                        T2,
                                        std::uint32_t,
                                        std::uint32_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int64_t,
                                        T2,
                                        std::int64_t,
                                        std::int64_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint64_t,
                                        T2,
                                        std::uint64_t,
                                        std::uint64_t>,
        td_ns::DefaultResultEntry<void>>::result_type;

    static constexpr bool is_defined = !std::is_same_v<value_type, void>;
};

namespace hyperparam_detail
{

namespace vsu_ns = dpctl::tensor::kernels::vec_size_utils;

using vsu_ns::BinaryContigHyperparameterSetEntry;
using vsu_ns::ContigHyperparameterSetDefault;

template <typename argTy1, typename argTy2>
struct BitwiseRightShiftContigHyperparameterSet
{
    using value_type =
        typename std::disjunction<ContigHyperparameterSetDefault<4u, 2u>>;

    constexpr static auto vec_sz = value_type::vec_sz;
    constexpr static auto n_vecs = value_type::n_vecs;
};

} // namespace hyperparam_detail

template <typename argT1,
          typename argT2,
          typename resT,
          std::uint8_t vec_sz,
          std::uint8_t n_vecs>
class bitwise_right_shift_contig_kernel;

template <typename argTy1, typename argTy2>
sycl::event bitwise_right_shift_contig_impl(
    sycl::queue &exec_q,
    std::size_t nelems,
    const char *arg1_p,
    ssize_t arg1_offset,
    const char *arg2_p,
    ssize_t arg2_offset,
    char *res_p,
    ssize_t res_offset,
    const std::vector<sycl::event> &depends = {})
{
    using BitwiseRSHS =
        hyperparam_detail::BitwiseRightShiftContigHyperparameterSet<argTy1,
                                                                    argTy2>;
    constexpr std::uint8_t vec_sz = BitwiseRSHS::vec_sz;
    constexpr std::uint8_t n_vecs = BitwiseRSHS::n_vecs;

    return elementwise_common::binary_contig_impl<
        argTy1, argTy2, BitwiseRightShiftOutputType,
        BitwiseRightShiftContigFunctor, bitwise_right_shift_contig_kernel,
        vec_sz, n_vecs>(exec_q, nelems, arg1_p, arg1_offset, arg2_p,
                        arg2_offset, res_p, res_offset, depends);
}

template <typename fnT, typename T1, typename T2>
struct BitwiseRightShiftContigFactory
{
    fnT get()
    {
        if constexpr (!BitwiseRightShiftOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = bitwise_right_shift_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2>
struct BitwiseRightShiftTypeMapFactory
{
    /*! @brief get typeid for output type of operator()>(x, y), always bool
     */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename BitwiseRightShiftOutputType<T1, T2>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename resT, typename IndexerT>
class bitwise_right_shift_strided_kernel;

template <typename argTy1, typename argTy2>
sycl::event bitwise_right_shift_strided_impl(
    sycl::queue &exec_q,
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
    return elementwise_common::binary_strided_impl<
        argTy1, argTy2, BitwiseRightShiftOutputType,
        BitwiseRightShiftStridedFunctor, bitwise_right_shift_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg1_p, arg1_offset, arg2_p,
        arg2_offset, res_p, res_offset, depends, additional_depends);
}

template <typename fnT, typename T1, typename T2>
struct BitwiseRightShiftStridedFactory
{
    fnT get()
    {
        if constexpr (!BitwiseRightShiftOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = bitwise_right_shift_strided_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename argT, typename resT>
struct BitwiseRightShiftInplaceFunctor
{
    static_assert(std::is_integral_v<argT>);
    static_assert(!std::is_same_v<argT, bool>);

    using supports_sg_loadstore = typename std::true_type;
    using supports_vec = typename std::true_type;

    void operator()(resT &res, const argT &in) const { impl(res, in); }

    template <int vec_sz>
    void operator()(sycl::vec<resT, vec_sz> &res,
                    const sycl::vec<argT, vec_sz> &in) const
    {
#pragma unroll
        for (int i = 0; i < vec_sz; ++i) {
            impl(res[i], in[i]);
        }
    }

private:
    void impl(resT &res, const argT &in) const
    {
        static constexpr argT res_bitsize = static_cast<argT>(sizeof(resT) * 8);
        static constexpr resT zero = resT(0);

        // bitshift op with second operand negative, or >= bitwidth(argT1) is UB
        // array API spec mandates 0
        if constexpr (std::is_unsigned_v<argT>) {
            (in < res_bitsize) ? (res >>= in) : res = zero;
        }
        else {
            (in < argT(0)) ? res = zero
                           : ((in < res_bitsize) ? (res >>= in)
                              : (res < resT(0))  ? res = resT(-1)
                                                 : res = zero);
        }
    }
};

template <typename argT,
          typename resT,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
using BitwiseRightShiftInplaceContigFunctor =
    elementwise_common::BinaryInplaceContigFunctor<
        argT,
        resT,
        BitwiseRightShiftInplaceFunctor<argT, resT>,
        vec_sz,
        n_vecs,
        enable_sg_loadstore>;

template <typename argT, typename resT, typename IndexerT>
using BitwiseRightShiftInplaceStridedFunctor =
    elementwise_common::BinaryInplaceStridedFunctor<
        argT,
        resT,
        IndexerT,
        BitwiseRightShiftInplaceFunctor<argT, resT>>;

template <typename argT,
          typename resT,
          std::uint8_t vec_sz,
          std::uint8_t n_vecs>
class bitwise_right_shift_inplace_contig_kernel;

/* @brief Types supported by in-place bitwise right shift */
template <typename argTy, typename resTy>
struct BitwiseRightShiftInplaceTypePairSupport
{
    /* value if true a kernel for <argTy, resTy> must be instantiated  */
    static constexpr bool is_defined = std::disjunction<
        td_ns::TypePairDefinedEntry<argTy, std::int8_t, resTy, std::int8_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint8_t, resTy, std::uint8_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int16_t, resTy, std::int16_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint16_t, resTy, std::uint16_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int32_t, resTy, std::int32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint32_t, resTy, std::uint32_t>,
        td_ns::TypePairDefinedEntry<argTy, std::int64_t, resTy, std::int64_t>,
        td_ns::TypePairDefinedEntry<argTy, std::uint64_t, resTy, std::uint64_t>,
        // fall-through
        td_ns::NotDefinedEntry>::is_defined;
};

template <typename fnT, typename argT, typename resT>
struct BitwiseRightShiftInplaceTypeMapFactory
{
    /*! @brief get typeid for output type of x >>= y */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        if constexpr (BitwiseRightShiftInplaceTypePairSupport<
                          argT, resT>::is_defined) {
            return td_ns::GetTypeid<resT>{}.get();
        }
        else {
            return td_ns::GetTypeid<void>{}.get();
        }
    }
};

template <typename argTy, typename resTy>
sycl::event bitwise_right_shift_inplace_contig_impl(
    sycl::queue &exec_q,
    std::size_t nelems,
    const char *arg_p,
    ssize_t arg_offset,
    char *res_p,
    ssize_t res_offset,
    const std::vector<sycl::event> &depends = {})
{
    using BitwiseRSHS =
        hyperparam_detail::BitwiseRightShiftContigHyperparameterSet<resTy,
                                                                    argTy>;

    // res = OP(res, arg)
    static constexpr std::uint8_t vec_sz = BitwiseRSHS::vec_sz;
    static constexpr std::uint8_t n_vecs = BitwiseRSHS::n_vecs;

    return elementwise_common::binary_inplace_contig_impl<
        argTy, resTy, BitwiseRightShiftInplaceContigFunctor,
        bitwise_right_shift_inplace_contig_kernel, vec_sz, n_vecs>(
        exec_q, nelems, arg_p, arg_offset, res_p, res_offset, depends);
}

template <typename fnT, typename T1, typename T2>
struct BitwiseRightShiftInplaceContigFactory
{
    fnT get()
    {
        if constexpr (!BitwiseRightShiftInplaceTypePairSupport<
                          T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = bitwise_right_shift_inplace_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename resT, typename argT, typename IndexerT>
class bitwise_right_shift_inplace_strided_kernel;

template <typename argTy, typename resTy>
sycl::event bitwise_right_shift_inplace_strided_impl(
    sycl::queue &exec_q,
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
    return elementwise_common::binary_inplace_strided_impl<
        argTy, resTy, BitwiseRightShiftInplaceStridedFunctor,
        bitwise_right_shift_inplace_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T1, typename T2>
struct BitwiseRightShiftInplaceStridedFactory
{
    fnT get()
    {
        if constexpr (!BitwiseRightShiftInplaceTypePairSupport<
                          T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = bitwise_right_shift_inplace_strided_impl<T1, T2>;
            return fn;
        }
    }
};

} // namespace dpctl::tensor::kernels::bitwise_right_shift
