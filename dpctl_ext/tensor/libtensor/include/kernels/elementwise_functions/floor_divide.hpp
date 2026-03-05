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
/// This file defines kernels for elementwise evaluation of FLOOR_DIVIDE(x1, x2)
/// function.
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

#include "utils/offset_utils.hpp"
#include "utils/type_dispatch_building.hpp"
#include "utils/type_utils.hpp"

namespace dpctl::tensor::kernels::floor_divide
{

using dpctl::tensor::ssize_t;
namespace td_ns = dpctl::tensor::type_dispatch;
namespace tu_ns = dpctl::tensor::type_utils;

template <typename argT1, typename argT2, typename resT>
struct FloorDivideFunctor
{
    using supports_sg_loadstore = std::true_type;
    using supports_vec = std::true_type;

    resT operator()(const argT1 &in1, const argT2 &in2) const
    {
        if constexpr (std::is_integral_v<argT1> || std::is_integral_v<argT2>) {
            if (in2 == argT2(0)) {
                return resT(0);
            }
            if constexpr (std::is_signed_v<argT1> || std::is_signed_v<argT2>) {
                auto div = in1 / in2;
                auto mod = in1 % in2;
                auto corr = (mod != 0 && l_xor(mod < 0, in2 < 0));
                return (div - corr);
            }
            else {
                return (in1 / in2);
            }
        }
        else {
            auto div = in1 / in2;
            return (div == resT(0)) ? div : resT(sycl::floor(div));
        }
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz>
        operator()(const sycl::vec<argT1, vec_sz> &in1,
                   const sycl::vec<argT2, vec_sz> &in2) const
    {
        if constexpr (std::is_integral_v<resT>) {
            sycl::vec<resT, vec_sz> res;
#pragma unroll
            for (int i = 0; i < vec_sz; ++i) {
                if (in2[i] == argT2(0)) {
                    res[i] = resT(0);
                }
                else {
                    res[i] = in1[i] / in2[i];
                    if constexpr (std::is_signed_v<resT>) {
                        auto mod = in1[i] % in2[i];
                        auto corr = (mod != 0 && l_xor(mod < 0, in2[i] < 0));
                        res[i] -= corr;
                    }
                }
            }
            return res;
        }
        else {
            auto tmp = in1 / in2;
            using tmpT = typename decltype(tmp)::element_type;
#pragma unroll
            for (int i = 0; i < vec_sz; ++i) {
                if (in2[i] != argT2(0)) {
                    tmp[i] = sycl::floor(tmp[i]);
                }
            }
            if constexpr (std::is_same_v<resT, tmpT>) {
                return tmp;
            }
            else {
                using dpctl::tensor::type_utils::vec_cast;
                return vec_cast<resT, tmpT, vec_sz>(tmp);
            }
        }
    }

private:
    bool l_xor(bool b1, bool b2) const
    {
        return (b1 != b2);
    }
};

template <typename argT1,
          typename argT2,
          typename resT,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
using FloorDivideContigFunctor = elementwise_common::BinaryContigFunctor<
    argT1,
    argT2,
    resT,
    FloorDivideFunctor<argT1, argT2, resT>,
    vec_sz,
    n_vecs,
    enable_sg_loadstore>;

template <typename argT1, typename argT2, typename resT, typename IndexerT>
using FloorDivideStridedFunctor = elementwise_common::BinaryStridedFunctor<
    argT1,
    argT2,
    resT,
    IndexerT,
    FloorDivideFunctor<argT1, argT2, resT>>;

template <typename T1, typename T2>
struct FloorDivideOutputType
{
    using value_type = typename std::disjunction<
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint8_t,
                                        T2,
                                        std::uint8_t,
                                        std::uint8_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int8_t,
                                        T2,
                                        std::int8_t,
                                        std::int8_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint16_t,
                                        T2,
                                        std::uint16_t,
                                        std::uint16_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int16_t,
                                        T2,
                                        std::int16_t,
                                        std::int16_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint32_t,
                                        T2,
                                        std::uint32_t,
                                        std::uint32_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int32_t,
                                        T2,
                                        std::int32_t,
                                        std::int32_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint64_t,
                                        T2,
                                        std::uint64_t,
                                        std::uint64_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int64_t,
                                        T2,
                                        std::int64_t,
                                        std::int64_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        sycl::half,
                                        T2,
                                        sycl::half,
                                        sycl::half>,
        td_ns::BinaryTypeMapResultEntry<T1, float, T2, float, float>,
        td_ns::BinaryTypeMapResultEntry<T1, double, T2, double, double>,
        td_ns::DefaultResultEntry<void>>::result_type;

    static constexpr bool is_defined = !std::is_same_v<value_type, void>;
};

namespace hyperparam_detail
{

namespace vsu_ns = dpctl::tensor::kernels::vec_size_utils;

using vsu_ns::BinaryContigHyperparameterSetEntry;
using vsu_ns::ContigHyperparameterSetDefault;

template <typename argTy1, typename argTy2>
struct FloorDivideContigHyperparameterSet
{
    using value_type =
        typename std::disjunction<ContigHyperparameterSetDefault<4u, 2u>>;

    constexpr static auto vec_sz = value_type::vec_sz;
    constexpr static auto n_vecs = value_type::n_vecs;
};

} // end of namespace hyperparam_detail

template <typename argT1,
          typename argT2,
          typename resT,
          std::uint8_t vec_sz,
          std::uint8_t n_vecs>
class floor_divide_contig_kernel;

template <typename argTy1, typename argTy2>
sycl::event
    floor_divide_contig_impl(sycl::queue &exec_q,
                             std::size_t nelems,
                             const char *arg1_p,
                             ssize_t arg1_offset,
                             const char *arg2_p,
                             ssize_t arg2_offset,
                             char *res_p,
                             ssize_t res_offset,
                             const std::vector<sycl::event> &depends = {})
{
    using FloorDivideHS =
        hyperparam_detail::FloorDivideContigHyperparameterSet<argTy1, argTy2>;
    static constexpr std::uint8_t vec_sz = FloorDivideHS::vec_sz;
    static constexpr std::uint8_t n_vecs = FloorDivideHS::n_vecs;

    return elementwise_common::binary_contig_impl<
        argTy1, argTy2, FloorDivideOutputType, FloorDivideContigFunctor,
        floor_divide_contig_kernel, vec_sz, n_vecs>(
        exec_q, nelems, arg1_p, arg1_offset, arg2_p, arg2_offset, res_p,
        res_offset, depends);
}

template <typename fnT, typename T1, typename T2>
struct FloorDivideContigFactory
{
    fnT get()
    {
        if constexpr (!FloorDivideOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = floor_divide_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2>
struct FloorDivideTypeMapFactory
{
    /*! @brief get typeid for output type of floor_divide(T1 x, T2 y) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename FloorDivideOutputType<T1, T2>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename resT, typename IndexerT>
class floor_divide_strided_kernel;

template <typename argTy1, typename argTy2>
sycl::event floor_divide_strided_impl(
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
        argTy1, argTy2, FloorDivideOutputType, FloorDivideStridedFunctor,
        floor_divide_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg1_p, arg1_offset, arg2_p,
        arg2_offset, res_p, res_offset, depends, additional_depends);
}

template <typename fnT, typename T1, typename T2>
struct FloorDivideStridedFactory
{
    fnT get()
    {
        if constexpr (!FloorDivideOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = floor_divide_strided_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename argT, typename resT>
struct FloorDivideInplaceFunctor
{
    using supports_sg_loadstore = std::true_type;
    using supports_vec = std::true_type;

    void operator()(resT &in1, const argT &in2) const
    {
        if constexpr (std::is_integral_v<resT>) {
            if (in2 == argT(0)) {
                in1 = 0;
                return;
            }
            if constexpr (std::is_signed_v<resT>) {
                auto tmp = in1;
                in1 /= in2;
                auto mod = tmp % in2;
                auto corr = (mod != 0 && l_xor(mod < 0, in2 < 0));
                in1 -= corr;
            }
            else {
                in1 /= in2;
            }
        }
        else {
            in1 /= in2;
            if (in1 == resT(0)) {
                return;
            }
            in1 = sycl::floor(in1);
        }
    }

    template <int vec_sz>
    void operator()(sycl::vec<resT, vec_sz> &in1,
                    const sycl::vec<argT, vec_sz> &in2) const
    {
        if constexpr (std::is_integral_v<resT>) {
#pragma unroll
            for (int i = 0; i < vec_sz; ++i) {
                if (in2[i] == argT(0)) {
                    in1[i] = 0;
                }
                else {
                    if constexpr (std::is_signed_v<resT>) {
                        auto tmp = in1[i];
                        in1[i] /= in2[i];
                        auto mod = tmp % in2[i];
                        auto corr = (mod != 0 && l_xor(mod < 0, in2[i] < 0));
                        in1[i] -= corr;
                    }
                    else {
                        in1[i] /= in2[i];
                    }
                }
            }
        }
        else {
            in1 /= in2;
#pragma unroll
            for (int i = 0; i < vec_sz; ++i) {
                if (in2[i] != argT(0)) {
                    in1[i] = sycl::floor(in1[i]);
                }
            }
        }
    }

private:
    bool l_xor(bool b1, bool b2) const
    {
        return (b1 != b2);
    }
};

template <typename argT,
          typename resT,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
using FloorDivideInplaceContigFunctor =
    elementwise_common::BinaryInplaceContigFunctor<
        argT,
        resT,
        FloorDivideInplaceFunctor<argT, resT>,
        vec_sz,
        n_vecs,
        enable_sg_loadstore>;

template <typename argT, typename resT, typename IndexerT>
using FloorDivideInplaceStridedFunctor =
    elementwise_common::BinaryInplaceStridedFunctor<
        argT,
        resT,
        IndexerT,
        FloorDivideInplaceFunctor<argT, resT>>;

template <typename argT,
          typename resT,
          std::uint8_t vec_sz,
          std::uint8_t n_vecs>
class floor_divide_inplace_contig_kernel;

/* @brief Types supported by in-place floor division */
template <typename argTy, typename resTy>
struct FloorDivideInplaceTypePairSupport
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
        td_ns::TypePairDefinedEntry<argTy, sycl::half, resTy, sycl::half>,
        td_ns::TypePairDefinedEntry<argTy, float, resTy, float>,
        td_ns::TypePairDefinedEntry<argTy, double, resTy, double>,
        // fall-through
        td_ns::NotDefinedEntry>::is_defined;
};

template <typename fnT, typename argT, typename resT>
struct FloorDivideInplaceTypeMapFactory
{
    /*! @brief get typeid for output type of x //= y */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        if constexpr (FloorDivideInplaceTypePairSupport<argT, resT>::is_defined)
        {
            return td_ns::GetTypeid<resT>{}.get();
        }
        else {
            return td_ns::GetTypeid<void>{}.get();
        }
    }
};

template <typename argTy, typename resTy>
sycl::event floor_divide_inplace_contig_impl(
    sycl::queue &exec_q,
    std::size_t nelems,
    const char *arg_p,
    ssize_t arg_offset,
    char *res_p,
    ssize_t res_offset,
    const std::vector<sycl::event> &depends = {})
{
    using FloorDivideHS =
        hyperparam_detail::FloorDivideContigHyperparameterSet<resTy, argTy>;

    static constexpr std::uint8_t vec_sz = FloorDivideHS::vec_sz;
    static constexpr std::uint8_t n_vecs = FloorDivideHS::n_vecs;

    return elementwise_common::binary_inplace_contig_impl<
        argTy, resTy, FloorDivideInplaceContigFunctor,
        floor_divide_inplace_contig_kernel, vec_sz, n_vecs>(
        exec_q, nelems, arg_p, arg_offset, res_p, res_offset, depends);
}

template <typename fnT, typename T1, typename T2>
struct FloorDivideInplaceContigFactory
{
    fnT get()
    {
        if constexpr (!FloorDivideInplaceTypePairSupport<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = floor_divide_inplace_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename resT, typename argT, typename IndexerT>
class floor_divide_inplace_strided_kernel;

template <typename argTy, typename resTy>
sycl::event floor_divide_inplace_strided_impl(
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
        argTy, resTy, FloorDivideInplaceStridedFunctor,
        floor_divide_inplace_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T1, typename T2>
struct FloorDivideInplaceStridedFactory
{
    fnT get()
    {
        if constexpr (!FloorDivideInplaceTypePairSupport<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = floor_divide_inplace_strided_impl<T1, T2>;
            return fn;
        }
    }
};

} // namespace dpctl::tensor::kernels::floor_divide
