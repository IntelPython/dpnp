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
/// This file defines kernels for elementwise evaluation of POW(x1, x2)
/// function.
//===---------------------------------------------------------------------===//

#pragma once
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>

#include "sycl_complex.hpp"
#include "vec_size_util.hpp"

#include "utils/offset_utils.hpp"
#include "utils/type_dispatch_building.hpp"
#include "utils/type_utils.hpp"

#include "kernels/dpctl_tensor_types.hpp"
#include "kernels/elementwise_functions/common.hpp"
#include "kernels/elementwise_functions/common_inplace.hpp"

namespace dpctl::tensor::kernels::pow
{

using dpctl::tensor::ssize_t;
namespace td_ns = dpctl::tensor::type_dispatch;
namespace tu_ns = dpctl::tensor::type_utils;

template <typename argT1, typename argT2, typename resT>
struct PowFunctor
{

    using supports_sg_loadstore = std::negation<
        std::disjunction<tu_ns::is_complex<argT1>, tu_ns::is_complex<argT2>>>;
    using supports_vec = std::negation<
        std::disjunction<tu_ns::is_complex<argT1>, tu_ns::is_complex<argT2>>>;

    resT operator()(const argT1 &in1, const argT2 &in2) const
    {
        if constexpr (std::is_integral_v<argT1> || std::is_integral_v<argT2>) {
            auto tmp1 = in1;
            auto tmp2 = in2;
            if constexpr (std::is_signed_v<argT2>) {
                if (tmp2 < 0) {
                    // invalid; return 0
                    return resT(0);
                }
            }
            resT res = 1;
            if (tmp1 == 1 || tmp2 == 0) {
                return res;
            }
            while (tmp2 > 0) {
                if (tmp2 & 1) {
                    res *= tmp1;
                }
                tmp2 >>= 1;
                tmp1 *= tmp1;
            }
            return res;
        }
        else if constexpr (tu_ns::is_complex<argT1>::value &&
                           tu_ns::is_complex<argT2>::value)
        {
            using realT1 = typename argT1::value_type;
            using realT2 = typename argT2::value_type;

            return exprm_ns::pow(exprm_ns::complex<realT1>(in1),
                                 exprm_ns::complex<realT2>(in2));
        }
        else {
            return sycl::pow(in1, in2);
        }
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz>
        operator()(const sycl::vec<argT1, vec_sz> &in1,
                   const sycl::vec<argT2, vec_sz> &in2) const
    {
        if constexpr (std::is_integral_v<argT1> || std::is_integral_v<argT2>) {
            sycl::vec<resT, vec_sz> res;
#pragma unroll
            for (int i = 0; i < vec_sz; ++i) {
                auto tmp1 = in1[i];
                auto tmp2 = in2[i];
                if constexpr (std::is_signed_v<argT2>) {
                    if (tmp2 < 0) {
                        // invalid; yield 0
                        res[i] = 0;
                        continue;
                    }
                }
                resT res_tmp = 1;
                if (tmp1 == 1 || tmp2 == 0) {
                    res[i] = res_tmp;
                    continue;
                }
                while (tmp2 > 0) {
                    if (tmp2 & 1) {
                        res_tmp *= tmp1;
                    }
                    tmp2 >>= 1;
                    tmp1 *= tmp1;
                }
                res[i] = res_tmp;
            }
            return res;
        }
        else {
            auto res = sycl::pow(in1, in2);
            if constexpr (std::is_same_v<resT,
                                         typename decltype(res)::element_type>)
            {
                return res;
            }
            else {
                using dpctl::tensor::type_utils::vec_cast;

                return vec_cast<resT, typename decltype(res)::element_type,
                                vec_sz>(res);
            }
        }
    }
};

template <typename argT1,
          typename argT2,
          typename resT,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
using PowContigFunctor =
    elementwise_common::BinaryContigFunctor<argT1,
                                            argT2,
                                            resT,
                                            PowFunctor<argT1, argT2, resT>,
                                            vec_sz,
                                            n_vecs,
                                            enable_sg_loadstore>;

template <typename argT1, typename argT2, typename resT, typename IndexerT>
using PowStridedFunctor =
    elementwise_common::BinaryStridedFunctor<argT1,
                                             argT2,
                                             resT,
                                             IndexerT,
                                             PowFunctor<argT1, argT2, resT>>;

template <typename T1, typename T2>
struct PowOutputType
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
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::complex<float>,
                                        T2,
                                        std::complex<float>,
                                        std::complex<float>>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::complex<double>,
                                        T2,
                                        std::complex<double>,
                                        std::complex<double>>,
        td_ns::DefaultResultEntry<void>>::result_type;

    static constexpr bool is_defined = !std::is_same_v<value_type, void>;
};

namespace hyperparam_detail
{

namespace vsu_ns = dpctl::tensor::kernels::vec_size_utils;

using vsu_ns::BinaryContigHyperparameterSetEntry;
using vsu_ns::ContigHyperparameterSetDefault;

template <typename argTy1, typename argTy2>
struct PowContigHyperparameterSet
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
class pow_contig_kernel;

template <typename argTy1, typename argTy2>
sycl::event pow_contig_impl(sycl::queue &exec_q,
                            std::size_t nelems,
                            const char *arg1_p,
                            ssize_t arg1_offset,
                            const char *arg2_p,
                            ssize_t arg2_offset,
                            char *res_p,
                            ssize_t res_offset,
                            const std::vector<sycl::event> &depends = {})
{
    using PowHS = hyperparam_detail::PowContigHyperparameterSet<argTy1, argTy2>;
    static constexpr std::uint8_t vec_sz = PowHS::vec_sz;
    static constexpr std::uint8_t n_vecs = PowHS::n_vecs;

    return elementwise_common::binary_contig_impl<
        argTy1, argTy2, PowOutputType, PowContigFunctor, pow_contig_kernel,
        vec_sz, n_vecs>(exec_q, nelems, arg1_p, arg1_offset, arg2_p,
                        arg2_offset, res_p, res_offset, depends);
}

template <typename fnT, typename T1, typename T2>
struct PowContigFactory
{
    fnT get()
    {
        if constexpr (!PowOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = pow_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2>
struct PowTypeMapFactory
{
    /*! @brief get typeid for output type of std::pow(T1 x, T2 y) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename PowOutputType<T1, T2>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename resT, typename IndexerT>
class pow_strided_kernel;

template <typename argTy1, typename argTy2>
sycl::event pow_strided_impl(sycl::queue &exec_q,
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
        argTy1, argTy2, PowOutputType, PowStridedFunctor, pow_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg1_p, arg1_offset, arg2_p,
        arg2_offset, res_p, res_offset, depends, additional_depends);
}

template <typename fnT, typename T1, typename T2>
struct PowStridedFactory
{
    fnT get()
    {
        if constexpr (!PowOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = pow_strided_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename argT, typename resT>
struct PowInplaceFunctor
{

    using supports_sg_loadstore = std::negation<
        std::disjunction<tu_ns::is_complex<argT>, tu_ns::is_complex<resT>>>;
    using supports_vec = std::negation<
        std::disjunction<tu_ns::is_complex<argT>, tu_ns::is_complex<resT>>>;

    void operator()(resT &res, const argT &in)
    {
        if constexpr (std::is_integral_v<argT> || std::is_integral_v<resT>) {
            auto tmp1 = res;
            auto tmp2 = in;
            if constexpr (std::is_signed_v<argT>) {
                if (tmp2 < 0) {
                    // invalid; return 0
                    res = 0;
                    return;
                }
            }
            if (tmp1 == 1) {
                return;
            }
            if (tmp2 == 0) {
                res = 1;
                return;
            }
            resT res_tmp = 1;
            while (tmp2 > 0) {
                if (tmp2 & 1) {
                    res_tmp *= tmp1;
                }
                tmp2 >>= 1;
                tmp1 *= tmp1;
            }
            res = res_tmp;
        }
        else if constexpr (tu_ns::is_complex<argT>::value &&
                           tu_ns::is_complex<resT>::value)
        {
            using r_resT = typename resT::value_type;
            using r_argT = typename argT::value_type;

            res = exprm_ns::pow(exprm_ns::complex<r_resT>(res),
                                exprm_ns::complex<r_argT>(in));
        }
        else {
            res = sycl::pow(res, in);
        }
        return;
    }

    template <int vec_sz>
    void operator()(sycl::vec<resT, vec_sz> &res,
                    const sycl::vec<argT, vec_sz> &in)
    {
        if constexpr (std::is_integral_v<argT> || std::is_integral_v<resT>) {
#pragma unroll
            for (int i = 0; i < vec_sz; ++i) {
                auto tmp1 = res[i];
                auto tmp2 = in[i];
                if constexpr (std::is_signed_v<argT>) {
                    if (tmp2 < 0) {
                        // invalid; return 0
                        res[i] = 0;
                        continue;
                    }
                }
                if (tmp1 == 1) {
                    continue;
                }
                if (tmp2 == 0) {
                    res[i] = 1;
                    continue;
                }
                resT res_tmp = 1;
                while (tmp2 > 0) {
                    if (tmp2 & 1) {
                        res_tmp *= tmp1;
                    }
                    tmp2 >>= 1;
                    tmp1 *= tmp1;
                }
                res[i] = res_tmp;
            }
        }
        else {
            res = sycl::pow(res, in);
        }
    }
};

template <typename argT,
          typename resT,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
using PowInplaceContigFunctor = elementwise_common::BinaryInplaceContigFunctor<
    argT,
    resT,
    PowInplaceFunctor<argT, resT>,
    vec_sz,
    n_vecs,
    enable_sg_loadstore>;

template <typename argT, typename resT, typename IndexerT>
using PowInplaceStridedFunctor =
    elementwise_common::BinaryInplaceStridedFunctor<
        argT,
        resT,
        IndexerT,
        PowInplaceFunctor<argT, resT>>;

template <typename argT,
          typename resT,
          std::uint8_t vec_sz,
          std::uint8_t n_vecs>
class pow_inplace_contig_kernel;

/* @brief Types supported by in-place pow */
template <typename argTy, typename resTy>
struct PowInplaceTypePairSupport
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
        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<float>,
                                    resTy,
                                    std::complex<float>>,
        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<double>,
                                    resTy,
                                    std::complex<double>>,
        // fall-through
        td_ns::NotDefinedEntry>::is_defined;
};

template <typename fnT, typename argT, typename resT>
struct PowInplaceTypeMapFactory
{
    /*! @brief get typeid for output type of x **= y */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        if constexpr (PowInplaceTypePairSupport<argT, resT>::is_defined) {
            return td_ns::GetTypeid<resT>{}.get();
        }
        else {
            return td_ns::GetTypeid<void>{}.get();
        }
    }
};

template <typename argTy, typename resTy>
sycl::event
    pow_inplace_contig_impl(sycl::queue &exec_q,
                            std::size_t nelems,
                            const char *arg_p,
                            ssize_t arg_offset,
                            char *res_p,
                            ssize_t res_offset,
                            const std::vector<sycl::event> &depends = {})
{
    using PowHS = hyperparam_detail::PowContigHyperparameterSet<resTy, argTy>;
    static constexpr std::uint8_t vec_sz = PowHS::vec_sz;
    static constexpr std::uint8_t n_vecs = PowHS::n_vecs;

    return elementwise_common::binary_inplace_contig_impl<
        argTy, resTy, PowInplaceContigFunctor, pow_inplace_contig_kernel,
        vec_sz, n_vecs>(exec_q, nelems, arg_p, arg_offset, res_p, res_offset,
                        depends);
}

template <typename fnT, typename T1, typename T2>
struct PowInplaceContigFactory
{
    fnT get()
    {
        if constexpr (!PowInplaceTypePairSupport<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = pow_inplace_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename resT, typename argT, typename IndexerT>
class pow_inplace_strided_kernel;

template <typename argTy, typename resTy>
sycl::event
    pow_inplace_strided_impl(sycl::queue &exec_q,
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
        argTy, resTy, PowInplaceStridedFunctor, pow_inplace_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T1, typename T2>
struct PowInplaceStridedFactory
{
    fnT get()
    {
        if constexpr (!PowInplaceTypePairSupport<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = pow_inplace_strided_impl<T1, T2>;
            return fn;
        }
    }
};

} // namespace dpctl::tensor::kernels::pow
