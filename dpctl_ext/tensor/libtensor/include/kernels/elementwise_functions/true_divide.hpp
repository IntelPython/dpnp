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
/// This file defines kernels for elementwise evaluation of DIVIDE(x1, x2)
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

#include "kernels/dpctl_tensor_types.hpp"
#include "kernels/elementwise_functions/common.hpp"
#include "kernels/elementwise_functions/common_inplace.hpp"

#include "utils/type_dispatch_building.hpp"
#include "utils/type_utils.hpp"

namespace dpctl::tensor::kernels::true_divide
{

using dpctl::tensor::ssize_t;
namespace td_ns = dpctl::tensor::type_dispatch;
namespace tu_ns = dpctl::tensor::type_utils;

template <typename argT1, typename argT2, typename resT>
struct TrueDivideFunctor
{

    using supports_sg_loadstore = std::negation<
        std::disjunction<tu_ns::is_complex<argT1>, tu_ns::is_complex<argT2>>>;
    using supports_vec = std::negation<
        std::disjunction<tu_ns::is_complex<argT1>, tu_ns::is_complex<argT2>>>;

    resT operator()(const argT1 &in1, const argT2 &in2) const
    {
        if constexpr (tu_ns::is_complex<argT1>::value &&
                      tu_ns::is_complex<argT2>::value)
        {
            using realT1 = typename argT1::value_type;
            using realT2 = typename argT2::value_type;

            return exprm_ns::complex<realT1>(in1) /
                   exprm_ns::complex<realT2>(in2);
        }
        else if constexpr (tu_ns::is_complex<argT1>::value &&
                           !tu_ns::is_complex<argT2>::value)
        {
            using realT1 = typename argT1::value_type;

            return exprm_ns::complex<realT1>(in1) / in2;
        }
        else if constexpr (!tu_ns::is_complex<argT1>::value &&
                           tu_ns::is_complex<argT2>::value)
        {
            using realT2 = typename argT2::value_type;

            return in1 / exprm_ns::complex<realT2>(in2);
        }
        else {
            return in1 / in2;
        }
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz>
        operator()(const sycl::vec<argT1, vec_sz> &in1,
                   const sycl::vec<argT2, vec_sz> &in2) const
    {
        auto tmp = in1 / in2;
        if constexpr (std::is_same_v<resT,
                                     typename decltype(tmp)::element_type>) {
            return tmp;
        }
        else {
            using dpctl::tensor::type_utils::vec_cast;

            return vec_cast<resT, typename decltype(tmp)::element_type, vec_sz>(
                tmp);
        }
    }
};

template <typename argT1,
          typename argT2,
          typename resT,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
using TrueDivideContigFunctor = elementwise_common::BinaryContigFunctor<
    argT1,
    argT2,
    resT,
    TrueDivideFunctor<argT1, argT2, resT>,
    vec_sz,
    n_vecs,
    enable_sg_loadstore>;

template <typename argT1, typename argT2, typename resT, typename IndexerT>
using TrueDivideStridedFunctor = elementwise_common::BinaryStridedFunctor<
    argT1,
    argT2,
    resT,
    IndexerT,
    TrueDivideFunctor<argT1, argT2, resT>>;

template <typename T1, typename T2>
struct TrueDivideOutputType
{
    using value_type = typename std::disjunction<
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
                                        std::complex<float>,
                                        T2,
                                        float,
                                        std::complex<float>>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        float,
                                        T2,
                                        std::complex<float>,
                                        std::complex<float>>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::complex<double>,
                                        T2,
                                        std::complex<double>,
                                        std::complex<double>>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        double,
                                        T2,
                                        std::complex<double>,
                                        std::complex<double>>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::complex<double>,
                                        T2,
                                        double,
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
struct TrueDivideContigHyperparameterSet
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
class true_divide_contig_kernel;

template <typename argTy1, typename argTy2>
sycl::event
    true_divide_contig_impl(sycl::queue &exec_q,
                            std::size_t nelems,
                            const char *arg1_p,
                            ssize_t arg1_offset,
                            const char *arg2_p,
                            ssize_t arg2_offset,
                            char *res_p,
                            ssize_t res_offset,
                            const std::vector<sycl::event> &depends = {})
{
    using DivHS =
        hyperparam_detail::TrueDivideContigHyperparameterSet<argTy1, argTy2>;
    static constexpr std::uint8_t vec_sz = DivHS::vec_sz;
    static constexpr std::uint8_t n_vecs = DivHS::n_vecs;

    return elementwise_common::binary_contig_impl<
        argTy1, argTy2, TrueDivideOutputType, TrueDivideContigFunctor,
        true_divide_contig_kernel, vec_sz, n_vecs>(
        exec_q, nelems, arg1_p, arg1_offset, arg2_p, arg2_offset, res_p,
        res_offset, depends);
}

template <typename fnT, typename T1, typename T2>
struct TrueDivideContigFactory
{
    fnT get()
    {
        if constexpr (!TrueDivideOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = true_divide_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2>
struct TrueDivideTypeMapFactory
{
    /*! @brief get typeid for output type of divide(T1 x, T2 y) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename TrueDivideOutputType<T1, T2>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename resT, typename IndexerT>
class true_divide_strided_kernel;

template <typename argTy1, typename argTy2>
sycl::event
    true_divide_strided_impl(sycl::queue &exec_q,
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
        argTy1, argTy2, TrueDivideOutputType, TrueDivideStridedFunctor,
        true_divide_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg1_p, arg1_offset, arg2_p,
        arg2_offset, res_p, res_offset, depends, additional_depends);
}

template <typename fnT, typename T1, typename T2>
struct TrueDivideStridedFactory
{
    fnT get()
    {
        if constexpr (!TrueDivideOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = true_divide_strided_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename argT1, typename argT2, typename resT>
using TrueDivideContigMatrixContigRowBroadcastingFunctor =
    elementwise_common::BinaryContigMatrixContigRowBroadcastingFunctor<
        argT1,
        argT2,
        resT,
        TrueDivideFunctor<argT1, argT2, resT>>;

template <typename argT1, typename argT2, typename resT>
using TrueDivideContigRowContigMatrixBroadcastingFunctor =
    elementwise_common::BinaryContigRowContigMatrixBroadcastingFunctor<
        argT1,
        argT2,
        resT,
        TrueDivideFunctor<argT1, argT2, resT>>;

template <typename argT1, typename argT2, typename resT>
class true_divide_matrix_row_broadcast_sg_krn;

template <typename argT1, typename argT2, typename resT>
class true_divide_row_matrix_broadcast_sg_krn;

template <typename argT1, typename argT2, typename resT>
sycl::event true_divide_contig_matrix_contig_row_broadcast_impl(
    sycl::queue &exec_q,
    std::vector<sycl::event> &host_tasks,
    std::size_t n0,
    std::size_t n1,
    const char *mat_p, // typeless pointer to (n0, n1) C-contiguous matrix
    ssize_t mat_offset,
    const char *vec_p, // typeless pointer to (n1,) contiguous row
    ssize_t vec_offset,
    char *res_p, // typeless pointer to (n0, n1) result C-contig. matrix,
                 //    res[i,j] = mat[i,j] / vec[j]
    ssize_t res_offset,
    const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::binary_contig_matrix_contig_row_broadcast_impl<
        argT1, argT2, resT, TrueDivideContigMatrixContigRowBroadcastingFunctor,
        true_divide_matrix_row_broadcast_sg_krn>(
        exec_q, host_tasks, n0, n1, mat_p, mat_offset, vec_p, vec_offset, res_p,
        res_offset, depends);
}

template <typename fnT, typename T1, typename T2>
struct TrueDivideContigMatrixContigRowBroadcastFactory
{
    fnT get()
    {
        if constexpr (!TrueDivideOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using resT = typename TrueDivideOutputType<T1, T2>::value_type;
            if constexpr (dpctl::tensor::type_utils::is_complex<T1>::value ||
                          dpctl::tensor::type_utils::is_complex<T2>::value ||
                          dpctl::tensor::type_utils::is_complex<resT>::value)
            {
                fnT fn = nullptr;
                return fn;
            }
            else {
                fnT fn =
                    true_divide_contig_matrix_contig_row_broadcast_impl<T1, T2,
                                                                        resT>;
                return fn;
            }
        }
    }
};

template <typename argT1, typename argT2, typename resT>
sycl::event true_divide_contig_row_contig_matrix_broadcast_impl(
    sycl::queue &exec_q,
    std::vector<sycl::event> &host_tasks,
    std::size_t n0,
    std::size_t n1,
    const char *vec_p, // typeless pointer to (n1,) contiguous row
    ssize_t vec_offset,
    const char *mat_p, // typeless pointer to (n0, n1) C-contiguous matrix
    ssize_t mat_offset,
    char *res_p, // typeless pointer to (n0, n1) result C-contig. matrix,
                 //    res[i,j] = mat[i,j] + vec[j]
    ssize_t res_offset,
    const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::binary_contig_row_contig_matrix_broadcast_impl<
        argT1, argT2, resT, TrueDivideContigRowContigMatrixBroadcastingFunctor,
        true_divide_row_matrix_broadcast_sg_krn>(
        exec_q, host_tasks, n0, n1, vec_p, vec_offset, mat_p, mat_offset, res_p,
        res_offset, depends);
};

template <typename fnT, typename T1, typename T2>
struct TrueDivideContigRowContigMatrixBroadcastFactory
{
    fnT get()
    {
        if constexpr (!TrueDivideOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using resT = typename TrueDivideOutputType<T1, T2>::value_type;
            if constexpr (dpctl::tensor::type_utils::is_complex<T1>::value ||
                          dpctl::tensor::type_utils::is_complex<T2>::value ||
                          dpctl::tensor::type_utils::is_complex<resT>::value)
            {
                fnT fn = nullptr;
                return fn;
            }
            else {
                fnT fn =
                    true_divide_contig_row_contig_matrix_broadcast_impl<T1, T2,
                                                                        resT>;
                return fn;
            }
        }
    }
};

template <typename argT, typename resT>
struct TrueDivideInplaceFunctor
{

    using supports_sg_loadstore = std::negation<
        std::disjunction<tu_ns::is_complex<argT>, tu_ns::is_complex<resT>>>;
    using supports_vec = std::negation<
        std::disjunction<tu_ns::is_complex<argT>, tu_ns::is_complex<resT>>>;

    void operator()(resT &res, const argT &in)
    {
        if constexpr (tu_ns::is_complex<resT>::value) {
            if constexpr (tu_ns::is_complex<argT>::value) {
                using res_rT = typename resT::value_type;
                using arg_rT = typename argT::value_type;

                auto res1 = exprm_ns::complex<res_rT>(res);
                res1 /= exprm_ns::complex<arg_rT>(in);
                res = res1;
            }
            else {
                using res_rT = typename resT::value_type;

                auto res1 = exprm_ns::complex<res_rT>(res);
                res1 /= in;
                res = res1;
            }
        }
        else {
            res /= in;
        }
    }

    template <int vec_sz>
    void operator()(sycl::vec<resT, vec_sz> &res,
                    const sycl::vec<argT, vec_sz> &in)
    {
        res /= in;
    }
};

/* @brief Types supported by in-place divide */
template <typename argTy, typename resTy>
struct TrueDivideInplaceTypePairSupport
{

    /* value if true a kernel for <argTy, resTy> must be instantiated  */
    static constexpr bool is_defined = std::disjunction<
        td_ns::TypePairDefinedEntry<argTy, sycl::half, resTy, sycl::half>,
        td_ns::TypePairDefinedEntry<argTy, float, resTy, float>,
        td_ns::TypePairDefinedEntry<argTy, double, resTy, double>,
        td_ns::TypePairDefinedEntry<argTy, float, resTy, std::complex<float>>,
        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<float>,
                                    resTy,
                                    std::complex<float>>,
        td_ns::TypePairDefinedEntry<argTy, double, resTy, std::complex<double>>,
        td_ns::TypePairDefinedEntry<argTy,
                                    std::complex<double>,
                                    resTy,
                                    std::complex<double>>,
        // fall-through
        td_ns::NotDefinedEntry>::is_defined;
};

template <typename fnT, typename argT, typename resT>
struct TrueDivideInplaceTypeMapFactory
{
    /*! @brief get typeid for output type of divide(T1 x, T2 y) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        if constexpr (TrueDivideInplaceTypePairSupport<argT, resT>::is_defined)
        {
            return td_ns::GetTypeid<resT>{}.get();
        }
        else {
            return td_ns::GetTypeid<void>{}.get();
        }
    }
};

template <typename argT,
          typename resT,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
using TrueDivideInplaceContigFunctor =
    elementwise_common::BinaryInplaceContigFunctor<
        argT,
        resT,
        TrueDivideInplaceFunctor<argT, resT>,
        vec_sz,
        n_vecs,
        enable_sg_loadstore>;

template <typename argT, typename resT, typename IndexerT>
using TrueDivideInplaceStridedFunctor =
    elementwise_common::BinaryInplaceStridedFunctor<
        argT,
        resT,
        IndexerT,
        TrueDivideInplaceFunctor<argT, resT>>;

template <typename argT,
          typename resT,
          std::uint8_t vec_sz,
          std::uint8_t n_vecs>
class true_divide_inplace_contig_kernel;

template <typename argTy, typename resTy>
sycl::event true_divide_inplace_contig_impl(
    sycl::queue &exec_q,
    std::size_t nelems,
    const char *arg_p,
    ssize_t arg_offset,
    char *res_p,
    ssize_t res_offset,
    const std::vector<sycl::event> &depends = {})
{
    using DivHS =
        hyperparam_detail::TrueDivideContigHyperparameterSet<resTy, argTy>;
    static constexpr std::uint8_t vec_sz = DivHS::vec_sz;
    static constexpr std::uint8_t n_vecs = DivHS::vec_sz;

    return elementwise_common::binary_inplace_contig_impl<
        argTy, resTy, TrueDivideInplaceContigFunctor,
        true_divide_inplace_contig_kernel, vec_sz, n_vecs>(
        exec_q, nelems, arg_p, arg_offset, res_p, res_offset, depends);
}

template <typename fnT, typename T1, typename T2>
struct TrueDivideInplaceContigFactory
{
    fnT get()
    {
        if constexpr (!TrueDivideInplaceTypePairSupport<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = true_divide_inplace_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename resT, typename argT, typename IndexerT>
class true_divide_inplace_strided_kernel;

template <typename argTy, typename resTy>
sycl::event true_divide_inplace_strided_impl(
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
        argTy, resTy, TrueDivideInplaceStridedFunctor,
        true_divide_inplace_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T1, typename T2>
struct TrueDivideInplaceStridedFactory
{
    fnT get()
    {
        if constexpr (!TrueDivideInplaceTypePairSupport<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = true_divide_inplace_strided_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename argT, typename resT>
class true_divide_inplace_row_matrix_broadcast_sg_krn;

template <typename argT, typename resT>
using TrueDivideInplaceRowMatrixBroadcastingFunctor =
    elementwise_common::BinaryInplaceRowMatrixBroadcastingFunctor<
        argT,
        resT,
        TrueDivideInplaceFunctor<argT, resT>>;

template <typename argT, typename resT>
sycl::event true_divide_inplace_row_matrix_broadcast_impl(
    sycl::queue &exec_q,
    std::vector<sycl::event> &host_tasks,
    std::size_t n0,
    std::size_t n1,
    const char *vec_p, // typeless pointer to (n1,) contiguous row
    ssize_t vec_offset,
    char *mat_p, // typeless pointer to (n0, n1) C-contiguous matrix
    ssize_t mat_offset,
    const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::binary_inplace_row_matrix_broadcast_impl<
        argT, resT, TrueDivideInplaceRowMatrixBroadcastingFunctor,
        true_divide_inplace_row_matrix_broadcast_sg_krn>(
        exec_q, host_tasks, n0, n1, vec_p, vec_offset, mat_p, mat_offset,
        depends);
}

template <typename fnT, typename T1, typename T2>
struct TrueDivideInplaceRowMatrixBroadcastFactory
{
    fnT get()
    {
        if constexpr (!TrueDivideInplaceTypePairSupport<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            if constexpr (dpctl::tensor::type_utils::is_complex<T1>::value ||
                          dpctl::tensor::type_utils::is_complex<T2>::value)
            {
                fnT fn = nullptr;
                return fn;
            }
            else {
                fnT fn = true_divide_inplace_row_matrix_broadcast_impl<T1, T2>;
                return fn;
            }
        }
    }
};

} // namespace dpctl::tensor::kernels::true_divide
