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
/// This file defines kernels for elementwise evaluation of inequality of
/// tensor elements.
//===---------------------------------------------------------------------===//

#pragma once
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>

#include "vec_size_util.hpp"

#include "utils/offset_utils.hpp"
#include "utils/type_dispatch_building.hpp"
#include "utils/type_utils.hpp"

#include "kernels/dpctl_tensor_types.hpp"
#include "kernels/elementwise_functions/common.hpp"

namespace dpctl::tensor::kernels::not_equal
{

using dpctl::tensor::ssize_t;
namespace td_ns = dpctl::tensor::type_dispatch;
namespace tu_ns = dpctl::tensor::type_utils;

template <typename argT1, typename argT2, typename resT>
struct NotEqualFunctor
{
    static_assert(std::is_same_v<resT, bool>);

    using supports_sg_loadstore = std::negation<
        std::disjunction<tu_ns::is_complex<argT1>, tu_ns::is_complex<argT2>>>;
    using supports_vec = std::conjunction<
        std::is_same<argT1, argT2>,
        std::negation<std::disjunction<tu_ns::is_complex<argT1>,
                                       tu_ns::is_complex<argT2>>>>;

    resT operator()(const argT1 &in1, const argT2 &in2) const
    {
        if constexpr (std::is_integral_v<argT1> && std::is_integral_v<argT2> &&
                      std::is_signed_v<argT1> != std::is_signed_v<argT2>)
        {
            if constexpr (std::is_signed_v<argT1> && !std::is_signed_v<argT2>) {
                return (in1 < 0) ? true : (static_cast<argT2>(in1) != in2);
            }
            else {
                if constexpr (!std::is_signed_v<argT1> &&
                              std::is_signed_v<argT2>)
                {
                    return (in2 < 0) ? true : (in1 != static_cast<argT1>(in2));
                }
            }
        }
        else {
            return (in1 != in2);
        }
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz>
        operator()(const sycl::vec<argT1, vec_sz> &in1,
                   const sycl::vec<argT2, vec_sz> &in2) const
    {
        auto tmp = (in1 != in2);
        if constexpr (std::is_same_v<resT,
                                     typename decltype(tmp)::element_type>)
        {
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
using NotEqualContigFunctor =
    elementwise_common::BinaryContigFunctor<argT1,
                                            argT2,
                                            resT,
                                            NotEqualFunctor<argT1, argT2, resT>,
                                            vec_sz,
                                            n_vecs,
                                            enable_sg_loadstore>;

template <typename argT1, typename argT2, typename resT, typename IndexerT>
using NotEqualStridedFunctor = elementwise_common::BinaryStridedFunctor<
    argT1,
    argT2,
    resT,
    IndexerT,
    NotEqualFunctor<argT1, argT2, resT>>;

template <typename T1, typename T2>
struct NotEqualOutputType
{
    using value_type = typename std::disjunction<
        td_ns::BinaryTypeMapResultEntry<T1, bool, T2, bool, bool>,
        td_ns::
            BinaryTypeMapResultEntry<T1, std::uint8_t, T2, std::uint8_t, bool>,
        td_ns::BinaryTypeMapResultEntry<T1, std::int8_t, T2, std::int8_t, bool>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint16_t,
                                        T2,
                                        std::uint16_t,
                                        bool>,
        td_ns::
            BinaryTypeMapResultEntry<T1, std::int16_t, T2, std::int16_t, bool>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint32_t,
                                        T2,
                                        std::uint32_t,
                                        bool>,
        td_ns::
            BinaryTypeMapResultEntry<T1, std::int32_t, T2, std::int32_t, bool>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint64_t,
                                        T2,
                                        std::uint64_t,
                                        bool>,
        td_ns::
            BinaryTypeMapResultEntry<T1, std::int64_t, T2, std::int64_t, bool>,
        td_ns::
            BinaryTypeMapResultEntry<T1, std::uint64_t, T2, std::int64_t, bool>,
        td_ns::
            BinaryTypeMapResultEntry<T1, std::int64_t, T2, std::uint64_t, bool>,
        td_ns::BinaryTypeMapResultEntry<T1, sycl::half, T2, sycl::half, bool>,
        td_ns::BinaryTypeMapResultEntry<T1, float, T2, float, bool>,
        td_ns::BinaryTypeMapResultEntry<T1, double, T2, double, bool>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::complex<float>,
                                        T2,
                                        std::complex<float>,
                                        bool>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::complex<double>,
                                        T2,
                                        std::complex<double>,
                                        bool>,
        td_ns::DefaultResultEntry<void>>::result_type;

    static constexpr bool is_defined = !std::is_same_v<value_type, void>;
};

namespace hyperparam_detail
{

namespace vsu_ns = dpctl::tensor::kernels::vec_size_utils;

using vsu_ns::BinaryContigHyperparameterSetEntry;
using vsu_ns::ContigHyperparameterSetDefault;

template <typename argTy1, typename argTy2>
struct NotEqualContigHyperparameterSet
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
class not_equal_contig_kernel;

template <typename argTy1, typename argTy2>
sycl::event not_equal_contig_impl(sycl::queue &exec_q,
                                  std::size_t nelems,
                                  const char *arg1_p,
                                  ssize_t arg1_offset,
                                  const char *arg2_p,
                                  ssize_t arg2_offset,
                                  char *res_p,
                                  ssize_t res_offset,
                                  const std::vector<sycl::event> &depends = {})
{
    using NotEqHS =
        hyperparam_detail::NotEqualContigHyperparameterSet<argTy1, argTy2>;
    static constexpr std::uint8_t vec_sz = NotEqHS::vec_sz;
    static constexpr std::uint8_t n_vecs = NotEqHS::n_vecs;

    return elementwise_common::binary_contig_impl<
        argTy1, argTy2, NotEqualOutputType, NotEqualContigFunctor,
        not_equal_contig_kernel, vec_sz, n_vecs>(
        exec_q, nelems, arg1_p, arg1_offset, arg2_p, arg2_offset, res_p,
        res_offset, depends);
}

template <typename fnT, typename T1, typename T2>
struct NotEqualContigFactory
{
    fnT get()
    {
        if constexpr (!NotEqualOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = not_equal_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2>
struct NotEqualTypeMapFactory
{
    /*! @brief get typeid for output type of operator()!=(x, y), always bool */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename NotEqualOutputType<T1, T2>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename resT, typename IndexerT>
class not_equal_strided_kernel;

template <typename argTy1, typename argTy2>
sycl::event
    not_equal_strided_impl(sycl::queue &exec_q,
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
        argTy1, argTy2, NotEqualOutputType, NotEqualStridedFunctor,
        not_equal_strided_kernel>(exec_q, nelems, nd, shape_and_strides, arg1_p,
                                  arg1_offset, arg2_p, arg2_offset, res_p,
                                  res_offset, depends, additional_depends);
}

template <typename fnT, typename T1, typename T2>
struct NotEqualStridedFactory
{
    fnT get()
    {
        if constexpr (!NotEqualOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = not_equal_strided_impl<T1, T2>;
            return fn;
        }
    }
};

} // namespace dpctl::tensor::kernels::not_equal
