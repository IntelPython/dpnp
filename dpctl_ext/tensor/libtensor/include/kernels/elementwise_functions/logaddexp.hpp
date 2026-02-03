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
/// This file defines kernels for elementwise evaluation of LOGADDEXP(x1, x2)
/// function.
//===---------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>

#include "common.hpp"
#include "vec_size_util.hpp"

#include "utils/math_utils.hpp"
#include "utils/offset_utils.hpp"
#include "utils/type_dispatch_building.hpp"
#include "utils/type_utils.hpp"

#include "kernels/dpctl_tensor_types.hpp"

namespace dpctl::tensor::kernels::logaddexp
{
using dpctl::tensor::ssize_t;
namespace td_ns = dpctl::tensor::type_dispatch;
namespace tu_ns = dpctl::tensor::type_utils;

using dpctl::tensor::type_utils::is_complex;
using dpctl::tensor::type_utils::vec_cast;

template <typename argT1, typename argT2, typename resT>
struct LogAddExpFunctor
{
    using supports_sg_loadstore = std::true_type;
    using supports_vec = std::true_type;

    resT operator()(const argT1 &in1, const argT2 &in2) const
    {
        using dpctl::tensor::math_utils::logaddexp;
        return logaddexp<resT>(in1, in2);
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz>
        operator()(const sycl::vec<argT1, vec_sz> &in1,
                   const sycl::vec<argT2, vec_sz> &in2) const
    {
        sycl::vec<resT, vec_sz> res;
        auto diff = in1 - in2; // take advantange of faster vec arithmetic

#pragma unroll
        for (int i = 0; i < vec_sz; ++i) {
            if (std::isfinite(diff[i])) {
                res[i] = std::max<resT>(in1[i], in2[i]) +
                         impl_finite<resT>(-sycl::fabs(diff[i]));
            }
            else {
                using dpctl::tensor::math_utils::logaddexp;
                res[i] = logaddexp<resT>(in1[i], in2[i]);
            }
        }

        return res;
    }

private:
    template <typename T>
    T impl_finite(T const &in) const
    {
        return (in > 0) ? (in + sycl::log1p(sycl::exp(-in)))
                        : sycl::log1p(sycl::exp(in));
    }
};

template <typename argT1,
          typename argT2,
          typename resT,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
using LogAddExpContigFunctor = elementwise_common::BinaryContigFunctor<
    argT1,
    argT2,
    resT,
    LogAddExpFunctor<argT1, argT2, resT>,
    vec_sz,
    n_vecs,
    enable_sg_loadstore>;

template <typename argT1, typename argT2, typename resT, typename IndexerT>
using LogAddExpStridedFunctor = elementwise_common::BinaryStridedFunctor<
    argT1,
    argT2,
    resT,
    IndexerT,
    LogAddExpFunctor<argT1, argT2, resT>>;

template <typename T1, typename T2>
struct LogAddExpOutputType
{
    using value_type = typename std::disjunction<
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
struct LogAddExpContigHyperparameterSet
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
class logaddexp_contig_kernel;

template <typename argTy1, typename argTy2>
sycl::event logaddexp_contig_impl(sycl::queue &exec_q,
                                  std::size_t nelems,
                                  const char *arg1_p,
                                  ssize_t arg1_offset,
                                  const char *arg2_p,
                                  ssize_t arg2_offset,
                                  char *res_p,
                                  ssize_t res_offset,
                                  const std::vector<sycl::event> &depends = {})
{
    using LogAddExpHS =
        hyperparam_detail::LogAddExpContigHyperparameterSet<argTy1, argTy2>;
    static constexpr std::uint8_t vec_sz = LogAddExpHS::vec_sz;
    static constexpr std::uint8_t n_vecs = LogAddExpHS::n_vecs;

    return elementwise_common::binary_contig_impl<
        argTy1, argTy2, LogAddExpOutputType, LogAddExpContigFunctor,
        logaddexp_contig_kernel, vec_sz, n_vecs>(
        exec_q, nelems, arg1_p, arg1_offset, arg2_p, arg2_offset, res_p,
        res_offset, depends);
}

template <typename fnT, typename T1, typename T2>
struct LogAddExpContigFactory
{
    fnT get()
    {
        if constexpr (!LogAddExpOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = logaddexp_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2>
struct LogAddExpTypeMapFactory
{
    /*! @brief get typeid for output type of logaddexp(T1 x, T2 y) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename LogAddExpOutputType<T1, T2>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename resT, typename IndexerT>
class logaddexp_strided_kernel;

template <typename argTy1, typename argTy2>
sycl::event
    logaddexp_strided_impl(sycl::queue &exec_q,
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
        argTy1, argTy2, LogAddExpOutputType, LogAddExpStridedFunctor,
        logaddexp_strided_kernel>(exec_q, nelems, nd, shape_and_strides, arg1_p,
                                  arg1_offset, arg2_p, arg2_offset, res_p,
                                  res_offset, depends, additional_depends);
}

template <typename fnT, typename T1, typename T2>
struct LogAddExpStridedFactory
{
    fnT get()
    {
        if constexpr (!LogAddExpOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = logaddexp_strided_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename argT1, typename argT2, typename resT>
class logaddexp_matrix_row_broadcast_sg_krn;

} // namespace dpctl::tensor::kernels::logaddexp
