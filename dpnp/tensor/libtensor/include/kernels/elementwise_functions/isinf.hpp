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
/// This file defines kernels for elementwise evaluation of ISINF(x)
/// function that tests whether a tensor element is an infinity.
//===---------------------------------------------------------------------===//

#pragma once
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>

#include "vec_size_util.hpp"

#include "kernels/dpctl_tensor_types.hpp"
#include "kernels/elementwise_functions/common.hpp"

#include "utils/type_dispatch_building.hpp"
#include "utils/type_utils.hpp"

namespace dpctl::tensor::kernels::isinf
{

using dpctl::tensor::ssize_t;
namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;
using dpctl::tensor::type_utils::vec_cast;

template <typename argT, typename resT>
struct IsInfFunctor
{
    static_assert(std::is_same_v<resT, bool>);

    using is_constant = typename std::disjunction<std::is_same<argT, bool>,
                                                  std::is_integral<argT>>;
    static constexpr resT constant_value = false;
    using supports_vec =
        typename std::disjunction<std::is_same<argT, sycl::half>,
                                  std::is_floating_point<argT>>;
    using supports_sg_loadstore = typename std::negation<
        std::disjunction<is_complex<resT>, is_complex<argT>>>;

    resT operator()(const argT &in) const
    {
        if constexpr (is_complex<argT>::value) {
            const bool real_isinf = std::isinf(std::real(in));
            const bool imag_isinf = std::isinf(std::imag(in));
            return (real_isinf || imag_isinf);
        }
        else if constexpr (std::is_same<argT, bool>::value ||
                           std::is_integral<argT>::value) {
            return constant_value;
        }
        else if constexpr (std::is_same_v<argT, sycl::half>) {
            return sycl::isinf(in);
        }
        else {
            return std::isinf(in);
        }
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz> operator()(const sycl::vec<argT, vec_sz> &in) const
    {
        auto const &res_vec = sycl::isinf(in);

        using deducedT = typename std::remove_cv_t<
            std::remove_reference_t<decltype(res_vec)>>::element_type;

        return vec_cast<bool, deducedT, vec_sz>(res_vec);
    }
};

template <typename argT,
          typename resT = bool,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
using IsInfContigFunctor =
    elementwise_common::UnaryContigFunctor<argT,
                                           resT,
                                           IsInfFunctor<argT, resT>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using IsInfStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, IsInfFunctor<argTy, resTy>>;

template <typename argTy>
struct IsInfOutputType
{
    using value_type = bool;
};

namespace hyperparam_detail
{

namespace vsu_ns = dpctl::tensor::kernels::vec_size_utils;

using vsu_ns::ContigHyperparameterSetDefault;
using vsu_ns::UnaryContigHyperparameterSetEntry;

template <typename argTy>
struct IsInfContigHyperparameterSet
{
    using value_type =
        typename std::disjunction<ContigHyperparameterSetDefault<4u, 2u>>;

    constexpr static auto vec_sz = value_type::vec_sz;
    constexpr static auto n_vecs = value_type::n_vecs;
};

} // namespace hyperparam_detail

template <typename T1, typename T2, std::uint8_t vec_sz, std::uint8_t n_vecs>
class isinf_contig_kernel;

template <typename argTy>
sycl::event isinf_contig_impl(sycl::queue &exec_q,
                              std::size_t nelems,
                              const char *arg_p,
                              char *res_p,
                              const std::vector<sycl::event> &depends = {})
{
    using IsInfHS = hyperparam_detail::IsInfContigHyperparameterSet<argTy>;
    static constexpr std::uint8_t vec_sz = IsInfHS::vec_sz;
    static constexpr std::uint8_t n_vecs = IsInfHS::n_vecs;

    return elementwise_common::unary_contig_impl<
        argTy, IsInfOutputType, IsInfContigFunctor, isinf_contig_kernel, vec_sz,
        n_vecs>(exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T>
struct IsInfContigFactory
{
    fnT get()
    {
        fnT fn = isinf_contig_impl<T>;
        return fn;
    }
};

template <typename fnT, typename T>
struct IsInfTypeMapFactory
{
    /*! @brief get typeid for output type of sycl::isinf(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename IsInfOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3>
class isinf_strided_kernel;

template <typename argTy>
sycl::event
    isinf_strided_impl(sycl::queue &exec_q,
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
    return elementwise_common::unary_strided_impl<
        argTy, IsInfOutputType, IsInfStridedFunctor, isinf_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T>
struct IsInfStridedFactory
{
    fnT get()
    {
        fnT fn = isinf_strided_impl<T>;
        return fn;
    }
};

} // namespace dpctl::tensor::kernels::isinf
