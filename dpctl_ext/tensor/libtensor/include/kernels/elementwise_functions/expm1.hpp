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
/// \file
/// This file defines kernels for elementwise evaluation of EXPM1(x) function.
//===---------------------------------------------------------------------===//

#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <sycl/sycl.hpp>

#include "vec_size_util.hpp"

#include "kernels/dpctl_tensor_types.hpp"
#include "kernels/elementwise_functions/common.hpp"

#include "utils/offset_utils.hpp"
#include "utils/type_dispatch_building.hpp"
#include "utils/type_utils.hpp"

namespace dpctl::tensor::kernels::expm1
{

using dpctl::tensor::ssize_t;
namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;

template <typename argT, typename resT>
struct Expm1Functor
{

    // is function constant for given argT
    using is_constant = typename std::false_type;
    // constant value, if constant
    // constexpr resT constant_value = resT{};
    // is function defined for sycl::vec
    using supports_vec = typename std::false_type;
    // do both argTy and resTy support sugroup store/load operation
    using supports_sg_loadstore = typename std::negation<
        std::disjunction<is_complex<resT>, is_complex<argT>>>;

    resT operator()(const argT &in) const
    {
        if constexpr (is_complex<argT>::value) {
            using realT = typename argT::value_type;
            // expm1(x + I*y) = expm1(x)*cos(y) - 2*sin(y / 2)^2 +
            // I*exp(x)*sin(y)
            const realT x = std::real(in);
            const realT y = std::imag(in);

            // special cases
            if (std::isinf(x)) {
                if (x > realT(0)) {
                    // positive infinity cases
                    if (!std::isfinite(y)) {
                        return resT{x, std::numeric_limits<realT>::quiet_NaN()};
                    }
                    else if (y == realT(0)) {
                        return in;
                    }
                    else {
                        return (resT{sycl::copysign(x, sycl::cos(y)),
                                     sycl::copysign(x, sycl::sin(y))});
                    }
                }
                else {
                    // negative infinity cases
                    if (!std::isfinite(y)) {
                        // copy sign of y to guarantee
                        // conj(expm1(x)) == expm1(conj(x))
                        return resT{realT(-1), sycl::copysign(realT(0), y)};
                    }
                    else {
                        return resT{realT(-1),
                                    sycl::copysign(realT(0), sycl::sin(y))};
                    }
                }
            }

            if (std::isnan(x)) {
                if (y == realT(0)) {
                    return in;
                }
                else {
                    return resT{std::numeric_limits<realT>::quiet_NaN(),
                                std::numeric_limits<realT>::quiet_NaN()};
                }
            }

            // x, y finite numbers
            const realT cosY_val = sycl::cos(y);
            const realT sinY_val = (y == 0) ? y : sycl::sin(y);
            const realT sinhalfY_val = (y == 0) ? y : sycl::sin(y / 2);

            const realT res_re =
                sycl::expm1(x) * cosY_val - 2 * sinhalfY_val * sinhalfY_val;
            realT res_im = sycl::exp(x) * sinY_val;
            return resT{res_re, res_im};
        }
        else {
            static_assert(std::is_floating_point_v<argT> ||
                          std::is_same_v<argT, sycl::half>);
            static_assert(std::is_same_v<argT, resT>);
            if (in == 0) {
                return in;
            }
            return sycl::expm1(in);
        }
    }
};

template <typename argTy,
          typename resTy = argTy,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
using Expm1ContigFunctor =
    elementwise_common::UnaryContigFunctor<argTy,
                                           resTy,
                                           Expm1Functor<argTy, resTy>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using Expm1StridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, Expm1Functor<argTy, resTy>>;

template <typename T>
struct Expm1OutputType
{
    using value_type = typename std::disjunction<
        td_ns::TypeMapResultEntry<T, sycl::half, sycl::half>,
        td_ns::TypeMapResultEntry<T, float, float>,
        td_ns::TypeMapResultEntry<T, double, double>,
        td_ns::TypeMapResultEntry<T, std::complex<float>, std::complex<float>>,
        td_ns::
            TypeMapResultEntry<T, std::complex<double>, std::complex<double>>,
        td_ns::DefaultResultEntry<void>>::result_type;

    static constexpr bool is_defined = !std::is_same_v<value_type, void>;
};

namespace hyperparam_detail
{

namespace vsu_ns = dpctl::tensor::kernels::vec_size_utils;

using vsu_ns::ContigHyperparameterSetDefault;
using vsu_ns::UnaryContigHyperparameterSetEntry;

template <typename argTy>
struct Expm1ContigHyperparameterSet
{
    using value_type =
        typename std::disjunction<ContigHyperparameterSetDefault<4u, 2u>>;

    constexpr static auto vec_sz = value_type::vec_sz;
    constexpr static auto n_vecs = value_type::n_vecs;
};

} // end of namespace hyperparam_detail

template <typename T1, typename T2, std::uint8_t vec_sz, std::uint8_t n_vecs>
class expm1_contig_kernel;

template <typename argTy>
sycl::event expm1_contig_impl(sycl::queue &exec_q,
                              std::size_t nelems,
                              const char *arg_p,
                              char *res_p,
                              const std::vector<sycl::event> &depends = {})
{
    using Expm1HS = hyperparam_detail::Expm1ContigHyperparameterSet<argTy>;
    static constexpr std::uint8_t vec_sz = Expm1HS::vec_sz;
    static constexpr std::uint8_t n_vecs = Expm1HS::n_vecs;

    return elementwise_common::unary_contig_impl<
        argTy, Expm1OutputType, Expm1ContigFunctor, expm1_contig_kernel, vec_sz,
        n_vecs>(exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T>
struct Expm1ContigFactory
{
    fnT get()
    {
        if constexpr (!Expm1OutputType<T>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = expm1_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T>
struct Expm1TypeMapFactory
{
    /*! @brief get typeid for output type of sycl::expm1(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename Expm1OutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3>
class expm1_strided_kernel;

template <typename argTy>
sycl::event
    expm1_strided_impl(sycl::queue &exec_q,
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
        argTy, Expm1OutputType, Expm1StridedFunctor, expm1_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T>
struct Expm1StridedFactory
{
    fnT get()
    {
        if constexpr (!Expm1OutputType<T>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = expm1_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace dpctl::tensor::kernels::expm1
