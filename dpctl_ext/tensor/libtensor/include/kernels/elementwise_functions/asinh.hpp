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
/// This file defines kernels for elementwise evaluation of ASINH(x) function.
//===---------------------------------------------------------------------===//

#pragma once
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>

#include "sycl_complex.hpp"
#include "vec_size_util.hpp"

#include "kernels/dpctl_tensor_types.hpp"
#include "kernels/elementwise_functions/common.hpp"

#include "utils/offset_utils.hpp"
#include "utils/type_dispatch_building.hpp"
#include "utils/type_utils.hpp"

namespace dpctl::tensor::kernels::asinh
{

using dpctl::tensor::ssize_t;
namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;

template <typename argT, typename resT>
struct AsinhFunctor
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

            static constexpr realT q_nan =
                std::numeric_limits<realT>::quiet_NaN();

            const realT x = std::real(in);
            const realT y = std::imag(in);

            if (std::isnan(x)) {
                /* asinh(NaN + I*+-Inf) = opt(+-)Inf + I*NaN */
                if (std::isinf(y)) {
                    return resT{y, q_nan};
                }
                /* asinh(NaN + I*0) = NaN + I*0 */
                if (y == realT(0)) {
                    return resT{q_nan, y};
                }
                /* All other cases involving NaN return NaN + I*NaN. */
                return resT{q_nan, q_nan};
            }

            if (std::isnan(y)) {
                /* asinh(+-Inf + I*NaN) = +-Inf + I*NaN */
                if (std::isinf(x)) {
                    return resT{x, q_nan};
                }
                /* All other cases involving NaN return NaN + I*NaN. */
                return resT{q_nan, q_nan};
            }

            /*
             * For large x or y including asinh(+-Inf + I*+-Inf)
             * asinh(in) = sign(x)*log(sign(x)*in) + O(1/in^2)   as in ->
             * infinity The above formula works for the imaginary part as well,
             * because Im(asinh(in)) = sign(x)*atan2(sign(x)*y, fabs(x)) +
             * O(y/in^3) as in -> infinity, uniformly in y
             */
            static constexpr realT r_eps =
                realT(1) / std::numeric_limits<realT>::epsilon();

            if (sycl::fabs(x) > r_eps || sycl::fabs(y) > r_eps) {
                using sycl_complexT = exprm_ns::complex<realT>;
                sycl_complexT log_in = (sycl::signbit(x))
                                           ? exprm_ns::log(sycl_complexT(-in))
                                           : exprm_ns::log(sycl_complexT(in));
                realT wx = log_in.real() + sycl::log(realT(2));
                realT wy = log_in.imag();

                const realT res_re = sycl::copysign(wx, x);
                const realT res_im = sycl::copysign(wy, y);
                return resT{res_re, res_im};
            }

            /* ordinary cases */
            return exprm_ns::asinh(exprm_ns::complex<realT>(in)); // asinh(in);
        }
        else {
            static_assert(std::is_floating_point_v<argT> ||
                          std::is_same_v<argT, sycl::half>);
            return sycl::asinh(in);
        }
    }
};

template <typename argTy,
          typename resTy = argTy,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
using AsinhContigFunctor =
    elementwise_common::UnaryContigFunctor<argTy,
                                           resTy,
                                           AsinhFunctor<argTy, resTy>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using AsinhStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, AsinhFunctor<argTy, resTy>>;

template <typename T>
struct AsinhOutputType
{
    using value_type = typename std::disjunction<
        td_ns::TypeMapResultEntry<T, sycl::half>,
        td_ns::TypeMapResultEntry<T, float>,
        td_ns::TypeMapResultEntry<T, double>,
        td_ns::TypeMapResultEntry<T, std::complex<float>>,
        td_ns::TypeMapResultEntry<T, std::complex<double>>,
        td_ns::DefaultResultEntry<void>>::result_type;

    static constexpr bool is_defined = !std::is_same_v<value_type, void>;
};

namespace hyperparam_detail
{

namespace vsu_ns = dpctl::tensor::kernels::vec_size_utils;

using vsu_ns::ContigHyperparameterSetDefault;
using vsu_ns::UnaryContigHyperparameterSetEntry;

template <typename argTy>
struct AsinhContigHyperparameterSet
{
    using value_type =
        typename std::disjunction<ContigHyperparameterSetDefault<4u, 2u>>;

    constexpr static auto vec_sz = value_type::vec_sz;
    constexpr static auto n_vecs = value_type::n_vecs;
};

} // end of namespace hyperparam_detail

template <typename T1, typename T2, std::uint8_t vec_sz, std::uint8_t n_vecs>
class asinh_contig_kernel;

template <typename argTy>
sycl::event asinh_contig_impl(sycl::queue &exec_q,
                              std::size_t nelems,
                              const char *arg_p,
                              char *res_p,
                              const std::vector<sycl::event> &depends = {})
{
    using AsinhHS = hyperparam_detail::AsinhContigHyperparameterSet<argTy>;
    static constexpr std::uint8_t vec_sz = AsinhHS::vec_sz;
    static constexpr std::uint8_t n_vec = AsinhHS::n_vecs;

    return elementwise_common::unary_contig_impl<
        argTy, AsinhOutputType, AsinhContigFunctor, asinh_contig_kernel, vec_sz,
        n_vec>(exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T>
struct AsinhContigFactory
{
    fnT get()
    {
        if constexpr (!AsinhOutputType<T>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = asinh_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T>
struct AsinhTypeMapFactory
{
    /*! @brief get typeid for output type of sycl::asinh(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename AsinhOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3>
class asinh_strided_kernel;

template <typename argTy>
sycl::event
    asinh_strided_impl(sycl::queue &exec_q,
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
        argTy, AsinhOutputType, AsinhStridedFunctor, asinh_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T>
struct AsinhStridedFactory
{
    fnT get()
    {
        if constexpr (!AsinhOutputType<T>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = asinh_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace dpctl::tensor::kernels::asinh
