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
/// This file defines kernels for elementwise evaluation of SIN(x) function.
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

#include "utils/type_dispatch_building.hpp"
#include "utils/type_utils.hpp"

namespace dpctl::tensor::kernels::sin
{

using dpctl::tensor::ssize_t;
namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;

template <typename argT, typename resT>
struct SinFunctor
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

            realT const &in_re = std::real(in);
            realT const &in_im = std::imag(in);

            const bool in_re_finite = std::isfinite(in_re);
            const bool in_im_finite = std::isfinite(in_im);
            /*
             * Handle the nearly-non-exceptional cases where
             * real and imaginary parts of input are finite.
             */
            if (in_re_finite && in_im_finite) {
                resT res =
                    exprm_ns::sin(exprm_ns::complex<realT>(in)); // sin(in);
                if (in_re == realT(0)) {
                    res.real(sycl::copysign(realT(0), in_re));
                }
                return res;
            }

            /*
             * since sin(in) = -I * sinh(I * in), for special cases,
             * we calculate real and imaginary parts of z = sinh(I * in) and
             * then return { imag(z) , -real(z) } which is sin(in).
             */
            const realT x = -in_im;
            const realT y = in_re;
            const bool xfinite = in_im_finite;
            const bool yfinite = in_re_finite;
            /*
             * sinh(+-0 +- I Inf) = sign(d(+-0, dNaN))0 + I dNaN.
             * The sign of 0 in the result is unspecified.  Choice = normally
             * the same as dNaN.
             *
             * sinh(+-0 +- I NaN) = sign(d(+-0, NaN))0 + I d(NaN).
             * The sign of 0 in the result is unspecified.  Choice = normally
             * the same as d(NaN).
             */
            if (x == realT(0) && !yfinite) {
                const realT sinh_im = q_nan;
                const realT sinh_re = sycl::copysign(realT(0), x * sinh_im);
                return resT{sinh_im, -sinh_re};
            }

            /*
             * sinh(+-Inf +- I 0) = +-Inf + I +-0.
             *
             * sinh(NaN +- I 0)   = d(NaN) + I +-0.
             */
            if (y == realT(0) && !xfinite) {
                if (std::isnan(x)) {
                    const realT sinh_re = x;
                    const realT sinh_im = y;
                    return resT{sinh_im, -sinh_re};
                }
                const realT sinh_re = x;
                const realT sinh_im = sycl::copysign(realT(0), y);
                return resT{sinh_im, -sinh_re};
            }

            /*
             * sinh(x +- I Inf) = dNaN + I dNaN.
             *
             * sinh(x + I NaN) = d(NaN) + I d(NaN).
             */
            if (xfinite && !yfinite) {
                const realT sinh_re = q_nan;
                const realT sinh_im = x * sinh_re;
                return resT{sinh_im, -sinh_re};
            }

            /*
             * sinh(+-Inf + I NaN)  = +-Inf + I d(NaN).
             * The sign of Inf in the result is unspecified.  Choice = normally
             * the same as d(NaN).
             *
             * sinh(+-Inf +- I Inf) = +Inf + I dNaN.
             * The sign of Inf in the result is unspecified.
             * Choice = always - here for sinh to have positive result for
             * imaginary part of sin.
             *
             * sinh(+-Inf + I y)   = +-Inf cos(y) + I Inf sin(y)
             */
            if (std::isinf(x)) {
                if (!yfinite) {
                    const realT sinh_re = -x * x;
                    const realT sinh_im = x * (y - y);
                    return resT{sinh_im, -sinh_re};
                }
                const realT sinh_re = x * sycl::cos(y);
                const realT sinh_im =
                    std::numeric_limits<realT>::infinity() * sycl::sin(y);
                return resT{sinh_im, -sinh_re};
            }

            /*
             * sinh(NaN + I NaN)  = d(NaN) + I d(NaN).
             *
             * sinh(NaN +- I Inf) = d(NaN) + I d(NaN).
             *
             * sinh(NaN + I y)    = d(NaN) + I d(NaN).
             */
            const realT y_m_y = (y - y);
            const realT sinh_re = (x * x) * y_m_y;
            const realT sinh_im = (x + x) * y_m_y;
            return resT{sinh_im, -sinh_re};
        }
        else {
            static_assert(std::is_same_v<argT, resT>);
            if (in == 0) {
                return in;
            }
            return sycl::sin(in);
        }
    }
};

template <typename argTy,
          typename resTy = argTy,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
using SinContigFunctor =
    elementwise_common::UnaryContigFunctor<argTy,
                                           resTy,
                                           SinFunctor<argTy, resTy>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using SinStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, SinFunctor<argTy, resTy>>;

template <typename T>
struct SinOutputType
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
struct SinContigHyperparameterSet
{
    using value_type =
        typename std::disjunction<ContigHyperparameterSetDefault<4u, 2u>>;

    constexpr static auto vec_sz = value_type::vec_sz;
    constexpr static auto n_vecs = value_type::n_vecs;
};

} // end of namespace hyperparam_detail

template <typename T1, typename T2, std::uint8_t vec_sz, std::uint8_t n_vecs>
class sin_contig_kernel;

template <typename argTy>
sycl::event sin_contig_impl(sycl::queue &exec_q,
                            std::size_t nelems,
                            const char *arg_p,
                            char *res_p,
                            const std::vector<sycl::event> &depends = {})
{
    using SinHS = hyperparam_detail::SinContigHyperparameterSet<argTy>;
    static constexpr std::uint8_t vec_sz = SinHS::vec_sz;
    static constexpr std::uint8_t n_vecs = SinHS::n_vecs;

    return elementwise_common::unary_contig_impl<
        argTy, SinOutputType, SinContigFunctor, sin_contig_kernel, vec_sz,
        n_vecs>(exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T>
struct SinContigFactory
{
    fnT get()
    {
        if constexpr (!SinOutputType<T>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = sin_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T>
struct SinTypeMapFactory
{
    /*! @brief get typeid for output type of sycl::sin(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename SinOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3>
class sin_strided_kernel;

template <typename argTy>
sycl::event sin_strided_impl(sycl::queue &exec_q,
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
        argTy, SinOutputType, SinStridedFunctor, sin_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T>
struct SinStridedFactory
{
    fnT get()
    {
        if constexpr (!SinOutputType<T>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = sin_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace dpctl::tensor::kernels::sin
