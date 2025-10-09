//*****************************************************************************
// Copyright (c) 2024, Intel Corporation
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

#pragma once

#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>

#include <sycl/sycl.hpp>

// dpctl tensor headers
#include "kernels/elementwise_functions/sycl_complex.hpp"
#include "utils/type_utils.hpp"

namespace dpnp::kernels::sinc
{
namespace tu_ns = dpctl::tensor::type_utils;

namespace impl
{
template <typename Tp>
inline Tp sin(const Tp &in)
{
    if constexpr (tu_ns::is_complex<Tp>::value) {
        using realTp = typename Tp::value_type;

        constexpr realTp q_nan = std::numeric_limits<realTp>::quiet_NaN();

        realTp const &in_re = std::real(in);
        realTp const &in_im = std::imag(in);

        const bool in_re_finite = sycl::isfinite(in_re);
        const bool in_im_finite = sycl::isfinite(in_im);
        /*
         * Handle the nearly-non-exceptional cases where
         * real and imaginary parts of input are finite.
         */
        if (in_re_finite && in_im_finite) {
            Tp res = exprm_ns::sin(exprm_ns::complex<realTp>(in)); // sin(in);
            if (in_re == realTp(0)) {
                res.real(sycl::copysign(realTp(0), in_re));
            }
            return res;
        }

        /*
         * since sin(in) = -I * sinh(I * in), for special cases,
         * we calculate real and imaginary parts of z = sinh(I * in) and
         * then return { imag(z) , -real(z) } which is sin(in).
         */
        const realTp x = -in_im;
        const realTp y = in_re;
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
        if (x == realTp(0) && !yfinite) {
            const realTp sinh_im = q_nan;
            const realTp sinh_re = sycl::copysign(realTp(0), x * sinh_im);
            return Tp{sinh_im, -sinh_re};
        }

        /*
         * sinh(+-Inf +- I 0) = +-Inf + I +-0.
         *
         * sinh(NaN +- I 0)   = d(NaN) + I +-0.
         */
        if (y == realTp(0) && !xfinite) {
            if (std::isnan(x)) {
                const realTp sinh_re = x;
                const realTp sinh_im = y;
                return Tp{sinh_im, -sinh_re};
            }
            const realTp sinh_re = x;
            const realTp sinh_im = sycl::copysign(realTp(0), y);
            return Tp{sinh_im, -sinh_re};
        }

        /*
         * sinh(x +- I Inf) = dNaN + I dNaN.
         *
         * sinh(x + I NaN) = d(NaN) + I d(NaN).
         */
        if (xfinite && !yfinite) {
            const realTp sinh_re = q_nan;
            const realTp sinh_im = x * sinh_re;
            return Tp{sinh_im, -sinh_re};
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
                const realTp sinh_re = -x * x;
                const realTp sinh_im = x * (y - y);
                return Tp{sinh_im, -sinh_re};
            }
            const realTp sinh_re = x * sycl::cos(y);
            const realTp sinh_im =
                std::numeric_limits<realTp>::infinity() * sycl::sin(y);
            return Tp{sinh_im, -sinh_re};
        }

        /*
         * sinh(NaN + I NaN)  = d(NaN) + I d(NaN).
         *
         * sinh(NaN +- I Inf) = d(NaN) + I d(NaN).
         *
         * sinh(NaN + I y)    = d(NaN) + I d(NaN).
         */
        const realTp y_m_y = (y - y);
        const realTp sinh_re = (x * x) * y_m_y;
        const realTp sinh_im = (x + x) * y_m_y;
        return Tp{sinh_im, -sinh_re};
    }
    else {
        if (in == Tp(0)) {
            return in;
        }
        return sycl::sin(in);
    }
}
} // namespace impl

template <typename argT, typename Tp>
struct SincFunctor
{
    // is function constant for given argT
    using is_constant = typename std::false_type;
    // constant value, if constant
    // constexpr Tp constant_value = Tp{};
    // is function defined for sycl::vec
    using supports_vec = typename std::false_type;
    // do both argT and Tp support subgroup store/load operation
    using supports_sg_loadstore = typename std::negation<
        std::disjunction<tu_ns::is_complex<Tp>, tu_ns::is_complex<argT>>>;

    Tp operator()(const argT &x) const
    {
        constexpr argT pi =
            static_cast<argT>(3.1415926535897932384626433832795029L);
        const argT y = pi * x;

        if (y == argT(0)) {
            return Tp(1);
        }

        if constexpr (tu_ns::is_complex<argT>::value) {
            return impl::sin(y) / y;
        }
        else {
            return sycl::sinpi(x) / y;
        }
    }
};
} // namespace dpnp::kernels::sinc
