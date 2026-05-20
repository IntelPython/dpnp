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
/// This file defines complex math functions.
//===----------------------------------------------------------------------===//

#pragma once

#include <cmath>
#include <complex>
#include <limits>

#include <sycl/sycl.hpp>

#include "sycl_complex.hpp" // for exprm_ns

namespace dpnp::tensor::kernels::complex_math
{
static constexpr double ln2 = 0.6931471805599453094172321214581765L;
static constexpr double pi = 3.1415926535897932384626433832795029L;

template <typename realT>
static constexpr realT q_nan = std::numeric_limits<realT>::quiet_NaN();

template <typename T>
T cacos(const T &in)
{
    using realT = typename T::value_type;
    using sycl_complexT = exprm_ns::complex<realT>;

    const realT x = std::real(in);
    const realT y = std::imag(in);

    if (std::isnan(x)) {
        // acos(NaN + I*+-Inf) = NaN + I*-+Inf
        if (std::isinf(y)) {
            return T{q_nan<realT>, -y};
        }

        // all other cases involving NaN return NaN + I*NaN
        return T{q_nan<realT>, q_nan<realT>};
    }

    if (std::isnan(y)) {
        // acos(+-Inf + I*NaN) = NaN + I*opt(-)Inf
        if (std::isinf(x)) {
            return T{q_nan<realT>, -std::numeric_limits<realT>::infinity()};
        }

        // acos(0 + I*NaN) = PI/2 + I*NaN with inexact
        if (x == realT(0)) {
            static constexpr realT pio2 = realT(pi) / realT(2); // PI/2
            return T{pio2, q_nan<realT>};
        }

        // all other cases involving NaN return NaN + I*NaN
        return T{q_nan<realT>, q_nan<realT>};
    }

    /*
     * For large x or y including acos(+-Inf + I*+-Inf).
     * exprm_ns::acos(x) is based on calculating log(x + sqrt(x^2 - 1)),
     * so r_eps = sqrt(1/eps)/2 is appropriate precision loss point.
     */
    const realT r_eps =
        sycl::sqrt(realT(1) / std::numeric_limits<realT>::epsilon()) / 2;
    if (sycl::fabs(x) > r_eps || sycl::fabs(y) > r_eps) {
        sycl_complexT log_in = exprm_ns::log(sycl_complexT(in));

        const realT wx = log_in.real();
        const realT wy = log_in.imag();
        const realT rx = sycl::fabs(wy);

        realT ry = wx + realT(ln2);
        return T{rx, (sycl::signbit(y)) ? ry : -ry};
    }

    // ordinary cases
    return exprm_ns::acos(sycl_complexT(in)); // sycl::acos(in);
}

template <typename T>
T casinh(const T &in)
{
    using realT = typename T::value_type;
    using sycl_complexT = exprm_ns::complex<realT>;

    const realT x = std::real(in);
    const realT y = std::imag(in);

    if (std::isnan(x)) {
        // asinh(NaN + I*+-Inf) = opt(+-)Inf + I*NaN
        if (std::isinf(y)) {
            return T{y, q_nan<realT>};
        }
        // asinh(NaN + I*0) = NaN + I*0
        if (y == realT(0)) {
            return T{q_nan<realT>, y};
        }
        // all other cases involving NaN return NaN + I*NaN
        return T{q_nan<realT>, q_nan<realT>};
    }

    if (std::isnan(y)) {
        // asinh(+-Inf + I*NaN) = +-Inf + I*NaN
        if (std::isinf(x)) {
            return T{x, q_nan<realT>};
        }
        // all other cases involving NaN return NaN + I*NaN
        return T{q_nan<realT>, q_nan<realT>};
    }

    /*
     * For large x or y including asinh(+-Inf + I*+-Inf)
     * asinh(in) = sign(x)*log(sign(x)*in) + O(1/in^2)   as in -> infinity
     * The above formula works for the imaginary part as well, because
     * Im(asinh(in)) = sign(x)*atan2(sign(x)*y, fabs(x)) + O(y/in^3)
     * as in -> infinity, uniformly in y.
     *
     * exprm_ns::asinh(x) is based on calculating log(x + sqrt(x^2 + 1)),
     * so r_eps = sqrt(1/eps)/2 is appropriate precision loss point.
     */
    const realT r_eps =
        sycl::sqrt(realT(1) / std::numeric_limits<realT>::epsilon()) / 2;
    if (sycl::fabs(x) > r_eps || sycl::fabs(y) > r_eps) {
        sycl_complexT log_in = (sycl::signbit(x))
                                   ? exprm_ns::log(sycl_complexT(-in))
                                   : exprm_ns::log(sycl_complexT(in));
        realT wx = log_in.real() + realT(ln2);
        realT wy = log_in.imag();

        const realT res_re = sycl::copysign(wx, x);
        const realT res_im = sycl::copysign(wy, y);
        return T{res_re, res_im};
    }

    // ordinary cases
    return exprm_ns::asinh(sycl_complexT(in));
}

template <typename T>
T casin(const T &in)
{
    /*
     * casin(z) = reverse(casinh(reverse(z))),
     * where reverse(x + I*y) = y + I*x = I*conj(z)
     */

    // reverse(z): swap real and imaginary parts
    T reversed{std::imag(in), std::real(in)};

    // compute asinh of reversed input
    T w = casinh(reversed);

    // reverse result back: swap real and imaginary parts
    return T{std::imag(w), std::real(w)};
}
} // namespace dpnp::tensor::kernels::complex_math
