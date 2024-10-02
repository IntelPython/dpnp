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

#include <sycl/ext/intel/math.hpp>
#include <sycl/sycl.hpp>

/**
 * Version of SYCL DPC++ 2025.1 compiler where an issue with
 * sycl::ext::intel::math::cyl_bessel_i0(x) is fully resolved.
 */
#ifndef __SYCL_COMPILER_BESSEL_I0_SUPPORT
// TODO: update with proper version
#define __SYCL_COMPILER_BESSEL_I0_SUPPORT 20241030L
#endif

namespace dpnp::kernels::i0
{
/**
 * The below implementation of Bessel function of order 0
 * is based on the source code from https://github.com/gcc-mirror/gcc
 */
namespace impl
{
/**
 * @brief This routine returns the cylindrical Bessel functions
 *        of order 0: \f$ J_0 \f$ or \f$ I_0 \f$
 *        by series expansion.
 *
 * The modified cylindrical Bessel function is:
 * @f[
 *  Z_0(x) = \sum_{k=0}^{\infty}
 *           \frac{\sigma^k (x/2)^{2k}}{k!\Gamma(k+1)}
 * @f]
 * where \f$ \sigma = +1 \f$ or\f$  -1 \f$ for
 * \f$ Z = I \f$ or \f$ J \f$ respectively.
 *
 * See Abramowitz & Stegun, 9.1.10
 *     Abramowitz & Stegun, 9.6.7
 * (1) Handbook of Mathematical Functions,
 *     ed. Milton Abramowitz and Irene A. Stegun,
 *     Dover Publications,
 *     Equation 9.1.10 p. 360 and Equation 9.6.10 p. 375
 *
 * @param x   The argument of the Bessel function.
 * @param sgn The sign of the alternate terms
 *              -1 for the Bessel function of the first kind.
 *              +1 for the modified Bessel function of the first kind.
 * @return The output Bessel function.
 */
template <typename Tp>
inline Tp cyl_bessel_ij_0_series(Tp x, Tp sgn, unsigned int max_iter)
{
    if (x == Tp(0)) {
        return Tp(1);
    }

    const Tp x2 = x / Tp(2);
    Tp fact = -sycl::lgamma(Tp(1));
    fact = std::exp(fact);

    const Tp xx4 = sgn * x2 * x2;
    Tp Jn = Tp(1);
    Tp term = Tp(1);

    for (unsigned int i = 1; i < max_iter; ++i) {
        term *= xx4 / (Tp(i) * Tp(i));
        Jn += term;
        if (sycl::fabs(term / Jn) < std::numeric_limits<Tp>::epsilon()) {
            break;
        }
    }
    return fact * Jn;
}

/**
 * @brief  Compute the modified Bessel functions @f$ I_0(0) @f$ and
 *         @f$ K_0(0) @f$ and their first derivatives
 *         @f$ I'_0(0) @f$ and @f$ K'_0(0) @f$ respectively.
 *
 * @param  Inu  The output regular modified Bessel function.
 * @param  Knu  The output irregular modified Bessel function.
 * @param  Ipnu The output derivative of the regular
 *                modified Bessel function.
 * @param  Kpnu The output derivative of the irregular
 *                modified Bessel function.
 */
template <typename Tp>
inline void bessel_ik_0_x_0(Tp &Inu, Tp &Knu, Tp &Ipnu, Tp &Kpnu)
{
    Inu = Tp(1);
    Ipnu = Tp(0);
    Knu = std::numeric_limits<Tp>::infinity();
    Kpnu = -std::numeric_limits<Tp>::infinity();
}

/**
 * @brief Compute the gamma functions required by the Temme series
 *        expansions of @f$ N_\nu(x) @f$ and @f$ K_\nu(x) @f$.
 * @f[
 *   \Gamma_1 = \frac{1}{2\mu}
 *              [\frac{1}{\Gamma(1 - \mu)} - \frac{1}{\Gamma(1 + \mu)}]
 * @f]
 * and
 * @f[
 *   \Gamma_2 = \frac{1}{2}
 *              [\frac{1}{\Gamma(1 - \mu)} + \frac{1}{\Gamma(1 + \mu)}]
 * @f]
 * where @f$ -1/2 <= \mu <= 1/2 @f$ is @f$ \mu = \nu - N @f$ and @f$ N @f$.
 * is the nearest integer to @f$ \nu @f$.
 * The values of \f$ \Gamma(1 + \mu) \f$ and \f$ \Gamma(1 - \mu) \f$
 * are returned as well.
 *
 * The accuracy requirements on this are exquisite.
 *
 * @param mu     The input parameter of the gamma functions.
 * @param gam1   The output function \f$ \Gamma_1(\mu) \f$
 * @param gam2   The output function \f$ \Gamma_2(\mu) \f$
 * @param gampl  The output function \f$ \Gamma(1 + \mu) \f$
 * @param gammi  The output function \f$ \Gamma(1 - \mu) \f$
 */
template <typename Tp>
inline void gamma_temme(Tp mu, Tp &gam1, Tp &gam2, Tp &gampl, Tp &gammi)
{
    gampl = Tp(1) / sycl::tgamma(Tp(1) + mu);
    gammi = Tp(1) / sycl::tgamma(Tp(1) - mu);

    if (sycl::fabs(mu) < std::numeric_limits<Tp>::epsilon())
        // constant Euler's constant @f$ \gamma_E @f$.
        gam1 = -static_cast<Tp>(0.5772156649015328606065120900824024L);
    else
        gam1 = (gammi - gampl) / (Tp(2) * mu);

    gam2 = (gammi + gampl) / (Tp(2));
}

/**
 * @brief  Compute the modified Bessel functions @f$ I_0(x) @f$ and
 *         @f$ K_0(x) @f$ and their first derivatives
 *         @f$ I'_0(x) @f$ and @f$ K'_0(x) @f$ respectively.
 *         These four functions are computed together for numerical
 *         stability.
 *
 * @param  x    The argument of the Bessel functions.
 * @param  Inu  The output regular modified Bessel function.
 * @param  Knu  The output irregular modified Bessel function.
 * @param  Ipnu The output derivative of the regular
 *                modified Bessel function.
 * @param  Kpnu The output derivative of the irregular
 *                modified Bessel function.
 */
template <typename Tp>
inline void bessel_ik_0(Tp x, Tp &Inu, Tp &Knu, Tp &Ipnu, Tp &Kpnu)
{
    if (x == Tp(0)) {
        bessel_ik_0_x_0(Inu, Knu, Ipnu, Kpnu);
        return;
    }

    constexpr Tp __nu = Tp(0);

    constexpr Tp eps = std::numeric_limits<Tp>::epsilon();
    constexpr Tp fp_min = Tp(10) * eps;
    constexpr int max_iter = 15000;
    constexpr Tp x_min = Tp(2);

    constexpr int nl = int(0.5L);

    const Tp mu = -nl;
    const Tp mu2 = mu * mu;
    const Tp xi = Tp(1) / x;
    const Tp xi2 = Tp(2) * xi;
    Tp h = fp_min;

    Tp b = Tp(0);
    Tp d = Tp(0);
    Tp c = h;
    int i;
    for (i = 1; i <= max_iter; ++i) {
        b += xi2;
        d = Tp(1) / (b + d);
        c = b + Tp(1) / c;

        const Tp del = c * d;
        h *= del;
        if (sycl::fabs(del - Tp(1)) < eps) {
            break;
        }
    }
    if (i > max_iter) {
        // argument `x` is too large
        bessel_ik_0_x_0(Inu, Knu, Ipnu, Kpnu);
        return;
    }

    Tp Inul = fp_min;
    Tp Ipnul = h * Inul;
    Tp Inul1 = Inul;
    Tp Ipnu1 = Ipnul;
    Tp fact = Tp(0);
    for (int l = nl; l >= 1; --l) {
        const Tp Inutemp = fact * Inul + Ipnul;
        fact -= xi;
        Ipnul = fact * Inutemp + Inul;
        Inul = Inutemp;
    }

    constexpr Tp pi = static_cast<Tp>(3.1415926535897932384626433832795029L);
    Tp f = Ipnul / Inul;
    Tp Kmu, Knu1;
    if (x < x_min) {
        const Tp x2 = x / Tp(2);
        const Tp pimu = pi * mu;
        const Tp fact =
            (sycl::fabs(pimu) < eps ? Tp(1) : pimu / sycl::sin(pimu));

        Tp d = -sycl::log(x2);
        Tp e = mu * d;
        const Tp fact2 = (sycl::fabs(e) < eps ? Tp(1) : sycl::sinh(e) / e);

        Tp gam1, gam2, gampl, gammi;
        gamma_temme(mu, gam1, gam2, gampl, gammi);

        Tp ff = fact * (gam1 * sycl::cosh(e) + gam2 * fact2 * d);
        Tp sum = ff;
        e = sycl::exp(e);

        Tp p = e / (Tp(2) * gampl);
        Tp q = Tp(1) / (Tp(2) * e * gammi);
        Tp c = Tp(1);
        d = x2 * x2;
        Tp sum1 = p;
        int i;
        for (i = 1; i <= max_iter; ++i) {
            ff = (i * ff + p + q) / (i * i - mu2);
            c *= d / i;
            p /= i - mu;
            q /= i + mu;
            const Tp del = c * ff;
            sum += del;
            const Tp __del1 = c * (p - i * ff);
            sum1 += __del1;
            if (sycl::fabs(del) < eps * sycl::fabs(sum)) {
                break;
            }
        }
        if (i > max_iter) {
            // Bessel k series failed to converge
            bessel_ik_0_x_0(Inu, Knu, Ipnu, Kpnu);
            return;
        }
        Kmu = sum;
        Knu1 = sum1 * xi2;
    }
    else {
        Tp b = Tp(2) * (Tp(1) + x);
        Tp d = Tp(1) / b;
        Tp delh = d;
        Tp h = delh;
        Tp q1 = Tp(0);
        Tp q2 = Tp(1);
        Tp a1 = Tp(0.25L) - mu2;
        Tp q = c = a1;
        Tp a = -a1;
        Tp s = Tp(1) + q * delh;
        int i;
        for (i = 2; i <= max_iter; ++i) {
            a -= 2 * (i - 1);
            c = -a * c / i;
            const Tp qnew = (q1 - b * q2) / a;
            q1 = q2;
            q2 = qnew;
            q += c * qnew;
            b += Tp(2);
            d = Tp(1) / (b + a * d);
            delh = (b * d - Tp(1)) * delh;
            h += delh;
            const Tp dels = q * delh;
            s += dels;
            if (sycl::fabs(dels / s) < eps) {
                break;
            }
        }
        if (i > max_iter) {
            // Steed's method failed
            bessel_ik_0_x_0(Inu, Knu, Ipnu, Kpnu);
            return;
        }
        h = a1 * h;
        Kmu = std::sqrt(pi / (Tp(2) * x)) * std::exp(-x) / s;
        Knu1 = Kmu * (mu + x + Tp(0.5L) - h) * xi;
    }

    Tp Kpmu = mu * xi * Kmu - Knu1;
    Tp Inumu = xi / (f * Kmu - Kpmu);
    Inu = Inumu * Inul1 / Inul;
    Ipnu = Inumu * Ipnu1 / Inul;
    for (i = 1; i <= nl; ++i) {
        const Tp Knutemp = (mu + i) * xi2 * Knu1 + Kmu;
        Kmu = Knu1;
        Knu1 = Knutemp;
    }
    Knu = Kmu;
    Kpnu = -Knu1;
}

/**
 * @brief Return the regular modified Bessel function of order 0:
 *        \f$ I_0(x) \f$.
 *
 *  The regular modified cylindrical Bessel function is:
 *  @f[
 *    I_0(x) = \sum_{k=0}^{\infty}
 *                 \frac{(x/2)^{2k}}{k!\Gamma(k+1)}
 *  @f]
 *
 * @param x The argument of the regular modified Bessel function.
 * @return The output regular modified Bessel function.
 */
template <typename Tp>
inline Tp cyl_bessel_0(Tp x)
{
    if (sycl::isnan(x)) {
        return std::numeric_limits<Tp>::quiet_NaN();
    }

    x = sycl::fabs(x);

    if (x * x < Tp(10)) {
        return cyl_bessel_ij_0_series(x, +Tp(1), 200);
    }
    else {
        Tp I_nu, K_nu, Ip_nu, Kp_nu;
        bessel_ik_0(x, I_nu, K_nu, Ip_nu, Kp_nu);
        return I_nu;
    }
}
} // namespace impl

template <typename argT, typename resT>
struct I0Functor
{
    // is function constant for given argT
    using is_constant = typename std::false_type;
    // constant value, if constant
    // constexpr resT constant_value = resT{};
    // is function defined for sycl::vec
    using supports_vec = typename std::false_type;
    // do both argT and resT support subgroup store/load operation
    using supports_sg_loadstore = typename std::true_type;

    resT operator()(const argT &x) const
    {
#if __SYCL_COMPILER_VERSION >= __SYCL_COMPILER_BESSEL_I0_SUPPORT
        return sycl::ext::intel::math::cyl_bessel_i0(x);
#else
        return impl::cyl_bessel_0(x);
#endif
    }
};
} // namespace dpnp::kernels::i0
