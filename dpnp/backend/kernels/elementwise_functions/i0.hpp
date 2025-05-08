//*****************************************************************************
// Copyright (c) 2024-2025, Intel Corporation
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

#include <sycl/sycl.hpp>

/**
 * Version of SYCL DPC++ 2025.1 compiler where an issue with
 * sycl::ext::intel::math::cyl_bessel_i0(x) is fully resolved.
 */
#ifndef __SYCL_COMPILER_BESSEL_I0_SUPPORT
#define __SYCL_COMPILER_BESSEL_I0_SUPPORT 20241208L
#endif

// Include <sycl/ext/intel/math.hpp> only when targeting Intel devices.
// This header relies on intel-specific types like _iml_half_internal,
// which are not supported on non-intel backends (e.g., CUDA, AMD)
#if defined(__SPIR__) && defined(__INTEL_LLVM_COMPILER) &&                     \
    (__SYCL_COMPILER_VERSION >= __SYCL_COMPILER_BESSEL_I0_SUPPORT)
#include <sycl/ext/intel/math.hpp>
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
 *        of order 0 by series expansion.
 *
 * @param x The argument of the Bessel function.
 * @return The output Bessel function.
 */
template <typename Tp>
inline Tp cyl_bessel_ij_0_series(const Tp x, const unsigned int max_iter)
{
    const Tp x2 = x / Tp(2);
    const Tp fact = sycl::exp(-sycl::lgamma(Tp(1)));

    const Tp xx4 = x2 * x2;
    Tp Jn = Tp(1);
    Tp term = Tp(1);
    constexpr Tp eps = std::numeric_limits<Tp>::epsilon();

    for (unsigned int i = 1; i < max_iter; ++i) {
        term *= xx4 / (Tp(i) * Tp(i));
        Jn += term;
        if (sycl::fabs(term / Jn) < eps) {
            break;
        }
    }
    return fact * Jn;
}

/**
 * @brief Compute the modified Bessel functions.
 *
 * @param x The argument of the Bessel functions.
 * @return The output Bessel function.
 */
template <typename Tp>
inline Tp bessel_ik_0(Tp x)
{
    constexpr Tp eps = std::numeric_limits<Tp>::epsilon();
    constexpr Tp fp_min = Tp(10) * eps;
    constexpr int max_iter = 15000;
    constexpr Tp x_min = Tp(2);

    const Tp mu = Tp(0);
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
        return std::numeric_limits<Tp>::infinity();
    }

    Tp Inul = fp_min;
    const Tp Inul1 = Inul;
    const Tp Ipnul = h * Inul;

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

        // compute the gamma functions required by the Temme series expansions
        constexpr Tp gam1 =
            -static_cast<Tp>(0.5772156649015328606065120900824024L);
        const Tp gam2 = Tp(1) / sycl::tgamma(Tp(1));

        Tp ff = fact * (gam1 * sycl::cosh(e) + gam2 * fact2 * d);
        Tp sum = ff;
        e = sycl::exp(e);

        Tp p = e / (Tp(2) * gam2);
        Tp q = Tp(1) / (Tp(2) * e * gam2);
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
            return std::numeric_limits<Tp>::quiet_NaN();
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
            return std::numeric_limits<Tp>::quiet_NaN();
        }
        h = a1 * h;
        Kmu = sycl::sqrt(pi / (Tp(2) * x)) * sycl::exp(-x) / s;
        Knu1 = Kmu * (mu + x + Tp(0.5L) - h) * xi;
    }

    Tp Kpmu = mu * xi * Kmu - Knu1;
    Tp Inumu = xi / (f * Kmu - Kpmu);
    return Inumu * Inul1 / Inul;
}

/**
 * @brief Return the regular modified Bessel function of order 0.
 *
 * @param x The argument of the regular modified Bessel function.
 * @return The output regular modified Bessel function.
 */
template <typename Tp>
inline Tp cyl_bessel_i0(Tp x)
{
    if (sycl::isnan(x)) {
        return std::numeric_limits<Tp>::quiet_NaN();
    }

    if (sycl::isinf(x)) {
        // return +inf per any input infinity
        return std::numeric_limits<Tp>::infinity();
    }

    if (x == Tp(0)) {
        return Tp(1);
    }

    if (x * x < Tp(10)) {
        return cyl_bessel_ij_0_series<Tp>(x, 200);
    }
    return bessel_ik_0(sycl::fabs(x));
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
#if defined(__SPIR__) && defined(__INTEL_LLVM_COMPILER) &&                     \
    (__SYCL_COMPILER_VERSION >= __SYCL_COMPILER_BESSEL_I0_SUPPORT)
        using sycl::ext::intel::math::cyl_bessel_i0;
#else
        using impl::cyl_bessel_i0;
#endif

        if constexpr (std::is_same_v<resT, sycl::half>) {
            return static_cast<resT>(cyl_bessel_i0<float>(float(x)));
        }
        else {
            return cyl_bessel_i0(x);
        }
    }
};
} // namespace dpnp::kernels::i0
