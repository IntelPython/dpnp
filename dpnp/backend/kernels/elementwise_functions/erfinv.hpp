//*****************************************************************************
// Copyright (c) 2025, Intel Corporation
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

#include <limits>
#include <sycl/sycl.hpp>

namespace dpnp::kernels::erfs::impl
{
template <typename Tp>
inline Tp polevl(Tp x, const Tp *coeff, int i)
{
    Tp p = *coeff++;

    do {
        p = p * x + *coeff++;
    } while (--i);
    return p;
}

template <typename Tp>
inline Tp p1evl(Tp x, const Tp *coeff, int i)
{
    Tp p = x + *coeff++;

    while (--i) {
        p = p * x + *coeff++;
    }
    return p;
}

template <typename Tp>
inline Tp ndtri(Tp y0)
{
    Tp y;
    int code = 1;

    if (y0 == 0.0) {
        return -std::numeric_limits<Tp>::infinity();
    }
    else if (y0 == 1.0) {
        return std::numeric_limits<Tp>::infinity();
    }
    else if (y0 < 0.0 || y0 > 1.0) {
        return std::numeric_limits<Tp>::quiet_NaN();
    }

    // exp(-2)
    constexpr Tp exp_minus2 = 0.13533528323661269189399949497248L;
    if (y0 > (1.0 - exp_minus2)) {
        y = 1.0 - y0;
        code = 0;
    }
    else {
        y = y0;
    }

    if (y > exp_minus2) {
        // sqrt(2*pi)
        constexpr Tp root_2_pi = 2.50662827463100050241576528481105L;

        // approximation for 0 <= |y - 0.5| <= 3/8
        constexpr Tp p[] = {
            -5.99633501014107895267E1, 9.80010754185999661536E1,
            -5.66762857469070293439E1, 1.39312609387279679503E1,
            -1.23916583867381258016E0,
        };
        constexpr Tp q[] = {
            1.95448858338141759834E0, 4.67627912898881538453E0,
            8.63602421390890590575E1, -2.25462687854119370527E2,
            2.00260212380060660359E2, -8.20372256168333339912E1,
            1.59056225126211695515E1, -1.18331621121330003142E0,
        };

        y -= 0.5;
        Tp y2 = y * y;
        Tp x = y + y * (y2 * polevl(y2, p, 4) / p1evl(y2, q, 8));
        return x * root_2_pi;
    }

    Tp x = sycl::sqrt(-2.0 * sycl::log(y));
    Tp x0 = x - sycl::log(x) / x;
    Tp z = 1.0 / x;

    Tp x1;
    if (x < 8.0) {
        // approximation for 2 <= sqrt(-2*log(y)) < 8
        constexpr Tp p[] = {
            4.05544892305962419923E0,   3.15251094599893866154E1,
            5.71628192246421288162E1,   4.40805073893200834700E1,
            1.46849561928858024014E1,   2.18663306850790267539E0,
            -1.40256079171354495875E-1, -3.50424626827848203418E-2,
            -8.57456785154685413611E-4,
        };

        constexpr Tp q[] = {
            1.57799883256466749731E1,   4.53907635128879210584E1,
            4.13172038254672030440E1,   1.50425385692907503408E1,
            2.50464946208309415979E0,   -1.42182922854787788574E-1,
            -3.80806407691578277194E-2, -9.33259480895457427372E-4,
        };

        x1 = z * polevl(z, p, 8) / p1evl(z, q, 8);
    }
    else {
        // approximation for 8 <= sqrt(-2*log(y)) < 64
        constexpr Tp p[] = {
            3.23774891776946035970E0,  6.91522889068984211695E0,
            3.93881025292474443415E0,  1.33303460815807542389E0,
            2.01485389549179081538E-1, 1.23716634817820021358E-2,
            3.01581553508235416007E-4, 2.65806974686737550832E-6,
            6.23974539184983293730E-9,
        };

        constexpr Tp q[] = {
            6.02427039364742014255E0,  3.67983563856160859403E0,
            1.37702099489081330271E0,  2.16236993594496635890E-1,
            1.34204006088543189037E-2, 3.28014464682127739104E-4,
            2.89247864745380683936E-6, 6.79019408009981274425E-9,
        };

        x1 = z * polevl(z, p, 8) / p1evl(z, q, 8);
    }

    x = x0 - x1;
    if (code != 0) {
        x = -x;
    }
    return x;
}

template <typename Tp>
inline Tp erfinv(Tp y)
{
    static_assert(std::is_floating_point_v<Tp>,
                  "erfinv requires a floating-point type");

    constexpr Tp lower = -1;
    constexpr Tp upper = 1;

    constexpr Tp thresh = 1e-7;

    // For small arguments, use the Taylor expansion.
    // Otherwise, y + 1 loses precision for |y| << 1.
    if ((-thresh < y) && (y < thresh)) {
        // 2/sqrt(pi)
        constexpr Tp inv_sqrtpi = 1.1283791670955125738961589031215452L;
        return y / inv_sqrtpi;
    }

    if ((lower < y) && (y < upper)) {
        // 1/sqrt(2)
        constexpr Tp one_div_root_2 = 0.7071067811865475244008443621048490L;
        return ndtri(0.5 * (y + 1)) * one_div_root_2;
    }
    else if (y == lower) {
        return -std::numeric_limits<Tp>::infinity();
    }
    else if (y == upper) {
        return std::numeric_limits<Tp>::infinity();
    }
    else if (sycl::isnan(y)) {
        return y;
    }
    return std::numeric_limits<Tp>::quiet_NaN();
}

template <typename Tp>
inline Tp erfcinv(Tp y)
{
    static_assert(std::is_floating_point_v<Tp>,
                  "erfcinv requires a floating-point type");

    constexpr Tp lower = 0;
    constexpr Tp upper = 2;

    if ((lower < y) && (y < upper)) {
        // 1/sqrt(2)
        constexpr Tp one_div_root_2 = 0.7071067811865475244008443621048490L;
        return -ndtri(0.5 * y) * one_div_root_2;
    }
    else if (y == lower) {
        return std::numeric_limits<Tp>::infinity();
    }
    else if (y == upper) {
        return -std::numeric_limits<Tp>::infinity();
    }
    else if (sycl::isnan(y)) {
        return y;
    }
    return std::numeric_limits<Tp>::quiet_NaN();
}
} // namespace dpnp::kernels::erfs::impl
