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
// Most of the sampling algorithms below are adapted from NumPy's
// numpy/random/src/distributions/distributions.c, which carries this notice:
//
// Copyright (c) 2005-2025, NumPy Developers.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//        copyright notice, this list of conditions and the following
//        disclaimer in the documentation and/or other materials provided
//        with the distribution.
//
//     * Neither the name of the NumPy Developers nor the names of any
//        contributors may be used to endorse or promote products derived
//        from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//*****************************************************************************

#pragma once

#include <cstddef>
#include <limits>
#include <type_traits>

#include <sycl/sycl.hpp>

namespace dpnp::kernels::rng
{
// A distribution parameter: either one value for every sample, or a
// C-contiguous array holding one value per sample.
template <typename T>
struct Param
{
    const T *ptr = nullptr;
    T value{};

    T operator[](std::size_t i) const { return ptr ? ptr[i] : value; }
};

// Standard normal variates by the Box-Muller transform. The second value
// of each pair is cached, so one instance must live for the whole run of a
// work-item.
template <typename T>
class StandardNormal
{
private:
    T cached_{};
    bool has_cached_ = false;

public:
    template <typename BitGenT>
    T operator()(BitGenT &bg)
    {
        if (has_cached_) {
            has_cached_ = false;
            return cached_;
        }

        // u1 is on (0, 1], so the logarithm is always finite. In float, it
        // takes a whole 32-bit word, at the midpoints of 2**-32 steps, so
        // that |z| reaches 6.76 rather than the 5.77 of 24 random bits.
        T u1;
        if constexpr (std::is_same_v<T, float>) {
            const float w = static_cast<float>(bg.next_uint32());
            u1 = (w + 0.5f) * 0x1p-32f;
        }
        else {
            u1 = T(1) - bg.template next_uniform<T>();
        }
        const T u2 = bg.template next_uniform<T>();

        const T r = sycl::sqrt(T(-2) * sycl::log(u1));
        const T phi = T(2) * u2;

        cached_ = r * sycl::cospi(phi);
        has_cached_ = true;
        return r * sycl::sinpi(phi);
    }
};

// NumPy's "inv" method; 1 - u is on (0, 1], so the result is finite
template <typename T, typename BitGenT>
T standard_exponential(BitGenT &bg)
{
    return -sycl::log1p(-bg.template next_uniform<T>());
}

// Marsaglia and Tsang for shape > 1, and a rejection method built on the
// exponential distribution for shape < 1, as in NumPy's
// random_standard_gamma()
template <typename T, typename BitGenT>
T standard_gamma(BitGenT &bg, StandardNormal<T> &normal, const T shape)
{
    if (shape == T(1)) {
        return standard_exponential<T>(bg);
    }
    else if (shape == T(0)) {
        return T(0);
    }
    else if (!(shape > T(0))) {
        // invalid, and for a shape just below zero every draw would be
        // rejected
        return std::numeric_limits<T>::quiet_NaN();
    }
    else if (shape < T(1)) {
        for (;;) {
            const T U = bg.template next_uniform<T>();
            const T V = standard_exponential<T>(bg);
            if (U <= T(1) - shape) {
                const T X = sycl::pow(U, T(1) / shape);
                if (X <= V) {
                    return X;
                }
            }
            else {
                const T Y = -sycl::log((T(1) - U) / shape);
                const T X = sycl::pow(T(1) - shape + shape * Y, T(1) / shape);
                if (X <= (V + Y)) {
                    return X;
                }
            }
        }
    }

    const T b = shape - T(1) / T(3);
    const T c = T(1) / sycl::sqrt(T(9) * b);
    for (;;) {
        T X, V;
        do {
            X = normal(bg);
            V = T(1) + c * X;
        } while (V <= T(0));

        V = V * V * V;
        const T U = bg.template next_uniform<T>();
        if (U < T(1) - T(0.0331) * (X * X) * (X * X)) {
            return b * V;
        }
        // log(0) is fine here
        if (sycl::log(U) < T(0.5) * X * X + b * (T(1) - V + sycl::log(V))) {
            return b * V;
        }
    }
}

// Johnk's algorithm when both parameters are at most one, and a ratio of
// gamma variates otherwise, as in NumPy's random_beta()
template <typename T, typename BitGenT>
T beta(BitGenT &bg, StandardNormal<T> &normal, const T a, const T b)
{
    if (!(a > T(0)) || !(b > T(0))) {
        // invalid input, which Johnk's algorithm would reject forever
        return std::numeric_limits<T>::quiet_NaN();
    }
    if ((a <= T(1)) && (b <= T(1))) {
        // Below this threshold the result is 0 or 1 with overwhelming
        // probability, so a single uniform draw decides which. The
        // threshold underflows to zero, disabling the shortcut, in float.
        constexpr T tiny_threshold = static_cast<T>(3e-103);
        if (a < tiny_threshold && b < tiny_threshold) {
            const T U = bg.template next_uniform<T>();
            return static_cast<T>((a + b) * U < a);
        }

        for (;;) {
            const T U = bg.template next_uniform<T>();
            const T V = bg.template next_uniform<T>();
            const T X = sycl::pow(U, T(1) / a);
            const T Y = sycl::pow(V, T(1) / b);
            const T XpY = X + Y;
            // reject if both U and V are 0
            if ((XpY <= T(1)) && (U + V > T(0))) {
                T res;

                // NumPy tests X > 0 and Y > 0 here. Devices that flush
                // subnormals to zero can see a subnormal X as positive
                // while X + Y flushes to zero, so require normal values.
                constexpr T min_normal = std::numeric_limits<T>::min();
                if ((X >= min_normal) && (Y >= min_normal)) {
                    res = X / XpY;
                }
                else {
                    // X or Y underflowed, so work with logarithms
                    const T logX = sycl::log(U) / a;
                    const T logY = sycl::log(V) / b;
                    const T delta = logX - logY;
                    if (delta > T(0)) {
                        res = sycl::exp(-sycl::log1p(sycl::exp(-delta)));
                    }
                    else {
                        res = sycl::exp(delta - sycl::log1p(sycl::exp(delta)));
                    }
                }

                // Division and exp need not be correctly rounded on the
                // device, and can land an ulp above 1; NaN is kept
                return (res > T(1)) ? T(1) : res;
            }
        }
    }

    const T Ga = standard_gamma<T>(bg, normal, a);
    const T Gb = standard_gamma<T>(bg, normal, b);
    // as above; this is Ga / Ga when Gb underflows
    const T res = Ga / (Ga + Gb);
    return (res > T(1)) ? T(1) : res;
}

template <typename T, typename BitGenT>
T chisquare(BitGenT &bg, StandardNormal<T> &normal, const T df)
{
    return T(2) * standard_gamma<T>(bg, normal, df / T(2));
}

template <typename T, typename BitGenT>
T standard_cauchy(BitGenT &bg, StandardNormal<T> &normal)
{
    const T num = normal(bg);
    return num / normal(bg);
}

template <typename T, typename BitGenT>
T standard_t(BitGenT &bg, StandardNormal<T> &normal, const T df)
{
    const T num = normal(bg);
    const T denom = standard_gamma<T>(bg, normal, df / T(2));
    return sycl::sqrt(df / T(2)) * num / sycl::sqrt(denom);
}

template <typename T, typename BitGenT>
T f(BitGenT &bg, StandardNormal<T> &normal, const T dfnum, const T dfden)
{
    const T num = chisquare<T>(bg, normal, dfnum) * dfden;
    return num / (chisquare<T>(bg, normal, dfden) * dfnum);
}

template <typename T, typename BitGenT>
T laplace(BitGenT &bg, const T loc, const T scale)
{
    for (;;) {
        const T U = bg.template next_uniform<T>();
        if (U >= T(0.5)) {
            return loc - scale * sycl::log(T(2) - U - U);
        }
        else if (U > T(0)) {
            return loc + scale * sycl::log(U + U);
        }
        // reject U == 0
    }
}

template <typename T, typename BitGenT>
T gumbel(BitGenT &bg, const T loc, const T scale)
{
    for (;;) {
        const T U = T(1) - bg.template next_uniform<T>();
        if (U < T(1)) {
            return loc - scale * sycl::log(-sycl::log(U));
        }
        // reject U == 1
    }
}

template <typename T, typename BitGenT>
T logistic(BitGenT &bg, const T loc, const T scale)
{
    for (;;) {
        const T U = bg.template next_uniform<T>();
        if (U > T(0)) {
            return loc + scale * sycl::log(U / (T(1) - U));
        }
        // reject U == 0
    }
}

// NumPy's random_wald() takes X = mean + mean/(2*scale)*(Y - sqrt(4*scale*Y +
// Y*Y)), which cancels, and overflows in float, for large Y. This is the same
// value as mean*4*scale/(sqrt(Y) + sqrt(Y + 4*scale))**2.
template <typename T, typename BitGenT>
T wald(BitGenT &bg, StandardNormal<T> &normal, const T mean, const T scale)
{
    const T n = normal(bg);
    const T Y = mean * n * n;
    const T s = sycl::sqrt(Y) + sycl::sqrt(Y + T(4) * scale);
    const T X = mean * (T(4) * scale / (s * s));
    const T U = bg.template next_uniform<T>();
    if (U <= mean / (mean + X)) {
        return X;
    }
    return mean * (mean / X);
}

template <typename T, typename BitGenT>
T triangular(BitGenT &bg, const T left, const T mode, const T right)
{
    const T base = right - left;
    const T leftbase = mode - left;
    const T ratio = leftbase / base;
    const T leftprod = leftbase * base;
    const T rightprod = (right - mode) * base;

    const T U = bg.template next_uniform<T>();
    if (U <= ratio) {
        return left + sycl::sqrt(U * leftprod);
    }
    return right - sycl::sqrt((T(1) - U) * rightprod);
}

// Best and Fisher's algorithm, with the fallbacks of NumPy's
// random_vonmises() for very small and very large kappa
template <typename T, typename BitGenT>
T vonmises(BitGenT &bg, StandardNormal<T> &normal, const T mu, const T kappa)
{
    constexpr T pi = T(3.141592653589793238462643383279502884L);

    if (sycl::isnan(kappa)) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (kappa < T(1e-8)) {
        // the distribution is uniform on the circle
        return pi * (T(2) * bg.template next_uniform<T>() - T(1));
    }

    T s;
    if (kappa < T(1e-5)) {
        // as in NumPy, whose form of rho is zero in double precision until
        // 1.4e-8
        s = T(1) / kappa + kappa;
    }
    else if (kappa <= T(1e6)) {
        // rho = (r - sqrt(2 r)) / (2 kappa), without the cancellation which
        // makes it zero in single precision for kappa below about 4e-4,
        // where every draw would be rejected
        const T r = T(1) + sycl::sqrt(T(1) + T(4) * kappa * kappa);
        const T rho = T(2) * kappa / (r + sycl::sqrt(T(2) * r));
        s = (T(1) + rho * rho) / (T(2) * rho);
    }
    else {
        // a wrapped normal approximates the distribution for large kappa
        T result = mu + sycl::sqrt(T(1) / kappa) * normal(bg);
        if (result < -pi) {
            result += T(2) * pi;
        }
        if (result > pi) {
            result -= T(2) * pi;
        }
        return result;
    }

    T W;
    for (;;) {
        const T U = bg.template next_uniform<T>();
        const T Z = sycl::cospi(U);
        W = (T(1) + s * Z) / (s + Z);
        const T Y = kappa * (s - W);
        const T V = bg.template next_uniform<T>();
        // V == 0 is fine: Y >= 0 always accepts and Y < 0 always rejects
        if ((Y * (T(2) - Y) - V >= T(0)) ||
            (sycl::log(Y / V) + T(1) - Y >= T(0))) {
            break;
        }
    }

    const T U = bg.template next_uniform<T>();
    T result = sycl::acos(W);
    if (U < T(0.5)) {
        result = -result;
    }
    result += mu;

    const bool neg = (result < T(0));
    T mod = sycl::fabs(result);
    mod = sycl::fmod(mod + pi, T(2) * pi) - pi;
    return neg ? -mod : mod;
}
} // namespace dpnp::kernels::rng
