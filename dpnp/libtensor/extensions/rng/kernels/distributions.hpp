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
#include <cstdint>
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

        // u1 is on (0, 1], so the logarithm is always finite
        const T u1 = T(1) - bg.template next_uniform<T>();
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

// Lemire's method for a uniform integer on [0, rng], rng < 2**32 - 1
template <typename BitGenT>
std::uint32_t bounded_lemire_uint32(BitGenT &bg, const std::uint32_t rng)
{
    const std::uint32_t rng_excl = rng + 1;

    std::uint64_t m = static_cast<std::uint64_t>(bg.next_uint32()) * rng_excl;
    std::uint32_t leftover = static_cast<std::uint32_t>(m);

    if (leftover < rng_excl) {
        const std::uint32_t threshold =
            (std::numeric_limits<std::uint32_t>::max() - rng) % rng_excl;

        while (leftover < threshold) {
            m = static_cast<std::uint64_t>(bg.next_uint32()) * rng_excl;
            leftover = static_cast<std::uint32_t>(m);
        }
    }
    return static_cast<std::uint32_t>(m >> 32);
}

// Lemire's method for a uniform integer on [0, rng], rng < 2**64 - 1
template <typename BitGenT>
std::uint64_t bounded_lemire_uint64(BitGenT &bg, const std::uint64_t rng)
{
    const std::uint64_t rng_excl = rng + 1;

    std::uint64_t x = bg.next_uint64();
    std::uint64_t leftover = x * rng_excl;

    if (leftover < rng_excl) {
        const std::uint64_t threshold =
            (std::numeric_limits<std::uint64_t>::max() - rng) % rng_excl;

        while (leftover < threshold) {
            x = bg.next_uint64();
            leftover = x * rng_excl;
        }
    }
    return sycl::mul_hi(x, rng_excl);
}

// A uniform integer on [0, rng], as in NumPy's random_bounded_uint64()
template <typename BitGenT>
std::uint64_t bounded_uint64(BitGenT &bg, const std::uint64_t rng)
{
    if (rng == 0) {
        return 0;
    }
    else if (rng <= std::numeric_limits<std::uint32_t>::max()) {
        if (rng == std::numeric_limits<std::uint32_t>::max()) {
            return bg.next_uint32();
        }
        return bounded_lemire_uint32(bg, static_cast<std::uint32_t>(rng));
    }
    else if (rng == std::numeric_limits<std::uint64_t>::max()) {
        return bg.next_uint64();
    }
    return bounded_lemire_uint64(bg, rng);
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

// Poisson variates by multiplication of uniforms, for lam < 10
// The error of Stirling's approximation, log(n!) - log(sqrt(2*pi*n)*(n/e)**n)
template <typename T>
T stirlerr(const T n)
{
    constexpr T S0 = T(1) / T(12);
    constexpr T S1 = T(1) / T(360);
    constexpr T S2 = T(1) / T(1260);
    constexpr T S3 = T(1) / T(1680);
    constexpr T S4 = T(1) / T(1188);
    // log(sqrt(2*pi))
    constexpr T ln_sqrt_2pi = T(0.918938533204672741780329736406);

    if (n <= T(15)) {
        return sycl::lgamma(n + T(1)) - (n + T(0.5)) * sycl::log(n) + n -
               ln_sqrt_2pi;
    }
    const T nn = n * n;
    return (S0 - (S1 - (S2 - (S3 - S4 / nn) / nn) / nn) / nn) / n;
}

// Loader's deviance term x*log(x/np) + np - x, given d = x - np, which does
// not cancel for x close to np
template <typename T>
T bd0(const T x, const T np, const T d)
{
    if (sycl::fabs(d) < T(0.1) * (x + np)) {
        T v = d / (x + np);
        T s = d * v;
        T ej = T(2) * x * v;
        v = v * v;
        for (int j = 1; j < 1000; ++j) {
            ej *= v;
            const T s1 = s + ej / static_cast<T>(2 * j + 1);
            if (s1 == s) {
                return s1;
            }
            s = s1;
        }
        return s;
    }
    return x * sycl::log(x / np) - d;
}

// log(a!) - log(b!). The terms of the difference are of the order of
// (a - b) * log(a), not of a * log(a), so the result keeps its precision
// for large a and b, where lgamma(a + 1) - lgamma(b + 1) loses all of it.
template <typename T>
T logfactorial_diff(const std::int64_t a, const std::int64_t b)
{
    if (a == b) {
        return T(0);
    }
    if (sycl::min(a, b) < 16) {
        return sycl::lgamma(static_cast<T>(a) + T(1)) -
               sycl::lgamma(static_cast<T>(b) + T(1));
    }

    // with log(n!) = stirlerr(n) + (n + 1/2)*log(n) - n + log(sqrt(2*pi))
    const T af = static_cast<T>(a);
    const T bf = static_cast<T>(b);
    const T d = static_cast<T>(a - b);
    return stirlerr(af) - stirlerr(bf) - bd0(bf, af, -d) +
           T(0.5) * sycl::log1p(d / bf) + d * sycl::log(af);
}

template <typename T, typename BitGenT>
std::int64_t poisson_mult(BitGenT &bg, const T lam)
{
    const T enlam = sycl::exp(-lam);
    std::int64_t X = 0;
    T prod = T(1);
    for (;;) {
        prod *= bg.template next_uniform<T>();
        if (prod > enlam) {
            X += 1;
        }
        else {
            return X;
        }
    }
}

// Hormann's transformed rejection with squeeze (PTRS), for lam >= 10.
//
// Unlike NumPy's random_poisson_ptrs(), this keeps k as the integer part of
// lam plus an offset, and evaluates the log-pmf in Loader's form, as with
// floats both lam + offset and -lam + k*log(lam) - log(k!) lose too much.
template <typename T, typename BitGenT>
std::int64_t poisson_ptrs(BitGenT &bg, const T lam)
{
    constexpr T ln_2pi = T(1.83787706640934548356065947281);
    // the largest T below 2**63, which lam may round up to in float
    constexpr T lam_cap =
        T(0x1p63) - T(0x1p63) * std::numeric_limits<T>::epsilon() / T(2);

    const T slam = sycl::sqrt(lam);
    const T b = T(0.931) + T(2.53) * slam;
    const T a = T(-0.059) + T(0.02483) * b;
    const T invalpha = T(1.1239) + T(1.1328) / (b - T(3.4));
    const T vr = T(0.9277) - T(3.6224) / (b - T(2));

    const T lam_fl = sycl::fmin(sycl::floor(lam), lam_cap);
    const std::int64_t lam_i = static_cast<std::int64_t>(lam_fl);
    const T lam_f = lam - lam_fl;

    for (;;) {
        const T U = bg.template next_uniform<T>() - T(0.5);
        const T V = bg.template next_uniform<T>();
        const T us = T(0.5) - sycl::fabs(U);
        const T dk = sycl::floor((T(2) * a / us + b) * U + lam_f + T(0.43));

        // k = lam_i + dk must be a non-negative int64; NumPy casts first,
        // which is undefined out of range. This also rejects us == 0.
        if (!(sycl::fabs(dk) < T(0x1p62))) {
            continue;
        }
        const std::int64_t dk_i = static_cast<std::int64_t>(dk);
        if ((dk_i < -lam_i) ||
            (dk_i > std::numeric_limits<std::int64_t>::max() - lam_i)) {
            continue;
        }
        const std::int64_t k = lam_i + dk_i;

        if ((us >= T(0.07)) && (V <= vr)) {
            return k;
        }
        if ((us < T(0.013)) && (V > us)) {
            continue;
        }

        // log(lam**k * exp(-lam) / k!)
        T log_pmf;
        if (k == 0) {
            log_pmf = -lam;
        }
        else {
            const T kf = static_cast<T>(k);
            log_pmf = -stirlerr(kf) - bd0(kf, lam, static_cast<T>(dk) - lam_f) -
                      T(0.5) * (ln_2pi + sycl::log(kf));
        }

        // log(V) == log(0.0) is fine here
        if ((sycl::log(V) + sycl::log(invalpha) -
             sycl::log(a / (us * us) + b)) <= log_pmf) {
            return k;
        }
    }
}

template <typename T, typename BitGenT>
std::int64_t poisson(BitGenT &bg, const T lam)
{
    if (lam >= T(0x1p63)) {
        // invalid input, which PTRS would reject forever; this is the value
        // of the out of range cast in NumPy's legacy code. NaN gives 0 below,
        // as there.
        return std::numeric_limits<std::int64_t>::min();
    }
    if (lam >= T(10)) {
        return poisson_ptrs<T>(bg, lam);
    }
    else if (lam == T(0)) {
        return 0;
    }
    return poisson_mult<T>(bg, lam);
}

template <typename T, typename BitGenT>
T noncentral_chisquare(BitGenT &bg,
                       StandardNormal<T> &normal,
                       const T df,
                       const T nonc)
{
    if (sycl::isnan(nonc)) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (nonc == T(0)) {
        return chisquare<T>(bg, normal, df);
    }
    if (T(1) < df) {
        const T Chi2 = chisquare<T>(bg, normal, df - T(1));
        const T n = normal(bg) + sycl::sqrt(nonc);
        return Chi2 + n * n;
    }
    const std::int64_t i = poisson<T>(bg, nonc / T(2));
    return chisquare<T>(bg, normal, df + T(2) * static_cast<T>(i));
}

template <typename T, typename BitGenT>
T noncentral_f(BitGenT &bg,
               StandardNormal<T> &normal,
               const T dfnum,
               const T dfden,
               const T nonc)
{
    const T t = noncentral_chisquare<T>(bg, normal, dfnum, nonc) * dfden;
    return t / (chisquare<T>(bg, normal, dfden) * dfnum);
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
        // with double precision rho is zero until 1.4e-8
        s = T(1) / kappa + kappa;
    }
    else if (kappa <= T(1e6)) {
        const T r = T(1) + sycl::sqrt(T(1) + T(4) * kappa * kappa);
        const T rho = (r - sycl::sqrt(T(2) * r)) / (T(2) * kappa);
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

template <typename T, typename BitGenT>
std::int64_t binomial_inversion(BitGenT &bg, const std::int64_t n, const T p)
{
    const T q = T(1) - p;
    const T qn = sycl::exp(static_cast<T>(n) * sycl::log(q));
    const T np = static_cast<T>(n) * p;
    const T bound_f =
        sycl::fmin(static_cast<T>(n), np + T(10) * sycl::sqrt(np * q + T(1)));
    const std::int64_t bound = static_cast<std::int64_t>(bound_f);

    std::int64_t X = 0;
    T px = qn;
    T U = bg.template next_uniform<T>();
    while (U > px) {
        X++;
        if (X > bound) {
            X = 0;
            px = qn;
            U = bg.template next_uniform<T>();
        }
        else {
            U -= px;
            px = (static_cast<T>(n - X + 1) * p * px) / (static_cast<T>(X) * q);
        }
    }
    return X;
}

// Kachitvichyanukul and Schmeiser's BTPE algorithm, for p <= 0.5. The steps
// are numbered as in NumPy's random_binomial_btpe().
template <typename T, typename BitGenT>
std::int64_t binomial_btpe(BitGenT &bg, const std::int64_t n, const T p)
{
    const T nf = static_cast<T>(n);
    const T r = sycl::fmin(p, T(1) - p);
    const T q = T(1) - r;
    const T fm = nf * r + r;
    const std::int64_t m = static_cast<std::int64_t>(sycl::floor(fm));
    const T mf = static_cast<T>(m);
    const T p1 =
        sycl::floor(T(2.195) * sycl::sqrt(nf * r * q) - T(4.6) * q) + T(0.5);
    const T xm = mf + T(0.5);
    const T xl = xm - p1;
    const T xr = xm + p1;
    const T c = T(0.134) + T(20.5) / (T(15.3) + mf);
    T a = (fm - xl) / (fm - xl * r);
    const T laml = a * (T(1) + a / T(2));
    a = (xr - fm) / (xr * q);
    const T lamr = a * (T(1) + a / T(2));
    const T p2 = p1 * (T(1) + T(2) * c);
    const T p3 = p2 + c / laml;
    const T p4 = p3 + c / lamr;
    const T nrq = nf * r * q;

    for (;;) {
        // Step 10
        const T u = bg.template next_uniform<T>() * p4;
        T v = bg.template next_uniform<T>();
        std::int64_t y;
        if (u <= p1) {
            return static_cast<std::int64_t>(sycl::floor(xm - p1 * v + u));
        }
        else if (u <= p2) {
            // Step 20
            const T x = xl + (u - p1) / c;
            v = v * c + T(1) - sycl::fabs(mf - x + T(0.5)) / p1;
            if (v > T(1)) {
                continue;
            }
            y = static_cast<std::int64_t>(sycl::floor(x));
        }
        else if (u <= p3) {
            // Step 30; reject v == 0 before the undefined cast
            if (v == T(0)) {
                continue;
            }
            const T yf = sycl::floor(xl + sycl::log(v) / laml);
            if (yf < T(0)) {
                continue;
            }
            y = static_cast<std::int64_t>(yf);
            v = v * (u - p2) * laml;
        }
        else {
            // Step 40; as above
            if (v == T(0)) {
                continue;
            }
            const T yf = sycl::floor(xr - sycl::log(v) / lamr);
            if (yf > nf) {
                continue;
            }
            y = static_cast<std::int64_t>(yf);
            v = v * (u - p3) * lamr;
        }

        // Step 50
        const std::int64_t k = (y > m) ? y - m : m - y;
        const T kf = static_cast<T>(k);
        if (!((k > 20) && (kf < nrq / T(2) - T(1)))) {
            const T s = r / q;
            const T aa = s * (nf + T(1));
            T F = T(1);
            if (m < y) {
                for (std::int64_t i = m + 1; i <= y; ++i) {
                    F *= (aa / static_cast<T>(i) - s);
                }
            }
            else if (m > y) {
                for (std::int64_t i = y + 1; i <= m; ++i) {
                    F /= (aa / static_cast<T>(i) - s);
                }
            }
            if (v > F) {
                continue;
            }
            return y;
        }

        // Step 52
        const T rho =
            (kf / nrq) *
            ((kf * (kf / T(3) + T(0.625)) + T(0.16666666666666666)) / nrq +
             T(0.5));
        const T t = -kf * kf / (T(2) * nrq);
        // log(0.0) is fine here
        const T A = sycl::log(v);
        if (A < (t - rho)) {
            return y;
        }
        if (A > (t + rho)) {
            continue;
        }

        const T yf = static_cast<T>(y);
        const T x1 = yf + T(1);
        const T f1 = mf + T(1);
        const T z = nf + T(1) - mf;
        const T w = nf - yf + T(1);
        const T x2 = x1 * x1;
        const T f2 = f1 * f1;
        const T z2 = z * z;
        const T w2 = w * w;

        auto stirling = [](const T v1, const T v2) {
            return (T(13680) -
                    (T(462) - (T(132) - (T(99) - T(140) / v2) / v2) / v2) /
                        v2) /
                   v1 / T(166320);
        };

        if (A >
            (xm * sycl::log(f1 / x1) + (nf - mf + T(0.5)) * sycl::log(z / w) +
             (yf - mf) * sycl::log(w * r / (x1 * q)) + stirling(f1, f2) +
             stirling(z, z2) + stirling(x1, x2) + stirling(w, w2))) {
            continue;
        }
        return y;
    }
}

template <typename T, typename BitGenT>
std::int64_t binomial(BitGenT &bg, const std::int64_t n, const T p)
{
    // Invalid params are only possible on the device, where they are not
    // checked. BTPE rejects forever for a NaN p.
    if (!(n >= 0) || !(p >= T(0) && p <= T(1))) {
        return std::numeric_limits<std::int64_t>::min();
    }
    if ((n == 0) || (p == T(0))) {
        return 0;
    }

    const T nf = static_cast<T>(n);
    if (p <= T(0.5)) {
        if (p * nf <= T(30)) {
            return binomial_inversion<T>(bg, n, p);
        }
        return binomial_btpe<T>(bg, n, p);
    }

    const T q = T(1) - p;
    if (q * nf <= T(30)) {
        return n - binomial_inversion<T>(bg, n, q);
    }
    return n - binomial_btpe<T>(bg, n, q);
}

template <typename T, typename BitGenT>
std::int64_t negative_binomial(BitGenT &bg,
                               StandardNormal<T> &normal,
                               const T n,
                               const T p)
{
    const T Y = standard_gamma<T>(bg, normal, n) * ((T(1) - p) / p);
    return poisson<T>(bg, Y);
}

template <typename T, typename BitGenT>
std::int64_t geometric(BitGenT &bg, const T p)
{
    if (p >= T(0.333333333333333333333333)) {
        // search
        std::int64_t X = 1;
        T sum = p;
        T prod = p;
        const T q = T(1) - p;
        const T U = bg.template next_uniform<T>();
        // rounding can stall the sum below U, until prod underflows
        while ((U > sum) && (prod > T(0))) {
            prod *= q;
            sum += prod;
            X++;
        }
        return X;
    }

    // inversion
    const T z = sycl::ceil(-standard_exponential<T>(bg) / sycl::log1p(-p));
    // the smallest value above INT64_MAX in both float and double
    if (z >= T(9.223372036854775808e18)) {
        return std::numeric_limits<std::int64_t>::max();
    }
    return static_cast<std::int64_t>(z);
}

template <typename T, typename BitGenT>
std::int64_t logseries(BitGenT &bg, const T p)
{
    const T r = sycl::log1p(-p);
    for (;;) {
        const T V = bg.template next_uniform<T>();
        if (V >= p) {
            return 1;
        }
        const T U = bg.template next_uniform<T>();
        const T q = -sycl::expm1(r * U);
        if (V <= (q * q)) {
            if (V == T(0)) {
                continue;
            }
            const T result = sycl::floor(T(1) + sycl::log(V) / sycl::log(q));
            if (!(result >= T(1) && result < T(9.223372036854775808e18))) {
                continue;
            }
            return static_cast<std::int64_t>(result);
        }
        if (V >= q) {
            return 1;
        }
        return 2;
    }
}

template <typename T, typename BitGenT>
std::int64_t zipf(BitGenT &bg, const T a)
{
    if (!(a > T(1))) {
        // invalid, and every draw would be rejected
        return 0;
    }
    if (a >= T(1025)) {
        // every draw is 1
        return 1;
    }

    // the smallest value above INT64_MAX in both float and double
    constexpr T int64_bound = T(9.223372036854775808e18);

    const T am1 = a - T(1);
    const T b = sycl::pow(T(2), am1);
    // reject values that do not fit an int64, so this models a Zipf
    // distribution truncated to INT64_MAX
    const T Umin = sycl::pow(int64_bound, -am1);
    for (;;) {
        // U is on (Umin, 1], and Umin may be zero
        const T U01 = bg.template next_uniform<T>();
        const T U = U01 * Umin + (T(1) - U01);
        const T V = bg.template next_uniform<T>();
        const T X = sycl::floor(sycl::pow(U, T(-1) / am1));
        if (!(X >= T(1) && X < int64_bound)) {
            continue;
        }
        const T t = sycl::pow(T(1) + T(1) / X, am1);
        if (V * X * (t - T(1)) / (b - T(1)) <= t / b) {
            return static_cast<std::int64_t>(X);
        }
    }
}

// Draw `sample` items without replacement one by one, for small samples
template <typename T, typename BitGenT>
std::int64_t hypergeometric_sample(BitGenT &bg,
                                   const std::int64_t good,
                                   const std::int64_t bad,
                                   const std::int64_t sample)
{
    const std::int64_t total = good + bad;
    std::int64_t computed_sample =
        (sample > total / 2) ? total - sample : sample;

    std::int64_t remaining_total = total;
    std::int64_t remaining_good = good;
    while ((computed_sample > 0) && (remaining_good > 0) &&
           (remaining_total > remaining_good)) {
        --remaining_total;
        if (static_cast<std::int64_t>(bounded_uint64(
                bg, static_cast<std::uint64_t>(remaining_total))) <
            remaining_good) {
            --remaining_good;
        }
        --computed_sample;
    }

    if (remaining_total == remaining_good) {
        // only good items are left
        remaining_good -= computed_sample;
    }

    return (sample > total / 2) ? remaining_good : good - remaining_good;
}

// Stadlober's ratio-of-uniforms method (HRUA).
//
// Unlike NumPy's hypergeometric_hrua(), the mode m and the candidate K are
// computed in integers, as the integer part of the mean plus an offset, and
// g - gp as a sum of stable differences of log-factorials, so that neither
// loses its precision with floats and large populations. NumPy limits good
// and bad to below 10**9, so the integer products cannot overflow.
template <typename T, typename BitGenT>
std::int64_t hypergeometric_hrua(BitGenT &bg,
                                 const std::int64_t good,
                                 const std::int64_t bad,
                                 const std::int64_t sample)
{
    // D1 = 2*sqrt(2/e), D2 = 3 - 2*sqrt(3/e)
    constexpr T D1 = T(1.7155277699214135);
    constexpr T D2 = T(0.8989161620588988);

    const std::int64_t popsize = good + bad;
    const std::int64_t computed_sample = sycl::min(sample, popsize - sample);
    const std::int64_t mingoodbad = sycl::min(good, bad);
    const std::int64_t maxgoodbad = sycl::max(good, bad);

    const T p = static_cast<T>(mingoodbad) / static_cast<T>(popsize);
    const T q = static_cast<T>(maxgoodbad) / static_cast<T>(popsize);

    // the mean mu = computed_sample * p, and a = mu + 1/2 = a_i + a_f
    const std::int64_t mu_num = computed_sample * mingoodbad;
    const std::int64_t a_i = mu_num / popsize;
    const T a_f =
        static_cast<T>(mu_num % popsize) / static_cast<T>(popsize) + T(0.5);

    const T var = static_cast<T>(popsize - computed_sample) *
                  static_cast<T>(computed_sample) * p * q /
                  static_cast<T>(popsize - 1);
    const T c = sycl::sqrt(var + T(0.5));
    const T h = D1 * c + D2;

    const std::int64_t m =
        (computed_sample + 1) * (mingoodbad + 1) / (popsize + 2);
    const std::int64_t b = sycl::min(
        sycl::min(computed_sample, mingoodbad) + 1,
        a_i + static_cast<std::int64_t>(sycl::floor(a_f + T(16) * c)));

    std::int64_t K;
    for (;;) {
        const T U = bg.template next_uniform<T>();
        const T V = bg.template next_uniform<T>();
        // X = a + h*(V - 1/2)/U = a_i + dX
        const T dX = a_f + h * (V - T(0.5)) / U;

        // fast rejection of X < 0 or X >= b; this also rejects U == 0
        if (!(dX >= -static_cast<T>(a_i) - T(1) &&
              dX < static_cast<T>(b - a_i) + T(1))) {
            continue;
        }
        K = a_i + static_cast<std::int64_t>(sycl::floor(dX));
        if ((K < 0) || (K >= b)) {
            continue;
        }

        // g - gp, where g and gp are sums of four log-factorials at m and K
        const T t =
            logfactorial_diff<T>(m, K) +
            logfactorial_diff<T>(mingoodbad - m, mingoodbad - K) +
            logfactorial_diff<T>(computed_sample - m, computed_sample - K) +
            logfactorial_diff<T>(maxgoodbad - computed_sample + m,
                                 maxgoodbad - computed_sample + K);

        // fast acceptance
        if ((U * (T(4) - U) - T(3)) <= t) {
            break;
        }
        // fast rejection
        if (U * (U - t) >= T(1)) {
            continue;
        }
        if (T(2) * sycl::log(U) <= t) {
            break;
        }
    }

    if (good > bad) {
        K = computed_sample - K;
    }
    if (computed_sample < sample) {
        K = good - K;
    }
    return K;
}

template <typename T, typename BitGenT>
std::int64_t hypergeometric(BitGenT &bg,
                            const std::int64_t good,
                            const std::int64_t bad,
                            const std::int64_t sample)
{
    // Invalid params are only possible on the device, where they are not
    // checked. HRUA's integer products overflow for good or bad of 10**9 or
    // more, after which it may reject forever.
    constexpr std::int64_t max_good_bad = 1000000000;
    if (!((good >= 0) && (good < max_good_bad) && (bad >= 0) &&
          (bad < max_good_bad) && (sample >= 0) && (sample <= good + bad))) {
        return std::numeric_limits<std::int64_t>::min();
    }
    if ((sample >= 10) && (sample <= good + bad - 10)) {
        return hypergeometric_hrua<T>(bg, good, bad, sample);
    }
    return hypergeometric_sample<T>(bg, good, bad, sample);
}
} // namespace dpnp::kernels::rng
