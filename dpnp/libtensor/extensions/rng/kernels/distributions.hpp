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
                // device, and can land an ulp above 1
                return sycl::fmin(res, T(1));
            }
        }
    }

    const T Ga = standard_gamma<T>(bg, normal, a);
    const T Gb = standard_gamma<T>(bg, normal, b);
    // as above; this is Ga / Ga when Gb underflows
    return sycl::fmin(Ga / (Ga + Gb), T(1));
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

// Every sampler takes two parameters, (a, b), and ignores those it has no
// use for. operator() returns the sample for output element `i`; it may
// update the sampler's own state, so each work-item works on a copy.

// The raw 64-bit output of the bit generator; ignores both parameters
template <typename T>
class RawSampler
{
public:
    RawSampler(const Param<T> &, const Param<T> &) {}

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t)
    {
        return bg.next_uint64();
    }
};

// Uniform on [a, b)
template <typename T>
class UniformSampler
{
private:
    Param<T> low_;
    Param<T> high_;

public:
    UniformSampler(const Param<T> &low, const Param<T> &high)
        : low_(low), high_(high)
    {
    }

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t i)
    {
        const T low = low_[i];
        return low + (high_[i] - low) * bg.template next_uniform<T>();
    }
};

// Normal with mean a and standard deviation b
template <typename T>
class NormalSampler
{
private:
    Param<T> loc_;
    Param<T> scale_;
    StandardNormal<T> normal_{};

public:
    NormalSampler(const Param<T> &loc, const Param<T> &scale)
        : loc_(loc), scale_(scale)
    {
    }

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t i)
    {
        return loc_[i] + scale_[i] * normal_(bg);
    }
};

// Log-normal: exp of a normal with mean a and standard deviation b
template <typename T>
class LognormalSampler
{
private:
    Param<T> mean_;
    Param<T> sigma_;
    StandardNormal<T> normal_{};

public:
    LognormalSampler(const Param<T> &mean, const Param<T> &sigma)
        : mean_(mean), sigma_(sigma)
    {
    }

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t i)
    {
        return sycl::exp(mean_[i] + sigma_[i] * normal_(bg));
    }
};

// Exponential with scale a
template <typename T>
class ExponentialSampler
{
private:
    Param<T> scale_;

public:
    ExponentialSampler(const Param<T> &scale, const Param<T> &) : scale_(scale)
    {
    }

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t i)
    {
        return scale_[i] * standard_exponential<T>(bg);
    }
};

// Gamma with shape a and scale b
template <typename T>
class GammaSampler
{
private:
    Param<T> shape_;
    Param<T> scale_;
    StandardNormal<T> normal_{};

public:
    GammaSampler(const Param<T> &shape, const Param<T> &scale)
        : shape_(shape), scale_(scale)
    {
    }

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t i)
    {
        return scale_[i] * standard_gamma<T>(bg, normal_, shape_[i]);
    }
};

// Beta with shape parameters a and b
template <typename T>
class BetaSampler
{
private:
    Param<T> a_;
    Param<T> b_;
    StandardNormal<T> normal_{};

public:
    BetaSampler(const Param<T> &a, const Param<T> &b) : a_(a), b_(b) {}

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t i)
    {
        return beta<T>(bg, normal_, a_[i], b_[i]);
    }
};

// Uniform integers on the closed interval [a, b]
template <typename T>
class IntegersSampler
{
private:
    Param<T> low_;
    Param<T> high_;

public:
    IntegersSampler(const Param<T> &low, const Param<T> &high)
        : low_(low), high_(high)
    {
    }

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t i)
    {
        // Widen to 64 bits with the sign of T, then do the arithmetic
        // modulo 2**64, which is exact for any low <= high.
        using wideT = std::conditional_t<std::is_signed_v<T>, std::int64_t,
                                         std::uint64_t>;

        const std::uint64_t off =
            static_cast<std::uint64_t>(static_cast<wideT>(low_[i]));
        const std::uint64_t rng =
            static_cast<std::uint64_t>(static_cast<wideT>(high_[i])) - off;

        return static_cast<T>(off + bounded_uint64(bg, rng));
    }
};
} // namespace dpnp::kernels::rng
