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

#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>

#include <sycl/sycl.hpp>

#include "distributions.hpp"

namespace dpnp::kernels::rng
{
// A sampler holds the parameters of a distribution; operator() returns the
// sample for output element `i`. It may update its own state, such as a
// cached normal, so each work-item works on a copy.
//
// `result_type` is the type of the samples, and `param_types` lists the
// type of each parameter, in the order the constructor takes them.
//
// `halves_per_item` is the number of 32-bit halves of raw 64-bit outputs
// the sampler takes for every element, if that number is fixed, and 0
// otherwise.
//
// `words_per_item` is the number of 32-bit words the sampler takes per
// element, on average over items_per_work_item consecutive elements of one
// work-item, if that number is fixed whatever the parameters and the
// draws, and 0 otherwise. A rejection loop, even one that almost never
// repeats, makes it 0.
template <typename ResT, typename... ParamTs>
struct SamplerBase
{
    using result_type = ResT;
    using param_types = std::tuple<ParamTs...>;

    static constexpr unsigned halves_per_item = 0;
    static constexpr unsigned words_per_item = 0;
};

// The number of 32-bit words one uniform value of type T takes
template <typename T>
inline constexpr unsigned uniform_words = sizeof(T) / 4;

// The raw 64-bit, or 32-bit, output of the bit generator
template <typename T>
class RawSampler : public SamplerBase<T>
{
public:
    static constexpr unsigned halves_per_item = sizeof(T) / 4;
    static constexpr unsigned words_per_item = halves_per_item;

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t)
    {
        if constexpr (sizeof(T) == 8) {
            return bg.next_uint64();
        }
        else {
            return bg.next_uint32();
        }
    }
};

// Uniform on [low, high)
template <typename T>
class UniformSampler : public SamplerBase<T, T, T>
{
private:
    Param<T> low_;
    Param<T> high_;

public:
    static constexpr unsigned halves_per_item = sizeof(T) / 4;
    static constexpr unsigned words_per_item = halves_per_item;

    UniformSampler(const Param<T> &low, const Param<T> &high)
        : low_(low), high_(high)
    {
    }

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t i)
    {
        // no fused multiply-add, to round as NumPy does
#pragma clang fp contract(off)
        const T low = low_[i];
        return low + (high_[i] - low) * bg.template next_uniform<T>();
    }
};

// Uniform integers on the closed interval [low, high]
template <typename T>
class IntegersSampler : public SamplerBase<T, T, T>
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

// Normal with mean `loc` and standard deviation `scale`
template <typename T>
class NormalSampler : public SamplerBase<T, T, T>
{
private:
    Param<T> loc_;
    Param<T> scale_;
    StandardNormal<T> normal_{};

public:
    // one normal, half a pair of uniform values, per element
    static constexpr unsigned words_per_item = uniform_words<T>;

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

// Log-normal: exp of a normal with mean `mean` and standard deviation `sigma`
template <typename T>
class LognormalSampler : public SamplerBase<T, T, T>
{
private:
    Param<T> mean_;
    Param<T> sigma_;
    StandardNormal<T> normal_{};

public:
    // one normal, half a pair of uniform values, per element
    static constexpr unsigned words_per_item = uniform_words<T>;

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

template <typename T>
class StandardCauchySampler : public SamplerBase<T>
{
private:
    StandardNormal<T> normal_{};

public:
    // two normals per element
    static constexpr unsigned words_per_item = 2 * uniform_words<T>;

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t)
    {
        return standard_cauchy<T>(bg, normal_);
    }
};

// Wald (inverse Gaussian) with mean `mean` and shape `scale`
template <typename T>
class WaldSampler : public SamplerBase<T, T, T>
{
private:
    Param<T> mean_;
    Param<T> scale_;
    StandardNormal<T> normal_{};

public:
    // one normal and one uniform value per element
    static constexpr unsigned words_per_item = 2 * uniform_words<T>;

    WaldSampler(const Param<T> &mean, const Param<T> &scale)
        : mean_(mean), scale_(scale)
    {
    }

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t i)
    {
        return wald<T>(bg, normal_, mean_[i], scale_[i]);
    }
};

// Exponential with scale `scale`
template <typename T>
class ExponentialSampler : public SamplerBase<T, T>
{
private:
    Param<T> scale_;

public:
    // one uniform value per element
    static constexpr unsigned words_per_item = uniform_words<T>;

    explicit ExponentialSampler(const Param<T> &scale) : scale_(scale) {}

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t i)
    {
        return scale_[i] * standard_exponential<T>(bg);
    }
};

// Pareto II (Lomax) with shape `a`
template <typename T>
class ParetoSampler : public SamplerBase<T, T>
{
private:
    Param<T> a_;

public:
    // one uniform value per element
    static constexpr unsigned words_per_item = uniform_words<T>;

    explicit ParetoSampler(const Param<T> &a) : a_(a) {}

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t i)
    {
        return sycl::expm1(standard_exponential<T>(bg) / a_[i]);
    }
};

// Weibull with shape `a`
template <typename T>
class WeibullSampler : public SamplerBase<T, T>
{
private:
    Param<T> a_;

public:
    // one uniform value per element
    static constexpr unsigned words_per_item = uniform_words<T>;

    explicit WeibullSampler(const Param<T> &a) : a_(a) {}

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t i)
    {
        const T a = a_[i];
        const T E = standard_exponential<T>(bg);
        if (a == T(0)) {
            return T(0);
        }
        return sycl::pow(E, T(1) / a);
    }
};

// Power distribution with exponent `a` - 1, on [0, 1]
template <typename T>
class PowerSampler : public SamplerBase<T, T>
{
private:
    Param<T> a_;

public:
    // one uniform value per element
    static constexpr unsigned words_per_item = uniform_words<T>;

    explicit PowerSampler(const Param<T> &a) : a_(a) {}

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t i)
    {
        const T E = standard_exponential<T>(bg);
        return sycl::pow(-sycl::expm1(-E), T(1) / a_[i]);
    }
};

// Rayleigh with mode `scale`
template <typename T>
class RayleighSampler : public SamplerBase<T, T>
{
private:
    Param<T> scale_;

public:
    // one uniform value per element
    static constexpr unsigned words_per_item = uniform_words<T>;

    explicit RayleighSampler(const Param<T> &scale) : scale_(scale) {}

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t i)
    {
        return scale_[i] * sycl::sqrt(T(2) * standard_exponential<T>(bg));
    }
};

// Gamma with shape `shape` and scale `scale`
template <typename T>
class GammaSampler : public SamplerBase<T, T, T>
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

// Beta with shape parameters `a` and `b`
template <typename T>
class BetaSampler : public SamplerBase<T, T, T>
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

// Chi-square with `df` degrees of freedom
template <typename T>
class ChisquareSampler : public SamplerBase<T, T>
{
private:
    Param<T> df_;
    StandardNormal<T> normal_{};

public:
    explicit ChisquareSampler(const Param<T> &df) : df_(df) {}

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t i)
    {
        return chisquare<T>(bg, normal_, df_[i]);
    }
};

// Student's t with `df` degrees of freedom
template <typename T>
class StandardTSampler : public SamplerBase<T, T>
{
private:
    Param<T> df_;
    StandardNormal<T> normal_{};

public:
    explicit StandardTSampler(const Param<T> &df) : df_(df) {}

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t i)
    {
        return standard_t<T>(bg, normal_, df_[i]);
    }
};

// F with `dfnum` and `dfden` degrees of freedom
template <typename T>
class FSampler : public SamplerBase<T, T, T>
{
private:
    Param<T> dfnum_;
    Param<T> dfden_;
    StandardNormal<T> normal_{};

public:
    FSampler(const Param<T> &dfnum, const Param<T> &dfden)
        : dfnum_(dfnum), dfden_(dfden)
    {
    }

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t i)
    {
        return f<T>(bg, normal_, dfnum_[i], dfden_[i]);
    }
};

// Noncentral chi-square with `df` degrees of freedom and noncentrality `nonc`
template <typename T>
class NoncentralChisquareSampler : public SamplerBase<T, T, T>
{
private:
    Param<T> df_;
    Param<T> nonc_;
    StandardNormal<T> normal_{};

public:
    NoncentralChisquareSampler(const Param<T> &df, const Param<T> &nonc)
        : df_(df), nonc_(nonc)
    {
    }

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t i)
    {
        return noncentral_chisquare<T>(bg, normal_, df_[i], nonc_[i]);
    }
};

// Noncentral F with `dfnum` and `dfden` degrees of freedom and noncentrality
// `nonc`
template <typename T>
class NoncentralFSampler : public SamplerBase<T, T, T, T>
{
private:
    Param<T> dfnum_;
    Param<T> dfden_;
    Param<T> nonc_;
    StandardNormal<T> normal_{};

public:
    NoncentralFSampler(const Param<T> &dfnum,
                       const Param<T> &dfden,
                       const Param<T> &nonc)
        : dfnum_(dfnum), dfden_(dfden), nonc_(nonc)
    {
    }

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t i)
    {
        return noncentral_f<T>(bg, normal_, dfnum_[i], dfden_[i], nonc_[i]);
    }
};

// Laplace with location `loc` and scale `scale`
template <typename T>
class LaplaceSampler : public SamplerBase<T, T, T>
{
private:
    Param<T> loc_;
    Param<T> scale_;

public:
    LaplaceSampler(const Param<T> &loc, const Param<T> &scale)
        : loc_(loc), scale_(scale)
    {
    }

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t i)
    {
        return laplace<T>(bg, loc_[i], scale_[i]);
    }
};

// Gumbel with location `loc` and scale `scale`
template <typename T>
class GumbelSampler : public SamplerBase<T, T, T>
{
private:
    Param<T> loc_;
    Param<T> scale_;

public:
    GumbelSampler(const Param<T> &loc, const Param<T> &scale)
        : loc_(loc), scale_(scale)
    {
    }

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t i)
    {
        return gumbel<T>(bg, loc_[i], scale_[i]);
    }
};

// Logistic with location `loc` and scale `scale`
template <typename T>
class LogisticSampler : public SamplerBase<T, T, T>
{
private:
    Param<T> loc_;
    Param<T> scale_;

public:
    LogisticSampler(const Param<T> &loc, const Param<T> &scale)
        : loc_(loc), scale_(scale)
    {
    }

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t i)
    {
        return logistic<T>(bg, loc_[i], scale_[i]);
    }
};

// Triangular on [left, right] with mode `mode`
template <typename T>
class TriangularSampler : public SamplerBase<T, T, T, T>
{
private:
    Param<T> left_;
    Param<T> mode_;
    Param<T> right_;

public:
    // one uniform value per element
    static constexpr unsigned words_per_item = uniform_words<T>;

    TriangularSampler(const Param<T> &left,
                      const Param<T> &mode,
                      const Param<T> &right)
        : left_(left), mode_(mode), right_(right)
    {
    }

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t i)
    {
        return triangular<T>(bg, left_[i], mode_[i], right_[i]);
    }
};

// von Mises with mode `mu` and concentration `kappa`, on [-pi, pi]
template <typename T>
class VonmisesSampler : public SamplerBase<T, T, T>
{
private:
    Param<T> mu_;
    Param<T> kappa_;
    StandardNormal<T> normal_{};

public:
    VonmisesSampler(const Param<T> &mu, const Param<T> &kappa)
        : mu_(mu), kappa_(kappa)
    {
    }

    template <typename BitGenT>
    T operator()(BitGenT &bg, const std::size_t i)
    {
        return vonmises<T>(bg, normal_, mu_[i], kappa_[i]);
    }
};

// The discrete samplers below give int64 samples, and compute in the
// floating-point type T.

// Poisson with rate `lam`
template <typename T>
class PoissonSampler : public SamplerBase<std::int64_t, T>
{
private:
    Param<T> lam_;

public:
    explicit PoissonSampler(const Param<T> &lam) : lam_(lam) {}

    template <typename BitGenT>
    std::int64_t operator()(BitGenT &bg, const std::size_t i)
    {
        return poisson<T>(bg, lam_[i]);
    }
};

// The number of failures before `n` successes of probability `p`
template <typename T>
class NegativeBinomialSampler : public SamplerBase<std::int64_t, T, T>
{
private:
    Param<T> n_;
    Param<T> p_;
    StandardNormal<T> normal_{};

public:
    NegativeBinomialSampler(const Param<T> &n, const Param<T> &p) : n_(n), p_(p)
    {
    }

    template <typename BitGenT>
    std::int64_t operator()(BitGenT &bg, const std::size_t i)
    {
        return negative_binomial<T>(bg, normal_, n_[i], p_[i]);
    }
};

// The number of successes of `n` trials of probability `p`
template <typename T>
class BinomialSampler : public SamplerBase<std::int64_t, std::int64_t, T>
{
private:
    Param<std::int64_t> n_;
    Param<T> p_;

public:
    BinomialSampler(const Param<std::int64_t> &n, const Param<T> &p)
        : n_(n), p_(p)
    {
    }

    template <typename BitGenT>
    std::int64_t operator()(BitGenT &bg, const std::size_t i)
    {
        return binomial<T>(bg, n_[i], p_[i]);
    }
};

// The number of trials of probability `p` up to the first success
template <typename T>
class GeometricSampler : public SamplerBase<std::int64_t, T>
{
private:
    Param<T> p_;

public:
    explicit GeometricSampler(const Param<T> &p) : p_(p) {}

    template <typename BitGenT>
    std::int64_t operator()(BitGenT &bg, const std::size_t i)
    {
        return geometric<T>(bg, p_[i]);
    }
};

// Logarithmic series with shape `p`
template <typename T>
class LogseriesSampler : public SamplerBase<std::int64_t, T>
{
private:
    Param<T> p_;

public:
    explicit LogseriesSampler(const Param<T> &p) : p_(p) {}

    template <typename BitGenT>
    std::int64_t operator()(BitGenT &bg, const std::size_t i)
    {
        return logseries<T>(bg, p_[i]);
    }
};

// Zipf with exponent `a`
template <typename T>
class ZipfSampler : public SamplerBase<std::int64_t, T>
{
private:
    Param<T> a_;

public:
    explicit ZipfSampler(const Param<T> &a) : a_(a) {}

    template <typename BitGenT>
    std::int64_t operator()(BitGenT &bg, const std::size_t i)
    {
        return zipf<T>(bg, a_[i]);
    }
};

// The number of good items of `sample` drawn without replacement from `good`
// good and `bad` bad items
template <typename T>
class HypergeometricSampler
    : public SamplerBase<std::int64_t, std::int64_t, std::int64_t, std::int64_t>
{
private:
    Param<std::int64_t> good_;
    Param<std::int64_t> bad_;
    Param<std::int64_t> sample_;

public:
    HypergeometricSampler(const Param<std::int64_t> &good,
                          const Param<std::int64_t> &bad,
                          const Param<std::int64_t> &sample)
        : good_(good), bad_(bad), sample_(sample)
    {
    }

    template <typename BitGenT>
    std::int64_t operator()(BitGenT &bg, const std::size_t i)
    {
        return hypergeometric<T>(bg, good_[i], bad_[i], sample_[i]);
    }
};
} // namespace dpnp::kernels::rng
