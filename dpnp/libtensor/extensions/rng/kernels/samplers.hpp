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
#include <tuple>

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
// `words_per_item` is the number of 32-bit words the sampler takes per
// element, on average over the items_per_work_item consecutive elements of one
// work-item, if that number is fixed whatever the parameters and the
// draws, and 0 otherwise. A rejection loop, even one that almost never
// repeats, makes it 0.
template <typename ResT, typename... ParamTs>
struct SamplerBase
{
    using result_type = ResT;
    using param_types = std::tuple<ParamTs...>;

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
    static constexpr unsigned words_per_item = sizeof(T) / 4;

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
    static constexpr unsigned words_per_item = sizeof(T) / 4;

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
} // namespace dpnp::kernels::rng
