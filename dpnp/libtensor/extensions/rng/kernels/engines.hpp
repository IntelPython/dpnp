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
#include <type_traits>

#include <sycl/sycl.hpp>

#include <oneapi/mkl/rng/device.hpp>

namespace dpnp::kernels::rng
{
namespace mkl_rng_dev = oneapi::mkl::rng::device;

// The maximum number of 64-bit seed words any supported engine takes.
inline constexpr std::size_t max_seed_words = 1;

// Host-side state of a bit generator, passed by value into kernels.
//
// A generator never keeps state on the device: every kernel launch rebuilds
// one engine per work-item from this description. Work-item `w` of a launch
// owns the counter block `counter + w * 2**64` (for Philox, in units of
// 128-bit counter increments), so the stream consumed by a work-item cannot
// overlap the stream of any other work-item, of this or any later launch,
// no matter how many draws a rejection sampler takes.
struct EngineState
{
    std::uint64_t seed[max_seed_words];
    std::uint64_t counter_lo;
    std::uint64_t counter_hi;
};

// Supported engines. The values are part of the Python-facing API of the
// extension and must stay in sync with dpnp.random's bit generators.
enum class engine_id : int
{
    philox4x32x10 = 0,
};
inline constexpr int num_engines = 1;

struct Philox4x32x10
{
    using engine_t = mkl_rng_dev::philox4x32x10<1>;

    // Philox takes the key and the initial 128-bit counter as the seed
    // words {key, counter_lo, counter_hi}, so a work-item's block is set up
    // directly, with no skip-ahead.
    static engine_t make(const EngineState &state, std::uint64_t subsequence)
    {
        return engine_t(
            {state.seed[0], state.counter_lo, state.counter_hi + subsequence});
    }
};

// Adapts a oneMKL device engine to the next_uint32/next_uint64/next_float/
// next_double interface NumPy's distribution algorithms are written against.
template <typename EngineT>
class BitGen
{
private:
    EngineT engine_;
    mkl_rng_dev::bits<std::uint32_t> bits_{};

public:
    explicit BitGen(const EngineT &engine) : engine_(engine) {}

    std::uint32_t next_uint32()
    {
        return mkl_rng_dev::generate(bits_, engine_);
    }

    std::uint64_t next_uint64()
    {
        const std::uint64_t hi = next_uint32();
        const std::uint64_t lo = next_uint32();
        return (hi << 32) | lo;
    }

    // 24 random bits, uniform on [0, 1)
    float next_float()
    {
        return static_cast<float>(next_uint32() >> 8) * (1.0f / 16777216.0f);
    }

    // 53 random bits, uniform on [0, 1)
    double next_double()
    {
        return static_cast<double>(next_uint64() >> 11) *
               (1.0 / 9007199254740992.0);
    }

    template <typename T>
    T next_uniform()
    {
        if constexpr (std::is_same_v<T, double>) {
            return next_double();
        }
        else {
            return next_float();
        }
    }
};
} // namespace dpnp::kernels::rng
