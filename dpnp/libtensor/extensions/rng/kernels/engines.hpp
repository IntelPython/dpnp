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
inline constexpr std::size_t max_seed_words = 10;

// Host-side state of a bit generator, passed by value into kernels.
//
// A generator never keeps state on the device: every kernel launch rebuilds
// one engine per work-item from this description, and each work-item draws
// from its own part of the stream, which cannot overlap the part of any
// other work-item, of this or any later launch, no matter how many draws a
// rejection sampler takes. For Philox, work-item `w` of a launch owns the
// counter block `counter + w * 2**64`, in units of 128-bit counter
// increments. PCG64DXSM keeps its whole state in the seed words.
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
    pcg64dxsm = 1,
};
inline constexpr int num_engines = 2;

// The conversions of raw bits to uniform floating-point values, for a bit
// generator `Derived` providing next_uint32() and next_uint64(). They are
// those of NumPy.
template <typename Derived>
class UniformBits
{
public:
    // 24 random bits, uniform on [0, 1)
    float next_float()
    {
        return static_cast<float>(self().next_uint32() >> 8) *
               (1.0f / 16777216.0f);
    }

    // 53 random bits, uniform on [0, 1)
    double next_double()
    {
        return static_cast<double>(self().next_uint64() >> 11) *
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

private:
    Derived &self() { return static_cast<Derived &>(*this); }
};

// Adapts a oneMKL device engine to the next_uint32/next_uint64/next_float/
// next_double interface NumPy's distribution algorithms are written against.
template <typename EngineT>
class BitGen : public UniformBits<BitGen<EngineT>>
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
};

// The first N words of the stream of BitGen<EngineT>, for a oneMKL device
// engine of vector size N, generated together in one engine call. The
// sampler must draw no more than N words (see SamplerBase::words_per_item);
// their positions are then known at compile time in a fully unrolled loop,
// so the words stay in registers. That needs everything touching them
// inlined: when the compiler instead returned the generator from an
// out-of-line call, the words lived in private memory and the kernel ran
// 2-3 times slower.
template <typename EngineT>
class BlockBitGen : public UniformBits<BlockBitGen<EngineT>>
{
private:
    static constexpr int N = EngineT::vec_size;

    sycl::vec<std::uint32_t, N> words_;
    int pos_ = 0;

public:
    __attribute__((always_inline)) explicit BlockBitGen(EngineT engine)
    {
        mkl_rng_dev::bits<std::uint32_t> bits;
        words_ = mkl_rng_dev::generate(bits, engine);
    }

    __attribute__((always_inline)) std::uint32_t next_uint32()
    {
        return words_[pos_++];
    }

    __attribute__((always_inline)) std::uint64_t next_uint64()
    {
        const std::uint64_t hi = next_uint32();
        const std::uint64_t lo = next_uint32();
        return (hi << 32) | lo;
    }
};

struct Philox4x32x10
{
    using engine_t = mkl_rng_dev::philox4x32x10<1>;
    using bitgen_t = BitGen<engine_t>;

    // Whether the fixed-consumption samplers draw the stream in order,
    // see SampleFunctor
    static constexpr bool ordered = false;

    // The most words make_block can generate together
    static constexpr unsigned max_block_words = 16;

    // Philox takes the key and the initial 128-bit counter as the seed
    // words {key, counter_lo, counter_hi}, so a work-item's block is set up
    // directly, with no skip-ahead.
    static bitgen_t make(const EngineState &state, std::uint64_t subsequence)
    {
        return bitgen_t(engine_t(
            {state.seed[0], state.counter_lo, state.counter_hi + subsequence}));
    }

    template <unsigned Words>
    using block_engine_t = mkl_rng_dev::philox4x32x10<Words>;

    // The stream of make(), with its first `Words` words generated together
    template <unsigned Words>
    __attribute__((always_inline)) static BlockBitGen<block_engine_t<Words>>
        make_block(const EngineState &state, std::uint64_t subsequence)
    {
        static_assert(Words == 8 || Words == 16);
        return BlockBitGen<block_engine_t<Words>>(block_engine_t<Words>(
            {state.seed[0], state.counter_lo, state.counter_hi + subsequence}));
    }
};

// An unsigned 128-bit integer, with the arithmetic modulo 2**128 the LCG of
// PCG64 needs
struct UInt128
{
    std::uint64_t lo;
    std::uint64_t hi;
};

inline UInt128 operator+(const UInt128 &a, const UInt128 &b)
{
    const std::uint64_t lo = a.lo + b.lo;
    return {lo, a.hi + b.hi + (lo < a.lo ? 1 : 0)};
}

inline UInt128 operator*(const UInt128 &a, const UInt128 &b)
{
    return {a.lo * b.lo, sycl::mul_hi(a.lo, b.lo) + a.lo * b.hi + a.hi * b.lo};
}

// The affine map x -> mult * x + inc modulo 2**128, as a step of an LCG
struct Affine128
{
    UInt128 mult;
    UInt128 inc;

    UInt128 operator()(const UInt128 &x) const { return mult * x + inc; }
};

// The map `m` applied `k` times, in O(log k) steps (F. Brown, "Random
// number generation with arbitrary stride", 1994)
inline Affine128 affine_power(Affine128 m, std::uint64_t k)
{
    Affine128 acc{{1, 0}, {0, 0}};
    while (k != 0) {
        if (k & 1) {
            acc = {acc.mult * m.mult, acc.inc * m.mult + m.inc};
        }
        m = {m.mult * m.mult, (m.mult + UInt128{1, 0}) * m.inc};
        k >>= 1;
    }
    return acc;
}

// NumPy's PCG64DXSM: a 128-bit LCG with the cheap 64-bit multiplier and an
// odd increment, whose output is the DXSM permutation of the state before
// the step. 32-bit draws take the low half of an output, and keep its high
// half for the next one, as NumPy does.
class Pcg64Dxsm : public UniformBits<Pcg64Dxsm>
{
private:
    UInt128 state_;
    UInt128 inc_;
    bool has_uint32_ = false;
    std::uint32_t uinteger_ = 0;

public:
    static constexpr std::uint64_t cheap_multiplier = 0xda942042e4dd58b5ULL;

    Pcg64Dxsm(const UInt128 &state, const UInt128 &inc)
        : state_(state), inc_(inc)
    {
    }

    // Use `value` for the next 32-bit draw
    void set_buffered(std::uint32_t value)
    {
        has_uint32_ = true;
        uinteger_ = value;
    }

    std::uint64_t next_uint64()
    {
        std::uint64_t hi = state_.hi;
        const std::uint64_t lo = state_.lo | 1;
        state_ = state_ * UInt128{cheap_multiplier, 0} + inc_;

        hi ^= hi >> 32;
        hi *= cheap_multiplier;
        hi ^= hi >> 48;
        hi *= lo;
        return hi;
    }

    std::uint32_t next_uint32()
    {
        if (has_uint32_) {
            has_uint32_ = false;
            return uinteger_;
        }
        const std::uint64_t next = next_uint64();
        set_buffered(static_cast<std::uint32_t>(next >> 32));
        return static_cast<std::uint32_t>(next);
    }
};

struct PCG64DXSM
{
    using bitgen_t = Pcg64Dxsm;

    // The samplers that take a fixed number of 32-bit halves of raw outputs
    // per element draw the stream in order, as NumPy does
    static constexpr bool ordered = true;

    // No make_block
    static constexpr unsigned max_block_words = 0;

    // The seed words are the state, the increment, the map of the stride of
    // NumPy's jumped() (the step applied (phi - 1) * 2**128 times), each as
    // {lo, hi}, then NumPy's has_uint32 and uinteger.
    static UInt128 word128(const EngineState &state, std::size_t i)
    {
        return {state.seed[i], state.seed[i + 1]};
    }

    // The stream jumped `subsequence` times, as by NumPy's jumped(), for the
    // samplers whose draws per element vary
    static bitgen_t make(const EngineState &state, std::uint64_t subsequence)
    {
        const Affine128 stride{word128(state, 4), word128(state, 6)};
        return bitgen_t(affine_power(stride, subsequence)(word128(state, 0)),
                        word128(state, 2));
    }

    // The stream at element `first` of a sampler taking `halves` 32-bit
    // halves of raw outputs per element
    static bitgen_t
        make_at(const EngineState &state, std::uint64_t first, unsigned halves)
    {
        const UInt128 inc = word128(state, 2);
        const Affine128 step{{Pcg64Dxsm::cheap_multiplier, 0}, inc};
        const UInt128 start = word128(state, 0);

        if (halves == 2) {
            // 64-bit draws, which leave the buffered half alone
            return bitgen_t(affine_power(step, first)(start), inc);
        }

        // 32-bit draws, the first of which is the buffered half, if any
        const bool has_uint32 = state.seed[8] != 0;
        if (has_uint32 && first == 0) {
            bitgen_t bg(start, inc);
            bg.set_buffered(static_cast<std::uint32_t>(state.seed[9]));
            return bg;
        }
        const std::uint64_t half = has_uint32 ? first - 1 : first;
        bitgen_t bg(affine_power(step, half / 2)(start), inc);
        if (half % 2 != 0) {
            // start with the high half of an output
            bg.next_uint32();
        }
        return bg;
    }
};
} // namespace dpnp::kernels::rng
