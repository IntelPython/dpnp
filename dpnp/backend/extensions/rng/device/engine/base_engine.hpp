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

#include <sycl/sycl.hpp>

namespace dpnp::backend::ext::rng::device::engine
{
class EngineType
{
public:
    enum Type : std::uint8_t
    {
        MRG32k3a = 0,
        PHILOX4x32x10,
        MCG31M1,
        MCG59,
        Base, // must be the last always
    };

    EngineType() = default;
    constexpr EngineType(Type type) : type_(type) {}

    constexpr std::uint8_t id() const
    {
        return static_cast<std::uint8_t>(type_);
    }

    static constexpr std::uint8_t base_id()
    {
        return EngineType(Base).id();
    }

private:
    Type type_;
};

// A total number of supported engines == EngineType::Base
constexpr std::uint8_t no_of_engines = EngineType::base_id();

class EngineBase
{
private:
    sycl::queue q_{};
    std::vector<std::uint64_t> seed_vec{};
    std::vector<std::uint64_t> offset_vec{};

    void validate_vec_size(const std::size_t size)
    {
        if (size > max_vec_n) {
            throw std::runtime_error("TODO: add text");
        }
    }

public:
    EngineBase() {}

    EngineBase(sycl::queue &q, std::uint64_t seed, std::uint64_t offset)
        : q_(q), seed_vec(1, seed), offset_vec(1, offset)
    {
    }

    EngineBase(sycl::queue &q,
               std::vector<std::uint64_t> &seeds,
               std::uint64_t offset)
        : q_(q), seed_vec(seeds), offset_vec(1, offset)
    {
        validate_vec_size(seeds.size());
    }

    EngineBase(sycl::queue &q,
               std::vector<std::uint32_t> &seeds,
               std::uint64_t offset)
        : q_(q), offset_vec(1, offset)
    {
        validate_vec_size(seeds.size());

        seed_vec.reserve(seeds.size());
        seed_vec.assign(seeds.begin(), seeds.end());
    }

    EngineBase(sycl::queue &q,
               std::uint64_t seed,
               std::vector<std::uint64_t> &offsets)
        : q_(q), seed_vec(1, seed), offset_vec(offsets)
    {
        validate_vec_size(offsets.size());
    }

    EngineBase(sycl::queue &q,
               std::vector<std::uint64_t> &seeds,
               std::vector<std::uint64_t> &offsets)
        : q_(q), seed_vec(seeds), offset_vec(offsets)
    {
        validate_vec_size(seeds.size());
        validate_vec_size(offsets.size());
    }

    EngineBase(sycl::queue &q,
               std::vector<std::uint32_t> &seeds,
               std::vector<std::uint64_t> &offsets)
        : q_(q), offset_vec(offsets)
    {
        validate_vec_size(seeds.size());
        validate_vec_size(offsets.size());

        seed_vec.reserve(seeds.size());
        seed_vec.assign(seeds.begin(), seeds.end());
    }

    virtual ~EngineBase() {}

    virtual EngineType get_type() const noexcept
    {
        return EngineType::Base;
    }

    sycl::queue &get_queue() noexcept
    {
        return q_;
    }

    std::vector<std::uint64_t> &get_seeds() noexcept
    {
        return seed_vec;
    }

    std::vector<std::uint64_t> &get_offsets() noexcept
    {
        return offset_vec;
    }

    //
    static constexpr std::uint8_t max_vec_n = 1;
};
} // namespace dpnp::backend::ext::rng::device::engine
