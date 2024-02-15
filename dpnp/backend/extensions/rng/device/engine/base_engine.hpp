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
class EngineType {
public:
    enum Type : std::uint8_t {
        MRG32k3a = 0,
        PHILOX4x32x10,
        MCG31M1,
        MCG59,
        Base, // must be the last always
    };

    EngineType() = default;
    constexpr EngineType(Type type) : type_(type) {}

    constexpr std::uint8_t id() const {
        return static_cast<std::uint8_t>(type_);
    }

    static constexpr std::uint8_t base_id() {
        return EngineType(Base).id();
    }

private:
  Type type_;
};

// A total number of supported engines == EngineType::Base
constexpr int no_of_engines = EngineType::base_id();

class EngineBase {
public:
    virtual ~EngineBase() {}
    virtual sycl::queue &get_queue() = 0;

    virtual EngineType get_type() const noexcept {
        return EngineType::Base;
    }

    virtual std::vector<std::uint64_t> get_seeds() const noexcept {
        return std::vector<std::uint64_t>();
    }

    virtual std::vector<std::uint64_t> get_offsets() const noexcept {
        return std::vector<std::uint64_t>();
    }
};
} // dpnp::backend::ext::rng::device::engine
