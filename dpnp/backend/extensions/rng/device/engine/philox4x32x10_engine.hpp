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

#include "base_engine.hpp"


namespace dpnp::backend::ext::rng::device::engine
{
class PHILOX4x32x10 : public EngineBase {
public:
    PHILOX4x32x10(sycl::queue &q, std::uint64_t seed, std::uint64_t offset = 0) :
        EngineBase(q, seed, offset) {}

    PHILOX4x32x10(sycl::queue &q, std::vector<std::uint64_t> &seeds, std::uint64_t offset = 0) :
        EngineBase(q, seeds, offset) {}

    PHILOX4x32x10(sycl::queue &q, std::uint64_t seed, std::vector<std::uint64_t> &offsets) :
        EngineBase(q, seed, offsets) {}

    PHILOX4x32x10(sycl::queue &q, std::vector<std::uint64_t> &seeds, std::vector<std::uint64_t> &offsets) :
        EngineBase(q, seeds, offsets) {}

    virtual EngineType get_type() const noexcept override {
        return EngineType::PHILOX4x32x10;
    }
};
} // dpnp::backend::ext::rng::device::engine