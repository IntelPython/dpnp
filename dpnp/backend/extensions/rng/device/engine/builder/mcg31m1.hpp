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

#include <oneapi/mkl/rng/device.hpp>

#include "base_builder.hpp"
#include "base_engine.hpp"

namespace dpnp::backend::ext::rng::device::engine::builder
{
namespace mkl_rng_dev = oneapi::mkl::rng::device;

template <std::int32_t VecSize>
class Builder<mkl_rng_dev::mcg31m1<VecSize>>
    : public BaseBuilder<mkl_rng_dev::mcg31m1<VecSize>,
                         std::uint32_t,
                         std::uint64_t>
{
public:
    using EngineType = mkl_rng_dev::mcg31m1<VecSize>;

    Builder(EngineBase *engine)
        : BaseBuilder<EngineType, std::uint32_t, std::uint64_t>(engine)
    {
    }
};
} // namespace dpnp::backend::ext::rng::device::engine::builder
