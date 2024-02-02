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
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>

#include <dpctl4pybind11.hpp>

class EngineBase {
public:
    virtual ~EngineBase() {}
    virtual sycl::queue get_queue() = 0;
    virtual std::string print() = 0;
    // auto get_engine() {
    //     return nullptr;
    // }
};

class MRG32k3a : public EngineBase {
public:
    sycl::queue q_;
    const std::uint32_t seed_;
    const std::uint64_t offset_;

// public:
    MRG32k3a(sycl::queue &q, std::uint32_t seed, std::uint64_t offset = 0) : q_(q), seed_(seed), offset_(offset) {}

    sycl::queue get_queue() override {
        return q_;
    }

    std::string print() override {
        return "seed = " + std::to_string(seed_) + ", offset = " + std::to_string(offset_);
    }

    // auto get_engine() override {
    //     return oneapi::mkl::rng::device::mrg32k3a<8>(seed_, offset_);
    // }

    // using engine_type = oneapi::mkl::rng::device::mrg32k3a<8>;
};

namespace dpnp::backend::ext::rng::device
{
extern std::pair<sycl::event, sycl::event> gaussian(EngineBase *engine,
                                             const std::uint8_t method_id,
                                             const std::uint32_t seed,
                                             const double mean,
                                             const double stddev,
                                             const std::uint64_t n,
                                             dpctl::tensor::usm_ndarray res,
                                             const std::vector<sycl::event> &depends = {});

extern void init_gaussian_dispatch_table(void);
} // namespace dpnp::backend::ext::rng::device
