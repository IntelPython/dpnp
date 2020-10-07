//*****************************************************************************
// Copyright (c) 2016-2020, Intel Corporation
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

#include <iostream>

#include <ctime>
#include <mkl_sycl.hpp>

#include <backend_iface.hpp>
#include "queue_sycl.hpp"

#include "rng_engine.hpp"

namespace mkl_rng = oneapi::mkl::rng;

mkl_rng::mt19937* engine_rng::mt19937_engine = nullptr;

void engine_rng::engine_rng_init()
{
    if (mt19937_engine)
    {
        engine_rng::destroy();
    }
    // TODO:
    // choose engine as is in numpy
    // seed number
    // TODO:
    // mem leak
    size_t seed = std::time(nullptr);
    mt19937_engine = new mkl_rng::mt19937(DPNP_QUEUE, seed);
}

void engine_rng::engine_rng_init(size_t seed)
{
    if (mt19937_engine)
    {
        engine_rng::destroy();
    }
    mt19937_engine = new mkl_rng::mt19937(DPNP_QUEUE, seed);
}

void dpnp_engine_rng_initialize()
{
    engine_rng::engine_rng_init();
}

void dpnp_engine_rng_initialize(size_t seed)
{
    engine_rng::engine_rng_init(seed);
}
