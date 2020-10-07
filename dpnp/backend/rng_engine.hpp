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

#pragma once
#ifndef RNG_ENGINE_H
#define RNG_ENGINE_H

#include <iostream>

#include <ctime>
#include <mkl_sycl.hpp>

#include "backend_utils.hpp"
#include "queue_sycl.hpp"

namespace mkl_rng = oneapi::mkl::rng;

#define DPNP_RNG_ENGINE engine_rng::get_engine()

/**
 * This is
 * TODO:
 * add docs
 */

class engine_rng
{
    // TODO:
    // static
    static mkl_rng::mt19937* mt19937_engine;

    static void destroy()
    {
        delete mt19937_engine;
        mt19937_engine = nullptr;
    }

public:
    engine_rng()
    {
        mt19937_engine = nullptr;
    }

    ~engine_rng()
    {
        engine_rng::destroy();
    }

    /**
     * Explicitly disallow copying
     */
    engine_rng(const engine_rng&) = delete;
    engine_rng& operator=(const engine_rng&) = delete;

    static void engine_rng_init();
    static void engine_rng_init(size_t seed);

    static mkl_rng::mt19937& get_engine()
    {
        if (!mt19937_engine)
        {
            engine_rng_init();
        }
        return *mt19937_engine;
    }
};

#endif // RNG_ENGINE_H
