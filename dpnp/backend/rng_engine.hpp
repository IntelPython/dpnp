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

#ifndef RNG_ENGINE_H
#define RNG_ENGINE_H

#include <CL/sycl.hpp>
#include <mkl_sycl.hpp>

#include "backend_utils.hpp"
#include "queue_sycl.hpp"

namespace mkl_rng = oneapi::mkl::rng;

/**
 * This is 
 * TODO:
 *
 */

class engine_rng
{
    static mkl_rng::mt19937* mt19937_engine;
    static size_t seed;

public:
    engine_rng()
    {
        size_t seed = 1;
        mkl_rng::mt19937 engine(DPNP_QUEUE, seed);
        mt19937_engine = &engine;
    }

    virtual ~engine_rng()
    {
        delete mt19937_engine;
        mt19937_engine = nullptr;
    }

    /**
     * Explicitly disallow copying
     */
    engine_rng(const engine_rng&) = delete;
    engine_rng& operator=(const engine_rng&) = delete;

    static mkl_rng::mt19937& get_engine()
    {
        return * mt19937_engine;
    }
    
};

#endif // RNG_ENGINE_H