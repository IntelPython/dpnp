//*****************************************************************************
// Copyright (c) 2022-2023, Intel Corporation
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

#include "dpnp_random_state.hpp"
#include <oneapi/mkl/rng.hpp>

namespace mkl_rng = oneapi::mkl::rng;

void MT19937_InitScalarSeed(mt19937_struct *mt19937, DPCTLSyclQueueRef q_ref, uint32_t seed)
{
    sycl::queue *q = reinterpret_cast<sycl::queue *>(q_ref);
    mt19937->engine = new mkl_rng::mt19937(*q, seed);
}

void MT19937_InitVectorSeed(mt19937_struct *mt19937, DPCTLSyclQueueRef q_ref, uint32_t *seed, unsigned int n) {
    sycl::queue *q = reinterpret_cast<sycl::queue *>(q_ref);
    
    switch (n) {
        case 1: mt19937->engine = new mkl_rng::mt19937(*q, {seed[0]}); break;
        case 2: mt19937->engine = new mkl_rng::mt19937(*q, {seed[0], seed[1]}); break;
        case 3: mt19937->engine = new mkl_rng::mt19937(*q, {seed[0], seed[1], seed[2]}); break;
        default:
        // TODO need to get rid of the limitation for seed vector length
        throw std::runtime_error("Too long seed vector");
    }
}

void MT19937_Delete(mt19937_struct *mt19937) {
    mkl_rng::mt19937 *engine = static_cast<mkl_rng::mt19937 *>(mt19937->engine);
    mt19937->engine = nullptr;
    delete engine;
}

void MCG59_InitScalarSeed(mcg59_struct* mcg59, DPCTLSyclQueueRef q_ref, uint64_t seed)
{
    sycl::queue* q = reinterpret_cast<sycl::queue*>(q_ref);
    mcg59->engine = new mkl_rng::mcg59(*q, seed);
}

void MCG59_Delete(mcg59_struct* mcg59)
{
    mkl_rng::mcg59* engine = static_cast<mkl_rng::mcg59*>(mcg59->engine);
    mcg59->engine = nullptr;
    delete engine;
}
