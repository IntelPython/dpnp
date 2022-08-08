//*****************************************************************************
// Copyright (c) 2016-2022, Intel Corporation
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
#ifndef BACKEND_RANDOM_STATE_H // Cython compatibility
#define BACKEND_RANDOM_STATE_H

#include <CL/sycl.hpp>
#include <oneapi/mkl/rng.hpp>

namespace mkl_rng = oneapi::mkl::rng;

struct mt19937_struct
{
    mkl_rng::mt19937* engine;
};

void MT19937_InitScalarSeed(mt19937_struct *mt19937, DPCTLSyclQueueRef QRef, uint32_t seed=1)
{
    sycl::queue q = *(reinterpret_cast<sycl::queue *>(QRef));
    mt19937->engine = new mkl_rng::mt19937(q, seed);
    return;
}

void MT19937_InitVectorSeed(mt19937_struct *mt19937, DPCTLSyclQueueRef QRef, uint32_t * seed, unsigned int n) {
    sycl::queue q = *(reinterpret_cast<sycl::queue *>(QRef));
    
    switch (n) {
        case 1: mt19937->engine = new mkl_rng::mt19937(q, {seed[0]}); break;
        case 2: mt19937->engine = new mkl_rng::mt19937(q, {seed[0], seed[1]}); break;
        case 3: mt19937->engine = new mkl_rng::mt19937(q, {seed[0], seed[1], seed[2]}); break;
        default:
        throw std::runtime_error("Too long seed vector");
    }
    return;
}

void MT19937_Delete(mt19937_struct *mt19937) {
    delete mt19937->engine;
    mt19937->engine = nullptr;
    return;
}

#endif // BACKEND_RANDOM_STATE_H
