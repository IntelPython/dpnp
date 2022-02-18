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

#include <CL/sycl.hpp>
#include <oneapi/mkl/rng.hpp>

namespace mkl_rng = oneapi::mkl::rng;

class IRandomState
{
public:
    void * engine;
    virtual void InitScalarSeed(DPCTLSyclQueueRef, size_t) = 0;
    virtual void InitVectorSeed(DPCTLSyclQueueRef QRef, unsigned int * seed, int n) = 0;
    virtual void Delete() = 0;
};

class mt19937_class : public IRandomState
{
public:
    mkl_rng::mt19937* engine;
    void InitScalarSeed(DPCTLSyclQueueRef QRef, size_t seed=1)
    {
        sycl::queue q = *(reinterpret_cast<sycl::queue *>(QRef));
        this->engine = new mkl_rng::mt19937(q, seed);
        return;
    }

    void InitVectorSeed(DPCTLSyclQueueRef QRef, unsigned int * seed, int n) {
        sycl::queue q = *(reinterpret_cast<sycl::queue *>(QRef));
        if (n==1) {
	        this->engine = new mkl_rng::mt19937(q, {seed[0]});
        } else if (n==2) {
	        this->engine = new mkl_rng::mt19937(q, {seed[0], seed[1]});
        } else if (n==3) {
	        this->engine = new mkl_rng::mt19937(q, {seed[0], seed[1], seed[2]});
        } else {
	    throw std::runtime_error("Too long of a seed vector");
        }
        return;
    }

    void Delete() {
        delete this->engine;
        this->engine = nullptr;
        return;
    }
};
