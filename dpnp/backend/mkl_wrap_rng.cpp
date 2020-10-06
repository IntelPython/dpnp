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

#include <ctime>
#include <iostream>
#include <vector>

#include <mkl_sycl.hpp>

#include <backend_iface.hpp>
#include "backend_utils.hpp"
#include "queue_sycl.hpp"
#include "rng_engine.hpp"

namespace mkl_rng = oneapi::mkl::rng;

// TODO:
// add mean and std params ?
template <typename _DataType, typename _Engine>
void mkl_rng_gaussian(void* result, size_t size, void* engine)
{
    if (!size)
    {
        return;
    }
    _DataType* result1 = reinterpret_cast<_DataType*>(result);
    engine_rng* engn_rng = reinterpret_cast<engine_rng*>(engine);
    _Engine* engine1 = reinterpret_cast<_Engine*>(engn_rng->get_engine());

    const _DataType mean = _DataType(0.0);
    const _DataType stddev = _DataType(1.0);

    mkl_rng::gaussian<_DataType> distribution(mean, stddev);
    // perform generation
    mkl_rng::generate(distribution, *engine1, size, result1);

    DPNP_QUEUE.wait();
}

template <typename _DataType, typename _Engine>
void mkl_rng_uniform(void* result, long low, long high, size_t size, void* engine)
{
    if (!size)
    {
        return;
    }
    _DataType* result1 = reinterpret_cast<_DataType*>(result);
    engine_rng* engn_rng = reinterpret_cast<engine_rng*>(engine);
    _Engine* engine1 = reinterpret_cast<_Engine*>(engn_rng->get_engine());

    // set left bound of distribution
    const _DataType a = (_DataType(low));
    // set right bound of distribution
    const _DataType b = (_DataType(high));

    mkl_rng::uniform<_DataType> distribution(a, b);
    try
    {
        // perform generation
        mkl_rng::generate(distribution, *engine1, size, result1);
        DPNP_QUEUE.wait_and_throw();
    }
    catch (cl::sycl::exception const& e)
    {
        std::cerr << "Caught synchronous SYCL exception during mkl_rng_uniform_mt19937():\n"
                  << e.what() << "\nOpenCL status: " << e.get_cl_code() << std::endl;
    }
}

void* rng_engine_init()
{
    return new engine_rng();
}

void* rng_engine_init(size_t seed)
{
    return new engine_rng(seed);
}

void* rng_engine_set_seed(size_t seed, void* engine)
{
    engine_rng* engine_ptr = reinterpret_cast<engine_rng*>(engine);
    engine_ptr->set_seed(seed);

    return engine_ptr;
}

template void mkl_rng_gaussian<double, mkl_rng::mt19937>(void* result, size_t size, void* engine);
template void mkl_rng_gaussian<float, mkl_rng::mt19937>(void* result, size_t size, void* engine);
template void mkl_rng_gaussian<double, mkl_rng::philox4x32x10>(void* result, size_t size, void* engine);
template void mkl_rng_gaussian<float, mkl_rng::philox4x32x10>(void* result, size_t size, void* engine);

template void mkl_rng_uniform<int, mkl_rng::mt19937>(void* result, long low, long high, size_t size, void* engine);
template void mkl_rng_uniform<float, mkl_rng::mt19937>(void* result, long low, long high, size_t size, void* engine);
template void mkl_rng_uniform<double, mkl_rng::mt19937>(void* result, long low, long high, size_t size, void* engine);

template void
    mkl_rng_uniform<int, mkl_rng::philox4x32x10>(void* result, long low, long high, size_t size, void* engine);
template void
    mkl_rng_uniform<float, mkl_rng::philox4x32x10>(void* result, long low, long high, size_t size, void* engine);
template void
    mkl_rng_uniform<double, mkl_rng::philox4x32x10>(void* result, long low, long high, size_t size, void* engine);
