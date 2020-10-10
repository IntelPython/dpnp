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

#include <backend_iface.hpp>
#include "backend_utils.hpp"
#include "queue_sycl.hpp"

namespace mkl_rng = oneapi::mkl::rng;

template <typename _DataType, typename _Distribution>
sycl::event mkl_rng_generate_custom(_Distribution& distribution, void* result, size_t size)
{
    sycl::event event_out;
    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    EngineOptions engine_type = DPNP_RNG_ENGINE_TYPE;
    switch(engine_type)
    {
        case EngineOptions::ARS5:
        {
            mkl_rng::ars5 * engine_ptr = reinterpret_cast<mkl_rng::ars5 *>(DPNP_RNG_ENGINE);
            event_out = mkl_rng::generate(distribution, *engine_ptr, size, result1);
            break;
        }
        case EngineOptions::MCG32M1:
        {
            mkl_rng::mcg31m1 * engine_ptr = reinterpret_cast<mkl_rng::mcg31m1 *>(DPNP_RNG_ENGINE);
            event_out = mkl_rng::generate(distribution, *engine_ptr, size, result1);
            break;
        }
        case EngineOptions::MCG59:
        {
            mkl_rng::mcg59 * engine_ptr = reinterpret_cast<mkl_rng::mcg59 *>(DPNP_RNG_ENGINE);
            event_out = mkl_rng::generate(distribution, *engine_ptr, size, result1);
            break;
        }
        case EngineOptions::MRG32K3A:
        {
            mkl_rng::mrg32k3a * engine_ptr = reinterpret_cast<mkl_rng::mrg32k3a *>(DPNP_RNG_ENGINE);
            event_out = mkl_rng::generate(distribution, *engine_ptr, size, result1);
            break;
        }
        case EngineOptions::MT19937:
        {
            mkl_rng::mt19937 * engine_ptr = reinterpret_cast<mkl_rng::mt19937 *>(DPNP_RNG_ENGINE);
            event_out = mkl_rng::generate(distribution, *engine_ptr, size, result1);
            break;
        }
        case EngineOptions::MT2203:
        {
            mkl_rng::mt2203 * engine_ptr = reinterpret_cast<mkl_rng::mt2203 *>(DPNP_RNG_ENGINE);
            event_out = mkl_rng::generate(distribution, *engine_ptr, size, result1);
            break;
        }
        case EngineOptions::NIEDERREITER:
        {
            mkl_rng::niederreiter * engine_ptr = reinterpret_cast<mkl_rng::niederreiter *>(DPNP_RNG_ENGINE);
            event_out = mkl_rng::generate(distribution, *engine_ptr, size, result1);
            break;
        }
        case EngineOptions::NONDETERMINISTIC:
        {
            mkl_rng::nondeterministic * engine_ptr = reinterpret_cast<mkl_rng::nondeterministic *>(DPNP_RNG_ENGINE);
            event_out = mkl_rng::generate(distribution, *engine_ptr, size, result1);
            break;
        }
        case EngineOptions::PHILOX4X32X10:
        {
            mkl_rng::philox4x32x10 * engine_ptr = reinterpret_cast<mkl_rng::philox4x32x10 *>(DPNP_RNG_ENGINE);
            event_out = mkl_rng::generate(distribution, *engine_ptr, size, result1);
            break;
        }
        case EngineOptions::R250:
        {
            mkl_rng::r250 * engine_ptr = reinterpret_cast<mkl_rng::r250 *>(DPNP_RNG_ENGINE);
            event_out = mkl_rng::generate(distribution, *engine_ptr, size, result1);
            break;
        }
        case EngineOptions::SFMT19937:
        {
            mkl_rng::sfmt19937 * engine_ptr = reinterpret_cast<mkl_rng::sfmt19937 *>(DPNP_RNG_ENGINE);
            event_out = mkl_rng::generate(distribution, *engine_ptr, size, result1);
            break;
        }
        case EngineOptions::SOBOL:
        {
            mkl_rng::sobol * engine_ptr = reinterpret_cast<mkl_rng::sobol *>(DPNP_RNG_ENGINE);
            event_out = mkl_rng::generate(distribution, *engine_ptr, size, result1);
            break;
        }
        case EngineOptions::WICHMANN_HILL:
        {
            mkl_rng::wichmann_hill * engine_ptr = reinterpret_cast<mkl_rng::wichmann_hill *>(DPNP_RNG_ENGINE);
            event_out = mkl_rng::generate(distribution, *engine_ptr, size, result1);
            break;
        }
    }
    return event_out;
}

// TODO:
// add mean and std params ?
template <typename _DataType>
void mkl_rng_gaussian(void* result, size_t size)
{
    if (!size)
    {
        return;
    }
    sycl::event event_out;

    const _DataType mean = _DataType(0.0);
    const _DataType stddev = _DataType(1.0);

    mkl_rng::gaussian<_DataType> distribution(mean, stddev);

    event_out = mkl_rng_generate_custom<_DataType, mkl_rng::gaussian<_DataType>>(distribution, result, size);
    event_out.wait();
}

template <typename _DataType>
void mkl_rng_uniform(void* result, long low, long high, size_t size)
{
    if (!size)
    {
        return;
    }
    sycl::event event_out;

    // set left bound of distribution
    const _DataType a = (_DataType(low));
    // set right bound of distribution
    const _DataType b = (_DataType(high));

    mkl_rng::uniform<_DataType> distribution(a, b);

    event_out = mkl_rng_generate_custom<_DataType, mkl_rng::uniform<_DataType>>(distribution, result, size);
    event_out.wait();
}

template void mkl_rng_gaussian<double>(void* result, size_t size);
template void mkl_rng_gaussian<float>(void* result, size_t size);

template void mkl_rng_uniform<int>(void* result, long low, long high, size_t size);
template void mkl_rng_uniform<float>(void* result, long low, long high, size_t size);
template void mkl_rng_uniform<double>(void* result, long low, long high, size_t size);
