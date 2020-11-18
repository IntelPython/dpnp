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
#include "backend_fptr.hpp"
#include "backend_utils.hpp"
#include "queue_sycl.hpp"

namespace mkl_rng = oneapi::mkl::rng;

template <typename _DataType>
void custom_rng_beta_c(void* result, _DataType a, _DataType b, size_t size)
{
    if (!size)
    {
        return;
    }

    _DataType displacement = _DataType(0.0);

    _DataType scalefactor = _DataType(1.0);

    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    mkl_rng::beta<_DataType> distribution(a, b, displacement, scalefactor);
    // perform generation
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_out.wait();
}

template <typename _DataType>
void custom_rng_binomial_c(void* result, int ntrial, double p, size_t size)
{
    if (!size)
    {
        return;
    }
    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    mkl_rng::binomial<_DataType> distribution(ntrial, p);
    // perform generation
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_out.wait();
}

template <typename _DataType>
void custom_rng_chi_square_c(void* result, int df, size_t size)
{
    if (!size)
    {
        return;
    }
    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    mkl_rng::chi_square<_DataType> distribution(df);
    // perform generation
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_out.wait();
}

template <typename _DataType>
void custom_rng_exponential_c(void* result, _DataType beta, size_t size)
{
    if (!size)
    {
        return;
    }

    // set displacement a
    const _DataType a = (_DataType(0.0));

    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    mkl_rng::exponential<_DataType> distribution(a, beta);
    // perform generation
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_out.wait();
}

template <typename _DataType>
void custom_rng_gamma_c(void* result, _DataType shape, _DataType scale, size_t size)
{
    if (!size)
    {
        return;
    }

    // set displacement a
    const _DataType a = (_DataType(0.0));

    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    mkl_rng::gamma<_DataType> distribution(shape, a, scale);
    // perform generation
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_out.wait();
}

template <typename _DataType>
void custom_rng_gaussian_c(void* result, _DataType mean, _DataType stddev, size_t size)
{
    if (!size)
    {
        return;
    }
    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    mkl_rng::gaussian<_DataType> distribution(mean, stddev);
    // perform generation
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_out.wait();
}

template <typename _DataType>
void custom_rng_negative_binomial_c(void* result, double a, double p, size_t size)
{
    if (!size)
    {
        return;
    }
    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    mkl_rng::negative_binomial<_DataType> distribution(a, p);
    // perform generation
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_out.wait();
}

template <typename _DataType>
void custom_rng_uniform_c(void* result, long low, long high, size_t size)
{
    if (!size)
    {
        return;
    }
    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    // set left bound of distribution
    const _DataType a = (_DataType(low));
    // set right bound of distribution
    const _DataType b = (_DataType(high));

    mkl_rng::uniform<_DataType> distribution(a, b);
    // perform generation
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_out.wait();
}

void func_map_init_random(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_BETA][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_rng_beta_c<double>};
    fmap[DPNPFuncName::DPNP_FN_BETA][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_rng_beta_c<float>};

    fmap[DPNPFuncName::DPNP_FN_BINOMIAL][eft_INT][eft_INT] = {eft_INT, (void*)custom_rng_binomial_c<int>};

    fmap[DPNPFuncName::DPNP_FN_CHISQUARE][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_rng_chi_square_c<double>};
    fmap[DPNPFuncName::DPNP_FN_CHISQUARE][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_rng_chi_square_c<float>};

    fmap[DPNPFuncName::DPNP_FN_EXPONENTIAL][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_rng_exponential_c<double>};
    fmap[DPNPFuncName::DPNP_FN_EXPONENTIAL][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_rng_exponential_c<float>};

    fmap[DPNPFuncName::DPNP_FN_GAMMA][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_rng_gamma_c<double>};
    fmap[DPNPFuncName::DPNP_FN_GAMMA][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_rng_gamma_c<float>};

    fmap[DPNPFuncName::DPNP_FN_GAUSSIAN][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_rng_gaussian_c<double>};
    fmap[DPNPFuncName::DPNP_FN_GAUSSIAN][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_rng_gaussian_c<float>};

    fmap[DPNPFuncName::DPNP_FN_NEGATIVE_BINOMIAL][eft_INT][eft_INT] = {eft_INT,
                                                                       (void*)custom_rng_negative_binomial_c<int>};

    fmap[DPNPFuncName::DPNP_FN_UNIFORM][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_rng_uniform_c<double>};
    fmap[DPNPFuncName::DPNP_FN_UNIFORM][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_rng_uniform_c<float>};
    fmap[DPNPFuncName::DPNP_FN_UNIFORM][eft_INT][eft_INT] = {eft_INT, (void*)custom_rng_uniform_c<int>};

    return;
}
