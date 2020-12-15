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

#include <vector>

namespace mkl_rng = oneapi::mkl::rng;
namespace mkl_blas = oneapi::mkl::blas;

template <typename _DataType>
void dpnp_rng_beta_c(void* result, _DataType a, _DataType b, size_t size)
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
void dpnp_rng_binomial_c(void* result, int ntrial, double p, size_t size)
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
void dpnp_rng_chi_square_c(void* result, int df, size_t size)
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
void dpnp_rng_exponential_c(void* result, _DataType beta, size_t size)
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
void dpnp_rng_gamma_c(void* result, _DataType shape, _DataType scale, size_t size)
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
void dpnp_rng_gaussian_c(void* result, _DataType mean, _DataType stddev, size_t size)
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
void dpnp_rng_geometric_c(void* result, float p, size_t size)
{
    if (!size)
    {
        return;
    }
    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    mkl_rng::geometric<_DataType> distribution(p);
    // perform generation
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_out.wait();
}

template <typename _DataType>
void dpnp_rng_gumbel_c(void* result, double loc, double scale, size_t size)
{
    cl::sycl::event event;
    if (!size)
    {
        return;
    }

    const _DataType alpha = (_DataType(-1.0));
    const _DataType stride = (_DataType(1.0));
    _DataType* result1 = reinterpret_cast<_DataType*>(result);
    loc = loc * (double(-1.0));

    mkl_rng::gumbel<_DataType> distribution(loc, scale);
    // perform generation
    event = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event.wait();
    event = mkl_blas::scal(DPNP_QUEUE, size, alpha, result1, stride);
    event.wait();
}

template <typename _DataType>
void dpnp_rng_hypergeometric_c(void* result, int l, int s, int m, size_t size)
{
    if (!size)
    {
        return;
    }
    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    mkl_rng::hypergeometric<_DataType> distribution(l, s, m);
    // perform generation
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_out.wait();
}

template <typename _DataType>
void dpnp_rng_laplace_c(void* result, double loc, double scale, size_t size)
{
    if (!size)
    {
        return;
    }
    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    mkl_rng::laplace<_DataType> distribution(loc, scale);
    // perform generation
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_out.wait();
}

template <typename _DataType>
void dpnp_rng_lognormal_c(void* result, _DataType mean, _DataType stddev, size_t size)
{
    if (!size)
    {
        return;
    }
    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    const _DataType displacement = _DataType(0.0);

    const _DataType scalefactor = _DataType(1.0);

    mkl_rng::lognormal<_DataType> distribution(mean, stddev, displacement, scalefactor);
    // perform generation
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_out.wait();
}

template <typename _DataType>
void dpnp_rng_multinomial_c(void* result, int ntrial, const double* p_vector, const size_t p_vector_size, size_t size)
{
    if (!size)
    {
        return;
    }
    std::int32_t* result1 = reinterpret_cast<std::int32_t*>(result);
    std::vector<double> p(p_vector, p_vector + p_vector_size);

    mkl_rng::multinomial<std::int32_t> distribution(ntrial, p);
    // size = size
    // `result` is a array for random numbers
    // `size` is a `result`'s len. `size = n * p.size()`
    // `n` is a number of random values to be generated.
    size_t n = size / p.size();
    // perform generation
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, n, result1);
    event_out.wait();
}

template <typename _DataType>
void dpnp_rng_multivariate_normal_c(void* result,
                                    const int dimen,
                                    const double* mean_vector,
                                    const size_t mean_vector_size,
                                    const double* cov_vector,
                                    const size_t cov_vector_size,
                                    size_t size)
{
    if (!size)
    {
        return;
    }
    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    std::vector<double> mean(mean_vector, mean_vector + mean_vector_size);
    std::vector<double> cov(cov_vector, cov_vector + cov_vector_size);

    // `result` is a array for random numbers
    // `size` is a `result`'s len.
    // `size1` is a number of random values to be generated for each dimension.
    mkl_rng::gaussian_mv<_DataType> distribution(dimen, mean, cov);
    size_t size1 = size / dimen;

    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size1, result1);
    event_out.wait();
}

template <typename _DataType>
void dpnp_rng_negative_binomial_c(void* result, double a, double p, size_t size)
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
void dpnp_rng_normal_c(void* result, _DataType mean, _DataType stddev, size_t size)
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
void dpnp_rng_poisson_c(void* result, double lambda, size_t size)
{
    if (!size)
    {
        return;
    }
    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    mkl_rng::poisson<_DataType> distribution(lambda);
    // perform generation
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_out.wait();
}

template <typename _DataType>
void dpnp_rng_rayleigh_c(void* result, _DataType scale, size_t size)
{
    if (!size)
    {
        return;
    }

    // set displacement a
    const _DataType a = (_DataType(0.0));

    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    mkl_rng::rayleigh<_DataType> distribution(a, scale);
    // perform generation
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_out.wait();
}

template <typename _DataType>
void dpnp_rng_standard_cauchy_c(void* result, size_t size)
{
    if (!size)
    {
        return;
    }
    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    const _DataType displacement = _DataType(0.0);

    const _DataType scalefactor = _DataType(1.0);

    mkl_rng::cauchy<_DataType> distribution(displacement, scalefactor);
    // perform generation
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_out.wait();
}

template <typename _DataType>
void dpnp_rng_standard_exponential_c(void* result, size_t size)
{
    if (!size)
    {
        return;
    }

    // set displacement a
    const _DataType beta = (_DataType(1.0));

    dpnp_rng_exponential_c(result, beta, size);
}

template <typename _DataType>
void dpnp_rng_standard_gamma_c(void* result, _DataType shape, size_t size)
{
    if (!size)
    {
        return;
    }

    const _DataType scale = _DataType(1.0);

    dpnp_rng_gamma_c(result, shape, scale, size);
}

template <typename _DataType>
void dpnp_rng_standard_normal_c(void* result, size_t size)
{
    if (!size)
    {
        return;
    }

    const _DataType mean = _DataType(0.0);
    const _DataType stddev = _DataType(1.0);

    dpnp_rng_normal_c(result, mean, stddev, size);
}

template <typename _DataType>
void dpnp_rng_uniform_c(void* result, long low, long high, size_t size)
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

template <typename _DataType>
void dpnp_rng_weibull_c(void* result, double alpha, size_t size)
{
    if (!size)
    {
        return;
    }
    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    // set displacement a
    const _DataType a = (_DataType(0.0));

    // set beta
    const _DataType beta = (_DataType(1.0));

    mkl_rng::weibull<_DataType> distribution(alpha, a, beta);
    // perform generation
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_out.wait();
}

void func_map_init_random(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_RNG_BETA][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_beta_c<double>};
    fmap[DPNPFuncName::DPNP_FN_RNG_BETA][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_rng_beta_c<float>};

    fmap[DPNPFuncName::DPNP_FN_RNG_BINOMIAL][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_rng_binomial_c<int>};

    fmap[DPNPFuncName::DPNP_FN_RNG_CHISQUARE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_chi_square_c<double>};
    fmap[DPNPFuncName::DPNP_FN_RNG_CHISQUARE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_rng_chi_square_c<float>};

    fmap[DPNPFuncName::DPNP_FN_RNG_EXPONENTIAL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_exponential_c<double>};
    fmap[DPNPFuncName::DPNP_FN_RNG_EXPONENTIAL][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_rng_exponential_c<float>};

    fmap[DPNPFuncName::DPNP_FN_RNG_GAMMA][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_gamma_c<double>};
    fmap[DPNPFuncName::DPNP_FN_RNG_GAMMA][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_rng_gamma_c<float>};

    fmap[DPNPFuncName::DPNP_FN_RNG_GAUSSIAN][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_gaussian_c<double>};
    fmap[DPNPFuncName::DPNP_FN_RNG_GAUSSIAN][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_rng_gaussian_c<float>};

    fmap[DPNPFuncName::DPNP_FN_RNG_GEOMETRIC][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_rng_geometric_c<int>};

    fmap[DPNPFuncName::DPNP_FN_RNG_GUMBEL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_gumbel_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_HYPERGEOMETRIC][eft_INT][eft_INT] = {eft_INT,
                                                                        (void*)dpnp_rng_hypergeometric_c<int>};

    fmap[DPNPFuncName::DPNP_FN_RNG_LAPLACE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_laplace_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_LOGNORMAL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_lognormal_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_MULTINOMIAL][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_rng_multinomial_c<int>};

    fmap[DPNPFuncName::DPNP_FN_RNG_MULTIVARIATE_NORMAL][eft_DBL][eft_DBL] = {
        eft_DBL, (void*)dpnp_rng_multivariate_normal_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_NEGATIVE_BINOMIAL][eft_INT][eft_INT] = {eft_INT,
                                                                           (void*)dpnp_rng_negative_binomial_c<int>};

    fmap[DPNPFuncName::DPNP_FN_RNG_NORMAL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_normal_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_POISSON][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_rng_poisson_c<int>};

    fmap[DPNPFuncName::DPNP_FN_RNG_RAYLEIGH][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_rayleigh_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_STANDARD_CAUCHY][eft_DBL][eft_DBL] = {eft_DBL,
                                                                         (void*)dpnp_rng_standard_cauchy_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_STANDARD_EXPONENTIAL][eft_DBL][eft_DBL] = {
        eft_DBL, (void*)dpnp_rng_standard_exponential_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_STANDARD_GAMMA][eft_DBL][eft_DBL] = {eft_DBL,
                                                                        (void*)dpnp_rng_standard_gamma_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_STANDARD_NORMAL][eft_DBL][eft_DBL] = {eft_DBL,
                                                                         (void*)dpnp_rng_standard_normal_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_UNIFORM][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_uniform_c<double>};
    fmap[DPNPFuncName::DPNP_FN_RNG_UNIFORM][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_rng_uniform_c<float>};
    fmap[DPNPFuncName::DPNP_FN_RNG_UNIFORM][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_rng_uniform_c<int>};

    fmap[DPNPFuncName::DPNP_FN_RNG_WEIBULL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_weibull_c<double>};

    return;
}
