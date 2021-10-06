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

#include <cassert>
#include <cmath>
#include <mkl_vsl.h>
#include <stdexcept>
#include <vector>

#include <dpnp_iface.hpp>

#include "dpnp_fptr.hpp"
#include "dpnp_utils.hpp"
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"

namespace mkl_blas = oneapi::mkl::blas;
namespace mkl_rng = oneapi::mkl::rng;
namespace mkl_vm = oneapi::mkl::vm;

/**
 * Use get/set functions to access/modify this variable
 */
static VSLStreamStatePtr rng_stream = nullptr;

static void set_rng_stream(size_t seed = 1)
{
    if (rng_stream)
    {
        vslDeleteStream(&rng_stream);
        rng_stream = nullptr;
    }

    vslNewStream(&rng_stream, VSL_BRNG_MT19937, seed);
}

static VSLStreamStatePtr get_rng_stream()
{
    if (!rng_stream)
    {
        set_rng_stream();
    }

    return rng_stream;
}

void dpnp_rng_srand_c(size_t seed)
{
    backend_sycl::backend_sycl_rng_engine_init(seed);
    set_rng_stream(seed);
}

template <typename _DataType>
void dpnp_rng_beta_c(void* result, const _DataType a, const _DataType b, const size_t size)
{
    if (!size)
    {
        return;
    }

    _DataType displacement = _DataType(0.0);

    _DataType scalefactor = _DataType(1.0);

    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    if (dpnp_queue_is_cpu_c())
    {
        mkl_rng::beta<_DataType> distribution(a, b, displacement, scalefactor);
        // perform generation
        auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
        event_out.wait();
    }
    else
    {
        int errcode =
            vdRngBeta(VSL_RNG_METHOD_BETA_CJA, get_rng_stream(), size, result1, a, b, displacement, scalefactor);
        if (errcode != VSL_STATUS_OK)
        {
            throw std::runtime_error("DPNP RNG Error: dpnp_rng_beta_c() failed.");
        }
    }

    return;
}

template <typename _DataType>
void dpnp_rng_binomial_c(void* result, const int ntrial, const double p, const size_t size)
{
    if (result == nullptr)
    {
        return;
    }
    if (!size)
    {
        return;
    }

    if (ntrial == 0 || p == 0)
    {
        dpnp_zeros_c<_DataType>(result, size);
    }
    else if (p == 1)
    {
        _DataType* fill_value = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(sizeof(_DataType)));
        fill_value[0] = static_cast<_DataType>(ntrial);
        dpnp_initval_c<_DataType>(result, fill_value, size);
        dpnp_memory_free_c(fill_value);
    }
    else
    {
        _DataType* result1 = reinterpret_cast<_DataType*>(result);
        if (dpnp_queue_is_cpu_c())
        {
            mkl_rng::binomial<_DataType> distribution(ntrial, p);
            auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
            event_out.wait();
        }
        else
        {
            int errcode = viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, get_rng_stream(), size, result1, ntrial, p);
            if (errcode != VSL_STATUS_OK)
            {
                throw std::runtime_error("DPNP RNG Error: dpnp_rng_binomial_c() failed.");
            }
        }
    }
    return;
}

template <typename _DataType>
void dpnp_rng_chisquare_c(void* result, const int df, const size_t size)
{
    if (!size)
    {
        return;
    }
    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    if (dpnp_queue_is_cpu_c())
    {
        mkl_rng::chi_square<_DataType> distribution(df);
        // perform generation
        auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
        event_out.wait();
    }
    else
    {
        int errcode = vdRngChiSquare(VSL_RNG_METHOD_CHISQUARE_CHI2GAMMA, get_rng_stream(), size, result1, df);
        if (errcode != VSL_STATUS_OK)
        {
            throw std::runtime_error("DPNP RNG Error: dpnp_rng_chisquare_c() failed.");
        }
    }
}

template <typename _DataType>
void dpnp_rng_exponential_c(void* result, const _DataType beta, const size_t size)
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
void dpnp_rng_f_c(void* result, const _DataType df_num, const _DataType df_den, const size_t size)
{
    if (!size)
    {
        return;
    }
    std::vector<cl::sycl::event> no_deps;

    const _DataType d_zero = (_DataType(0.0));

    _DataType shape = 0.5 * df_num;
    _DataType scale = 2.0 / df_num;
    _DataType* den = nullptr;

    DPNPC_ptr_adapter<_DataType> result1_ptr(result, size, true, true);
    _DataType* result1 = result1_ptr.get_ptr();

    if (dpnp_queue_is_cpu_c())
    {
        mkl_rng::gamma<_DataType> gamma_distribution1(shape, d_zero, scale);
        auto event_out = mkl_rng::generate(gamma_distribution1, DPNP_RNG_ENGINE, size, result1);
        event_out.wait();

        den = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(size * sizeof(_DataType)));
        shape = 0.5 * df_den;
        scale = 2.0 / df_den;
        mkl_rng::gamma<_DataType> gamma_distribution2(shape, d_zero, scale);
        event_out = mkl_rng::generate(gamma_distribution2, DPNP_RNG_ENGINE, size, den);
        event_out.wait();

        event_out = mkl_vm::div(DPNP_QUEUE, size, result1, den, result1, no_deps, mkl_vm::mode::ha);
        event_out.wait();

        dpnp_memory_free_c(den);
    }
    else
    {
        int errcode =
            vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, get_rng_stream(), size, result1, shape, d_zero, scale);
        if (errcode != VSL_STATUS_OK)
        {
            throw std::runtime_error("DPNP RNG Error: dpnp_rng_f_c() failed.");
        }
        den = (_DataType*)mkl_malloc(size * sizeof(_DataType), 64);
        shape = 0.5 * df_den;
        scale = 2.0 / df_den;
        errcode = vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, get_rng_stream(), size, den, shape, d_zero, scale);
        if (errcode != VSL_STATUS_OK)
        {
            throw std::runtime_error("DPNP RNG Error: dpnp_rng_f_c() failed.");
        }
        vmdDiv(size, result1, den, result1, VML_HA);
        mkl_free(den);
    }
}

template <typename _DataType>
void dpnp_rng_gamma_c(void* result, const _DataType shape, const _DataType scale, const size_t size)
{
    if (!size || result == nullptr)
    {
        return;
    }

    if (shape == 0.0 || scale == 0.0)
    {
        dpnp_zeros_c<_DataType>(result, size);
    }
    else
    {
        _DataType* result1 = reinterpret_cast<_DataType*>(result);
        const _DataType a = (_DataType(0.0));

        if (dpnp_queue_is_cpu_c())
        {
            mkl_rng::gamma<_DataType> distribution(shape, a, scale);
            // perform generation
            auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
            event_out.wait();
        }
        else
        {
            int errcode = vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM, get_rng_stream(), size, result1, shape, a, scale);
            if (errcode != VSL_STATUS_OK)
            {
                throw std::runtime_error("DPNP RNG Error: dpnp_rng_gamma_c() failed.");
            }
        }
    }
}

template <typename _DataType>
void dpnp_rng_gaussian_c(void* result, const _DataType mean, const _DataType stddev, const size_t size)
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
void dpnp_rng_geometric_c(void* result, const float p, const size_t size)
{
    if (!size || !result)
    {
        return;
    }

    if (p == 1.0)
    {
        dpnp_ones_c<_DataType>(result, size);
    }
    else
    {
        _DataType* result1 = reinterpret_cast<_DataType*>(result);
        mkl_rng::geometric<_DataType> distribution(p);
        // perform generation
        auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
        event_out.wait();
    }
}

template <typename _KernelNameSpecialization>
class dpnp_blas_scal_c_kernel;

template <typename _DataType>
void dpnp_rng_gumbel_c(void* result, const double loc, const double scale, const size_t size)
{
    if (!size || !result)
    {
        return;
    }

    if (scale == 0.0)
    {
        _DataType* fill_value = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(sizeof(_DataType)));
        fill_value[0] = static_cast<_DataType>(loc);
        dpnp_initval_c<_DataType>(result, fill_value, size);
        dpnp_memory_free_c(fill_value);
    }
    else
    {
        const _DataType alpha = (_DataType(-1.0));
        std::int64_t incx = 1;
        _DataType* result1 = reinterpret_cast<_DataType*>(result);
        double negloc = loc * (double(-1.0));

        mkl_rng::gumbel<_DataType> distribution(negloc, scale);
        auto event_distribution = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);

        // OK for CPU and segfault for GPU device
        // event = mkl_blas::scal(DPNP_QUEUE, size, alpha, result1, incx);
        cl::sycl::event prod_event;
        if (dpnp_queue_is_cpu_c())
        {
            prod_event = mkl_blas::scal(DPNP_QUEUE, size, alpha, result1, incx, {event_distribution});
        }
        else
        {
            // for (size_t i = 0; i < size; i++) result1[i] *= alpha;
            cl::sycl::range<1> gws(size);
            auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
                size_t i = global_id[0];
                result1[i] *= alpha;
            };
            auto kernel_func = [&](cl::sycl::handler& cgh) {
                cgh.depends_on({event_distribution});
                cgh.parallel_for<class dpnp_blas_scal_c_kernel<_DataType>>(gws, kernel_parallel_for_func);
            };
            prod_event = DPNP_QUEUE.submit(kernel_func);
        }
        prod_event.wait();
    }
}

template <typename _DataType>
void dpnp_rng_hypergeometric_c(void* result, const int l, const int s, const int m, const size_t size)
{
    if (!size || !result)
    {
        return;
    }

    if (m == 0)
    {
        dpnp_zeros_c<_DataType>(result, size);
    }
    else if (l == m)
    {
        _DataType* fill_value = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(sizeof(_DataType)));
        fill_value[0] = static_cast<_DataType>(s);
        dpnp_initval_c<_DataType>(result, fill_value, size);
        dpnp_memory_free_c(fill_value);
    }
    else
    {
        _DataType* result1 = reinterpret_cast<_DataType*>(result);
        if (dpnp_queue_is_cpu_c())
        {
            mkl_rng::hypergeometric<_DataType> distribution(l, s, m);
            // perform generation
            auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
            event_out.wait();
        }
        else
        {
            int errcode =
                viRngHypergeometric(VSL_RNG_METHOD_HYPERGEOMETRIC_H2PE, get_rng_stream(), size, result1, l, s, m);
            if (errcode != VSL_STATUS_OK)
            {
                throw std::runtime_error("DPNP RNG Error: dpnp_rng_hypergeometric_c() failed.");
            }
        }
    }
}

template <typename _DataType>
void dpnp_rng_laplace_c(void* result, const double loc, const double scale, const size_t size)
{
    if (!size || !result)
    {
        return;
    }

    if (scale == 0.0)
    {
        dpnp_zeros_c<_DataType>(result, size);
    }
    else
    {
        _DataType* result1 = reinterpret_cast<_DataType*>(result);
        mkl_rng::laplace<_DataType> distribution(loc, scale);
        // perform generation
        auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
        event_out.wait();
    }
}

template <typename _KernelNameSpecialization>
class dpnp_rng_logistic_c_kernel;

/*   Logistic(loc, scale) ~ loc + scale * log(u/(1.0 - u)) */
template <typename _DataType>
void dpnp_rng_logistic_c(void* result, const double loc, const double scale, const size_t size)
{
    if (!size || !result)
    {
        return;
    }

    const _DataType d_zero = _DataType(0.0);
    const _DataType d_one = _DataType(1.0);

    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    mkl_rng::uniform<_DataType> distribution(d_zero, d_one);
    auto event_distribution = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);

    cl::sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
        size_t i = global_id[0];
        result1[i] = cl::sycl::log(result1[i] / (1.0 - result1[i]));
        result1[i] = loc + scale * result1[i];
    };
    auto kernel_func = [&](cl::sycl::handler& cgh) {
        cgh.depends_on({event_distribution});
        cgh.parallel_for<class dpnp_rng_logistic_c_kernel<_DataType>>(gws, kernel_parallel_for_func);
    };
    auto event = DPNP_QUEUE.submit(kernel_func);
    event.wait();
}

template <typename _DataType>
void dpnp_rng_lognormal_c(void* result, const _DataType mean, const _DataType stddev, const size_t size)
{
    if (!size || !result)
    {
        return;
    }
    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    if (stddev == 0.0)
    {
        _DataType* fill_value = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(sizeof(_DataType)));
        fill_value[0] = static_cast<_DataType>(std::exp(mean + (stddev * stddev) / 2));
        dpnp_initval_c<_DataType>(result, fill_value, size);
        dpnp_memory_free_c(fill_value);
    }
    else
    {
        const _DataType displacement = _DataType(0.0);
        const _DataType scalefactor = _DataType(1.0);

        mkl_rng::lognormal<_DataType> distribution(mean, stddev, displacement, scalefactor);
        auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
        event_out.wait();
    }
    return;
}

template <typename _DataType>
void dpnp_rng_multinomial_c(
    void* result, const int ntrial, const double* p_vector, const size_t p_vector_size, const size_t size)
{
    if (!size || !result)
    {
        return;
    }

    if (ntrial == 0)
    {
        dpnp_zeros_c<_DataType>(result, size);
    }
    else
    {
        std::int32_t* result1 = reinterpret_cast<std::int32_t*>(result);
        std::vector<double> p(p_vector, p_vector + p_vector_size);
        // size = size
        // `result` is a array for random numbers
        // `size` is a `result`'s len. `size = n * p.size()`
        // `n` is a number of random values to be generated.
        size_t n = size / p.size();

        if (dpnp_queue_is_cpu_c())
        {
            mkl_rng::multinomial<std::int32_t> distribution(ntrial, p);
            // perform generation
            auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, n, result1);
            event_out.wait();
        }
        else
        {
            int errcode = viRngMultinomial(
                VSL_RNG_METHOD_MULTINOMIAL_MULTPOISSON, get_rng_stream(), n, result1, ntrial, p_vector_size, p_vector);
            if (errcode != VSL_STATUS_OK)
            {
                throw std::runtime_error("DPNP RNG Error: dpnp_rng_multinomial_c() failed.");
            }
        }
    }
    return;
}

template <typename _DataType>
void dpnp_rng_multivariate_normal_c(void* result,
                                    const int dimen,
                                    const double* mean_vector,
                                    const size_t mean_vector_size,
                                    const double* cov_vector,
                                    const size_t cov_vector_size,
                                    const size_t size)
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
    size_t size1 = size / dimen;

    mkl_rng::gaussian_mv<_DataType> distribution(dimen, mean, cov);
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size1, result1);
    event_out.wait();
}

template <typename _DataType>
void dpnp_rng_negative_binomial_c(void* result, const double a, const double p, const size_t size)
{
    if (!size)
    {
        return;
    }
    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    if (dpnp_queue_is_cpu_c())
    {
        mkl_rng::negative_binomial<_DataType> distribution(a, p);
        // perform generation
        auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
        event_out.wait();
    }
    else
    {
        int errcode = viRngNegbinomial(VSL_RNG_METHOD_NEGBINOMIAL_NBAR, get_rng_stream(), size, result1, a, p);
        if (errcode != VSL_STATUS_OK)
        {
            throw std::runtime_error("DPNP RNG Error: dpnp_rng_negative_binomial_c() failed.");
        }
    }
}

template <typename _DataType>
void dpnp_rng_noncentral_chisquare_c(void* result, const _DataType df, const _DataType nonc, const size_t size)
{
    if (!size || !result)
    {
        return;
    }
    DPNPC_ptr_adapter<_DataType> result1_ptr(result, size, true, true);
    _DataType* result1 = result1_ptr.get_ptr();

    const _DataType d_zero = _DataType(0.0);
    const _DataType d_one = _DataType(1.0);
    const _DataType d_two = _DataType(2.0);

    if (dpnp_queue_is_cpu_c())
    {
        _DataType shape, loc;
        size_t i;

        if (df > 1)
        {
            _DataType* nvec = nullptr;

            shape = 0.5 * (df - 1.0);
            /* res has chi^2 with (df - 1) */
            mkl_rng::gamma<_DataType> gamma_distribution(shape, d_zero, d_two);
            auto event_gamma_distr = mkl_rng::generate(gamma_distribution, DPNP_RNG_ENGINE, size, result1);

            nvec = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(size * sizeof(_DataType)));

            loc = sqrt(nonc);

            mkl_rng::gaussian<_DataType> gaussian_distribution(loc, d_one);
            auto event_gaussian_distr = mkl_rng::generate(gaussian_distribution, DPNP_RNG_ENGINE, size, nvec);

            /* squaring could result in an overflow */
            auto event_sqr_out =
                mkl_vm::sqr(DPNP_QUEUE, size, nvec, nvec, {event_gamma_distr, event_gaussian_distr}, mkl_vm::mode::ha);
            auto event_add_out =
                mkl_vm::add(DPNP_QUEUE, size, result1, nvec, result1, {event_sqr_out}, mkl_vm::mode::ha);
            dpnp_memory_free_c(nvec);
            event_add_out.wait();
        }
        else if (df < 1)
        {
            /* noncentral_chisquare(df, nonc) ~ G( df/2 + Poisson(nonc/2), 2) */
            double lambda;
            int* pvec = nullptr;
            pvec = reinterpret_cast<int*>(dpnp_memory_alloc_c(size * sizeof(int)));
            lambda = 0.5 * nonc;

            mkl_rng::poisson<int> poisson_distribution(lambda);
            auto event_out = mkl_rng::generate(poisson_distribution, DPNP_RNG_ENGINE, size, pvec);
            event_out.wait();

            shape = 0.5 * df;

            if (0.125 * size > sqrt(lambda))
            {
                size_t* idx = nullptr;
                _DataType* tmp = nullptr;
                idx = reinterpret_cast<size_t*>(dpnp_memory_alloc_c(size * sizeof(size_t)));
                for (i = 0; i < size; i++)
                    idx[i] = i;

                std::sort(idx, idx + size, [pvec](size_t i1, size_t i2) { return pvec[i1] < pvec[i2]; });
                /* idx now contains original indexes of ordered Poisson outputs */

                /* allocate workspace to store samples of gamma, enough to hold entire output */
                tmp = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(size * sizeof(_DataType)));
                for (i = 0; i < size;)
                {
                    size_t k, j;
                    int cv = pvec[idx[i]];

                    // TODO vectorize
                    for (j = i + 1; (j < size) && (pvec[idx[j]] == cv); j++)
                    {
                    }
                    // assert(j > i);
                    if (j <= i)
                    {
                        throw std::runtime_error("DPNP RNG Error: dpnp_rng_noncentral_chisquare_c() failed.");
                    }
                    mkl_rng::gamma<_DataType> gamma_distribution(shape + cv, d_zero, d_two);
                    event_out = mkl_rng::generate(gamma_distribution, DPNP_RNG_ENGINE, j - i, tmp);
                    event_out.wait();

                    // TODO vectorize
                    for (k = i; k < j; k++)
                        result1[idx[k]] = tmp[k - i];

                    i = j;
                }

                dpnp_memory_free_c(tmp);
                dpnp_memory_free_c(idx);
            }
            else
            {
                for (i = 0; i < size; i++)
                {
                    mkl_rng::gamma<_DataType> gamma_distribution(shape + pvec[i], d_zero, d_two);
                    event_out = mkl_rng::generate(gamma_distribution, DPNP_RNG_ENGINE, 1, result1 + 1);
                    event_out.wait();
                }
            }
            dpnp_memory_free_c(pvec);
        }
        else
        {
            /* noncentral_chisquare(1, nonc) ~ (Z + sqrt(nonc))**2 for df == 1 */
            loc = sqrt(nonc);
            mkl_rng::gaussian<_DataType> gaussian_distribution(loc, d_one);
            auto event_gaussian_distr = mkl_rng::generate(gaussian_distribution, DPNP_RNG_ENGINE, size, result1);
            auto event_out = mkl_vm::sqr(DPNP_QUEUE, size, result1, result1, {event_gaussian_distr}, mkl_vm::mode::ha);
            event_out.wait();
        }
    }
    else
    {
        double shape, loc;
        int errcode;
        size_t i;

        if (df > 1)
        {
            double* nvec = nullptr;

            shape = 0.5 * (df - 1.0);
            /* res has chi^2 with (df - 1) */
            errcode =
                vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, get_rng_stream(), size, result1, shape, d_zero, d_two);
            // TODO
            // refactor this check, smth like: void status_assert(errcode, VSL_STATUS_OK, func_name)
            // or just redesign and return int status:
            // void dpnp_rng_*_c(..., int& status);
            // or
            // int dpnp_rng_*_c(...);
            if (errcode != VSL_STATUS_OK)
            {
                throw std::runtime_error("DPNP RNG Error: dpnp_rng_noncentral_chisquare_c() failed.");
            }

            nvec = (double*)mkl_malloc(size * sizeof(double), 64);
            loc = sqrt(nonc);
            errcode = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, get_rng_stream(), size, nvec, loc, d_one);
            if (errcode != VSL_STATUS_OK)
            {
                throw std::runtime_error("DPNP RNG Error: dpnp_rng_noncentral_chisquare_c() failed.");
            }
            /* squaring could result in an overflow */
            vmdSqr(size, nvec, nvec, VML_HA);
            vmdAdd(size, result1, nvec, result1, VML_HA);

            mkl_free(nvec);
        }
        else if (df < 1)
        {
            /* noncentral_chisquare(df, nonc) ~ G( df/2 + Poisson(nonc/2), 2) */
            double lambda;
            int* pvec = nullptr;
            pvec = (int*)mkl_malloc(size * sizeof(int), 64);

            lambda = 0.5 * nonc;
            errcode = viRngPoisson(VSL_RNG_METHOD_POISSON_PTPE, get_rng_stream(), size, pvec, lambda);
            if (errcode != VSL_STATUS_OK)
            {
                throw std::runtime_error("DPNP RNG Error: dpnp_rng_noncentral_chisquare_c() failed.");
            }

            shape = 0.5 * df;

            if (0.125 * size > sqrt(lambda))
            {
                size_t* idx = nullptr;
                double* tmp = nullptr;

                idx = (size_t*)mkl_malloc(size * sizeof(size_t), 64);
                if (idx == nullptr)
                {
                    throw std::runtime_error("DPNP RNG Error: dpnp_rng_noncentral_chisquare_c() failed.");
                }

                for (i = 0; i < size; i++)
                    idx[i] = i;

                std::sort(idx, idx + size, [pvec](size_t i1, size_t i2) { return pvec[i1] < pvec[i2]; });
                /* idx now contains original indexes of ordered Poisson outputs */

                /* allocate workspace to store samples of gamma, enough to hold entire output */
                tmp = (double*)mkl_malloc(size * sizeof(double), 64);
                for (i = 0; i < size;)
                {
                    size_t k, j;
                    int cv = pvec[idx[i]];

                    for (j = i + 1; (j < size) && (pvec[idx[j]] == cv); j++)
                    {
                    }
                    // assert(j > i);
                    if (j <= i)
                    {
                        throw std::runtime_error("DPNP RNG Error: dpnp_rng_noncentral_chisquare_c() failed.");
                    }
                    errcode = vdRngGamma(
                        VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, get_rng_stream(), j - i, tmp, shape + cv, d_zero, d_two);
                    if (errcode != VSL_STATUS_OK)
                    {
                        throw std::runtime_error("DPNP RNG Error: dpnp_rng_noncentral_chisquare_c() failed.");
                    }

                    for (k = i; k < j; k++)
                        result1[idx[k]] = tmp[k - i];

                    i = j;
                }
                mkl_free(tmp);
                mkl_free(idx);
            }
            else
            {
                for (i = 0; i < size; i++)
                {
                    errcode = vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE,
                                         get_rng_stream(),
                                         1,
                                         result1 + i,
                                         shape + pvec[i],
                                         d_zero,
                                         d_two);
                    if (errcode != VSL_STATUS_OK)
                    {
                        throw std::runtime_error("DPNP RNG Error: dpnp_rng_noncentral_chisquare_c() failed.");
                    }
                }
            }
            mkl_free(pvec);
        }
        else
        {
            /* noncentral_chisquare(1, nonc) ~ (Z + sqrt(nonc))**2 for df == 1 */
            loc = sqrt(nonc);
            errcode = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, get_rng_stream(), size, result1, loc, d_one);
            if (errcode != VSL_STATUS_OK)
            {
                throw std::runtime_error("DPNP RNG Error: dpnp_rng_noncentral_chisquare_c() failed.");
            }
            /* squaring could result in an overflow */
            vmdSqr(size, result1, result1, VML_HA);
        }
    }
}

template <typename _DataType>
void dpnp_rng_normal_c(void* result, const _DataType mean, const _DataType stddev, const size_t size)
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
void dpnp_rng_pareto_c(void* result, const double alpha, const size_t size)
{
    if (!size)
    {
        return;
    }
    std::vector<cl::sycl::event> no_deps;

    const _DataType d_zero = _DataType(0.0);
    const _DataType d_one = _DataType(1.0);
    _DataType neg_rec_alp = -1.0 / alpha;

    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    mkl_rng::uniform<_DataType> distribution(d_zero, d_one);
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_out.wait();

    event_out = mkl_vm::powx(DPNP_QUEUE, size, result1, neg_rec_alp, result1, no_deps, mkl_vm::mode::ha);
    event_out.wait();
}

template <typename _DataType>
void dpnp_rng_poisson_c(void* result, const double lambda, const size_t size)
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
void dpnp_rng_power_c(void* result, const double alpha, const size_t size)
{
    if (!size)
    {
        return;
    }
    std::vector<cl::sycl::event> no_deps;

    const _DataType d_zero = _DataType(0.0);
    const _DataType d_one = _DataType(1.0);
    _DataType neg_rec_alp = 1.0 / alpha;

    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    mkl_rng::uniform<_DataType> distribution(d_zero, d_one);
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_out.wait();

    event_out = mkl_vm::powx(DPNP_QUEUE, size, result1, neg_rec_alp, result1, no_deps, mkl_vm::mode::ha);
    event_out.wait();
}

template <typename _DataType>
void dpnp_rng_rayleigh_c(void* result, const _DataType scale, const size_t size)
{
    if (!size)
    {
        return;
    }

    std::vector<cl::sycl::event> no_deps;

    const _DataType a = 0.0;
    const _DataType beta = 2.0;

    DPNPC_ptr_adapter<_DataType> result1_ptr(result, size, true, true);
    _DataType* result1 = result1_ptr.get_ptr();

    mkl_rng::exponential<_DataType> distribution(a, beta);

    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_out.wait();
    event_out = mkl_vm::sqrt(DPNP_QUEUE, size, result1, result1, no_deps, mkl_vm::mode::ha);
    event_out.wait();
    // with MKL
    // event_out = mkl_blas::axpy(DPNP_QUEUE, size, scale, result1, 1, result1, 1);
    // event_out.wait();
    for (size_t i = 0; i < size; i++)
    {
        result1[i] *= scale;
    }
}

template <typename _DataType>
void dpnp_rng_shuffle_c(
    void* result, const size_t itemsize, const size_t ndim, const size_t high_dim_size, const size_t size)
{
    if (!result)
    {
        return;
    }

    if (!size || !ndim || !(high_dim_size > 1))
    {
        return;
    }

    DPNPC_ptr_adapter<char> result1_ptr(result, size, true, true);
    char* result1 = result1_ptr.get_ptr();

    size_t uvec_size = high_dim_size - 1;
    double* Uvec = reinterpret_cast<double*>(dpnp_memory_alloc_c(uvec_size * sizeof(double)));
    mkl_rng::uniform<double> uniform_distribution(0.0, 1.0);
    auto uniform_event = mkl_rng::generate(uniform_distribution, DPNP_RNG_ENGINE, uvec_size, Uvec);
    uniform_event.wait();

    if (ndim == 1)
    {
        // Fast, statically typed path: shuffle the underlying buffer.
        // Only for non-empty, 1d objects of class ndarray (subclasses such
        // as MaskedArrays may not support this approach).
        char* buf = reinterpret_cast<char*>(dpnp_memory_alloc_c(itemsize * sizeof(char)));
        for (size_t i = uvec_size; i > 0; i--)
        {
            size_t j = (size_t)(floor((i + 1) * Uvec[i - 1]));
            if (i != j)
            {
                auto memcpy1 =
                    DPNP_QUEUE.submit([&](cl::sycl::handler& h) { h.memcpy(buf, result1 + j * itemsize, itemsize); });
                auto memcpy2 = DPNP_QUEUE.submit([&](cl::sycl::handler& h) {
                    h.depends_on({memcpy1});
                    h.memcpy(result1 + j * itemsize, result1 + i * itemsize, itemsize);
                });
                auto memcpy3 = DPNP_QUEUE.submit([&](cl::sycl::handler& h) {
                    h.depends_on({memcpy2});
                    h.memcpy(result1 + i * itemsize, buf, itemsize);
                });
                memcpy3.wait();
            }
        }
        dpnp_memory_free_c(buf);
    }
    else
    {
        // Multidimensional ndarrays require a bounce buffer.
        size_t step_size = (size / high_dim_size) * itemsize; // size in bytes for x[i] element
        char* buf = reinterpret_cast<char*>(dpnp_memory_alloc_c(step_size * sizeof(char)));
        for (size_t i = uvec_size; i > 0; i--)
        {
            size_t j = (size_t)(floor((i + 1) * Uvec[i - 1]));
            if (j < i)
            {
                auto memcpy1 =
                    DPNP_QUEUE.submit([&](cl::sycl::handler& h) { h.memcpy(buf, result1 + j * step_size, step_size); });
                auto memcpy2 = DPNP_QUEUE.submit([&](cl::sycl::handler& h) {
                    h.depends_on({memcpy1});
                    h.memcpy(result1 + j * step_size, result1 + i * step_size, step_size);
                });
                auto memcpy3 = DPNP_QUEUE.submit([&](cl::sycl::handler& h) {
                    h.depends_on({memcpy2});
                    h.memcpy(result1 + i * step_size, buf, step_size);
                });
                memcpy3.wait();
            }
        }
        dpnp_memory_free_c(buf);
    }
    dpnp_memory_free_c(Uvec);
}

template <typename _DataType>
void dpnp_rng_standard_cauchy_c(void* result, const size_t size)
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
void dpnp_rng_standard_exponential_c(void* result, const size_t size)
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
void dpnp_rng_standard_gamma_c(void* result, const _DataType shape, const size_t size)
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
void dpnp_rng_standard_t_c(void* result, const _DataType df, const size_t size)
{
    if (!size || !result)
    {
        return;
    }

    _DataType* result1 = reinterpret_cast<_DataType*>(result);
    const _DataType d_zero = 0.0, d_one = 1.0;
    _DataType shape = df / 2;
    _DataType* sn = nullptr;

    if (dpnp_queue_is_cpu_c())
    {
        mkl_rng::gamma<_DataType> gamma_distribution(shape, d_zero, 1.0 / shape);
        auto gamma_distr_event = mkl_rng::generate(gamma_distribution, DPNP_RNG_ENGINE, size, result1);

        auto invsqrt_event = mkl_vm::invsqrt(DPNP_QUEUE, size, result1, result1, {gamma_distr_event}, mkl_vm::mode::ha);

        sn = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(size * sizeof(_DataType)));

        mkl_rng::gaussian<_DataType> gaussian_distribution(d_zero, d_one);
        auto gaussian_distr_event = mkl_rng::generate(gaussian_distribution, DPNP_RNG_ENGINE, size, sn);

        auto event_out = mkl_vm::mul(
            DPNP_QUEUE, size, result1, sn, result1, {invsqrt_event, gaussian_distr_event}, mkl_vm::mode::ha);
        dpnp_memory_free_c(sn);
        event_out.wait();
    }
    else
    {
        int errcode = vdRngGamma(
            VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, get_rng_stream(), size, result1, shape, d_zero, 1.0 / shape);

        if (errcode != VSL_STATUS_OK)
        {
            throw std::runtime_error("DPNP RNG Error: dpnp_rng_standard_t_c() failed.");
        }

        vmdInvSqrt(size, result1, result1, VML_HA);

        sn = (_DataType*)mkl_malloc(size * sizeof(_DataType), 64);

        errcode = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, get_rng_stream(), size, sn, d_zero, d_one);
        if (errcode != VSL_STATUS_OK)
        {
            throw std::runtime_error("DPNP RNG Error: dpnp_rng_standard_t_c() failed.");
        }
        vmdMul(size, result1, sn, result1, VML_HA);
        mkl_free(sn);
    }
}

template <typename _KernelNameSpecialization>
class dpnp_rng_triangular_ration_acceptance_c_kernel;

template <typename _DataType>
void dpnp_rng_triangular_c(
    void* result, const _DataType x_min, const _DataType x_mode, const _DataType x_max, const size_t size)
{
    if (!size)
    {
        return;
    }
    _DataType* result1 = reinterpret_cast<_DataType*>(result);
    const _DataType d_zero = (_DataType(0));
    const _DataType d_one = (_DataType(1));

    _DataType ratio, lpr, rpr;

    mkl_rng::uniform<_DataType> uniform_distribution(d_zero, d_one);
    auto event_uniform = mkl_rng::generate(uniform_distribution, DPNP_RNG_ENGINE, size, result1);

    {
        _DataType wtot, wl, wr;

        wtot = x_max - x_min;
        wl = x_mode - x_min;
        wr = x_max - x_mode;

        ratio = wl / wtot;
        lpr = wl * wtot;
        rpr = wr * wtot;
    }

    if (!(0 <= ratio && ratio <= 1))
    {
        throw std::runtime_error("DPNP RNG Error: dpnp_rng_triangular_c() failed.");
    }

    cl::sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
        size_t i = global_id[0];
        if (ratio <= 0)
        {
            result1[i] = x_max - cl::sycl::sqrt(result1[i] * rpr);
        }
        else if (ratio >= 1)
        {
            result1[i] = x_min + cl::sycl::sqrt(result1[i] * lpr);
        }
        else
        {
            _DataType ui = result1[i];
            if (ui > ratio)
            {
                result1[i] = x_max - cl::sycl::sqrt((1.0 - ui) * rpr);
            }
            else
            {
                result1[i] = x_min + cl::sycl::sqrt(ui * lpr);
            }
        }
    };
    auto kernel_func = [&](cl::sycl::handler& cgh) {
        cgh.depends_on({event_uniform});
        cgh.parallel_for<class dpnp_rng_triangular_ration_acceptance_c_kernel<_DataType>>(gws,
                                                                                          kernel_parallel_for_func);
    };
    auto event_ration_acceptance = DPNP_QUEUE.submit(kernel_func);
    event_ration_acceptance.wait();
}

template <typename _DataType>
void dpnp_rng_uniform_c(void* result, const long low, const long high, const size_t size)
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

#ifndef M_PI
/*  128-bits worth of pi */
#define M_PI 3.141592653589793238462643383279502884197
#endif

template <typename _DataType>
void dpnp_rng_vonmises_large_kappa_c(void* result, const _DataType mu, const _DataType kappa, const size_t size)
{
    if (!size || !result)
    {
        return;
    }

    DPNPC_ptr_adapter<_DataType> result1_ptr(result, size, true, true);
    _DataType* result1 = result1_ptr.get_ptr();

    _DataType r_over_two_kappa, recip_two_kappa;
    _DataType s_minus_one, hpt, r_over_two_kappa_minus_one, rho_minus_one;
    _DataType* Uvec = nullptr;
    _DataType* Vvec = nullptr;
    const _DataType d_zero = 0.0, d_one = 1.0;

    assert(kappa > 1.0);

    recip_two_kappa = 1 / (2 * kappa);

    /* variables here are dwindling to zero as kappa grows */
    hpt = sqrt(1 + recip_two_kappa * recip_two_kappa);
    r_over_two_kappa_minus_one = recip_two_kappa * (1 + recip_two_kappa / (1 + hpt));
    r_over_two_kappa = 1 + r_over_two_kappa_minus_one;
    rho_minus_one = r_over_two_kappa_minus_one - sqrt(2 * r_over_two_kappa * recip_two_kappa);
    s_minus_one = rho_minus_one * (0.5 * rho_minus_one / (1 + rho_minus_one));

    Uvec = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(size * sizeof(_DataType)));
    Vvec = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(size * sizeof(_DataType)));

    for (size_t n = 0; n < size;)
    {
        size_t diff_size = size - n;
        mkl_rng::uniform<_DataType> uniform_distribution_u(d_zero, 0.5 * M_PI);
        auto event_out = mkl_rng::generate(uniform_distribution_u, DPNP_RNG_ENGINE, diff_size, Uvec);
        event_out.wait();
        // TODO
        // use deps case
        mkl_rng::uniform<_DataType> uniform_distribution_v(d_zero, d_one);
        event_out = mkl_rng::generate(uniform_distribution_v, DPNP_RNG_ENGINE, diff_size, Vvec);
        event_out.wait();

        // TODO
        // kernel
        for (size_t i = 0; i < diff_size; i++)
        {
            _DataType sn, cn, sn2, cn2;
            _DataType neg_W_minus_one, V, Y;

            sn = sin(Uvec[i]);
            cn = cos(Uvec[i]);
            V = Vvec[i];
            sn2 = sn * sn;
            cn2 = cn * cn;

            neg_W_minus_one = s_minus_one * sn2 / (0.5 * s_minus_one + cn2);
            Y = kappa * (s_minus_one + neg_W_minus_one);

            if ((Y * (2 - Y) >= V) || (log(Y / V) + 1 >= Y))
            {
                Y = neg_W_minus_one * (2 - neg_W_minus_one);
                if (Y < 0)
                    Y = 0.0;
                else if (Y > 1.0)
                    Y = 1.0;

                result1[n++] = asin(sqrt(Y));
            }
        }
    }

    dpnp_memory_free_c(Uvec);

    mkl_rng::uniform<_DataType> uniform_distribution(d_zero, d_one);
    auto uniform_distr_event = mkl_rng::generate(uniform_distribution, DPNP_RNG_ENGINE, size, Vvec);

    cl::sycl::range<1> gws(size);

    auto paral_kernel_acceptance = [&](cl::sycl::handler& cgh) {
        cgh.depends_on({uniform_distr_event});
        cgh.parallel_for(gws, [=](cl::sycl::id<1> global_id) {
            size_t i = global_id[0];
            double mod, resi;
            resi = (Vvec[i] < 0.5) ? mu - result1[i] : mu + result1[i];
            mod = cl::sycl::fabs(resi);
            mod = (cl::sycl::fmod(mod + M_PI, 2 * M_PI) - M_PI);
            result1[i] = (resi < 0) ? -mod : mod;
        });
    };
    auto acceptance_event = DPNP_QUEUE.submit(paral_kernel_acceptance);
    acceptance_event.wait();

    dpnp_memory_free_c(Vvec);
    return;
}

template <typename _DataType>
void dpnp_rng_vonmises_small_kappa_c(void* result, const _DataType mu, const _DataType kappa, const size_t size)
{
    if (!size || !result)
    {
        return;
    }

    DPNPC_ptr_adapter<_DataType> result1_ptr(result, size, true, true);
    _DataType* result1 = result1_ptr.get_ptr();

    _DataType rho_over_kappa, rho, r, s_kappa;
    _DataType* Uvec = nullptr;
    _DataType* Vvec = nullptr;

    const _DataType d_zero = 0.0, d_one = 1.0;

    assert(0. < kappa <= 1.0);

    r = 1 + sqrt(1 + 4 * kappa * kappa);
    rho_over_kappa = (2) / (r + sqrt(2 * r));
    rho = rho_over_kappa * kappa;

    /* s times kappa */
    s_kappa = (1 + rho * rho) / (2 * rho_over_kappa);

    Uvec = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(size * sizeof(_DataType)));
    Vvec = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(size * sizeof(_DataType)));

    for (size_t n = 0; n < size;)
    {
        size_t diff_size = size - n;
        mkl_rng::uniform<_DataType> uniform_distribution_u(d_zero, M_PI);
        auto event_out = mkl_rng::generate(uniform_distribution_u, DPNP_RNG_ENGINE, diff_size, Uvec);
        event_out.wait();
        // TODO
        // use deps case
        mkl_rng::uniform<_DataType> uniform_distribution_v(d_zero, d_one);
        event_out = mkl_rng::generate(uniform_distribution_v, DPNP_RNG_ENGINE, diff_size, Vvec);
        event_out.wait();

        // TODO
        // kernel
        for (size_t i = 0; i < diff_size; i++)
        {
            _DataType Z, W, Y, V;
            Z = cos(Uvec[i]);
            V = Vvec[i];
            W = (kappa + s_kappa * Z) / (s_kappa + kappa * Z);
            Y = s_kappa - kappa * W;
            if ((Y * (2 - Y) >= V) || (log(Y / V) + 1 >= Y))
            {
                result1[n++] = acos(W);
            }
        }
    }

    dpnp_memory_free_c(Uvec);

    mkl_rng::uniform<_DataType> uniform_distribution(d_zero, d_one);
    auto uniform_distr_event = mkl_rng::generate(uniform_distribution, DPNP_RNG_ENGINE, size, Vvec);

    cl::sycl::range<1> gws(size);
    auto paral_kernel_acceptance = [&](cl::sycl::handler& cgh) {
        cgh.depends_on({uniform_distr_event});
        cgh.parallel_for(gws, [=](cl::sycl::id<1> global_id) {
            size_t i = global_id[0];
            double mod, resi;
            resi = (Vvec[i] < 0.5) ? mu - result1[i] : mu + result1[i];
            mod = cl::sycl::fabs(resi);
            mod = (cl::sycl::fmod(mod + M_PI, 2 * M_PI) - M_PI);
            result1[i] = (resi < 0) ? -mod : mod;
        });
    };
    auto acceptance_event = DPNP_QUEUE.submit(paral_kernel_acceptance);
    acceptance_event.wait();

    dpnp_memory_free_c(Vvec);
    return;
}

/* Vonmisses uses the rejection algorithm compared against the wrapped
   Cauchy distribution suggested by Best and Fisher and documented in
   Chapter 9 of Luc's Non-Uniform Random Variate Generation.
   http://cg.scs.carleton.ca/~luc/rnbookindex.html
   (but corrected to match the algorithm in R and Python)
*/
template <typename _DataType>
void dpnp_rng_vonmises_c(void* result, const _DataType mu, const _DataType kappa, const size_t size)
{
    if (kappa > 1.0)
        dpnp_rng_vonmises_large_kappa_c<_DataType>(result, mu, kappa, size);
    else
        dpnp_rng_vonmises_small_kappa_c<_DataType>(result, mu, kappa, size);
}

template <typename _KernelNameSpecialization>
class dpnp_rng_wald_acceptance_kernel1;

template <typename _KernelNameSpecialization>
class dpnp_rng_wald_acceptance_kernel2;

template <typename _DataType>
void dpnp_rng_wald_c(void* result, const _DataType mean, const _DataType scale, const size_t size)
{
    if (!size)
    {
        return;
    }

    _DataType* result1 = reinterpret_cast<_DataType*>(result);
    _DataType* uvec = nullptr;

    const _DataType d_zero = _DataType(0.0);
    const _DataType d_one = _DataType(1.0);
    _DataType gsc = sqrt(0.5 * mean / scale);

    mkl_rng::gaussian<_DataType> gaussian_distribution(d_zero, gsc);

    auto gaussian_dstr_event = mkl_rng::generate(gaussian_distribution, DPNP_RNG_ENGINE, size, result1);

    /* Y = mean/(2 scale) * Z^2 */
    auto sqr_event = mkl_vm::sqr(DPNP_QUEUE, size, result1, result1, {gaussian_dstr_event}, mkl_vm::mode::ha);

    cl::sycl::range<1> gws(size);
    auto acceptance_kernel1 = [=](cl::sycl::id<1> global_id) {
        size_t i = global_id[0];
        if (result1[i] <= 2.0)
        {
            result1[i] = 1.0 + result1[i] + cl::sycl::sqrt(result1[i] * (result1[i] + 2.0));
        }
        else
        {
            result1[i] = 1.0 + result1[i] * (1.0 + cl::sycl::sqrt(1.0 + 2.0 / result1[i]));
        }
    };
    auto parallel_for_acceptance1 = [&](cl::sycl::handler& cgh) {
        cgh.depends_on({sqr_event});
        cgh.parallel_for<class dpnp_rng_wald_acceptance_kernel1<_DataType>>(gws, acceptance_kernel1);
    };
    auto event_ration_acceptance = DPNP_QUEUE.submit(parallel_for_acceptance1);

    uvec = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(size * sizeof(_DataType)));

    mkl_rng::uniform<_DataType> uniform_distribution(d_zero, d_one);
    auto uniform_distr_event = mkl_rng::generate(uniform_distribution, DPNP_RNG_ENGINE, size, uvec);

    auto acceptance_kernel2 = [=](cl::sycl::id<1> global_id) {
        size_t i = global_id[0];
        if (uvec[i] * (1.0 + result1[i]) <= result1[i])
            result1[i] = mean / result1[i];
        else
            result1[i] = mean * result1[i];
    };
    auto parallel_for_acceptance2 = [&](cl::sycl::handler& cgh) {
        cgh.depends_on({event_ration_acceptance, uniform_distr_event});
        cgh.parallel_for<class dpnp_rng_wald_acceptance_kernel2<_DataType>>(gws, acceptance_kernel2);
    };
    auto event_out = DPNP_QUEUE.submit(parallel_for_acceptance2);
    event_out.wait();

    dpnp_memory_free_c(uvec);
}

template <typename _DataType>
void dpnp_rng_weibull_c(void* result, const double alpha, const size_t size)
{
    if (!size)
    {
        return;
    }

    if (alpha == 0)
    {
        dpnp_zeros_c<_DataType>(result, size);
    }
    else
    {
        _DataType* result1 = reinterpret_cast<_DataType*>(result);
        const _DataType a = (_DataType(0.0));
        const _DataType beta = (_DataType(1.0));

        mkl_rng::weibull<_DataType> distribution(alpha, a, beta);
        auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
        event_out.wait();
    }
}

template <typename _DataType>
void dpnp_rng_zipf_c(void* result, const _DataType a, const size_t size)
{
    if (!size)
    {
        return;
    }

    cl::sycl::event event_out;

    size_t i, n_accepted, batch_size;
    _DataType T, U, V, am1, b;
    _DataType *Uvec = nullptr, *Vvec = nullptr;
    long X;
    const _DataType d_zero = 0.0;
    const _DataType d_one = 1.0;

    DPNPC_ptr_adapter<_DataType> result1_ptr(result, size, true, true);
    _DataType* result1 = result1_ptr.get_ptr();

    am1 = a - d_one;
    b = pow(2.0, am1);

    Uvec = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(size * 2 * sizeof(_DataType)));
    Vvec = Uvec + size;

    // TODO
    // kernel for acceptance
    for (n_accepted = 0; n_accepted < size;)
    {
        batch_size = size - n_accepted;

        mkl_rng::uniform<_DataType> uniform_distribution(d_zero, d_one);
        event_out = mkl_rng::generate(uniform_distribution, DPNP_RNG_ENGINE, batch_size, Uvec);
        event_out.wait();
        event_out = mkl_rng::generate(uniform_distribution, DPNP_RNG_ENGINE, batch_size, Vvec);
        event_out.wait();
        for (i = 0; i < batch_size; i++)
        {
            U = d_one - Uvec[i];
            V = Vvec[i];
            X = (long)floor(pow(U, (-1.0) / am1));
            /* The real result may be above what can be represented in a signed
             * long. It will get casted to -sys.maxint-1. Since this is
             * a straightforward rejection algorithm, we can just reject this value
             * in the rejection condition below. This function then models a Zipf
             * distribution truncated to sys.maxint.
             */
            T = pow(d_one + d_one / X, am1);
            if ((X > 0) && ((V * X) * (T - d_one) / (b - d_one) <= T / b))
            {
                result1[n_accepted++] = X;
            }
        }
    }

    dpnp_memory_free_c(Uvec);
}

void func_map_init_random(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_RNG_BETA][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_beta_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_BINOMIAL][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_rng_binomial_c<int>};

    fmap[DPNPFuncName::DPNP_FN_RNG_CHISQUARE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_chisquare_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_EXPONENTIAL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_exponential_c<double>};
    fmap[DPNPFuncName::DPNP_FN_RNG_EXPONENTIAL][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_rng_exponential_c<float>};

    fmap[DPNPFuncName::DPNP_FN_RNG_F][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_f_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_GAMMA][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_gamma_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_GAUSSIAN][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_gaussian_c<double>};
    fmap[DPNPFuncName::DPNP_FN_RNG_GAUSSIAN][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_rng_gaussian_c<float>};

    fmap[DPNPFuncName::DPNP_FN_RNG_GEOMETRIC][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_rng_geometric_c<int>};

    fmap[DPNPFuncName::DPNP_FN_RNG_GUMBEL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_gumbel_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_HYPERGEOMETRIC][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_rng_hypergeometric_c<int>};

    fmap[DPNPFuncName::DPNP_FN_RNG_LAPLACE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_laplace_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_LOGISTIC][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_logistic_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_LOGNORMAL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_lognormal_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_MULTINOMIAL][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_rng_multinomial_c<int>};

    fmap[DPNPFuncName::DPNP_FN_RNG_MULTIVARIATE_NORMAL][eft_DBL][eft_DBL] = {
        eft_DBL, (void*)dpnp_rng_multivariate_normal_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_NEGATIVE_BINOMIAL][eft_INT][eft_INT] = {eft_INT,
                                                                           (void*)dpnp_rng_negative_binomial_c<int>};

    fmap[DPNPFuncName::DPNP_FN_RNG_NONCENTRAL_CHISQUARE][eft_DBL][eft_DBL] = {
        eft_DBL, (void*)dpnp_rng_noncentral_chisquare_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_NORMAL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_normal_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_PARETO][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_pareto_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_POISSON][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_rng_poisson_c<int>};

    fmap[DPNPFuncName::DPNP_FN_RNG_POWER][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_power_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_RAYLEIGH][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_rayleigh_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_SHUFFLE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_shuffle_c<double>};
    fmap[DPNPFuncName::DPNP_FN_RNG_SHUFFLE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_rng_shuffle_c<float>};
    fmap[DPNPFuncName::DPNP_FN_RNG_SHUFFLE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_rng_shuffle_c<int>};
    fmap[DPNPFuncName::DPNP_FN_RNG_SHUFFLE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_rng_shuffle_c<long>};

    fmap[DPNPFuncName::DPNP_FN_RNG_SRAND][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_srand_c};

    fmap[DPNPFuncName::DPNP_FN_RNG_STANDARD_CAUCHY][eft_DBL][eft_DBL] = {eft_DBL,
                                                                         (void*)dpnp_rng_standard_cauchy_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_STANDARD_EXPONENTIAL][eft_DBL][eft_DBL] = {
        eft_DBL, (void*)dpnp_rng_standard_exponential_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_STANDARD_GAMMA][eft_DBL][eft_DBL] = {eft_DBL,
                                                                        (void*)dpnp_rng_standard_gamma_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_STANDARD_NORMAL][eft_DBL][eft_DBL] = {eft_DBL,
                                                                         (void*)dpnp_rng_standard_normal_c<double>};
    fmap[DPNPFuncName::DPNP_FN_RNG_STANDARD_T][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_standard_t_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_TRIANGULAR][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_triangular_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_UNIFORM][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_uniform_c<double>};
    fmap[DPNPFuncName::DPNP_FN_RNG_UNIFORM][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_rng_uniform_c<float>};
    fmap[DPNPFuncName::DPNP_FN_RNG_UNIFORM][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_rng_uniform_c<int>};

    fmap[DPNPFuncName::DPNP_FN_RNG_VONMISES][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_vonmises_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_WALD][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_wald_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_WEIBULL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_weibull_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_ZIPF][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_zipf_c<double>};

    return;
}
