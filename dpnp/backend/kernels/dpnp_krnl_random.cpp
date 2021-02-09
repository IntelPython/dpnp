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

#include <cmath>
#include <mkl_vsl.h>
#include <stdexcept>
#include <vector>

#include <dpnp_iface.hpp>

#include "dpnp_fptr.hpp"
#include "dpnp_utils.hpp"
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
INP_DLLEXPORT void dpnp_rng_beta_c(void* result, const _DataType a, const _DataType b, const size_t size)
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
    if (!size)
    {
        return;
    }
    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    if (dpnp_queue_is_cpu_c())
    {
        mkl_rng::binomial<_DataType> distribution(ntrial, p);
        // perform generation
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
    cl::sycl::vector_class<cl::sycl::event> no_deps;

    const _DataType d_zero = (_DataType(0.0));

    _DataType shape = 0.5 * df_num;
    _DataType scale = 2.0 / df_num;
    _DataType* den = nullptr;

    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    if (dpnp_queue_is_cpu_c())
    {
        mkl_rng::gamma<_DataType> gamma_distribution1(shape, d_zero, scale);
        auto event_out = mkl_rng::generate(gamma_distribution1, DPNP_RNG_ENGINE, size, result1);
        event_out.wait();

        den = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(size * sizeof(_DataType)));
        if (den == nullptr)
        {
            throw std::runtime_error("DPNP RNG Error: dpnp_rng_f_c() failed.");
        }
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
        if (den == nullptr)
        {
            throw std::runtime_error("DPNP RNG Error: dpnp_rng_f_c() failed.");
        }
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
    if (!size)
    {
        return;
    }

    // set displacement a
    const _DataType a = (_DataType(0.0));

    _DataType* result1 = reinterpret_cast<_DataType*>(result);

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

template <typename _KernelNameSpecialization>
class dpnp_blas_scal_c_kernel;

template <typename _DataType>
void dpnp_rng_gumbel_c(void* result, const double loc, const double scale, const size_t size)
{
    cl::sycl::event event;
    if (!size)
    {
        return;
    }

    const _DataType alpha = (_DataType(-1.0));
    std::int64_t incx = 1;
    _DataType* result1 = reinterpret_cast<_DataType*>(result);
    double negloc = loc * (double(-1.0));

    mkl_rng::gumbel<_DataType> distribution(negloc, scale);
    event = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event.wait();

    // OK for CPU and segfault for GPU device
    // event = mkl_blas::scal(DPNP_QUEUE, size, alpha, result1, incx);
    if (dpnp_queue_is_cpu_c())
    {
        event = mkl_blas::scal(DPNP_QUEUE, size, alpha, result1, incx);
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
            cgh.parallel_for<class dpnp_blas_scal_c_kernel<_DataType>>(gws, kernel_parallel_for_func);
        };
        event = DPNP_QUEUE.submit(kernel_func);
    }
    event.wait();
}

template <typename _DataType>
void dpnp_rng_hypergeometric_c(void* result, const int l, const int s, const int m, const size_t size)
{
    if (!size)
    {
        return;
    }
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
        int errcode = viRngHypergeometric(VSL_RNG_METHOD_HYPERGEOMETRIC_H2PE, get_rng_stream(), size, result1, l, s, m);
        if (errcode != VSL_STATUS_OK)
        {
            throw std::runtime_error("DPNP RNG Error: dpnp_rng_hypergeometric_c() failed.");
        }
    }
}

template <typename _DataType>
void dpnp_rng_laplace_c(void* result, const double loc, const double scale, const size_t size)
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

/*   Logistic(loc, scale) ~ loc + scale * log(u/(1.0 - u)) */
template <typename _DataType>
void dpnp_rng_logistic_c(void* result, const double loc, const double scale, const size_t size)
{
    if (!size)
    {
        return;
    }
    cl::sycl::vector_class<cl::sycl::event> no_deps;

    const _DataType d_zero = _DataType(0.0);
    const _DataType d_one = _DataType(1.0);

    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    mkl_rng::uniform<_DataType> distribution(d_zero, d_one);
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_out.wait();

    for (size_t i = 0; i < size; i++)
        result1[i] = log(result1[i] / (1.0 - result1[i]));

    for (size_t i = 0; i < size; i++)
        result1[i] = loc + scale * result1[i];
}

template <typename _DataType>
void dpnp_rng_lognormal_c(void* result, const _DataType mean, const _DataType stddev, const size_t size)
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
void dpnp_rng_multinomial_c(
    void* result, const int ntrial, const double* p_vector, const size_t p_vector_size, const size_t size)
{
    if (!size)
    {
        return;
    }
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

    if (dpnp_queue_is_cpu_c())
    {
        mkl_rng::gaussian_mv<_DataType> distribution(dimen, mean, cov);
        auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size1, result1);
        event_out.wait();
    }
    else
    {
        int errcode = vdRngGaussianMV(VSL_RNG_METHOD_GAUSSIANMV_BOXMULLER2,
                                      get_rng_stream(),
                                      size1,
                                      result1,
                                      dimen,
                                      VSL_MATRIX_STORAGE_FULL,
                                      mean_vector,
                                      cov_vector);
        if (errcode != VSL_STATUS_OK)
        {
            throw std::runtime_error("DPNP RNG Error: dpnp_rng_multivariate_normal_c() failed.");
        }
    }
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
    cl::sycl::vector_class<cl::sycl::event> no_deps;

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
    cl::sycl::vector_class<cl::sycl::event> no_deps;

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

    cl::sycl::vector_class<cl::sycl::event> no_deps;

    const _DataType a = 0.0;
    const _DataType beta = 2.0;

    _DataType* result1 = reinterpret_cast<_DataType*>(result);

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
    if (!(size) || !(high_dim_size > 1))
    {
        return;
    }

    char* result1 = reinterpret_cast<char*>(result);

    double* Uvec = nullptr;

    size_t uvec_size = high_dim_size - 1;
    // TODO
    // nullptr check will be removed after dpnp_memory_alloc_c update
    Uvec = reinterpret_cast<double*>(dpnp_memory_alloc_c(uvec_size * sizeof(double)));
    if (Uvec == nullptr)
    {
        throw std::runtime_error("DPNP RNG Error: dpnp_rng_shuffle_c() failed.");
    }
    mkl_rng::uniform<double> uniform_distribution(0.0, 1.0);
    auto uniform_event = mkl_rng::generate(uniform_distribution, DPNP_RNG_ENGINE, uvec_size, Uvec);
    uniform_event.wait();

    if (ndim == 1)
    {
        // Fast, statically typed path: shuffle the underlying buffer.
        // Only for non-empty, 1d objects of class ndarray (subclasses such
        // as MaskedArrays may not support this approach).
        // TODO
        // kernel
        char* buf = nullptr;
        buf = reinterpret_cast<char*>(dpnp_memory_alloc_c(itemsize * sizeof(char)));
        // TODO
        // nullptr check will be removed after dpnp_memory_alloc_c update
        if (buf == nullptr)
        {
            throw std::runtime_error("DPNP RNG Error: dpnp_rng_shuffle_c() failed.");
        }
        for (size_t i = uvec_size; i > 0; i--)
        {
            size_t j = (size_t)(floor((i + 1) * Uvec[i - 1]));
            memcpy(buf, result1 + j * itemsize, itemsize);
            memcpy(result1 + j * itemsize, result1 + i * itemsize, itemsize);
            memcpy(result1 + i * itemsize, buf, itemsize);
        }

        dpnp_memory_free_c(buf);
    }
    else
    {
        // Multidimensional ndarrays require a bounce buffer.
        // TODO
        // kernel
        char* buf = nullptr;
        size_t step_size = (size / high_dim_size) * itemsize; // size in bytes for x[i] element
        buf = reinterpret_cast<char*>(dpnp_memory_alloc_c(step_size * sizeof(char)));
        // TODO
        // nullptr check will be removed after dpnp_memory_alloc_c update
        if (buf == nullptr)
        {
            throw std::runtime_error("DPNP RNG Error: dpnp_rng_shuffle_c() failed.");
        }
        for (size_t i = uvec_size; i > 0; i--)
        {
            size_t j = (size_t)(floor((i + 1) * Uvec[i - 1]));
            if (j < i)
            {
                memcpy(buf, result1 + j * step_size, step_size);
                memcpy(result1 + j * step_size, result1 + i * step_size, step_size);
                memcpy(result1 + i * step_size, buf, step_size);
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
    if (!size)
    {
        return;
    }
    cl::sycl::vector_class<cl::sycl::event> no_deps;

    _DataType* result1 = reinterpret_cast<_DataType*>(result);
    const _DataType d_zero = 0.0, d_one = 1.0;
    _DataType shape = df / 2;
    _DataType* sn = nullptr;

    if (dpnp_queue_is_cpu_c())
    {
        mkl_rng::gamma<_DataType> gamma_distribution(shape, d_zero, 1.0 / shape);
        auto event_out = mkl_rng::generate(gamma_distribution, DPNP_RNG_ENGINE, size, result1);
        event_out.wait();
        event_out = mkl_vm::invsqrt(DPNP_QUEUE, size, result1, result1, no_deps, mkl_vm::mode::ha);
        event_out.wait();

        sn = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(size * sizeof(_DataType)));
        if (sn == nullptr)
        {
            throw std::runtime_error("DPNP RNG Error: dpnp_rng_standard_t_c() failed.");
        }

        mkl_rng::gaussian<_DataType> gaussian_distribution(d_zero, d_one);
        event_out = mkl_rng::generate(gaussian_distribution, DPNP_RNG_ENGINE, size, sn);
        event_out.wait();

        event_out = mkl_vm::mul(DPNP_QUEUE, size, result1, sn, result1, no_deps, mkl_vm::mode::ha);
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
        if (sn == nullptr)
        {
            throw std::runtime_error("DPNP RNG Error: dpnp_rng_standard_t_c() failed.");
        }

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
    auto event_out = mkl_rng::generate(uniform_distribution, DPNP_RNG_ENGINE, size, result1);
    event_out.wait();

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
        cgh.parallel_for<class dpnp_rng_triangular_ration_acceptance_c_kernel<_DataType>>(gws,
                                                                                          kernel_parallel_for_func);
    };
    event_out = DPNP_QUEUE.submit(kernel_func);
    event_out.wait();
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

template <typename _DataType>
void dpnp_rng_weibull_c(void* result, const double alpha, const size_t size)
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

    fmap[DPNPFuncName::DPNP_FN_RNG_WEIBULL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_weibull_c<double>};

    return;
}
