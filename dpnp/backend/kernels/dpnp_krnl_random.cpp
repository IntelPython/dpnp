//*****************************************************************************
// Copyright (c) 2016-2023, Intel Corporation
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
#include "dpnp_random_state.hpp"

static_assert(INTEL_MKL_VERSION >= __INTEL_MKL_2023_VERSION_REQUIRED,
              "MKL does not meet minimum version requirement");

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

template <typename _DistrType, typename _EngineType, typename _DataType>
static inline DPCTLSyclEventRef
    dpnp_rng_generate(const _DistrType& distr, _EngineType& engine, const int64_t size, _DataType* result)
{
    DPCTLSyclEventRef event_ref = nullptr;
    sycl::event event;

    // perform rng generation
    try
    {
        event = mkl_rng::generate<_DistrType, _EngineType>(distr, engine, size, result);
        event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);
    }
    catch (const std::exception& e)
    {
        // TODO: add error reporting
        return event_ref;
    }
    return DPCTLEvent_Copy(event_ref);
}

template <typename _EngineType, typename _DataType>
static inline DPCTLSyclEventRef dpnp_rng_generate_uniform(
    _EngineType& engine, sycl::queue* q, const _DataType a, const _DataType b, const int64_t size, _DataType* result)
{
    DPCTLSyclEventRef event_ref = nullptr;

    if constexpr (std::is_same<_DataType, int32_t>::value)
    {
        if (q->get_device().has(sycl::aspect::fp64))
        {
            /**
             * A note from oneMKL for oneapi::mkl::rng::uniform (Discrete):
             * The oneapi::mkl::rng::uniform_method::standard uses the s BRNG type on GPU devices.
             * This might cause the produced numbers to have incorrect statistics (due to rounding error)
             * when abs(b-a) > 2^23 || abs(b) > 2^23 || abs(a) > 2^23. To get proper statistics for this case,
             * use the oneapi::mkl::rng::uniform_method::accurate method instead.
             */
            using method_type = mkl_rng::uniform_method::accurate;
            mkl_rng::uniform<_DataType, method_type> distribution(a, b);

            // perform generation
            try
            {
                sycl::event event = mkl_rng::generate(distribution, engine, size, result);

                event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);
                return DPCTLEvent_Copy(event_ref);
            }
            catch (const oneapi::mkl::unsupported_device&)
            {
                // fall through to try with uniform_method::standard
            }
            catch (const oneapi::mkl::unimplemented&)
            {
                // fall through to try with uniform_method::standard
            }
            catch (const std::exception& e)
            {
                // TODO: add error reporting
                return event_ref;
            }
        }
    }

    // uniform_method::standard is a method used by default
    using method_type = mkl_rng::uniform_method::standard;
    mkl_rng::uniform<_DataType, method_type> distribution(a, b);

    // perform generation
    return dpnp_rng_generate(distribution, engine, size, result);
}

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_beta_c(DPCTLSyclQueueRef q_ref,
                                  void* result,
                                  const _DataType a,
                                  const _DataType b,
                                  const size_t size,
                                  const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!size)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    _DataType displacement = _DataType(0.0);

    _DataType scalefactor = _DataType(1.0);

    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    mkl_rng::beta<_DataType> distribution(a, b, displacement, scalefactor);
    // perform generation
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);

    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event_out);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_rng_beta_c(void* result, const _DataType a, const _DataType b, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_beta_c<_DataType>(q_ref,
                                                             result,
                                                             a,
                                                             b,
                                                             size,
                                                             dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_beta_default_c)(void*, const _DataType, const _DataType, const size_t) = dpnp_rng_beta_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_beta_ext_c)(DPCTLSyclQueueRef,
                                         void*,
                                         const _DataType,
                                         const _DataType,
                                         const size_t,
                                         const DPCTLEventVectorRef) = dpnp_rng_beta_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_binomial_c(DPCTLSyclQueueRef q_ref,
                                      void* result,
                                      const int ntrial,
                                      const double p,
                                      const size_t size,
                                      const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (result == nullptr)
    {
        return event_ref;
    }
    if (!size)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    if (ntrial == 0 || p == 0)
    {
        event_ref = dpnp_zeros_c<_DataType>(q_ref, result, size, dep_event_vec_ref);
    }
    else if (p == 1)
    {
        _DataType* fill_value = reinterpret_cast<_DataType*>(sycl::malloc_shared(sizeof(_DataType), q));
        fill_value[0] = static_cast<_DataType>(ntrial);

        event_ref = dpnp_initval_c<_DataType>(q_ref, result, fill_value, size, dep_event_vec_ref);
        DPCTLEvent_Wait(event_ref);
        dpnp_memory_free_c(q_ref, fill_value);
    }
    else
    {
        _DataType* result1 = reinterpret_cast<_DataType*>(result);
        mkl_rng::binomial<_DataType> distribution(ntrial, p);
        auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
        event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event_out);
    }
    return DPCTLEvent_Copy(event_ref);

}

template <typename _DataType>
void dpnp_rng_binomial_c(void* result, const int ntrial, const double p, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_binomial_c<_DataType>(q_ref,
                                                                 result,
                                                                 ntrial,
                                                                 p,
                                                                 size,
                                                                 dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_binomial_default_c)(void*, const int, const double, const size_t) = dpnp_rng_binomial_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_binomial_ext_c)(DPCTLSyclQueueRef,
                                             void*,
                                             const int,
                                             const double,
                                             const size_t,
                                             const DPCTLEventVectorRef) = dpnp_rng_binomial_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_chisquare_c(DPCTLSyclQueueRef q_ref,
                                       void* result,
                                       const int df,
                                       const size_t size,
                                       const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!size)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    mkl_rng::chi_square<_DataType> distribution(df);
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);

    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event_out);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_rng_chisquare_c(void* result, const int df, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_chisquare_c<_DataType>(q_ref,
                                                                  result,
                                                                  df,
                                                                  size,
                                                                  dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_chisquare_default_c)(void*, const int, const size_t) = dpnp_rng_chisquare_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_chisquare_ext_c)(DPCTLSyclQueueRef,
                                              void*,
                                              const int,
                                              const size_t,
                                              const DPCTLEventVectorRef) = dpnp_rng_chisquare_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_exponential_c(DPCTLSyclQueueRef q_ref,
                                         void* result,
                                         const _DataType beta,
                                         const size_t size,
                                         const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!size)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    // set displacement a
    const _DataType a = (_DataType(0.0));

    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    mkl_rng::exponential<_DataType> distribution(a, beta);
    // perform generation
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);

    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event_out);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_rng_exponential_c(void* result, const _DataType beta, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_exponential_c<_DataType>(q_ref,
                                                                    result,
                                                                    beta,
                                                                    size,
                                                                    dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_exponential_default_c)(void*, const _DataType, const size_t) = dpnp_rng_exponential_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_exponential_ext_c)(DPCTLSyclQueueRef,
                                              void*,
                                              const _DataType,
                                              const size_t,
                                              const DPCTLEventVectorRef) = dpnp_rng_exponential_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_f_c(DPCTLSyclQueueRef q_ref,
                               void* result,
                               const _DataType df_num,
                               const _DataType df_den,
                               const size_t size,
                               const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!size)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));
    std::vector<sycl::event> no_deps;

    const _DataType d_zero = (_DataType(0.0));

    _DataType shape = 0.5 * df_num;
    _DataType scale = 2.0 / df_num;
    _DataType* den = nullptr;

    DPNPC_ptr_adapter<_DataType> result1_ptr(q_ref, result, size, true, true);
    _DataType* result1 = result1_ptr.get_ptr();

    mkl_rng::gamma<_DataType> gamma_distribution1(shape, d_zero, scale);
    auto event_gamma_distribution1 = mkl_rng::generate(gamma_distribution1, DPNP_RNG_ENGINE, size, result1);

    den = reinterpret_cast<_DataType*>(sycl::malloc_shared(size * sizeof(_DataType), q));
    shape = 0.5 * df_den;
    scale = 2.0 / df_den;
    mkl_rng::gamma<_DataType> gamma_distribution2(shape, d_zero, scale);
    auto event_gamma_distribution2 = mkl_rng::generate(gamma_distribution2, DPNP_RNG_ENGINE, size, den);

    auto event_out = mkl_vm::div(q,
                                 size,
                                 result1,
                                 den,
                                 result1,
                                 {event_gamma_distribution1, event_gamma_distribution2},
                                 mkl_vm::mode::ha);
    event_out.wait();

    sycl::free(den, q);

    return event_ref;
}

template <typename _DataType>
void dpnp_rng_f_c(void* result, const _DataType df_num, const _DataType df_den, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_f_c<_DataType>(q_ref,
                                                          result,
                                                          df_num,
                                                          df_den,
                                                          size,
                                                          dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_f_default_c)(void*, const _DataType, const _DataType, const size_t) = dpnp_rng_f_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_f_ext_c)(DPCTLSyclQueueRef,
                                              void*,
                                              const _DataType,
                                              const _DataType,
                                              const size_t,
                                              const DPCTLEventVectorRef) = dpnp_rng_f_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_gamma_c(DPCTLSyclQueueRef q_ref,
                      void* result,
                      const _DataType shape,
                      const _DataType scale,
                      const size_t size,
                      const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;
    sycl::event event_out;

    if (!size || result == nullptr)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    if (shape == 0.0 || scale == 0.0)
    {
        event_ref = dpnp_zeros_c<_DataType>(q_ref, result, size, dep_event_vec_ref);
    }
    else
    {
        _DataType* result1 = reinterpret_cast<_DataType*>(result);
        const _DataType a = (_DataType(0.0));

        mkl_rng::gamma<_DataType> distribution(shape, a, scale);
        event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
        event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event_out);
    }

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_rng_gamma_c(void* result, const _DataType shape, const _DataType scale, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_gamma_c<_DataType>(q_ref,
                                                              result,
                                                              shape,
                                                              scale,
                                                              size,
                                                              dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_gamma_default_c)(void*, const _DataType, const _DataType, const size_t) = dpnp_rng_gamma_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_gamma_ext_c)(DPCTLSyclQueueRef,
                                          void*,
                                          const _DataType,
                                          const _DataType,
                                          const size_t,
                                          const DPCTLEventVectorRef) = dpnp_rng_gamma_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_gaussian_c(DPCTLSyclQueueRef q_ref,
                                      void* result,
                                      const _DataType mean,
                                      const _DataType stddev,
                                      const size_t size,
                                      const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!size)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    mkl_rng::gaussian<_DataType> distribution(mean, stddev);
    // perform generation
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event_out);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_rng_gaussian_c(void* result, const _DataType mean, const _DataType stddev, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_gaussian_c<_DataType>(q_ref,
                                                                 result,
                                                                 mean,
                                                                 stddev,
                                                                 size,
                                                                 dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_gaussian_default_c)(void*,
                                    const _DataType,
                                    const _DataType,
                                    const size_t) = dpnp_rng_gaussian_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_gaussian_ext_c)(DPCTLSyclQueueRef,
                                             void*,
                                             const _DataType,
                                             const _DataType,
                                             const size_t,
                                             const DPCTLEventVectorRef) = dpnp_rng_gaussian_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_geometric_c(DPCTLSyclQueueRef q_ref,
                                       void* result,
                                       const float p,
                                       const size_t size,
                                       const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;
    sycl::event event_out;

    if (!size || !result)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    if (p == 1.0)
    {
        event_ref = dpnp_ones_c<_DataType>(q_ref, result, size, dep_event_vec_ref);
    }
    else
    {
        _DataType* result1 = reinterpret_cast<_DataType*>(result);
        mkl_rng::geometric<_DataType> distribution(p);
        // perform generation
        event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
        event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event_out);
    }
    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_rng_geometric_c(void* result, const float p, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_geometric_c<_DataType>(q_ref,
                                                                  result,
                                                                  p,
                                                                  size,
                                                                  dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_geometric_default_c)(void*, const float, const size_t) = dpnp_rng_geometric_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_geometric_ext_c)(DPCTLSyclQueueRef,
                                              void*,
                                              const float,
                                              const size_t,
                                              const DPCTLEventVectorRef) = dpnp_rng_geometric_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_gumbel_c(DPCTLSyclQueueRef q_ref,
                                    void* result,
                                    const double loc,
                                    const double scale,
                                    const size_t size,
                                    const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;
    sycl::event event_out;

    if (!size || !result)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    if (scale == 0.0)
    {
        _DataType* fill_value = reinterpret_cast<_DataType*>(sycl::malloc_shared(sizeof(_DataType), q));
        fill_value[0] = static_cast<_DataType>(loc);

        event_ref = dpnp_initval_c<_DataType>(q_ref, result, fill_value, size, dep_event_vec_ref);
        DPCTLEvent_Wait(event_ref);
        dpnp_memory_free_c(q_ref, fill_value);
    }
    else
    {
        const _DataType alpha = (_DataType(-1.0));
        std::int64_t incx = 1;
        _DataType* result1 = reinterpret_cast<_DataType*>(result);
        double negloc = loc * (double(-1.0));

        mkl_rng::gumbel<_DataType> distribution(negloc, scale);
        auto event_distribution = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);

        event_out = mkl_blas::scal(q, size, alpha, result1, incx, {event_distribution});
        event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event_out);
    }
    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_rng_gumbel_c(void* result, const double loc, const double scale, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_gumbel_c<_DataType>(q_ref,
                                                               result,
                                                               loc,
                                                               scale,
                                                               size,
                                                               dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_gumbel_default_c)(void*, const double, const double, const size_t) = dpnp_rng_gumbel_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_gumbel_ext_c)(DPCTLSyclQueueRef,
                                           void*,
                                           const double,
                                           const double,
                                           const size_t,
                                           const DPCTLEventVectorRef) = dpnp_rng_gumbel_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_hypergeometric_c(DPCTLSyclQueueRef q_ref,
                                            void* result,
                                            const int l,
                                            const int s,
                                            const int m,
                                            const size_t size,
                                            const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;
    sycl::event event_out;

    if (!size || !result)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    if (m == 0)
    {
        event_ref = dpnp_zeros_c<_DataType>(q_ref, result, size, dep_event_vec_ref);
    }
    else if (l == m)
    {
        _DataType* fill_value = reinterpret_cast<_DataType*>(sycl::malloc_shared(sizeof(_DataType), q));
        fill_value[0] = static_cast<_DataType>(s);

        event_ref = dpnp_initval_c<_DataType>(q_ref, result, fill_value, size, dep_event_vec_ref);
        DPCTLEvent_Wait(event_ref);
        dpnp_memory_free_c(q_ref, fill_value);
    }
    else
    {
        _DataType* result1 = reinterpret_cast<_DataType*>(result);
        mkl_rng::hypergeometric<_DataType> distribution(l, s, m);
        event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
        event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event_out);

    }
    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_rng_hypergeometric_c(void* result, const int l, const int s, const int m, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_hypergeometric_c<_DataType>(q_ref,
                                                                       result,
                                                                       l,
                                                                       s,
                                                                       m,
                                                                       size,
                                                                       dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_hypergeometric_default_c)(void*,
                                          const int,
                                          const int,
                                          const int,
                                          const size_t) = dpnp_rng_hypergeometric_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_hypergeometric_ext_c)(DPCTLSyclQueueRef,
                                                   void*,
                                                   const int,
                                                   const int,
                                                   const int,
                                                   const size_t,
                                                   const DPCTLEventVectorRef) = dpnp_rng_hypergeometric_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_laplace_c(DPCTLSyclQueueRef q_ref,
                                     void* result,
                                     const double loc,
                                     const double scale,
                                     const size_t size,
                                     const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;
    sycl::event event_out;

    if (!size || !result)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    if (scale == 0.0)
    {
        event_ref = dpnp_zeros_c<_DataType>(q_ref, result, size, dep_event_vec_ref);
    }
    else
    {
        _DataType* result1 = reinterpret_cast<_DataType*>(result);
        mkl_rng::laplace<_DataType> distribution(loc, scale);
        // perform generation
        event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
        event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event_out);
    }
    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_rng_laplace_c(void* result, const double loc, const double scale, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_laplace_c<_DataType>(q_ref,
                                                                result,
                                                                loc,
                                                                scale,
                                                                size,
                                                                dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_laplace_default_c)(void*, const double, const double, const size_t) = dpnp_rng_laplace_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_laplace_ext_c)(DPCTLSyclQueueRef,
                                            void*,
                                            const double,
                                            const double,
                                            const size_t,
                                            const DPCTLEventVectorRef) = dpnp_rng_laplace_c<_DataType>;

template <typename _KernelNameSpecialization>
class dpnp_rng_logistic_c_kernel;

/*   Logistic(loc, scale) ~ loc + scale * log(u/(1.0 - u)) */
template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_logistic_c(DPCTLSyclQueueRef q_ref,
                                      void* result,
                                      const double loc,
                                      const double scale,
                                      const size_t size,
                                      const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!size || !result)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    const _DataType d_zero = _DataType(0.0);
    const _DataType d_one = _DataType(1.0);

    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    mkl_rng::uniform<_DataType> distribution(d_zero, d_one);
    auto event_distribution = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);

    sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
        size_t i = global_id[0];
        result1[i] = sycl::log(result1[i] / (1.0 - result1[i]));
        result1[i] = loc + scale * result1[i];
    };
    auto kernel_func = [&](sycl::handler& cgh) {
        cgh.depends_on({event_distribution});
        cgh.parallel_for<class dpnp_rng_logistic_c_kernel<_DataType>>(gws, kernel_parallel_for_func);
    };
    auto event = q.submit(kernel_func);
    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_rng_logistic_c(void* result, const double loc, const double scale, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_logistic_c<_DataType>(q_ref,
                                                                 result,
                                                                 loc,
                                                                 scale,
                                                                 size,
                                                                 dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_logistic_default_c)(void*, const double, const double, const size_t) = dpnp_rng_logistic_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_logistic_ext_c)(DPCTLSyclQueueRef,
                                             void*,
                                             const double,
                                             const double,
                                             const size_t,
                                             const DPCTLEventVectorRef) = dpnp_rng_logistic_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_lognormal_c(DPCTLSyclQueueRef q_ref,
                                       void* result,
                                       const _DataType mean,
                                       const _DataType stddev,
                                       const size_t size,
                                       const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;
    sycl::event event_out;

    if (!size || !result)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    if (stddev == 0.0)
    {
        _DataType* fill_value = reinterpret_cast<_DataType*>(sycl::malloc_shared(sizeof(_DataType), q));
        fill_value[0] = static_cast<_DataType>(std::exp(mean + (stddev * stddev) / 2));

        event_ref = dpnp_initval_c<_DataType>(q_ref, result, fill_value, size, dep_event_vec_ref);
        DPCTLEvent_Wait(event_ref);
        dpnp_memory_free_c(q_ref, fill_value);
    }
    else
    {
        const _DataType displacement = _DataType(0.0);
        const _DataType scalefactor = _DataType(1.0);

        mkl_rng::lognormal<_DataType> distribution(mean, stddev, displacement, scalefactor);
        event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
        event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event_out);
    }
    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_rng_lognormal_c(void* result, const _DataType mean, const _DataType stddev, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_lognormal_c<_DataType>(q_ref,
                                                                  result,
                                                                  mean,
                                                                  stddev,
                                                                  size,
                                                                  dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_lognormal_default_c)(void*,
                                     const _DataType,
                                     const _DataType,
                                     const size_t) = dpnp_rng_lognormal_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_lognormal_ext_c)(DPCTLSyclQueueRef,
                                              void*,
                                              const _DataType,
                                              const _DataType,
                                              const size_t,
                                              const DPCTLEventVectorRef) = dpnp_rng_lognormal_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_multinomial_c(DPCTLSyclQueueRef q_ref,
                                         void* result,
                                         const int ntrial,
                                         void* p_in,
                                         const size_t p_size,
                                         const size_t size,
                                         const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;
    sycl::event event_out;

    if (!size || !result || (ntrial < 0))
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    if (ntrial == 0)
    {
        event_ref = dpnp_zeros_c<_DataType>(q_ref, result, size, dep_event_vec_ref);
    }
    else
    {
        DPNPC_ptr_adapter<double> p_ptr(q_ref, p_in, p_size, true);
        double* p_data = p_ptr.get_ptr();

        // size = size
        // `result` is a array for random numbers
        // `size` is a `result`'s len. `size = n * p_size`
        // `n` is a number of random values to be generated.
        size_t n = size / p_size;

        size_t is_cpu_queue = dpnp_queue_is_cpu_c();

        // math library supports the distribution generation on GPU device with input parameters
        // which follow the condition
        if (is_cpu_queue || (!is_cpu_queue && (p_size >= ((size_t)ntrial * 16)) && (ntrial <= 16)))
        {
            DPNPC_ptr_adapter<_DataType> result_ptr(q_ref, result, size, true, true);
            _DataType* result1 = result_ptr.get_ptr();

            auto p = sycl::span<double>{p_data, p_size};
            mkl_rng::multinomial<_DataType> distribution(ntrial, p);

            // perform generation
            event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, n, result1);
            event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event_out);

            p_ptr.depends_on(event_out);
            result_ptr.depends_on(event_out);
        }
        else
        {
            DPNPC_ptr_adapter<_DataType> result_ptr(q_ref, result, size, true, true);
            _DataType* result1 = result_ptr.get_ptr();
            int errcode = viRngMultinomial(
                VSL_RNG_METHOD_MULTINOMIAL_MULTPOISSON, get_rng_stream(), n, result1, ntrial, p_size, p_data);
            if (errcode != VSL_STATUS_OK)
            {
                throw std::runtime_error("DPNP RNG Error: dpnp_rng_multinomial_c() failed.");
            }
        }
    }
    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_rng_multinomial_c(
    void* result, const int ntrial, void* p_in, const size_t p_size, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_multinomial_c<_DataType>(q_ref,
                                                                    result,
                                                                    ntrial,
                                                                    p_in,
                                                                    p_size,
                                                                    size,
                                                                    dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_multinomial_default_c)(void*,
                                       const int,
                                       void*,
                                       const size_t,
                                       const size_t) = dpnp_rng_multinomial_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_multinomial_ext_c)(DPCTLSyclQueueRef,
                                              void*,
                                              const int,
                                              void*,
                                              const size_t,
                                              const size_t,
                                              const DPCTLEventVectorRef) = dpnp_rng_multinomial_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_multivariate_normal_c(DPCTLSyclQueueRef q_ref,
                                                 void* result,
                                                 const int dimen,
                                                 void* mean_in,
                                                 const size_t mean_size,
                                                 void* cov_in,
                                                 const size_t cov_size,
                                                 const size_t size,
                                                 const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!size)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    DPNPC_ptr_adapter<double> mean_ptr(q_ref, mean_in, mean_size, true);
    double* mean_data = mean_ptr.get_ptr();
    DPNPC_ptr_adapter<double> cov_ptr(q_ref, cov_in, cov_size, true);
    double* cov_data = cov_ptr.get_ptr();

    _DataType* result1 = static_cast<_DataType *>(result);

    auto mean = sycl::span<double>{mean_data, mean_size};
    auto cov = sycl::span<double>{cov_data, cov_size};

    // `result` is a array for random numbers
    // `size` is a `result`'s len.
    // `size1` is a number of random values to be generated for each dimension.
    size_t size1 = size / dimen;

    mkl_rng::gaussian_mv<_DataType> distribution(dimen, mean, cov);
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size1, result1);
    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event_out);

    mean_ptr.depends_on(event_out);
    cov_ptr.depends_on(event_out);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_rng_multivariate_normal_c(void* result,
                                    const int dimen,
                                    void* mean_in,
                                    const size_t mean_size,
                                    void* cov_in,
                                    const size_t cov_size,
                                    const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_multivariate_normal_c<_DataType>(q_ref,
                                                                            result,
                                                                            dimen,
                                                                            mean_in,
                                                                            mean_size,
                                                                            cov_in,
                                                                            cov_size,
                                                                            size,
                                                                            dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_multivariate_normal_default_c)(void*,
                                               const int,
                                               void*,
                                               const size_t,
                                               void*,
                                               const size_t,
                                               const size_t) = dpnp_rng_multivariate_normal_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_multivariate_normal_ext_c)(DPCTLSyclQueueRef,
                                                        void*,
                                                        const int,
                                                        void*,
                                                        const size_t,
                                                        void*,
                                                        const size_t,
                                                        const size_t,
                                                        const DPCTLEventVectorRef) = dpnp_rng_multivariate_normal_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_negative_binomial_c(DPCTLSyclQueueRef q_ref,
                                               void* result,
                                               const double a,
                                               const double p,
                                               const size_t size,
                                               const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!size)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    _DataType* result1 = reinterpret_cast<_DataType*>(result);
    mkl_rng::negative_binomial<_DataType> distribution(a, p);
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event_out);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_rng_negative_binomial_c(void* result, const double a, const double p, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_negative_binomial_c<_DataType>(q_ref,
                                                                          result,
                                                                          a,
                                                                          p,
                                                                          size,
                                                                          dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_negative_binomial_default_c)(void*,
                                             const double,
                                             const double,
                                             const size_t) = dpnp_rng_negative_binomial_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_negative_binomial_ext_c)(
    DPCTLSyclQueueRef,
    void*,
    const double,
    const double,
    const size_t,
    const DPCTLEventVectorRef) = dpnp_rng_negative_binomial_c<_DataType>;

template <typename _KernelNameSpecialization>
class dpnp_rng_noncentral_chisquare_c_kernel1;
template <typename _KernelNameSpecialization>
class dpnp_rng_noncentral_chisquare_c_kernel2;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_noncentral_chisquare_c(DPCTLSyclQueueRef q_ref,
                                                  void* result,
                                                  const _DataType df,
                                                  const _DataType nonc,
                                                  const size_t size,
                                                  const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;
    sycl::event event_out;

    if (!size || !result)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    DPNPC_ptr_adapter<_DataType> result1_ptr(q_ref, result, size, false, true);
    _DataType* result1 = result1_ptr.get_ptr();

    const _DataType d_zero = _DataType(0.0);
    const _DataType d_one = _DataType(1.0);
    const _DataType d_two = _DataType(2.0);

    _DataType shape, loc;
    size_t i;

    if (df > 1)
    {
        _DataType* nvec = nullptr;

        shape = 0.5 * (df - 1.0);
        /* res has chi^2 with (df - 1) */
        mkl_rng::gamma<_DataType> gamma_distribution(shape, d_zero, d_two);
        auto event_gamma_distr = mkl_rng::generate(gamma_distribution, DPNP_RNG_ENGINE, size, result1);

        nvec = reinterpret_cast<_DataType*>(sycl::malloc_shared(size * sizeof(_DataType), q));

        loc = sqrt(nonc);

        mkl_rng::gaussian<_DataType> gaussian_distribution(loc, d_one);
        auto event_gaussian_distr = mkl_rng::generate(gaussian_distribution, DPNP_RNG_ENGINE, size, nvec);

        /* squaring could result in an overflow */
        auto event_sqr_out =
            mkl_vm::sqr(q, size, nvec, nvec, {event_gamma_distr, event_gaussian_distr}, mkl_vm::mode::ha);
        auto event_add_out = mkl_vm::add(q, size, result1, nvec, result1, {event_sqr_out}, mkl_vm::mode::ha);
        event_add_out.wait();
        sycl::free(nvec, q);
    }
    else if (df < 1)
    {
        /* noncentral_chisquare(df, nonc) ~ G( df/2 + Poisson(nonc/2), 2) */
        double lambda;
        int* pvec = nullptr;
        pvec = reinterpret_cast<int*>(sycl::malloc_shared(size * sizeof(int), q));
        lambda = 0.5 * nonc;

        mkl_rng::poisson<int> poisson_distribution(lambda);
        event_out = mkl_rng::generate(poisson_distribution, DPNP_RNG_ENGINE, size, pvec);
        event_out.wait();

        shape = 0.5 * df;
        if (0.125 * size > sqrt(lambda))
        {
            size_t* idx = nullptr;
            _DataType* tmp = nullptr;
            idx = reinterpret_cast<size_t*>(sycl::malloc_shared(size * sizeof(size_t), q));

            sycl::range<1> gws1(size);
            auto kernel_parallel_for_func1 = [=](sycl::id<1> global_id) {
                size_t i = global_id[0];
                idx[i] = i;
            };
            auto kernel_func1 = [&](sycl::handler& cgh) {
                cgh.parallel_for<class dpnp_rng_noncentral_chisquare_c_kernel1<_DataType>>(gws1,
                                                                                           kernel_parallel_for_func1);
            };
            event_out = q.submit(kernel_func1);
            event_out.wait();

            std::sort(idx, idx + size, [pvec](size_t i1, size_t i2) { return pvec[i1] < pvec[i2]; });
            /* idx now contains original indexes of ordered Poisson outputs */

            /* allocate workspace to store samples of gamma, enough to hold entire output */
            tmp = reinterpret_cast<_DataType*>(sycl::malloc_shared(size * sizeof(_DataType), q));
            for (i = 0; i < size;)
            {
                size_t j;
                int cv = pvec[idx[i]];
                // TODO vectorize
                for (j = i + 1; (j < size) && (pvec[idx[j]] == cv); j++)
                {
                }

                if (j <= i)
                {
                    throw std::runtime_error("DPNP RNG Error: dpnp_rng_noncentral_chisquare_c() failed.");
                }
                mkl_rng::gamma<_DataType> gamma_distribution(shape + cv, d_zero, d_two);
                event_out = mkl_rng::generate(gamma_distribution, DPNP_RNG_ENGINE, j - i, tmp);
                event_out.wait();

                sycl::range<1> gws2(j - i);
                auto kernel_parallel_for_func2 = [=](sycl::id<1> global_id) {
                    size_t index = global_id[0];
                    result1[idx[index + i]] = tmp[index];
                };
                auto kernel_func2 = [&](sycl::handler& cgh) {
                    cgh.parallel_for<class dpnp_rng_noncentral_chisquare_c_kernel2<_DataType>>(
                        gws2, kernel_parallel_for_func2);
                };
                event_out = q.submit(kernel_func2);
                event_out.wait();

                i = j;
            }
            sycl::free(tmp, q);
            sycl::free(idx, q);
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
        sycl::free(pvec, q);
    }
    else
    {
        /* noncentral_chisquare(1, nonc) ~ (Z + sqrt(nonc))**2 for df == 1 */
        loc = sqrt(nonc);
        mkl_rng::gaussian<_DataType> gaussian_distribution(loc, d_one);
        auto event_gaussian_distr = mkl_rng::generate(gaussian_distribution, DPNP_RNG_ENGINE, size, result1);
        event_out = mkl_vm::sqr(q, size, result1, result1, {event_gaussian_distr}, mkl_vm::mode::ha);
        event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event_out);
    }
    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_rng_noncentral_chisquare_c(void* result, const _DataType df, const _DataType nonc, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_noncentral_chisquare_c<_DataType>(q_ref,
                                                                             result,
                                                                             df,
                                                                             nonc,
                                                                             size,
                                                                             dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_noncentral_chisquare_default_c)(void*,
                                                const _DataType,
                                                const _DataType,
                                                const size_t) = dpnp_rng_noncentral_chisquare_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_noncentral_chisquare_ext_c)(
    DPCTLSyclQueueRef,
    void*,
    const _DataType,
    const _DataType,
    const size_t,
    const DPCTLEventVectorRef) = dpnp_rng_noncentral_chisquare_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_normal_c(DPCTLSyclQueueRef q_ref,
                                    void* result_out,
                                    const double mean_in,
                                    const double stddev_in,
                                    const int64_t size,
                                    void* random_state_in,
                                    const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;
    sycl::queue* q = reinterpret_cast<sycl::queue*>(q_ref);

    if (!size)
    {
        return event_ref;
    }
    assert(q != nullptr);

    _DataType* result = static_cast<_DataType*>(result_out);

    // set mean of distribution
    const _DataType mean = static_cast<_DataType>(mean_in);
    // set standard deviation of distribution
    const _DataType stddev = static_cast<_DataType>(stddev_in);

    mkl_rng::gaussian<_DataType> distribution(mean, stddev);

    if (q->get_device().is_cpu())
    {
        mt19937_struct* random_state = static_cast<mt19937_struct*>(random_state_in);
        mkl_rng::mt19937* engine = static_cast<mkl_rng::mt19937*>(random_state->engine);

        // perform generation with MT19937 engine
        event_ref = dpnp_rng_generate(distribution, *engine, size, result);
    }
    else
    {
        mcg59_struct* random_state = static_cast<mcg59_struct*>(random_state_in);
        mkl_rng::mcg59* engine = static_cast<mkl_rng::mcg59*>(random_state->engine);

        // perform generation with MCG59 engine
        event_ref = dpnp_rng_generate(distribution, *engine, size, result);
    }
    return event_ref;
}

template <typename _DataType>
void dpnp_rng_normal_c(void* result, const _DataType mean, const _DataType stddev, const size_t size)
{
    sycl::queue* q = &DPNP_QUEUE;
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(q);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = nullptr;

    if (q->get_device().is_cpu())
    {
        mt19937_struct* mt19937 = new mt19937_struct();
        mt19937->engine = &DPNP_RNG_ENGINE;

        event_ref = dpnp_rng_normal_c<_DataType>(
            q_ref, result, mean, stddev, static_cast<int64_t>(size), mt19937, dep_event_vec_ref);
        DPCTLEvent_WaitAndThrow(event_ref);
        DPCTLEvent_Delete(event_ref);
        delete mt19937;
    }
    else
    {
        // MCG59 engine is assumed to provide a better performance on GPU than MT19937
        mcg59_struct* mcg59 = new mcg59_struct();
        mcg59->engine = &DPNP_RNG_MCG59_ENGINE;

        event_ref = dpnp_rng_normal_c<_DataType>(
            q_ref, result, mean, stddev, static_cast<int64_t>(size), mcg59, dep_event_vec_ref);
        DPCTLEvent_WaitAndThrow(event_ref);
        DPCTLEvent_Delete(event_ref);
        delete mcg59;
    }
}

template <typename _DataType>
void (*dpnp_rng_normal_default_c)(void*,
                                  const _DataType,
                                  const _DataType,
                                  const size_t) = dpnp_rng_normal_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_normal_ext_c)(DPCTLSyclQueueRef,
                                           void*,
                                           const double,
                                           const double,
                                           const int64_t,
                                           void*,
                                           const DPCTLEventVectorRef) = dpnp_rng_normal_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_pareto_c(DPCTLSyclQueueRef q_ref,
                                    void* result,
                                    const double alpha,
                                    const size_t size,
                                    const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!size)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));
    std::vector<sycl::event> no_deps;

    const _DataType d_zero = _DataType(0.0);
    const _DataType d_one = _DataType(1.0);
    _DataType neg_rec_alp = -1.0 / alpha;

    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    mkl_rng::uniform<_DataType> distribution(d_zero, d_one);
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_out.wait();

    event_out = mkl_vm::powx(q, size, result1, neg_rec_alp, result1, no_deps, mkl_vm::mode::ha);
    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event_out);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_rng_pareto_c(void* result, const double alpha, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_pareto_c<_DataType>(q_ref,
                                                               result,
                                                               alpha,
                                                               size,
                                                               dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_pareto_default_c)(void*, const double, const size_t) = dpnp_rng_pareto_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_pareto_ext_c)(DPCTLSyclQueueRef,
                                           void*,
                                           const double,
                                           const size_t,
                                           const DPCTLEventVectorRef) = dpnp_rng_pareto_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_poisson_c(DPCTLSyclQueueRef q_ref,
                                     void* result,
                                     const double lambda,
                                     const size_t size,
                                     const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!size)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    mkl_rng::poisson<_DataType> distribution(lambda);
    // perform generation
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event_out);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_rng_poisson_c(void* result, const double lambda, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_poisson_c<_DataType>(q_ref,
                                                                result,
                                                                lambda,
                                                                size,
                                                                dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_poisson_default_c)(void*, const double, const size_t) = dpnp_rng_poisson_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_poisson_ext_c)(DPCTLSyclQueueRef,
                                            void*,
                                            const double,
                                            const size_t,
                                            const DPCTLEventVectorRef) = dpnp_rng_poisson_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_power_c(DPCTLSyclQueueRef q_ref,
                                   void* result,
                                   const double alpha,
                                   const size_t size,
                                   const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!size)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));
    std::vector<sycl::event> no_deps;

    const _DataType d_zero = _DataType(0.0);
    const _DataType d_one = _DataType(1.0);
    _DataType neg_rec_alp = 1.0 / alpha;

    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    mkl_rng::uniform<_DataType> distribution(d_zero, d_one);
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_out.wait();

    event_out = mkl_vm::powx(q, size, result1, neg_rec_alp, result1, no_deps, mkl_vm::mode::ha);
    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event_out);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_rng_power_c(void* result, const double alpha, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_power_c<_DataType>(q_ref,
                                                              result,
                                                              alpha,
                                                              size,
                                                              dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_power_default_c)(void*, const double, const size_t) = dpnp_rng_power_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_power_ext_c)(DPCTLSyclQueueRef,
                                          void*,
                                          const double,
                                          const size_t,
                                          const DPCTLEventVectorRef) = dpnp_rng_power_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_rayleigh_c(DPCTLSyclQueueRef q_ref,
                                      void* result,
                                      const _DataType scale,
                                      const size_t size,
                                      const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!size)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));
    std::vector<sycl::event> no_deps;

    const _DataType a = 0.0;
    const _DataType beta = 2.0;

    DPNPC_ptr_adapter<_DataType> result1_ptr(q_ref, result, size);
    _DataType* result1 = result1_ptr.get_ptr();

    mkl_rng::exponential<_DataType> distribution(a, beta);

    auto exponential_rng_event = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    auto sqrt_event = mkl_vm::sqrt(q, size, result1, result1, {exponential_rng_event}, mkl_vm::mode::ha);
    auto scal_event = mkl_blas::scal(q, size, scale, result1, 1, {sqrt_event});
    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&scal_event);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_rng_rayleigh_c(void* result, const _DataType scale, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_rayleigh_c<_DataType>(q_ref,
                                                                 result,
                                                                 scale,
                                                                 size,
                                                                 dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_rayleigh_default_c)(void*, const _DataType, const size_t) = dpnp_rng_rayleigh_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_rayleigh_ext_c)(DPCTLSyclQueueRef,
                                             void*,
                                             const _DataType,
                                             const size_t,
                                             const DPCTLEventVectorRef) = dpnp_rng_rayleigh_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_shuffle_c(DPCTLSyclQueueRef q_ref,
                                     void* result,
                                     const size_t itemsize,
                                     const size_t ndim,
                                     const size_t high_dim_size,
                                     const size_t size,
                                     const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!result)
    {
        return event_ref;
    }

    if (!size || !ndim || !(high_dim_size > 1))
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    DPNPC_ptr_adapter<char> result1_ptr(q_ref, result, size * itemsize, true, true);
    char* result1 = result1_ptr.get_ptr();

    size_t uvec_size = high_dim_size - 1;
    double* Uvec = reinterpret_cast<double*>(sycl::malloc_shared(uvec_size * sizeof(double), q));
    mkl_rng::uniform<double> uniform_distribution(0.0, 1.0);
    auto uniform_event = mkl_rng::generate(uniform_distribution, DPNP_RNG_ENGINE, uvec_size, Uvec);
    uniform_event.wait();

    if (ndim == 1)
    {
        // Fast, statically typed path: shuffle the underlying buffer.
        // Only for non-empty, 1d objects of class ndarray (subclasses such
        // as MaskedArrays may not support this approach).
        void* buf = sycl::malloc_device(itemsize, q);
        for (size_t i = uvec_size; i > 0; i--)
        {
            size_t j = (size_t)(floor((i + 1) * Uvec[i - 1]));
            if (i != j)
            {
                auto memcpy1 = q.memcpy(buf, result1 + j * itemsize, itemsize);
                auto memcpy2 = q.memcpy(result1 + j * itemsize, result1 + i * itemsize, itemsize, memcpy1);
                q.memcpy(result1 + i * itemsize, buf, itemsize, memcpy2).wait();
            }
        }
        sycl::free(buf, q);
    }
    else
    {
        // Multidimensional ndarrays require a bounce buffer.
        size_t step_size = (size / high_dim_size) * itemsize; // size in bytes for x[i] element
        void* buf = sycl::malloc_device(step_size, q);
        for (size_t i = uvec_size; i > 0; i--)
        {
            size_t j = (size_t)(floor((i + 1) * Uvec[i - 1]));
            if (j < i)
            {
                auto memcpy1 = q.memcpy(buf, result1 + j * step_size, step_size);
                auto memcpy2 = q.memcpy(result1 + j * step_size, result1 + i * step_size, step_size, memcpy1);
                q.memcpy(result1 + i * step_size, buf, step_size, memcpy2).wait();
            }
        }
        sycl::free(buf, q);
    }

    sycl::free(Uvec, q);

    return event_ref;
}

template <typename _DataType>
void dpnp_rng_shuffle_c(
    void* result, const size_t itemsize, const size_t ndim, const size_t high_dim_size, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_shuffle_c<_DataType>(q_ref,
                                                                result,
                                                                itemsize,
                                                                ndim,
                                                                high_dim_size,
                                                                size,
                                                                dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_shuffle_default_c)(void*,
                                   const size_t,
                                   const size_t,
                                   const size_t,
                                   const size_t) = dpnp_rng_shuffle_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_shuffle_ext_c)(DPCTLSyclQueueRef,
                                            void*,
                                            const size_t,
                                            const size_t,
                                            const size_t,
                                            const size_t,
                                            const DPCTLEventVectorRef) = dpnp_rng_shuffle_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_standard_cauchy_c(DPCTLSyclQueueRef q_ref,
                                             void* result,
                                             const size_t size,
                                             const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!size)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    _DataType* result1 = reinterpret_cast<_DataType*>(result);

    const _DataType displacement = _DataType(0.0);

    const _DataType scalefactor = _DataType(1.0);

    mkl_rng::cauchy<_DataType> distribution(displacement, scalefactor);
    // perform generation
    auto event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event_out);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_rng_standard_cauchy_c(void* result, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_standard_cauchy_c<_DataType>(q_ref,
                                                                        result,
                                                                        size,
                                                                        dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_standard_cauchy_default_c)(void*, const size_t) = dpnp_rng_standard_cauchy_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_standard_cauchy_ext_c)(DPCTLSyclQueueRef,
                                                    void*,
                                                    const size_t,
                                                    const DPCTLEventVectorRef) = dpnp_rng_standard_cauchy_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_standard_exponential_c(DPCTLSyclQueueRef q_ref,
                                                  void* result,
                                                  const size_t size,
                                                  const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!size)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    // set displacement a
    const _DataType beta = (_DataType(1.0));

    event_ref = dpnp_rng_exponential_c(q_ref, result, beta, size, dep_event_vec_ref);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_rng_standard_exponential_c(void* result, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_standard_exponential_c<_DataType>(q_ref,
                                                                             result,
                                                                             size,
                                                                             dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_standard_exponential_default_c)(void*, const size_t) = dpnp_rng_standard_exponential_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_standard_exponential_ext_c)(
    DPCTLSyclQueueRef,
    void*,
    const size_t,
    const DPCTLEventVectorRef) = dpnp_rng_standard_exponential_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_standard_gamma_c(DPCTLSyclQueueRef q_ref,
                                            void* result,
                                            const _DataType shape,
                                            const size_t size,
                                            const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!size)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    const _DataType scale = _DataType(1.0);

    event_ref = dpnp_rng_gamma_c(q_ref, result, shape, scale, size, dep_event_vec_ref);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_rng_standard_gamma_c(void* result, const _DataType shape, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_standard_gamma_c<_DataType>(q_ref,
                                                                       result,
                                                                       shape,
                                                                       size,
                                                                       dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_standard_gamma_default_c)(void*, const _DataType, const size_t) = dpnp_rng_standard_gamma_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_standard_gamma_ext_c)(DPCTLSyclQueueRef,
                                                   void*,
                                                   const _DataType,
                                                   const size_t,
                                                   const DPCTLEventVectorRef) = dpnp_rng_standard_gamma_c<_DataType>;

template <typename _DataType>
void dpnp_rng_standard_normal_c(void* result, size_t size)
{
    dpnp_rng_normal_c(result, _DataType(0.0), _DataType(1.0), size);
}

template <typename _DataType>
void (*dpnp_rng_standard_normal_default_c)(void*, const size_t) = dpnp_rng_standard_normal_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_standard_t_c(DPCTLSyclQueueRef q_ref,
                                        void* result,
                                        const _DataType df,
                                        const size_t size,
                                        const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!size || !result)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    _DataType* result1 = reinterpret_cast<_DataType*>(result);
    const _DataType d_zero = 0.0, d_one = 1.0;
    _DataType shape = df / 2;
    _DataType* sn = nullptr;

    mkl_rng::gamma<_DataType> gamma_distribution(shape, d_zero, 1.0 / shape);
    auto gamma_distr_event = mkl_rng::generate(gamma_distribution, DPNP_RNG_ENGINE, size, result1);

    auto invsqrt_event = mkl_vm::invsqrt(q, size, result1, result1, {gamma_distr_event}, mkl_vm::mode::ha);

    sn = reinterpret_cast<_DataType*>(sycl::malloc_shared(size * sizeof(_DataType), q));

    mkl_rng::gaussian<_DataType> gaussian_distribution(d_zero, d_one);
    auto gaussian_distr_event = mkl_rng::generate(gaussian_distribution, DPNP_RNG_ENGINE, size, sn);

    auto event_out =
        mkl_vm::mul(q, size, result1, sn, result1, {invsqrt_event, gaussian_distr_event}, mkl_vm::mode::ha);
    event_out.wait();

    sycl::free(sn, q);

    return event_ref;
}

template <typename _DataType>
void dpnp_rng_standard_t_c(void* result, const _DataType df, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_standard_t_c<_DataType>(q_ref,
                                                                   result,
                                                                   df,
                                                                   size,
                                                                   dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_standard_t_default_c)(void*, const _DataType, const size_t) = dpnp_rng_standard_t_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_standard_t_ext_c)(DPCTLSyclQueueRef,
                                               void*,
                                               const _DataType,
                                               const size_t,
                                               const DPCTLEventVectorRef) = dpnp_rng_standard_t_c<_DataType>;

template <typename _KernelNameSpecialization>
class dpnp_rng_triangular_ration_acceptance_c_kernel;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_triangular_c(DPCTLSyclQueueRef q_ref,
                                        void* result,
                                        const _DataType x_min,
                                        const _DataType x_mode,
                                        const _DataType x_max,
                                        const size_t size,
                                        const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!size)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

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

    sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
        size_t i = global_id[0];
        if (ratio <= 0)
        {
            result1[i] = x_max - sycl::sqrt(result1[i] * rpr);
        }
        else if (ratio >= 1)
        {
            result1[i] = x_min + sycl::sqrt(result1[i] * lpr);
        }
        else
        {
            _DataType ui = result1[i];
            if (ui > ratio)
            {
                result1[i] = x_max - sycl::sqrt((1.0 - ui) * rpr);
            }
            else
            {
                result1[i] = x_min + sycl::sqrt(ui * lpr);
            }
        }
    };
    auto kernel_func = [&](sycl::handler& cgh) {
        cgh.depends_on({event_uniform});
        cgh.parallel_for<class dpnp_rng_triangular_ration_acceptance_c_kernel<_DataType>>(gws,
                                                                                          kernel_parallel_for_func);
    };
    auto event_ration_acceptance = q.submit(kernel_func);
    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event_ration_acceptance);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_rng_triangular_c(
    void* result, const _DataType x_min, const _DataType x_mode, const _DataType x_max, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_triangular_c<_DataType>(q_ref,
                                                                   result,
                                                                   x_min,
                                                                   x_mode,
                                                                   x_max,
                                                                   size,
                                                                   dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_triangular_default_c)(void*,
                                      const _DataType,
                                      const _DataType,
                                      const _DataType,
                                      const size_t) = dpnp_rng_triangular_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_triangular_ext_c)(DPCTLSyclQueueRef,
                                               void*,
                                               const _DataType,
                                               const _DataType,
                                               const _DataType,
                                               const size_t,
                                               const DPCTLEventVectorRef) = dpnp_rng_triangular_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_uniform_c(DPCTLSyclQueueRef q_ref,
                                     void* result_out,
                                     const double low,
                                     const double high,
                                     const int64_t size,
                                     void* random_state_in,
                                     const DPCTLEventVectorRef dep_event_vec_ref)
{
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!size)
    {
        return event_ref;
    }

    sycl::queue* q = reinterpret_cast<sycl::queue*>(q_ref);

    _DataType* result = static_cast<_DataType*>(result_out);

    // set left bound of distribution
    const _DataType a = static_cast<_DataType>(low);
    // set right bound of distribution
    const _DataType b = static_cast<_DataType>(high);

    if (q->get_device().is_cpu())
    {
        mt19937_struct* random_state = static_cast<mt19937_struct*>(random_state_in);
        mkl_rng::mt19937* engine = static_cast<mkl_rng::mt19937*>(random_state->engine);

        // perform generation with MT19937 engine
        event_ref = dpnp_rng_generate_uniform(*engine, q, a, b, size, result);
    }
    else
    {
        mcg59_struct* random_state = static_cast<mcg59_struct*>(random_state_in);
        mkl_rng::mcg59* engine = static_cast<mkl_rng::mcg59*>(random_state->engine);

        // perform generation with MCG59 engine
        event_ref = dpnp_rng_generate_uniform(*engine, q, a, b, size, result);
    }
    return event_ref;
}

template <typename _DataType>
void dpnp_rng_uniform_c(void* result, const long low, const long high, const size_t size)
{
    sycl::queue* q = &DPNP_QUEUE;
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(q);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = nullptr;

    if (q->get_device().is_cpu())
    {
        mt19937_struct* mt19937 = new mt19937_struct();
        mt19937->engine = &DPNP_RNG_ENGINE;

        event_ref = dpnp_rng_uniform_c<_DataType>(q_ref,
                                                  result,
                                                  static_cast<double>(low),
                                                  static_cast<double>(high),
                                                  static_cast<int64_t>(size),
                                                  mt19937,
                                                  dep_event_vec_ref);
        DPCTLEvent_WaitAndThrow(event_ref);
        DPCTLEvent_Delete(event_ref);
        delete mt19937;
    }
    else
    {
        // MCG59 engine is assumed to provide a better performance on GPU than MT19937
        mcg59_struct* mcg59 = new mcg59_struct();
        mcg59->engine = &DPNP_RNG_MCG59_ENGINE;

        event_ref = dpnp_rng_uniform_c<_DataType>(q_ref,
                                                  result,
                                                  static_cast<double>(low),
                                                  static_cast<double>(high),
                                                  static_cast<int64_t>(size),
                                                  mcg59,
                                                  dep_event_vec_ref);
        DPCTLEvent_WaitAndThrow(event_ref);
        DPCTLEvent_Delete(event_ref);
        delete mcg59;
    }
}

template <typename _DataType>
void (*dpnp_rng_uniform_default_c)(void*, const long, const long, const size_t) = dpnp_rng_uniform_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_uniform_ext_c)(DPCTLSyclQueueRef,
                                            void*,
                                            const double,
                                            const double,
                                            const int64_t,
                                            void*,
                                            const DPCTLEventVectorRef) = dpnp_rng_uniform_c<_DataType>;

#ifndef M_PI
/*  128-bits worth of pi */
#define M_PI 3.141592653589793238462643383279502884197
#endif

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_vonmises_large_kappa_c(DPCTLSyclQueueRef q_ref,
                                                  void* result,
                                                  const _DataType mu,
                                                  const _DataType kappa,
                                                  const size_t size,
                                                  const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!size || !result)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    DPNPC_ptr_adapter<_DataType> result1_ptr(q_ref, result, size, true, true);
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

    Uvec = reinterpret_cast<_DataType*>(sycl::malloc_shared(size * sizeof(_DataType), q));
    Vvec = reinterpret_cast<_DataType*>(sycl::malloc_shared(size * sizeof(_DataType), q));

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

    sycl::free(Uvec, q);

    mkl_rng::uniform<_DataType> uniform_distribution(d_zero, d_one);
    auto uniform_distr_event = mkl_rng::generate(uniform_distribution, DPNP_RNG_ENGINE, size, Vvec);

    sycl::range<1> gws(size);

    auto paral_kernel_acceptance = [&](sycl::handler& cgh) {
        cgh.depends_on({uniform_distr_event});
        cgh.parallel_for(gws, [=](sycl::id<1> global_id) {
            size_t i = global_id[0];
            double mod, resi;
            resi = (Vvec[i] < 0.5) ? mu - result1[i] : mu + result1[i];
            mod = sycl::fabs(resi);
            mod = (sycl::fmod(mod + M_PI, 2 * M_PI) - M_PI);
            result1[i] = (resi < 0) ? -mod : mod;
        });
    };
    auto acceptance_event = q.submit(paral_kernel_acceptance);
    acceptance_event.wait();

    sycl::free(Vvec, q);
    return event_ref;
}

template <typename _DataType>
void dpnp_rng_vonmises_large_kappa_c(void* result, const _DataType mu, const _DataType kappa, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_vonmises_large_kappa_c<_DataType>(q_ref,
                                                                             result,
                                                                             mu,
                                                                             kappa,
                                                                             size,
                                                                             dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_vonmises_large_kappa_default_c)(void*,
                                                const _DataType,
                                                const _DataType,
                                                const size_t) = dpnp_rng_vonmises_large_kappa_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_vonmises_large_kappa_ext_c)(
    DPCTLSyclQueueRef,
    void*,
    const _DataType,
    const _DataType,
    const size_t,
    const DPCTLEventVectorRef) = dpnp_rng_vonmises_large_kappa_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_vonmises_small_kappa_c(DPCTLSyclQueueRef q_ref,
                                                  void* result,
                                                  const _DataType mu,
                                                  const _DataType kappa,
                                                  const size_t size,
                                                  const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!size || !result)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    DPNPC_ptr_adapter<_DataType> result1_ptr(q_ref, result, size, true, true);
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

    Uvec = reinterpret_cast<_DataType*>(sycl::malloc_shared(size * sizeof(_DataType), q));
    Vvec = reinterpret_cast<_DataType*>(sycl::malloc_shared(size * sizeof(_DataType), q));

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

    sycl::free(Uvec, q);

    mkl_rng::uniform<_DataType> uniform_distribution(d_zero, d_one);
    auto uniform_distr_event = mkl_rng::generate(uniform_distribution, DPNP_RNG_ENGINE, size, Vvec);

    sycl::range<1> gws(size);
    auto paral_kernel_acceptance = [&](sycl::handler& cgh) {
        cgh.depends_on({uniform_distr_event});
        cgh.parallel_for(gws, [=](sycl::id<1> global_id) {
            size_t i = global_id[0];
            double mod, resi;
            resi = (Vvec[i] < 0.5) ? mu - result1[i] : mu + result1[i];
            mod = sycl::fabs(resi);
            mod = (sycl::fmod(mod + M_PI, 2 * M_PI) - M_PI);
            result1[i] = (resi < 0) ? -mod : mod;
        });
    };
    auto acceptance_event = q.submit(paral_kernel_acceptance);
    acceptance_event.wait();

    sycl::free(Vvec, q);
    return event_ref;
}

template <typename _DataType>
void dpnp_rng_vonmises_small_kappa_c(void* result, const _DataType mu, const _DataType kappa, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_vonmises_small_kappa_c<_DataType>(q_ref,
                                                                             result,
                                                                             mu,
                                                                             kappa,
                                                                             size,
                                                                             dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_vonmises_small_kappa_default_c)(void*,
                                                const _DataType,
                                                const _DataType,
                                                const size_t) = dpnp_rng_vonmises_small_kappa_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_vonmises_small_kappa_ext_c)(
    DPCTLSyclQueueRef,
    void*,
    const _DataType,
    const _DataType,
    const size_t,
    const DPCTLEventVectorRef) = dpnp_rng_vonmises_small_kappa_c<_DataType>;

/* Vonmisses uses the rejection algorithm compared against the wrapped
   Cauchy distribution suggested by Best and Fisher and documented in
   Chapter 9 of Luc's Non-Uniform Random Variate Generation.
   http://cg.scs.carleton.ca/~luc/rnbookindex.html
   (but corrected to match the algorithm in R and Python)
*/
template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_vonmises_c(DPCTLSyclQueueRef q_ref,
                                      void* result,
                                      const _DataType mu,
                                      const _DataType kappa,
                                      const size_t size,
                                      const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    if (kappa > 1.0)
        dpnp_rng_vonmises_large_kappa_c<_DataType>(result, mu, kappa, size);
    else
        dpnp_rng_vonmises_small_kappa_c<_DataType>(result, mu, kappa, size);

    return event_ref;
}

template <typename _DataType>
void dpnp_rng_vonmises_c(void* result, const _DataType mu, const _DataType kappa, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_vonmises_c<_DataType>(q_ref,
                                                                 result,
                                                                 mu,
                                                                 kappa,
                                                                 size,
                                                                 dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_vonmises_default_c)(void*,
                                    const _DataType,
                                    const _DataType,
                                    const size_t) = dpnp_rng_vonmises_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_vonmises_ext_c)(DPCTLSyclQueueRef,
                                             void*,
                                             const _DataType,
                                             const _DataType,
                                             const size_t,
                                             const DPCTLEventVectorRef) = dpnp_rng_vonmises_c<_DataType>;

template <typename _KernelNameSpecialization>
class dpnp_rng_wald_acceptance_kernel1;

template <typename _KernelNameSpecialization>
class dpnp_rng_wald_acceptance_kernel2;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_wald_c(DPCTLSyclQueueRef q_ref,
                                  void* result,
                                  const _DataType mean,
                                  const _DataType scale,
                                  const size_t size,
                                  const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!size)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    _DataType* result1 = reinterpret_cast<_DataType*>(result);
    _DataType* uvec = nullptr;

    const _DataType d_zero = _DataType(0.0);
    const _DataType d_one = _DataType(1.0);
    _DataType gsc = sqrt(0.5 * mean / scale);

    mkl_rng::gaussian<_DataType> gaussian_distribution(d_zero, gsc);

    auto gaussian_dstr_event = mkl_rng::generate(gaussian_distribution, DPNP_RNG_ENGINE, size, result1);

    /* Y = mean/(2 scale) * Z^2 */
    auto sqr_event = mkl_vm::sqr(q, size, result1, result1, {gaussian_dstr_event}, mkl_vm::mode::ha);

    sycl::range<1> gws(size);
    auto acceptance_kernel1 = [=](sycl::id<1> global_id) {
        size_t i = global_id[0];
        if (result1[i] <= 2.0)
        {
            result1[i] = 1.0 + result1[i] + sycl::sqrt(result1[i] * (result1[i] + 2.0));
        }
        else
        {
            result1[i] = 1.0 + result1[i] * (1.0 + sycl::sqrt(1.0 + 2.0 / result1[i]));
        }
    };
    auto parallel_for_acceptance1 = [&](sycl::handler& cgh) {
        cgh.depends_on({sqr_event});
        cgh.parallel_for<class dpnp_rng_wald_acceptance_kernel1<_DataType>>(gws, acceptance_kernel1);
    };
    auto event_ration_acceptance = q.submit(parallel_for_acceptance1);

    uvec = reinterpret_cast<_DataType*>(sycl::malloc_shared(size * sizeof(_DataType), q));

    mkl_rng::uniform<_DataType> uniform_distribution(d_zero, d_one);
    auto uniform_distr_event = mkl_rng::generate(uniform_distribution, DPNP_RNG_ENGINE, size, uvec);

    auto acceptance_kernel2 = [=](sycl::id<1> global_id) {
        size_t i = global_id[0];
        if (uvec[i] * (1.0 + result1[i]) <= result1[i])
            result1[i] = mean / result1[i];
        else
            result1[i] = mean * result1[i];
    };
    auto parallel_for_acceptance2 = [&](sycl::handler& cgh) {
        cgh.depends_on({event_ration_acceptance, uniform_distr_event});
        cgh.parallel_for<class dpnp_rng_wald_acceptance_kernel2<_DataType>>(gws, acceptance_kernel2);
    };
    auto event_out = q.submit(parallel_for_acceptance2);
    event_out.wait();

    sycl::free(uvec, q);

    return event_ref;
}

template <typename _DataType>
void dpnp_rng_wald_c(void* result, const _DataType mean, const _DataType scale, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_wald_c<_DataType>(q_ref,
                                                             result,
                                                             mean,
                                                             scale,
                                                             size,
                                                             dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}


template <typename _DataType>
void (*dpnp_rng_wald_default_c)(void*, const _DataType, const _DataType, const size_t) = dpnp_rng_wald_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_wald_ext_c)(DPCTLSyclQueueRef,
                                         void*,
                                         const _DataType,
                                         const _DataType,
                                         const size_t,
                                         const DPCTLEventVectorRef) = dpnp_rng_wald_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_weibull_c(DPCTLSyclQueueRef q_ref,
                                     void* result,
                                     const double alpha,
                                     const size_t size,
                                     const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;
    sycl::event event_out;

    if (!size)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    if (alpha == 0)
    {
        event_ref = dpnp_zeros_c<_DataType>(q_ref, result, size, dep_event_vec_ref);
    }
    else
    {
        _DataType* result1 = reinterpret_cast<_DataType*>(result);
        const _DataType a = (_DataType(0.0));
        const _DataType beta = (_DataType(1.0));

        mkl_rng::weibull<_DataType> distribution(alpha, a, beta);
        event_out = mkl_rng::generate(distribution, DPNP_RNG_ENGINE, size, result1);
        event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event_out);
    }
    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_rng_weibull_c(void* result, const double alpha, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_weibull_c<_DataType>(q_ref,
                                                                result,
                                                                alpha,
                                                                size,
                                                                dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_weibull_default_c)(void*, const double, const size_t) = dpnp_rng_weibull_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_weibull_ext_c)(DPCTLSyclQueueRef,
                                            void*,
                                            const double,
                                            const size_t,
                                            const DPCTLEventVectorRef) = dpnp_rng_weibull_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_rng_zipf_c(DPCTLSyclQueueRef q_ref,
                                  void* result,
                                  const _DataType a,
                                  const size_t size,
                                  const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!size)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));
    sycl::event event_out;

    size_t i, n_accepted, batch_size;
    _DataType T, U, V, am1, b;
    _DataType *Uvec = nullptr, *Vvec = nullptr;
    long X;
    const _DataType d_zero = 0.0;
    const _DataType d_one = 1.0;

    DPNPC_ptr_adapter<_DataType> result1_ptr(q_ref, result, size, true, true);
    _DataType* result1 = result1_ptr.get_ptr();

    am1 = a - d_one;
    b = pow(2.0, am1);

    Uvec = reinterpret_cast<_DataType*>(sycl::malloc_shared(size * 2 * sizeof(_DataType), q));
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

    sycl::free(Uvec, q);

    return event_ref;
}

template <typename _DataType>
void dpnp_rng_zipf_c(void* result, const _DataType a, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_rng_zipf_c<_DataType>(q_ref,
                                                             result,
                                                             a,
                                                             size,
                                                             dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_rng_zipf_default_c)(void*, const _DataType, const size_t) = dpnp_rng_zipf_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_rng_zipf_ext_c)(DPCTLSyclQueueRef,
                                         void*,
                                         const _DataType,
                                         const size_t,
                                         const DPCTLEventVectorRef) = dpnp_rng_zipf_c<_DataType>;

void func_map_init_random(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_RNG_BETA][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_beta_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_BETA_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_beta_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_BINOMIAL][eft_INT][eft_INT] = {eft_INT,
                                                                  (void*)dpnp_rng_binomial_default_c<int32_t>};

    fmap[DPNPFuncName::DPNP_FN_RNG_BINOMIAL_EXT][eft_INT][eft_INT] = {eft_INT,
                                                                      (void*)dpnp_rng_binomial_ext_c<int32_t>};         

    fmap[DPNPFuncName::DPNP_FN_RNG_CHISQUARE][eft_DBL][eft_DBL] = {eft_DBL,
                                                                   (void*)dpnp_rng_chisquare_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_CHISQUARE_EXT][eft_DBL][eft_DBL] = {eft_DBL,
                                                                       (void*)dpnp_rng_chisquare_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_EXPONENTIAL][eft_DBL][eft_DBL] = {eft_DBL,
                                                                     (void*)dpnp_rng_exponential_default_c<double>};
    fmap[DPNPFuncName::DPNP_FN_RNG_EXPONENTIAL][eft_FLT][eft_FLT] = {eft_FLT,
                                                                     (void*)dpnp_rng_exponential_default_c<float>};

    fmap[DPNPFuncName::DPNP_FN_RNG_EXPONENTIAL_EXT][eft_DBL][eft_DBL] = {eft_DBL,
                                                                         (void*)dpnp_rng_exponential_ext_c<double>};
    fmap[DPNPFuncName::DPNP_FN_RNG_EXPONENTIAL_EXT][eft_FLT][eft_FLT] = {eft_FLT,
                                                                         (void*)dpnp_rng_exponential_ext_c<float>};

    fmap[DPNPFuncName::DPNP_FN_RNG_F][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_f_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_F_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_f_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_GAMMA][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_gamma_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_GAMMA_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_gamma_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_GAUSSIAN][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_gaussian_default_c<double>};
    fmap[DPNPFuncName::DPNP_FN_RNG_GAUSSIAN][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_rng_gaussian_default_c<float>};

    fmap[DPNPFuncName::DPNP_FN_RNG_GAUSSIAN_EXT][eft_DBL][eft_DBL] = {eft_DBL,
                                                                      (void*)dpnp_rng_gaussian_ext_c<double>};
    fmap[DPNPFuncName::DPNP_FN_RNG_GAUSSIAN_EXT][eft_FLT][eft_FLT] = {eft_FLT,
                                                                      (void*)dpnp_rng_gaussian_ext_c<float>};

    fmap[DPNPFuncName::DPNP_FN_RNG_GEOMETRIC][eft_INT][eft_INT] = {eft_INT,
                                                                   (void*)dpnp_rng_geometric_default_c<int32_t>};

    fmap[DPNPFuncName::DPNP_FN_RNG_GEOMETRIC_EXT][eft_INT][eft_INT] = {eft_INT,
                                                                   (void*)dpnp_rng_geometric_ext_c<int32_t>};

    fmap[DPNPFuncName::DPNP_FN_RNG_GUMBEL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_gumbel_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_GUMBEL_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_gumbel_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_HYPERGEOMETRIC][eft_INT][eft_INT] = {
        eft_INT, (void*)dpnp_rng_hypergeometric_default_c<int32_t>};

    fmap[DPNPFuncName::DPNP_FN_RNG_HYPERGEOMETRIC_EXT][eft_INT][eft_INT] = {
        eft_INT, (void*)dpnp_rng_hypergeometric_ext_c<int32_t>};

    fmap[DPNPFuncName::DPNP_FN_RNG_LAPLACE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_laplace_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_LAPLACE_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_laplace_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_LOGISTIC][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_logistic_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_LOGISTIC_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_logistic_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_LOGNORMAL][eft_DBL][eft_DBL] = {eft_DBL,
                                                                   (void*)dpnp_rng_lognormal_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_LOGNORMAL_EXT][eft_DBL][eft_DBL] = {eft_DBL,
                                                                       (void*)dpnp_rng_lognormal_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_MULTINOMIAL][eft_INT][eft_INT] = {eft_INT,
                                                                     (void*)dpnp_rng_multinomial_default_c<int32_t>};

    fmap[DPNPFuncName::DPNP_FN_RNG_MULTINOMIAL_EXT][eft_INT][eft_INT] = {eft_INT,
                                                                         (void*)dpnp_rng_multinomial_ext_c<int32_t>};

    fmap[DPNPFuncName::DPNP_FN_RNG_MULTIVARIATE_NORMAL][eft_DBL][eft_DBL] = {
        eft_DBL, (void*)dpnp_rng_multivariate_normal_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_NEGATIVE_BINOMIAL][eft_INT][eft_INT] = {
        eft_INT, (void*)dpnp_rng_negative_binomial_default_c<int32_t>};

    fmap[DPNPFuncName::DPNP_FN_RNG_NEGATIVE_BINOMIAL_EXT][eft_INT][eft_INT] = {
        eft_INT, (void*)dpnp_rng_negative_binomial_ext_c<int32_t>};

    fmap[DPNPFuncName::DPNP_FN_RNG_NONCENTRAL_CHISQUARE][eft_DBL][eft_DBL] = {
        eft_DBL, (void*)dpnp_rng_noncentral_chisquare_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_NONCENTRAL_CHISQUARE_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void*)dpnp_rng_noncentral_chisquare_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_NORMAL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_normal_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_NORMAL_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_normal_ext_c<double>};
    fmap[DPNPFuncName::DPNP_FN_RNG_NORMAL_EXT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_rng_normal_ext_c<float>};

    fmap[DPNPFuncName::DPNP_FN_RNG_PARETO][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_pareto_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_PARETO_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_pareto_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_POISSON][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_rng_poisson_default_c<int32_t>};

    fmap[DPNPFuncName::DPNP_FN_RNG_POISSON_EXT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_rng_poisson_ext_c<int32_t>};

    fmap[DPNPFuncName::DPNP_FN_RNG_POWER][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_power_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_POWER_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_power_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_RAYLEIGH][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_rayleigh_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_RAYLEIGH_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_rayleigh_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_SHUFFLE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_shuffle_default_c<double>};
    fmap[DPNPFuncName::DPNP_FN_RNG_SHUFFLE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_rng_shuffle_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_RNG_SHUFFLE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_rng_shuffle_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_RNG_SHUFFLE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_rng_shuffle_default_c<int64_t>};

    fmap[DPNPFuncName::DPNP_FN_RNG_SHUFFLE_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_shuffle_ext_c<double>};
    fmap[DPNPFuncName::DPNP_FN_RNG_SHUFFLE_EXT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_rng_shuffle_ext_c<float>};
    fmap[DPNPFuncName::DPNP_FN_RNG_SHUFFLE_EXT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_rng_shuffle_ext_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_RNG_SHUFFLE_EXT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_rng_shuffle_ext_c<int64_t>};

    fmap[DPNPFuncName::DPNP_FN_RNG_SRAND][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_srand_c};

    fmap[DPNPFuncName::DPNP_FN_RNG_STANDARD_CAUCHY][eft_DBL][eft_DBL] = {
        eft_DBL, (void*)dpnp_rng_standard_cauchy_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_STANDARD_CAUCHY_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void*)dpnp_rng_standard_cauchy_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_STANDARD_EXPONENTIAL][eft_DBL][eft_DBL] = {
        eft_DBL, (void*)dpnp_rng_standard_exponential_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_STANDARD_EXPONENTIAL_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void*)dpnp_rng_standard_exponential_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_STANDARD_GAMMA][eft_DBL][eft_DBL] = {
        eft_DBL, (void*)dpnp_rng_standard_gamma_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_STANDARD_GAMMA_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void*)dpnp_rng_standard_gamma_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_STANDARD_NORMAL][eft_DBL][eft_DBL] = {
        eft_DBL, (void*)dpnp_rng_standard_normal_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_STANDARD_T][eft_DBL][eft_DBL] = {
        eft_DBL, (void*)dpnp_rng_standard_t_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_STANDARD_T_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void*)dpnp_rng_standard_t_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_TRIANGULAR][eft_DBL][eft_DBL] = {eft_DBL,
                                                                    (void*)dpnp_rng_triangular_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_TRIANGULAR_EXT][eft_DBL][eft_DBL] = {eft_DBL,
                                                                    (void*)dpnp_rng_triangular_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_UNIFORM][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_uniform_default_c<double>};
    fmap[DPNPFuncName::DPNP_FN_RNG_UNIFORM][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_rng_uniform_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_RNG_UNIFORM][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_rng_uniform_default_c<int32_t>};

    fmap[DPNPFuncName::DPNP_FN_RNG_UNIFORM_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_uniform_ext_c<double>};
    fmap[DPNPFuncName::DPNP_FN_RNG_UNIFORM_EXT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_rng_uniform_ext_c<float>};
    fmap[DPNPFuncName::DPNP_FN_RNG_UNIFORM_EXT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_rng_uniform_ext_c<int32_t>};

    fmap[DPNPFuncName::DPNP_FN_RNG_VONMISES][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_vonmises_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_VONMISES_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_vonmises_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_WALD][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_wald_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_WALD_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_wald_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_WEIBULL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_weibull_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_WEIBULL_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_weibull_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_ZIPF][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_zipf_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RNG_ZIPF_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_rng_zipf_ext_c<double>};

    return;
}
