//*****************************************************************************
// Copyright (c) 2016-2024, Intel Corporation
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

/*
 * This header file is for interface Cython with C++.
 * It should not contains any backend specific headers (like SYCL or math
 * library) because all included headers will be exposed in Cython compilation
 * procedure
 *
 * We would like to avoid backend specific things in higher level Cython
 * modules. Any backend interface functions and types should be defined here.
 *
 * Also, this file should contains documentation on functions and types
 * which are used in the interface
 */

#pragma once
#ifndef BACKEND_IFACE_RANDOM_H // Cython compatibility
#define BACKEND_IFACE_RANDOM_H

/**
 * @defgroup BACKEND_RANDOM_API Backend C++ library interface RANDOM API
 * @{
 * This section describes Backend API for RANDOM NUMBER GENERATION (RNG) part.
 * @}
 */

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (beta
 * distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  a                   Alpha, shape param.
 * @param [in]  b                   Beta, scalefactor.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_beta_c(DPCTLSyclQueueRef q_ref,
                    void *result,
                    const _DataType a,
                    const _DataType b,
                    const size_t size,
                    const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_beta_c(void *result,
                                   const _DataType a,
                                   const _DataType b,
                                   const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (binomial
 * distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  ntrial              Number of independent trials.
 * @param [in]  p                   Success probability p of a single trial.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_binomial_c(DPCTLSyclQueueRef q_ref,
                        void *result,
                        const int ntrial,
                        const double p,
                        const size_t size,
                        const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_binomial_c(void *result,
                                       const int ntrial,
                                       const double p,
                                       const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (chi-square
 * distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  df                  Degrees of freedom.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_chisquare_c(DPCTLSyclQueueRef q_ref,
                         void *result,
                         const int df,
                         const size_t size,
                         const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void
    dpnp_rng_chisquare_c(void *result, const int df, const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (exponential
 * distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  beta                Beta, scalefactor.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_exponential_c(DPCTLSyclQueueRef q_ref,
                           void *result,
                           const _DataType beta,
                           const size_t size,
                           const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_exponential_c(void *result,
                                          const _DataType beta,
                                          const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (F
 * distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  df_num              Degrees of freedom in numerator.
 * @param [in]  df_den              Degrees of freedom in denominator.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_f_c(DPCTLSyclQueueRef q_ref,
                 void *result,
                 const _DataType df_num,
                 const _DataType df_den,
                 const size_t size,
                 const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_f_c(void *result,
                                const _DataType df_num,
                                const _DataType df_den,
                                const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (gamma
 * distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  shape               The shape of the gamma distribution.
 * @param [in]  scale               The scale of the gamma distribution.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_gamma_c(DPCTLSyclQueueRef q_ref,
                     void *result,
                     const _DataType shape,
                     const _DataType scale,
                     const size_t size,
                     const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_gamma_c(void *result,
                                    const _DataType shape,
                                    const _DataType scale,
                                    const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (gaussian
 * continuous distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  mean                Mean value.
 * @param [in]  stddev              Standard deviation.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_gaussian_c(DPCTLSyclQueueRef q_ref,
                        void *result,
                        const _DataType mean,
                        const _DataType stddev,
                        const size_t size,
                        const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_gaussian_c(void *result,
                                       const _DataType mean,
                                       const _DataType stddev,
                                       const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (Geometric
 * distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  p                   The probability of success of an individual
 * trial.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_geometric_c(DPCTLSyclQueueRef q_ref,
                         void *result,
                         const float p,
                         const size_t size,
                         const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void
    dpnp_rng_geometric_c(void *result, const float p, const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (Gumbel
 * distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  loc                 The location of the mode of the
 * distribution.
 * @param [in]  scale               The scale parameter of the distribution.
 * Default is 1.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_gumbel_c(DPCTLSyclQueueRef q_ref,
                      void *result,
                      const double loc,
                      const double scale,
                      const size_t size,
                      const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_gumbel_c(void *result,
                                     const double loc,
                                     const double scale,
                                     const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (hypergeometric
 * distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  l                   Lot size of l.
 * @param [in]  s                   Size of sampling without replacement.
 * @param [in]  m                   Number of marked elements m.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_hypergeometric_c(DPCTLSyclQueueRef q_ref,
                              void *result,
                              const int l,
                              const int s,
                              const int m,
                              const size_t size,
                              const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_hypergeometric_c(void *result,
                                             const int l,
                                             const int s,
                                             const int m,
                                             const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (laplace
 * distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  loc                 The position of the distribution peak.
 * @param [in]  scale               The exponential decay.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_laplace_c(DPCTLSyclQueueRef q_ref,
                       void *result,
                       const double loc,
                       const double scale,
                       const size_t size,
                       const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_laplace_c(void *result,
                                      const double loc,
                                      const double scale,
                                      const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (logistic
 * distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  loc                 The position of the distribution peak.
 * @param [in]  scale               The exponential decay.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_logistic_c(DPCTLSyclQueueRef q_ref,
                        void *result,
                        const double loc,
                        double const scale,
                        const size_t size,
                        const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_logistic_c(void *result,
                                       const double loc,
                                       double const scale,
                                       const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (lognormal
 * distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  mean                Mean value.
 * @param [in]  stddev              Standard deviation.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_lognormal_c(DPCTLSyclQueueRef q_ref,
                         void *result,
                         const _DataType mean,
                         const _DataType stddev,
                         const size_t size,
                         const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_lognormal_c(void *result,
                                        const _DataType mean,
                                        const _DataType stddev,
                                        const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (multinomial
 * distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  ntrial              Number of independent trials.
 * @param [in]  p_in                Probability of possible outcomes (k length).
 * @param [in]  p_size              Length of `p_in`.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_multinomial_c(DPCTLSyclQueueRef q_ref,
                           void *result,
                           const int ntrial,
                           const double *p_in,
                           const size_t p_size,
                           const size_t size,
                           const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_multinomial_c(void *result,
                                          const int ntrial,
                                          const double *p_in,
                                          const size_t p_size,
                                          const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (multivariate
 * normal distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  dimen               Dimension of output random vectors.
 * @param [in]  mean_in             Mean array of dimension.
 * @param [in]  mean_size           Length of `mean_in`.
 * @param [in]  cov                 Variance-covariance matrix.
 * @param [in]  cov_size            Length of `cov_in`.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_multivariate_normal_c(DPCTLSyclQueueRef q_ref,
                                   void *result,
                                   const int dimen,
                                   const double *mean_in,
                                   const size_t mean_size,
                                   const double *cov_in,
                                   const size_t cov_size,
                                   const size_t size,
                                   const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_multivariate_normal_c(void *result,
                                                  const int dimen,
                                                  const double *mean_in,
                                                  const size_t mean_size,
                                                  const double *cov_in,
                                                  const size_t cov_size,
                                                  const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (negative
 * binomial distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  a                   The first distribution parameter a, > 0.
 * @param [in]  p                   The second distribution parameter p, >= 0
 * and <=1.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 *
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_negative_binomial_c(DPCTLSyclQueueRef q_ref,
                                 void *result,
                                 const double a,
                                 const double p,
                                 const size_t size,
                                 const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_negative_binomial_c(void *result,
                                                const double a,
                                                const double p,
                                                const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (noncentral
 * chisquare distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  df                  Degrees of freedom.
 * @param [in]  nonc                Non-centrality param.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef dpnp_rng_noncentral_chisquare_c(
    DPCTLSyclQueueRef q_ref,
    void *result,
    const _DataType df,
    const _DataType nonc,
    const size_t size,
    const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_noncentral_chisquare_c(void *result,
                                                   const _DataType df,
                                                   const _DataType nonc,
                                                   const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (normal
 * continuous distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result_out          Output array.
 * @param [in]  mean                Mean value.
 * @param [in]  stddev              Standard deviation.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  random_state_in     Pointer on random state.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_normal_c(DPCTLSyclQueueRef q_ref,
                      void *result_out,
                      const double mean,
                      const double stddev,
                      const int64_t size,
                      void *random_state_in,
                      const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_normal_c(void *result,
                                     const _DataType mean,
                                     const _DataType stddev,
                                     const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (Pareto
 * distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  alpha               Shape of the distribution, alpha.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_pareto_c(DPCTLSyclQueueRef q_ref,
                      void *result,
                      const double alpha,
                      const size_t size,
                      const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void
    dpnp_rng_pareto_c(void *result, const double alpha, const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (poisson
 * distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  lambda              Distribution parameter lambda.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_poisson_c(DPCTLSyclQueueRef q_ref,
                       void *result,
                       const double lambda,
                       const size_t size,
                       const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void
    dpnp_rng_poisson_c(void *result, const double lambda, const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (power
 * distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  alpha               Shape of the distribution, alpha.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_power_c(DPCTLSyclQueueRef q_ref,
                     void *result,
                     const double alpha,
                     const size_t size,
                     const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void
    dpnp_rng_power_c(void *result, const double alpha, const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (rayleigh
 * distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  scale               Distribution parameter, scalefactor.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_rayleigh_c(DPCTLSyclQueueRef q_ref,
                        void *result,
                        const _DataType scale,
                        const size_t size,
                        const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void
    dpnp_rng_rayleigh_c(void *result, const _DataType scale, const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random in-place shuffle.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Input/output array.
 * @param [in]  itemsize            Length of `result` array element in bytes.
 * @param [in]  ndim                Number of array dimensions in `result`
 * arrays.
 * @param [in]  high_dim_size       Number of elements in `result` arrays higher
 * dimension, or len(result).
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_shuffle_c(DPCTLSyclQueueRef q_ref,
                       void *result,
                       const size_t itemsize,
                       const size_t ndim,
                       const size_t high_dim_size,
                       const size_t size,
                       const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_shuffle_c(void *result,
                                      const size_t itemsize,
                                      const size_t ndim,
                                      const size_t high_dim_size,
                                      const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief initializer for basic random number generator.
 *
 * @param [in]  seed    The seed value.
 */
INP_DLLEXPORT void dpnp_rng_srand_c(size_t seed = 1);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (standard
 * cauchy distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_standard_cauchy_c(DPCTLSyclQueueRef q_ref,
                               void *result,
                               const size_t size,
                               const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_standard_cauchy_c(void *result, const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (standard
 * exponential distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef dpnp_rng_standard_exponential_c(
    DPCTLSyclQueueRef q_ref,
    void *result,
    const size_t size,
    const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_standard_exponential_c(void *result,
                                                   const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (standard gamma
 * distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  shape               Shape value.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_standard_gamma_c(DPCTLSyclQueueRef q_ref,
                              void *result,
                              const _DataType shape,
                              const size_t size,
                              const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_standard_gamma_c(void *result,
                                             const _DataType shape,
                                             const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (standard
 * normal distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result_out          Output array.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_standard_normal_c(void *result, const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (standard
 * Student's t distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  df                  Degrees of freedom.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_standard_t_c(DPCTLSyclQueueRef q_ref,
                          void *result,
                          const _DataType df,
                          const size_t size,
                          const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void
    dpnp_rng_standard_t_c(void *result, const _DataType df, const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (Triangular
 * distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  x_min               Lower limit.
 * @param [in]  x_mode              The value where the peak of the distribution
 * occurs.
 * @param [in]  x_max               Upper limit.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_triangular_c(DPCTLSyclQueueRef q_ref,
                          void *result,
                          const _DataType x_min,
                          const _DataType x_mode,
                          const _DataType x_max,
                          const size_t size,
                          const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_triangular_c(void *result,
                                         const _DataType x_min,
                                         const _DataType x_mode,
                                         const _DataType x_max,
                                         const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (uniform
 * distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result_out          Output array.
 * @param [in]  low                 Left bound of array values.
 * @param [in]  high                Right bound of array values.
 * @param [in]  size                Number of elements in `result` array.
 * @param [in]  random_state_in     Pointer on random state.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_uniform_c(DPCTLSyclQueueRef q_ref,
                       void *result_out,
                       const double low,
                       const double high,
                       const int64_t size,
                       void *random_state_in,
                       const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_uniform_c(void *result,
                                      const long low,
                                      const long high,
                                      const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (Vonmises
 * distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  mu                  Mode of the distribution.
 * @param [in]  kappa               Dispersion of the distribution.
 * @param [in]  size                Number of elements in `result` array.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_vonmises_c(DPCTLSyclQueueRef q_ref,
                        void *result,
                        const _DataType mu,
                        const _DataType kappa,
                        const size_t size,
                        const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_vonmises_c(void *result,
                                       const _DataType mu,
                                       const _DataType kappa,
                                       const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (Wald's
 * distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  mean                Mean value.
 * @param [in]  scale               The scale of the distribution.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_wald_c(DPCTLSyclQueueRef q_ref,
                    void *result,
                    const _DataType mean,
                    const _DataType scale,
                    size_t size,
                    const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_wald_c(void *result,
                                   const _DataType mean,
                                   const _DataType scale,
                                   size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (weibull
 * distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  alpha               Shape parameter of the distribution, alpha.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_weibull_c(DPCTLSyclQueueRef q_ref,
                       void *result,
                       const double alpha,
                       const size_t size,
                       const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void
    dpnp_rng_weibull_c(void *result, const double alpha, const size_t size);

/**
 * @ingroup BACKEND_RANDOM_API
 * @brief math library implementation of random number generator (Zipf
 * distribution)
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  a                   Distribution parameter.
 * @param [in]  size                Number of elements in `result` arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_rng_zipf_c(DPCTLSyclQueueRef q_ref,
                    void *result,
                    const _DataType a,
                    const size_t size,
                    const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void
    dpnp_rng_zipf_c(void *result, const _DataType a, const size_t size);

#endif // BACKEND_IFACE_RANDOM_H
