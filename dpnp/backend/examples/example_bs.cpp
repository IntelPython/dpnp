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

/**
 * Example BS.
 *
 * This example shows simple usage of the DPNP C++ Backend library
 * to calculate black scholes algorithm like in Python version
 *
 * Possible compile line:
 * clang++ -g -fPIC dpnp/backend/examples/example_bs.cpp -Idpnp -Idpnp/backend/include -Ldpnp -Wl,-rpath='$ORIGIN'/dpnp -ldpnp_backend_c -o example_bs
 *
 */

#include <cmath>
#include <iostream>

#include "dpnp_iface.hpp"

double* black_scholes_put(double* S,
                          double* K,
                          double* T,
                          double* sigmas,
                          double* r_sigma_sigma_2,
                          double* nrs,
                          double* sqrt2,
                          double* ones,
                          double* twos,
                          const size_t size)
{
    double* d1 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_divide_c<double, double, double>(S, K, d1, size); // S/K
    dpnp_log_c<double, double>(d1, d1, size);              // np.log(S/K)

    double* bs_put = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_multiply_c<double, double, double>(r_sigma_sigma_2, T, bs_put, size); // r_sigma_sigma_2*T
    dpnp_add_c<double, double, double>(d1, bs_put, d1, size);                  // np.log(S/K) + r_sigma_sigma_2*T

    dpnp_sqrt_c<double, double>(T, bs_put, size);                          // np.sqrt(T)
    dpnp_multiply_c<double, double, double>(sigmas, bs_put, bs_put, size); // sigmas*np.sqrt(T)

    // (np.log(S/K) + r_sigma_sigma_2*T) / (sigmas*np.sqrt(T))
    dpnp_divide_c<double, double, double>(d1, bs_put, d1, size);

    double* d2 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_sqrt_c<double, double>(T, bs_put, size);                          // np.sqrt(T)
    dpnp_multiply_c<double, double, double>(sigmas, bs_put, bs_put, size); // sigmas*np.sqrt(T)
    dpnp_subtract_c<double, double, double>(d1, bs_put, d2, size);         // d1 - sigmas*np.sqrt(T)

    double* cdf_d1 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_divide_c<double, double, double>(d1, sqrt2, cdf_d1, size); // d1 / sqrt2
    dpnp_erf_c<double>(cdf_d1, cdf_d1, size);                       // np.erf(d1 / sqrt2)
    dpnp_add_c<double, double, double>(ones, cdf_d1, cdf_d1, size); // ones + np.erf(d1 / sqrt2)
    dpnp_add_c<double, double, double>(ones, cdf_d1, cdf_d1, size); // (ones + np.erf(d1 / sqrt2)) / twos
    dpnp_memory_free_c(d1);

    double* cdf_d2 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_divide_c<double, double, double>(d2, sqrt2, cdf_d2, size); // d2 / sqrt2
    dpnp_erf_c<double>(cdf_d2, cdf_d2, size);                       // np.erf(d2 / sqrt2)
    dpnp_add_c<double, double, double>(ones, cdf_d2, cdf_d2, size); // ones + np.erf(d2 / sqrt2)
    dpnp_add_c<double, double, double>(ones, cdf_d2, cdf_d2, size); // (ones + np.erf(d2 / sqrt2)) / twos
    dpnp_memory_free_c(d2);

    double* bs_call = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_multiply_c<double, double, double>(S, cdf_d1, bs_call, size); // S*cdf_d1
    dpnp_memory_free_c(cdf_d1);

    dpnp_multiply_c<double, double, double>(nrs, T, bs_put, size);    // nrs*T
    dpnp_exp_c<double, double>(bs_put, bs_put, size);                 // np.exp(nrs*T)
    dpnp_multiply_c<double, double, double>(K, bs_put, bs_put, size); // K*np.exp(nrs*T)

    // K*np.exp(nrs*T)*cdf_d2
    dpnp_multiply_c<double, double, double>(bs_put, cdf_d2, bs_put, size);
    dpnp_memory_free_c(cdf_d2);

    // S*cdf_d1 - K*np.exp(nrs*T)*cdf_d2
    dpnp_subtract_c<double, double, double>(bs_call, bs_put, bs_call, size);

    dpnp_multiply_c<double, double, double>(nrs, T, bs_put, size);     // nrs*T
    dpnp_exp_c<double, double>(bs_put, bs_put, size);                  // np.exp(nrs*T)
    dpnp_multiply_c<double, double, double>(K, bs_put, bs_put, size);  // K*np.exp(nrs*T)
    dpnp_subtract_c<double, double, double>(bs_put, S, bs_put, size);  // K*np.exp(nrs*T) - S
    dpnp_add_c<double, double, double>(bs_put, bs_call, bs_put, size); // K*np.exp(nrs*T) - S + bs_call
    dpnp_memory_free_c(bs_call);

    return bs_put;
}

int main(int, char**)
{
    const size_t SIZE = 256;

    const size_t SEED = 7777777;
    const long SL = 10, SH = 50;
    const long KL = 10, KH = 50;
    const long TL = 1, TH = 2;
    const double RISK_FREE = 0.1;
    const double VOLATILITY = 0.2;

    dpnp_queue_initialize_c(QueueOptions::GPU_SELECTOR);
    std::cout << "SYCL queue is CPU: " << dpnp_queue_is_cpu_c() << std::endl;

    double* S = (double*)dpnp_memory_alloc_c(SIZE * sizeof(double));
    double* K = (double*)dpnp_memory_alloc_c(SIZE * sizeof(double));
    double* T = (double*)dpnp_memory_alloc_c(SIZE * sizeof(double));

    dpnp_rng_srand_c(SEED);
    dpnp_rng_uniform_c<double>(S, SL, SH, SIZE); // np.random.uniform(SL, SH, SIZE)
    dpnp_rng_uniform_c<double>(K, KL, KH, SIZE); // np.random.uniform(KL, KH, SIZE)
    dpnp_rng_uniform_c<double>(T, TL, TH, SIZE); // np.random.uniform(TL, TH, SIZE)

    double* r = (double*)dpnp_memory_alloc_c(1 * sizeof(double));
    r[0] = RISK_FREE;

    double* sigma = (double*)dpnp_memory_alloc_c(1 * sizeof(double));
    sigma[0] = VOLATILITY;

    double* rss2 = (double*)dpnp_memory_alloc_c(1 * sizeof(double));
    rss2[0] = RISK_FREE + VOLATILITY * VOLATILITY / 2.;

    double* nr = (double*)dpnp_memory_alloc_c(1 * sizeof(double));
    nr[0] = -RISK_FREE;

    double* sq2 = (double*)dpnp_memory_alloc_c(1 * sizeof(double));
    sq2[0] = sqrt(2.);

    double* one = (double*)dpnp_memory_alloc_c(1 * sizeof(double));
    one[0] = 1.;

    double* two = (double*)dpnp_memory_alloc_c(1 * sizeof(double));
    two[0] = 2.;

    double* sigmas = (double*)dpnp_memory_alloc_c(SIZE * sizeof(double));
    double* r_sigma_sigma_2 = (double*)dpnp_memory_alloc_c(SIZE * sizeof(double));
    double* nrs = (double*)dpnp_memory_alloc_c(SIZE * sizeof(double));
    double* sqrt2 = (double*)dpnp_memory_alloc_c(SIZE * sizeof(double));
    double* ones = (double*)dpnp_memory_alloc_c(SIZE * sizeof(double));
    double* twos = (double*)dpnp_memory_alloc_c(SIZE * sizeof(double));

    dpnp_full_c<double>(sigma, sigmas, SIZE);         // np.full((SIZE,), sigma, dtype=DTYPE)
    dpnp_full_c<double>(rss2, r_sigma_sigma_2, SIZE); // np.full((SIZE,), r + sigma*sigma/2., dtype=DTYPE)
    dpnp_full_c<double>(nr, nrs, SIZE);               // np.full((SIZE,), -r, dtype=DTYPE)
    dpnp_full_c<double>(sq2, sqrt2, SIZE);            // np.full((SIZE,), np.sqrt(2), dtype=DTYPE)
    dpnp_full_c<double>(one, ones, SIZE);             // np.full((SIZE,), 1, dtype=DTYPE)
    dpnp_full_c<double>(two, twos, SIZE);             // np.full((SIZE,), 2, dtype=DTYPE)

    dpnp_memory_free_c(one);
    dpnp_memory_free_c(two);
    dpnp_memory_free_c(sq2);
    dpnp_memory_free_c(nr);
    dpnp_memory_free_c(rss2);
    dpnp_memory_free_c(sigma);
    dpnp_memory_free_c(r);

    double* bs_put = black_scholes_put(S, K, T, sigmas, r_sigma_sigma_2, nrs, sqrt2, ones, twos, SIZE);

    std::cout << std::endl;
    for (size_t i = 0; i < 10; ++i)
    {
        std::cout << bs_put[i] << ", ";
    }
    std::cout << std::endl;

    dpnp_memory_free_c(bs_put);

    dpnp_memory_free_c(twos);
    dpnp_memory_free_c(ones);
    dpnp_memory_free_c(sqrt2);
    dpnp_memory_free_c(nrs);
    dpnp_memory_free_c(r_sigma_sigma_2);
    dpnp_memory_free_c(sigmas);

    dpnp_memory_free_c(T);
    dpnp_memory_free_c(K);
    dpnp_memory_free_c(S);

    return 0;
}
