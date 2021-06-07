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

void black_scholes(double* price,
                   double* strike,
                   double* t,
                   double* mrs,
                   double* vol_vol_twos,
                   double* quarters,
                   double* ones,
                   double* halfs,
                   double* call,
                   double* put,
                   const size_t size)
{
    double* P = price;
    double* S = strike;
    double* T = t;

    double* PdivS = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_divide_c<double, double, double>(P, S, PdivS, size); // P / S
    double* a = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_log_c<double, double>(PdivS, a, size); // np.log(P / S)
    dpnp_memory_free_c(PdivS);

    double* b = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_multiply_c<double, double, double>(T, mrs, b, size); // T * mrs

    double* z = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_multiply_c<double, double, double>(T, vol_vol_twos, z, size); // T * vol_vol_twos

    double* c = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_multiply_c<double, double, double>(quarters, z, c, size); // quarters * z

    double* sqrt_z = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_sqrt_c<double, double>(z, sqrt_z, size); // np.sqrt(z)
    dpnp_memory_free_c(z);
    double* y = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_divide_c<double, double, double>(ones, sqrt_z, y, size); // ones/np.sqrt(z)
    dpnp_memory_free_c(sqrt_z);

    double* a_sub_b = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_subtract_c<double, double, double>(a, b, a_sub_b, size); // a - b
    dpnp_memory_free_c(a);
    double* a_sub_b_add_c = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_add_c<double, double, double>(a_sub_b, c, a_sub_b_add_c, size); // a - b + c
    double* w1 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_multiply_c<double, double, double>(a_sub_b_add_c, y, w1, size); // (a - b + c) * y
    dpnp_memory_free_c(a_sub_b_add_c);

    double* a_sub_b_sub_c = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_subtract_c<double, double, double>(a_sub_b, c, a_sub_b_sub_c, size); // a - b - c
    dpnp_memory_free_c(a_sub_b);
    dpnp_memory_free_c(c);
    double* w2 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_multiply_c<double, double, double>(a_sub_b_sub_c, y, w2, size); // (a - b - c) * y
    dpnp_memory_free_c(a_sub_b_sub_c);
    dpnp_memory_free_c(y);

    double* erf_w1 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_erf_c<double>(w1, erf_w1, size); // np.erf(w1)
    dpnp_memory_free_c(w1);
    double* halfs_mul_erf_w1 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_multiply_c<double, double, double>(halfs, erf_w1, halfs_mul_erf_w1, size); // halfs * np.erf(w1)
    dpnp_memory_free_c(erf_w1);
    double* d1 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_add_c<double, double, double>(halfs, halfs_mul_erf_w1, d1, size); // halfs + halfs * np.erf(w1)
    dpnp_memory_free_c(halfs_mul_erf_w1);

    double* erf_w2 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_erf_c<double>(w2, erf_w2, size); // np.erf(w2)
    dpnp_memory_free_c(w2);
    double* halfs_mul_erf_w2 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_multiply_c<double, double, double>(halfs, erf_w2, halfs_mul_erf_w2, size); // halfs * np.erf(w2)
    dpnp_memory_free_c(erf_w2);
    double* d2 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_add_c<double, double, double>(halfs, halfs_mul_erf_w2, d2, size); // halfs + halfs * np.erf(w2)
    dpnp_memory_free_c(halfs_mul_erf_w2);

    double* exp_b = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_exp_c<double, double>(b, exp_b, size); // np.exp(b)
    double* Se = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_multiply_c<double, double, double>(exp_b, S, Se, size); // np.exp(b) * S
    dpnp_memory_free_c(exp_b);
    dpnp_memory_free_c(b);

    double* Pmul_d1 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_multiply_c<double, double, double>(P, d1, Pmul_d1, size); // P * d1
    dpnp_memory_free_c(d1);
    double* Se_mul_d2 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_multiply_c<double, double, double>(Se, d2, Se_mul_d2, size); // Se * d2
    dpnp_memory_free_c(d2);
    double* r = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_subtract_c<double, double, double>(Pmul_d1, Se_mul_d2, r, size); // P * d1 - Se * d2
    dpnp_memory_free_c(Se_mul_d2);
    dpnp_memory_free_c(Pmul_d1);

    dpnp_copyto_c<double, double>(call, r, size); // call[:] = r
    double* r_subP = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_subtract_c<double, double, double>(r, P, r_subP, size); // r - P
    dpnp_memory_free_c(r);
    double* r_subPaddSe = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    dpnp_add_c<double, double, double>(r_subP, Se, r_subPaddSe, size); // r - P + Se
    dpnp_memory_free_c(r_subP);
    dpnp_memory_free_c(Se);
    dpnp_copyto_c<double, double>(put, r_subPaddSe, size); // put[:] = r - P + Se
    dpnp_memory_free_c(r_subPaddSe);
}

int main(int, char**)
{
    const size_t SIZE = 256;

    const size_t SEED = 7777777;
    const long PL = 10, PH = 50;
    const long SL = 10, SH = 50;
    const long TL = 1, TH = 2;
    const double RISK_FREE = 0.1;
    const double VOLATILITY = 0.2;

    dpnp_queue_initialize_c(QueueOptions::GPU_SELECTOR);
    std::cout << "SYCL queue is CPU: " << dpnp_queue_is_cpu_c() << std::endl;

    double* price = (double*)dpnp_memory_alloc_c(SIZE * sizeof(double));
    double* strike = (double*)dpnp_memory_alloc_c(SIZE * sizeof(double));
    double* t = (double*)dpnp_memory_alloc_c(SIZE * sizeof(double));

    dpnp_rng_srand_c(SEED);                           // np.random.seed(SEED)
    dpnp_rng_uniform_c<double>(price, PL, PH, SIZE);  // np.random.uniform(PL, PH, SIZE)
    dpnp_rng_uniform_c<double>(strike, SL, SH, SIZE); // np.random.uniform(SL, SH, SIZE)
    dpnp_rng_uniform_c<double>(t, TL, TH, SIZE);      // np.random.uniform(TL, TH, SIZE)

    double* zero = (double*)dpnp_memory_alloc_c(1 * sizeof(double));
    zero[0] = 0.;

    double* mone = (double*)dpnp_memory_alloc_c(1 * sizeof(double));
    mone[0] = -1.;

    double* mr = (double*)dpnp_memory_alloc_c(1 * sizeof(double));
    mr[0] = -RISK_FREE;

    double* vol_vol_two = (double*)dpnp_memory_alloc_c(1 * sizeof(double));
    vol_vol_two[0] = VOLATILITY * VOLATILITY * 2.;

    double* quarter = (double*)dpnp_memory_alloc_c(1 * sizeof(double));
    quarter[0] = 0.25;

    double* one = (double*)dpnp_memory_alloc_c(1 * sizeof(double));
    one[0] = 1.;

    double* half = (double*)dpnp_memory_alloc_c(1 * sizeof(double));
    half[0] = 0.5;

    double* call = (double*)dpnp_memory_alloc_c(SIZE * sizeof(double));
    double* put = (double*)dpnp_memory_alloc_c(SIZE * sizeof(double));
    double* mrs = (double*)dpnp_memory_alloc_c(SIZE * sizeof(double));
    double* vol_vol_twos = (double*)dpnp_memory_alloc_c(SIZE * sizeof(double));
    double* quarters = (double*)dpnp_memory_alloc_c(SIZE * sizeof(double));
    double* ones = (double*)dpnp_memory_alloc_c(SIZE * sizeof(double));
    double* halfs = (double*)dpnp_memory_alloc_c(SIZE * sizeof(double));

    dpnp_full_c<double>(zero, call, SIZE);                // np.full(SIZE, 0., dtype=DTYPE)
    dpnp_full_c<double>(mone, put, SIZE);                 // np.full(SIZE, -1., dtype=DTYPE)
    dpnp_full_c<double>(mr, mrs, SIZE);                   // np.full(SIZE, -RISK_FREE, dtype=DTYPE)
    dpnp_full_c<double>(vol_vol_two, vol_vol_twos, SIZE); // np.full(SIZE, VOLATILITY * VOLATILITY * 2., dtype=DTYPE)
    dpnp_full_c<double>(quarter, quarters, SIZE);         // np.full(SIZE, 0.25, dtype=DTYPE)
    dpnp_full_c<double>(one, ones, SIZE);                 // np.full(SIZE, 1., dtype=DTYPE)
    dpnp_full_c<double>(half, halfs, SIZE);               // np.full(SIZE, 0.5, dtype=DTYPE)

    dpnp_memory_free_c(half);
    dpnp_memory_free_c(one);
    dpnp_memory_free_c(quarter);
    dpnp_memory_free_c(vol_vol_two);
    dpnp_memory_free_c(mr);
    dpnp_memory_free_c(mone);
    dpnp_memory_free_c(zero);

    black_scholes(price, strike, t, mrs, vol_vol_twos, quarters, ones, halfs, call, put, SIZE);

    std::cout << std::endl;
    for (size_t i = 0; i < 10; ++i)
    {
        std::cout << call[i] << ", ";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < 10; ++i)
    {
        std::cout << put[i] << ", ";
    }
    std::cout << std::endl;

    dpnp_memory_free_c(halfs);
    dpnp_memory_free_c(ones);
    dpnp_memory_free_c(quarters);
    dpnp_memory_free_c(vol_vol_twos);
    dpnp_memory_free_c(mrs);
    dpnp_memory_free_c(put);
    dpnp_memory_free_c(call);

    dpnp_memory_free_c(t);
    dpnp_memory_free_c(strike);
    dpnp_memory_free_c(price);

    return 0;
}
