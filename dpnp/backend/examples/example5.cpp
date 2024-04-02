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

/**
 * Example 5.
 *
 * This example shows simple usage of the DPNP C++ Backend RNG library
 *
 * Possible compile line:
 * . /opt/intel/oneapi/setvars.sh
 * g++ -g dpnp/backend/examples/example5.cpp -Idpnp -Idpnp/backend/include
 * -Ldpnp -Wl,-rpath='$ORIGIN'/dpnp -ldpnp_backend_c -o example5
 *
 */

#include <iostream>

#include <dpnp_iface.hpp>

void print_dpnp_array(double *arr, size_t size)
{
    std::cout << std::endl;
    for (size_t i = 0; i < size; ++i) {
        std::cout << arr[i] << ", ";
    }
    std::cout << std::endl;
}

int main(int, char **)
{
    const size_t size = 256;

    double *result = (double *)dpnp_memory_alloc_c(size * sizeof(double));

    size_t seed = 10;
    long low = 1;
    long high = 120;

    std::cout << "Uniform distr. params:\nlow is " << low << ", high is "
              << high << std::endl;

    std::cout << "Results, when seed is the same (10) for all random number "
                 "generations:";
    for (size_t i = 0; i < 4; ++i) {
        dpnp_rng_srand_c(seed);
        dpnp_rng_uniform_c<double>(result, low, high, size);
        print_dpnp_array(result, 10);
    }

    std::cout << std::endl << "Results, when seed is random:";
    dpnp_rng_srand_c();
    for (size_t i = 0; i < 4; ++i) {
        dpnp_rng_uniform_c<double>(result, low, high, size);
        print_dpnp_array(result, 10);
    }

    dpnp_memory_free_c(result);

    return 0;
}
