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

/**
 * Example 11.
 *
 * This example shows simple usage of the DPNP C++ Backend library RNG shuffle
 * function for one and ndim arrays.
 *
 * Possible compile line:
 * g++ -g dpnp/backend/examples/example11.cpp -Idpnp -Idpnp/backend/include
 * -Ldpnp -Wl,-rpath='$ORIGIN'/dpnp -ldpnp_backend_c -o example11
 *
 */

#include <iostream>

#include <dpnp_iface.hpp>

template <typename T>
void print_dpnp_array(T *arr, size_t size)
{
    std::cout << std::endl;
    for (size_t i = 0; i < size; ++i) {
        std::cout << arr[i] << ", ";
    }
    std::cout << std::endl;
}

int main(int, char **)
{
    // Two cases:
    // 1) array size = 100, ndim = 1, high_dim_size = 10 (aka ndarray with shape
    // (100,) ) 2) array size = 100, ndim = 2, high_dim_size = 20 (e.g. ndarray
    // with shape (20, 5) and len(array) = 20 )
    const size_t ndim_cases = 2;
    const size_t itemsize = sizeof(double);
    const size_t ndim[ndim_cases] = {1, 2};
    const size_t high_dim_size[ndim_cases] = {100, 20};
    const size_t size = 100;
    const size_t seed = 1234;

    // DPNPC dpnp_rng_shuffle_c
    // DPNPC interface
    double *array_1 =
        reinterpret_cast<double *>(dpnp_memory_alloc_c(size * sizeof(double)));
    for (size_t i = 0; i < ndim_cases; i++) {
        std::cout << "\nREPRODUCE: DPNPC dpnp_rng_shuffle_c:";
        std::cout << "\nDIMS: " << ndim[i] << std::endl;
        // init array 0, 1, 2, 3, 4, 5, 6, ....
        dpnp_arange_c<double>(0, 1, array_1, size);
        // print before shuffle
        std::cout << "\nINPUT array:";
        print_dpnp_array(array_1, size);
        dpnp_rng_srand_c(seed);
        dpnp_rng_shuffle_c<double>(array_1, itemsize, ndim[i], high_dim_size[i],
                                   size);
        // print shuffle result
        std::cout << "\nSHUFFLE INPUT array:";
        print_dpnp_array(array_1, size);
    }
    dpnp_memory_free_c(array_1);
}
