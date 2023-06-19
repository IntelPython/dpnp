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
 * Example 7.
 *
 * This example shows simple usage of the DPNP C++ Backend library
 * to calculate eigenvalues and eigenvectors of a symmetric matrix
 *
 * Possible compile line:
 * . /opt/intel/oneapi/setvars.sh
 * g++ -g dpnp/backend/examples/example7.cpp -Idpnp -Idpnp/backend/include -Ldpnp -Wl,-rpath='$ORIGIN'/dpnp -ldpnp_backend_c -o example7
 *
 */

#include <iostream>

#include "dpnp_iface.hpp"

int main(int, char**)
{
    const size_t size = 2;
    size_t len = size * size;

    dpnp_queue_initialize_c(QueueOptions::CPU_SELECTOR);

    float* array = (float*)dpnp_memory_alloc_c(len * sizeof(float));
    float* result1 = (float*)dpnp_memory_alloc_c(size * sizeof(float));
    float* result2 = (float*)dpnp_memory_alloc_c(len * sizeof(float));

    /* init input diagonal array like:
    1, 0, 0,
    0, 2, 0,
    0, 0, 3
    */
    for (size_t i = 0; i < len; ++i)
    {
        array[i] = 0;
    }
    for (size_t i = 0; i < size; ++i)
    {
        array[size * i + i] = i + 1;
    }

    dpnp_eig_c<float, float>(array, result1, result2, size);

    std::cout << "eigen values" << std::endl;
    for (size_t i = 0; i < size; ++i)
    {
        std::cout << result1[i] << ", ";
    }
    std::cout << std::endl;

    dpnp_memory_free_c(result2);
    dpnp_memory_free_c(result1);
    dpnp_memory_free_c(array);

    return 0;
}
