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
 * Example 9.
 *
 * TODO explanation of the example
 *
 * Possible compile line:
 * . /opt/intel/oneapi/setvars.sh
 * g++ -g dpnp/backend/examples/example8.cpp -Idpnp -Idpnp/backend/include
 * -Ldpnp -Wl,-rpath='$ORIGIN'/dpnp -ldpnp_backend_c -o example8
 *
 */
#include <iostream>

#include "dpnp_iface.hpp"

int main(int, char **)
{
    const size_t size = 16;

    dpnp_queue_initialize_c(QueueOptions::GPU_SELECTOR);

    double *array = (double *)dpnp_memory_alloc_c(size * sizeof(double));
    long *result = (long *)dpnp_memory_alloc_c(size * sizeof(long));

    std::cout << "array" << std::endl;
    for (size_t i = 0; i < size; ++i) {
        array[i] = (double)(size - i) / 2;
        std::cout << array[i] << ", ";
    }
    std::cout << std::endl;

    dpnp_argsort_c<double, long>(array, result, size);

    std::cout << "array with 'sorted' indeces" << std::endl;
    for (size_t i = 0; i < size; ++i) {
        std::cout << result[i] << ", ";
    }
    std::cout << std::endl;

    dpnp_memory_free_c(result);
    dpnp_memory_free_c(array);

    return 0;
}
