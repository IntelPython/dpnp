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
 * Example 3.
 *
 * This example shows simple usage of the DPNP C++ Backend library
 * to calculate cos of input vector elements
 *
 * Possible compile line:
 * . /opt/intel/oneapi/setvars.sh
 * g++ -g dpnp/backend/examples/example3.cpp -Idpnp -Idpnp/backend/include
 * -Ldpnp -Wl,-rpath='$ORIGIN'/dpnp -ldpnp_backend_c -o example3
 *
 */

#include <iostream>

#include "dpnp_iface.hpp"

int main(int, char **)
{
    const size_t size = 256;

    dpnp_queue_initialize_c();
    std::cout << "SYCL queue is CPU: " << dpnp_queue_is_cpu_c() << std::endl;

    int *array1 = (int *)dpnp_memory_alloc_c(size * sizeof(int));
    double *result = (double *)dpnp_memory_alloc_c(size * sizeof(double));

    for (size_t i = 0; i < 10; ++i) {
        array1[i] = i;
        result[i] = 0;
        std::cout << ", " << array1[i];
    }
    std::cout << std::endl;

    const long ndim = 1;
    shape_elem_type *shape = reinterpret_cast<shape_elem_type *>(
        dpnp_memory_alloc_c(ndim * sizeof(shape_elem_type)));
    shape[0] = size;
    shape_elem_type *strides = reinterpret_cast<shape_elem_type *>(
        dpnp_memory_alloc_c(ndim * sizeof(shape_elem_type)));
    strides[0] = 1;

    dpnp_cos_c<int, double>(result, size, ndim, shape, strides, array1, size,
                            ndim, shape, strides, NULL);

    for (size_t i = 0; i < 10; ++i) {
        std::cout << ", " << result[i];
    }
    std::cout << std::endl;

    dpnp_memory_free_c(result);
    dpnp_memory_free_c(array1);

    return 0;
}
