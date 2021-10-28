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
 * Example 11.
 *
 * Possible compile line:
 * dpcpp -g -fPIC   -W -Wextra -Wshadow -Wall -Wstrict-prototypes -Wformat -Wformat-security -fsycl -fsycl-device-code-split=per_kernel -O0 -fno-delete-null-pointer-checks -fstack-protector-strong -fno-strict-overflow -std=gnu++17 dpnp/backend/examples/example11.cpp -Idpnp -Idpnp/backend/include -Ldpnp -Wl,-rpath='$ORIGIN'/dpnp -ldpnp_backend_c -o example11 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_core
 */
#include <iostream>
#include <time.h>

#include <dpnp_iface.hpp>
#include <dpnp_iface_fptr.hpp>

#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>

namespace mkl_dft = oneapi::mkl::dft;
namespace mkl_rng = oneapi::mkl::rng;
namespace mkl_vm = oneapi::mkl::vm;

template <typename T>
void print_dpnp_array(T* arr, size_t size)
{
    std::cout << std::endl;
    for (size_t i = 0; i < size; ++i)
    {
        std::cout << arr[i] << ", ";
    }
    std::cout << std::endl;
}

template <typename T>
void init(T* x, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        x[i] = static_cast<T>(i);
    }
}

#if 0
/* For 1dim array */
template <typename _DataType>
void shuffle_c(
    void* result, const size_t itemsize, const size_t ndim, const size_t high_dim_size, const size_t size, cl::sycl::queue& queue, size_t seed)
{
    if (!result)
    {
        return;
    }

    if (!size || !ndim || !(high_dim_size > 1))
    {
        return;
    }

    // cl::sycl::queue queue{cl::sycl::gpu_selector()};

    mkl_rng::mt19937 rng_engine = mkl_rng::mt19937(queue, seed);

    // DPNPC_ptr_adapter<char> result1_ptr(result, size * itemsize, true, true);
    // char* result1 = result1_ptr.get_ptr();
    char* result1 = reinterpret_cast<char*>(result);

    size_t uvec_size = high_dim_size - 1;
    // double* array_1 = reinterpret_cast<double*>(malloc_shared(result_size * sizeof(double), queue));
    // double* Uvec = reinterpret_cast<double*>(dpnp_memory_alloc_c(uvec_size * sizeof(double)));

    double* Uvec = reinterpret_cast<double*>(malloc_shared(uvec_size * sizeof(double), queue));

    mkl_rng::uniform<double> uniform_distribution(0.0, 1.0);

    auto uniform_event = mkl_rng::generate(uniform_distribution, rng_engine, uvec_size, Uvec);
    uniform_event.wait();
    // just for debug
    // print_dpnp_array(Uvec, uvec_size);


    // Fast, statically typed path: shuffle the underlying buffer.
    // Only for non-empty, 1d objects of class ndarray (subclasses such
    // as MaskedArrays may not support this approach).
    char* buf = reinterpret_cast<char*>(malloc_shared(sizeof(char), queue));
    for (size_t i = uvec_size; i > 0; i--)
    {
        size_t j = (size_t)(floor((i + 1) * Uvec[i - 1]));
        if (i != j)
        {
            auto memcpy1 =
                queue.submit([&](cl::sycl::handler& h) { h.memcpy(buf, result1 + j * itemsize, itemsize); });
            auto memcpy2 = queue.submit([&](cl::sycl::handler& h) {
                h.depends_on({memcpy1});
                h.memcpy(result1 + j * itemsize, result1 + i * itemsize, itemsize);
            });
            auto memcpy3 = queue.submit([&](cl::sycl::handler& h) {
                h.depends_on({memcpy2});
                h.memcpy(result1 + i * itemsize, buf, itemsize);
            });
            memcpy3.wait();
        }
    }
    free(buf, queue);
    free(Uvec, queue);
    return;
}
#endif

int main(int, char**)
{
    // Two cases:
    // 1) array size = 100, ndim = 1, high_dim_size = 10 (aka ndarray with shape (100,) )
    // 2) array size = 100, ndim = 2, high_dim_size = 20 (aka ndarray with shape (20, 5) and len(array) = 20 )
    const size_t ndim_cases = 2;
    const size_t itemsize = sizeof(double);
    const size_t ndim[ndim_cases] = {1, 2};
    const size_t high_dim_size[ndim_cases] = {100, 20};
    const size_t size = 100;
    const size_t seed = 1234;

#if 0
    /*
      reproducer for shuffling array (as is in dpnp_rng_shuffle_c)
      for 1dim array
      the same logic for ndim arrays
      using mathlib interface
    */
    std::cout << "\n///////////////////////////////////////////////\n";
    cl::sycl::queue queue{cl::sycl::gpu_selector()};
    std::cout << "\nREPRODUCE: logic as is in DPNPC dpnp_rng_shuffle_c:";
    std::cout << "\nDIMS: " << ndim[0] <<std::endl;
    double* array_1 = reinterpret_cast<double*>(malloc_shared(size * sizeof(double), queue));
    std::cout << "\nINPUT array 1:";
    init<double>(array_1, size);
    print_dpnp_array(array_1, size);
    shuffle_c<double>(array_1, itemsize, ndim[0], high_dim_size[0], size, queue, seed);
    std::cout << "\nSHUFFLE INPUT array:";
    print_dpnp_array(array_1, size);
    free(array_1, queue);
#endif

    std::cout << "\n///////////////////////////////////////////////\n";
    // DPNPC dpnp_rng_shuffle_c
    // DPNPC interface
    double* array_2 = reinterpret_cast<double*>(dpnp_memory_alloc_c(size * sizeof(double)));
    for (size_t i = 0; i < ndim_cases; i++)
    {
        std::cout << "\nREPRODUCE: DPNPC dpnp_rng_shuffle_c:";
        std::cout << "\nDIMS: " << ndim[i] <<std::endl;
        // init array 0, 1, 2, 3, 4, 5, 6, ....
        init<double>(array_2, size);
        // print before shuffle
        std::cout << "\nINPUT array 2:";
        print_dpnp_array(array_2, size);
        dpnp_rng_srand_c(seed);
        dpnp_rng_shuffle_c<double>(array_2, itemsize, ndim[i], high_dim_size[i], size);
        // print shuffle result
        std::cout << "\nSHUFFLE INPUT array 2:";
        print_dpnp_array(array_2, size); 
    }
    dpnp_memory_free_c(array_2);
}
