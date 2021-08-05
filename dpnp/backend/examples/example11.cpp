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
 * Example 10.
 *
 * Possible compile line:
 * clang++ -fsycl dpnp/backend/examples/example10.cpp -Idpnp -Idpnp/backend/include -Ldpnp -Wl,-rpath='$ORIGIN'/dpnp
 * -ldpnp_backend_c -o example10 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_core
 */

#include <iostream>
#include <time.h>

#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>

#include <dpnp_iface.hpp>

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
        x[2 * i + 0] = static_cast<T>(i);
        x[2 * i + 1] = 0;
    }
}

// TODO:
void test_dpnp_fft(const size_t size, const size_t iters);

void test_mkl_fft(const size_t size, const size_t iters)
{
    cl::sycl::event event;

    clock_t start, end;

    double dev_time_used = 0.0;
    double sum_dev_time_used = 0.0;
    const size_t result_size = size * 2;

    cl::sycl::queue queue{cl::sycl::gpu_selector()};

    double* array_1 = reinterpret_cast<double*>(malloc_shared(result_size * sizeof(double), queue));

    init<double>(array_1, result_size);

    double* result = reinterpret_cast<double*>(malloc_shared(result_size * sizeof(double), queue));

    for (size_t i = 0; i < iters; ++i)
    {
        start = clock();

        oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX> desc(
            result_size);

        desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0 / result_size));

        // enum value DFTI_NOT_INPLACE from math library C interface
        // instead of mkl_dft::config_value::NOT_INPLACE
        desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
        desc.commit(queue);
        event = oneapi::mkl::dft::compute_forward(desc, array_1, result);
        event.wait();

        end = clock();

        sum_dev_time_used += ((double)(end - start)) / CLOCKS_PER_SEC;
        // TODO: cumulative addition error
        // div error
    }

    std::cout << "array_1: ";
    print_dpnp_array(array_1, 10);
    std::cout << std::endl;
    std::cout << "res: ";
    print_dpnp_array(result, 10);

    free(result, queue);
    free(array_1, queue);

    std::cout << "\nMKL time: ";
    dev_time_used = sum_dev_time_used / iters;
    std::cout << dev_time_used << std::endl;

    return;
}

int main(int, char**)
{
    const size_t size = 1000;
    const size_t iters = 30;

    // TODO:
    // test_dpnp_fft(size, iters);

    test_mkl_fft(size, iters);

    return 0;
}
