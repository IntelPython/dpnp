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

#include <chrono>
#include <exception>
#include <iostream>

#include <backend/backend_iface.hpp>
#include "queue_sycl.hpp"

cl::sycl::queue* backend_sycl::queue = nullptr;

/**
 * Function push the SYCL kernels to be linked (final stage of the compilation) for the current queue
 *
 * TODO it is not the best idea to just a call some kernel. Needs better solution.
 */
static long dpnp_custom_kernels_link()
{
    /* must use memory pre-allocated at the current queue */
    long* value_ptr = reinterpret_cast<long*>(dpnp_memory_alloc_c(1 * sizeof(long)));
    long* result_ptr = reinterpret_cast<long*>(dpnp_memory_alloc_c(1 * sizeof(long)));
    long result = 1;

    *value_ptr = 2;

    custom_elemwise_square_c<long>(value_ptr, result_ptr, 1);

    result = *result_ptr;

    dpnp_memory_free_c(result_ptr);
    dpnp_memory_free_c(value_ptr);

    return result;
}

// Catch asynchronous exceptions
static void exception_handler(cl::sycl::exception_list exceptions)
{
    for (std::exception_ptr const& e : exceptions)
    {
        try
        {
            std::rethrow_exception(e);
        }
        catch (cl::sycl::exception const& e)
        {
            std::cout << "Intel NumPy. Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
        }
    }
};

void backend_sycl::backend_sycl_queue_init(QueueOptions selector)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    if (queue)
    {
        backend_sycl::destroy();
    }

    cl::sycl::device dev;

    if (QueueOptions::CPU_SELECTOR == selector)
    {
        dev = cl::sycl::device(cl::sycl::cpu_selector());
    }
    else
    {
        dev = cl::sycl::device(cl::sycl::gpu_selector());
    }

    queue = new cl::sycl::queue(dev, exception_handler);

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_queue_init = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    dpnp_custom_kernels_link();
    std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_kernels_link =
        std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);

    std::cout << "Running on: " << DPNP_QUEUE.get_device().get_info<sycl::info::device::name>() << "\n"
              << "queue initialization time: " << time_queue_init.count() << " (sec.)\n"
              << "SYCL kernels link time: " << time_kernels_link.count() << " (sec.)\n"
              << std::endl;
}

void dpnp_queue_initialize_c(QueueOptions selector)
{
    backend_sycl::backend_sycl_queue_init(selector);
}

#include <cstring>
/**
 * Experimental interface. DO NOT USE IT!
 */
void* get_backend_function_name(const char* func_name, const char* type_name)
{
    /** Implement it in this way to allow easier play with it */
    const char* supported_func_name = "dpnp_dot";
    const char* supported_type1_name = "double";
    const char* supported_type2_name = "float";
    const char* supported_type3_name = "long";
    const char* supported_type4_name = "int";

    /** of coerce it will be converted into std::map later */
    if (!strncmp(func_name, supported_func_name, strlen(supported_func_name)))
    {
        if (!strncmp(type_name, supported_type1_name, strlen(supported_type1_name)))
        {
            return reinterpret_cast<void*>(mkl_blas_dot_c<double>);
        }
        else if (!strncmp(type_name, supported_type2_name, strlen(supported_type2_name)))
        {
            return reinterpret_cast<void*>(mkl_blas_dot_c<float>);
        }
        else if (!strncmp(type_name, supported_type3_name, strlen(supported_type3_name)))
        {
            return reinterpret_cast<void*>(custom_blas_dot_c<long>);
        }
        else if (!strncmp(type_name, supported_type4_name, strlen(supported_type4_name)))
        {
            return reinterpret_cast<void*>(custom_blas_dot_c<int>);
        }
    }

    throw std::runtime_error("Intel NumPy Error: Unsupported function call");
}
