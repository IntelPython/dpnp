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

#include <dpnp_iface.hpp>
#include "queue_sycl.hpp"

#if defined(DPNP_LOCAL_QUEUE)
cl::sycl::queue* backend_sycl::queue = nullptr;
#endif
mkl_rng::mt19937* backend_sycl::rng_engine = nullptr;

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

    dpnp_square_c<long>(value_ptr, result_ptr, 1);

    result = *result_ptr;

    dpnp_memory_free_c(result_ptr);
    dpnp_memory_free_c(value_ptr);

    return result;
}

#if defined(DPNP_LOCAL_QUEUE)
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
            std::cout << "DPNP. Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
        }
    }
};
#endif

void backend_sycl::backend_sycl_queue_init(QueueOptions selector)
{
#if defined(DPNP_LOCAL_QUEUE)
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
#else
    (void) selector;
#endif

    std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
    dpnp_custom_kernels_link();
    std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_kernels_link =
        std::chrono::duration_cast<std::chrono::duration<double>>(t4 - t3);

    std::cout << "Running on: " << DPNP_QUEUE.get_device().get_info<sycl::info::device::name>() << "\n";
#if defined(DPNP_LOCAL_QUEUE)
    std::cout << "queue initialization time: " << time_queue_init.count() << " (sec.)\n";
#else
    std::cout << "DPCtrl SYCL queue used\n";
#endif
    std::cout << "SYCL kernels link time: " << time_kernels_link.count() << " (sec.)\n" << std::endl;
}

bool backend_sycl::backend_sycl_is_cpu()
{
    cl::sycl::queue& qptr = get_queue();

    if (qptr.is_host() || qptr.get_device().is_cpu() || qptr.get_device().is_host())
    {
        return true;
    }

    return false;
}

void backend_sycl::backend_sycl_rng_engine_init(size_t seed)
{
    if (rng_engine)
    {
        backend_sycl::destroy_rng_engine();
    }
    rng_engine = new mkl_rng::mt19937(DPNP_QUEUE, seed);
}

void dpnp_queue_initialize_c(QueueOptions selector)
{
    backend_sycl::backend_sycl_queue_init(selector);
}

size_t dpnp_queue_is_cpu_c()
{
    return backend_sycl::backend_sycl_is_cpu();
}

void dpnp_srand_c(size_t seed)
{
    backend_sycl::backend_sycl_rng_engine_init(seed);
}
