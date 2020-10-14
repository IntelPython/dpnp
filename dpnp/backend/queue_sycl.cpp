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

#include <backend_iface.hpp>
#include "queue_sycl.hpp"

#if defined(DPNP_LOCAL_QUEUE)
cl::sycl::queue* backend_sycl::queue = nullptr;
#endif

EngineOptions backend_sycl::engine_type = EngineOptions::MT19937;
bool backend_sycl::reseed = false;
size_t backend_sycl::reseed_val;
std::array<void*, ENGINE_OPTIONS_NUMBER> backend_sycl::engines;

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
            std::cout << "Intel NumPy. Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
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
        backend_sycl::destroy_queue();
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

void backend_sycl::backend_sycl_rng_engine_init()
{
    size_t seed;

    if (engines[engine_type])
    {
        backend_sycl::destroy_rng_engine();
    }

    if(reseed)
    {
        seed = reseed_val;
        reseed = false;
    }
    else
    {
        // TODO:
        // choose engine as is in numpy
        // seed number
        // TODO:
        // mem leak
        seed = std::time(nullptr);
    }
    switch(engine_type)
    {
        case EngineOptions::ARS5:
        {
            engines[engine_type] = new mkl_rng::ars5(DPNP_QUEUE, seed);
            break;
        }
        case EngineOptions::MCG32M1:
        {
            engines[engine_type] = new mkl_rng::mcg31m1(DPNP_QUEUE, seed);
            break;
        }
        case EngineOptions::MCG59:
        {
            engines[engine_type] = new mkl_rng::mcg59(DPNP_QUEUE, seed);
            break;
        }
        case EngineOptions::MRG32K3A:
        {
            engines[engine_type] = new mkl_rng::mrg32k3a(DPNP_QUEUE, seed);
            break;
        }
        case EngineOptions::MT19937:
        {
            engines[engine_type] = new mkl_rng::mt19937(DPNP_QUEUE, seed);
            break;
        }
        case EngineOptions::MT2203:
        {
            engines[engine_type] = new mkl_rng::mt2203(DPNP_QUEUE, seed);
            break;
        }
        case EngineOptions::NIEDERREITER:
        {
            engines[engine_type] = new mkl_rng::niederreiter(DPNP_QUEUE, seed);
            break;
        }
        case EngineOptions::NONDETERMINISTIC:
        {
            engines[engine_type] = new mkl_rng::nondeterministic(DPNP_QUEUE);
            break;
        }
        case EngineOptions::PHILOX4X32X10:
        {
            engines[engine_type] = new mkl_rng::philox4x32x10(DPNP_QUEUE, seed);
            break;
        }
        case EngineOptions::R250:
        {
            engines[engine_type] = new mkl_rng::r250(DPNP_QUEUE, seed);
            break;
        }
        case EngineOptions::SFMT19937:
        {
            engines[engine_type] = new mkl_rng::sfmt19937(DPNP_QUEUE, seed);
            break;
        }
        case EngineOptions::SOBOL:
        {
            engines[engine_type] = new mkl_rng::sobol(DPNP_QUEUE, seed);
            break;
        }
        case EngineOptions::WICHMANN_HILL:
        {
            engines[engine_type] = new mkl_rng::wichmann_hill(DPNP_QUEUE, seed);
            break;
        }
    }
}

void dpnp_queue_initialize_c(QueueOptions selector)
{
    backend_sycl::backend_sycl_queue_init(selector);
}

void dpnp_set_rng(EngineOptions engine_type)
{
    backend_sycl::set_engine_type(engine_type);
}

void dpnp_srand()
{
    backend_sycl::backend_sycl_rng_engine_init();
}

void dpnp_srand(size_t seed)
{
    backend_sycl::set_seed(seed);
    backend_sycl::backend_sycl_rng_engine_init();
}
