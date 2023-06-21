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

#include <chrono>
#include <exception>
#include <iostream>

#include "dpnp_iface.hpp"
#include "dpnp_utils.hpp"
#include "queue_sycl.hpp"

#if defined(DPNP_LOCAL_QUEUE)
sycl::queue *backend_sycl::queue = nullptr;
#endif
mkl_rng::mt19937 *backend_sycl::rng_engine     = nullptr;
mkl_rng::mcg59 *backend_sycl::rng_mcg59_engine = nullptr;

static void dpnpc_show_mathlib_version()
{
#if 1
    const int len = 256;
    std::string mathlib_version_str(len, 0x0);

    char *buf = const_cast<char *>(
        mathlib_version_str.c_str()); // TODO avoid write into the container

    mkl_get_version_string(buf, len);

    std::cout << "Math backend version: " << mathlib_version_str << std::endl;
#else
    // Failed to load library under Python environment die to unresolved symbol
    MKLVersion version;

    mkl_get_version(&version);

    std::cout << "Math backend version: " << version.MajorVersion << "."
              << version.UpdateVersion << "." << version.MinorVersion
              << std::endl;
#endif
}

#if (not defined(NDEBUG)) && defined(DPNP_LOCAL_QUEUE)
static void show_available_sycl_devices()
{
    const std::vector<sycl::device> devices = sycl::device::get_devices();

    std::cout << "Available SYCL devices:" << std::endl;
    for (std::vector<sycl::device>::const_iterator it = devices.cbegin();
         it != devices.cend(); ++it)
    {
        std::cout
            // not yet implemented error << " " <<
            // it->has(sycl::aspect::usm_shared_allocations)  << " "
            << " - id=" << it->get_info<sycl::info::device::vendor_id>()
            << ", type="
            << static_cast<pi_uint64>(
                   it->get_info<sycl::info::device::device_type>())
            << ", gws="
            << it->get_info<sycl::info::device::max_work_group_size>()
            << ", cu=" << it->get_info<sycl::info::device::max_compute_units>()
            << ", name=" << it->get_info<sycl::info::device::name>()
            << std::endl;
    }
}
#endif

#if defined(DPNP_LOCAL_QUEUE)
static sycl::device get_default_sycl_device()
{
    int dpnpc_queue_gpu = 0;
    sycl::device dev    = sycl::device(sycl::cpu_selector());

    const char *dpnpc_queue_gpu_var = getenv("DPNPC_QUEUE_GPU");
    if (dpnpc_queue_gpu_var != NULL) {
        dpnpc_queue_gpu = atoi(dpnpc_queue_gpu_var);
    }

    if (dpnpc_queue_gpu) {
        dev = sycl::device(sycl::gpu_selector());
    }

    return dev;
}
#endif

#if defined(DPNPC_TOUCH_KERNEL_TO_LINK)
/**
 * Function push the SYCL kernels to be linked (final stage of the compilation)
 * for the current queue
 *
 * TODO it is not the best idea to just a call some kernel. Needs better
 * solution.
 */
static long dpnp_kernels_link()
{
    /* must use memory pre-allocated at the current queue */
    long *value_ptr =
        reinterpret_cast<long *>(dpnp_memory_alloc_c(1 * sizeof(long)));
    long *result_ptr =
        reinterpret_cast<long *>(dpnp_memory_alloc_c(1 * sizeof(long)));
    long result = 1;

    *value_ptr = 2;

    dpnp_square_c<long>(value_ptr, result_ptr, 1);

    result = *result_ptr;

    dpnp_memory_free_c(result_ptr);
    dpnp_memory_free_c(value_ptr);

    return result;
}
#endif

#if defined(DPNP_LOCAL_QUEUE)
// Catch asynchronous exceptions
static void exception_handler(sycl::exception_list exceptions)
{
    for (std::exception_ptr const &e : exceptions) {
        try {
            std::rethrow_exception(e);
        } catch (sycl::exception const &e) {
            std::cout << "DPNP. Caught asynchronous SYCL exception:\n"
                      << e.what() << std::endl;
        }
    }
};
#endif

void backend_sycl::backend_sycl_queue_init(QueueOptions selector)
{
#if defined(DPNP_LOCAL_QUEUE)
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();

    if (queue) {
        backend_sycl::destroy();
    }

    sycl::device dev;

#if not defined(NDEBUG)
    show_available_sycl_devices();
#endif

    if (QueueOptions::CPU_SELECTOR == selector) {
        dev = sycl::device(sycl::cpu_selector());
    }
    else if (QueueOptions::GPU_SELECTOR == selector) {
        dev = sycl::device(sycl::gpu_selector());
    }
    else {
        dev = get_default_sycl_device();
    }

    if (is_verbose_mode()) {
        sycl::property_list properties{
            sycl::property::queue::enable_profiling()};
        queue = new sycl::queue(dev, exception_handler, properties);
    }
    else {
        queue = new sycl::queue(dev, exception_handler);
    }

    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_queue_init =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
#else
    (void)selector;
#endif

    std::chrono::high_resolution_clock::time_point t3 =
        std::chrono::high_resolution_clock::now();
#if defined(DPNPC_TOUCH_KERNEL_TO_LINK)
    // Remove pre-link kernel library at startup time
    dpnp_kernels_link();
#endif
    std::chrono::high_resolution_clock::time_point t4 =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_kernels_link =
        std::chrono::duration_cast<std::chrono::duration<double>>(t4 - t3);

    std::cout << "Running on: "
              << DPNP_QUEUE.get_device().get_info<sycl::info::device::name>()
              << "\n";
#if defined(DPNP_LOCAL_QUEUE)
    std::cout << "queue initialization time: " << time_queue_init.count()
              << " (sec.)\n";
#else
    std::cout << "DPCtrl SYCL queue used\n";
#endif
    std::cout << "SYCL kernels link time: " << time_kernels_link.count()
              << " (sec.)\n";
    dpnpc_show_mathlib_version();

    std::cout << std::endl;
}

bool backend_sycl::backend_sycl_is_cpu()
{
    sycl::queue &qptr = get_queue();

    if (qptr.get_device().is_cpu()) {
        return true;
    }

    return false;
}

void backend_sycl::backend_sycl_rng_engine_init(size_t seed)
{
    if (rng_engine) {
        backend_sycl::destroy_rng_engine();
    }
    rng_engine       = new mkl_rng::mt19937(DPNP_QUEUE, seed);
    rng_mcg59_engine = new mkl_rng::mcg59(DPNP_QUEUE, seed);
}

void dpnp_queue_initialize_c(QueueOptions selector)
{
    backend_sycl::backend_sycl_queue_init(selector);
}

size_t dpnp_queue_is_cpu_c()
{
    return backend_sycl::backend_sycl_is_cpu();
}
