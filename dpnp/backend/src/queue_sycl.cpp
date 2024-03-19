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

#include <chrono>
#include <exception>
#include <iostream>

#include "dpnp_iface.hpp"
#include "dpnp_utils.hpp"
#include "queue_sycl.hpp"

[[maybe_unused]] static void dpnpc_show_mathlib_version()
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

#if (not defined(NDEBUG))
[[maybe_unused]] static void show_available_sycl_devices()
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

size_t dpnp_queue_is_cpu_c()
{
    const auto &be = backend_sycl::get();
    return be.backend_sycl_is_cpu();
}
