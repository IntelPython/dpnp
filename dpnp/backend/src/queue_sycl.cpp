//*****************************************************************************
// Copyright (c) 2016-2025, Intel Corporation
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
[[maybe_unused]] static std::string
    device_type_to_str(sycl::info::device_type devTy)
{
    std::stringstream ss;
    switch (devTy) {
    case sycl::info::device_type::cpu:
        ss << "cpu";
        break;
    case sycl::info::device_type::gpu:
        ss << "gpu";
        break;
    case sycl::info::device_type::accelerator:
        ss << "accelerator";
        break;
    case sycl::info::device_type::custom:
        ss << "custom";
        break;
    default:
        ss << "unknown";
    }
    return ss.str();
}

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
            << device_type_to_str(
                   it->get_info<sycl::info::device::device_type>())
            << ", gws="
            << it->get_info<sycl::info::device::max_work_group_size>()
            << ", cu=" << it->get_info<sycl::info::device::max_compute_units>()
            << ", name=" << it->get_info<sycl::info::device::name>()
            << std::endl;
    }
}
#endif

size_t dpnp_queue_is_cpu_c()
{
    const auto &be = backend_sycl::get();
    return be.backend_sycl_is_cpu();
}
