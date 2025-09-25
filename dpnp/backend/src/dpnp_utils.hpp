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

#pragma once
#ifndef BACKEND_UTILS_H // Cython compatibility
#define BACKEND_UTILS_H

#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>

#include <dpnp_iface_fptr.hpp>

/**
 * Version of SYCL DPC++ 2023 compiler where a return type of sycl::abs() is
 * changed from unsigned integer to signed one of input vector.
 */
#ifndef __SYCL_COMPILER_VECTOR_ABS_CHANGED
#define __SYCL_COMPILER_VECTOR_ABS_CHANGED 20230503L
#endif

/**
 * Version of Intel MKL at which transition to OneMKL release 2023.0.0 occurs.
 */
#ifndef __INTEL_MKL_2023_0_0_VERSION_REQUIRED
#define __INTEL_MKL_2023_0_0_VERSION_REQUIRED 20230000
#endif

/**
 * @defgroup BACKEND_UTILS Backend C++ library utilities
 * @{
 * This section describes utilities used in Backend API.
 * @}
 */

/**
 * @ingroup BACKEND_UTILS
 * @brief check support of type T by SYCL device.
 *
 * To check if sycl::device may use templated type T.
 *
 * @param [in]  q  sycl::device which is examined for type support.
 *
 * @exception std::runtime_error    type T is out of support by the queue.
 */
template <typename T>
static inline void validate_type_for_device(const sycl::device &d)
{
    if constexpr (std::is_same_v<T, double>) {
        if (!d.has(sycl::aspect::fp64)) {
            throw std::runtime_error("Device " +
                                     d.get_info<sycl::info::device::name>() +
                                     " does not support type 'double'");
        }
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>) {
        if (!d.has(sycl::aspect::fp64)) {
            throw std::runtime_error(
                "Device " + d.get_info<sycl::info::device::name>() +
                " does not support type 'complex<double>'");
        }
    }
    else if constexpr (std::is_same_v<T, sycl::half>) {
        if (!d.has(sycl::aspect::fp16)) {
            throw std::runtime_error("Device " +
                                     d.get_info<sycl::info::device::name>() +
                                     " does not support type 'half'");
        }
    }
}

/**
 * @ingroup BACKEND_UTILS
 * @brief check support of type T by SYCL queue.
 *
 * To check if sycl::queue assigned to a device may use templated type T.
 *
 * @param [in]  q  sycl::queue which is examined for type support.
 *
 * @exception std::runtime_error    type T is out of support by the queue.
 */
template <typename T>
static inline void validate_type_for_device(const sycl::queue &q)
{
    validate_type_for_device<T>(q.get_device());
}

/**
 * @ingroup BACKEND_UTILS
 * @brief print std::vector to std::ostream.
 *
 * To print std::vector with POD types to std::out.
 *
 * @param [in]  vec  std::vector with types with ability to be printed.
 */
template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &vec)
{
    std::string delimiter;
    out << "{";
    // std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(out, ", "));
    // out << "\b\b}"; // last two 'backspaces' needs to eliminate last
    // delimiter. ex: {2, 3, 4, }
    for (auto &elem : vec) {
        out << delimiter << elem;
        if (delimiter.empty()) {
            delimiter.assign(", ");
        }
    }
    out << "}";

    return out;
}

/**
 * @ingroup BACKEND_UTILS
 * @brief print @ref DPNPFuncType to std::ostream.
 *
 * To print DPNPFuncType type to std::out.
 * TODO implement string representation of the enum
 *
 * @param [in]  elem  DPNPFuncType value to be printed.
 */
template <typename T>
std::ostream &operator<<(std::ostream &out, DPNPFuncType elem)
{
    out << static_cast<size_t>(elem);

    return out;
}

#endif // BACKEND_UTILS_H
