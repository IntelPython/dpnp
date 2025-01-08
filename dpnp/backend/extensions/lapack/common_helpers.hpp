//*****************************************************************************
// Copyright (c) 2023-2025, Intel Corporation
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

#include <stdexcept>

#include <pybind11/pybind11.h>
#include <sycl/sycl.hpp>

#include <complex>
#include <cstring>
#include <stdexcept>

// dpctl tensor headers
#include "utils/sycl_alloc_utils.hpp"

namespace dpnp::extensions::lapack::helper
{
namespace py = pybind11;

template <typename T>
struct value_type_of
{
    using type = T;
};

template <typename T>
struct value_type_of<std::complex<T>>
{
    using type = T;
};

// Rounds up the number `value` to the nearest multiple of `mult`.
template <typename intT>
inline intT round_up_mult(intT value, intT mult)
{
    intT q = (value + (mult - 1)) / mult;
    return q * mult;
}

// Checks if the shape array has any non-zero dimension.
inline bool check_zeros_shape(int ndim, const py::ssize_t *shape)
{
    size_t src_nelems(1);

    for (int i = 0; i < ndim; ++i) {
        src_nelems *= static_cast<size_t>(shape[i]);
    }
    return src_nelems == 0;
}

// Allocate the memory for the pivot indices
inline std::int64_t *alloc_ipiv(const std::int64_t n, sycl::queue &exec_q)
{
    std::int64_t *ipiv = nullptr;

    try {
        ipiv = sycl::malloc_device<std::int64_t>(n, exec_q);
        if (!ipiv) {
            throw std::runtime_error("Device allocation for ipiv failed");
        }
    } catch (sycl::exception const &e) {
        if (ipiv != nullptr)
            dpctl::tensor::alloc_utils::sycl_free_noexcept(ipiv, exec_q);
        throw std::runtime_error(
            std::string(
                "Unexpected SYCL exception caught during ipiv allocation: ") +
            e.what());
    }

    return ipiv;
}

// Allocate the total memory for the total pivot indices with proper alignment
// for batch implementations
template <typename T>
inline std::int64_t *alloc_ipiv_batch(const std::int64_t n,
                                      std::int64_t n_linear_streams,
                                      sycl::queue &exec_q)
{
    // Get padding size to ensure memory allocations are aligned to 256 bytes
    // for better performance
    const std::int64_t padding = 256 / sizeof(T);

    // Calculate the total size needed for the pivot indices array for all
    // linear streams with proper alignment
    size_t alloc_ipiv_size = round_up_mult(n_linear_streams * n, padding);

    return alloc_ipiv(alloc_ipiv_size, exec_q);
}

// Allocate the memory for the scratchpad
template <typename T>
inline T *alloc_scratchpad(std::int64_t scratchpad_size, sycl::queue &exec_q)
{
    T *scratchpad = nullptr;

    try {
        if (scratchpad_size > 0) {
            scratchpad = sycl::malloc_device<T>(scratchpad_size, exec_q);
            if (!scratchpad) {
                throw std::runtime_error(
                    "Device allocation for scratchpad failed");
            }
        }
    } catch (sycl::exception const &e) {
        if (scratchpad != nullptr) {
            dpctl::tensor::alloc_utils::sycl_free_noexcept(scratchpad, exec_q);
        }
        throw std::runtime_error(std::string("Unexpected SYCL exception caught "
                                             "during scratchpad allocation: ") +
                                 e.what());
    }

    return scratchpad;
}

// Allocate the total scratchpad memory with proper alignment for batch
// implementations
template <typename T>
inline T *alloc_scratchpad_batch(std::int64_t scratchpad_size,
                                 std::int64_t n_linear_streams,
                                 sycl::queue &exec_q)
{
    // Get padding size to ensure memory allocations are aligned to 256 bytes
    // for better performance
    const std::int64_t padding = 256 / sizeof(T);

    // Calculate the total scratchpad memory size needed for all linear
    // streams with proper alignment
    const size_t alloc_scratch_size =
        round_up_mult(n_linear_streams * scratchpad_size, padding);

    return alloc_scratchpad<T>(alloc_scratch_size, exec_q);
}
} // namespace dpnp::extensions::lapack::helper
