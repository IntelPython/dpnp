//*****************************************************************************
// Copyright (c) 2016, Intel Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// - Neither the name of the copyright holder nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
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
#ifndef QUEUE_SYCL_H // Cython compatibility
#define QUEUE_SYCL_H

// #pragma clang diagnostic push
// #pragma clang diagnostic ignored "-Wpass-failed"
#include <sycl/sycl.hpp>
// #pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wreorder-ctor"
#include <oneapi/mkl.hpp>
#pragma clang diagnostic pop

#include "dpnp_pstl.hpp" // this header must be included after <mkl.hpp>

#include "verbose.hpp"

#define DPNP_QUEUE backend_sycl::get_queue()

/**
 * This is container for the SYCL queue and related functions like queue
 * initialization and maintenance. The queue could not be initialized as a
 * global object. Global object initialization order is undefined. This class
 * postpone initialization of the SYCL queue.
 */
class backend_sycl
{
public:
    ~backend_sycl() {}

    static backend_sycl &get()
    {
#if defined(_WIN32) && INTEL_MKL_VERSION == 20260000
        // TODO: remove once MKLD-19835 is resolved
        // mt19937 (oneMKL 2026.0) destructor crashes during DLL_PROCESS_DETACH
        // on Windows (Battlemage/Level Zero). Use a heap-allocated
        // process-lifetime singleton to skip destructor; OS reclaims memory.
        static backend_sycl *backend = new backend_sycl{};
        return *backend;
#else
        static backend_sycl backend{};
        return backend;
#endif
    }

    static sycl::queue &get_queue()
    {
        auto &be = backend_sycl::get();
        return be.queue_;
    }

    bool backend_sycl_is_cpu() const
    {
        const auto &dev = queue_.get_device();
        return dev.is_cpu();
    }

private:
    backend_sycl()
        : queue_{sycl::default_selector_v,
                 (is_verbose_mode())
                     ? sycl::property_list{sycl::property::queue::
                                               enable_profiling()}
                     : sycl::property_list{}}
    {
    }

    backend_sycl(backend_sycl const &) = default;
    backend_sycl &operator=(backend_sycl const &) = default;
    backend_sycl &operator=(backend_sycl &&) = default;

    sycl::queue queue_;
};

#endif // QUEUE_SYCL_H
