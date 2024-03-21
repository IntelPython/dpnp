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

#pragma once
#ifndef QUEUE_SYCL_H // Cython compatibility
#define QUEUE_SYCL_H

//#pragma clang diagnostic push
//#pragma clang diagnostic ignored "-Wpass-failed"
#include <sycl/sycl.hpp>
//#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wreorder-ctor"
#include <oneapi/mkl.hpp>
#pragma clang diagnostic pop

#include <utility>

#include "dpnp_pstl.hpp" // this header must be included after <mkl.hpp>

#include "verbose.hpp"

namespace mkl_rng = oneapi::mkl::rng;

#define DPNP_QUEUE            backend_sycl::get_queue()
#define DPNP_RNG_ENGINE       backend_sycl::get_rng_engine()
#define DPNP_RNG_MCG59_ENGINE backend_sycl::get_rng_mcg59_engine()

/**
 * This is container for the SYCL queue, random number generation engine and
 * related functions like queue and engine initialization and maintenance. The
 * queue could not be initialized as a global object. Global object
 * initialization order is undefined. This class postpone initialization of the
 * SYCL queue and mt19937 random number generation engine.
 */
class backend_sycl
{
public:
    ~backend_sycl() {}

    static backend_sycl &get()
    {
        static backend_sycl backend{};
        return backend;
    }

    static sycl::queue &get_queue()
    {
        auto &be = backend_sycl::get();
        return be.queue_;
    }

    static mkl_rng::mt19937 &get_rng_engine()
    {
        auto &be = backend_sycl::get();
        return be.rng_mt19937_engine_;
    }

    static mkl_rng::mcg59 &get_rng_mcg59_engine()
    {
        auto &be = backend_sycl::get();
        return be.rng_mcg59_engine_;
    }

    template <typename SeedT>
    void set_rng_engines_seed(const SeedT &seed)
    {
        mkl_rng::mt19937 rng_eng_mt19937(queue_, seed);
        mkl_rng::mcg59 rng_eng_mcg59(queue_, seed);

        // now that instances are created, let's move them
        rng_mt19937_engine_ = std::move(rng_eng_mt19937);
        rng_mcg59_engine_ = std::move(rng_eng_mcg59);
    }

    bool backend_sycl_is_cpu() const
    {
        const auto &dev = queue_.get_device();
        return dev.is_cpu();
    }

private:
    static constexpr std::size_t default_seed = 1;

    backend_sycl()
        : queue_{sycl::default_selector_v,
                 (is_verbose_mode())
                     ? sycl::property_list{sycl::property::queue::
                                               enable_profiling()}
                     : sycl::property_list{}},
          rng_mt19937_engine_{queue_, default_seed}, rng_mcg59_engine_{
                                                         queue_, default_seed}
    {
    }

    backend_sycl(backend_sycl const &) = default;
    backend_sycl &operator=(backend_sycl const &) = default;
    backend_sycl &operator=(backend_sycl &&) = default;

    sycl::queue queue_;
    mkl_rng::mt19937 rng_mt19937_engine_;
    mkl_rng::mcg59 rng_mcg59_engine_;
};

#endif // QUEUE_SYCL_H
