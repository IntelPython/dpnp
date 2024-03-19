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
#include <memory>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wreorder-ctor"
#include <oneapi/mkl.hpp>
#pragma clang diagnostic pop

#include <ctime>

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

    static backend_sycl& get()
    {
        static backend_sycl backend{};
        return backend;
    }

    static sycl::queue& get_queue()
    {
        auto &be = backend_sycl::get();
        return *(be.queue_ptr);
    } 

    static mkl_rng::mt19937& get_rng_engine()
    {
        auto &be = backend_sycl::get();
        return *(be.rng_mt19937_engine_ptr);
    } 

    static mkl_rng::mcg59& get_rng_mcg59_engine()
    {
        auto &be = backend_sycl::get();
        return *(be.rng_mcg59_engine_ptr);
    }

    template <typename SeedT>
    void set_rng_engines_seed(const SeedT &seed) 
    {
        auto rng_eng_mt19937 = 
            std::make_shared<mkl_rng::mt19937>(*queue_ptr, seed);
        if (!rng_eng_mt19937) {
            throw std::runtime_error(
                "Could not create MT19937 engine with given seed"
            );
        }
        auto rng_eng_mcg59 = 
            std::make_shared<mkl_rng::mcg59>(*queue_ptr, seed);
        if (!rng_eng_mcg59) {
            throw std::runtime_error(
                "Could not create MCG59 engine with given seed"
            );
        }

        rng_mt19937_engine_ptr.swap(rng_eng_mt19937);
        rng_mcg59_engine_ptr.swap(rng_eng_mcg59);
    }

    bool backend_sycl_is_cpu() const 
    {
        const sycl::queue &q = *queue_ptr;
        return q.get_device().is_cpu();
    }

private:
    backend_sycl() 
        : queue_ptr{}, rng_mt19937_engine_ptr{}, rng_mcg59_engine_ptr{} 
    {
        const sycl::property_list &prop = (is_verbose_mode()) ? 
              sycl::property_list{sycl::property::queue::enable_profiling()}
            : sycl::property_list{};
        queue_ptr = std::make_shared<sycl::queue>(sycl::default_selector_v, prop);

        if (!queue_ptr) {
            throw std::runtime_error(
                "Could not create queue for default-selected device"
            );
        }

        constexpr std::size_t default_seed = 1;
        rng_mt19937_engine_ptr = std::make_shared<mkl_rng::mt19937>(*queue_ptr, default_seed);
        if (!rng_mt19937_engine_ptr) {
            throw std::runtime_error(
                "Could not create MT19937 engine"
            );
        }

        rng_mcg59_engine_ptr = std::make_shared<mkl_rng::mcg59>(*queue_ptr, default_seed);
        if (!rng_mcg59_engine_ptr) {
            throw std::runtime_error(
                "Could not create MCG59 engine"
            );
        }
    }

    backend_sycl(backend_sycl const &) = default;
    backend_sycl &operator=(backend_sycl const &) = default;
    backend_sycl &operator=(backend_sycl &&) = default;

    std::shared_ptr<sycl::queue> queue_ptr;
    std::shared_ptr<mkl_rng::mt19937> rng_mt19937_engine_ptr;
    std::shared_ptr<mkl_rng::mcg59> rng_mcg59_engine_ptr;
};

#endif // QUEUE_SYCL_H
