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

#define DPNP_QUEUE            backend_sycl_singleton::get_queue()
#define DPNP_RNG_ENGINE       backend_sycl_singleton::get_rng_engine()
#define DPNP_RNG_MCG59_ENGINE backend_sycl_singleton::get_rng_mcg59_engine()

class backend_sycl_singleton {
public:
    ~backend_sycl_singleton() {}

    static backend_sycl_singleton& get() {
        static backend_sycl_singleton backend = lookup();
        return backend;
    }

    static sycl::queue& get_queue() {
        auto &be = backend_sycl_singleton::get();
        return *(be.queue_ptr);
    } 

    static mkl_rng::mt19937& get_rng_engine() {
        auto &be = backend_sycl_singleton::get();
        return *(be.rng_mt19937_engine_ptr);
    } 

    static mkl_rng::mcg59& get_rng_mcg59_engine() {
        auto &be = backend_sycl_singleton::get();
        return *(be.rng_mcg59_engine_ptr);
    }

    template <typename SeedT>
    void set_rng_engines_seed(const SeedT &seed) {
        auto rng_eng_mt19937 = std::make_shared<mkl_rng::mt19937>(*queue_ptr, seed);
        if (!rng_eng_mt19937) {
            throw std::runtime_error(
                "Could not create MT19937 engine with given seed"
            );
        }
        auto rng_eng_mcg59 = std::make_shared<mkl_rng::mcg59>(*queue_ptr, seed);
        if (!rng_eng_mcg59) {
            throw std::runtime_error(
                "Could not create MCG59 engine with given seed"
            );
        }

        rng_mt19937_engine_ptr.swap(rng_eng_mt19937);
        rng_mcg59_engine_ptr.swap(rng_eng_mcg59);
    }

    bool backend_sycl_is_cpu() const {
        if (queue_ptr) {
            const sycl::queue &q = *queue_ptr;

            return q.get_device().is_cpu();
        }
        return false;     
    }

private:
    backend_sycl_singleton() : 
        queue_ptr{}, rng_mt19937_engine_ptr{}, rng_mcg59_engine_ptr{} 
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

    static backend_sycl_singleton& lookup() {
        static backend_sycl_singleton backend{};
        return backend;
    }

    std::shared_ptr<sycl::queue> queue_ptr;
    std::shared_ptr<mkl_rng::mt19937> rng_mt19937_engine_ptr;
    std::shared_ptr<mkl_rng::mcg59> rng_mcg59_engine_ptr;
};

/**
 * This is container for the SYCL queue, random number generation engine and
 * related functions like queue and engine initialization and maintenance. The
 * queue could not be initialized as a global object. Global object
 * initialization order is undefined. This class postpone initialization of the
 * SYCL queue and mt19937 random number generation engine.
 */
#if 0
class backend_sycl
{
    /**< contains SYCL queue pointer initialized in
        @ref backend_sycl_queue_init */
    static sycl::queue *queue;
    /**< RNG MT19937 engine ptr. initialized in @ref
                        backend_sycl_rng_engine_init */
    static mkl_rng::mt19937 *rng_engine;
    /**< RNG MCG59 engine ptr. initialized in @ref
        backend_sycl_rng_engine_init */
    static mkl_rng::mcg59 *rng_mcg59_engine;

    static void destroy()
    {
        backend_sycl::destroy_rng_engine();
        delete queue;
    }

    static void destroy_rng_engine()
    {
        delete rng_engine;
        delete rng_mcg59_engine;

        rng_engine = nullptr;
        rng_mcg59_engine = nullptr;
    }

public:
    backend_sycl()
    {
        queue = nullptr;
        rng_engine = nullptr;
        rng_mcg59_engine = nullptr;
    }

    virtual ~backend_sycl()
    {
        backend_sycl::destroy();
    }

    /**
     * Explicitly disallow copying
     */
    backend_sycl(const backend_sycl &) = delete;
    backend_sycl &operator=(const backend_sycl &) = delete;

    /**
     * Initialize @ref queue
     */
    static void backend_sycl_queue_init(
        QueueOptions selector = QueueOptions::AUTO_SELECTOR);

    /**
     * Return True if current @ref queue is related to cpu device
     */
    static bool backend_sycl_is_cpu();

    /**
     * Initialize @ref rng_engine and @ref rng_mcg59_engine
     */
    static void backend_sycl_rng_engine_init(size_t seed = 1);

    /**
     * Return the @ref queue to the user
     */
    static sycl::queue &get_queue()
    {
        if (!queue) {
            backend_sycl_queue_init();
        }

        return *queue;
    }

    /**
     * Return the @ref rng_engine to the user
     */
    static mkl_rng::mt19937 &get_rng_engine()
    {
        if (!rng_engine) {
            backend_sycl_rng_engine_init();
        }
        return *rng_engine;
    }

    /**
     * Return the @ref rng_mcg59_engine to the user
     */
    static mkl_rng::mcg59 &get_rng_mcg59_engine()
    {
        if (!rng_engine) {
            backend_sycl_rng_engine_init();
        }
        return *rng_mcg59_engine;
    }
};
#endif

#endif // QUEUE_SYCL_H
