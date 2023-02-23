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

#pragma once
#ifndef QUEUE_SYCL_H // Cython compatibility
#define QUEUE_SYCL_H

//#pragma clang diagnostic push
//#pragma clang diagnostic ignored "-Wpass-failed"
#include <CL/sycl.hpp>
//#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wreorder-ctor"
#include <oneapi/mkl.hpp>
#pragma clang diagnostic pop

#include <ctime>

#if !defined(DPNP_LOCAL_QUEUE)
#if defined __has_include
#if __has_include(<dpctl_sycl_interface.h>)
#include <dpctl_sycl_interface.h>
#else
#include <dpctl_sycl_queue_manager.h>
#endif
#endif
#endif

#include "dpnp_pstl.hpp" // this header must be included after <mkl.hpp>

#include "verbose.hpp"

namespace mkl_rng = oneapi::mkl::rng;

#define DPNP_QUEUE      backend_sycl::get_queue()
#define DPNP_RNG_ENGINE backend_sycl::get_rng_engine()

/**
 * This is container for the SYCL queue, random number generation engine and related functions like queue and engine
 * initialization and maintenance.
 * The queue could not be initialized as a global object. Global object initialization order is undefined.
 * This class postpone initialization of the SYCL queue and mt19937 random number generation engine.
 */
class backend_sycl
{
#if defined(DPNP_LOCAL_QUEUE)
    static sycl::queue* queue; /**< contains SYCL queue pointer initialized in @ref backend_sycl_queue_init */
#endif
    static mkl_rng::mt19937* rng_engine; /**< RNG engine ptr. initialized in @ref backend_sycl_rng_engine_init */

    static void destroy()
    {
        backend_sycl::destroy_rng_engine();
#if defined(DPNP_LOCAL_QUEUE)
        delete queue;
        queue = nullptr;
#endif
    }

    static void destroy_rng_engine()
    {
        delete rng_engine;
        rng_engine = nullptr;
    }

public:
    backend_sycl()
    {
#if defined(DPNP_LOCAL_QUEUE)
        queue = nullptr;
        rng_engine = nullptr;
#endif
    }

    virtual ~backend_sycl()
    {
        backend_sycl::destroy();
    }

    /**
     * Explicitly disallow copying
     */
    backend_sycl(const backend_sycl&) = delete;
    backend_sycl& operator=(const backend_sycl&) = delete;

    /**
     * Initialize @ref queue
     */
    static void backend_sycl_queue_init(QueueOptions selector = QueueOptions::CPU_SELECTOR);

    /**
     * Return True if current @ref queue is related to cpu device
     */
    static bool backend_sycl_is_cpu();

    /**
     * Initialize @ref rng_engine
     */
    static void backend_sycl_rng_engine_init(size_t seed = 1);

    /**
     * Return the @ref queue to the user
     */
    static sycl::queue& get_queue()
    {
#if defined(DPNP_LOCAL_QUEUE)
        if (!queue)
        {
            backend_sycl_queue_init();
        }

        return *queue;
#else
        // temporal solution. Started from Sept-2020
        DPCTLSyclQueueRef DPCtrl_queue = DPCTLQueueMgr_GetCurrentQueue();
        if (DPCtrl_queue == nullptr)
        {
            std::string reason = (DPCTLQueueMgr_GetQueueStackSize() == static_cast<size_t>(-1))
                                     ? ": the queue stack is empty, probably no device is available."
                                     : ".";
            throw std::runtime_error("Failed to create a copy of SYCL queue with default device" + reason);
        }
        return *(reinterpret_cast<sycl::queue*>(DPCtrl_queue));
#endif
    }

    /**
     * Return the @ref rng_engine to the user
     */
    static mkl_rng::mt19937& get_rng_engine()
    {
        if (!rng_engine)
        {
            backend_sycl_rng_engine_init();
        }
        return *rng_engine;
    }
};

#endif // QUEUE_SYCL_H
