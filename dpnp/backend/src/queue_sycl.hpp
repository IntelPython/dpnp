//*****************************************************************************
// Copyright (c) 2016-2020, Intel Corporation
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
#include <dpctl_sycl_queue_manager.h>
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
    static cl::sycl::queue* queue; /**< contains SYCL queue pointer initialized in @ref backend_sycl_queue_init */
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
     * Return True if current @ref queue is related to cpu or host device
     */
    static bool backend_sycl_is_cpu();

    /**
     * Initialize @ref rng_engine
     */
    static void backend_sycl_rng_engine_init(size_t seed = 1);

    /**
     * Return the @ref queue to the user
     */
    static cl::sycl::queue& get_queue()
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
        return *(reinterpret_cast<cl::sycl::queue*>(DPCtrl_queue));
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


/**
 * @ingroup BACKEND_UTILS
 * @brief Type of memory pointer
 *
 * Return status of given pointer
 *
 * @param [in] src_ptr  Input pointer
 *
 * @return              Output pointer
 */
static inline void * get_memory_pointer(cl::sycl::queue& queue, const void* src_ptr)
{
    cl::sycl::usm::alloc ptr_type; // enum class alloc { host = 0, device = 1, shared = 2, unknown = 3 };
    void* dst_ptr = const_cast<void*>(src_ptr);

    ptr_type = cl::sycl::get_pointer_type(src_ptr, queue.get_context());

    std::cout << "Input pointer=0x" << src_ptr
              << "\n\t type=" << (int) ptr_type
              << "\n\t queue_in_order=" << queue.is_in_order()
              << "\n\t queue_is_host=" << queue.is_host()
              << "\n\t device.is_host=" << queue.get_device().is_host()
              << "\n\t device.is_cpu=" << queue.get_device().is_cpu()
              << "\n\t device.is_gpu=" << queue.get_device().is_gpu()
              << "\n\t device.is_accelerator=" << queue.get_device().is_accelerator()
              << std::endl;

    if (queue.is_host() && (ptr_type == cl::sycl::usm::alloc::device))
    { // move from Device memory to Host
        // dst_ptr = alloc()
        queue.memcpy(dst_ptr, src_ptr, 1);

    }
    else if(!queue.is_host() && ((ptr_type == cl::sycl::usm::alloc::host) || (ptr_type == cl::sycl::usm::alloc::unknown)))
    { // Queue is not on host and memory out of device. Need to copy to device.
        // dst_ptr = alloc()
        queue.memcpy(dst_ptr, src_ptr, 1);
    }

    return dst_ptr;
}

#endif // QUEUE_SYCL_H
