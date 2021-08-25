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

template <typename _DataType>
class DPNPC_ptr_converter final
{
    void* aux_ptr = nullptr;
    void* orig_ptr = nullptr;
    size_t size_in_bytes = 0;
    bool allocated = false;
    bool copy_back = false;

public:
    DPNPC_ptr_converter(cl::sycl::queue* execution_queue, const void* src_ptr, const size_t size, bool result = false)
    {
        copy_back = result;
        orig_ptr = const_cast<void*>(src_ptr);
        size_in_bytes = size * sizeof(_DataType);

        // enum class alloc { host = 0, device = 1, shared = 2, unknown = 3 };
        cl::sycl::usm::alloc src_ptr_type = cl::sycl::usm::alloc::unknown;
        if (execution_queue)
        {
            src_ptr_type = cl::sycl::get_pointer_type(src_ptr, execution_queue->get_context());
        }
#if 1
        std::cerr << "DPNPC_ptr_converter:";
        std::cerr << "\n\t pointer=" << src_ptr;
        std::cerr << "\n\t size=" << size;
        std::cerr << "\n\t size_in_bytes=" << size_in_bytes;
        std::cerr << "\n\t pointer type=" << (long)src_ptr_type;
        if (execution_queue)
        {
            std::cerr << "\n\t queue inorder=" << execution_queue->is_in_order();
            std::cerr << "\n\t queue is_host=" << execution_queue->is_host();
            std::cerr << "\n\t queue device is_host=" << execution_queue->get_device().is_host();
            std::cerr << "\n\t queue device is_cpu=" << execution_queue->get_device().is_cpu();
            std::cerr << "\n\t queue device is_gpu=" << execution_queue->get_device().is_gpu();
            std::cerr << "\n\t queue device is_accelerator=" << execution_queue->get_device().is_accelerator();
        }
        else
        {
            std::cerr << "\n\t no queue provided";
        }
        std::cerr << std::endl;
#endif

        if (src_ptr_type == cl::sycl::usm::alloc::unknown)
        {
            aux_ptr = dpnp_memory_alloc_c(size_in_bytes);
            dpnp_memory_memcpy_c(aux_ptr, src_ptr, size_in_bytes);
            allocated = true;
            // std::cerr << "DPNPC_ptr_converter::alloc_memory at=" << aux_ptr << std::endl;
        }
        else
        {
            aux_ptr = const_cast<void*>(src_ptr);
        }
    }

    DPNPC_ptr_converter() = delete;

    ~DPNPC_ptr_converter()
    {
        if (allocated)
        {
            // std::cerr << "DPNPC_ptr_converter::free_memory at=" << aux_ptr << std::endl;
            if (copy_back)
            {
                dpnp_memory_memcpy_c(orig_ptr, aux_ptr, size_in_bytes);
            }

            dpnp_memory_free_c(aux_ptr);
            aux_ptr = nullptr;
            allocated = false;
        }
    }

    _DataType* get_ptr() const
    {
        return reinterpret_cast<_DataType*>(aux_ptr);
    }
};

#endif // QUEUE_SYCL_H
