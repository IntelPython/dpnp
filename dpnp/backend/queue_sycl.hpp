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

#include <CL/sycl.hpp>
#include <mkl_sycl.hpp>

#include <ctime>
#include <vector>
#include <array>
#include <map>
#include <utility>
#include <variant>
#include <mkl_sycl.hpp>

#if !defined(DPNP_LOCAL_QUEUE)
#include <dppl_sycl_queue_manager.h>
#endif

namespace mkl_rng = oneapi::mkl::rng;

#include "backend_pstl.hpp" // this header must be included after <mkl_sycl.hpp>

#define DPNP_QUEUE backend_sycl::get_queue()
#define DPNP_RNG_ENGINE backend_sycl::get_engine()
#define DPNP_RNG_ENGINE_TYPE backend_sycl::get_engine_type()

/**
 * This is container for the SYCL queue and related functions like queue initialization and maintenance
 * The queue could not be initialized as a global object. Global object initialization order is undefined.
 * This class postpone initialization of the SYCL queue
 */
class backend_sycl
{
#if defined(DPNP_LOCAL_QUEUE)
    static cl::sycl::queue* queue; /**< contains SYCL queue pointer initialized in @ref backend_sycl_queue_init */
#endif
    static EngineOptions engine_type;
    // will be in pair
    static size_t reseed_val;
    static bool reseed;
    static std::array<void*, ENGINE_OPTIONS_NUMBER> engines;

    static void destroy_queue()
    {
#if defined(DPNP_LOCAL_QUEUE)
        delete queue;
        queue = nullptr;
#endif
    }

    static void destroy_rng_engine()
    {
        switch(engine_type)
        {
            case EngineOptions::ARS5:
            {
                delete reinterpret_cast<mkl_rng::ars5*>(engines[engine_type]);
                break;
            }
            case EngineOptions::MCG32M1:
            {
                delete reinterpret_cast<mkl_rng::mcg31m1*>(engines[engine_type]);
                break;
            }
            case EngineOptions::MCG59:
            {
                delete reinterpret_cast<mkl_rng::mcg59*>(engines[engine_type]);
                break;
            }
            case EngineOptions::MRG32K3A:
            {
                delete reinterpret_cast<mkl_rng::mrg32k3a*>(engines[engine_type]);
                break;
            }
            case EngineOptions::MT19937:
            {
                delete reinterpret_cast<mkl_rng::mt19937*>(engines[engine_type]);
                break;
            }
            case EngineOptions::MT2203:
            {
                delete reinterpret_cast<mkl_rng::mt2203*>(engines[engine_type]);
                break;
            }
            case EngineOptions::NIEDERREITER:
            {
                delete reinterpret_cast<mkl_rng::niederreiter*>(engines[engine_type]);
                break;
            }
            case EngineOptions::NONDETERMINISTIC:
            {
                delete reinterpret_cast<mkl_rng::nondeterministic*>(engines[engine_type]);
                break;
            }
            case EngineOptions::PHILOX4X32X10:
            {
                delete reinterpret_cast<mkl_rng::philox4x32x10*>(engines[engine_type]);
                break;
            }
            case EngineOptions::R250:
            {
                delete reinterpret_cast<mkl_rng::r250*>(engines[engine_type]);
                break;
            }
            case EngineOptions::SFMT19937:
            {
                delete reinterpret_cast<mkl_rng::sfmt19937*>(engines[engine_type]);
                break;
            }
            case EngineOptions::SOBOL:
            {
                delete reinterpret_cast<mkl_rng::sobol*>(engines[engine_type]);
                break;
            }
            case EngineOptions::WICHMANN_HILL:
            {
                delete reinterpret_cast<mkl_rng::wichmann_hill*>(engines[engine_type]);
                break;
            }
        }
        engines[engine_type] = nullptr;
    }

public:
    backend_sycl()
    {
#if defined(DPNP_LOCAL_QUEUE)
        queue = nullptr;
#endif
        engine_type = EngineOptions::MT19937;
        engines.fill(nullptr);
        reseed = false;
    }

    virtual ~backend_sycl()
    {
        // TODO:
        // delete for all ptrs
        backend_sycl::destroy_queue();
        // will use iterator for enum class
        //for(int i = EngineOptions::ARS5; i <= EngineOptions::WICHMANN_HILL; ++i)
        //{
        //    engine_type = i;
        //    backend_sycl::destroy_rng_engine();
        //}

        backend_sycl::destroy_rng_engine();
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

    static void backend_sycl_rng_engine_init();
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
        DPPLSyclQueueRef DPCtrl_queue = DPPLQueueMgr_GetCurrentQueue();
        return *(reinterpret_cast<cl::sycl::queue*>(DPCtrl_queue));
#endif
    }
    static void * get_engine()
    {
        if (!engines[engine_type] || reseed)
        {
            backend_sycl_rng_engine_init();
        }
        return engines[engine_type];
    }

    static EngineOptions& get_engine_type()
    {
        return engine_type;
    }

    static void set_engine_type(EngineOptions engine_type_)
    {
        engine_type = engine_type_;
    }

    static void set_seed(size_t value)
    {
        reseed = true;
        reseed_val = value;
    }
};

#endif // QUEUE_SYCL_H
