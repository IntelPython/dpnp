//*****************************************************************************
// Copyright (c) 2022, Intel Corporation
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

/*
 * This header file is for interface Cython with C++.
 * It should not contains any backend specific headers (like SYCL or math library) because
 * all included headers will be exposed in Cython compilation procedure
 *
 * We would like to avoid backend specific things in higher level Cython modules.
 * Any backend interface functions and types should be defined here.
 *
 * Also, this file should contains documentation on functions and types
 * which are used in the interface
 */

#pragma once
#ifndef BACKEND_RANDOM_STATE_H // Cython compatibility
#define BACKEND_RANDOM_STATE_H

#ifdef _WIN32
#define INP_DLLEXPORT __declspec(dllexport)
#else
#define INP_DLLEXPORT
#endif

#include <dpctl_sycl_interface.h>

// Structure storing MKL engine for MT199374x32x10 algorithm
struct mt19937_struct
{
    void* engine;
};

/**
 * @ingroup BACKEND_API
 * @brief Create a MKL engine from scalar seed.
 *
 * Invoke a common seed initialization of the engine for MT199374x32x10 algorithm.
 *
 * @param [in]  mt19937       A structure with MKL engine which will be filled with generated value by MKL.
 * @param [in]  q_ref         A refference on SYCL queue which will be used to obtain random numbers.
 * @param [in]  seed          An initial condition of the generator state.
 */
INP_DLLEXPORT void MT19937_InitScalarSeed(mt19937_struct *mt19937, DPCTLSyclQueueRef q_ref, uint32_t seed = 1);

/**
 * @ingroup BACKEND_API
 * @brief Create a MKL engine from seed vector.
 *
 * Invoke an extended seed initialization of the engine for MT199374x32x10 algorithm..
 *
 * @param [in]  mt19937       A structure with MKL engine which will be filled with generated value by MKL.
 * @param [in]  q_ref         A refference on SYCL queue which will be used to obtain random numbers.
 * @param [in]  seed          A vector with the initial conditions of the generator state.
 * @param [in]  n             Length of the vector.
 */
INP_DLLEXPORT void MT19937_InitVectorSeed(mt19937_struct *mt19937, DPCTLSyclQueueRef q_ref, uint32_t * seed, unsigned int n);

/**
 * @ingroup BACKEND_API
 * @brief Release a MKL engine.
 *
 * Release all resource required for storing of the MKL engine.
 *
 * @param [in]  mt19937       A structure with the MKL engine.
 */
INP_DLLEXPORT void MT19937_Delete(mt19937_struct *mt19937);

#endif // BACKEND_RANDOM_STATE_H
