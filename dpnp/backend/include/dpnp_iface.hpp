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

/*
 * This header file is for interface Cython with C++.
 * It should not contains any backend specific headers (like SYCL or math
 * library) because all included headers will be exposed in Cython compilation
 * procedure
 *
 * We would like to avoid backend specific things in higher level Cython
 * modules. Any backend interface functions and types should be defined here.
 *
 * Also, this file should contains documentation on functions and types
 * which are used in the interface
 */

#pragma once
#ifndef BACKEND_IFACE_H // Cython compatibility
#define BACKEND_IFACE_H

#include <cstdint>
#include <vector>

#ifdef _WIN32
#define INP_DLLEXPORT __declspec(dllexport)
#else
#define INP_DLLEXPORT
#endif

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

typedef ssize_t shape_elem_type;

#include <dpctl_sycl_interface.h>

/**
 * @defgroup BACKEND_API Backend C++ library interface API
 * @{
 * This section describes Backend API.
 * @}
 */

/**
 * @ingroup BACKEND_API
 * @brief Return a partitioned copy of an array.
 *
 * @param [in]  q_ref             Reference to SYCL queue.
 * @param [in]  array             Input array.
 * @param [in]  array2            Copy input array.
 * @param [out] result            Result array.
 * @param [in]  kth               Element index to partition by.
 * @param [in]  shape             Shape of input array.
 * @param [in]  ndim              Number of elements in shape.
 * @param [in]  dep_event_vec_ref Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_partition_c(DPCTLSyclQueueRef q_ref,
                     void *array,
                     void *array2,
                     void *result,
                     const size_t kth,
                     const shape_elem_type *shape,
                     const size_t ndim,
                     const DPCTLEventVectorRef dep_event_vec_ref);

#endif // BACKEND_IFACE_H
