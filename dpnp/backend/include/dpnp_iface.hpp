//*****************************************************************************
// Copyright (c) 2016-2025, Intel Corporation
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

#include "dpnp_iface_random.hpp"

/**
 * @defgroup BACKEND_API Backend C++ library interface API
 * @{
 * This section describes Backend API.
 * @}
 */

/**
 * @ingroup BACKEND_API
 * @brief SYCL queue device status.
 *
 * Return 1 if current @ref queue is related to cpu device. return 0 otherwise.
 */
INP_DLLEXPORT size_t dpnp_queue_is_cpu_c();

/**
 * @ingroup BACKEND_API
 * @brief SYCL queue memory allocation.
 *
 * Memory allocation on the SYCL backend.
 *
 * @param [in]  size_in_bytes  Number of bytes for requested memory allocation.
 * @param [in]  q_ref          Reference to SYCL queue.
 *
 * @return  A pointer to newly created memory on SYCL device.
 */
INP_DLLEXPORT char *dpnp_memory_alloc_c(DPCTLSyclQueueRef q_ref,
                                        size_t size_in_bytes);
INP_DLLEXPORT char *dpnp_memory_alloc_c(size_t size_in_bytes);

INP_DLLEXPORT void dpnp_memory_free_c(DPCTLSyclQueueRef q_ref, void *ptr);
INP_DLLEXPORT void dpnp_memory_free_c(void *ptr);

INP_DLLEXPORT void dpnp_memory_memcpy_c(DPCTLSyclQueueRef q_ref,
                                        void *dst,
                                        const void *src,
                                        size_t size_in_bytes);
INP_DLLEXPORT void
    dpnp_memory_memcpy_c(void *dst, const void *src, size_t size_in_bytes);

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

template <typename _DataType>
INP_DLLEXPORT void dpnp_partition_c(void *array,
                                    void *array2,
                                    void *result,
                                    const size_t kth,
                                    const shape_elem_type *shape,
                                    const size_t ndim);

/**
 * @ingroup BACKEND_API
 * @brief implementation of creating filled with value array function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result1             Output array.
 * @param [in]  value               Value in array.
 * @param [in]  size                Number of elements in array.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_initval_c(DPCTLSyclQueueRef q_ref,
                   void *result1,
                   void *value,
                   size_t size,
                   const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_initval_c(void *result1, void *value, size_t size);

#define MACRO_1ARG_1TYPE_OP(__name__, __operation1__, __operation2__)          \
    template <typename _DataType>                                              \
    INP_DLLEXPORT DPCTLSyclEventRef __name__(                                  \
        DPCTLSyclQueueRef q_ref, void *result_out, const size_t result_size,   \
        const size_t result_ndim, const shape_elem_type *result_shape,         \
        const shape_elem_type *result_strides, const void *input1_in,          \
        const size_t input1_size, const size_t input1_ndim,                    \
        const shape_elem_type *input1_shape,                                   \
        const shape_elem_type *input1_strides, const size_t *where,            \
        const DPCTLEventVectorRef dep_event_vec_ref);                          \
                                                                               \
    template <typename _DataType>                                              \
    INP_DLLEXPORT void __name__(                                               \
        void *result_out, const size_t result_size, const size_t result_ndim,  \
        const shape_elem_type *result_shape,                                   \
        const shape_elem_type *result_strides, const void *input1_in,          \
        const size_t input1_size, const size_t input1_ndim,                    \
        const shape_elem_type *input1_shape,                                   \
        const shape_elem_type *input1_strides, const size_t *where);

#include <dpnp_gen_1arg_1type_tbl.hpp>

/**
 * @ingroup BACKEND_API
 * @brief modf function.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array1_in           Input array.
 * @param [out] result1_out         Output array 1.
 * @param [out] result2_out         Output array 2.
 * @param [in]  size                Number of elements in input arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType_input, typename _DataType_output>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_modf_c(DPCTLSyclQueueRef q_ref,
                void *array1_in,
                void *result1_out,
                void *result2_out,
                size_t size,
                const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType_input, typename _DataType_output>
INP_DLLEXPORT void dpnp_modf_c(void *array1_in,
                               void *result1_out,
                               void *result2_out,
                               size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Implementation of ones function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  size                Number of elements in the output array.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_ones_c(DPCTLSyclQueueRef q_ref,
                void *result,
                size_t size,
                const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_ones_c(void *result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Implementation of ones_like function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  size                Number of elements in the output array.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_ones_like_c(DPCTLSyclQueueRef q_ref,
                     void *result,
                     size_t size,
                     const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_ones_like_c(void *result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Implementation of zeros function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  size                Number of elements in the output array.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_zeros_c(DPCTLSyclQueueRef q_ref,
                 void *result,
                 size_t size,
                 const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_zeros_c(void *result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Implementation of zeros_like function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  size                Number of elements in the output array.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_zeros_like_c(DPCTLSyclQueueRef q_ref,
                      void *result,
                      size_t size,
                      const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_zeros_like_c(void *result, size_t size);

#endif // BACKEND_IFACE_H
