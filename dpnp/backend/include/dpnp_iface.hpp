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

#include "dpnp_iface_fft.hpp"
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
 * @brief Test whether all array elements along a given axis evaluate to True.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array               Input array.
 * @param [out] result              Output array.
 * @param [in]  size                Number of input elements in `array`.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_all_c(DPCTLSyclQueueRef q_ref,
               const void *array,
               void *result,
               const size_t size,
               const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void
    dpnp_all_c(const void *array, void *result, const size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Returns True if two arrays are element-wise equal within a tolerance.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array1_in           First input array.
 * @param [in]  array2_in           Second input array.
 * @param [out] result1             Output array.
 * @param [in]  size                Number of input elements in `array`.
 * @param [in]  rtol                The relative tolerance parameter.
 * @param [in]  atol                The absolute tolerance parameter.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType1, typename _DataType2, typename _ResultType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_allclose_c(DPCTLSyclQueueRef q_ref,
                    const void *array1_in,
                    const void *array2_in,
                    void *result1,
                    const size_t size,
                    double rtol,
                    double atol,
                    const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType1, typename _DataType2, typename _ResultType>
INP_DLLEXPORT void dpnp_allclose_c(const void *array1_in,
                                   const void *array2_in,
                                   void *result1,
                                   const size_t size,
                                   double rtol,
                                   double atol);

/**
 * @ingroup BACKEND_API
 * @brief Test whether any array element along a given axis evaluates to True.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array               Input array.
 * @param [out] result              Output array.
 * @param [in]  size                Number of input elements in `array`.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_any_c(DPCTLSyclQueueRef q_ref,
               const void *array,
               void *result,
               const size_t size,
               const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void
    dpnp_any_c(const void *array, void *result, const size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Array initialization
 *
 * Input array, step based, initialization procedure.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  start               Start of initialization sequence
 * @param [in]  step                Step for initialization sequence
 * @param [out] result1             Output array.
 * @param [in]  size                Number of elements in input arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_arange_c(DPCTLSyclQueueRef q_ref,
                  size_t start,
                  size_t step,
                  void *result1,
                  size_t size,
                  const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void
    dpnp_arange_c(size_t start, size_t step, void *result1, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Copy of the array, cast to a specified type.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array               Input array.
 * @param [out] result              Output array.
 * @param [in]  size                Number of input elements in `array`.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_astype_c(DPCTLSyclQueueRef q_ref,
                  const void *array,
                  void *result,
                  const size_t size,
                  const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void
    dpnp_astype_c(const void *array, void *result, const size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Implementation of full function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array_in            Input one-element array.
 * @param [out] result              Output array.
 * @param [in]  size                Number of elements in the output array.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_full_c(DPCTLSyclQueueRef q_ref,
                void *array_in,
                void *result,
                const size_t size,
                const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_full_c(void *array_in, void *result, const size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Implementation of full_like function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array_in            Input one-element array.
 * @param [out] result              Output array.
 * @param [in]  size                Number of elements in the output array.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_full_like_c(DPCTLSyclQueueRef q_ref,
                     void *array_in,
                     void *result,
                     size_t size,
                     const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_full_like_c(void *array_in, void *result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Matrix multiplication.
 *
 * Matrix multiplication procedure.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result_out          Output array.
 * @param [in]  result_size         Size of output array.
 * @param [in]  result_ndim         Number of output array dimensions.
 * @param [in]  result_shape        Shape of output array.
 * @param [in]  result_strides      Strides of output array.
 * @param [in]  input1_in           First input array.
 * @param [in]  input1_size         Size of first input array.
 * @param [in]  input1_ndim         Number of first input array dimensions.
 * @param [in]  input1_shape        Shape of first input array.
 * @param [in]  input1_strides      Strides of first input array.
 * @param [in]  input2_in           Second input array.
 * @param [in]  input2_size         Size of second input array.
 * @param [in]  input2_ndim         Number of second input array dimensions.
 * @param [in]  input2_shape        Shape of second input array.
 * @param [in]  input2_strides      Strides of second input array.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_matmul_c(DPCTLSyclQueueRef q_ref,
                  void *result_out,
                  const size_t result_size,
                  const size_t result_ndim,
                  const shape_elem_type *result_shape,
                  const shape_elem_type *result_strides,
                  const void *input1_in,
                  const size_t input1_size,
                  const size_t input1_ndim,
                  const shape_elem_type *input1_shape,
                  const shape_elem_type *input1_strides,
                  const void *input2_in,
                  const size_t input2_size,
                  const size_t input2_ndim,
                  const shape_elem_type *input2_shape,
                  const shape_elem_type *input2_strides,
                  const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_matmul_c(void *result_out,
                                 const size_t result_size,
                                 const size_t result_ndim,
                                 const shape_elem_type *result_shape,
                                 const shape_elem_type *result_strides,
                                 const void *input1_in,
                                 const size_t input1_size,
                                 const size_t input1_ndim,
                                 const shape_elem_type *input1_shape,
                                 const shape_elem_type *input1_strides,
                                 const void *input2_in,
                                 const size_t input2_size,
                                 const size_t input2_ndim,
                                 const shape_elem_type *input2_shape,
                                 const shape_elem_type *input2_strides);

/**
 * @ingroup BACKEND_API
 * @brief Compute the variance along the specified axis, while ignoring NaNs.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array               Input array.
 * @param [in]  mask_arr            Input mask array when elem is nan.
 * @param [out] result              Output array.
 * @param [in]  result_size         Output array size.
 * @param [in]  size                Number of elements in input arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_nanvar_c(DPCTLSyclQueueRef q_ref,
                  void *array,
                  void *mask_arr,
                  void *result,
                  const size_t result_size,
                  size_t size,
                  const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_nanvar_c(void *array,
                                 void *mask_arr,
                                 void *result,
                                 const size_t result_size,
                                 size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Return the indices of the elements that are non-zero.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array1              Input array.
 * @param [out] result1             Output array.
 * @param [in]  result_size         Output array size.
 * @param [in]  shape               Shape of input array.
 * @param [in]  ndim                Number of elements in shape.
 * @param [in]  j                   Number input array.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_nonzero_c(DPCTLSyclQueueRef q_ref,
                   const void *array1,
                   void *result1,
                   const size_t result_size,
                   const shape_elem_type *shape,
                   const size_t ndim,
                   const size_t j,
                   const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_nonzero_c(const void *array1,
                                  void *result1,
                                  const size_t result_size,
                                  const shape_elem_type *shape,
                                  const size_t ndim,
                                  const size_t j);

/**
 * @ingroup BACKEND_API
 * @brief absolute function.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  input1_in           Input array.
 * @param [out] result1             Output array.
 * @param [in]  size                Number of elements in input arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_elemwise_absolute_c(DPCTLSyclQueueRef q_ref,
                             const void *input1_in,
                             void *result1,
                             size_t size,
                             const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void
    dpnp_elemwise_absolute_c(const void *input1_in, void *result1, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Custom implementation of dot function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result_out          Output array.
 * @param [in]  result_size         Size of output array.
 * @param [in]  result_ndim         Number of output array dimensions.
 * @param [in]  result_shape        Shape of output array.
 * @param [in]  result_strides      Strides of output array.
 * @param [in]  input1_in           First input array.
 * @param [in]  input1_size         Size of first input array.
 * @param [in]  input1_ndim         Number of first input array dimensions.
 * @param [in]  input1_shape        Shape of first input array.
 * @param [in]  input1_strides      Strides of first input array.
 * @param [in]  input2_in           Second input array.
 * @param [in]  input2_size         Size of second input array.
 * @param [in]  input2_ndim         Number of second input array dimensions.
 * @param [in]  input2_shape        Shape of second input array.
 * @param [in]  input2_strides      Strides of second input array.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType_output,
          typename _DataType_input1,
          typename _DataType_input2>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_dot_c(DPCTLSyclQueueRef q_ref,
               void *result_out,
               const size_t result_size,
               const size_t result_ndim,
               const shape_elem_type *result_shape,
               const shape_elem_type *result_strides,
               const void *input1_in,
               const size_t input1_size,
               const size_t input1_ndim,
               const shape_elem_type *input1_shape,
               const shape_elem_type *input1_strides,
               const void *input2_in,
               const size_t input2_size,
               const size_t input2_ndim,
               const shape_elem_type *input2_shape,
               const shape_elem_type *input2_strides,
               const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType_output,
          typename _DataType_input1,
          typename _DataType_input2>
INP_DLLEXPORT void dpnp_dot_c(void *result_out,
                              const size_t result_size,
                              const size_t result_ndim,
                              const shape_elem_type *result_shape,
                              const shape_elem_type *result_strides,
                              const void *input1_in,
                              const size_t input1_size,
                              const size_t input1_ndim,
                              const shape_elem_type *input1_shape,
                              const shape_elem_type *input1_strides,
                              const void *input2_in,
                              const size_t input2_size,
                              const size_t input2_ndim,
                              const shape_elem_type *input2_shape,
                              const shape_elem_type *input2_strides);

/**
 * @ingroup BACKEND_API
 * @brief Custom implementation of cross function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result_out          Output array.
 * @param [in]  input1_in           First input array.
 * @param [in]  input1_size         Size of first input array.
 * @param [in]  input1_shape        Shape of first input array.
 * @param [in]  input1_shape_ndim   Number of first array dimensions.
 * @param [in]  input2_in           Second input array.
 * @param [in]  input2_size         Shape of second input array.
 * @param [in]  input2_shape        Shape of first input array.
 * @param [in]  input2_shape_ndim   Number of second array dimensions.
 * @param [in]  where               Mask array.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType_output,
          typename _DataType_input1,
          typename _DataType_input2>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_cross_c(DPCTLSyclQueueRef q_ref,
                 void *result_out,
                 const void *input1_in,
                 const size_t input1_size,
                 const shape_elem_type *input1_shape,
                 const size_t input1_shape_ndim,
                 const void *input2_in,
                 const size_t input2_size,
                 const shape_elem_type *input2_shape,
                 const size_t input2_shape_ndim,
                 const size_t *where,
                 const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType_output,
          typename _DataType_input1,
          typename _DataType_input2>
INP_DLLEXPORT void dpnp_cross_c(void *result_out,
                                const void *input1_in,
                                const size_t input1_size,
                                const shape_elem_type *input1_shape,
                                const size_t input1_shape_ndim,
                                const void *input2_in,
                                const size_t input2_size,
                                const shape_elem_type *input2_shape,
                                const size_t input2_shape_ndim,
                                const size_t *where);

/**
 * @ingroup BACKEND_API
 * @brief Custom implementation of cumprod function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array1_in           Input array.
 * @param [out] result1             Output array.
 * @param [in]  size                Number of elements in input arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 *
 */
template <typename _DataType_input, typename _DataType_output>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_cumprod_c(DPCTLSyclQueueRef q_ref,
                   void *array1_in,
                   void *result1,
                   size_t size,
                   const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType_input, typename _DataType_output>
INP_DLLEXPORT void dpnp_cumprod_c(void *array1_in, void *result1, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Custom implementation of cumsum function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array1_in           Input array.
 * @param [out] result1             Output array.
 * @param [in]  size                Number of elements in input arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 *
 */
template <typename _DataType_input, typename _DataType_output>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_cumsum_c(DPCTLSyclQueueRef q_ref,
                  void *array1_in,
                  void *result1,
                  size_t size,
                  const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType_input, typename _DataType_output>
INP_DLLEXPORT void dpnp_cumsum_c(void *array1_in, void *result1, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief The differences between consecutive elements of an array.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result_out          Output array.
 * @param [in]  result_size         Size of output array.
 * @param [in]  result_ndim         Number of output array dimensions.
 * @param [in]  result_shape        Shape of output array.
 * @param [in]  result_strides      Strides of output array.
 * @param [in]  input1_in           First input array.
 * @param [in]  input1_size         Size of first input array.
 * @param [in]  input1_ndim         Number of first input array dimensions.
 * @param [in]  input1_shape        Shape of first input array.
 * @param [in]  input1_strides      Strides of first input array.
 * @param [in]  where               Mask array.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType_input, typename _DataType_output>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_ediff1d_c(DPCTLSyclQueueRef q_ref,
                   void *result_out,
                   const size_t result_size,
                   const size_t result_ndim,
                   const shape_elem_type *result_shape,
                   const shape_elem_type *result_strides,
                   const void *input1_in,
                   const size_t input1_size,
                   const size_t input1_ndim,
                   const shape_elem_type *input1_shape,
                   const shape_elem_type *input1_strides,
                   const size_t *where,
                   const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType_input, typename _DataType_output>
INP_DLLEXPORT void dpnp_ediff1d_c(void *result_out,
                                  const size_t result_size,
                                  const size_t result_ndim,
                                  const shape_elem_type *result_shape,
                                  const shape_elem_type *result_strides,
                                  const void *input1_in,
                                  const size_t input1_size,
                                  const size_t input1_ndim,
                                  const shape_elem_type *input1_shape,
                                  const shape_elem_type *input1_strides,
                                  const size_t *where);

/**
 * @ingroup BACKEND_API
 * @brief Compute summary of input array elements.
 *
 * Input array is expected as @ref _DataType_input type and assume result as
 * @ref _DataType_output type. The function creates no memory.
 *
 * Empty @ref input_shape means scalar.
 *
 * @param [in]  q_ref             Reference to SYCL queue.
 * @param [out] result_out        Output array pointer. @ref _DataType_output
 * type is expected
 * @param [in]  input_in          Input array pointer. @ref _DataType_input type
 * is expected
 * @param [in]  input_shape       Shape of @ref input_in
 * @param [in]  input_shape_ndim  Number of elements in @ref input_shape
 * @param [in]  axes              Array of axes to apply to @ref input_shape
 * @param [in]  axes_ndim         Number of elements in @ref axes
 * @param [in]  initial           Pointer to initial value for the algorithm.
 * @ref _DataType_input is expected
 * @param [in]  where             mask array
 * @param [in]  dep_event_vec_ref Reference to vector of SYCL events.
 */
template <typename _DataType_output, typename _DataType_input>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_sum_c(DPCTLSyclQueueRef q_ref,
               void *result_out,
               const void *input_in,
               const shape_elem_type *input_shape,
               const size_t input_shape_ndim,
               const shape_elem_type *axes,
               const size_t axes_ndim,
               const void *initial,
               const long *where,
               const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType_output, typename _DataType_input>
INP_DLLEXPORT void dpnp_sum_c(void *result_out,
                              const void *input_in,
                              const shape_elem_type *input_shape,
                              const size_t input_shape_ndim,
                              const shape_elem_type *axes,
                              const size_t axes_ndim,
                              const void *initial,
                              const long *where);

/**
 * @ingroup BACKEND_API
 * @brief Custom implementation of count_nonzero function
 *
 * @param [in]  q_ref             Reference to SYCL queue.
 * @param [in]  array1_in         Input array.
 * @param [out] result1_out       Output array.
 * @param [in]  size              Number of elements in input arrays.
 * @param [in]  dep_event_vec_ref Reference to vector of SYCL events.
 *
 */
template <typename _DataType_input, typename _DataType_output>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_count_nonzero_c(DPCTLSyclQueueRef q_ref,
                         void *array1_in,
                         void *result1_out,
                         size_t size,
                         const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType_input, typename _DataType_output>
INP_DLLEXPORT void
    dpnp_count_nonzero_c(void *array1_in, void *result1_out, size_t size);

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
 * @brief Place of array elements
 *
 * @param [in]  q_ref             Reference to SYCL queue.
 * @param [in]  arr               Input array.
 * @param [in]  mask              Mask array.
 * @param [in]  vals              Vals array.
 * @param [in]  arr_size          Number of input elements in `arr`.
 * @param [in]  vals_size         Number of input elements in `vals`.
 * @param [in]  dep_event_vec_ref Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_place_c(DPCTLSyclQueueRef q_ref,
                 void *arr,
                 long *mask,
                 void *vals,
                 const size_t arr_size,
                 const size_t vals_size,
                 const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_place_c(void *arr,
                                long *mask,
                                void *vals,
                                const size_t arr_size,
                                const size_t vals_size);

/**
 * @ingroup BACKEND_API
 * @brief Compute Product of input array elements.
 *
 * Input array is expected as @ref _DataType_input type and assume result as
 * @ref _DataType_output type. The function creates no memory.
 *
 * Empty @ref input_shape means scalar.
 *
 * @param [in]  q_ref             Reference to SYCL queue.
 * @param [out] result_out        Output array pointer. @ref _DataType_output
 * type is expected
 * @param [in]  input_in          Input array pointer. @ref _DataType_input type
 * is expected
 * @param [in]  input_shape       Shape of @ref input_in
 * @param [in]  input_shape_ndim  Number of elements in @ref input_shape
 * @param [in]  axes              Array of axes to apply to @ref input_shape
 * @param [in]  axes_ndim         Number of elements in @ref axes
 * @param [in]  initial           Pointer to initial value for the algorithm.
 * @ref _DataType_input is expected
 * @param [in]  where             mask array
 * @param [in]  dep_event_vec_ref Reference to vector of SYCL events.
 */
template <typename _DataType_output, typename _DataType_input>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_prod_c(DPCTLSyclQueueRef q_ref,
                void *result_out,
                const void *input_in,
                const shape_elem_type *input_shape,
                const size_t input_shape_ndim,
                const shape_elem_type *axes,
                const size_t axes_ndim,
                const void *initial,
                const long *where,
                const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType_output, typename _DataType_input>
INP_DLLEXPORT void dpnp_prod_c(void *result_out,
                               const void *input_in,
                               const shape_elem_type *input_shape,
                               const size_t input_shape_ndim,
                               const shape_elem_type *axes,
                               const size_t axes_ndim,
                               const void *initial,
                               const long *where);

/**
 * @ingroup BACKEND_API
 * @brief Range of values (maximum - minimum) along an axis.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result_out          Output array.
 * @param [in]  result_size         Size of output array.
 * @param [in]  result_ndim         Number of output array dimensions.
 * @param [in]  result_shape        Shape of output array.
 * @param [in]  result_strides      Strides of output array.
 * @param [in]  input_in            First input array.
 * @param [in]  input_size          Size of first input array.
 * @param [in]  input_ndim          Number of first input array dimensions.
 * @param [in]  input_shape         Shape of first input array.
 * @param [in]  input_strides       Strides of first input array.
 * @param [in]  axis                Axis.
 * @param [in]  naxis               Number of elements in axis.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_ptp_c(DPCTLSyclQueueRef q_ref,
               void *result_out,
               const size_t result_size,
               const size_t result_ndim,
               const shape_elem_type *result_shape,
               const shape_elem_type *result_strides,
               const void *input_in,
               const size_t input_size,
               const size_t input_ndim,
               const shape_elem_type *input_shape,
               const shape_elem_type *input_strides,
               const shape_elem_type *axis,
               const size_t naxis,
               const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_ptp_c(void *result_out,
                              const size_t result_size,
                              const size_t result_ndim,
                              const shape_elem_type *result_shape,
                              const shape_elem_type *result_strides,
                              const void *input_in,
                              const size_t input_size,
                              const size_t input_ndim,
                              const shape_elem_type *input_shape,
                              const shape_elem_type *input_strides,
                              const shape_elem_type *axis,
                              const size_t naxis);

/**
 * @ingroup BACKEND_API
 * @brief Replaces specified elements of an array with given values.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array               Input array.
 * @param [in]  ind                 Target indices, interpreted as integers.
 * @param [in]  v                   Values to place in array at target indices.
 * @param [in]  size                Number of input elements in `array`.
 * @param [in]  size_ind            Number of input elements in `ind`.
 * @param [in]  size_v              Number of input elements in `v`.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType, typename _IndecesType, typename _ValueType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_put_c(DPCTLSyclQueueRef q_ref,
               void *array,
               void *ind,
               void *v,
               const size_t size,
               const size_t size_ind,
               const size_t size_v,
               const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType, typename _IndecesType, typename _ValueType>
INP_DLLEXPORT void dpnp_put_c(void *array,
                              void *ind,
                              void *v,
                              const size_t size,
                              const size_t size_ind,
                              const size_t size_v);

/**
 * @ingroup BACKEND_API
 * @brief Put values into the destination array by matching 1d index and data
 * slices.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  arr_in              Input array.
 * @param [in]  indices_in          Indices to change along each 1d slice of
 * arr.
 * @param [in]  values_in           Values to insert at those indices.
 * @param [in]  axis                The axis to take 1d slices along.
 * @param [in]  shape               Shape of input array.
 * @param [in]  ndim                Number of input array dimensions.
 * @param [in]  size_indices        Size of indices.
 * @param [in]  values_size         Size of values.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_put_along_axis_c(DPCTLSyclQueueRef q_ref,
                          void *arr_in,
                          long *indices_in,
                          void *values_in,
                          size_t axis,
                          const shape_elem_type *shape,
                          size_t ndim,
                          size_t size_indices,
                          size_t values_size,
                          const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_put_along_axis_c(void *arr_in,
                                         long *indices_in,
                                         void *values_in,
                                         size_t axis,
                                         const shape_elem_type *shape,
                                         size_t ndim,
                                         size_t size_indices,
                                         size_t values_size);

/**
 * @ingroup BACKEND_API
 * @brief Compute the eigenvalues and right eigenvectors of a square array.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array_in            Input array[size][size]
 * @param [out] result1             The eigenvalues, each repeated according to
 * its multiplicity
 * @param [out] result2             The normalized (unit "length") eigenvectors
 * @param [in]  size                One dimension of square [size][size] array
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_eig_c(DPCTLSyclQueueRef q_ref,
               const void *array_in,
               void *result1,
               void *result2,
               size_t size,
               const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void
    dpnp_eig_c(const void *array_in, void *result1, void *result2, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Compute the eigenvalues of a square array.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array_in            Input array[size][size]
 * @param [out] result1             The eigenvalues, each repeated according to
 * its multiplicity
 * @param [in]  size                One dimension of square [size][size] array
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_eigvals_c(DPCTLSyclQueueRef q_ref,
                   const void *array_in,
                   void *result1,
                   size_t size,
                   const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void
    dpnp_eigvals_c(const void *array_in, void *result1, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Return a 2-D array with ones on the diagonal and zeros elsewhere.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              The eigenvalues, each repeated according to
 * its multiplicity
 * @param [in]  k                   Index of the diagonal
 * @param [in]  shape               Shape of result
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_eye_c(DPCTLSyclQueueRef q_ref,
               void *result,
               int k,
               const shape_elem_type *res_shape,
               const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void
    dpnp_eye_c(void *result, int k, const shape_elem_type *res_shape);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of argsort function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array               Input array with data.
 * @param [out] result              Output array with indices.
 * @param [in]  size                Number of elements in input arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType, typename _idx_DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_argsort_c(DPCTLSyclQueueRef q_ref,
                   void *array,
                   void *result,
                   size_t size,
                   const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType, typename _idx_DataType>
INP_DLLEXPORT void dpnp_argsort_c(void *array, void *result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of searchsorted function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  array               Input array with data.
 * @param [in]  v                   Input values to insert into array.
 * @param [in]  side                Param for choosing a case of searching for
 * elements.
 * @param [in]  arr_size            Number of elements in input arrays.
 * @param [in]  v_size              Number of elements in input values arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType, typename _IndexingType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_searchsorted_c(DPCTLSyclQueueRef q_ref,
                        void *result,
                        const void *array,
                        const void *v,
                        bool side,
                        const size_t arr_size,
                        const size_t v_size,
                        const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType, typename _IndexingType>
INP_DLLEXPORT void dpnp_searchsorted_c(void *result,
                                       const void *array,
                                       const void *v,
                                       bool side,
                                       const size_t arr_size,
                                       const size_t v_size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of sort function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array               Input array with data.
 * @param [out] result              Output array with indices.
 * @param [in]  size                Number of elements in input arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_sort_c(DPCTLSyclQueueRef q_ref,
                void *array,
                void *result,
                size_t size,
                const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_sort_c(void *array, void *result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of cholesky function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array               Input array with data.
 * @param [out] result              Output array.
 * @param [in]  size                Number of elements in input arrays.
 * @param [in]  data_size           Last element of shape arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_cholesky_c(DPCTLSyclQueueRef q_ref,
                    void *array1_in,
                    void *result1,
                    const size_t size,
                    const size_t data_size,
                    const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_cholesky_c(void *array1_in,
                                   void *result1,
                                   const size_t size,
                                   const size_t data_size);

/**
 * @ingroup BACKEND_API
 * @brief correlate function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result_out          Output array.
 * @param [in]  input1_in           First input array.
 * @param [in]  input1_size         Size of first input array.
 * @param [in]  input1_shape        Shape of first input array.
 * @param [in]  input1_shape_ndim   Number of first array dimensions.
 * @param [in]  input2_in           Second input array.
 * @param [in]  input2_size         Shape of second input array.
 * @param [in]  input2_shape        Shape of first input array.
 * @param [in]  input2_shape_ndim   Number of second array dimensions.
 * @param [in]  where               Mask array.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType_output,
          typename _DataType_input1,
          typename _DataType_input2>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_correlate_c(DPCTLSyclQueueRef q_ref,
                     void *result_out,
                     const void *input1_in,
                     const size_t input1_size,
                     const shape_elem_type *input1_shape,
                     const size_t input1_shape_ndim,
                     const void *input2_in,
                     const size_t input2_size,
                     const shape_elem_type *input2_shape,
                     const size_t input2_shape_ndim,
                     const size_t *where,
                     const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType_output,
          typename _DataType_input1,
          typename _DataType_input2>
INP_DLLEXPORT void dpnp_correlate_c(void *result_out,
                                    const void *input1_in,
                                    const size_t input1_size,
                                    const shape_elem_type *input1_shape,
                                    const size_t input1_shape_ndim,
                                    const void *input2_in,
                                    const size_t input2_size,
                                    const shape_elem_type *input2_shape,
                                    const size_t input2_shape_ndim,
                                    const size_t *where);

/**
 * @ingroup BACKEND_API
 * @brief Custom implementation of cov function with math library and PSTL
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array               Input array.
 * @param [out] result              Output array.
 * @param [in]  nrows               Number of rows in input array.
 * @param [in]  ncols               Number of columns in input array.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_cov_c(DPCTLSyclQueueRef q_ref,
               void *array1_in,
               void *result1,
               size_t nrows,
               size_t ncols,
               const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void
    dpnp_cov_c(void *array1_in, void *result1, size_t nrows, size_t ncols);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of det function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array               Input array with data.
 * @param [out] result              Output array.
 * @param [in]  shape               Shape of input array.
 * @param [in]  ndim                Number of elements in shape.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_det_c(DPCTLSyclQueueRef q_ref,
               void *array1_in,
               void *result1,
               shape_elem_type *shape,
               size_t ndim,
               const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_det_c(void *array1_in,
                              void *result1,
                              shape_elem_type *shape,
                              size_t ndim);

/**
 * @ingroup BACKEND_API
 * @brief Construct an array from an index array and a list of arrays to choose
 * from.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result1             Output array.
 * @param [in]  array1_in           Input array with data.
 * @param [in]  choices             Choice arrays.
 * @param [in]  size                Input array size.
 * @param [in]  choices_size        Choices size.
 * @param [in]  choice_size         Choice size.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType1, typename _DataType2>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_choose_c(DPCTLSyclQueueRef q_ref,
                  void *result1,
                  void *array1_in,
                  void **choices,
                  size_t size,
                  size_t choices_size,
                  size_t choice_size,
                  const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType1, typename _DataType2>
INP_DLLEXPORT void dpnp_choose_c(void *result1,
                                 void *array1_in,
                                 void **choices,
                                 size_t size,
                                 size_t choices_size,
                                 size_t choice_size);

/**
 * @ingroup BACKEND_API
 * @brief Extract a diagonal or construct a diagonal array.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array               Input array with data.
 * @param [out] result              Output array.
 * @param [in]  k                   Diagonal in question.
 * @param [in]  shape               Shape of input array.
 * @param [in]  res_shape           Shape of result array.
 * @param [in]  ndim                Number of elements in shape of input array.
 * @param [in]  res_ndim            Number of elements in shape of result array.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_diag_c(DPCTLSyclQueueRef q_ref,
                void *array,
                void *result,
                const int k,
                shape_elem_type *shape,
                shape_elem_type *res_shape,
                const size_t ndim,
                const size_t res_ndim,
                const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_diag_c(void *array,
                               void *result,
                               const int k,
                               shape_elem_type *shape,
                               shape_elem_type *res_shape,
                               const size_t ndim,
                               const size_t res_ndim);

/**
 * @ingroup BACKEND_API
 * @brief Return the indices to access the main diagonal of an array.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result1             Output array.
 * @param [in]  size                Size of array.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_diag_indices_c(DPCTLSyclQueueRef q_ref,
                        void *result1,
                        size_t size,
                        const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_diag_indices_c(void *result1, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of diagonal function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array1_in           Input array with data.
 * @param [in]  input1_size         Input1 data size.
 * @param [out] result1             Output array.
 * @param [in]  offset              Offset of the diagonal from the main
 * diagonal.
 * @param [in]  shape               Shape of input array.
 * @param [in]  res_shape           Shape of output array.
 * @param [in]  res_ndim            Number of elements in shape.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_diagonal_c(DPCTLSyclQueueRef q_ref,
                    void *array1_in,
                    const size_t input1_size,
                    void *result1,
                    const size_t offset,
                    shape_elem_type *shape,
                    shape_elem_type *res_shape,
                    const size_t res_ndim,
                    const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_diagonal_c(void *array1_in,
                                   const size_t input1_size,
                                   void *result1,
                                   const size_t offset,
                                   shape_elem_type *shape,
                                   shape_elem_type *res_shape,
                                   const size_t res_ndim);

/**
 * @ingroup BACKEND_API
 * @brief Implementation of identity function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result1             Output array.
 * @param [in]  n                   Number of rows (and columns) in n x n
 * output.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_identity_c(DPCTLSyclQueueRef q_ref,
                    void *result1,
                    const size_t n,
                    const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_identity_c(void *result1, const size_t n);

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

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of inv function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array1_in           Input array with data.
 * @param [out] result1             Output array.
 * @param [in]  shape               Shape of input array.
 * @param [in]  ndim                Number of elements in shape.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_inv_c(DPCTLSyclQueueRef q_ref,
               void *array1_in,
               void *result1,
               shape_elem_type *shape,
               size_t ndim,
               const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void dpnp_inv_c(void *array1_in,
                              void *result1,
                              shape_elem_type *shape,
                              size_t ndim);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of matrix_rank function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array1_in           Input array with data.
 * @param [out] result1             Output array.
 * @param [in]  shape               Shape of input array.
 * @param [in]  ndim                Number of elements in shape.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_matrix_rank_c(DPCTLSyclQueueRef q_ref,
                       void *array1_in,
                       void *result1,
                       shape_elem_type *shape,
                       size_t ndim,
                       const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_matrix_rank_c(void *array1_in,
                                      void *result1,
                                      shape_elem_type *shape,
                                      size_t ndim);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of max function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array1_in           Input array with data.
 * @param [out] result1             Output array.
 * @param [in]  result_size         Output array size.
 * @param [in]  shape               Shape of input array.
 * @param [in]  ndim                Number of elements in shape.
 * @param [in]  axis                Axis.
 * @param [in]  naxis               Number of elements in axis.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_max_c(DPCTLSyclQueueRef q_ref,
               void *array1_in,
               void *result1,
               const size_t result_size,
               const shape_elem_type *shape,
               size_t ndim,
               const shape_elem_type *axis,
               size_t naxis,
               const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_max_c(void *array1_in,
                              void *result1,
                              const size_t result_size,
                              const shape_elem_type *shape,
                              size_t ndim,
                              const shape_elem_type *axis,
                              size_t naxis);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of mean function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array               Input array with data.
 * @param [out] result              Output array.
 * @param [in]  shape               Shape of input array.
 * @param [in]  ndim                Number of elements in shape.
 * @param [in]  axis                Axis.
 * @param [in]  naxis               Number of elements in axis.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_mean_c(DPCTLSyclQueueRef q_ref,
                void *array,
                void *result,
                const shape_elem_type *shape,
                size_t ndim,
                const shape_elem_type *axis,
                size_t naxis,
                const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void dpnp_mean_c(void *array,
                               void *result,
                               const shape_elem_type *shape,
                               size_t ndim,
                               const shape_elem_type *axis,
                               size_t naxis);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of median function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array               Input array with data.
 * @param [out] result              Output array.
 * @param [in]  shape               Shape of input array.
 * @param [in]  ndim                Number of elements in shape.
 * @param [in]  axis                Axis.
 * @param [in]  naxis               Number of elements in axis.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_median_c(DPCTLSyclQueueRef q_ref,
                  void *array,
                  void *result,
                  const shape_elem_type *shape,
                  size_t ndim,
                  const shape_elem_type *axis,
                  size_t naxis,
                  const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void dpnp_median_c(void *array,
                                 void *result,
                                 const shape_elem_type *shape,
                                 size_t ndim,
                                 const shape_elem_type *axis,
                                 size_t naxis);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of min function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array               Input array with data.
 * @param [out] result              Output array.
 * @param [in]  result_size         Output array size.
 * @param [in]  shape               Shape of input array.
 * @param [in]  ndim                Number of elements in shape.
 * @param [in]  axis                Axis.
 * @param [in]  naxis               Number of elements in axis.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_min_c(DPCTLSyclQueueRef q_ref,
               void *array,
               void *result,
               const size_t result_size,
               const shape_elem_type *shape,
               size_t ndim,
               const shape_elem_type *axis,
               size_t naxis,
               const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_min_c(void *array,
                              void *result,
                              const size_t result_size,
                              const shape_elem_type *shape,
                              size_t ndim,
                              const shape_elem_type *axis,
                              size_t naxis);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of argmax function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array               Input array with data.
 * @param [out] result              Output array with indices.
 * @param [in]  size                Number of elements in input array.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType, typename _idx_DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_argmax_c(DPCTLSyclQueueRef q_ref,
                  void *array,
                  void *result,
                  size_t size,
                  const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType, typename _idx_DataType>
INP_DLLEXPORT void dpnp_argmax_c(void *array, void *result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of argmin function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array               Input array with data.
 * @param [out] result              Output array with indices.
 * @param [in]  size                Number of elements in input array.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType, typename _idx_DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_argmin_c(DPCTLSyclQueueRef q_ref,
                  void *array,
                  void *result,
                  size_t size,
                  const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType, typename _idx_DataType>
INP_DLLEXPORT void dpnp_argmin_c(void *array, void *result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of around function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  input_in            Input array with data.
 * @param [out] result_out          Output array with indices.
 * @param [in]  input_size          Number of elements in input arrays.
 * @param [in]  decimals            Number of decimal places to round. Support
 * only with default value 0.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_around_c(DPCTLSyclQueueRef q_ref,
                  const void *input_in,
                  void *result_out,
                  const size_t input_size,
                  const int decimals,
                  const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_around_c(const void *input_in,
                                 void *result_out,
                                 const size_t input_size,
                                 const int decimals);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of std function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array               Input array with data.
 * @param [out] result              Output array with indices.
 * @param [in]  shape               Shape of input array.
 * @param [in]  ndim                Number of elements in shape.
 * @param [in]  axis                Axis.
 * @param [in]  naxis               Number of elements in axis.
 * @param [in]  ddof                Delta degrees of freedom.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_std_c(DPCTLSyclQueueRef q_ref,
               void *array,
               void *result,
               const shape_elem_type *shape,
               size_t ndim,
               const shape_elem_type *axis,
               size_t naxis,
               size_t ddof,
               const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void dpnp_std_c(void *array,
                              void *result,
                              const shape_elem_type *shape,
                              size_t ndim,
                              const shape_elem_type *axis,
                              size_t naxis,
                              size_t ddof);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of take function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array               Input array with data.
 * @param [in]  array1_size         Input array size.
 * @param [in]  indices             Input array with indices.
 * @param [out] result              Output array.
 * @param [in]  size                Number of elements in the input array.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType, typename _IndecesType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_take_c(DPCTLSyclQueueRef q_ref,
                void *array,
                const size_t array1_size,
                void *indices,
                void *result,
                size_t size,
                const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType, typename _IndecesType>
INP_DLLEXPORT void dpnp_take_c(void *array,
                               const size_t array1_size,
                               void *indices,
                               void *result,
                               size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of trace function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array               Input array with data.
 * @param [out] result              Output array.
 * @param [in]  shape               Shape of input array.
 * @param [in]  ndim                Number of elements in array.shape.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_trace_c(DPCTLSyclQueueRef q_ref,
                 const void *array,
                 void *result,
                 const shape_elem_type *shape,
                 const size_t ndim,
                 const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void dpnp_trace_c(const void *array,
                                void *result,
                                const shape_elem_type *shape,
                                const size_t ndim);

/**
 * @ingroup BACKEND_API
 * @brief An array with ones at and below the given diagonal and zeros
 * elsewhere.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result              Output array.
 * @param [in]  N                   Number of rows in the array.
 * @param [in]  M                   Number of columns in the array.
 * @param [in]  k                   The sub-diagonal at and below which the
 * array is filled.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_tri_c(DPCTLSyclQueueRef q_ref,
               void *result,
               const size_t N,
               const size_t M,
               const int k,
               const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void
    dpnp_tri_c(void *result, const size_t N, const size_t M, const int k);

/**
 * @ingroup BACKEND_API
 * @brief Lower triangle of an array.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array               Input array with data.
 * @param [out] result              Output array.
 * @param [in]  k                   Diagonal above which to zero elements.
 * @param [in]  shape               Shape of input array.
 * @param [in]  res_shape           Shape of result array.
 * @param [in]  ndim                Number of elements in array.shape.
 * @param [in]  res_ndim            Number of elements in res_shape.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_tril_c(DPCTLSyclQueueRef q_ref,
                void *array,
                void *result,
                const int k,
                shape_elem_type *shape,
                shape_elem_type *res_shape,
                const size_t ndim,
                const size_t res_ndim,
                const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_tril_c(void *array,
                               void *result,
                               const int k,
                               shape_elem_type *shape,
                               shape_elem_type *res_shape,
                               const size_t ndim,
                               const size_t res_ndim);

/**
 * @ingroup BACKEND_API
 * @brief Upper triangle of an array.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array               Input array with data.
 * @param [out] result              Output array.
 * @param [in]  k                   Diagonal above which to zero elements.
 * @param [in]  shape               Shape of input array.
 * @param [in]  res_shape           Shape of result array.
 * @param [in]  ndim                Number of elements in array.shape.
 * @param [in]  res_ndim            Number of elements in res_shape.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_triu_c(DPCTLSyclQueueRef q_ref,
                void *array,
                void *result,
                const int k,
                shape_elem_type *shape,
                shape_elem_type *res_shape,
                const size_t ndim,
                const size_t res_ndim,
                const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_triu_c(void *array,
                               void *result,
                               const int k,
                               shape_elem_type *shape,
                               shape_elem_type *res_shape,
                               const size_t ndim,
                               const size_t res_ndim);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of var function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array               Input array with data.
 * @param [out] result              Output array with indices.
 * @param [in]  shape               Shape of input array.
 * @param [in]  ndim                Number of elements in shape.
 * @param [in]  axis                Axis.
 * @param [in]  naxis               Number of elements in axis.
 * @param [in]  ddof                Delta degrees of freedom.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_var_c(DPCTLSyclQueueRef q_ref,
               void *array,
               void *result,
               const shape_elem_type *shape,
               size_t ndim,
               const shape_elem_type *axis,
               size_t naxis,
               size_t ddof,
               const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void dpnp_var_c(void *array,
                              void *result,
                              const shape_elem_type *shape,
                              size_t ndim,
                              const shape_elem_type *axis,
                              size_t naxis,
                              size_t ddof);

/**
 * @ingroup BACKEND_API
 * @brief Implementation of invert function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array1_in           Input array.
 * @param [out] result1             Output array.
 * @param [in]  size                Number of elements in the input array.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_invert_c(DPCTLSyclQueueRef q_ref,
                  void *array1_in,
                  void *result,
                  size_t size,
                  const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_invert_c(void *array1_in, void *result, size_t size);

#define MACRO_2ARG_1TYPE_OP(__name__, __operation__)                           \
    template <typename _DataType>                                              \
    INP_DLLEXPORT DPCTLSyclEventRef __name__(                                  \
        DPCTLSyclQueueRef q_ref, void *result_out, const size_t result_size,   \
        const size_t result_ndim, const shape_elem_type *result_shape,         \
        const shape_elem_type *result_strides, const void *input1_in,          \
        const size_t input1_size, const size_t input1_ndim,                    \
        const shape_elem_type *input1_shape,                                   \
        const shape_elem_type *input1_strides, const void *input2_in,          \
        const size_t input2_size, const size_t input2_ndim,                    \
        const shape_elem_type *input2_shape,                                   \
        const shape_elem_type *input2_strides, const size_t *where,            \
        const DPCTLEventVectorRef dep_event_vec_ref);                          \
                                                                               \
    template <typename _DataType>                                              \
    INP_DLLEXPORT void __name__(                                               \
        void *result_out, const size_t result_size, const size_t result_ndim,  \
        const shape_elem_type *result_shape,                                   \
        const shape_elem_type *result_strides, const void *input1_in,          \
        const size_t input1_size, const size_t input1_ndim,                    \
        const shape_elem_type *input1_shape,                                   \
        const shape_elem_type *input1_strides, const void *input2_in,          \
        const size_t input2_size, const size_t input2_ndim,                    \
        const shape_elem_type *input2_shape,                                   \
        const shape_elem_type *input2_strides, const size_t *where);

#include <dpnp_gen_2arg_1type_tbl.hpp>

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

#define MACRO_1ARG_2TYPES_OP(__name__, __operation1__, __operation2__)         \
    template <typename _DataType_input, typename _DataType_output>             \
    INP_DLLEXPORT DPCTLSyclEventRef __name__(                                  \
        DPCTLSyclQueueRef q_ref, void *result_out, const size_t result_size,   \
        const size_t result_ndim, const shape_elem_type *result_shape,         \
        const shape_elem_type *result_strides, const void *input1_in,          \
        const size_t input1_size, const size_t input1_ndim,                    \
        const shape_elem_type *input1_shape,                                   \
        const shape_elem_type *input1_strides, const size_t *where,            \
        const DPCTLEventVectorRef dep_event_vec_ref);                          \
                                                                               \
    template <typename _DataType_input, typename _DataType_output>             \
    INP_DLLEXPORT void __name__(                                               \
        void *result_out, const size_t result_size, const size_t result_ndim,  \
        const shape_elem_type *result_shape,                                   \
        const shape_elem_type *result_strides, const void *input1_in,          \
        const size_t input1_size, const size_t input1_ndim,                    \
        const shape_elem_type *input1_shape,                                   \
        const shape_elem_type *input1_strides, const size_t *where);

#include <dpnp_gen_1arg_2type_tbl.hpp>

#define MACRO_2ARG_2TYPES_LOGIC_OP(__name__, __operation__)                    \
    template <typename _DataType_output, typename _DataType_input1,            \
              typename _DataType_input2>                                       \
    INP_DLLEXPORT DPCTLSyclEventRef __name__(                                  \
        DPCTLSyclQueueRef q_ref, void *result_out, const size_t result_size,   \
        const size_t result_ndim, const shape_elem_type *result_shape,         \
        const shape_elem_type *result_strides, const void *input1_in,          \
        const size_t input1_size, const size_t input1_ndim,                    \
        const shape_elem_type *input1_shape,                                   \
        const shape_elem_type *input1_strides, const void *input2_in,          \
        const size_t input2_size, const size_t input2_ndim,                    \
        const shape_elem_type *input2_shape,                                   \
        const shape_elem_type *input2_strides, const size_t *where,            \
        const DPCTLEventVectorRef dep_event_vec_ref);

#include <dpnp_gen_2arg_2type_tbl.hpp>

#define MACRO_2ARG_3TYPES_OP(__name__, __operation__, __vec_operation__,       \
                             __vec_types__, __mkl_operation__, __mkl_types__)  \
    template <typename _DataType_output, typename _DataType_input1,            \
              typename _DataType_input2>                                       \
    INP_DLLEXPORT DPCTLSyclEventRef __name__(                                  \
        DPCTLSyclQueueRef q_ref, void *result_out, const size_t result_size,   \
        const size_t result_ndim, const shape_elem_type *result_shape,         \
        const shape_elem_type *result_strides, const void *input1_in,          \
        const size_t input1_size, const size_t input1_ndim,                    \
        const shape_elem_type *input1_shape,                                   \
        const shape_elem_type *input1_strides, const void *input2_in,          \
        const size_t input2_size, const size_t input2_ndim,                    \
        const shape_elem_type *input2_shape,                                   \
        const shape_elem_type *input2_strides, const size_t *where,            \
        const DPCTLEventVectorRef dep_event_vec_ref);                          \
                                                                               \
    template <typename _DataType_output, typename _DataType_input1,            \
              typename _DataType_input2>                                       \
    INP_DLLEXPORT void __name__(                                               \
        void *result_out, const size_t result_size, const size_t result_ndim,  \
        const shape_elem_type *result_shape,                                   \
        const shape_elem_type *result_strides, const void *input1_in,          \
        const size_t input1_size, const size_t input1_ndim,                    \
        const shape_elem_type *input1_shape,                                   \
        const shape_elem_type *input1_strides, const void *input2_in,          \
        const size_t input2_size, const size_t input2_ndim,                    \
        const shape_elem_type *input2_shape,                                   \
        const shape_elem_type *input2_strides, const size_t *where);

#include <dpnp_gen_2arg_3type_tbl.hpp>

/**
 * @ingroup BACKEND_API
 * @brief fill_diagonal function.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array1_in           Input array.
 * @param [in]  val                 Value to write on the diagonal.
 * @param [in]  shape               Input shape.
 * @param [in]  ndim                Number of elements in shape.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_fill_diagonal_c(DPCTLSyclQueueRef q_ref,
                         void *array1_in,
                         void *val,
                         shape_elem_type *shape,
                         const size_t ndim,
                         const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_fill_diagonal_c(void *array1_in,
                                        void *val,
                                        shape_elem_type *shape,
                                        const size_t ndim);

/**
 * @ingroup BACKEND_API
 * @brief floor_divide function.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result_out          Output array.
 * @param [in]  input1_in           First input array.
 * @param [in]  input1_size         Size of first input array.
 * @param [in]  input1_shape        Shape of first input array.
 * @param [in]  input1_shape_ndim   Number of first array dimensions.
 * @param [in]  input2_in           Second input array.
 * @param [in]  input2_size         Shape of second input array.
 * @param [in]  input2_shape        Shape of first input array.
 * @param [in]  input2_shape_ndim   Number of second array dimensions.
 * @param [in]  where               Mask array.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType_input1,
          typename _DataType_input2,
          typename _DataType_output>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_floor_divide_c(DPCTLSyclQueueRef q_ref,
                        void *result_out,
                        const void *input1_in,
                        const size_t input1_size,
                        const shape_elem_type *input1_shape,
                        const size_t input1_shape_ndim,
                        const void *input2_in,
                        const size_t input2_size,
                        const shape_elem_type *input2_shape,
                        const size_t input2_shape_ndim,
                        const size_t *where,
                        const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType_input1,
          typename _DataType_input2,
          typename _DataType_output>
INP_DLLEXPORT void dpnp_floor_divide_c(void *result_out,
                                       const void *input1_in,
                                       const size_t input1_size,
                                       const shape_elem_type *input1_shape,
                                       const size_t input1_shape_ndim,
                                       const void *input2_in,
                                       const size_t input2_size,
                                       const shape_elem_type *input2_shape,
                                       const size_t input2_shape_ndim,
                                       const size_t *where);

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
 * @brief remainder function.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [out] result_out          Output array.
 * @param [in]  input1_in           First input array.
 * @param [in]  input1_size         Size of first input array.
 * @param [in]  input1_shape        Shape of first input array.
 * @param [in]  input1_shape_ndim   Number of first array dimensions.
 * @param [in]  input2_in           Second input array.
 * @param [in]  input2_size         Shape of second input array.
 * @param [in]  input2_shape        Shape of first input array.
 * @param [in]  input2_shape_ndim   Number of second array dimensions.
 * @param [in]  where               Mask array.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType_output,
          typename _DataType_input1,
          typename _DataType_input2>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_remainder_c(DPCTLSyclQueueRef q_ref,
                     void *result_out,
                     const void *input1_in,
                     const size_t input1_size,
                     const shape_elem_type *input1_shape,
                     const size_t input1_shape_ndim,
                     const void *input2_in,
                     const size_t input2_size,
                     const shape_elem_type *input2_shape,
                     const size_t input2_shape_ndim,
                     const size_t *where,
                     const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType_output,
          typename _DataType_input1,
          typename _DataType_input2>
INP_DLLEXPORT void dpnp_remainder_c(void *result_out,
                                    const void *input1_in,
                                    const size_t input1_size,
                                    const shape_elem_type *input1_shape,
                                    const size_t input1_shape_ndim,
                                    const void *input2_in,
                                    const size_t input2_size,
                                    const shape_elem_type *input2_shape,
                                    const size_t input2_shape_ndim,
                                    const size_t *where);

/**
 * @ingroup BACKEND_API
 * @brief repeat elements of an array.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array_in            Input array.
 * @param [out] result              Output array.
 * @param [in]  repeats             The number of repetitions for each element.
 * @param [in]  size                Number of elements in input arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_repeat_c(DPCTLSyclQueueRef q_ref,
                  const void *array_in,
                  void *result,
                  const size_t repeats,
                  const size_t size,
                  const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void dpnp_repeat_c(const void *array_in,
                                 void *result,
                                 const size_t repeats,
                                 const size_t size);

/**
 * @ingroup BACKEND_API
 * @brief transpose function. Permute axes of the input to the output with
 * elements permutation.
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array1_in           Input array.
 * @param [in]  input_shape         Input shape.
 * @param [in]  result_shape        Output shape.
 * @param [in]  permute_axes        Order of axis by it's id as it should be
 * presented in output.
 * @param [in]  ndim                Number of elements in shapes and axes.
 * @param [out] result1             Output array.
 * @param [in]  size                Number of elements in input arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 */
template <typename _DataType>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_elemwise_transpose_c(DPCTLSyclQueueRef q_ref,
                              void *array1_in,
                              const shape_elem_type *input_shape,
                              const shape_elem_type *result_shape,
                              const shape_elem_type *permute_axes,
                              size_t ndim,
                              void *result1,
                              size_t size,
                              const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType>
INP_DLLEXPORT void
    dpnp_elemwise_transpose_c(void *array1_in,
                              const shape_elem_type *input_shape,
                              const shape_elem_type *result_shape,
                              const shape_elem_type *permute_axes,
                              size_t ndim,
                              void *result1,
                              size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Custom implementation of trapz function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array1_in           First input array.
 * @param [in]  array2_in           Second input array.
 * @param [out] result1             Output array.
 * @param [in]  dx                  The spacing between sample points.
 * @param [in]  array1_size         Number of elements in first input array.
 * @param [in]  array2_size         Number of elements in second input arrays.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 *
 */
template <typename _DataType_input1,
          typename _DataType_input2,
          typename _DataType_output>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_trapz_c(DPCTLSyclQueueRef q_ref,
                 const void *array1_in,
                 const void *array2_in,
                 void *result1,
                 double dx,
                 size_t array1_size,
                 size_t array2_size,
                 const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType_input1,
          typename _DataType_input2,
          typename _DataType_output>
INP_DLLEXPORT void dpnp_trapz_c(const void *array1_in,
                                const void *array2_in,
                                void *result1,
                                double dx,
                                size_t array1_size,
                                size_t array2_size);

/**
 * @ingroup BACKEND_API
 * @brief Implementation of vander function
 *
 * @param [in]  q_ref               Reference to SYCL queue.
 * @param [in]  array_in            Input array.
 * @param [out] result              Output array.
 * @param [in]  size_in             Number of elements in the input array.
 * @param [in]  N                   Number of columns in the output.
 * @param [in]  increasing          Order of the powers of the columns.
 * @param [in]  dep_event_vec_ref   Reference to vector of SYCL events.
 *
 */
template <typename _DataType_input, typename _DataType_output>
INP_DLLEXPORT DPCTLSyclEventRef
    dpnp_vander_c(DPCTLSyclQueueRef q_ref,
                  const void *array1_in,
                  void *result1,
                  const size_t size_in,
                  const size_t N,
                  const int increasing,
                  const DPCTLEventVectorRef dep_event_vec_ref);

template <typename _DataType_input, typename _DataType_output>
INP_DLLEXPORT void dpnp_vander_c(const void *array1_in,
                                 void *result1,
                                 const size_t size_in,
                                 const size_t N,
                                 const int increasing);

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
