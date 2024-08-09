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
