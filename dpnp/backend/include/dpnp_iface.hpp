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
#ifndef BACKEND_IFACE_H // Cython compatibility
#define BACKEND_IFACE_H

#include <cstdint>
#include <vector>

#include "dpnp_iface_fft.hpp"
#include "dpnp_iface_random.hpp"

#ifdef _WIN32
#define INP_DLLEXPORT __declspec(dllexport)
#else
#define INP_DLLEXPORT
#endif

/**
 * @defgroup BACKEND_API Backend C++ library interface API
 * @{
 * This section describes Backend API.
 * @}
 */

/**
 * @ingroup BACKEND_API
 * @brief SYCL queue initialization selector.
 *
 * The structure defines the parameters that are used for the library initialization
 * by @ref dpnp_queue_initialize_c "dpnp_queue_initialize".
 */
enum class QueueOptions : uint32_t
{
    CPU_SELECTOR, /**< CPU side execution mode */
    GPU_SELECTOR, /**< Intel GPU side execution mode */
    AUTO_SELECTOR /**< Automatic selection based on environment variable with @ref CPU_SELECTOR default */
};

/**
 * @ingroup BACKEND_API
 * @brief SYCL queue initialization.
 *
 * Global SYCL queue initialization.
 *
 * @param [in]  selector       Select type @ref QueueOptions of the SYCL queue. Default @ref AUTO_SELECTOR
 */
INP_DLLEXPORT void dpnp_queue_initialize_c(QueueOptions selector = QueueOptions::AUTO_SELECTOR);

/**
 * @ingroup BACKEND_API
 * @brief SYCL queue device status.
 *
 * Return 1 if current @ref queue is related to cpu or host device. return 0 otherwise.
 */
INP_DLLEXPORT size_t dpnp_queue_is_cpu_c();

/**
 * @ingroup BACKEND_API
 * @brief SYCL queue memory allocation.
 *
 * Memory allocation on the SYCL backend.
 *
 * @param [in]  size_in_bytes  Number of bytes for requested memory allocation.
 *
 * @return  A pointer to newly created memory on @ref dpnp_queue_initialize_c "initialized SYCL device".
 */
INP_DLLEXPORT char* dpnp_memory_alloc_c(size_t size_in_bytes);

INP_DLLEXPORT void dpnp_memory_free_c(void* ptr);
void dpnp_memory_memcpy_c(void* dst, const void* src, size_t size_in_bytes);

/**
 * @ingroup BACKEND_API
 * @brief Test whether all array elements along a given axis evaluate to True.
 *
 * @param [in]  array       Input array.
 * @param [out] result      Output array.
 * @param [in]  size        Number of input elements in `array`.
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void dpnp_all_c(const void* array, void* result, const size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Test whether all array elements along a given axis evaluate to True.
 *
 * @param [in]  array1_in      First input array.
 * @param [in]  array2_in      Second input array.
 * @param [out] result1        Output array.
 * @param [in]  size           Number of input elements in `array`.
 * @param [in]  rtol_val       The relative tolerance parameter.
 * @param [in]  atol_val       The absolute tolerance parameter.
 */
template <typename _DataType1, typename _DataType2, typename _ResultType>
INP_DLLEXPORT void dpnp_allclose_c(
    const void* array1_in, const void* array2_in, void* result1, const size_t size, double rtol, double atol);

/**
 * @ingroup BACKEND_API
 * @brief Test whether any array element along a given axis evaluates to True.
 *
 * @param [in]  array       Input array.
 * @param [out] result      Output array.
 * @param [in]  size        Number of input elements in `array`.
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void dpnp_any_c(const void* array, void* result, const size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Array initialization
 *
 * Input array, step based, initialization procedure.
 *
 * @param [in]  start     Start of initialization sequence
 * @param [in]  step      Step for initialization sequence
 * @param [out] result1   Output array.
 * @param [in]  size      Number of elements in input arrays.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_arange_c(size_t start, size_t step, void* result1, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Copy of the array, cast to a specified type.
 *
 * @param [in]  array       Input array.
 * @param [out] result      Output array.
 * @param [in]  size        Number of input elements in `array`.
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void dpnp_astype_c(const void* array, void* result, const size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Implementation of full function
 *
 * @param [in]  array_in  Input one-element array.
 * @param [out] result    Output array.
 * @param [in]  size      Number of elements in the output array.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_full_c(void* array_in, void* result, const size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Implementation of full_like function
 *
 * @param [in]  array_in  Input one-element array.
 * @param [out] result    Output array.
 * @param [in]  size      Number of elements in the output array.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_full_like_c(void* array_in, void* result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Matrix multiplication.
 *
 * Matrix multiplication procedure.
 *
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
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_matmul_c(void* result_out,
                                 const size_t result_size,
                                 const size_t result_ndim,
                                 const size_t* result_shape,
                                 const size_t* result_strides,
                                 const void* input1_in,
                                 const size_t input1_size,
                                 const size_t input1_ndim,
                                 const size_t* input1_shape,
                                 const size_t* input1_strides,
                                 const void* input2_in,
                                 const size_t input2_size,
                                 const size_t input2_ndim,
                                 const size_t* input2_shape,
                                 const size_t* input2_strides);

/**
 * @ingroup BACKEND_API
 * @brief Compute the variance along the specified axis, while ignoring NaNs.
 *
 * @param [in]  array     Input array.
 * @param [in]  mask_arr  Input mask array when elem is nan.
 * @param [out] result    Output array.
 * @param [in]  result_size    Output array size.
 * @param [in]  size      Number of elements in input arrays.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_nanvar_c(void* array, void* mask_arr, void* result, const size_t result_size, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Return the indices of the elements that are non-zero.
 *
 * @param [in]  array1    Input array.
 * @param [out] result1   Output array.
 * @param [in]  result_size   Output array size.
 * @param [in]  shape     Shape of input array.
 * @param [in]  ndim      Number of elements in shape.
 * @param [in]  j         Number input array.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_nonzero_c(const void* array1,
                                  void* result1,
                                  const size_t result_size,
                                  const size_t* shape,
                                  const size_t ndim,
                                  const size_t j);

/**
 * @ingroup BACKEND_API
 * @brief absolute function.
 *
 * @param [in]  input1_in    Input array.
 * @param [out] result1      Output array.
 * @param [in]  size         Number of elements in input arrays.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_elemwise_absolute_c(const void* input1_in, void* result1, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Custom implementation of dot function
 *
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
 */
template <typename _DataType_output, typename _DataType_input1, typename _DataType_input2>
INP_DLLEXPORT void dpnp_dot_c(void* result_out,
                              const size_t result_size,
                              const size_t result_ndim,
                              const size_t* result_shape,
                              const size_t* result_strides,
                              const void* input1_in,
                              const size_t input1_size,
                              const size_t input1_ndim,
                              const size_t* input1_shape,
                              const size_t* input1_strides,
                              const void* input2_in,
                              const size_t input2_size,
                              const size_t input2_ndim,
                              const size_t* input2_shape,
                              const size_t* input2_strides);

/**
 * @ingroup BACKEND_API
 * @brief Custom implementation of cross function
 *
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
 * @param [out] result1             Output array.
 * @param [in]  size                Number of elements in input arrays.
 */
template <typename _DataType_output, typename _DataType_input1, typename _DataType_input2>
INP_DLLEXPORT void dpnp_cross_c(void* result_out,
                                const void* input1_in,
                                const size_t input1_size,
                                const size_t* input1_shape,
                                const size_t input1_shape_ndim,
                                const void* input2_in,
                                const size_t input2_size,
                                const size_t* input2_shape,
                                const size_t input2_shape_ndim,
                                const size_t* where);

/**
 * @ingroup BACKEND_API
 * @brief Custom implementation of cumprod function
 *
 * @param [in]  array1_in  Input array.
 * @param [out] result1    Output array.
 * @param [in]  size       Number of elements in input arrays.
 *
 */
template <typename _DataType_input, typename _DataType_output>
INP_DLLEXPORT void dpnp_cumprod_c(void* array1_in, void* result1, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Custom implementation of cumsum function
 *
 * @param [in]  array1_in  Input array.
 * @param [out] result1    Output array.
 * @param [in]  size       Number of elements in input arrays.
 *
 */
template <typename _DataType_input, typename _DataType_output>
INP_DLLEXPORT void dpnp_cumsum_c(void* array1_in, void* result1, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Compute summary of input array elements.
 *
 * Input array is expected as @ref _DataType_input type and assume result as @ref _DataType_output type.
 * The function creates no memory.
 *
 * Empty @ref input_shape means scalar.
 *
 * @param [out] result_out        Output array pointer. @ref _DataType_output type is expected
 * @param [in]  input_in          Input array pointer. @ref _DataType_input type is expected
 * @param [in]  input_shape       Shape of @ref input_in
 * @param [in]  input_shape_ndim  Number of elements in @ref input_shape
 * @param [in]  axes              Array of axes to apply to @ref input_shape
 * @param [in]  axes_ndim         Number of elements in @ref axes
 * @param [in]  initial           Pointer to initial value for the algorithm. @ref _DataType_input is expected
 * @param [in]  where             mask array
 */
template <typename _DataType_output, typename _DataType_input>
INP_DLLEXPORT void dpnp_sum_c(void* result_out,
                              const void* input_in,
                              const size_t* input_shape,
                              const size_t input_shape_ndim,
                              const long* axes,
                              const size_t axes_ndim,
                              const void* initial,
                              const long* where);

/**
 * @ingroup BACKEND_API
 * @brief Custom implementation of count_nonzero function
 *
 * @param [in]  array1_in     Input array.
 * @param [out] result1_out   Output array.
 * @param [in]  size          Number of elements in input arrays.
 *
 */
template <typename _DataType_input, typename _DataType_output>
INP_DLLEXPORT void dpnp_count_nonzero_c(void* array1_in, void* result1_out, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Place of array elements
 *
 * @param [in]  array       Input array.
 * @param [in]  array2      Copy input array.
 * @param [out]  result     Result array.
 * @param [in]  kth         Element index to partition by.
 * @param [in]  shape       Shape of input array.
 * @param [in]  ndim        Number of elements in shape.
 */
template <typename _DataType>
INP_DLLEXPORT void
    dpnp_partition_c(void* array, void* array2, void* result, const size_t kth, const size_t* shape, const size_t ndim);

/**
 * @ingroup BACKEND_API
 * @brief Place of array elements
 *
 * @param [in]  arr         Input array.
 * @param [in]  mask        Mask array.
 * @param [in]  vals        Vals array.
 * @param [in]  arr_size    Number of input elements in `arr`.
 * @param [in]  vals_size   Number of input elements in `vals`.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_place_c(void* arr, long* mask, void* vals, const size_t arr_size, const size_t vals_size);

/**
 * @ingroup BACKEND_API
 * @brief Compute Product of input array elements.
 *
 * Input array is expected as @ref _DataType_input type and assume result as @ref _DataType_output type.
 * The function creates no memory.
 *
 * Empty @ref input_shape means scalar.
 *
 * @param [out] result_out        Output array pointer. @ref _DataType_output type is expected
 * @param [in]  input_in          Input array pointer. @ref _DataType_input type is expected
 * @param [in]  input_shape       Shape of @ref input_in
 * @param [in]  input_shape_ndim  Number of elements in @ref input_shape
 * @param [in]  axes              Array of axes to apply to @ref input_shape
 * @param [in]  axes_ndim         Number of elements in @ref axes
 * @param [in]  initial           Pointer to initial value for the algorithm. @ref _DataType_input is expected
 * @param [in]  where             mask array
 */
template <typename _DataType_output, typename _DataType_input>
INP_DLLEXPORT void dpnp_prod_c(void* result_out,
                               const void* input_in,
                               const size_t* input_shape,
                               const size_t input_shape_ndim,
                               const long* axes,
                               const size_t axes_ndim,
                               const void* initial,
                               const long* where);

/**
 * @ingroup BACKEND_API
 * @brief Product of array elements
 *
 * @param [in]  array       Input array.
 * @param [in]  ind         Target indices, interpreted as integers.
 * @param [in]  v           Values to place in array at target indices.
 * @param [in]  size        Number of input elements in `array`.
 * @param [in]  size_ind    Number of input elements in `ind`.
 * @param [in]  size_v      Number of input elements in `v`.
 */
template <typename _DataType, typename _IndecesType, typename _ValueType>
INP_DLLEXPORT void
    dpnp_put_c(void* array, void* ind, void* v, const size_t size, const size_t size_ind, const size_t size_v);

/**
 * @ingroup BACKEND_API
 * @brief Product of array elements
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_put_along_axis_c(void* arr_in,
                                         long* indices_in,
                                         void* values_in,
                                         size_t axis,
                                         const size_t* shape,
                                         size_t ndim,
                                         size_t size_indices,
                                         size_t values_size);

/**
 * @ingroup BACKEND_API
 * @brief Compute the eigenvalues and right eigenvectors of a square array.
 *
 * @param [in]  array_in  Input array[size][size]
 * @param [out] result1   The eigenvalues, each repeated according to its multiplicity
 * @param [out] result2   The normalized (unit "length") eigenvectors
 * @param [in]  size      One dimension of square [size][size] array
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void dpnp_eig_c(const void* array_in, void* result1, void* result2, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Compute the eigenvalues of a square array.
 *
 * @param [in]  array_in  Input array[size][size]
 * @param [out] result1   The eigenvalues, each repeated according to its multiplicity
 * @param [in]  size      One dimension of square [size][size] array
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void dpnp_eigvals_c(const void* array_in, void* result1, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of argsort function
 *
 * @param [in]  array   Input array with data.
 * @param [out] result  Output array with indeces.
 * @param [in]  size    Number of elements in input arrays.
 */
template <typename _DataType, typename _idx_DataType>
INP_DLLEXPORT void dpnp_argsort_c(void* array, void* result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of searchsorted function
 *
 * @param [out] result      Output array.
 * @param [in]  array       Input array with data.
 * @param [in]  v           Input values to insert into array.
 * @param [in]  side        Param for choosing a case of searching for elements.
 * @param [in]  arr_size    Number of elements in input arrays.
 * @param [in]  v_size      Number of elements in input values arrays.
 */
template <typename _DataType, typename _IndexingType>
INP_DLLEXPORT void dpnp_searchsorted_c(
    void* result, const void* array, const void* v, bool side, const size_t arr_size, const size_t v_size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of sort function
 *
 * @param [in]  array   Input array with data.
 * @param [out] result  Output array with indeces.
 * @param [in]  size    Number of elements in input arrays.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_sort_c(void* array, void* result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of cholesky function
 *
 * @param [in]  array       Input array with data.
 * @param [out] result      Output array.
 * @param [in]  size        Number of elements in input arrays.
 * @param [in]  data_size   Last element of shape arrays.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_cholesky_c(void* array1_in, void* result1, const size_t size, const size_t data_size);

/**
 * @ingroup BACKEND_API
 * @brief correlate function
 *
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
 * @param [out] result1             Output array.
 * @param [in]  size                Number of elements in input arrays.
 */
template <typename _DataType_output, typename _DataType_input1, typename _DataType_input2>
INP_DLLEXPORT void dpnp_correlate_c(void* result_out,
                                    const void* input1_in,
                                    const size_t input1_size,
                                    const size_t* input1_shape,
                                    const size_t input1_shape_ndim,
                                    const void* input2_in,
                                    const size_t input2_size,
                                    const size_t* input2_shape,
                                    const size_t input2_shape_ndim,
                                    const size_t* where);

/**
 * @ingroup BACKEND_API
 * @brief Custom implementation of cov function with math library and PSTL
 *
 * @param [in]  array       Input array.
 * @param [out] result      Output array.
 * @param [in]  nrows       Number of rows in input array.
 * @param [in]  ncols       Number of columns in input array.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_cov_c(void* array1_in, void* result1, size_t nrows, size_t ncols);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of det function
 *
 * @param [in]  array   Input array with data.
 * @param [out] result  Output array.
 * @param [in]  shape   Shape of input array.
 * @param [in]  ndim    Number of elements in shape.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_det_c(void* array1_in, void* result1, size_t* shape, size_t ndim);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of take function
 *
 * @param [out] result        Output array.
 * @param [in]  array         Input array with data.
 * @param [in]  choices       Choice arrays.
 * @param [in]  size          Input array size.
 * @param [in]  choices_size  Choices size.
 * @param [in]  choice_size  Choices size.
 */
template <typename _DataType1, typename _DataType2>
INP_DLLEXPORT void
    dpnp_choose_c(void* result1, void* array1_in, void** choices, size_t size, size_t choices_size, size_t choice_size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of det function
 *
 * @param [in]  array          Input array with data.
 * @param [out] result         Output array.
 * @param [in]  k              Diagonal in question.
 * @param [in]  shape          Shape of input array.
 * @param [in]  shape_result   Shape of result array.
 * @param [in]  ndim           Number of elements in shape.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_diag_c(
    void* array, void* result, const int k, size_t* shape, size_t* res_shape, const size_t ndim, const size_t res_ndim);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of diagonal function
 *
 * @param [out] result      Output array.
 * @param [in]  size        Size of array.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_diag_indices_c(void* result1, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of diagonal function
 *
 * @param [in]  array   Input array with data.
 * @param [in]  input1_size   Input1 data size.
 * @param [out] result  Output array.
 * @param [in]  offset  Offset of the diagonal from the main diagonal.
 * @param [in]  shape   Shape of input array.
 * @param [in]  shape   Shape of output array.
 * @param [in]  ndim    Number of elements in shape.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_diagonal_c(void* array1_in,
                                   const size_t input1_size,
                                   void* result1,
                                   const size_t offset,
                                   size_t* shape,
                                   size_t* res_shape,
                                   const size_t res_ndim);

/**
 * @ingroup BACKEND_API
 * @brief Implementation of identity function
 *
 * @param [out] result1   Output array.
 * @param [in]  n         Number of rows (and columns) in n x n output.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_identity_c(void* result1, const size_t n);

/**
 * @ingroup BACKEND_API
 * @brief implementation of creating filled with value array function
 *
 * @param [out] result  Output array.
 * @param [in]  value   Value in array.
 * @param [in]  size    Number of elements in array.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_initval_c(void* result1, void* value, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of inv function
 *
 * @param [in]  array   Input array with data.
 * @param [out] result  Output array.
 * @param [in]  shape   Shape of input array.
 * @param [in]  ndim    Number of elements in shape.
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void dpnp_inv_c(void* array1_in, void* result1, size_t* shape, size_t ndim);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of matrix_rank function
 *
 * @param [in]  array   Input array with data.
 * @param [out] result  Output array.
 * @param [in]  shape   Shape of input array.
 * @param [in]  ndim    Number of elements in shape.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_matrix_rank_c(void* array1_in, void* result1, size_t* shape, size_t ndim);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of max function
 *
 * @param [in]  array   Input array with data.
 * @param [out] result  Output array.
 * @param [in]  result_size  Output array size.
 * @param [in]  shape   Shape of input array.
 * @param [in]  ndim    Number of elements in shape.
 * @param [in]  axis    Axis.
 * @param [in]  naxis   Number of elements in axis.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_max_c(void* array1_in,
                              void* result1,
                              const size_t result_size,
                              const size_t* shape,
                              size_t ndim,
                              const size_t* axis,
                              size_t naxis);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of mean function
 *
 * @param [in]  array   Input array with data.
 * @param [out] result  Output array.
 * @param [in]  shape   Shape of input array.
 * @param [in]  ndim    Number of elements in shape.
 * @param [in]  axis    Axis.
 * @param [in]  naxis   Number of elements in axis.
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void
    dpnp_mean_c(void* array, void* result, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of median function
 *
 * @param [in]  array   Input array with data.
 * @param [out] result  Output array.
 * @param [in]  shape   Shape of input array.
 * @param [in]  ndim    Number of elements in shape.
 * @param [in]  axis    Axis.
 * @param [in]  naxis   Number of elements in axis.
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void
    dpnp_median_c(void* array, void* result, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of min function
 *
 * @param [in]  array   Input array with data.
 * @param [out] result  Output array.
 * @param [in]  result_size  Output array size.
 * @param [in]  shape   Shape of input array.
 * @param [in]  ndim    Number of elements in shape.
 * @param [in]  axis    Axis.
 * @param [in]  naxis   Number of elements in axis.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_min_c(void* array,
                              void* result,
                              const size_t result_size,
                              const size_t* shape,
                              size_t ndim,
                              const size_t* axis,
                              size_t naxis);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of argmax function
 *
 * @param [in]  array   Input array with data.
 * @param [out] result  Output array with indeces.
 * @param [in]  size    Number of elements in input array.
 */
template <typename _DataType, typename _idx_DataType>
INP_DLLEXPORT void dpnp_argmax_c(void* array, void* result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of argmin function
 *
 * @param [in]  array   Input array with data.
 * @param [out] result  Output array with indeces.
 * @param [in]  size    Number of elements in input array.
 */
template <typename _DataType, typename _idx_DataType>
INP_DLLEXPORT void dpnp_argmin_c(void* array, void* result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of around function
 *
 * @param [in]  input_in     Input array with data.
 * @param [out] result_out   Output array with indeces.
 * @param [in]  input_size   Number of elements in input arrays.
 * @param [in]  decimals     Number of decimal places to round. Support only with default value 0.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_around_c(const void* input_in, void* result_out, const size_t input_size, const int decimals);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of std function
 *
 * @param [in]  array   Input array with data.
 * @param [out] result  Output array with indeces.
 * @param [in]  shape   Shape of input array.
 * @param [in]  ndim    Number of elements in shape.
 * @param [in]  axis    Axis.
 * @param [in]  naxis   Number of elements in axis.
 * @param [in]  ddof    Delta degrees of freedom.
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void dpnp_std_c(
    void* array, void* result, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis, size_t ddof);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of take function
 *
 * @param [in]  array   Input array with data.
 * @param [in]  array1_size   Input array size.
 * @param [in]  indices Input array with indices.
 * @param [out] result  Output array.
 * @param [in]  size    Number of elements in the input array.
 */
template <typename _DataType, typename _IndecesType>
INP_DLLEXPORT void dpnp_take_c(void* array, const size_t array1_size, void* indices, void* result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of trace function
 *
 * @param [in]  array      Input array with data.
 * @param [out] result     Output array.
 * @param [in]  shape      Shape of input array.
 * @param [in]  ndim       Number of elements in array.shape.
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void dpnp_trace_c(const void* array, void* result, const size_t* shape, const size_t ndim);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of take function
 *
 * @param [out] result  Output array.
 * @param [in]  N       Number of rows in the array.
 * @param [in]  M       Number of columns in the array.
 * @param [in]  k       The sub-diagonal at and below which the array is filled.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_tri_c(void* result, const size_t N, const size_t M, const int k);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of take function
 *
 * @param [in]  array      Input array with data.
 * @param [out] result     Output array.
 * @param [in]  k          Diagonal above which to zero elements.
 * @param [in]  shape      Shape of input array.
 * @param [in]  res_shape  Shape of result array.
 * @param [in]  ndim       Number of elements in array.shape.
 * @param [in]  res_ndim   Number of elements in res_shape.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_tril_c(
    void* array, void* result, const int k, size_t* shape, size_t* res_shape, const size_t ndim, const size_t res_ndim);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of take function
 *
 * @param [in]  array      Input array with data.
 * @param [out] result     Output array.
 * @param [in]  k          Diagonal above which to zero elements.
 * @param [in]  shape      Shape of input array.
 * @param [in]  res_shape  Shape of result array.
 * @param [in]  ndim       Number of elements in array.shape.
 * @param [in]  res_ndim   Number of elements in res_shape.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_triu_c(
    void* array, void* result, const int k, size_t* shape, size_t* res_shape, const size_t ndim, const size_t res_ndim);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of var function
 *
 * @param [in]  array   Input array with data.
 * @param [out] result  Output array with indeces.
 * @param [in]  shape   Shape of input array.
 * @param [in]  ndim    Number of elements in shape.
 * @param [in]  axis    Axis.
 * @param [in]  naxis   Number of elements in axis.
 * @param [in]  ddof    Delta degrees of freedom.
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void dpnp_var_c(
    void* array, void* result, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis, size_t ddof);

/**
 * @ingroup BACKEND_API
 * @brief Implementation of invert function
 *
 * @param [in]  array1_in  Input array.
 * @param [out] result1    Output array.
 * @param [in]  size       Number of elements in the input array.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_invert_c(void* array1_in, void* result, size_t size);

#define MACRO_2ARG_1TYPE_OP(__name__, __operation__)                                                                   \
    template <typename _DataType>                                                                                      \
    INP_DLLEXPORT void __name__(void* result_out,                                                                      \
                                const void* input1_in,                                                                 \
                                const size_t input1_size,                                                              \
                                const size_t* input1_shape,                                                            \
                                const size_t input1_shape_ndim,                                                        \
                                const void* input2_in,                                                                 \
                                const size_t input2_size,                                                              \
                                const size_t* input2_shape,                                                            \
                                const size_t input2_shape_ndim,                                                        \
                                const size_t* where);

#include <dpnp_gen_2arg_1type_tbl.hpp>

#define MACRO_1ARG_1TYPE_OP(__name__, __operation1__, __operation2__)                                                  \
    template <typename _DataType>                                                                                      \
    INP_DLLEXPORT void __name__(void* array1, void* result1, size_t size);

#include <dpnp_gen_1arg_1type_tbl.hpp>

#define MACRO_1ARG_2TYPES_OP(__name__, __operation1__, __operation2__)                                                 \
    template <typename _DataType_input, typename _DataType_output>                                                     \
    INP_DLLEXPORT void __name__(void* array1, void* result1, size_t size);

#include <dpnp_gen_1arg_2type_tbl.hpp>

#define MACRO_2ARG_3TYPES_OP(__name__, __operation1__, __operation2__)                                                 \
    template <typename _DataType_output, typename _DataType_input1, typename _DataType_input2>                         \
    INP_DLLEXPORT void __name__(void* result_out,                                                                      \
                                const void* input1_in,                                                                 \
                                const size_t input1_size,                                                              \
                                const size_t* input1_shape,                                                            \
                                const size_t input1_shape_ndim,                                                        \
                                const void* input2_in,                                                                 \
                                const size_t input2_size,                                                              \
                                const size_t* input2_shape,                                                            \
                                const size_t input2_shape_ndim,                                                        \
                                const size_t* where);

#include <dpnp_gen_2arg_3type_tbl.hpp>

/**
 * @ingroup BACKEND_API
 * @brief fill_diagonal function.
 *
 * @param [in]  array1_in    Input array.
 * @param [in]  val          Value to write on the diagonal.
 * @param [in]  shape        Input shape.
 * @param [in]  ndim         Number of elements in shape.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_fill_diagonal_c(void* array1_in, void* val, size_t* shape, const size_t ndim);

/**
 * @ingroup BACKEND_API
 * @brief floor_divide function.
 *
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
 * @param [out] result1             Output array.
 * @param [in]  size                Number of elements in input arrays.
 */
template <typename _DataType_input1, typename _DataType_input2, typename _DataType_output>
INP_DLLEXPORT void dpnp_floor_divide_c(void* result_out,
                                       const void* input1_in,
                                       const size_t input1_size,
                                       const size_t* input1_shape,
                                       const size_t input1_shape_ndim,
                                       const void* input2_in,
                                       const size_t input2_size,
                                       const size_t* input2_shape,
                                       const size_t input2_shape_ndim,
                                       const size_t* where);

/**
 * @ingroup BACKEND_API
 * @brief modf function.
 *
 * @param [in]  array1_in    Input array.
 * @param [out] result1_out  Output array 1.
 * @param [out] result2_out  Output array 2.
 * @param [in]  size         Number of elements in input arrays.
 */
template <typename _DataType_input, typename _DataType_output>
INP_DLLEXPORT void dpnp_modf_c(void* array1_in, void* result1_out, void* result2_out, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Implementation of ones function
 *
 * @param [out] result    Output array.
 * @param [in]  size      Number of elements in the output array.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_ones_c(void* result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Implementation of ones_like function
 *
 * @param [out] result    Output array.
 * @param [in]  size      Number of elements in the output array.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_ones_like_c(void* result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief remainder function.
 *
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
 * @param [out] result1             Output array.
 * @param [in]  size                Number of elements in input arrays.
 */
template <typename _DataType_output, typename _DataType_input1, typename _DataType_input2>
INP_DLLEXPORT void dpnp_remainder_c(void* result_out,
                                    const void* input1_in,
                                    const size_t input1_size,
                                    const size_t* input1_shape,
                                    const size_t input1_shape_ndim,
                                    const void* input2_in,
                                    const size_t input2_size,
                                    const size_t* input2_shape,
                                    const size_t input2_shape_ndim,
                                    const size_t* where);

/**
 * @ingroup BACKEND_API
 * @brief repeat elements of an array.
 *
 * @param [in]  array_in    Input array.
 * @param [out] result      Output array.
 * @param [in]  repeats      The number of repetitions for each element.
 * @param [in]  size         Number of elements in input arrays.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_repeat_c(const void* array_in, void* result, const size_t repeats, const size_t size);

/**
 * @ingroup BACKEND_API
 * @brief copyto function.
 *
 * @param [out] destination  Destination array.
 * @param [in]  source       Source array.
 * @param [in]  size         Number of elements in destination array.
 */
template <typename _DataType_dst, typename _DataType_src>
INP_DLLEXPORT void dpnp_copyto_c(void* destination, void* source, const size_t size);

/**
 * @ingroup BACKEND_API
 * @brief transpose function. Permute axes of the input to the output with elements permutation.
 *
 * @param [in]  array1_in    Input array.
 * @param [in]  input_shape  Input shape.
 * @param [in]  result_shape Output shape.
 * @param [in]  permute_axes Order of axis by it's id as it should be presented in output.
 * @param [in]  ndim         Number of elements in shapes and axes.
 * @param [out] result1      Output array.
 * @param [in]  size         Number of elements in input arrays.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_elemwise_transpose_c(void* array1_in,
                                             const size_t* input_shape,
                                             const size_t* result_shape,
                                             const size_t* permute_axes,
                                             size_t ndim,
                                             void* result1,
                                             size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Custom implementation of trapz function
 *
 * @param [in]  array1_in    First input array.
 * @param [in]  array2_in    Second input array.
 * @param [out] result1      Output array.
 * @param [in]  dx           The spacing between sample points.
 * @param [in]  array1_size  Number of elements in first input array.
 * @param [in]  array2_size  Number of elements in second input arrays.
 *
 */
template <typename _DataType_input1, typename _DataType_input2, typename _DataType_output>
INP_DLLEXPORT void dpnp_trapz_c(
    const void* array1_in, const void* array2_in, void* result1, double dx, size_t array1_size, size_t array2_size);

/**
 * @ingroup BACKEND_API
 * @brief Implementation of vander function
 *
 * @param [in]  array_in    Input array.
 * @param [out] result      Output array.
 * @param [in]  size_in     Number of elements in the input array.
 * @param [in]  N           Number of columns in the output.
 * @param [in]  increasing  Order of the powers of the columns.
 *
 */
template <typename _DataType_input, typename _DataType_output>
INP_DLLEXPORT void
    dpnp_vander_c(const void* array1_in, void* result1, const size_t size_in, const size_t N, const int increasing);

/**
 * @ingroup BACKEND_API
 * @brief Implementation of zeros function
 *
 * @param [out] result    Output array.
 * @param [in]  size      Number of elements in the output array.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_zeros_c(void* result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Implementation of zeros_like function
 *
 * @param [out] result    Output array.
 * @param [in]  size      Number of elements in the output array.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_zeros_like_c(void* result, size_t size);

#endif // BACKEND_IFACE_H
