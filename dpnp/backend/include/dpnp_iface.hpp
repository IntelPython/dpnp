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
 * @brief Matrix multiplication.
 *
 * Matrix multiplication procedure. Works with 2-D matrices
 *
 * @param [in]  array1    Input array.
 * @param [in]  array2    Input array.
 * @param [out] result1   Output array.
 * @param [in]  size      Number of elements in input arrays.
 */
template <typename _DataType>
INP_DLLEXPORT void
    dpnp_matmul_c(void* array1, void* array2, void* result1, size_t size_m, size_t size_n, size_t size_k);

/**
 * @ingroup BACKEND_API
 * @brief absolute function.
 *
 * @param [in]  array1_in    Input array.
 * @param [out] result1      Output array.
 * @param [in]  size         Number of elements in input arrays.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_elemwise_absolute_c(void* array1_in, void* result1, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Custom implementation of dot function
 *
 * @param [in]  array1  Input array.
 * @param [in]  array2  Input array.
 * @param [out] result1 Output array.
 * @param [in]  size    Number of elements in input arrays.
 */
template <typename _DataType_input1, typename _DataType_input2, typename _DataType_output>
INP_DLLEXPORT void dpnp_dot_c(void* array1, void* array2, void* result1, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Custom implementation of cross function
 *
 * @param [in]  array1_in  First input array.
 * @param [in]  array2_in  Second input array.
 * @param [out] result1 Output array.
 * @param [in]  size    Number of elements in input arrays.
 *
 */
template <typename _DataType_input1, typename _DataType_input2, typename _DataType_output>
INP_DLLEXPORT void dpnp_cross_c(void* array1_in, void* array2_in, void* result1, size_t size);

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
 * @brief Sum of array elements
 *
 * @param [in]  array  Input array.
 * @param [in]  size    Number of input elements in `array`.
 * @param [out] result Output array contains one element.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_sum_c(void* array, void* result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Product of array elements
 *
 * @param [in]  array  Input array.
 * @param [in]  size    Number of input elements in `array`.
 * @param [out] result Output array contains one element.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_prod_c(void* array, void* result, size_t size);

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
 * @param [in]  array1_in   Input array 1.
 * @param [in]  array2_in   Input array 2.
 * @param [out] result      Output array.
 * @param [in]  size        Number of elements in input arrays.
 */
template <typename _DataType_input1, typename _DataType_input2, typename _DataType_output>
INP_DLLEXPORT void dpnp_correlate_c(void* array1_in, void* array2_in, void* result, size_t size);

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
 * @brief math library implementation of diagonal function
 *
 * @param [in]  array   Input array with data.
 * @param [out] result  Output array.
 * @param [in]  offset  Offset of the diagonal from the main diagonal.
 * @param [in]  shape   Shape of input array.
 * @param [in]  shape   Shape of output array.
 * @param [in]  ndim    Number of elements in shape.
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_diagonal_c(void* array1_in, void* result1, const size_t offset, size_t* shape, size_t* res_shape, const size_t res_ndim);

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
template <typename _DataType>
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
 * @param [in]  shape   Shape of input array.
 * @param [in]  ndim    Number of elements in shape.
 * @param [in]  axis    Axis.
 * @param [in]  naxis   Number of elements in axis.
 */
template <typename _DataType>
INP_DLLEXPORT void
    dpnp_max_c(void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);

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
 * @param [in]  shape   Shape of input array.
 * @param [in]  ndim    Number of elements in shape.
 * @param [in]  axis    Axis.
 * @param [in]  naxis   Number of elements in axis.
 */
template <typename _DataType>
INP_DLLEXPORT void
    dpnp_min_c(void* array, void* result, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);

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
 * @param [in]  indices Input array with indices.
 * @param [out] result  Output array.
 * @param [in]  size    Number of elements in the input array.
 */
template <typename _DataType, typename _IndecesType>
INP_DLLEXPORT void dpnp_take_c(void* array, void* indices, void* result, size_t size);

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
    INP_DLLEXPORT void __name__(void* array1_in1, void* array2_in, void* result1, size_t size);

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
    template <typename _DataType_input1, typename _DataType_input2, typename _DataType_output>                         \
    INP_DLLEXPORT void __name__(void* array1, void* array2, void* result1, size_t size);

#include <dpnp_gen_2arg_3type_tbl.hpp>

/**
 * @ingroup BACKEND_API
 * @brief floor_divide function.
 *
 * @param [in]  array1_in    Input array 1.
 * @param [in]  array2_in    Input array 2.
 * @param [out] result1      Output array.
 * @param [in]  size         Number of elements in input arrays.
 */
template <typename _DataType_input1, typename _DataType_input2, typename _DataType_output>
INP_DLLEXPORT void dpnp_floor_divide_c(void* array1_in, void* array2_in, void* result1, size_t size);

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
 * @brief remainder function.
 *
 * @param [in]  array1_in    Input array 1.
 * @param [in]  array2_in    Input array 2.
 * @param [out] result1      Output array.
 * @param [in]  size         Number of elements in input arrays.
 */
template <typename _DataType_input1, typename _DataType_input2, typename _DataType_output>
INP_DLLEXPORT void dpnp_remainder_c(void* array1_in, void* array2_in, void* result1, size_t size);

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

#endif // BACKEND_IFACE_H
