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

#ifdef _WIN
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
    GPU_SELECTOR  /**< Intel GPU side execution mode */
};

/**
 * @ingroup BACKEND_API
 * @brief SYCL queue initialization.
 *
 * Global SYCL queue initialization.
 *
 * @param [in]  selector       Select type @ref QueueOptions of the SYCL queue.
 */
INP_DLLEXPORT void dpnp_queue_initialize_c(QueueOptions selector);

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
 * @brief Matrix multiplication.
 *
 * Matrix multiplication procedure. Works with 2-D matrices
 *
 * @param [in]  array1    Input array.
 *
 * @param [in]  array2    Input array.
 *
 * @param [out] result1   Output array.
 *
 * @param [in]  size      Number of elements in input arrays.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void
    custom_blas_gemm_c(void* array1, void* array2, void* result1, size_t size_m, size_t size_n, size_t size_k);

/**
 * @ingroup BACKEND_API
 * @brief absolute function.
 *
 * @param [in]  array1_in    Input array.
 *
 * @param [in]  input_shape  Input shape.
 *
 * @param [out] result1      Output array.
 *
 * @param [in]  size         Number of elements in input arrays.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void
    custom_elemwise_absolute_c(void* array1_in, const std::vector<long>& input_shape, void* result1, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Custom implementation of dot function
 *
 * @param [in]  array1  Input array.
 *
 * @param [in]  array2  Input array.
 *
 * @param [out] result1 Output array.
 *
 * @param [in]  size    Number of elements in input arrays.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void custom_blas_dot_c(void* array1, void* array2, void* result1, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Sum of array elements
 *
 * @param [in]  array  Input array.
 *
 * @param [in]  size    Number of input elements in `array`.
 *
 * @param [out] result Output array contains one element.
 */
template <typename _DataType>
INP_DLLEXPORT void custom_sum_c(void* array, void* result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Product of array elements
 *
 * @param [in]  array  Input array.
 *
 * @param [in]  size    Number of input elements in `array`.
 *
 * @param [out] result Output array contains one element.
 */
template <typename _DataType>
INP_DLLEXPORT void custom_prod_c(void* array, void* result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of eig function
 *
 * @param [in]  array1  Input array.
 *
 * @param [out] result1 Output array.
 *
 * @param [in]  size    Number of elements in input arrays.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void mkl_lapack_syevd_c(void* array1, void* result1, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of argsort function
 *
 * @param [in]  array   Input array with data.
 *
 * @param [out] result  Output array with indeces.
 *
 * @param [in]  size    Number of elements in input arrays.
 *
 */
template <typename _DataType, typename _idx_DataType>
INP_DLLEXPORT void custom_argsort_c(void* array, void* result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of sort function
 *
 * @param [in]  array   Input array with data.
 *
 * @param [out] result  Output array with indeces.
 *
 * @param [in]  size    Number of elements in input arrays.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void custom_sort_c(void* array, void* result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Custom implementation of cov function with math library and PSTL
 *
 * @param [in]  array       Input array.
 *
 * @param [out] result      Output array.
 *
 * @param [in]  nrows       Number of rows in input array.
 *
 * @param [in]  ncols       Number of columns in input array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void custom_cov_c(void* array1_in, void* result1, size_t nrows, size_t ncols);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of max function
 *
 * @param [in]  array   Input array with data.
 *
 * @param [out] result  Output array.
 *
 * @param [in]  shape   Shape of input array.
 *
 * @param [in]  ndim    Number of elements in shape.
 *
 * @param [in]  axis    Axis.
 *
 * @param [in]  naxis   Number of elements in axis.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void
    custom_max_c(void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of mean function
 *
 * @param [in]  array   Input array with data.
 *
 * @param [out] result  Output array.
 *
 * @param [in]  shape   Shape of input array.
 *
 * @param [in]  ndim    Number of elements in shape.
 *
 * @param [in]  axis    Axis.
 *
 * @param [in]  naxis   Number of elements in axis.
 *
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void
    custom_mean_c(void* array, void* result, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of median function
 *
 * @param [in]  array   Input array with data.
 *
 * @param [out] result  Output array.
 *
 * @param [in]  shape   Shape of input array.
 *
 * @param [in]  ndim    Number of elements in shape.
 *
 * @param [in]  axis    Axis.
 *
 * @param [in]  naxis   Number of elements in axis.
 *
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void
    custom_median_c(void* array, void* result, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of min function
 *
 * @param [in]  array   Input array with data.
 *
 * @param [out] result  Output array.
 *
 * @param [in]  shape   Shape of input array.
 *
 * @param [in]  ndim    Number of elements in shape.
 *
 * @param [in]  axis    Axis.
 *
 * @param [in]  naxis   Number of elements in axis.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void
    custom_min_c(void* array, void* result, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of argmax function
 *
 * @param [in]  array   Input array with data.
 *
 * @param [out] result  Output array with indeces.
 *
 * @param [in]  size    Number of elements in input array.
 *
 */
template <typename _DataType, typename _idx_DataType>
INP_DLLEXPORT void custom_argmax_c(void* array, void* result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of argmin function
 *
 * @param [in]  array   Input array with data.
 *
 * @param [out] result  Output array with indeces.
 *
 * @param [in]  size    Number of elements in input array.
 *
 */
template <typename _DataType, typename _idx_DataType>
INP_DLLEXPORT void custom_argmin_c(void* array, void* result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of std function
 *
 * @param [in]  array   Input array with data.
 *
 * @param [out] result  Output array with indeces.
 *
 * @param [in]  shape   Shape of input array.
 *
 * @param [in]  ndim    Number of elements in shape.
 *
 * @param [in]  axis    Axis.
 *
 * @param [in]  naxis   Number of elements in axis.
 *
 * @param [in]  ddof    Delta degrees of freedom.
 *
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void custom_std_c(
    void* array, void* result, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis, size_t ddof);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of var function
 *
 * @param [in]  array   Input array with data.
 *
 * @param [out] result  Output array with indeces.
 *
 * @param [in]  shape   Shape of input array.
 *
 * @param [in]  ndim    Number of elements in shape.
 *
 * @param [in]  axis    Axis.
 *
 * @param [in]  naxis   Number of elements in axis.
 *
 * @param [in]  ddof    Delta degrees of freedom.
 *
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void custom_var_c(
    void* array, void* result, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis, size_t ddof);

/**
 * @ingroup BACKEND_API
 * @brief Element wise function __name__
 *
 * __name__ function called with the SYCL backend.
 *
 * @param [in]  size  Number of elements in the input array.
 *
 */
#define MACRO_CUSTOM_1ARG_2TYPES_OP(__name__, __operation__)                                                           \
    template <typename _DataType_input, typename _DataType_output>                                                     \
    INP_DLLEXPORT void custom_elemwise_##__name__##_c(void* array1, void* result1, size_t size);

#include <custom_1arg_2type_tbl.hpp>

#define MACRO_CUSTOM_1ARG_1TYPE_OP(__name__, __operation__)                                                            \
    template <typename _DataType>                                                                                      \
    INP_DLLEXPORT void custom_elemwise_##__name__##_c(void* array1, void* result1, size_t size);

#include <custom_1arg_1type_tbl.hpp>

#define MACRO_CUSTOM_2ARG_3TYPES_OP(__name__, __operation__)                                                           \
    template <typename _DataType_input1, typename _DataType_input2, typename _DataType_output>                         \
    INP_DLLEXPORT void custom_elemwise_##__name__##_c(void* array1, void* array2, void* result1, size_t size);

#include <custom_2arg_3type_tbl.hpp>

/**
 * @ingroup BACKEND_API
 * @brief transpose function. Permute axes of the input to the output with elements permutation.
 *
 * @param [in]  array1_in    Input array.
 *
 * @param [in]  input_shape  Input shape.
 *
 * @param [in]  result_shape Output shape.
 *
 * @param [in]  permute_axes Order of axis by it's id as it should be presented in output.
 *
 * @param [out] result1      Output array.
 *
 * @param [in]  size         Number of elements in input arrays.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void custom_elemwise_transpose_c(void* array1_in,
                                               const std::vector<long>& input_shape,
                                               const std::vector<long>& result_shape,
                                               const std::vector<long>& permute_axes,
                                               void* result1,
                                               size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of random number generator (gaussian continious distribution)
 *
 * @param [in]  size   Number of elements in `result` arrays.
 *
 * @param [out] result Output array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void mkl_rng_gaussian(void* result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of random number generator (uniform distribution)
 *
 * @param [in]  low    Left bound of array values.
 *
 * @param [in]  high   Right bound of array values.
 *
 * @param [in]  size   Number of elements in `result` array.
 *
 * @param [out] result Output array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void mkl_rng_uniform(void* result, long low, long high, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief initializer for basic random number generator.
 *
 */
INP_DLLEXPORT void dpnp_engine_rng_initialize();

/**
 * @ingroup BACKEND_API
 * @brief initializer for basic random number generator.
 *
 * @param [in]  low    Left bound of array values.
 *
 */
INP_DLLEXPORT void dpnp_engine_rng_initialize(size_t seed);

#endif // BACKEND_IFACE_H
