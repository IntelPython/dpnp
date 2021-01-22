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
 *
 * @param [in]  step      Step for initialization sequence
 *
 * @param [out] result1   Output array.
 *
 * @param [in]  size      Number of elements in input arrays.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_arange_c(size_t start, size_t step, void* result1, size_t size);

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
    dpnp_matmul_c(void* array1, void* array2, void* result1, size_t size_m, size_t size_n, size_t size_k);

/**
 * @ingroup BACKEND_API
 * @brief absolute function.
 *
 * @param [in]  array1_in    Input array.
 *
 * @param [out] result1      Output array.
 *
 * @param [in]  size         Number of elements in input arrays.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_elemwise_absolute_c(void* array1_in, void* result1, size_t size);

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
template <typename _DataType_input1, typename _DataType_input2, typename _DataType_output>
INP_DLLEXPORT void dpnp_dot_c(void* array1, void* array2, void* result1, size_t size);

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
INP_DLLEXPORT void dpnp_sum_c(void* array, void* result, size_t size);

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
INP_DLLEXPORT void dpnp_prod_c(void* array, void* result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Compute the eigenvalues and right eigenvectors of a square array.
 *
 * @param [in]  array_in  Input array[size][size]
 *
 * @param [out] result1   The eigenvalues, each repeated according to its multiplicity
 *
 * @param [out] result2   The normalized (unit "length") eigenvectors
 *
 * @param [in]  size      One dimension of square [size][size] array
 *
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void dpnp_eig_c(const void* array_in, void* result1, void* result2, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief Compute the eigenvalues of a square array.
 *
 * @param [in]  array_in  Input array[size][size]
 *
 * @param [out] result1   The eigenvalues, each repeated according to its multiplicity
 *
 * @param [in]  size      One dimension of square [size][size] array
 *
 */
template <typename _DataType, typename _ResultType>
INP_DLLEXPORT void dpnp_eigvals_c(const void* array_in, void* result1, size_t size);

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
INP_DLLEXPORT void dpnp_argsort_c(void* array, void* result, size_t size);

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
INP_DLLEXPORT void dpnp_sort_c(void* array, void* result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of cholesky function
 *
 * @param [in]  array   Input array with data.
 *
 * @param [out] result  Output array.
 *
 * @param [in]  shape   Shape of input array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_cholesky_c(void* array1_in, void* result1, size_t* shape);

/**
 * @ingroup BACKEND_API
 * @brief correlate function
 *
 * @param [in]  array1_in   Input array 1.
 *
 * @param [in]  array2_in   Input array 2.
 *
 * @param [out] result      Output array.
 *
 * @param [in]  size        Number of elements in input arrays.
 *
 */
template <typename _DataType_input1, typename _DataType_input2, typename _DataType_output>
INP_DLLEXPORT void dpnp_correlate_c(void* array1_in, void* array2_in, void* result, size_t size);

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
INP_DLLEXPORT void dpnp_cov_c(void* array1_in, void* result1, size_t nrows, size_t ncols);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of det function
 *
 * @param [in]  array   Input array with data.
 *
 * @param [out] result  Output array.
 *
 * @param [in]  shape   Shape of input array.
 *
 * @param [in]  ndim    Number of elements in shape.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_det_c(void* array1_in, void* result1, size_t* shape, size_t ndim);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of inv function
 *
 * @param [in]  array   Input array with data.
 *
 * @param [out] result  Output array.
 *
 * @param [in]  shape   Shape of input array.
 *
 * @param [in]  ndim    Number of elements in shape.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_inv_c(void* array1_in, void* result1, size_t* shape, size_t ndim);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of matrix_rank function
 *
 * @param [in]  array   Input array with data.
 *
 * @param [out] result  Output array.
 *
 * @param [in]  shape   Shape of input array.
 *
 * @param [in]  ndim    Number of elements in shape.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_matrix_rank_c(void* array1_in, void* result1, size_t* shape, size_t ndim);

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
    dpnp_max_c(void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);

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
    dpnp_mean_c(void* array, void* result, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);

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
    dpnp_median_c(void* array, void* result, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);

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
    dpnp_min_c(void* array, void* result, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);

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
INP_DLLEXPORT void dpnp_argmax_c(void* array, void* result, size_t size);

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
INP_DLLEXPORT void dpnp_argmin_c(void* array, void* result, size_t size);

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
INP_DLLEXPORT void dpnp_std_c(
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
INP_DLLEXPORT void dpnp_var_c(
    void* array, void* result, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis, size_t ddof);

/**
 * @ingroup BACKEND_API
 * @brief Implementation of invert function
 *
 * @param [in]  array1_in  Input array.
 *
 * @param [out] result1    Output array.
 *
 * @param [in]  size       Number of elements in the input array.
 *
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
 *
 * @param [in]  array2_in    Input array 2.
 *
 * @param [out] result1      Output array.
 *
 * @param [in]  size         Number of elements in input arrays.
 *
 */
template <typename _DataType_input1, typename _DataType_input2, typename _DataType_output>
INP_DLLEXPORT void dpnp_floor_divide_c(void* array1_in, void* array2_in, void* result1, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief modf function.
 *
 * @param [in]  array1_in    Input array.
 *
 * @param [out] result1_out  Output array 1.
 *
 * @param [out] result2_out  Output array 2.
 *
 * @param [in]  size         Number of elements in input arrays.
 *
 */
template <typename _DataType_input, typename _DataType_output>
INP_DLLEXPORT void dpnp_modf_c(void* array1_in, void* result1_out, void* result2_out, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief remainder function.
 *
 * @param [in]  array1_in    Input array 1.
 *
 * @param [in]  array2_in    Input array 2.
 *
 * @param [out] result1      Output array.
 *
 * @param [in]  size         Number of elements in input arrays.
 *
 */
template <typename _DataType_input1, typename _DataType_input2, typename _DataType_output>
INP_DLLEXPORT void dpnp_remainder_c(void* array1_in, void* array2_in, void* result1, size_t size);

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
INP_DLLEXPORT void dpnp_elemwise_transpose_c(void* array1_in,
                                             const std::vector<long>& input_shape,
                                             const std::vector<long>& result_shape,
                                             const std::vector<long>& permute_axes,
                                             void* result1,
                                             size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of random number generator (beta distribution)
 *
 * @param [in]  size   Number of elements in `result` arrays.
 *
 * @param [in]  a      Alpha, shape param.
 *
 * @param [in]  b      Beta, scalefactor.
 *
 * @param [out] result Output array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_beta_c(void* result, _DataType a, _DataType b, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of random number generator (binomial distribution)
 *
 * @param [in]  size   Number of elements in `result` arrays.
 *
 * @param [in]  ntrial Number of independent trials.
 *
 * @param [in]  p      Success probability p of a single trial.
 *
 * @param [out] result Output array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_binomial_c(void* result, int ntrial, double p, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of random number generator (chi-square distribution)
 *
 * @param [in]  size   Number of elements in `result` arrays.
 *
 * @param [in]  df     Degrees of freedom.
 *
 * @param [out] result Output array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_chi_square_c(void* result, int df, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of random number generator (exponential distribution)
 *
 * @param [in]  size   Number of elements in `result` arrays.
 *
 * @param [in]  beta   Beta, scalefactor.
 *
 * @param [out] result Output array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_exponential_c(void* result, _DataType beta, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of random number generator (gamma distribution)
 *
 * @param [in]  size   Number of elements in `result` arrays.
 *
 * @param [in]  shape  The shape of the gamma distribution.
 *
 * @param [in]  scale  The scale of the gamma distribution.
 *
 * @param [out] result Output array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_gamma_c(void* result, _DataType shape, _DataType scale, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of random number generator (gaussian continious distribution)
 *
 * @param [in]  size   Number of elements in `result` arrays.
 *
 * @param [in]  mean   Mean value.
 *
 * @param [in]  stddev Standard deviation.
 *
 * @param [out] result Output array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_gaussian_c(void* result, _DataType mean, _DataType stddev, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of random number generator (hypergeometric distribution)
 *
 * @param [in]  size   Number of elements in `result` arrays.
 *
 * @param [in]  l      Lot size of l.
 *
 * @param [in]  s      Size of sampling without replacement.
 *
 * @param [in]  m      Number of marked elements m.
 *
 * @param [out] result Output array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_hypergeometric_c(void* result, int l, int s, int m, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of random number generator (geometric distribution)
 *
 * @param [in]  size   Number of elements in `result` arrays.
 *
 * @param [in]  p      Success probability p of a trial.
 *
 * @param [out] result Output array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_geometric_c(void* result, float p, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of random number generator (gumbel distribution)
 *
 * @param [in]  size   Number of elements in `result` arrays.
 *
 * @param [in]  loc    The location of the mode of the distribution.
 *
 * @param [in]  scale  The scale parameter of the distribution.
 *
 * @param [out] result Output array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_gumbel_c(void* result, double loc, double scale, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of random number generator (laplace distribution)
 *
 * @param [in]  size   Number of elements in `result` arrays.
 *
 * @param [in]  loc    The position of the distribution peak.
 *
 * @param [in]  scale  The exponential decay.
 *
 * @param [out] result Output array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_laplace_c(void* result, double loc, double scale, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of random number generator (lognormal distribution)
 *
 * @param [in]  size   Number of elements in `result` arrays.
 *
 * @param [in]  mean   Mean value.
 *
 * @param [in]  stddev Standard deviation.
 *
 * @param [out] result Output array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_lognormal_c(void* result, _DataType mean, _DataType stddev, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of random number generator (multinomial distribution)
 *
 * @param [in]  size          Number of elements in `result` arrays.
 *
 * @param [in]  ntrial        Number of independent trials.
 *
 * @param [in]  p_vector      Probability vector of possible outcomes (k length).
 *
 * @param [in]  p_vector_size Length of `p_vector`.
 *
 * @param [out] result        Output array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void
    dpnp_rng_multinomial_c(void* result, int ntrial, const double* p_vector, const size_t p_vector_size, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of random number generator (multinomial distribution)
 * TODO
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_multivariate_normal_c(void* result,
                                                  const int dimen,
                                                  const double* mean_vector,
                                                  const size_t mean_vector_size,
                                                  const double* cov_vector,
                                                  const size_t cov_vector_size,
                                                  size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of random number generator (negative binomial distribution)
 *
 * @param [in]  size   Number of elements in `result` arrays.
 *
 * @param [in]  a      The first distribution parameter a, > 0.
 *
 * @param [in]  p      The second distribution parameter p, >= 0 and <=1.
 *
 * @param [out] result Output array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_negative_binomial_c(void* result, double a, double p, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of random number generator (normal continious distribution)
 *
 * @param [in]  size   Number of elements in `result` arrays.
 *
 * @param [in]  mean   Mean value.
 *
 * @param [in]  stddev Standard deviation.
 *
 * @param [out] result Output array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_normal_c(void* result, _DataType mean, _DataType stddev, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of random number generator (poisson distribution)
 *
 * @param [in]  size   Number of elements in `result` arrays.
 *
 * @param [in]  lambda Distribution parameter lambda.
 *
 * @param [out] result Output array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_poisson_c(void* result, double lambda, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of random number generator (power distribution)
 *
 * @param [in]  size   Number of elements in `result` arrays.
 *
 * @param [in]  alpha  Shape of the distribution, alpha.
 *
 * @param [out] result Output array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_power_c(void* result, double alpha, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of random number generator (rayleigh distribution)
 *
 * @param [in]  size   Number of elements in `result` arrays.
 *
 * @param [in]  scale  Distribution parameter, scalefactor.
 *
 * @param [out] result Output array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_rayleigh_c(void* result, _DataType scale, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of random number generator (standard cauchy distribution)
 *
 * @param [in]  size   Number of elements in `result` arrays.
 *
 * @param [out] result Output array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_standard_cauchy_c(void* result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of random number generator (standard exponential distribution)
 *
 * @param [in]  size   Number of elements in `result` arrays.
 *
 * @param [out] result Output array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_standard_exponential_c(void* result, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of random number generator (standard gamma distribution)
 *
 * @param [in]  size   Number of elements in `result` arrays.
 *
 * @param [in]  shape  Shape value.
 *
 * @param [out] result Output array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_standard_gamma_c(void* result, _DataType shape, size_t size);


/**
 * @ingroup BACKEND_API
 * @brief math library implementation of random number generator (standard normal distribution)
 *
 * @param [in]  size   Number of elements in `result` arrays.
 *
 * @param [out] result Output array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_standard_normal_c(void* result, size_t size);

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
INP_DLLEXPORT void dpnp_rng_uniform_c(void* result, long low, long high, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief math library implementation of random number generator (weibull distribution)
 *
 * @param [in]  size   Number of elements in `result` arrays.
 *
 * @param [in]  alpha  Shape parameter of the distribution, alpha.
 *
 * @param [out] result Output array.
 *
 */
template <typename _DataType>
INP_DLLEXPORT void dpnp_rng_weibull_c(void* result, double alpha, size_t size);

/**
 * @ingroup BACKEND_API
 * @brief initializer for basic random number generator.
 *
 * @param [in]  seed    The seed value.
 *
 */
INP_DLLEXPORT void dpnp_srand_c(size_t seed = 1);

#endif // BACKEND_IFACE_H
