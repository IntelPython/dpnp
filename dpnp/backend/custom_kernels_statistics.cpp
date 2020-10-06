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

#include <iostream>
#include <mkl_blas_sycl.hpp>
#include <mkl_stats_sycl.hpp>

#include <backend_iface.hpp>
#include "backend_pstl.hpp"
#include "backend_utils.hpp"
#include "queue_sycl.hpp"

namespace mkl_blas = oneapi::mkl::blas::row_major;
namespace mkl_stats = oneapi::mkl::stats;

template <typename _DataType>
class custom_cov_c_kernel;

template <typename _DataType>
void custom_cov_c(void* array1_in, void* result1, size_t nrows, size_t ncols)
{
    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    if (!nrows || !ncols)
    {
        return;
    }

    auto policy = oneapi::dpl::execution::make_device_policy<class custom_cov_c_kernel<_DataType>>(DPNP_QUEUE);

    _DataType* mean = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(nrows * sizeof(_DataType)));
    for (size_t i = 0; i < nrows; ++i)
    {
        _DataType* row_start = array_1 + ncols * i;
        mean[i] = std::reduce(policy, row_start, row_start + ncols, _DataType(0), std::plus<_DataType>()) / ncols;
    }
    policy.queue().wait();

#if 0
    std::cout << "mean\n";
    for (size_t i = 0; i < nrows; ++i)
    {
        std::cout << " , " << mean[i];
    }
    std::cout << std::endl;
#endif

    _DataType* temp = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(nrows * ncols * sizeof(_DataType)));
    for (size_t i = 0; i < nrows; ++i)
    {
        size_t offset = ncols * i;
        _DataType* row_start = array_1 + offset;
        std::transform(policy, row_start, row_start + ncols, temp + offset, [=](_DataType x) { return x - mean[i]; });
    }
    policy.queue().wait();

#if 0
    std::cout << "temp\n";
    for (size_t i = 0; i < nrows; ++i)
    {
        for (size_t j = 0; j < ncols; ++j)
        {
            std::cout << " , " << temp[i * ncols + j];
        }
        std::cout << std::endl;
    }
#endif

    cl::sycl::event event_syrk;

    const _DataType alpha = _DataType(1) / (ncols - 1);
    const _DataType beta = _DataType(0);

    event_syrk = mkl_blas::syrk(DPNP_QUEUE,                       // queue &exec_queue,
                                oneapi::mkl::uplo::upper,         // uplo upper_lower,
                                oneapi::mkl::transpose::nontrans, // transpose trans,
                                nrows,                            // std::int64_t n,
                                ncols,                            // std::int64_t k,
                                alpha,                            // T alpha,
                                temp,                             //const T* a,
                                ncols,                            // std::int64_t lda,
                                beta,                             // T beta,
                                result,                           // T* c,
                                nrows);                           // std::int64_t ldc);
    event_syrk.wait();

#if 0 // serial fill lower elements on CPU
    for (size_t i = 1; i < nrows; ++i)
    {
        for (size_t j = 0; j < i; ++j)
        {
            result[i * nrows + j] = result[j * nrows + i];
        }
    }
#endif

    // fill lower elements
    cl::sycl::event event;
    cl::sycl::range<1> gws(nrows * nrows);
    auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
        const size_t idx = global_id[0];
        const size_t row_idx = idx / nrows;
        const size_t col_idx = idx - row_idx * nrows;
        if (col_idx < row_idx)
        {
            result[idx] = result[col_idx * nrows + row_idx];
        }
    };

    auto kernel_func = [&](cl::sycl::handler& cgh) {
        cgh.parallel_for<class custom_cov_c_kernel<_DataType>>(gws, kernel_parallel_for_func);
    };

    event = DPNP_QUEUE.submit(kernel_func);

    event.wait();

    dpnp_memory_free_c(mean);
    dpnp_memory_free_c(temp);
}

template void custom_cov_c<double>(void* array1_in, void* result1, size_t nrows, size_t ncols);

template <typename _DataType>
class custom_max_c_kernel;

template <typename _DataType>
void custom_max_c(void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis)
{
    __attribute__((unused)) void* tmp = (void*)(axis + naxis);

    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    size_t size = 1;
    for (size_t i = 0; i < ndim; ++i)
    {
        size *= shape[i];
    }

    if constexpr (std::is_same<_DataType, double>::value || std::is_same<_DataType, float>::value)
    {
        // Required initializing the result before call the function
        result[0] = array_1[0];

        // https://docs.oneapi.com/versions/latest/onemkl/mkl-stats-make_dataset.html
        auto dataset = mkl_stats::make_dataset<mkl_stats::layout::row_major>(1, size, array_1);

        // https://docs.oneapi.com/versions/latest/onemkl/mkl-stats-max.html
        cl::sycl::event event = mkl_stats::max(DPNP_QUEUE, dataset, result);

        event.wait();
    }
    else
    {
        auto policy = oneapi::dpl::execution::make_device_policy<class custom_max_c_kernel<_DataType>>(DPNP_QUEUE);

        _DataType* res = std::max_element(policy, array_1, array_1 + size);
        policy.queue().wait();

        result[0] = *res;
    }
}

template void custom_max_c<double>(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);
template void custom_max_c<float>(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);
template void custom_max_c<long>(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);
template void custom_max_c<int>(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);

template <typename _DataType, typename _ResultType>
void custom_mean_c(void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis)
{
    __attribute__((unused)) void* tmp = (void*)(axis + naxis);

    _ResultType* result = reinterpret_cast<_ResultType*>(result1);

    size_t size = 1;
    for (size_t i = 0; i < ndim; ++i)
    {
        size *= shape[i];
    }

    if (!size)
    {
        return;
    }

    if constexpr (std::is_same<_DataType, double>::value || std::is_same<_DataType, float>::value)
    {
        _ResultType* array = reinterpret_cast<_DataType*>(array1_in);

        // https://docs.oneapi.com/versions/latest/onemkl/mkl-stats-make_dataset.html
        auto dataset = mkl_stats::make_dataset<mkl_stats::layout::row_major>(1, size, array);

        // https://docs.oneapi.com/versions/latest/onemkl/mkl-stats-mean.html
        cl::sycl::event event = mkl_stats::mean(DPNP_QUEUE, dataset, result);

        event.wait();
    }
    else
    {
        _DataType* sum = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(1 * sizeof(_DataType)));

        custom_sum_c<_DataType>(array1_in, sum, size);

        result[0] = static_cast<_ResultType>(sum[0]) / static_cast<_ResultType>(size);

        dpnp_memory_free_c(sum);
    }
}

template void custom_mean_c<double, double>(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);
template void custom_mean_c<float, float>(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);
template void custom_mean_c<long, double>(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);
template void custom_mean_c<int, double>(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);

template <typename _DataType, typename _ResultType>
void custom_median_c(void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis)
{
    __attribute__((unused)) void* tmp = (void*)(axis + naxis);

    _ResultType* result = reinterpret_cast<_ResultType*>(result1);

    size_t size = 1;
    for (size_t i = 0; i < ndim; ++i)
    {
        size *= shape[i];
    }

    _DataType* sorted = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(size * sizeof(_DataType)));

    custom_sort_c<_DataType>(array1_in, sorted, size);

    if (size % 2 == 0)
    {
        result[0] = static_cast<_ResultType>(sorted[size / 2] + sorted[size / 2 - 1]) / 2;
    }
    else
    {
        result[0] = sorted[(size - 1) / 2];
    }

    dpnp_memory_free_c(sorted);

#if 0
    std::cout << "median result " << result[0] << "\n";
#endif
}

template void custom_median_c<double, double>(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);
template void custom_median_c<float, double>(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);
template void custom_median_c<long, double>(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);
template void custom_median_c<int, double>(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);

template <typename _DataType>
class custom_min_c_kernel;

template <typename _DataType>
void custom_min_c(void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis)
{
    __attribute__((unused)) void* tmp = (void*)(axis + naxis);

    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    size_t size = 1;
    for (size_t i = 0; i < ndim; ++i)
    {
        size *= shape[i];
    }
    if constexpr (std::is_same<_DataType, double>::value || std::is_same<_DataType, float>::value)
    {
        // Required initializing the result before call the function
        result[0] = array_1[0];

        // https://docs.oneapi.com/versions/latest/onemkl/mkl-stats-make_dataset.html
        auto dataset = mkl_stats::make_dataset<mkl_stats::layout::row_major>(1, size, array_1);

        // https://docs.oneapi.com/versions/latest/onemkl/mkl-stats-min.html
        cl::sycl::event event = mkl_stats::min(DPNP_QUEUE, dataset, result);

        event.wait();
    }
    else
    {
        auto policy = oneapi::dpl::execution::make_device_policy<class custom_min_c_kernel<_DataType>>(DPNP_QUEUE);

        _DataType* res = std::min_element(policy, array_1, array_1 + size);
        policy.queue().wait();

        result[0] = *res;
    }
}

template void custom_min_c<double>(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);
template void custom_min_c<float>(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);
template void custom_min_c<long>(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);
template void custom_min_c<int>(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis);

template <typename _DataType, typename _ResultType>
void custom_std_c(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis, size_t ddof)
{
    _DataType* array1 = reinterpret_cast<_DataType*>(array1_in);
    _ResultType* result = reinterpret_cast<_ResultType*>(result1);

    _ResultType* var = reinterpret_cast<_ResultType*>(dpnp_memory_alloc_c(1 * sizeof(_ResultType)));
    custom_var_c<_DataType, _ResultType>(array1, var, shape, ndim, axis, naxis, ddof);

    custom_elemwise_sqrt_c<_ResultType, _ResultType>(var, result, 1);

    dpnp_memory_free_c(var);

#if 0
    std::cout << "std result " << res[0] << "\n";
#endif
}

template void custom_std_c<int, double>(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis, size_t ddof);
template void custom_std_c<long, double>(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis, size_t ddof);
template void custom_std_c<float, float>(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis, size_t ddof);
template void custom_std_c<double, double>(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis, size_t ddof);

template <typename _DataType, typename _ResultType>
class custom_var_c_kernel;

template <typename _DataType, typename _ResultType>
void custom_var_c(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis, size_t ddof)
{
    cl::sycl::event event;
    _DataType* array1 = reinterpret_cast<_DataType*>(array1_in);
    _ResultType* result = reinterpret_cast<_ResultType*>(result1);

    _ResultType* mean = reinterpret_cast<_ResultType*>(dpnp_memory_alloc_c(1 * sizeof(_ResultType)));
    custom_mean_c<_DataType, _ResultType>(array1, mean, shape, ndim, axis, naxis);
    _ResultType mean_val = mean[0];

    size_t size = 1;
    for (size_t i = 0; i < ndim; ++i)
    {
        size *= shape[i];
    }

    _ResultType* squared_deviations = reinterpret_cast<_ResultType*>(dpnp_memory_alloc_c(size * sizeof(_ResultType)));

    cl::sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
        size_t i = global_id[0]; /*for (size_t i = 0; i < size; ++i)*/
        {
            _ResultType deviation = static_cast<_ResultType>(array1[i]) - mean_val;
            squared_deviations[i] = deviation * deviation;
        }
    };

    auto kernel_func = [&](cl::sycl::handler& cgh) {
        cgh.parallel_for<class custom_var_c_kernel<_DataType, _ResultType>>(gws, kernel_parallel_for_func);
    };

    event = DPNP_QUEUE.submit(kernel_func);

    event.wait();

    custom_mean_c<_ResultType, _ResultType>(squared_deviations, mean, shape, ndim, axis, naxis);
    mean_val = mean[0];

    result[0] = mean_val * size / static_cast<_ResultType>(size - ddof);

    dpnp_memory_free_c(mean);
    dpnp_memory_free_c(squared_deviations);

#if 0
    std::cout << "var result " << res[0] << "\n";
#endif
}

template void custom_var_c<int, double>(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis, size_t ddof);
template void custom_var_c<long, double>(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis, size_t ddof);
template void custom_var_c<float, float>(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis, size_t ddof);
template void custom_var_c<double, double>(
    void* array1_in, void* result1, const size_t* shape, size_t ndim, const size_t* axis, size_t naxis, size_t ddof);
