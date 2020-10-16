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

#include <backend_iface.hpp>
#include "backend_pstl.hpp"
#include "backend_utils.hpp"
#include "queue_sycl.hpp"

namespace mkl_blas = oneapi::mkl::blas::row_major;

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
    event = DPNP_QUEUE.submit([&](cl::sycl::handler& cgh) {
            cgh.parallel_for<class custom_cov_c_kernel<_DataType> >(
                gws,
                [=](cl::sycl::id<1> global_id)
            {
                const size_t idx = global_id[0];
                const size_t row_idx = idx / nrows;
                const size_t col_idx = idx - row_idx * nrows;
                if (col_idx < row_idx)
                {
                    result[idx] = result[col_idx * nrows + row_idx];
                }
            }); // parallel_for
    });         // queue.submit
    event.wait();

    dpnp_memory_free_c(mean);
    dpnp_memory_free_c(temp);
}

template void custom_cov_c<double>(void* array1_in, void* result1, size_t nrows, size_t ncols);

template <typename _DataType>
class custom_max_c_kernel;

template <typename _DataType>
void custom_max_c(void* array1_in, void* result1, size_t* shape, size_t ndim, size_t* axis, size_t naxis)
{
    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    size_t size = 1;
    for (size_t i = 0; i < ndim; ++i)
    {
        size *= shape[i];
    }

    auto policy = oneapi::dpl::execution::make_device_policy<class custom_max_c_kernel<_DataType>>(DPNP_QUEUE);

    _DataType* res = std::max_element(policy, array_1, array_1 + size);
    policy.queue().wait();

    result[0] = *res;

#if 0
    std::cout << "max result " << result[0] << "\n";
#endif
}

template void
    custom_max_c<double>(void* array1_in, void* result1, size_t* shape, size_t ndim, size_t* axis, size_t naxis);
template void
    custom_max_c<float>(void* array1_in, void* result1, size_t* shape, size_t ndim, size_t* axis, size_t naxis);
template void
    custom_max_c<long>(void* array1_in, void* result1, size_t* shape, size_t ndim, size_t* axis, size_t naxis);
template void custom_max_c<int>(void* array1_in, void* result1, size_t* shape, size_t ndim, size_t* axis, size_t naxis);

template <typename _DataType, typename _ResultType>
void custom_mean_c(void* array1_in, void* result1, size_t* shape, size_t ndim, size_t* axis, size_t naxis)
{
    _ResultType* result = reinterpret_cast<_ResultType*>(result1);

    size_t size = 1;
    for (size_t i = 0; i < ndim; ++i)
    {
        size *= shape[i];
    }

    _DataType* sum = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(1 * sizeof(_DataType)));

    custom_sum_c<_DataType>(array1_in, sum, size);

    result[0] = (_ResultType)(sum[0]) / size;

    dpnp_memory_free_c(sum);

#if 0
    std::cout << "mean result " << result[0] << "\n";
#endif
}

template void custom_mean_c<double, double>(
    void* array1_in, void* result1, size_t* shape, size_t ndim, size_t* axis, size_t naxis);
template void
    custom_mean_c<float, float>(void* array1_in, void* result1, size_t* shape, size_t ndim, size_t* axis, size_t naxis);
template void
    custom_mean_c<long, double>(void* array1_in, void* result1, size_t* shape, size_t ndim, size_t* axis, size_t naxis);
template void
    custom_mean_c<int, double>(void* array1_in, void* result1, size_t* shape, size_t ndim, size_t* axis, size_t naxis);

template <typename _DataType, typename _ResultType>
void custom_median_c(void* array1_in, void* result1, size_t* shape, size_t ndim, size_t* axis, size_t naxis)
{
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
        result[0] = (_ResultType)(sorted[size / 2] + sorted[size / 2 - 1]) / 2;
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
    void* array1_in, void* result1, size_t* shape, size_t ndim, size_t* axis, size_t naxis);
template void custom_median_c<float, double>(
    void* array1_in, void* result1, size_t* shape, size_t ndim, size_t* axis, size_t naxis);
template void custom_median_c<long, double>(
    void* array1_in, void* result1, size_t* shape, size_t ndim, size_t* axis, size_t naxis);
template void custom_median_c<int, double>(
    void* array1_in, void* result1, size_t* shape, size_t ndim, size_t* axis, size_t naxis);

template <typename _DataType>
class custom_min_c_kernel;

template <typename _DataType>
void custom_min_c(void* array1_in, void* result1, size_t* shape, size_t ndim, size_t* axis, size_t naxis)
{
    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    size_t size = 1;
    for (size_t i = 0; i < ndim; ++i)
    {
        size *= shape[i];
    }

    auto policy = oneapi::dpl::execution::make_device_policy<class custom_min_c_kernel<_DataType>>(DPNP_QUEUE);

    _DataType* res = std::min_element(policy, array_1, array_1 + size);
    policy.queue().wait();

    result[0] = *res;

#if 0
    std::cout << "min result " << result[0] << "\n";
#endif
}

template void
    custom_min_c<double>(void* array1_in, void* result1, size_t* shape, size_t ndim, size_t* axis, size_t naxis);
template void
    custom_min_c<float>(void* array1_in, void* result1, size_t* shape, size_t ndim, size_t* axis, size_t naxis);
template void
    custom_min_c<long>(void* array1_in, void* result1, size_t* shape, size_t ndim, size_t* axis, size_t naxis);
template void custom_min_c<int>(void* array1_in, void* result1, size_t* shape, size_t ndim, size_t* axis, size_t naxis);

template <typename _DataType>
class custom_min_axis_c_kernel;

template <typename _DataType>
void custom_min_axis_c(void* array1_in,
                       void* result1,
                       size_t* shape,
                       size_t ndim,
                       size_t* axis,
                       size_t naxis)
{
    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    size_t res_ndim = ndim - naxis;
    size_t res_shape[res_ndim];
    int ind = 0;
    for (size_t i = 0; i < ndim; i++)
    {
        bool found = false;
        for (size_t j = 0; j < naxis; j++)
        {
            if (axis[j] == i)
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            res_shape[ind] = shape[i];
            ind++;
        }
    }

    size_t size_input = 1;
    for (size_t i = 0; i < ndim; ++i)
    {
        size_input *= shape[i];
    }

    size_t input_shape_offsets[ndim];
    size_t acc = 1;
    for (size_t i = ndim - 1; i > 0; --i)
    {
        input_shape_offsets[i] = acc;
        acc *= shape[i];
    }
    input_shape_offsets[0] = acc;

    size_t output_shape_offsets[res_ndim];
    acc = 1;
    for (size_t i = res_ndim - 1; i > 0; --i)
    {
        output_shape_offsets[i] = acc;
        acc *= res_shape[i];
    }
    output_shape_offsets[0] = acc;

    size_t size_result = 1;
    for (size_t i = 0; i < res_ndim; ++i)
    {
        size_result *= res_shape[i];
    }

    //init result array
    for (size_t result_idx = 0; result_idx < size_result; ++result_idx)
    {
        size_t xyz[res_ndim];
        size_t remainder = result_idx;
        for (size_t i = 0; i < res_ndim; ++i)
        {
            xyz[i] = remainder / output_shape_offsets[i];
            remainder = remainder - xyz[i] * output_shape_offsets[i];
        }

        size_t source_axis[ndim];
        size_t result_axis_idx = 0;
        for (size_t idx = 0; idx < ndim; ++idx)
        {
            bool found = false;
            for (size_t i = 0; i < naxis; ++i)
            {
                if (axis[i] == idx)
                {
                    found = true;
                    break;
                }
            }
            if (found)
            {
                source_axis[idx] = 0;
            }
            else
            {
                source_axis[idx] = xyz[result_axis_idx];
                result_axis_idx++;
            }
        }

        size_t source_idx = 0;
        for (size_t i = 0; i < ndim; ++i)
        {
            source_idx += input_shape_offsets[i] * source_axis[i];
        }

        result[result_idx] = array_1[source_idx];
    }

    for (size_t source_idx = 0; source_idx < size_input; ++source_idx)
    {
        // reconstruct x,y,z from linear source_idx
        size_t xyz[ndim];
        size_t remainder = source_idx;
        for (size_t i = 0; i < ndim; ++i)
        {
            xyz[i] = remainder / input_shape_offsets[i];
            remainder = remainder - xyz[i] * input_shape_offsets[i];
        }

        // extract result axis
        size_t result_axis[res_ndim];
        size_t result_idx = 0;
        for (size_t idx = 0; idx < ndim; ++idx)
        {
            // try to find current idx in axis array
            bool found = false;
            for (size_t i = 0; i < naxis; ++i)
            {
                if (axis[i] == idx)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                result_axis[result_idx] = xyz[idx];
                result_idx++;
            }
        }

        // Construct result offset
        size_t result_offset = 0;
        for (size_t i = 0; i < res_ndim; ++i)
        {
            result_offset += output_shape_offsets[i] * result_axis[i];
        }

        if (result[result_offset] > array_1[source_idx])
        {
            result[result_offset] = array_1[source_idx];
        }
    }

#if 0
    std::cout << "min result " << result_array << "\n";
#endif
}

template void custom_min_axis_c<double>(void* array1_in,
                                        void* result1,
                                        size_t* shape,
                                        size_t ndim,
                                        size_t* axis,
                                        size_t naxis);
template void custom_min_axis_c<float>(void* array1_in,
                                       void* result1,
                                       size_t* shape,
                                       size_t ndim,
                                       size_t* axis,
                                       size_t naxis);
template void custom_min_axis_c<long>(void* array1_in,
                                      void* result1,
                                      size_t* shape,
                                      size_t ndim,
                                      size_t* axis,
                                      size_t naxis);
template void custom_min_axis_c<int>(void* array1_in,
                                     void* result1,
                                     size_t* shape,
                                     size_t ndim,
                                     size_t* axis,
                                     size_t naxis);
