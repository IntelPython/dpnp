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
