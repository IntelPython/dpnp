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

#include <algorithm>
#include <iostream>
#include <mkl_blas_sycl.hpp>

#include <backend/backend_iface.hpp>
#include "queue_sycl.hpp"

template <typename _DataType>
void dpnp_blas_gemm_c(void* array1_in, void* array2_in, void* result1, size_t size_m, size_t size_n, size_t size_k)
{
    cl::sycl::event status;
    // using std::max for these ldx variables is required by MKL
    const std::int64_t lda = std::max<size_t>(1UL, size_k); // First dimensions of array_1
    const std::int64_t ldb = std::max<size_t>(1UL, size_n); // First dimensions of array_2
    const std::int64_t ldc = std::max<size_t>(1UL, size_n); // Fast dimensions of result

    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* array_2 = reinterpret_cast<_DataType*>(array2_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

#if 0
    std::cout << ">>>>>>>>>>>>>>>>MKL dpnp_blas_gemm_c parameters:"
              << "\n"
              << "lda=" << lda << "\n"
              << "ldb=" << ldb << "\n"
              << "ldc=" << ldc << "\n"
              << "size_m=" << size_m << "\n"
              << "size_n=" << size_n << "\n"
              << "size_k=" << size_k << "\n"
              << "alfa=" << _DataType(1) << "\n"
              << "beta=" << _DataType(0) << "\n"
              << std::endl;

    std::cout << "array_1\n";
    for (size_t it1 = 0; it1 < size_m; ++it1)
    {
        for (size_t it2 = 0; it2 < size_k; ++it2)
        {
            std::cout << " , " << array_1[it1*size_k + it2];
        }
        std::cout << std::endl;
    }

    std::cout << "array_2\n";
    for (size_t it1 = 0; it1 < size_k; ++it1)
    {
        for (size_t it2 = 0; it2 < size_n; ++it2)
        {
            std::cout << " , " << array_2[it1*size_n + it2];
        }
        std::cout << std::endl;
    }

    std::cout << "result_1 before\n";
    for (size_t it1 = 0; it1 < size_m; ++it1)
    {
        for (size_t it2 = 0; it2 < size_n; ++it2)
        {
            result[it1*size_n + it2] = 0;
            std::cout << " , " << result[it1*size_n + it2];
        }
        std::cout << std::endl;
    }
#endif
    try
    {
        status = oneapi::mkl::blas::gemm(DPNP_QUEUE,
                                         oneapi::mkl::transpose::nontrans,
                                         oneapi::mkl::transpose::nontrans,
                                         size_n,
                                         size_m,
                                         size_k,
                                         _DataType(1),
                                         array_2,
                                         ldb,
                                         array_1,
                                         lda,
                                         _DataType(0),
                                         result,
                                         ldc);
    }
    catch (cl::sycl::exception const& e)
    {
        std::cerr << "Caught synchronous SYCL exception during dpnp_blas_gemm_c():\n"
                  << e.what() << "\nOpenCL status: " << e.get_cl_code() << std::endl;
    }

    status.wait();
#if 0
    std::cout << "result_1 after\n";
    for (size_t it1 = 0; it1 < size_m; ++it1)
    {
        for (size_t it2 = 0; it2 < size_n; ++it2)
        {
            std::cout << " , " << result[it1*size_n + it2];
        }
        std::cout << std::endl;
    }
#endif
}

template void dpnp_blas_gemm_c<float>(
    void* array1_in, void* array2_in, void* result1, size_t size_m, size_t size_n, size_t size_k);
template void dpnp_blas_gemm_c<double>(
    void* array1_in, void* array2_in, void* result1, size_t size_m, size_t size_n, size_t size_k);
