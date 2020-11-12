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

#include <cmath>
#include <iostream>
#include <type_traits>

#include <backend_iface.hpp>
#include "backend_fptr.hpp"
#include "backend_utils.hpp"
#include "queue_sycl.hpp"

namespace mkl_blas = oneapi::mkl::blas;
namespace mkl_lapack = oneapi::mkl::lapack;

template <typename _KernelNameSpecialization>
class dpnp_matmul_c_kernel;

template <typename _DataType>
void dpnp_matmul_c(void* array1_in, void* array2_in, void* result1, size_t size_m, size_t size_n, size_t size_k)
{
    cl::sycl::event event;
    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* array_2 = reinterpret_cast<_DataType*>(array2_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    if (!size_m || !size_n || !size_k)
    {
        return;
    }

    if constexpr (std::is_same<_DataType, double>::value || std::is_same<_DataType, float>::value)
    {
        // using std::max for these ldx variables is required by math library
        const std::int64_t lda = std::max<size_t>(1UL, size_k); // First dimensions of array_1
        const std::int64_t ldb = std::max<size_t>(1UL, size_n); // First dimensions of array_2
        const std::int64_t ldc = std::max<size_t>(1UL, size_n); // Fast dimensions of result

        event = mkl_blas::gemm(DPNP_QUEUE,
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
    else
    {
        // input1: M x K
        // input2: K x N
        // result: M x N
        const size_t dim_m = size_m; // shape1.front(); // First dimensions of array1
        const size_t dim_n = size_n; // shape2.back();  // Last dimensions of array2
        const size_t dim_k = size_k; // shape1.back(); // First dimensions of array2

        cl::sycl::range<2> gws(dim_m, dim_n); // dimensions are: "i" and "j"

        auto kernel_parallel_for_func = [=](cl::sycl::id<2> global_id) {
            size_t i = global_id[0]; //for (size_t i = 0; i < size; ++i)
            {
                size_t j = global_id[1]; //for (size_t j = 0; j < size; ++j)
                {
                    _DataType acc = _DataType(0);
                    for (size_t k = 0; k < dim_k; ++k)
                    {
                        const size_t index_1 = i * dim_k + k;
                        const size_t index_2 = k * dim_n + j;
                        acc += array_1[index_1] * array_2[index_2];
                    }
                    const size_t index_result = i * dim_n + j;
                    result[index_result] = acc;
                }
            }
        };

        auto kernel_func = [&](cl::sycl::handler& cgh) {
            cgh.parallel_for<class dpnp_matmul_c_kernel<_DataType>>(gws, kernel_parallel_for_func);
        };

        event = DPNP_QUEUE.submit(kernel_func);
    }
    event.wait();
}

template <typename _KernelNameSpecialization1, typename _KernelNameSpecialization2, typename _KernelNameSpecialization3>
class dpnp_dot_c_kernel;

template <typename _DataType_input1, typename _DataType_input2, typename _DataType_output>
void dpnp_dot_c(void* array1_in, void* array2_in, void* result1, size_t size)
{
    cl::sycl::event event;
    _DataType_input1* array_1 = reinterpret_cast<_DataType_input1*>(array1_in);
    _DataType_input2* array_2 = reinterpret_cast<_DataType_input2*>(array2_in);
    _DataType_output* result = reinterpret_cast<_DataType_output*>(result1);

    if (!size)
    {
        return;
    }

    if constexpr ((std::is_same<_DataType_input1, double>::value || std::is_same<_DataType_input1, float>::value) &&
                  std::is_same<_DataType_input2, _DataType_input1>::value &&
                  std::is_same<_DataType_output, _DataType_input1>::value)
    {
        event = mkl_blas::dot(DPNP_QUEUE,
                              size,
                              array_1,
                              1, // array_1 stride
                              array_2,
                              1, // array_2 stride
                              result);
        event.wait();
    }
    else
    {
        _DataType_output* local_mem =
            reinterpret_cast<_DataType_output*>(dpnp_memory_alloc_c(size * sizeof(_DataType_output)));

        // what about reduction??
        cl::sycl::range<1> gws(size);

        auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
            const size_t index = global_id[0];
            local_mem[index] = array_1[index] * array_2[index];
        };

        auto kernel_func = [&](cl::sycl::handler& cgh) {
            cgh.parallel_for<class dpnp_dot_c_kernel<_DataType_input1, _DataType_input2, _DataType_output>>(
                gws, kernel_parallel_for_func);
        };

        event = DPNP_QUEUE.submit(kernel_func);

        event.wait();

        auto policy = oneapi::dpl::execution::make_device_policy<
            class dpnp_dot_c_kernel<_DataType_input1, _DataType_input2, _DataType_output>>(DPNP_QUEUE);

        _DataType_output accumulator = 0;
        accumulator =
            std::reduce(policy, local_mem, local_mem + size, _DataType_output(0), std::plus<_DataType_output>());
        policy.queue().wait();

        result[0] = accumulator;

        free(local_mem, DPNP_QUEUE);
    }
}

template <typename _DataType, typename _ResultType>
void dpnp_eig_c(const void* array_in, void* result1, void* result2, size_t size)
{
    // TODO this kernel works with square 2-D array only

    // Kernel Type for calculation is double type
    // because interface requires float type but calculations are expected in double type

    if (!size)
    {
        return;
    }

    cl::sycl::event event;

    const _DataType* array = reinterpret_cast<const _DataType*>(array_in);
    _ResultType* result_val = reinterpret_cast<_ResultType*>(result1);
    _ResultType* result_vec = reinterpret_cast<_ResultType*>(result2);

    double* result_val_kern = reinterpret_cast<double*>(dpnp_memory_alloc_c(size * sizeof(double)));
    double* result_vec_kern = reinterpret_cast<double*>(dpnp_memory_alloc_c(size * size * sizeof(double)));

    // type conversion. Also, math library requires copy memory because override
    for (size_t it = 0; it < (size * size); ++it)
    {
        result_vec_kern[it] = array[it];
    }

    const std::int64_t lda = std::max<size_t>(1UL, size);

    const std::int64_t scratchpad_size = mkl_lapack::syevd_scratchpad_size<double>(
        DPNP_QUEUE, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, size, lda);

    double* scratchpad = reinterpret_cast<double*>(dpnp_memory_alloc_c(scratchpad_size * sizeof(double)));

    event = mkl_lapack::syevd(DPNP_QUEUE,               // queue
                              oneapi::mkl::job::vec,    // jobz
                              oneapi::mkl::uplo::upper, // uplo
                              size,                     // The order of the matrix A (0 <= n)
                              result_vec_kern,          // will be overwritten with eigenvectors
                              lda,
                              result_val_kern,
                              scratchpad,
                              scratchpad_size);
    event.wait();

    dpnp_memory_free_c(scratchpad);

    for (size_t it1 = 0; it1 < size; ++it1)
    {
        result_val[it1] = result_val_kern[it1];
        for (size_t it2 = 0; it2 < size; ++it2)
        {
            // copy + transpose
            result_vec[it2 * size + it1] = result_vec_kern[it1 * size + it2];
        }
    }

    dpnp_memory_free_c(result_val_kern);
    dpnp_memory_free_c(result_vec_kern);
}

void func_map_init_linalg(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_dot_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_dot_c<int, long, long>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_dot_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_dot_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_dot_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_dot_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_dot_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_dot_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_dot_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_dot_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_dot_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_dot_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_dot_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_dot_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_dot_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_dot_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_EIG][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_eig_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_EIG][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_eig_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_EIG][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_eig_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_EIG][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_eig_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_MATMUL][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_matmul_c<int>};
    fmap[DPNPFuncName::DPNP_FN_MATMUL][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_matmul_c<long>};
    fmap[DPNPFuncName::DPNP_FN_MATMUL][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_matmul_c<float>};
    fmap[DPNPFuncName::DPNP_FN_MATMUL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_matmul_c<double>};

    return;
}
