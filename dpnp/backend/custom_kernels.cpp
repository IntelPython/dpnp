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
#include <mkl_blas_sycl.hpp>

#include <backend/backend_iface.hpp>
#include "backend_pstl.hpp"
#include "backend_utils.hpp"
#include "queue_sycl.hpp"

template <typename _KernelNameSpecialization>
class custom_blas_gemm_c_kernel;

template <typename _DataType>
void custom_blas_gemm_c(void* array1_in, void* array2_in, void* result1, size_t size_m, size_t size_n, size_t size_k)
{
    cl::sycl::event event;
    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* array_2 = reinterpret_cast<_DataType*>(array2_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    // input1: M x K
    // input2: K x N
    // result: M x N
    const size_t dim_m = size_m; // shape1.front(); // First dimensions of array1
    const size_t dim_n = size_n; // shape2.back();  // Last dimensions of array2
    const size_t dim_k = size_k; // shape1.back(); // First dimensions of array2

    cl::sycl::range<2> gws(dim_m, dim_n); // dimensions are: "i" and "j"
    event = DPNP_QUEUE.submit([&](cl::sycl::handler& cgh) {
            cgh.parallel_for<class custom_blas_gemm_c_kernel<_DataType> >(
                gws,
                [=](cl::sycl::id<2> global_id)
            {
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
            }); // parallel_for
    });         // queue.submit

    event.wait();
}

template void custom_blas_gemm_c<long>(
    void* array1_in, void* array2_in, void* result1, size_t size_m, size_t size_n, size_t size_k);
template void custom_blas_gemm_c<int>(
    void* array1_in, void* array2_in, void* result1, size_t size_m, size_t size_n, size_t size_k);

template <typename _KernelNameSpecialization>
class custom_blas_dot_c_kernel;

template <typename _DataType>
void custom_blas_dot_c(void* array1_in, void* array2_in, void* result1, size_t size)
{
    cl::sycl::event event;
    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* array_2 = reinterpret_cast<_DataType*>(array2_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    _DataType* local_mem = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(size * sizeof(_DataType)));

    // what about reduction??
    cl::sycl::range<1> gws(size);
    event = DPNP_QUEUE.submit([&](cl::sycl::handler& cgh) {
            cgh.parallel_for<class custom_blas_dot_c_kernel<_DataType> >(gws, [=](cl::sycl::id<1> global_id)
            {
                const size_t index = global_id[0];
                local_mem[index] = array_1[index] * array_2[index];
            }     // kernel lambda
            );    // parallel_for
    }             // task lambda
    );            // queue.submit

    event.wait();

    for (size_t i = 1; i < size; ++i)
    {
        local_mem[0] += local_mem[i];
    }
    result[0] = local_mem[0];
    free(local_mem, DPNP_QUEUE);
}

template void custom_blas_dot_c<long>(void* array1_in, void* array2_in, void* result1, size_t size);
template void custom_blas_dot_c<int>(void* array1_in, void* array2_in, void* result1, size_t size);

template <typename _DataType, typename _idx_DataType>
struct _argsort_less
{
    _argsort_less(_DataType* data_ptr)
    {
        _data_ptr = data_ptr;
    }

    inline bool operator()(const _idx_DataType& idx1, const _idx_DataType& idx2)
    {
        return (_data_ptr[idx1] < _data_ptr[idx2]);
    }

private:
    _DataType* _data_ptr = nullptr;
};

// necessary for creating separate kernels inside execution with policy
template <typename _DataType, typename _idx_DataType>
struct device_policy_tmplt
{
};

template <typename _DataType, typename _idx_DataType>
void custom_argsort_c(void* array1_in, void* result1, size_t size)
{
    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _idx_DataType* result = reinterpret_cast<_idx_DataType*>(result1);

    for (size_t i = 0; i < size; ++i)
    {
        result[i] = i;
    }

    auto queue = DPNP_QUEUE;

    auto policy = oneapi::dpl::execution::make_device_policy<device_policy_tmplt<_DataType, _idx_DataType>>(queue);

    std::sort(policy, result, result + size, _argsort_less<_DataType, _idx_DataType>(array_1));

    queue.wait_and_throw(); // looks like it is necessary to sync after call of pstl
}

template void custom_argsort_c<double, long>(void* array1_in, void* result1, size_t size);
template void custom_argsort_c<float, long>(void* array1_in, void* result1, size_t size);
template void custom_argsort_c<long, long>(void* array1_in, void* result1, size_t size);
template void custom_argsort_c<int, long>(void* array1_in, void* result1, size_t size);
template void custom_argsort_c<double, int>(void* array1_in, void* result1, size_t size);
template void custom_argsort_c<float, int>(void* array1_in, void* result1, size_t size);
template void custom_argsort_c<long, int>(void* array1_in, void* result1, size_t size);
template void custom_argsort_c<int, int>(void* array1_in, void* result1, size_t size);

#if 0 // Example for OpenCL kernel
#include <map>
#include <typeindex>

static std::map<std::type_index, std::string> types_map = {{typeid(long), "long"}, {typeid(int), "int"}};

static const char* blas_gemm_naive =
    "//#define __KERNEL_TYPE__ long                                                \n"
    "#define __KERNEL_TYPE_ZERO__ 0                                                \n"
    "__kernel void blas_gemm_naive(__global __KERNEL_TYPE__* array_1,              \n"
    "                              __global __KERNEL_TYPE__* array_2,              \n"
    "                              __global __KERNEL_TYPE__* result,               \n"
    "                              unsigned long size)                             \n"
    "{                                                                             \n"
    "    size_t i = get_global_id(0); //for (size_t i = 0; i < size; ++i)          \n"
    "    {                                                                         \n"
    "        size_t j = get_global_id(1); //for (size_t j = 0; j < size; ++j)      \n"
    "        {                                                                     \n"
    "            __KERNEL_TYPE__ temp = __KERNEL_TYPE_ZERO__;                      \n"
    "            for (size_t k = 0; k < size; ++k)                                 \n"
    "            {                                                                 \n"
    "                const size_t index_1 = i * size + k;                          \n"
    "                const size_t index_2 = k * size + j;                          \n"
    "                temp += array_1[index_1] * array_2[index_2];                  \n"
    "            }                                                                 \n"
    "                                                                              \n"
    "            const size_t index_result = i * size + j;                         \n"
    "            result[index_result] = temp;                                      \n"
    "        }                                                                     \n"
    "    }                                                                         \n"
    "}                                                                             \n";

template <typename _DataType>
void custom_dgemm_c_opencl(void* array_1_in, void* array_2_in, void* result_1, size_t size)
{
    _DataType* array_1 = reinterpret_cast<_DataType*>(array_1_in);
    _DataType* array_2 = reinterpret_cast<_DataType*>(array_2_in);
    _DataType* result = reinterpret_cast<_DataType*>(result_1);

    std::string compile_time_options("-cl-std=CL1.2");
    compile_time_options += " -D__KERNEL_TYPE__=" + types_map.at(typeid(_DataType));

    cl::sycl::program program_src(DPNP_QUEUE.get_context());
    program_src.build_with_source(blas_gemm_naive, compile_time_options);

    cl::sycl::range<2> kernel_work_ids(size, size); // dimensions are: "i" and "j"
    DPNP_QUEUE.submit([&](cl::sycl::handler& cgh) {
        cgh.set_args(array_1, array_2, result, size);
        cgh.parallel_for(kernel_work_ids, program_src.get_kernel("blas_gemm_naive"));
    });

    DPNP_QUEUE.wait();
}

template void custom_dgemm_c_opencl<long>(void* array_1_in, void* array_2_in, void* result_1, size_t size);

#endif
