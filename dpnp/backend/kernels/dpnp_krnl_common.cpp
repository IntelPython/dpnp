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

#include <dpnp_iface.hpp>
#include "dpnp_fptr.hpp"
#include "dpnp_utils.hpp"
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"

namespace mkl_blas = oneapi::mkl::blas;
namespace mkl_blas_rm = oneapi::mkl::blas::row_major;
namespace mkl_lapack = oneapi::mkl::lapack;

template <typename _DataType, typename _ResultType>
class dpnp_astype_c_kernel;

template <typename _DataType, typename _ResultType>
void dpnp_astype_c(const void* array1_in, void* result1, const size_t size)
{
    cl::sycl::event event;
    DPNPC_ptr_adapter<_DataType> input1_ptr(array1_in, size);
    const _DataType* array_in = input1_ptr.get_ptr();
    _ResultType* result = reinterpret_cast<_ResultType*>(result1);

    if ((array_in == nullptr) || (result == nullptr))
    {
        return;
    }

    if (size == 0)
    {
        return;
    }

    cl::sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
        size_t i = global_id[0];
        result[i] = array_in[i];
    };

    auto kernel_func = [&](cl::sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_astype_c_kernel<_DataType, _ResultType>>(gws, kernel_parallel_for_func);
    };

    event = DPNP_QUEUE.submit(kernel_func);

    event.wait();
}

template <typename _KernelNameSpecialization1, typename _KernelNameSpecialization2, typename _KernelNameSpecialization3>
class dpnp_dot_c_kernel;

template <typename _DataType_output, typename _DataType_input1, typename _DataType_input2>
cl::sycl::event dot(cl::sycl::queue& queue,
                    _DataType_output* result_out,
                    _DataType_input1* input1_in,
                    _DataType_input2* input2_in,
                    size_t input1_strides,
                    size_t input2_strides,
                    size_t size,
                    const cl::sycl::vector_class<cl::sycl::event>& dependencies = {})
{
    (void)dependencies;

    cl::sycl::event event;

    if constexpr ((std::is_same<_DataType_input1, double>::value || std::is_same<_DataType_input1, float>::value) &&
                  std::is_same<_DataType_input2, _DataType_input1>::value &&
                  std::is_same<_DataType_output, _DataType_input1>::value)
    {
        event = oneapi::mkl::blas::dot(queue,
                                       size,
                                       input1_in,
                                       input1_strides, // input1 stride
                                       input2_in,
                                       input2_strides, // input2 stride
                                       result_out);
    }
    else
    {
#if LIBSYCL_VERSION_GREATER(5, 3, 0)
        event = queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>{size},
                             cl::sycl::reduction(result_out,
                                                 std::plus<_DataType_output>(),
                                                 cl::sycl::property::reduction::initialize_to_identity{}),
                             [=](cl::sycl::id<1> idx, auto& sum) {
                                 sum += static_cast<_DataType_output>(input1_in[idx * input1_strides]) *
                                        static_cast<_DataType_output>(input2_in[idx * input2_strides]);
                             });
        });
        // for some reason few such kernels cannot work in parallel
        // looks like a bug in level0 because with opencl works fine
        // that is why we call wait here
        event.wait();
#else
        _DataType_output* local_mem =
            reinterpret_cast<_DataType_output*>(dpnp_memory_alloc_c(size * sizeof(_DataType_output)));

        // what about reduction??
        cl::sycl::range<1> gws(size);

        auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
            const size_t index = global_id[0];
            local_mem[index] = input1_in[index * input1_strides] * input2_in[index * input2_strides];
        };

        auto kernel_func = [&](cl::sycl::handler& cgh) {
            cgh.parallel_for<class dpnp_dot_c_kernel<_DataType_output, _DataType_input1, _DataType_input2>>(
                gws, kernel_parallel_for_func);
        };

        event = DPNP_QUEUE.submit(kernel_func);

        event.wait();

        auto policy = oneapi::dpl::execution::make_device_policy<
            class dpnp_dot_c_kernel<_DataType_output, _DataType_input1, _DataType_input2>>(DPNP_QUEUE);

        _DataType_output accumulator = 0;
        accumulator =
            std::reduce(policy, local_mem, local_mem + size, _DataType_output(0), std::plus<_DataType_output>());
        policy.queue().wait();

        dpnp_memory_memcpy_c(result_out, &accumulator, sizeof(_DataType_output)); // result[0] = accumulator;

        free(local_mem, DPNP_QUEUE);
#endif
    }
    return event;
}

template <typename _DataType_output, typename _DataType_input1, typename _DataType_input2>
void dpnp_dot_c(void* result_out,
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
                const size_t* input2_strides)
{
    (void)result_strides;

    DPNPC_ptr_adapter<_DataType_input1> input1_ptr(input1_in, input1_size);
    DPNPC_ptr_adapter<_DataType_input2> input2_ptr(input2_in, input2_size);

    _DataType_input1* input1 = input1_ptr.get_ptr();
    _DataType_input2* input2 = input2_ptr.get_ptr();
    _DataType_output* result = reinterpret_cast<_DataType_output*>(result_out);

    if (!input1_size || !input2_size)
    {
        _DataType_output val = _DataType_output(0);
        dpnp_initval_c<_DataType_output>(result, &val, result_size);
        return;
    }

    // scalar
    if ((input1_ndim == 0) || (input2_ndim == 0))
    {
        // there is no support of strides in multiply function
        // so result can be wrong if input array has non-standard (c-contiguous) strides
        dpnp_multiply_c<_DataType_output, _DataType_input1, _DataType_input2>(result,
                                                                              input1_in,
                                                                              input1_size,
                                                                              input1_shape,
                                                                              input1_ndim,
                                                                              input2_in,
                                                                              input2_size,
                                                                              input2_shape,
                                                                              input2_ndim,
                                                                              NULL);
        return;
    }

    // if both arrays are vectors
    if ((input1_ndim == 1) && (input2_ndim == 1))
    {
        assert(input1_size == input2_size);
        cl::sycl::event event =
            dot(DPNP_QUEUE, result, input1, input2, input1_strides[0], input2_strides[0], input1_size);
        event.wait();
        return;
    }

    // 1D vector
    size_t ext_input1_ndim = input1_ndim == 1 ? 2 : input1_ndim;
    size_t* ext_input1_shape = new size_t[ext_input1_ndim];
    size_t* ext_input1_strides = new size_t[ext_input1_ndim];
    if (input1_ndim == 1)
    {
        ext_input1_shape[0] = 1;
        ext_input1_shape[1] = input1_shape[0];
        ext_input1_strides[0] = 0;
        ext_input1_strides[1] = input1_strides[0];
    }
    else
    {
        for (size_t i = 0; i < ext_input1_ndim; ++i)
        {
            ext_input1_shape[i] = input1_shape[i];
            ext_input1_strides[i] = input1_strides[i];
        }
    }
    size_t ext_input2_ndim = input2_ndim == 1 ? 2 : input2_ndim;
    size_t* ext_input2_shape = new size_t[ext_input2_ndim];
    size_t* ext_input2_strides = new size_t[ext_input2_ndim];
    if (input2_ndim == 1)
    {
        ext_input2_shape[0] = input2_shape[0];
        ext_input2_shape[1] = 1;
        ext_input2_strides[0] = input2_strides[0];
        ext_input2_strides[1] = 0;
    }
    else
    {
        for (size_t i = 0; i < ext_input2_ndim; ++i)
        {
            ext_input2_shape[i] = input2_shape[i];
            ext_input2_strides[i] = input2_strides[i];
        }
    }
    size_t ext_result_ndim = ((input1_ndim == 1) || (input2_ndim == 1)) ? 2 : result_ndim;
    size_t* ext_result_shape = new size_t[ext_result_ndim];
    if ((input1_ndim == 1) || (input2_ndim == 1))
    {
        ext_result_shape[0] = ext_input1_shape[0];
        ext_result_shape[1] = ext_input2_shape[1];
    }
    else
    {
        for (size_t i = 0; i < ext_result_ndim; ++i)
        {
            ext_result_shape[i] = result_shape[i];
        }
    }

    // check if GEMM can be executed (types)
    if constexpr ((std::is_same<_DataType_input1, double>::value || std::is_same<_DataType_input1, float>::value) &&
                  std::is_same<_DataType_input2, _DataType_input1>::value &&
                  std::is_same<_DataType_output, _DataType_input1>::value)
    {
        // check if GEMM can be executed (strides)
        // TODO: rewrite the condition in general case for ndims > 2
        // (looks like there are such another cases)
        if ((ext_input1_ndim == 2 && ext_input2_ndim == 2) &&
            (ext_input1_strides[0] == 1 || ext_input1_strides[1] == 1) &&
            (ext_input2_strides[0] == 1 || ext_input2_strides[1] == 1))
        {
// there is a difference of behavior with trans and sizes params in previous version of GEMM
// only new version is supported, in case of old version computation goes in common way
#if INTEL_MKL_VERSION >= 20210004
            oneapi::mkl::transpose trans1 =
                ext_input1_strides[0] == 1 ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;
            oneapi::mkl::transpose trans2 =
                ext_input2_strides[0] == 1 ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;

            const size_t size_m = ext_input1_shape[0];
            const size_t size_n = ext_input2_shape[1];
            const size_t size_k = ext_input1_shape[1];

            const std::int64_t lda =
                trans1 == oneapi::mkl::transpose::nontrans ? ext_input1_strides[0] : ext_input1_strides[1];
            const std::int64_t ldb =
                trans2 == oneapi::mkl::transpose::nontrans ? ext_input2_strides[0] : ext_input2_strides[1];
            ;
            // defenition of ldc will be another for result with non-standard (c-contiguous) strides
            // const std::int64_t ldc = result_strides[0] == 1 ? result_strides[1] : result_strides[0];
            const std::int64_t ldc = size_n;

            cl::sycl::event event = mkl_blas_rm::gemm(DPNP_QUEUE,
                                                      trans1,
                                                      trans2,
                                                      size_m,
                                                      size_n,
                                                      size_k,
                                                      _DataType_output(1), // alpha
                                                      input1,
                                                      lda,
                                                      input2,
                                                      ldb,
                                                      _DataType_output(0), // beta
                                                      result,
                                                      ldc);
            event.wait();
            return;
#endif
        }
    }

    // deprecated? can be replaced with std::vector<cl::sycl::event>
    cl::sycl::vector_class<cl::sycl::event> dot_events;
    // std::vector<cl::sycl::event> dot_events;
    dot_events.reserve(result_size);

    size_t dot_st1 = ext_input1_strides[ext_input1_ndim - 1];
    size_t dot_st2 = ext_input2_strides[ext_input2_ndim - 2];
    size_t dot_size = ext_input1_shape[ext_input1_ndim - 1];

    size_t* res_coords = new size_t[ext_result_ndim];
    size_t* result_offsets = new size_t[ext_result_ndim];
    get_shape_offsets_inkernel(ext_result_shape, ext_result_ndim, result_offsets);

    for (size_t i = 0; i < result_size; ++i)
    {
        get_xyz_by_id(i, ext_result_ndim, result_offsets, res_coords);

        _DataType_output* dot_res = result + i;

        _DataType_input1* dot_in1 = input1;
        for (size_t j = 0; j < ext_input1_ndim - 1; ++j)
        {
            dot_in1 = dot_in1 + res_coords[j] * ext_input1_strides[j];
        }

        _DataType_input2* dot_in2 = input2;
        for (size_t j = 0; j < ext_input2_ndim - 2; ++j)
        {
            dot_in2 = dot_in2 + res_coords[ext_input1_ndim - 1 + j] * ext_input2_strides[j];
        }
        dot_in2 = dot_in2 + res_coords[ext_input1_ndim + ext_input2_ndim - 3] * ext_input2_strides[ext_input2_ndim - 1];

        dot_events.push_back(dot(DPNP_QUEUE, dot_res, dot_in1, dot_in2, dot_st1, dot_st2, dot_size));
    }

    sycl::event::wait(dot_events);

    delete[] res_coords;
    delete[] result_offsets;
    delete[] ext_input1_shape;
    delete[] ext_input1_strides;
    delete[] ext_input2_shape;
    delete[] ext_input2_strides;
    delete[] ext_result_shape;
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
    DPNPC_ptr_adapter<_DataType> input1_ptr(array_in, size * size, true);
    DPNPC_ptr_adapter<_ResultType> result1_ptr(result1, size, true, true);
    DPNPC_ptr_adapter<_ResultType> result2_ptr(result2, size * size, true, true);
    const _DataType* array = input1_ptr.get_ptr();
    _ResultType* result_val = result1_ptr.get_ptr();
    _ResultType* result_vec = result2_ptr.get_ptr();

    double* result_val_kern = reinterpret_cast<double*>(dpnp_memory_alloc_c(size * sizeof(double)));
    double* result_vec_kern = reinterpret_cast<double*>(dpnp_memory_alloc_c(size * size * sizeof(double)));

    // type conversion. Also, math library requires copy memory because override
    for (size_t it = 0; it < (size * size); ++it)
    {
        result_vec_kern[it] = array[it]; // TODO use memcpy_c or input1_ptr(array_in, size, true)
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
        result_val[it1] = result_val_kern[it1]; // TODO use memcpy_c or dpnpc_transpose_c
        for (size_t it2 = 0; it2 < size; ++it2)
        {
            // copy + transpose
            result_vec[it2 * size + it1] = result_vec_kern[it1 * size + it2];
        }
    }

    dpnp_memory_free_c(result_val_kern);
    dpnp_memory_free_c(result_vec_kern);
}

template <typename _DataType, typename _ResultType>
void dpnp_eigvals_c(const void* array_in, void* result1, size_t size)
{
    // TODO this kernel works with square 2-D array only

    // Kernel Type for calculation is double type
    // because interface requires float type but calculations are expected in double type

    if (!size)
    {
        return;
    }

    cl::sycl::event event;
    DPNPC_ptr_adapter<_DataType> input1_ptr(array_in, size * size, true);
    DPNPC_ptr_adapter<_ResultType> result1_ptr(result1, size, true, true);
    const _DataType* array = input1_ptr.get_ptr();
    _ResultType* result_val = result1_ptr.get_ptr();

    double* result_val_kern = reinterpret_cast<double*>(dpnp_memory_alloc_c(size * sizeof(double)));
    double* result_vec_kern = reinterpret_cast<double*>(dpnp_memory_alloc_c(size * size * sizeof(double)));

    // type conversion. Also, math library requires copy memory because override
    for (size_t it = 0; it < (size * size); ++it)
    {
        result_vec_kern[it] = array[it]; // TODO same as previous kernel
    }

    const std::int64_t lda = std::max<size_t>(1UL, size);

    const std::int64_t scratchpad_size = mkl_lapack::syevd_scratchpad_size<double>(
        DPNP_QUEUE, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, size, lda);

    double* scratchpad = reinterpret_cast<double*>(dpnp_memory_alloc_c(scratchpad_size * sizeof(double)));

    event = mkl_lapack::syevd(DPNP_QUEUE,               // queue
                              oneapi::mkl::job::vec,    // jobz
                              oneapi::mkl::uplo::upper, // uplo
                              size,                     // The order of the matrix A (0 <= n)
                              result_vec_kern,
                              lda,
                              result_val_kern,
                              scratchpad,
                              scratchpad_size);
    event.wait();

    dpnp_memory_free_c(scratchpad);

    for (size_t it1 = 0; it1 < size; ++it1)
    {
        result_val[it1] = result_val_kern[it1];
    }

    dpnp_memory_free_c(result_val_kern);
}

template <typename _DataType>
class dpnp_initval_c_kernel;

template <typename _DataType>
void dpnp_initval_c(void* result1, void* value, size_t size)
{
    if (!size)
    {
        return;
    }

    DPNPC_ptr_adapter<_DataType> result1_ptr(result1, size);
    DPNPC_ptr_adapter<_DataType> value_ptr(value, 1);
    _DataType* result = result1_ptr.get_ptr();
    _DataType* val = value_ptr.get_ptr();

    cl::sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
        const size_t idx = global_id[0];
        result[idx] = *val;
    };

    auto kernel_func = [&](cl::sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_initval_c_kernel<_DataType>>(gws, kernel_parallel_for_func);
    };

    cl::sycl::event event = DPNP_QUEUE.submit(kernel_func);

    event.wait();
}

template <typename _KernelNameSpecialization>
class dpnp_matmul_c_kernel;

template <typename _DataType>
void dpnp_matmul_c(void* result_out,
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
                   const size_t* input2_strides)
{
    (void)result_size;
    (void)result_ndim;
    (void)result_shape;
    (void)result_strides;
    (void)input1_size;
    (void)input1_ndim;
    (void)input1_strides;
    (void)input2_size;
    (void)input2_ndim;
    (void)input2_strides;

    size_t size_m = input1_shape[0];
    size_t size_n = input2_shape[1];
    size_t size_k = input1_shape[1];

    if (!size_m || !size_n || !size_k)
    {
        return;
    }

    cl::sycl::event event;
    DPNPC_ptr_adapter<_DataType> input1_ptr(input1_in, size_m * size_k, true);
    DPNPC_ptr_adapter<_DataType> input2_ptr(input2_in, size_k * size_n, true);
    DPNPC_ptr_adapter<_DataType> result_ptr(result_out, size_m * size_n, true, true);
    _DataType* array_1 = input1_ptr.get_ptr();
    _DataType* array_2 = input2_ptr.get_ptr();
    _DataType* result = result_ptr.get_ptr();

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

void func_map_init_linalg(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_astype_c<bool, bool>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_BLN][eft_INT] = {eft_INT, (void*)dpnp_astype_c<bool, int>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_BLN][eft_LNG] = {eft_LNG, (void*)dpnp_astype_c<bool, long>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_BLN][eft_FLT] = {eft_FLT, (void*)dpnp_astype_c<bool, float>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_BLN][eft_DBL] = {eft_DBL, (void*)dpnp_astype_c<bool, double>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_INT][eft_BLN] = {eft_BLN, (void*)dpnp_astype_c<int, bool>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_astype_c<int, int>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_astype_c<int, long>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_INT][eft_FLT] = {eft_FLT, (void*)dpnp_astype_c<int, float>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_astype_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_LNG][eft_BLN] = {eft_BLN, (void*)dpnp_astype_c<long, bool>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_LNG][eft_INT] = {eft_INT, (void*)dpnp_astype_c<long, int>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_astype_c<long, long>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_LNG][eft_FLT] = {eft_FLT, (void*)dpnp_astype_c<long, float>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_astype_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_FLT][eft_BLN] = {eft_BLN, (void*)dpnp_astype_c<float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_FLT][eft_INT] = {eft_INT, (void*)dpnp_astype_c<float, int>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_FLT][eft_LNG] = {eft_LNG, (void*)dpnp_astype_c<float, long>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_astype_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_astype_c<float, double>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_DBL][eft_BLN] = {eft_BLN, (void*)dpnp_astype_c<double, bool>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_DBL][eft_INT] = {eft_INT, (void*)dpnp_astype_c<double, int>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_DBL][eft_LNG] = {eft_LNG, (void*)dpnp_astype_c<double, long>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_DBL][eft_FLT] = {eft_FLT, (void*)dpnp_astype_c<double, float>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_astype_c<double, double>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_C64][eft_C64] = {
        eft_C64, (void*)dpnp_astype_c<std::complex<float>, std::complex<float>>};
    fmap[DPNPFuncName::DPNP_FN_ASTYPE][eft_C128][eft_C128] = {
        eft_C128, (void*)dpnp_astype_c<std::complex<double>, std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_DOT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_dot_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_dot_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_dot_c<double, int, float>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_dot_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_dot_c<long, long, int>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_dot_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_dot_c<double, long, float>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_dot_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_dot_c<double, float, int>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_dot_c<double, float, long>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_dot_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_dot_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_dot_c<double, double, int>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_dot_c<double, double, long>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_dot_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_dot_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_EIG][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_eig_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_EIG][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_eig_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_EIG][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_eig_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_EIG][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_eig_c<double, double>};
    fmap[DPNPFuncName::DPNP_FN_EIGVALS][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_eigvals_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_EIGVALS][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_eigvals_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_EIGVALS][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_eigvals_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_EIGVALS][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_eigvals_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_INITVAL][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_initval_c<bool>};
    fmap[DPNPFuncName::DPNP_FN_INITVAL][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_initval_c<int>};
    fmap[DPNPFuncName::DPNP_FN_INITVAL][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_initval_c<long>};
    fmap[DPNPFuncName::DPNP_FN_INITVAL][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_initval_c<float>};
    fmap[DPNPFuncName::DPNP_FN_INITVAL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_initval_c<double>};
    fmap[DPNPFuncName::DPNP_FN_INITVAL][eft_C128][eft_C128] = {eft_C128, (void*)dpnp_initval_c<std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_MATMUL][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_matmul_c<int>};
    fmap[DPNPFuncName::DPNP_FN_MATMUL][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_matmul_c<long>};
    fmap[DPNPFuncName::DPNP_FN_MATMUL][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_matmul_c<float>};
    fmap[DPNPFuncName::DPNP_FN_MATMUL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_matmul_c<double>};

    return;
}
