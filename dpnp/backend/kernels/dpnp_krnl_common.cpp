//*****************************************************************************
// Copyright (c) 2016-2025, Intel Corporation
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
#include <exception>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#include "dpnp_fptr.hpp"
#include "dpnp_utils.hpp"
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"
#include <dpnp_iface.hpp>

/**
 * Version of SYCL DPC++ 2025.1 compiler where support of
 * sycl::ext::oneapi::experimental::properties was added.
 */
#ifndef __SYCL_COMPILER_REDUCTION_PROPERTIES_SUPPORT
#define __SYCL_COMPILER_REDUCTION_PROPERTIES_SUPPORT 20241208L
#endif

namespace mkl_blas = oneapi::mkl::blas;
namespace mkl_blas_cm = oneapi::mkl::blas::column_major;
namespace mkl_blas_rm = oneapi::mkl::blas::row_major;
namespace mkl_lapack = oneapi::mkl::lapack;

#if __SYCL_COMPILER_VERSION >= __SYCL_COMPILER_REDUCTION_PROPERTIES_SUPPORT
namespace syclex = sycl::ext::oneapi::experimental;
#endif

template <typename _KernelNameSpecialization1,
          typename _KernelNameSpecialization2,
          typename _KernelNameSpecialization3>
class dpnp_dot_c_kernel;

template <typename _DataType_output,
          typename _DataType_input1,
          typename _DataType_input2>
sycl::event dot(sycl::queue &queue,
                _DataType_output *result_out,
                _DataType_input1 *input1_in,
                _DataType_input2 *input2_in,
                size_t input1_strides,
                size_t input2_strides,
                size_t size,
                const std::vector<sycl::event> &dependencies = {})
{
    (void)dependencies;

    sycl::event event;

    if constexpr ((std::is_same<_DataType_input1, double>::value ||
                   std::is_same<_DataType_input1, float>::value) &&
                  std::is_same<_DataType_input2, _DataType_input1>::value &&
                  std::is_same<_DataType_output, _DataType_input1>::value)
    {
        event = oneapi::mkl::blas::dot(queue, size, input1_in,
                                       input1_strides, // input1 stride
                                       input2_in,
                                       input2_strides, // input2 stride
                                       result_out);
    }
    else {
#if LIBSYCL_VERSION_GREATER(5, 3, 0)
        event = queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::range<1>{size},
                sycl::reduction(
                    result_out, sycl::plus<_DataType_output>(),
#if __SYCL_COMPILER_VERSION >= __SYCL_COMPILER_REDUCTION_PROPERTIES_SUPPORT
                    syclex::properties(syclex::initialize_to_identity)
#else
                    sycl::property::reduction::initialize_to_identity {}
#endif
                        ),
                [=](sycl::id<1> idx, auto &sum) {
                    sum += static_cast<_DataType_output>(
                               input1_in[idx * input1_strides]) *
                           static_cast<_DataType_output>(
                               input2_in[idx * input2_strides]);
                });
        });
        // for some reason few such kernels cannot work in parallel
        // looks like a bug in level0 because with opencl works fine
        // that is why we call wait here
        event.wait();
#else
        _DataType_output *local_mem = reinterpret_cast<_DataType_output *>(
            sycl::malloc_shared(size * sizeof(_DataType_output), queue));

        // what about reduction??
        sycl::range<1> gws(size);

        auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
            const size_t index = global_id[0];
            local_mem[index] = input1_in[index * input1_strides] *
                               input2_in[index * input2_strides];
        };

        auto kernel_func = [&](sycl::handler &cgh) {
            cgh.parallel_for<class dpnp_dot_c_kernel<
                _DataType_output, _DataType_input1, _DataType_input2>>(
                gws, kernel_parallel_for_func);
        };

        event = queue.submit(kernel_func);

        event.wait();

        auto policy =
            oneapi::dpl::execution::make_device_policy<class dpnp_dot_c_kernel<
                _DataType_output, _DataType_input1, _DataType_input2>>(queue);

        _DataType_output accumulator = 0;
        accumulator =
            std::reduce(policy, local_mem, local_mem + size,
                        _DataType_output(0), std::plus<_DataType_output>());
        policy.queue().wait();

        queue.memcpy(result_out, &accumulator, sizeof(_DataType_output)).wait();

        sycl::free(local_mem, queue);
#endif
    }
    return event;
}

template <typename _DataType_output,
          typename _DataType_input1,
          typename _DataType_input2>
DPCTLSyclEventRef dpnp_dot_c(DPCTLSyclQueueRef q_ref,
                             void *result_out,
                             const size_t result_size,
                             const size_t result_ndim,
                             const shape_elem_type *result_shape,
                             const shape_elem_type *result_strides,
                             const void *input1_in,
                             const size_t input1_size,
                             const size_t input1_ndim,
                             const shape_elem_type *input1_shape,
                             const shape_elem_type *input1_strides,
                             const void *input2_in,
                             const size_t input2_size,
                             const size_t input2_ndim,
                             const shape_elem_type *input2_shape,
                             const shape_elem_type *input2_strides,
                             const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;
    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    _DataType_input1 *input1 =
        static_cast<_DataType_input1 *>(const_cast<void *>(input1_in));
    _DataType_input2 *input2 =
        static_cast<_DataType_input2 *>(const_cast<void *>(input2_in));
    _DataType_output *result = reinterpret_cast<_DataType_output *>(result_out);

    if (!input1_size || !input2_size) {
        _DataType_output val = _DataType_output(0);
        dpnp_initval_c<_DataType_output>(result, &val, result_size);
        return event_ref;
    }

    // scalar
    if ((input1_ndim == 0) || (input2_ndim == 0)) {
        // there is no support of strides in multiply function
        // so result can be wrong if input array has non-standard (c-contiguous)
        // strides
        dpnp_multiply_c<_DataType_output, _DataType_input1, _DataType_input2>(
            result, result_size, result_ndim, result_shape, result_strides,
            input1_in, input1_size, input1_ndim, input1_shape, input1_strides,
            input2_in, input2_size, input2_ndim, input2_shape, input2_strides,
            NULL);
        return event_ref;
    }

    // if both arrays are vectors
    if ((input1_ndim == 1) && (input2_ndim == 1)) {
        assert(input1_size == input2_size);

        sycl::event event = dot(q, result, input1, input2, input1_strides[0],
                                input2_strides[0], input1_size);

        event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);
        return DPCTLEvent_Copy(event_ref);
    }

    // 1D vector
    size_t ext_input1_ndim = input1_ndim == 1 ? 2 : input1_ndim;
    shape_elem_type *ext_input1_shape = new shape_elem_type[ext_input1_ndim];
    shape_elem_type *ext_input1_strides = new shape_elem_type[ext_input1_ndim];
    if (input1_ndim == 1) {
        ext_input1_shape[0] = 1;
        ext_input1_shape[1] = input1_shape[0];
        ext_input1_strides[0] = 0;
        ext_input1_strides[1] = input1_strides[0];
    }
    else {
        for (size_t i = 0; i < ext_input1_ndim; ++i) {
            ext_input1_shape[i] = input1_shape[i];
            ext_input1_strides[i] = input1_strides[i];
        }
    }
    size_t ext_input2_ndim = input2_ndim == 1 ? 2 : input2_ndim;
    shape_elem_type *ext_input2_shape = new shape_elem_type[ext_input2_ndim];
    shape_elem_type *ext_input2_strides = new shape_elem_type[ext_input2_ndim];
    if (input2_ndim == 1) {
        ext_input2_shape[0] = input2_shape[0];
        ext_input2_shape[1] = 1;
        ext_input2_strides[0] = input2_strides[0];
        ext_input2_strides[1] = 0;
    }
    else {
        for (size_t i = 0; i < ext_input2_ndim; ++i) {
            ext_input2_shape[i] = input2_shape[i];
            ext_input2_strides[i] = input2_strides[i];
        }
    }
    size_t ext_result_ndim =
        ((input1_ndim == 1) || (input2_ndim == 1)) ? 2 : result_ndim;
    shape_elem_type *ext_result_shape = new shape_elem_type[ext_result_ndim];
    shape_elem_type *ext_result_strides = new shape_elem_type[ext_result_ndim];
    if ((input1_ndim == 1) || (input2_ndim == 1)) {
        ext_result_shape[0] = ext_input1_shape[0];
        ext_result_shape[1] = ext_input2_shape[1];
        ext_result_strides[0] = 0;
        ext_result_strides[1] = result_strides[0];
    }
    else {
        for (size_t i = 0; i < ext_result_ndim; ++i) {
            ext_result_shape[i] = result_shape[i];
            ext_result_strides[i] = result_strides[i];
        }
    }

    // check if GEMM can be executed (types)
    if constexpr ((std::is_same<_DataType_input1, double>::value ||
                   std::is_same<_DataType_input1, float>::value) &&
                  std::is_same<_DataType_input2, _DataType_input1>::value &&
                  std::is_same<_DataType_output, _DataType_input1>::value)
    {
        // check if GEMM can be executed (strides)
        // TODO: rewrite the condition in general case for ndims > 2
        // (looks like there are such another cases)
        if (ext_input1_ndim == 2 && ext_input2_ndim == 2) {
            // OneMKL gemm supports only arrays contiguous on inner dimension,
            // so stride for at least one dimension should be equal to 1
            if ((ext_input1_strides[0] == 1 || ext_input1_strides[1] == 1) &&
                (ext_input2_strides[0] == 1 || ext_input2_strides[1] == 1) &&
                (ext_result_strides[0] == 1 || ext_result_strides[1] == 1))
            {
                const bool isRowmA =
                    (ext_input1_strides[1] == 1 || ext_input1_strides[0] == 0);
                const bool isRowmB =
                    (ext_input2_strides[1] == 1 || ext_input2_strides[1] == 0);
                const bool isRowmC =
                    (ext_result_strides[1] == 1 || ext_result_strides[0] == 0);

                oneapi::mkl::transpose transA =
                    (isRowmA != isRowmC) ? oneapi::mkl::transpose::trans
                                         : oneapi::mkl::transpose::nontrans;
                oneapi::mkl::transpose transB =
                    (isRowmB != isRowmC) ? oneapi::mkl::transpose::trans
                                         : oneapi::mkl::transpose::nontrans;

                const size_t size_m = ext_input1_shape[0];
                const size_t size_n = ext_input2_shape[1];
                const size_t size_k = ext_input1_shape[1];

                auto getLdaLdc = [](const bool isRown, shape_elem_type *strides,
                                    shape_elem_type *shapes) {
                    if (isRown) {
                        return (strides[0] != 0) ? strides[0] : shapes[1];
                    }
                    return strides[1];
                };

                const std::int64_t lda = static_cast<std::int64_t>(
                    getLdaLdc(isRowmA, ext_input1_strides, ext_input1_shape));
                const std::int64_t ldb = static_cast<std::int64_t>(
                    isRowmB ? ext_input2_strides[0] : ext_input2_strides[1]);
                const std::int64_t ldc = static_cast<std::int64_t>(
                    getLdaLdc(isRowmC, ext_result_strides, ext_result_shape));

                constexpr _DataType_output alpha = 1;
                constexpr _DataType_output beta = 0;

                std::stringstream error_msg;
                std::int64_t info = 0;

                try {
                    if (isRowmC) {
                        mkl_blas_rm::gemm(q, transA, transB, size_m, size_n,
                                          size_k, alpha, input1, lda, input2,
                                          ldb, beta, result, ldc)
                            .wait();
                    }
                    else {
                        mkl_blas_cm::gemm(q, transA, transB, size_m, size_n,
                                          size_k, alpha, input1, lda, input2,
                                          ldb, beta, result, ldc)
                            .wait();
                    }
                } catch (mkl_lapack::exception const &e) {
                    error_msg << "Unexpected MKL exception caught during "
                                 "gemm() call:\nreason: "
                              << e.what() << "\ninfo: " << e.info();
                    info = e.info();
                } catch (const std::exception &e) {
                    error_msg << "Unexpected SYCL exception caught during "
                                 "gemm() call:\n"
                              << e.what();
                    info = -1;
                }

                if (info != 0) // an unexpected error occurs
                {
                    throw std::runtime_error(error_msg.str());
                }

                delete[] ext_input1_shape;
                delete[] ext_input1_strides;
                delete[] ext_input2_shape;
                delete[] ext_input2_strides;
                delete[] ext_result_shape;
                delete[] ext_result_strides;
                return event_ref;
            }
        }
    }

    std::vector<sycl::event> dot_events;
    dot_events.reserve(result_size);

    size_t dot_st1 = ext_input1_strides[ext_input1_ndim - 1];
    size_t dot_st2 = ext_input2_strides[ext_input2_ndim - 2];
    size_t dot_size = ext_input1_shape[ext_input1_ndim - 1];

    shape_elem_type *res_coords = new shape_elem_type[ext_result_ndim];
    shape_elem_type *result_offsets = new shape_elem_type[ext_result_ndim];
    get_shape_offsets_inkernel(ext_result_shape, ext_result_ndim,
                               result_offsets);

    for (size_t i = 0; i < result_size; ++i) {
        get_xyz_by_id(i, ext_result_ndim, result_offsets, res_coords);

        _DataType_output *dot_res = result + i;

        _DataType_input1 *dot_in1 = input1;
        for (size_t j = 0; j < ext_input1_ndim - 1; ++j) {
            dot_in1 = dot_in1 + res_coords[j] * ext_input1_strides[j];
        }

        _DataType_input2 *dot_in2 = input2;
        for (size_t j = 0; j < ext_input2_ndim - 2; ++j) {
            dot_in2 = dot_in2 + res_coords[ext_input1_ndim - 1 + j] *
                                    ext_input2_strides[j];
        }
        dot_in2 = dot_in2 + res_coords[ext_input1_ndim + ext_input2_ndim - 3] *
                                ext_input2_strides[ext_input2_ndim - 1];

        dot_events.push_back(
            dot(q, dot_res, dot_in1, dot_in2, dot_st1, dot_st2, dot_size));
    }

    sycl::event::wait(dot_events);

    delete[] res_coords;
    delete[] result_offsets;
    delete[] ext_input1_shape;
    delete[] ext_input1_strides;
    delete[] ext_input2_shape;
    delete[] ext_input2_strides;
    delete[] ext_result_shape;
    delete[] ext_result_strides;

    return event_ref;
}

template <typename _DataType_output,
          typename _DataType_input1,
          typename _DataType_input2>
void dpnp_dot_c(void *result_out,
                const size_t result_size,
                const size_t result_ndim,
                const shape_elem_type *result_shape,
                const shape_elem_type *result_strides,
                const void *input1_in,
                const size_t input1_size,
                const size_t input1_ndim,
                const shape_elem_type *input1_shape,
                const shape_elem_type *input1_strides,
                const void *input2_in,
                const size_t input2_size,
                const size_t input2_ndim,
                const shape_elem_type *input2_shape,
                const shape_elem_type *input2_strides)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref =
        dpnp_dot_c<_DataType_output, _DataType_input1, _DataType_input2>(
            q_ref, result_out, result_size, result_ndim, result_shape,
            result_strides, input1_in, input1_size, input1_ndim, input1_shape,
            input1_strides, input2_in, input2_size, input2_ndim, input2_shape,
            input2_strides, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType_output,
          typename _DataType_input1,
          typename _DataType_input2>
void (*dpnp_dot_default_c)(void *,
                           const size_t,
                           const size_t,
                           const shape_elem_type *,
                           const shape_elem_type *,
                           const void *,
                           const size_t,
                           const size_t,
                           const shape_elem_type *,
                           const shape_elem_type *,
                           const void *,
                           const size_t,
                           const size_t,
                           const shape_elem_type *,
                           const shape_elem_type *) =
    dpnp_dot_c<_DataType_output, _DataType_input1, _DataType_input2>;

template <typename _DataType_output,
          typename _DataType_input1,
          typename _DataType_input2>
DPCTLSyclEventRef (*dpnp_dot_ext_c)(DPCTLSyclQueueRef,
                                    void *,
                                    const size_t,
                                    const size_t,
                                    const shape_elem_type *,
                                    const shape_elem_type *,
                                    const void *,
                                    const size_t,
                                    const size_t,
                                    const shape_elem_type *,
                                    const shape_elem_type *,
                                    const void *,
                                    const size_t,
                                    const size_t,
                                    const shape_elem_type *,
                                    const shape_elem_type *,
                                    const DPCTLEventVectorRef) =
    dpnp_dot_c<_DataType_output, _DataType_input1, _DataType_input2>;

template <typename _DataType>
class dpnp_initval_c_kernel;

template <typename _DataType>
DPCTLSyclEventRef dpnp_initval_c(DPCTLSyclQueueRef q_ref,
                                 void *result,
                                 void *value,
                                 size_t size,
                                 const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!size) {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));
    _DataType val = *(static_cast<_DataType *>(value));

    validate_type_for_device<_DataType>(q);

    auto event = q.fill<_DataType>(result, val, size);
    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_initval_c(void *result1, void *value, size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_initval_c<_DataType>(
        q_ref, result1, value, size, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_initval_default_c)(void *,
                               void *,
                               size_t) = dpnp_initval_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_initval_ext_c)(DPCTLSyclQueueRef,
                                        void *,
                                        void *,
                                        size_t,
                                        const DPCTLEventVectorRef) =
    dpnp_initval_c<_DataType>;

void func_map_init_linalg(func_map_t &fmap)
{

    fmap[DPNPFuncName::DPNP_FN_DOT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_dot_default_c<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_INT][eft_LNG] = {
        eft_LNG, (void *)dpnp_dot_default_c<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_INT][eft_FLT] = {
        eft_DBL, (void *)dpnp_dot_default_c<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_dot_default_c<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_LNG][eft_INT] = {
        eft_LNG, (void *)dpnp_dot_default_c<int64_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_dot_default_c<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_LNG][eft_FLT] = {
        eft_DBL, (void *)dpnp_dot_default_c<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_dot_default_c<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_FLT][eft_INT] = {
        eft_DBL, (void *)dpnp_dot_default_c<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_FLT][eft_LNG] = {
        eft_DBL, (void *)dpnp_dot_default_c<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_dot_default_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_dot_default_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_DBL][eft_INT] = {
        eft_DBL, (void *)dpnp_dot_default_c<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_DBL][eft_LNG] = {
        eft_DBL, (void *)dpnp_dot_default_c<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_DBL][eft_FLT] = {
        eft_DBL, (void *)dpnp_dot_default_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_DOT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_dot_default_c<double, double, double>};

    // needed for "dpnp_correlate_c" function in dpnp_krnl_statistics.cpp
    fmap[DPNPFuncName::DPNP_FN_DOT_EXT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_dot_ext_c<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_DOT_EXT][eft_INT][eft_LNG] = {
        eft_LNG, (void *)dpnp_dot_ext_c<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_DOT_EXT][eft_INT][eft_FLT] = {
        eft_DBL, (void *)dpnp_dot_ext_c<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_DOT_EXT][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_dot_ext_c<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_DOT_EXT][eft_LNG][eft_INT] = {
        eft_LNG, (void *)dpnp_dot_ext_c<int64_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_DOT_EXT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_dot_ext_c<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_DOT_EXT][eft_LNG][eft_FLT] = {
        eft_DBL, (void *)dpnp_dot_ext_c<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_DOT_EXT][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_dot_ext_c<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_DOT_EXT][eft_FLT][eft_INT] = {
        eft_DBL, (void *)dpnp_dot_ext_c<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_DOT_EXT][eft_FLT][eft_LNG] = {
        eft_DBL, (void *)dpnp_dot_ext_c<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_DOT_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_dot_ext_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_DOT_EXT][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_dot_ext_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_DOT_EXT][eft_DBL][eft_INT] = {
        eft_DBL, (void *)dpnp_dot_ext_c<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_DOT_EXT][eft_DBL][eft_LNG] = {
        eft_DBL, (void *)dpnp_dot_ext_c<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_DOT_EXT][eft_DBL][eft_FLT] = {
        eft_DBL, (void *)dpnp_dot_ext_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_DOT_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_dot_ext_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_INITVAL][eft_BLN][eft_BLN] = {
        eft_BLN, (void *)dpnp_initval_default_c<bool>};
    fmap[DPNPFuncName::DPNP_FN_INITVAL][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_initval_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_INITVAL][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_initval_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_INITVAL][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_initval_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_INITVAL][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_initval_default_c<double>};
    fmap[DPNPFuncName::DPNP_FN_INITVAL][eft_C128][eft_C128] = {
        eft_C128, (void *)dpnp_initval_default_c<std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_INITVAL_EXT][eft_BLN][eft_BLN] = {
        eft_BLN, (void *)dpnp_initval_ext_c<bool>};
    fmap[DPNPFuncName::DPNP_FN_INITVAL_EXT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_initval_ext_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_INITVAL_EXT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_initval_ext_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_INITVAL_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_initval_ext_c<float>};
    fmap[DPNPFuncName::DPNP_FN_INITVAL_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_initval_ext_c<double>};
    fmap[DPNPFuncName::DPNP_FN_INITVAL_EXT][eft_C64][eft_C64] = {
        eft_C64, (void *)dpnp_initval_ext_c<std::complex<float>>};
    fmap[DPNPFuncName::DPNP_FN_INITVAL_EXT][eft_C128][eft_C128] = {
        eft_C128, (void *)dpnp_initval_ext_c<std::complex<double>>};

    return;
}
