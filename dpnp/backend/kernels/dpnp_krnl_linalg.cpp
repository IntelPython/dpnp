//*****************************************************************************
// Copyright (c) 2016-2024, Intel Corporation
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
#include <list>

#include "dpnp_fptr.hpp"
#include "dpnp_utils.hpp"
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"
#include <dpnp_iface.hpp>

namespace mkl_blas = oneapi::mkl::blas::row_major;
namespace mkl_lapack = oneapi::mkl::lapack;

template <typename _DataType>
DPCTLSyclEventRef dpnp_cholesky_c(DPCTLSyclQueueRef q_ref,
                                  void *array1_in,
                                  void *result1,
                                  const size_t size,
                                  const size_t data_size,
                                  const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;
    if (!data_size) {
        return event_ref;
    }
    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    sycl::event event;

    DPNPC_ptr_adapter<_DataType> input1_ptr(q_ref, array1_in, size, true);
    DPNPC_ptr_adapter<_DataType> result_ptr(q_ref, result1, size, true, true);
    _DataType *in_array = input1_ptr.get_ptr();
    _DataType *result = result_ptr.get_ptr();

    size_t iters = size / (data_size * data_size);

    // math lib func overrides input
    _DataType *in_a = reinterpret_cast<_DataType *>(
        sycl::malloc_shared(data_size * data_size * sizeof(_DataType), q));

    for (size_t k = 0; k < iters; ++k) {
        for (size_t it = 0; it < data_size * data_size; ++it) {
            in_a[it] = in_array[k * (data_size * data_size) + it];
        }

        const std::int64_t n = data_size;

        const std::int64_t lda = std::max<size_t>(1UL, n);

        const std::int64_t scratchpad_size =
            mkl_lapack::potrf_scratchpad_size<_DataType>(
                q, oneapi::mkl::uplo::upper, n, lda);

        _DataType *scratchpad = reinterpret_cast<_DataType *>(
            sycl::malloc_shared(scratchpad_size * sizeof(_DataType), q));

        event = mkl_lapack::potrf(q, oneapi::mkl::uplo::upper, n, in_a, lda,
                                  scratchpad, scratchpad_size);

        event.wait();

        for (size_t i = 0; i < data_size; i++) {
            bool arg = false;
            for (size_t j = 0; j < data_size; j++) {
                if (i == j - 1) {
                    arg = true;
                }
                if (arg) {
                    in_a[i * data_size + j] = 0;
                }
            }
        }

        sycl::free(scratchpad, q);

        for (size_t t = 0; t < data_size * data_size; ++t) {
            result[k * (data_size * data_size) + t] = in_a[t];
        }
    }

    sycl::free(in_a, q);

    return event_ref;
}

template <typename _DataType>
void dpnp_cholesky_c(void *array1_in,
                     void *result1,
                     const size_t size,
                     const size_t data_size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_cholesky_c<_DataType>(
        q_ref, array1_in, result1, size, data_size, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_cholesky_default_c)(void *, void *, const size_t, const size_t) =
    dpnp_cholesky_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_det_c(DPCTLSyclQueueRef q_ref,
                             void *array1_in,
                             void *result1,
                             shape_elem_type *shape,
                             size_t ndim,
                             const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    const size_t input_size = std::accumulate(
        shape, shape + ndim, 1, std::multiplies<shape_elem_type>());
    if (!input_size) {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    size_t n = shape[ndim - 1];
    size_t size_out = 1;
    if (ndim != 2) {
        for (size_t i = 0; i < ndim - 2; i++) {
            size_out *= shape[i];
        }
    }

    DPNPC_ptr_adapter<_DataType> input1_ptr(q_ref, array1_in, input_size, true);
    DPNPC_ptr_adapter<_DataType> result_ptr(q_ref, result1, size_out, true,
                                            true);
    _DataType *array_1 = input1_ptr.get_ptr();
    _DataType *result = result_ptr.get_ptr();

    _DataType *matrix = new _DataType[n * n];
    _DataType *elems = new _DataType[n * n];

    for (size_t i = 0; i < size_out; i++) {
        if (size_out > 1) {
            for (size_t j = i * n * n; j < (i + 1) * n * n; j++) {
                elems[j - i * n * n] = array_1[j];
            }

            for (size_t j = 0; j < n; j++) {
                for (size_t k = 0; k < n; k++) {
                    matrix[j * n + k] = elems[j * n + k];
                }
            }
        }
        else {
            for (size_t j = 0; j < n; j++) {
                for (size_t k = 0; k < n; k++) {
                    matrix[j * n + k] = array_1[j * n + k];
                }
            }
        }

        _DataType det_val = 1;
        for (size_t l = 0; l < n; l++) {
            if (matrix[l * n + l] == 0) {
                for (size_t j = l; j < n; j++) {
                    if (matrix[j * n + l] != 0) {
                        for (size_t k = l; k < n; k++) {
                            _DataType c = matrix[l * n + k];
                            matrix[l * n + k] = -1 * matrix[j * n + k];
                            matrix[j * n + k] = c;
                        }
                        break;
                    }
                    if (j == n - 1 and matrix[j * n + l] == 0) {
                        det_val = 0;
                    }
                }
            }
            if (det_val != 0) {
                for (size_t j = l + 1; j < n; j++) {
                    _DataType quotient =
                        -(matrix[j * n + l] / matrix[l * n + l]);
                    for (size_t k = l + 1; k < n; k++) {
                        matrix[j * n + k] += quotient * matrix[l * n + k];
                    }
                }
            }
        }

        if (det_val != 0) {
            for (size_t l = 0; l < n; l++) {
                det_val *= matrix[l * n + l];
            }
        }

        result[i] = det_val;
    }

    delete[] elems;
    delete[] matrix;
    return event_ref;
}

template <typename _DataType>
void dpnp_det_c(void *array1_in,
                void *result1,
                shape_elem_type *shape,
                size_t ndim)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_det_c<_DataType>(
        q_ref, array1_in, result1, shape, ndim, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_det_default_c)(void *, void *, shape_elem_type *, size_t) =
    dpnp_det_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_det_ext_c)(DPCTLSyclQueueRef,
                                    void *,
                                    void *,
                                    shape_elem_type *,
                                    size_t,
                                    const DPCTLEventVectorRef) =
    dpnp_det_c<_DataType>;

template <typename _DataType, typename _ResultType>
DPCTLSyclEventRef dpnp_inv_c(DPCTLSyclQueueRef q_ref,
                             void *array1_in,
                             void *result1,
                             shape_elem_type *shape,
                             size_t ndim,
                             const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)ndim;
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    const size_t input_size = std::accumulate(
        shape, shape + ndim, 1, std::multiplies<shape_elem_type>());
    if (!input_size) {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    DPNPC_ptr_adapter<_DataType> input1_ptr(q_ref, array1_in, input_size, true);
    DPNPC_ptr_adapter<_ResultType> result_ptr(q_ref, result1, input_size, true,
                                              true);

    _DataType *array_1 = input1_ptr.get_ptr();
    _ResultType *result = result_ptr.get_ptr();

    size_t n = shape[0];

    _ResultType *a_arr = new _ResultType[n * n];
    _ResultType *e_arr = new _ResultType[n * n];

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            a_arr[i * n + j] = array_1[i * n + j];
            if (i == j) {
                e_arr[i * n + j] = 1;
            }
            else {
                e_arr[i * n + j] = 0;
            }
        }
    }

    for (size_t k = 0; k < n; ++k) {
        if (a_arr[k * n + k] == 0) {
            for (size_t i = k; i < n; ++i) {
                if (a_arr[i * n + k] != 0) {
                    for (size_t j = 0; j < n; ++j) {
                        float c = a_arr[k * n + j];
                        a_arr[k * n + j] = a_arr[i * n + j];
                        a_arr[i * n + j] = c;
                        float c_e = e_arr[k * n + j];
                        e_arr[k * n + j] = e_arr[i * n + j];
                        e_arr[i * n + j] = c_e;
                    }
                    break;
                }
            }
        }

        float temp = a_arr[k * n + k];

        for (size_t j = 0; j < n; ++j) {
            a_arr[k * n + j] = a_arr[k * n + j] / temp;
            e_arr[k * n + j] = e_arr[k * n + j] / temp;
        }

        for (size_t i = k + 1; i < n; ++i) {
            temp = a_arr[i * n + k];
            for (size_t j = 0; j < n; j++) {
                a_arr[i * n + j] = a_arr[i * n + j] - a_arr[k * n + j] * temp;
                e_arr[i * n + j] = e_arr[i * n + j] - e_arr[k * n + j] * temp;
            }
        }
    }

    for (size_t k = 0; k < n - 1; ++k) {
        size_t ind_k = n - 1 - k;
        for (size_t i = 0; i < ind_k; ++i) {
            size_t ind_i = ind_k - 1 - i;

            float temp = a_arr[ind_i * n + ind_k];
            for (size_t j = 0; j < n; ++j) {
                a_arr[ind_i * n + j] =
                    a_arr[ind_i * n + j] - a_arr[ind_k * n + j] * temp;
                e_arr[ind_i * n + j] =
                    e_arr[ind_i * n + j] - e_arr[ind_k * n + j] * temp;
            }
        }
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result[i * n + j] = e_arr[i * n + j];
        }
    }

    delete[] a_arr;
    delete[] e_arr;
    return event_ref;
}

template <typename _DataType, typename _ResultType>
void dpnp_inv_c(void *array1_in,
                void *result1,
                shape_elem_type *shape,
                size_t ndim)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_inv_c<_DataType, _ResultType>(
        q_ref, array1_in, result1, shape, ndim, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType, typename _ResultType>
void (*dpnp_inv_default_c)(void *, void *, shape_elem_type *, size_t) =
    dpnp_inv_c<_DataType, _ResultType>;

template <typename _DataType, typename _ResultType>
DPCTLSyclEventRef (*dpnp_inv_ext_c)(DPCTLSyclQueueRef,
                                    void *,
                                    void *,
                                    shape_elem_type *,
                                    size_t,
                                    const DPCTLEventVectorRef) =
    dpnp_inv_c<_DataType, _ResultType>;

template <typename _DataType1, typename _DataType2, typename _ResultType>
class dpnp_kron_c_kernel;

template <typename _DataType1, typename _DataType2, typename _ResultType>
DPCTLSyclEventRef dpnp_kron_c(DPCTLSyclQueueRef q_ref,
                              void *array1_in,
                              void *array2_in,
                              void *result1,
                              shape_elem_type *in1_shape,
                              shape_elem_type *in2_shape,
                              shape_elem_type *res_shape,
                              size_t ndim,
                              const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    const size_t input1_size = std::accumulate(
        in1_shape, in1_shape + ndim, 1, std::multiplies<shape_elem_type>());
    const size_t input2_size = std::accumulate(
        in2_shape, in2_shape + ndim, 1, std::multiplies<shape_elem_type>());
    const size_t result_size = std::accumulate(
        res_shape, res_shape + ndim, 1, std::multiplies<shape_elem_type>());
    if (!(result_size && input1_size && input2_size)) {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    DPNPC_ptr_adapter<_DataType1> input1_ptr(q_ref, array1_in, input1_size);
    DPNPC_ptr_adapter<_DataType2> input2_ptr(q_ref, array2_in, input2_size);
    DPNPC_ptr_adapter<_ResultType> result_ptr(q_ref, result1, result_size);

    _DataType1 *array1 = input1_ptr.get_ptr();
    _DataType2 *array2 = input2_ptr.get_ptr();
    _ResultType *result = result_ptr.get_ptr();

    shape_elem_type *_in1_shape = reinterpret_cast<shape_elem_type *>(
        sycl::malloc_shared(ndim * sizeof(shape_elem_type), q));
    shape_elem_type *_in2_shape = reinterpret_cast<shape_elem_type *>(
        sycl::malloc_shared(ndim * sizeof(shape_elem_type), q));

    q.memcpy(_in1_shape, in1_shape, ndim * sizeof(shape_elem_type)).wait();
    q.memcpy(_in2_shape, in2_shape, ndim * sizeof(shape_elem_type)).wait();

    shape_elem_type *in1_offsets = reinterpret_cast<shape_elem_type *>(
        sycl::malloc_shared(ndim * sizeof(shape_elem_type), q));
    shape_elem_type *in2_offsets = reinterpret_cast<shape_elem_type *>(
        sycl::malloc_shared(ndim * sizeof(shape_elem_type), q));
    shape_elem_type *res_offsets = reinterpret_cast<shape_elem_type *>(
        sycl::malloc_shared(ndim * sizeof(shape_elem_type), q));

    get_shape_offsets_inkernel(in1_shape, ndim, in1_offsets);
    get_shape_offsets_inkernel(in2_shape, ndim, in2_offsets);
    get_shape_offsets_inkernel(res_shape, ndim, res_offsets);

    sycl::range<1> gws(result_size);
    auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
        const size_t idx = global_id[0];

        size_t idx1 = 0;
        size_t idx2 = 0;
        size_t reminder = idx;
        for (size_t axis = 0; axis < ndim; ++axis) {
            const size_t res_axis = reminder / res_offsets[axis];
            reminder = reminder - res_axis * res_offsets[axis];

            const size_t in1_axis = res_axis / _in2_shape[axis];
            const size_t in2_axis = res_axis - in1_axis * _in2_shape[axis];

            idx1 += in1_axis * in1_offsets[axis];
            idx2 += in2_axis * in2_offsets[axis];
        }

        result[idx] = array1[idx1] * array2[idx2];
    };

    auto kernel_func = [&](sycl::handler &cgh) {
        cgh.parallel_for<
            class dpnp_kron_c_kernel<_DataType1, _DataType2, _ResultType>>(
            gws, kernel_parallel_for_func);
    };

    sycl::event event = q.submit(kernel_func);

    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType1, typename _DataType2, typename _ResultType>
void dpnp_kron_c(void *array1_in,
                 void *array2_in,
                 void *result1,
                 shape_elem_type *in1_shape,
                 shape_elem_type *in2_shape,
                 shape_elem_type *res_shape,
                 size_t ndim)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref =
        dpnp_kron_c<_DataType1, _DataType2, _ResultType>(
            q_ref, array1_in, array2_in, result1, in1_shape, in2_shape,
            res_shape, ndim, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType1, typename _DataType2, typename _ResultType>
void (*dpnp_kron_default_c)(void *,
                            void *,
                            void *,
                            shape_elem_type *,
                            shape_elem_type *,
                            shape_elem_type *,
                            size_t) =
    dpnp_kron_c<_DataType1, _DataType2, _ResultType>;

template <typename _DataType1, typename _DataType2, typename _ResultType>
DPCTLSyclEventRef (*dpnp_kron_ext_c)(DPCTLSyclQueueRef,
                                     void *,
                                     void *,
                                     void *,
                                     shape_elem_type *,
                                     shape_elem_type *,
                                     shape_elem_type *,
                                     size_t,
                                     const DPCTLEventVectorRef) =
    dpnp_kron_c<_DataType1, _DataType2, _ResultType>;

template <typename _DataType>
DPCTLSyclEventRef
    dpnp_matrix_rank_c(DPCTLSyclQueueRef q_ref,
                       void *array1_in,
                       void *result1,
                       shape_elem_type *shape,
                       size_t ndim,
                       const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    const size_t input_size = std::accumulate(
        shape, shape + ndim, 1, std::multiplies<shape_elem_type>());
    if (!input_size) {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    DPNPC_ptr_adapter<_DataType> input1_ptr(q_ref, array1_in, input_size, true);
    DPNPC_ptr_adapter<_DataType> result_ptr(q_ref, result1, 1, true, true);
    _DataType *array_1 = input1_ptr.get_ptr();
    _DataType *result = result_ptr.get_ptr();

    shape_elem_type elems = 1;
    if (ndim > 1) {
        elems = shape[0];
        for (size_t i = 1; i < ndim; i++) {
            if (shape[i] < elems) {
                elems = shape[i];
            }
        }
    }

    _DataType acc = 0;
    for (size_t i = 0; i < static_cast<size_t>(elems); i++) {
        size_t ind = 0;
        for (size_t j = 0; j < ndim; j++) {
            ind += (shape[j] - 1) * i;
        }
        acc += array_1[ind];
    }
    result[0] = acc;

    return event_ref;
}

template <typename _DataType>
void dpnp_matrix_rank_c(void *array1_in,
                        void *result1,
                        shape_elem_type *shape,
                        size_t ndim)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_matrix_rank_c<_DataType>(
        q_ref, array1_in, result1, shape, ndim, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_matrix_rank_default_c)(void *, void *, shape_elem_type *, size_t) =
    dpnp_matrix_rank_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_matrix_rank_ext_c)(DPCTLSyclQueueRef,
                                            void *,
                                            void *,
                                            shape_elem_type *,
                                            size_t,
                                            const DPCTLEventVectorRef) =
    dpnp_matrix_rank_c<_DataType>;

template <typename _InputDT, typename _ComputeDT>
DPCTLSyclEventRef dpnp_qr_c(DPCTLSyclQueueRef q_ref,
                            void *array1_in,
                            void *result1,
                            void *result2,
                            void *result3,
                            size_t size_m,
                            size_t size_n,
                            const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;
    if (!size_m || !size_n) {
        return event_ref;
    }
    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    sycl::event event;

    DPNPC_ptr_adapter<_InputDT> input1_ptr(q_ref, array1_in, size_m * size_n,
                                           true);
    _InputDT *in_array = input1_ptr.get_ptr();

    // math lib func overrides input
    _ComputeDT *in_a = reinterpret_cast<_ComputeDT *>(
        sycl::malloc_shared(size_m * size_n * sizeof(_ComputeDT), q));

    for (size_t i = 0; i < size_m; ++i) {
        for (size_t j = 0; j < size_n; ++j) {
            // TODO transpose? use dpnp_transpose_c()
            in_a[j * size_m + i] = in_array[i * size_n + j];
        }
    }

    const size_t min_size_m_n = std::min<size_t>(size_m, size_n);
    DPNPC_ptr_adapter<_ComputeDT> result1_ptr(
        q_ref, result1, size_m * min_size_m_n, true, true);
    DPNPC_ptr_adapter<_ComputeDT> result2_ptr(
        q_ref, result2, min_size_m_n * size_n, true, true);
    DPNPC_ptr_adapter<_ComputeDT> result3_ptr(q_ref, result3, min_size_m_n,
                                              true, true);
    _ComputeDT *res_q = result1_ptr.get_ptr();
    _ComputeDT *res_r = result2_ptr.get_ptr();
    _ComputeDT *tau = result3_ptr.get_ptr();

    const std::int64_t lda = size_m;

    const std::int64_t geqrf_scratchpad_size =
        mkl_lapack::geqrf_scratchpad_size<_ComputeDT>(q, size_m, size_n, lda);

    _ComputeDT *geqrf_scratchpad = reinterpret_cast<_ComputeDT *>(
        sycl::malloc_shared(geqrf_scratchpad_size * sizeof(_ComputeDT), q));

    std::vector<sycl::event> depends(1);
    set_barrier_event(q, depends);

    event = mkl_lapack::geqrf(q, size_m, size_n, in_a, lda, tau,
                              geqrf_scratchpad, geqrf_scratchpad_size, depends);
    event.wait();

    if (!depends.empty()) {
        verbose_print("oneapi::mkl::lapack::geqrf", depends.front(), event);
    }

    sycl::free(geqrf_scratchpad, q);

    // R
    size_t mrefl = min_size_m_n;
    for (size_t i = 0; i < mrefl; ++i) {
        for (size_t j = 0; j < size_n; ++j) {
            if (j >= i) {
                res_r[i * size_n + j] = in_a[j * size_m + i];
            }
            else {
                res_r[i * size_n + j] = _ComputeDT(0);
            }
        }
    }

    // Q
    const size_t nrefl = min_size_m_n;
    const std::int64_t orgqr_scratchpad_size =
        mkl_lapack::orgqr_scratchpad_size<_ComputeDT>(q, size_m, nrefl, nrefl,
                                                      lda);

    _ComputeDT *orgqr_scratchpad = reinterpret_cast<_ComputeDT *>(
        sycl::malloc_shared(orgqr_scratchpad_size * sizeof(_ComputeDT), q));

    set_barrier_event(q, depends);

    event = mkl_lapack::orgqr(q, size_m, nrefl, nrefl, in_a, lda, tau,
                              orgqr_scratchpad, orgqr_scratchpad_size, depends);
    event.wait();

    if (!depends.empty()) {
        verbose_print("oneapi::mkl::lapack::orgqr", depends.front(), event);
    }

    sycl::free(orgqr_scratchpad, q);

    for (size_t i = 0; i < size_m; ++i) {
        for (size_t j = 0; j < nrefl; ++j) {
            res_q[i * nrefl + j] = in_a[j * size_m + i];
        }
    }

    sycl::free(in_a, q);

    return event_ref;
}

template <typename _InputDT, typename _ComputeDT>
void dpnp_qr_c(void *array1_in,
               void *result1,
               void *result2,
               void *result3,
               size_t size_m,
               size_t size_n)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_qr_c<_InputDT, _ComputeDT>(
        q_ref, array1_in, result1, result2, result3, size_m, size_n,
        dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _InputDT, typename _ComputeDT>
void (*dpnp_qr_default_c)(void *, void *, void *, void *, size_t, size_t) =
    dpnp_qr_c<_InputDT, _ComputeDT>;

template <typename _InputDT, typename _ComputeDT>
DPCTLSyclEventRef (*dpnp_qr_ext_c)(DPCTLSyclQueueRef,
                                   void *,
                                   void *,
                                   void *,
                                   void *,
                                   size_t,
                                   size_t,
                                   const DPCTLEventVectorRef) =
    dpnp_qr_c<_InputDT, _ComputeDT>;

template <typename _InputDT, typename _ComputeDT, typename _SVDT>
DPCTLSyclEventRef dpnp_svd_c(DPCTLSyclQueueRef q_ref,
                             void *array1_in,
                             void *result1,
                             void *result2,
                             void *result3,
                             size_t size_m,
                             size_t size_n,
                             const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;
    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    sycl::event event;

    DPNPC_ptr_adapter<_InputDT> input1_ptr(
        q_ref, array1_in, size_m * size_n,
        true); // TODO no need this if use dpnp_copy_to()
    _InputDT *in_array = input1_ptr.get_ptr();

    // math lib gesvd func overrides input
    _ComputeDT *in_a = reinterpret_cast<_ComputeDT *>(
        sycl::malloc_shared(size_m * size_n * sizeof(_ComputeDT), q));
    for (size_t it = 0; it < size_m * size_n; ++it) {
        in_a[it] = in_array[it]; // TODO Type conversion. memcpy can not be used
                                 // directly. dpnp_copy_to() ?
    }

    DPNPC_ptr_adapter<_ComputeDT> result1_ptr(q_ref, result1, size_m * size_m,
                                              true, true);
    DPNPC_ptr_adapter<_SVDT> result2_ptr(q_ref, result2,
                                         std::min(size_m, size_n), true, true);
    DPNPC_ptr_adapter<_ComputeDT> result3_ptr(q_ref, result3, size_n * size_n,
                                              true, true);
    _ComputeDT *res_u = result1_ptr.get_ptr();
    _SVDT *res_s = result2_ptr.get_ptr();
    _ComputeDT *res_vt = result3_ptr.get_ptr();

    const std::int64_t m = size_m;
    const std::int64_t n = size_n;

    const std::int64_t lda = std::max<size_t>(1UL, n);
    const std::int64_t ldu = std::max<size_t>(1UL, m);
    const std::int64_t ldvt = std::max<size_t>(1UL, n);

    const std::int64_t scratchpad_size =
        mkl_lapack::gesvd_scratchpad_size<_ComputeDT>(
            q, oneapi::mkl::jobsvd::vectors, oneapi::mkl::jobsvd::vectors, n, m,
            lda, ldvt, ldu);

    _ComputeDT *scratchpad = reinterpret_cast<_ComputeDT *>(
        sycl::malloc_shared(scratchpad_size * sizeof(_ComputeDT), q));

    event =
        mkl_lapack::gesvd(q,
                          oneapi::mkl::jobsvd::vectors, // onemkl::job jobu,
                          oneapi::mkl::jobsvd::vectors, // onemkl::job jobvt,
                          n, m, in_a, lda, res_s, res_vt, ldvt, res_u, ldu,
                          scratchpad, scratchpad_size);

    event.wait();

    sycl::free(scratchpad, q);

    return event_ref;
}

template <typename _InputDT, typename _ComputeDT, typename _SVDT>
void dpnp_svd_c(void *array1_in,
                void *result1,
                void *result2,
                void *result3,
                size_t size_m,
                size_t size_n)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_svd_c<_InputDT, _ComputeDT, _SVDT>(
        q_ref, array1_in, result1, result2, result3, size_m, size_n,
        dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _InputDT, typename _ComputeDT, typename _SVDT>
void (*dpnp_svd_default_c)(void *, void *, void *, void *, size_t, size_t) =
    dpnp_svd_c<_InputDT, _ComputeDT, _SVDT>;

void func_map_init_linalg_func(func_map_t &fmap)
{
    fmap[DPNPFuncName::DPNP_FN_CHOLESKY][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_cholesky_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_CHOLESKY][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_cholesky_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_DET][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_det_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_DET][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_det_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_DET][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_det_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_DET][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_det_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_INV][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_inv_default_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_INV][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_inv_default_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_INV][eft_FLT][eft_FLT] = {
        eft_DBL, (void *)dpnp_inv_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_INV][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_inv_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_INV_EXT][eft_INT][eft_INT] = {
        get_default_floating_type(),
        (void *)dpnp_inv_ext_c<
            int32_t, func_type_map_t::find_type<get_default_floating_type()>>,
        get_default_floating_type<std::false_type>(),
        (void *)dpnp_inv_ext_c<
            int32_t, func_type_map_t::find_type<
                         get_default_floating_type<std::false_type>()>>};
    fmap[DPNPFuncName::DPNP_FN_INV_EXT][eft_LNG][eft_LNG] = {
        get_default_floating_type(),
        (void *)dpnp_inv_ext_c<
            int64_t, func_type_map_t::find_type<get_default_floating_type()>>,
        get_default_floating_type<std::false_type>(),
        (void *)dpnp_inv_ext_c<
            int64_t, func_type_map_t::find_type<
                         get_default_floating_type<std::false_type>()>>};
    fmap[DPNPFuncName::DPNP_FN_INV_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_inv_ext_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_INV_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_inv_ext_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_KRON][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_kron_default_c<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_INT][eft_LNG] = {
        eft_LNG, (void *)dpnp_kron_default_c<int32_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_INT][eft_FLT] = {
        eft_FLT, (void *)dpnp_kron_default_c<int32_t, float, float>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_kron_default_c<int32_t, double, double>};
    // fmap[DPNPFuncName::DPNP_FN_KRON][eft_INT][eft_C128] = {
    // eft_C128, (void*)dpnp_kron_default_c<int32_t, std::complex<double>,
    // std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_LNG][eft_INT] = {
        eft_LNG, (void *)dpnp_kron_default_c<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_kron_default_c<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_LNG][eft_FLT] = {
        eft_FLT, (void *)dpnp_kron_default_c<int64_t, float, float>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_kron_default_c<int64_t, double, double>};
    // fmap[DPNPFuncName::DPNP_FN_KRON][eft_LNG][eft_C128] = {
    // eft_C128, (void*)dpnp_kron_default_c<int64_t, std::complex<double>,
    // std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_FLT][eft_INT] = {
        eft_FLT, (void *)dpnp_kron_default_c<float, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_FLT][eft_LNG] = {
        eft_FLT, (void *)dpnp_kron_default_c<float, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_kron_default_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_kron_default_c<float, double, double>};
    // fmap[DPNPFuncName::DPNP_FN_KRON][eft_FLT][eft_C128] = {
    // eft_C128, (void*)dpnp_kron_default_c<float, std::complex<double>,
    // std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_DBL][eft_INT] = {
        eft_DBL, (void *)dpnp_kron_default_c<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_DBL][eft_LNG] = {
        eft_DBL, (void *)dpnp_kron_default_c<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_DBL][eft_FLT] = {
        eft_DBL, (void *)dpnp_kron_default_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_kron_default_c<double, double, double>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_DBL][eft_C128] = {
        eft_C128, (void *)dpnp_kron_default_c<double, std::complex<double>,
                                              std::complex<double>>};
    // fmap[DPNPFuncName::DPNP_FN_KRON][eft_C128][eft_INT] = {
    // eft_C128, (void*)dpnp_kron_default_c<std::complex<double>, int32_t,
    // std::complex<double>>};
    // fmap[DPNPFuncName::DPNP_FN_KRON][eft_C128][eft_LNG] = {
    // eft_C128, (void*)dpnp_kron_default_c<std::complex<double>, int64_t,
    // std::complex<double>>};
    // fmap[DPNPFuncName::DPNP_FN_KRON][eft_C128][eft_FLT] = {
    // eft_C128, (void*)dpnp_kron_default_c<std::complex<double>, float,
    // std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_C128][eft_DBL] = {
        eft_C128, (void *)dpnp_kron_default_c<std::complex<double>, double,
                                              std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_KRON][eft_C128][eft_C128] = {
        eft_C128,
        (void *)dpnp_kron_default_c<std::complex<double>, std::complex<double>,
                                    std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_KRON_EXT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_kron_ext_c<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_KRON_EXT][eft_INT][eft_LNG] = {
        eft_LNG, (void *)dpnp_kron_ext_c<int32_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_KRON_EXT][eft_INT][eft_FLT] = {
        eft_FLT, (void *)dpnp_kron_ext_c<int32_t, float, float>};
    fmap[DPNPFuncName::DPNP_FN_KRON_EXT][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_kron_ext_c<int32_t, double, double>};
    // fmap[DPNPFuncName::DPNP_FN_KRON_EXT][eft_INT][eft_C128] = {
    // eft_C128, (void*)dpnp_kron_ext_c<int32_t, std::complex<double>,
    // std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_KRON_EXT][eft_LNG][eft_INT] = {
        eft_LNG, (void *)dpnp_kron_ext_c<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_KRON_EXT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_kron_ext_c<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_KRON_EXT][eft_LNG][eft_FLT] = {
        eft_FLT, (void *)dpnp_kron_ext_c<int64_t, float, float>};
    fmap[DPNPFuncName::DPNP_FN_KRON_EXT][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_kron_ext_c<int64_t, double, double>};
    // fmap[DPNPFuncName::DPNP_FN_KRON_EXT][eft_LNG][eft_C128] = {
    // eft_C128, (void*)dpnp_kron_ext_c<int64_t, std::complex<double>,
    // std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_KRON_EXT][eft_FLT][eft_INT] = {
        eft_FLT, (void *)dpnp_kron_ext_c<float, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_KRON_EXT][eft_FLT][eft_LNG] = {
        eft_FLT, (void *)dpnp_kron_ext_c<float, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_KRON_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_kron_ext_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_KRON_EXT][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_kron_ext_c<float, double, double>};
    // fmap[DPNPFuncName::DPNP_FN_KRON_EXT][eft_FLT][eft_C128] = {
    // eft_C128, (void*)dpnp_kron_ext_c<float, std::complex<double>,
    // std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_KRON_EXT][eft_DBL][eft_INT] = {
        eft_DBL, (void *)dpnp_kron_ext_c<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_KRON_EXT][eft_DBL][eft_LNG] = {
        eft_DBL, (void *)dpnp_kron_ext_c<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_KRON_EXT][eft_DBL][eft_FLT] = {
        eft_DBL, (void *)dpnp_kron_ext_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_KRON_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_kron_ext_c<double, double, double>};
    fmap[DPNPFuncName::DPNP_FN_KRON_EXT][eft_DBL][eft_C128] = {
        eft_C128, (void *)dpnp_kron_ext_c<double, std::complex<double>,
                                          std::complex<double>>};
    // fmap[DPNPFuncName::DPNP_FN_KRON_EXT][eft_C128][eft_INT] = {
    // eft_C128, (void*)dpnp_kron_ext_c<std::complex<double>, int32_t,
    // std::complex<double>>};
    // fmap[DPNPFuncName::DPNP_FN_KRON_EXT][eft_C128][eft_LNG] = {
    // eft_C128, (void*)dpnp_kron_ext_c<std::complex<double>, int64_t,
    // std::complex<double>>};
    // fmap[DPNPFuncName::DPNP_FN_KRON_EXT][eft_C128][eft_FLT] = {
    // eft_C128, (void*)dpnp_kron_ext_c<std::complex<double>, float,
    // std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_KRON_EXT][eft_C128][eft_DBL] = {
        eft_C128, (void *)dpnp_kron_ext_c<std::complex<double>, double,
                                          std::complex<double>>};
    fmap[DPNPFuncName::DPNP_FN_KRON_EXT][eft_C128][eft_C128] = {
        eft_C128,
        (void *)dpnp_kron_ext_c<std::complex<double>, std::complex<double>,
                                std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_MATRIX_RANK][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_matrix_rank_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MATRIX_RANK][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_matrix_rank_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MATRIX_RANK][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_matrix_rank_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_MATRIX_RANK][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_matrix_rank_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_MATRIX_RANK_EXT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_matrix_rank_ext_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MATRIX_RANK_EXT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_matrix_rank_ext_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MATRIX_RANK_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_matrix_rank_ext_c<float>};
    fmap[DPNPFuncName::DPNP_FN_MATRIX_RANK_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_matrix_rank_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_QR][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_qr_default_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_QR][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_qr_default_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_QR][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_qr_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_QR][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_qr_default_c<double, double>};
    // fmap[DPNPFuncName::DPNP_FN_QR][eft_C128][eft_C128] = {
    // eft_C128, (void*)dpnp_qr_c<std::complex<double>, std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_QR_EXT][eft_INT][eft_INT] = {
        get_default_floating_type(),
        (void *)dpnp_qr_ext_c<
            int32_t, func_type_map_t::find_type<get_default_floating_type()>>,
        get_default_floating_type<std::false_type>(),
        (void *)dpnp_qr_ext_c<
            int32_t, func_type_map_t::find_type<
                         get_default_floating_type<std::false_type>()>>};
    fmap[DPNPFuncName::DPNP_FN_QR_EXT][eft_LNG][eft_LNG] = {
        get_default_floating_type(),
        (void *)dpnp_qr_ext_c<
            int64_t, func_type_map_t::find_type<get_default_floating_type()>>,
        get_default_floating_type<std::false_type>(),
        (void *)dpnp_qr_ext_c<
            int64_t, func_type_map_t::find_type<
                         get_default_floating_type<std::false_type>()>>};
    fmap[DPNPFuncName::DPNP_FN_QR_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_qr_ext_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_QR_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_qr_ext_c<double, double>};
    // fmap[DPNPFuncName::DPNP_FN_QR_EXT][eft_C128][eft_C128] = {
    // eft_C128, (void*)dpnp_qr_c<std::complex<double>, std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_SVD][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_svd_default_c<int32_t, double, double>};
    fmap[DPNPFuncName::DPNP_FN_SVD][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_svd_default_c<int64_t, double, double>};
    fmap[DPNPFuncName::DPNP_FN_SVD][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_svd_default_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_SVD][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_svd_default_c<double, double, double>};
    fmap[DPNPFuncName::DPNP_FN_SVD][eft_C128][eft_C128] = {
        eft_C128, (void *)dpnp_svd_default_c<std::complex<double>,
                                             std::complex<double>, double>};

    return;
}
