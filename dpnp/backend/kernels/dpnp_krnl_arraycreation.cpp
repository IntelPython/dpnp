//*****************************************************************************
// Copyright (c) 2016-2022, Intel Corporation
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

#include "dpnp_fptr.hpp"
#include "dpnp_iface.hpp"
#include "dpnp_utils.hpp"
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"

template <typename _KernelNameSpecialization>
class dpnp_arange_c_kernel;

template <typename _DataType>
DPCTLSyclEventRef dpnp_arange_c(DPCTLSyclQueueRef q_ref,
                                size_t start,
                                size_t step,
                                void* result1,
                                size_t size,
                                const DPCTLEventVectorRef dep_event_vec_ref)
{
    // parameter `size` used instead `stop` to avoid dependency on array length calculation algorithm
    // TODO: floating point (and negatives) types from `start` and `step`

    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!size)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));
    sycl::event event;

    validate_type_for_device<_DataType>(q);

    _DataType* result = reinterpret_cast<_DataType*>(result1);

    sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
        size_t i = global_id[0];

        result[i] = start + i * step;
    };

    auto kernel_func = [&](sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_arange_c_kernel<_DataType>>(gws, kernel_parallel_for_func);
    };

    event = q.submit(kernel_func);
    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_arange_c(size_t start, size_t step, void* result1, size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_arange_c<_DataType>(q_ref,
                                                           start,
                                                           step,
                                                           result1,
                                                           size,
                                                           dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_arange_default_c)(size_t, size_t, void*, size_t) = dpnp_arange_c<_DataType>;

// Explicit instantiation of the function, since dpnp_arange_c() is used by other template functions,
// but implicit instantiation is not applied anymore.
template DPCTLSyclEventRef dpnp_arange_c<int32_t>(DPCTLSyclQueueRef,
                                                  size_t,
                                                  size_t,
                                                  void*,
                                                  size_t,
                                                  const DPCTLEventVectorRef);

template DPCTLSyclEventRef dpnp_arange_c<int64_t>(DPCTLSyclQueueRef,
                                                  size_t,
                                                  size_t,
                                                  void*,
                                                  size_t,
                                                  const DPCTLEventVectorRef);

template DPCTLSyclEventRef dpnp_arange_c<float>(DPCTLSyclQueueRef,
                                                size_t,
                                                size_t,
                                                void*,
                                                size_t,
                                                const DPCTLEventVectorRef);

template DPCTLSyclEventRef dpnp_arange_c<double>(DPCTLSyclQueueRef,
                                                 size_t,
                                                 size_t,
                                                 void*,
                                                 size_t,
                                                 const DPCTLEventVectorRef);

template <typename _DataType>
DPCTLSyclEventRef dpnp_diag_c(DPCTLSyclQueueRef q_ref,
                              void* v_in,
                              void* result1,
                              const int k,
                              shape_elem_type* shape,
                              shape_elem_type* res_shape,
                              const size_t ndim,
                              const size_t res_ndim,
                              const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)res_ndim;
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;
    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    validate_type_for_device<_DataType>(q);

    const size_t input1_size = std::accumulate(shape, shape + ndim, 1, std::multiplies<shape_elem_type>());
    const size_t result_size = std::accumulate(res_shape, res_shape + res_ndim, 1, std::multiplies<shape_elem_type>());
    DPNPC_ptr_adapter<_DataType> input1_ptr(q_ref,v_in, input1_size, true);
    DPNPC_ptr_adapter<_DataType> result_ptr(q_ref,result1, result_size, true, true);
    _DataType* v = input1_ptr.get_ptr();
    _DataType* result = result_ptr.get_ptr();

    size_t init0 = std::max(0, -k);
    size_t init1 = std::max(0, k);

    if (ndim == 1)
    {
        for (size_t i = 0; i < static_cast<size_t>(shape[0]); ++i)
        {
            size_t ind = (init0 + i) * res_shape[1] + init1 + i;
            result[ind] = v[i];
        }
    }
    else
    {
        for (size_t i = 0; i < static_cast<size_t>(res_shape[0]); ++i)
        {
            size_t ind = (init0 + i) * shape[1] + init1 + i;
            result[i] = v[ind];
        }
    }
    return event_ref;
}

template <typename _DataType>
void dpnp_diag_c(void* v_in,
                 void* result1,
                 const int k,
                 shape_elem_type* shape,
                 shape_elem_type* res_shape,
                 const size_t ndim,
                 const size_t res_ndim)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_diag_c<_DataType>(q_ref,
                                                         v_in,
                                                         result1,
                                                         k,
                                                         shape,
                                                         res_shape,
                                                         ndim,
                                                         res_ndim,
                                                         dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_diag_default_c)(void*,
                            void*,
                            const int,
                            shape_elem_type*,
                            shape_elem_type*,
                            const size_t,
                            const size_t) = dpnp_diag_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_diag_ext_c)(DPCTLSyclQueueRef,
                                     void*,
                                     void*,
                                     const int,
                                     shape_elem_type*,
                                     shape_elem_type*,
                                     const size_t,
                                     const size_t,
                                     const DPCTLEventVectorRef) = dpnp_diag_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_eye_c(DPCTLSyclQueueRef q_ref,
                             void* result1,
                             int k,
                             const shape_elem_type* res_shape,
                             const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (result1 == nullptr)
    {
        return event_ref;
    }

    if (res_shape == nullptr)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    validate_type_for_device<_DataType>(q);

    size_t result_size = res_shape[0] * res_shape[1];

    DPNPC_ptr_adapter<_DataType> result_ptr(q_ref,result1, result_size, true, true);
    _DataType* result = result_ptr.get_ptr();

    int diag_val_;
    diag_val_ = std::min((int)res_shape[0], (int)res_shape[1]);
    diag_val_ = std::min(diag_val_, ((int)res_shape[0] + k));
    diag_val_ = std::min(diag_val_, ((int)res_shape[1] - k));

    size_t diag_val = (diag_val_ < 0) ? 0 : (size_t)diag_val_;

    for (size_t i = 0; i < result_size; ++i)
    {
        result[i] = 0;
        for (size_t j = 0; j < diag_val; ++j)
        {
            size_t ind = (k >= 0) ? (j * res_shape[1] + j + k) : (j - k) * res_shape[1] + j;
            if (i == ind)
            {
                result[i] = 1;
                break;
            }
        }
    }

    return event_ref;
}

template <typename _DataType>
void dpnp_eye_c(void* result1, int k, const shape_elem_type* res_shape)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_eye_c<_DataType>(q_ref,
                                                        result1,
                                                        k,
                                                        res_shape,
                                                        dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_eye_default_c)(void*, int, const shape_elem_type*) = dpnp_eye_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_full_c(DPCTLSyclQueueRef q_ref,
                              void* array_in,
                              void* result,
                              const size_t size,
                              const DPCTLEventVectorRef dep_event_vec_ref)
{
    return dpnp_initval_c<_DataType>(q_ref, result, array_in, size, dep_event_vec_ref);
}

template <typename _DataType>
void dpnp_full_c(void* array_in, void* result, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_full_c<_DataType>(q_ref,
                                                         array_in,
                                                         result,
                                                         size,
                                                         dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_full_default_c)(void*, void*, const size_t) = dpnp_full_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_full_like_c(DPCTLSyclQueueRef q_ref,
                                   void* array_in,
                                   void* result,
                                   const size_t size,
                                   const DPCTLEventVectorRef dep_event_vec_ref)
{
    return dpnp_full_c<_DataType>(q_ref, array_in, result, size, dep_event_vec_ref);
}

template <typename _DataType>
void dpnp_full_like_c(void* array_in, void* result, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_full_like_c<_DataType>(q_ref,
                                                              array_in,
                                                              result,
                                                              size,
                                                              dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_full_like_default_c)(void*, void*, const size_t) = dpnp_full_like_c<_DataType>;

template <typename _KernelNameSpecialization>
class dpnp_identity_c_kernel;

template <typename _DataType>
DPCTLSyclEventRef dpnp_identity_c(DPCTLSyclQueueRef q_ref,
                                  void* result1,
                                  const size_t n,
                                  const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (n == 0)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));
    sycl::event event;

    validate_type_for_device<_DataType>(q);

    _DataType* result = static_cast<_DataType *>(result1);

    sycl::range<2> gws(n, n);
    auto kernel_parallel_for_func = [=](sycl::id<2> global_id) {
        size_t i = global_id[0];
        size_t j = global_id[1];
        result[i * n + j] = i == j;
    };

    auto kernel_func = [&](sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_identity_c_kernel<_DataType>>(gws, kernel_parallel_for_func);
    };

    event = q.submit(kernel_func);
    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_identity_c(void* result1, const size_t n)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_identity_c<_DataType>(q_ref,
                                                             result1,
                                                             n,
                                                             dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_identity_default_c)(void*, const size_t) = dpnp_identity_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_identity_ext_c)(DPCTLSyclQueueRef,
                                         void*,
                                         const size_t,
                                         const DPCTLEventVectorRef) = dpnp_identity_c<_DataType>;

template <typename _DataType>
class dpnp_ones_c_kernel;

template <typename _DataType>
DPCTLSyclEventRef dpnp_ones_c(DPCTLSyclQueueRef q_ref,
                              void* result,
                              size_t size,
                              const DPCTLEventVectorRef dep_event_vec_ref)
{
    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    _DataType* fill_value = reinterpret_cast<_DataType*>(sycl::malloc_shared(sizeof(_DataType), q));
    fill_value[0] = 1;

    DPCTLSyclEventRef event_ref = dpnp_initval_c<_DataType>(q_ref, result, fill_value, size, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);

    sycl::free(fill_value, q);

    return nullptr;
}

template <typename _DataType>
void dpnp_ones_c(void* result, size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_ones_c<_DataType>(q_ref,
                                                         result,
                                                         size,
                                                         dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_ones_default_c)(void*, size_t) = dpnp_ones_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_ones_like_c(DPCTLSyclQueueRef q_ref,
                                   void* result,
                                   size_t size,
                                   const DPCTLEventVectorRef dep_event_vec_ref)
{
    return dpnp_ones_c<_DataType>(q_ref, result, size, dep_event_vec_ref);
}

template <typename _DataType>
void dpnp_ones_like_c(void* result, size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_ones_like_c<_DataType>(q_ref,
                                                              result,
                                                              size,
                                                              dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_ones_like_default_c)(void*, size_t) = dpnp_ones_like_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_ptp_c(DPCTLSyclQueueRef q_ref,
                             void* result1_out,
                             const size_t result_size,
                             const size_t result_ndim,
                             const shape_elem_type* result_shape,
                             const shape_elem_type* result_strides,
                             const void* input1_in,
                             const size_t input_size,
                             const size_t input_ndim,
                             const shape_elem_type* input_shape,
                             const shape_elem_type* input_strides,
                             const shape_elem_type* axis,
                             const size_t naxis,
                             const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)result_strides;
    (void)input_strides;
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;
    DPCTLSyclEventRef e1_ref = nullptr;
    DPCTLSyclEventRef e2_ref = nullptr;
    DPCTLSyclEventRef e3_ref = nullptr;

    if ((input1_in == nullptr) || (result1_out == nullptr))
    {
        return event_ref;
    }

    if (input_ndim < 1)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    validate_type_for_device<_DataType>(q);

    DPNPC_ptr_adapter<_DataType> input1_ptr(q_ref,input1_in, input_size, true);
    DPNPC_ptr_adapter<_DataType> result_ptr(q_ref,result1_out, result_size, false, true);
    _DataType* arr = input1_ptr.get_ptr();
    _DataType* result = result_ptr.get_ptr();

    _DataType* min_arr = reinterpret_cast<_DataType*>(sycl::malloc_shared(result_size * sizeof(_DataType), q));
    _DataType* max_arr = reinterpret_cast<_DataType*>(sycl::malloc_shared(result_size * sizeof(_DataType), q));

    e1_ref = dpnp_min_c<_DataType>(q_ref, arr, min_arr, result_size, input_shape, input_ndim, axis, naxis, NULL);
    e2_ref = dpnp_max_c<_DataType>(q_ref, arr, max_arr, result_size, input_shape, input_ndim, axis, naxis, NULL);

    shape_elem_type* _strides =
        reinterpret_cast<shape_elem_type*>(sycl::malloc_shared(result_ndim * sizeof(shape_elem_type), q));
    get_shape_offsets_inkernel(result_shape, result_ndim, _strides);

    e3_ref = dpnp_subtract_c<_DataType, _DataType, _DataType>(q_ref, result,
							      result_size,
							      result_ndim,
							      result_shape,
							      result_strides,
							      max_arr,
							      result_size,
							      result_ndim,
							      result_shape,
							      _strides,
							      min_arr,
							      result_size,
							      result_ndim,
							      result_shape,
							      _strides,
							      NULL, NULL);

    DPCTLEvent_Wait(e1_ref);
    DPCTLEvent_Wait(e2_ref);
    DPCTLEvent_Wait(e3_ref);
    DPCTLEvent_Delete(e1_ref);
    DPCTLEvent_Delete(e2_ref);
    DPCTLEvent_Delete(e3_ref);

    sycl::free(min_arr, q);
    sycl::free(max_arr, q);
    sycl::free(_strides, q);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_ptp_c(void* result1_out,
                const size_t result_size,
                const size_t result_ndim,
                const shape_elem_type* result_shape,
                const shape_elem_type* result_strides,
                const void* input1_in,
                const size_t input_size,
                const size_t input_ndim,
                const shape_elem_type* input_shape,
                const shape_elem_type* input_strides,
                const shape_elem_type* axis,
                const size_t naxis)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_ptp_c<_DataType>(q_ref,
                                                        result1_out,
                                                        result_size,
                                                        result_ndim,
                                                        result_shape,
                                                        result_strides,
                                                        input1_in,
                                                        input_size,
                                                        input_ndim,
                                                        input_shape,
                                                        input_strides,
                                                        axis,
                                                        naxis,
                                                        dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_ptp_default_c)(void*,
                           const size_t,
                           const size_t,
                           const shape_elem_type*,
                           const shape_elem_type*,
                           const void*,
                           const size_t,
                           const size_t,
                           const shape_elem_type*,
                           const shape_elem_type*,
                           const shape_elem_type*,
                           const size_t) = dpnp_ptp_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_ptp_ext_c)(DPCTLSyclQueueRef,
                                    void*,
                                    const size_t,
                                    const size_t,
                                    const shape_elem_type*,
                                    const shape_elem_type*,
                                    const void*,
                                    const size_t,
                                    const size_t,
                                    const shape_elem_type*,
                                    const shape_elem_type*,
                                    const shape_elem_type*,
                                    const size_t,
                                    const DPCTLEventVectorRef) = dpnp_ptp_c<_DataType>;

template <typename _DataType_input, typename _DataType_output>
DPCTLSyclEventRef dpnp_vander_c(DPCTLSyclQueueRef q_ref,
                                const void* array1_in,
                                void* result1,
                                const size_t size_in,
                                const size_t N,
                                const int increasing,
                                const DPCTLEventVectorRef dep_event_vec_ref)
{
    DPCTLSyclEventRef event_ref = nullptr;

    if ((array1_in == nullptr) || (result1 == nullptr))
        return event_ref;

    if (!size_in || !N)
        return event_ref;

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    validate_type_for_device<_DataType_input>(q);
    validate_type_for_device<_DataType_output>(q);

    DPNPC_ptr_adapter<_DataType_input> input1_ptr(q_ref,array1_in, size_in, true);
    DPNPC_ptr_adapter<_DataType_output> result_ptr(q_ref,result1, size_in * N, true, true);
    const _DataType_input* array_in = input1_ptr.get_ptr();
    _DataType_output* result = result_ptr.get_ptr();

    if (N == 1)
    {
        return dpnp_ones_c<_DataType_output>(q_ref, result, size_in, dep_event_vec_ref);
    }

    if (increasing)
    {
        for (size_t i = 0; i < size_in; ++i)
        {
            result[i * N] = 1;
        }
        for (size_t i = 1; i < N; ++i)
        {
            for (size_t j = 0; j < size_in; ++j)
            {
                result[j * N + i] = result[j * N + i - 1] * array_in[j];
            }
        }
    }
    else
    {
        for (size_t i = 0; i < size_in; ++i)
        {
            result[i * N + N - 1] = 1;
        }
        for (size_t i = N - 2; i > 0; --i)
        {
            for (size_t j = 0; j < size_in; ++j)
            {
                result[j * N + i] = result[j * N + i + 1] * array_in[j];
            }
        }

        for (size_t i = 0; i < size_in; ++i)
        {
            result[i * N] = result[i * N + 1] * array_in[i];
        }
    }

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType_input, typename _DataType_output>
void dpnp_vander_c(const void* array1_in, void* result1, const size_t size_in, const size_t N, const int increasing)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_vander_c<_DataType_input, _DataType_output>(q_ref,
                                                                                   array1_in,
                                                                                   result1,
                                                                                   size_in,
                                                                                   N,
                                                                                   increasing,
                                                                                   dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType_input, typename _DataType_output>
void (*dpnp_vander_default_c)(const void*,
                              void*,
                              const size_t,
                              const size_t,
                              const int) = dpnp_vander_c<_DataType_input, _DataType_output>;

template <typename _DataType_input, typename _DataType_output>
DPCTLSyclEventRef (*dpnp_vander_ext_c)(DPCTLSyclQueueRef,
                                       const void*,
                                       void*,
                                       const size_t,
                                       const size_t,
                                       const int,
                                       const DPCTLEventVectorRef) = dpnp_vander_c<_DataType_input, _DataType_output>;

template <typename _DataType, typename _ResultType>
class dpnp_trace_c_kernel;

template <typename _DataType, typename _ResultType>
DPCTLSyclEventRef dpnp_trace_c(DPCTLSyclQueueRef q_ref,
                               const void* array1_in,
                               void* result_in,
                               const shape_elem_type* shape_,
                               const size_t ndim,
                               const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!array1_in || !result_in || !shape_ || !ndim)
    {
        return event_ref;
    }

    const size_t last_dim = shape_[ndim - 1];
    const size_t size = std::accumulate(shape_, shape_ + (ndim - 1), 1, std::multiplies<shape_elem_type>());
    if (!size)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    validate_type_for_device<_DataType>(q);
    validate_type_for_device<_ResultType>(q);

    const _DataType* input = static_cast<const _DataType *>(array1_in);
    _ResultType* result = static_cast<_ResultType *>(result_in);

    sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](auto index) {
        size_t i = index[0];
        _ResultType acc = _ResultType(0);

        for (size_t j = 0; j < last_dim; ++j)
        {
            acc += input[i * last_dim + j];
        }

        result[i] = acc;
    };

    auto kernel_func = [&](sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_trace_c_kernel<_DataType, _ResultType>>(gws, kernel_parallel_for_func);
    };

    auto event = q.submit(kernel_func);
    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType, typename _ResultType>
void dpnp_trace_c(const void* array1_in, void* result_in, const shape_elem_type* shape_, const size_t ndim)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_trace_c<_DataType, _ResultType>(q_ref,
                                                                       array1_in,
                                                                       result_in,
                                                                       shape_,
                                                                       ndim,
                                                                       dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType, typename _ResultType>
void (*dpnp_trace_default_c)(const void*,
                             void*,
                             const shape_elem_type*,
                             const size_t) = dpnp_trace_c<_DataType, _ResultType>;

template <typename _DataType, typename _ResultType>
DPCTLSyclEventRef (*dpnp_trace_ext_c)(DPCTLSyclQueueRef,
                                      const void*,
                                      void*,
                                      const shape_elem_type*,
                                      const size_t,
                                      const DPCTLEventVectorRef) = dpnp_trace_c<_DataType, _ResultType>;

template <typename _DataType>
class dpnp_tri_c_kernel;

template <typename _DataType>
DPCTLSyclEventRef dpnp_tri_c(DPCTLSyclQueueRef q_ref,
                             void* result1,
                             const size_t N,
                             const size_t M,
                             const int k,
                             const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    sycl::event event;

    if (!result1 || !N || !M)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    validate_type_for_device<_DataType>(q);

    _DataType* result = static_cast<_DataType *>(result1);

    size_t idx = N * M;
    sycl::range<1> gws(idx);
    auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
        size_t ind = global_id[0];
        size_t i = ind / M;
        size_t j = ind % M;

        int val = i + k + 1;
        size_t diag_idx_ = (val > 0) ? (size_t)val : 0;
        size_t diag_idx = (M < diag_idx_) ? M : diag_idx_;

        if (j < diag_idx)
        {
            result[ind] = 1;
        }
        else
        {
            result[ind] = 0;
        }
    };

    auto kernel_func = [&](sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_tri_c_kernel<_DataType>>(gws, kernel_parallel_for_func);
    };

    event = q.submit(kernel_func);
    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_tri_c(void* result1, const size_t N, const size_t M, const int k)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_tri_c<_DataType>(q_ref,
                                                        result1,
                                                        N,
                                                        M,
                                                        k,
                                                        dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_tri_default_c)(void*,
                           const size_t,
                           const size_t,
                           const int) = dpnp_tri_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_tri_ext_c)(DPCTLSyclQueueRef,
                                    void*,
                                    const size_t,
                                    const size_t,
                                    const int,
                                    const DPCTLEventVectorRef) = dpnp_tri_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_tril_c(DPCTLSyclQueueRef q_ref,
                              void* array_in,
                              void* result1,
                              const int k,
                              shape_elem_type* shape,
                              shape_elem_type* res_shape,
                              const size_t ndim,
                              const size_t res_ndim,
                              const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if ((array_in == nullptr) || (result1 == nullptr))
    {
        return event_ref;
    }

    if ((shape == nullptr) || (res_shape == nullptr))
    {
        return event_ref;
    }

    if ((ndim == 0) || (res_ndim == 0))
    {
        return event_ref;
    }

    const size_t res_size = std::accumulate(res_shape, res_shape + res_ndim, 1, std::multiplies<shape_elem_type>());
    if (res_size == 0)
    {
        return event_ref;
    }

    const size_t input_size = std::accumulate(shape, shape + ndim, 1, std::multiplies<shape_elem_type>());
    if (input_size == 0)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    validate_type_for_device<_DataType>(q);

    DPNPC_ptr_adapter<_DataType> input1_ptr(q_ref,array_in, input_size, true);
    DPNPC_ptr_adapter<_DataType> result_ptr(q_ref,result1, res_size, true, true);
    _DataType* array_m = input1_ptr.get_ptr();
    _DataType* result = result_ptr.get_ptr();

    if (ndim == 1)
    {
        for (size_t i = 0; i < res_size; ++i)
        {
            size_t n = res_size;
            size_t val = i;
            int ids[res_ndim];
            for (size_t j = 0; j < res_ndim; ++j)
            {
                n /= res_shape[j];
                size_t p = val / n;
                ids[j] = p;
                if (p != 0)
                {
                    val = val - p * n;
                }
            }

            int diag_idx_ = (ids[res_ndim - 2] + k > -1) ? (ids[res_ndim - 2] + k) : -1;
            int values = res_shape[res_ndim - 1];
            int diag_idx = (values < diag_idx_) ? values : diag_idx_;

            if (ids[res_ndim - 1] <= diag_idx)
            {
                result[i] = array_m[ids[res_ndim - 1]];
            }
            else
            {
                result[i] = 0;
            }
        }
    }
    else
    {
        for (size_t i = 0; i < res_size; ++i)
        {
            size_t n = res_size;
            size_t val = i;
            int ids[res_ndim];
            for (size_t j = 0; j < res_ndim; ++j)
            {
                n /= res_shape[j];
                size_t p = val / n;
                ids[j] = p;
                if (p != 0)
                {
                    val = val - p * n;
                }
            }

            int diag_idx_ = (ids[res_ndim - 2] + k > -1) ? (ids[res_ndim - 2] + k) : -1;
            int values = res_shape[res_ndim - 1];
            int diag_idx = (values < diag_idx_) ? values : diag_idx_;

            if (ids[res_ndim - 1] <= diag_idx)
            {
                result[i] = array_m[i];
            }
            else
            {
                result[i] = 0;
            }
        }
    }
    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_tril_c(void* array_in,
                 void* result1,
                 const int k,
                 shape_elem_type* shape,
                 shape_elem_type* res_shape,
                 const size_t ndim,
                 const size_t res_ndim)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_tril_c<_DataType>(q_ref,
                                                         array_in,
                                                         result1,
                                                         k,
                                                         shape,
                                                         res_shape,
                                                         ndim,
                                                         res_ndim,
                                                         dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_tril_default_c)(void*,
                            void*,
                            const int,
                            shape_elem_type*,
                            shape_elem_type*,
                            const size_t,
                            const size_t) = dpnp_tril_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_tril_ext_c)(DPCTLSyclQueueRef,
                                     void*,
                                     void*,
                                     const int,
                                     shape_elem_type*,
                                     shape_elem_type*,
                                     const size_t,
                                     const size_t,
                                     const DPCTLEventVectorRef) = dpnp_tril_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_triu_c(DPCTLSyclQueueRef q_ref,
                              void* array_in,
                              void* result1,
                              const int k,
                              shape_elem_type* shape,
                              shape_elem_type* res_shape,
                              const size_t ndim,
                              const size_t res_ndim,
                              const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if ((array_in == nullptr) || (result1 == nullptr))
    {
        return event_ref;
    }

    if ((shape == nullptr) || (res_shape == nullptr))
    {
        return event_ref;
    }

    if ((ndim == 0) || (res_ndim == 0))
    {
        return event_ref;
    }

    const size_t res_size = std::accumulate(res_shape, res_shape + res_ndim, 1, std::multiplies<shape_elem_type>());
    if (res_size == 0)
    {
        return event_ref;
    }

    const size_t input_size = std::accumulate(shape, shape + ndim, 1, std::multiplies<shape_elem_type>());
    if (input_size == 0)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    validate_type_for_device<_DataType>(q);

    DPNPC_ptr_adapter<_DataType> input1_ptr(q_ref,array_in, input_size, true);
    DPNPC_ptr_adapter<_DataType> result_ptr(q_ref,result1, res_size, true, true);
    _DataType* array_m = input1_ptr.get_ptr();
    _DataType* result = result_ptr.get_ptr();

    if (ndim == 1)
    {
        for (size_t i = 0; i < res_size; ++i)
        {
            size_t n = res_size;
            size_t val = i;
            int ids[res_ndim];
            for (size_t j = 0; j < res_ndim; ++j)
            {
                n /= res_shape[j];
                size_t p = val / n;
                ids[j] = p;
                if (p != 0)
                {
                    val = val - p * n;
                }
            }

            int diag_idx_ = (ids[res_ndim - 2] + k > -1) ? (ids[res_ndim - 2] + k) : -1;
            int values = res_shape[res_ndim - 1];
            int diag_idx = (values < diag_idx_) ? values : diag_idx_;

            if (ids[res_ndim - 1] >= diag_idx)
            {
                result[i] = array_m[ids[res_ndim - 1]];
            }
            else
            {
                result[i] = 0;
            }
        }
    }
    else
    {
        for (size_t i = 0; i < res_size; ++i)
        {
            size_t n = res_size;
            size_t val = i;
            int ids[res_ndim];
            for (size_t j = 0; j < res_ndim; ++j)
            {
                n /= res_shape[j];
                size_t p = val / n;
                ids[j] = p;
                if (p != 0)
                {
                    val = val - p * n;
                }
            }

            int diag_idx_ = (ids[res_ndim - 2] + k > -1) ? (ids[res_ndim - 2] + k) : -1;
            int values = res_shape[res_ndim - 1];
            int diag_idx = (values < diag_idx_) ? values : diag_idx_;

            if (ids[res_ndim - 1] >= diag_idx)
            {
                result[i] = array_m[i];
            }
            else
            {
                result[i] = 0;
            }
        }
    }
    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_triu_c(void* array_in,
                 void* result1,
                 const int k,
                 shape_elem_type* shape,
                 shape_elem_type* res_shape,
                 const size_t ndim,
                 const size_t res_ndim)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_triu_c<_DataType>(q_ref,
                                                         array_in,
                                                         result1,
                                                         k,
                                                         shape,
                                                         res_shape,
                                                         ndim,
                                                         res_ndim,
                                                         dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_triu_default_c)(void*,
                            void*,
                            const int,
                            shape_elem_type*,
                            shape_elem_type*,
                            const size_t,
                            const size_t) = dpnp_triu_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_triu_ext_c)(DPCTLSyclQueueRef,
                                     void*,
                                     void*,
                                     const int,
                                     shape_elem_type*,
                                     shape_elem_type*,
                                     const size_t,
                                     const size_t,
                                     const DPCTLEventVectorRef) = dpnp_triu_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_zeros_c(DPCTLSyclQueueRef q_ref,
                               void* result,
                               size_t size,
                               const DPCTLEventVectorRef dep_event_vec_ref)
{
    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    _DataType* fill_value = reinterpret_cast<_DataType*>(sycl::malloc_shared(sizeof(_DataType), q));
    fill_value[0] = 0;

    DPCTLSyclEventRef event_ref = dpnp_initval_c<_DataType>(q_ref, result, fill_value, size, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);

    sycl::free(fill_value, q);

    return nullptr;
}

template <typename _DataType>
void dpnp_zeros_c(void* result, size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_zeros_c<_DataType>(q_ref,
                                                          result,
                                                          size,
                                                          dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_zeros_default_c)(void*, size_t) = dpnp_zeros_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_zeros_like_c(DPCTLSyclQueueRef q_ref,
                                    void* result,
                                    size_t size,
                                    const DPCTLEventVectorRef dep_event_vec_ref)
{
    return dpnp_zeros_c<_DataType>(q_ref, result, size, dep_event_vec_ref);
}

template <typename _DataType>
void dpnp_zeros_like_c(void* result, size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_zeros_like_c<_DataType>(q_ref,
                                                               result,
                                                               size,
                                                               dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_zeros_like_default_c)(void*, size_t) = dpnp_zeros_like_c<_DataType>;

void func_map_init_arraycreation(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_ARANGE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_arange_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARANGE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_arange_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARANGE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_arange_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_ARANGE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_arange_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_DIAG][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_diag_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_DIAG][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_diag_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_DIAG][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_diag_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_DIAG][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_diag_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_DIAG_EXT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_diag_ext_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_DIAG_EXT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_diag_ext_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_DIAG_EXT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_diag_ext_c<float>};
    fmap[DPNPFuncName::DPNP_FN_DIAG_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_diag_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_EYE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_eye_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_EYE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_eye_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_EYE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_eye_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_EYE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_eye_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_FULL][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_full_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_FULL][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_full_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_FULL][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_full_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_FULL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_full_default_c<double>};
    fmap[DPNPFuncName::DPNP_FN_FULL][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_full_default_c<bool>};
    fmap[DPNPFuncName::DPNP_FN_FULL][eft_C128][eft_C128] = {eft_C128,
                                                            (void*)dpnp_full_default_c<std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_FULL_LIKE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_full_like_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_FULL_LIKE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_full_like_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_FULL_LIKE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_full_like_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_FULL_LIKE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_full_like_default_c<double>};
    fmap[DPNPFuncName::DPNP_FN_FULL_LIKE][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_full_like_default_c<bool>};
    fmap[DPNPFuncName::DPNP_FN_FULL_LIKE][eft_C128][eft_C128] = {
        eft_C128, (void*)dpnp_full_like_default_c<std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_IDENTITY][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_identity_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_IDENTITY][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_identity_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_IDENTITY][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_identity_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_IDENTITY][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_identity_default_c<double>};
    fmap[DPNPFuncName::DPNP_FN_IDENTITY][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_identity_default_c<bool>};
    fmap[DPNPFuncName::DPNP_FN_IDENTITY][eft_C128][eft_C128] = {eft_C128,
                                                                (void*)dpnp_identity_default_c<std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_IDENTITY_EXT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_identity_ext_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_IDENTITY_EXT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_identity_ext_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_IDENTITY_EXT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_identity_ext_c<float>};
    fmap[DPNPFuncName::DPNP_FN_IDENTITY_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_identity_ext_c<double>};
    fmap[DPNPFuncName::DPNP_FN_IDENTITY_EXT][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_identity_ext_c<bool>};
    fmap[DPNPFuncName::DPNP_FN_IDENTITY_EXT][eft_C64][eft_C64] = {eft_C64,
                                                                    (void*)dpnp_identity_ext_c<std::complex<float>>};
    fmap[DPNPFuncName::DPNP_FN_IDENTITY_EXT][eft_C128][eft_C128] = {eft_C128,
                                                                    (void*)dpnp_identity_ext_c<std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_ONES][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_ones_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ONES][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_ones_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ONES][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_ones_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_ONES][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_ones_default_c<double>};
    fmap[DPNPFuncName::DPNP_FN_ONES][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_ones_default_c<bool>};
    fmap[DPNPFuncName::DPNP_FN_ONES][eft_C128][eft_C128] = {eft_C128,
                                                            (void*)dpnp_ones_default_c<std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_ONES_LIKE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_ones_like_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ONES_LIKE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_ones_like_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ONES_LIKE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_ones_like_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_ONES_LIKE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_ones_like_default_c<double>};
    fmap[DPNPFuncName::DPNP_FN_ONES_LIKE][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_ones_like_default_c<bool>};
    fmap[DPNPFuncName::DPNP_FN_ONES_LIKE][eft_C128][eft_C128] = {
        eft_C128, (void*)dpnp_ones_like_default_c<std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_PTP][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_ptp_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_PTP][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_ptp_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_PTP][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_ptp_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_PTP][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_ptp_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_PTP_EXT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_ptp_ext_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_PTP_EXT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_ptp_ext_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_PTP_EXT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_ptp_ext_c<float>};
    fmap[DPNPFuncName::DPNP_FN_PTP_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_ptp_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_VANDER][eft_INT][eft_INT] = {eft_LNG, (void*)dpnp_vander_default_c<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_VANDER][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_vander_default_c<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_VANDER][eft_FLT][eft_FLT] = {eft_DBL, (void*)dpnp_vander_default_c<float, double>};
    fmap[DPNPFuncName::DPNP_FN_VANDER][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_vander_default_c<double, double>};
    fmap[DPNPFuncName::DPNP_FN_VANDER][eft_BLN][eft_BLN] = {eft_LNG, (void*)dpnp_vander_default_c<bool, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_VANDER][eft_C128][eft_C128] = {
        eft_C128, (void*)dpnp_vander_default_c<std::complex<double>, std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_VANDER_EXT][eft_INT][eft_INT] = {eft_LNG, (void*)dpnp_vander_ext_c<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_VANDER_EXT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_vander_ext_c<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_VANDER_EXT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_vander_ext_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_VANDER_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_vander_ext_c<double, double>};
    fmap[DPNPFuncName::DPNP_FN_VANDER_EXT][eft_BLN][eft_BLN] = {eft_LNG, (void*)dpnp_vander_ext_c<bool, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_VANDER_EXT][eft_C64][eft_C64] = {
        eft_C64, (void*)dpnp_vander_ext_c<std::complex<float>, std::complex<float>>};
    fmap[DPNPFuncName::DPNP_FN_VANDER_EXT][eft_C128][eft_C128] = {
        eft_C128, (void*)dpnp_vander_ext_c<std::complex<double>, std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_TRACE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_trace_default_c<int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_TRACE][eft_LNG][eft_INT] = {eft_INT, (void*)dpnp_trace_default_c<int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_TRACE][eft_FLT][eft_INT] = {eft_INT, (void*)dpnp_trace_default_c<float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_TRACE][eft_DBL][eft_INT] = {eft_INT, (void*)dpnp_trace_default_c<double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_TRACE][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_trace_default_c<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_TRACE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_trace_default_c<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_TRACE][eft_FLT][eft_LNG] = {eft_LNG, (void*)dpnp_trace_default_c<float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_TRACE][eft_DBL][eft_LNG] = {eft_LNG, (void*)dpnp_trace_default_c<double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_TRACE][eft_INT][eft_FLT] = {eft_FLT, (void*)dpnp_trace_default_c<int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_TRACE][eft_LNG][eft_FLT] = {eft_FLT, (void*)dpnp_trace_default_c<int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_TRACE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_trace_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_TRACE][eft_DBL][eft_FLT] = {eft_FLT, (void*)dpnp_trace_default_c<double, float>};
    fmap[DPNPFuncName::DPNP_FN_TRACE][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_trace_default_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TRACE][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_trace_default_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TRACE][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_trace_default_c<float, double>};
    fmap[DPNPFuncName::DPNP_FN_TRACE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_trace_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_TRACE_EXT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_trace_ext_c<int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_TRACE_EXT][eft_LNG][eft_INT] = {eft_INT, (void*)dpnp_trace_ext_c<int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_TRACE_EXT][eft_FLT][eft_INT] = {eft_INT, (void*)dpnp_trace_ext_c<float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_TRACE_EXT][eft_DBL][eft_INT] = {eft_INT, (void*)dpnp_trace_ext_c<double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_TRACE_EXT][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_trace_ext_c<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_TRACE_EXT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_trace_ext_c<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_TRACE_EXT][eft_FLT][eft_LNG] = {eft_LNG, (void*)dpnp_trace_ext_c<float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_TRACE_EXT][eft_DBL][eft_LNG] = {eft_LNG, (void*)dpnp_trace_ext_c<double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_TRACE_EXT][eft_INT][eft_FLT] = {eft_FLT, (void*)dpnp_trace_ext_c<int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_TRACE_EXT][eft_LNG][eft_FLT] = {eft_FLT, (void*)dpnp_trace_ext_c<int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_TRACE_EXT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_trace_ext_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_TRACE_EXT][eft_DBL][eft_FLT] = {eft_FLT, (void*)dpnp_trace_ext_c<double, float>};
    fmap[DPNPFuncName::DPNP_FN_TRACE_EXT][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_trace_ext_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TRACE_EXT][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_trace_ext_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TRACE_EXT][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_trace_ext_c<float, double>};
    fmap[DPNPFuncName::DPNP_FN_TRACE_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_trace_ext_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_TRI][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_tri_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_TRI][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_tri_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_TRI][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_tri_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_TRI][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_tri_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_TRI_EXT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_tri_ext_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_TRI_EXT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_tri_ext_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_TRI_EXT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_tri_ext_c<float>};
    fmap[DPNPFuncName::DPNP_FN_TRI_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_tri_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_TRIL][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_tril_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_TRIL][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_tril_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_TRIL][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_tril_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_TRIL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_tril_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_TRIL_EXT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_tril_ext_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_TRIL_EXT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_tril_ext_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_TRIL_EXT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_tril_ext_c<float>};
    fmap[DPNPFuncName::DPNP_FN_TRIL_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_tril_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_TRIU][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_triu_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_TRIU][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_triu_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_TRIU][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_triu_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_TRIU][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_triu_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_TRIU_EXT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_triu_ext_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_TRIU_EXT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_triu_ext_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_TRIU_EXT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_triu_ext_c<float>};
    fmap[DPNPFuncName::DPNP_FN_TRIU_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_triu_ext_c<double>};

    fmap[DPNPFuncName::DPNP_FN_ZEROS][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_zeros_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ZEROS][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_zeros_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ZEROS][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_zeros_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_ZEROS][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_zeros_default_c<double>};
    fmap[DPNPFuncName::DPNP_FN_ZEROS][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_zeros_default_c<bool>};
    fmap[DPNPFuncName::DPNP_FN_ZEROS][eft_C128][eft_C128] = {eft_C128,
                                                             (void*)dpnp_zeros_default_c<std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_ZEROS_LIKE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_zeros_like_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ZEROS_LIKE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_zeros_like_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ZEROS_LIKE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_zeros_like_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_ZEROS_LIKE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_zeros_like_default_c<double>};
    fmap[DPNPFuncName::DPNP_FN_ZEROS_LIKE][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_zeros_like_default_c<bool>};
    fmap[DPNPFuncName::DPNP_FN_ZEROS_LIKE][eft_C128][eft_C128] = {
        eft_C128, (void*)dpnp_zeros_like_default_c<std::complex<double>>};

    return;
}
