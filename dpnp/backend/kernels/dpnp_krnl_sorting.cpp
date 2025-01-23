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

#include <iostream>

#include "dpnp_fptr.hpp"
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"
#include <dpnp_iface.hpp>

template <typename _DataType>
class dpnp_partition_c_kernel;

template <typename _DataType>
DPCTLSyclEventRef dpnp_partition_c(DPCTLSyclQueueRef q_ref,
                                   void *array1_in,
                                   void *array2_in,
                                   void *result1,
                                   const size_t kth,
                                   const shape_elem_type *shape_,
                                   const size_t ndim,
                                   const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if ((array1_in == nullptr) || (array2_in == nullptr) ||
        (result1 == nullptr)) {
        return event_ref;
    }

    if (ndim < 1) {
        return event_ref;
    }

    const size_t size = std::accumulate(shape_, shape_ + ndim, 1,
                                        std::multiplies<shape_elem_type>());
    size_t size_ = size / shape_[ndim - 1];

    if (size_ == 0) {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    if (ndim == 1) // 1d array with C-contiguous data
    {
        _DataType *arr = static_cast<_DataType *>(array1_in);
        _DataType *result = static_cast<_DataType *>(result1);

        auto policy = oneapi::dpl::execution::make_device_policy<
            dpnp_partition_c_kernel<_DataType>>(q);

        // fill the result array with data from input one
        q.memcpy(result, arr, size * sizeof(_DataType)).wait();

        // make a partial sorting such that:
        // 1. result[0 <= i < kth]    <= result[kth]
        // 2. result[kth <= i < size] >= result[kth]
        // event-blocking call, no need for wait()
        std::nth_element(policy, result, result + kth, result + size,
                         dpnp_less_comp());
        return event_ref;
    }

    DPNPC_ptr_adapter<_DataType> input1_ptr(q_ref, array1_in, size, true);
    DPNPC_ptr_adapter<_DataType> input2_ptr(q_ref, array2_in, size, true);
    DPNPC_ptr_adapter<_DataType> result1_ptr(q_ref, result1, size, true, true);
    _DataType *arr = input1_ptr.get_ptr();
    _DataType *arr2 = input2_ptr.get_ptr();
    _DataType *result = result1_ptr.get_ptr();

    auto arr_to_result_event = q.memcpy(result, arr, size * sizeof(_DataType));
    arr_to_result_event.wait();

    _DataType *matrix = new _DataType[shape_[ndim - 1]];

    for (size_t i = 0; i < size_; ++i) {
        size_t ind_begin = i * shape_[ndim - 1];
        size_t ind_end = (i + 1) * shape_[ndim - 1] - 1;

        for (size_t j = ind_begin; j < ind_end + 1; ++j) {
            size_t ind = j - ind_begin;
            matrix[ind] = arr2[j];
        }
        std::partial_sort(matrix, matrix + shape_[ndim - 1],
                          matrix + shape_[ndim - 1], dpnp_less_comp());
        for (size_t j = ind_begin; j < ind_end + 1; ++j) {
            size_t ind = j - ind_begin;
            arr2[j] = matrix[ind];
        }
    }

    shape_elem_type *shape = reinterpret_cast<shape_elem_type *>(
        sycl::malloc_shared(ndim * sizeof(shape_elem_type), q));
    auto memcpy_event = q.memcpy(shape, shape_, ndim * sizeof(shape_elem_type));

    memcpy_event.wait();

    sycl::range<2> gws(size_, kth + 1);
    auto kernel_parallel_for_func = [=](sycl::id<2> global_id) {
        size_t j = global_id[0];
        size_t k = global_id[1];

        _DataType val = arr2[j * shape[ndim - 1] + k];

        for (size_t i = 0; i < static_cast<size_t>(shape[ndim - 1]); ++i) {
            if (result[j * shape[ndim - 1] + i] == val) {
                _DataType change_val1 = result[j * shape[ndim - 1] + i];
                _DataType change_val2 = result[j * shape[ndim - 1] + k];
                result[j * shape[ndim - 1] + k] = change_val1;
                result[j * shape[ndim - 1] + i] = change_val2;
            }
        }
    };

    auto kernel_func = [&](sycl::handler &cgh) {
        cgh.depends_on({memcpy_event});
        cgh.parallel_for<class dpnp_partition_c_kernel<_DataType>>(
            gws, kernel_parallel_for_func);
    };

    auto event = q.submit(kernel_func);

    event.wait();

    delete[] matrix;
    sycl::free(shape, q);

    return event_ref;
}

template <typename _DataType>
void dpnp_partition_c(void *array1_in,
                      void *array2_in,
                      void *result1,
                      const size_t kth,
                      const shape_elem_type *shape_,
                      const size_t ndim)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref =
        dpnp_partition_c<_DataType>(q_ref, array1_in, array2_in, result1, kth,
                                    shape_, ndim, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_partition_default_c)(void *,
                                 void *,
                                 void *,
                                 const size_t,
                                 const shape_elem_type *,
                                 const size_t) = dpnp_partition_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_partition_ext_c)(DPCTLSyclQueueRef,
                                          void *,
                                          void *,
                                          void *,
                                          const size_t,
                                          const shape_elem_type *,
                                          const size_t,
                                          const DPCTLEventVectorRef) =
    dpnp_partition_c<_DataType>;

void func_map_init_sorting(func_map_t &fmap)
{
    fmap[DPNPFuncName::DPNP_FN_PARTITION][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_partition_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_PARTITION][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_partition_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_PARTITION][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_partition_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_PARTITION][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_partition_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_PARTITION_EXT][eft_BLN][eft_BLN] = {
        eft_BLN, (void *)dpnp_partition_ext_c<bool>};
    fmap[DPNPFuncName::DPNP_FN_PARTITION_EXT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_partition_ext_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_PARTITION_EXT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_partition_ext_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_PARTITION_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_partition_ext_c<float>};
    fmap[DPNPFuncName::DPNP_FN_PARTITION_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_partition_ext_c<double>};
    fmap[DPNPFuncName::DPNP_FN_PARTITION_EXT][eft_C64][eft_C64] = {
        eft_C64, (void *)dpnp_partition_ext_c<std::complex<float>>};
    fmap[DPNPFuncName::DPNP_FN_PARTITION_EXT][eft_C128][eft_C128] = {
        eft_C128, (void *)dpnp_partition_ext_c<std::complex<double>>};

    return;
}
