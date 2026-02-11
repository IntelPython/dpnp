//*****************************************************************************
// Copyright (c) 2016, Intel Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// - Neither the name of the copyright holder nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
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

    _DataType *arr = static_cast<_DataType *>(array1_in);
    _DataType *result = static_cast<_DataType *>(result1);

    auto policy = oneapi::dpl::execution::make_device_policy<
        dpnp_partition_c_kernel<_DataType>>(q);

    // fill the result array with data from input one
    q.memcpy(result, arr, size * sizeof(_DataType)).wait();

    for (size_t i = 0; i < size_; i++) {
        _DataType *bufptr = result + i * shape_[0];

        // for every slice it makes a partial sorting such that:
        // 1. result[0 <= i < kth]    <= result[kth]
        // 2. result[kth <= i < size] >= result[kth]
        // event-blocking call, no need for wait()
        std::nth_element(policy, bufptr, bufptr + kth, bufptr + size,
                         dpnp_less_comp());
    }
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
