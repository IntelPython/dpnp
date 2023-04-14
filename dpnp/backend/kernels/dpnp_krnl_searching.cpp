//*****************************************************************************
// Copyright (c) 2016-2023, Intel Corporation
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

#include <dpnp_iface.hpp>
#include "dpnp_fptr.hpp"
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"

template <typename _DataType, typename _idx_DataType>
class dpnp_argmax_c_kernel;

template <typename _DataType, typename _idx_DataType>
DPCTLSyclEventRef dpnp_argmax_c(DPCTLSyclQueueRef q_ref,
                                void* array1_in,
                                void* result1,
                                size_t size,
                                const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;
    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    DPNPC_ptr_adapter<_DataType> input1_ptr(q_ref, array1_in, size);
    _DataType* array_1 = input1_ptr.get_ptr();
    _idx_DataType* result = reinterpret_cast<_idx_DataType*>(result1);

    auto policy =
        oneapi::dpl::execution::make_device_policy<class dpnp_argmax_c_kernel<_DataType, _idx_DataType>>(q);

    _DataType* res = std::max_element(policy, array_1, array_1 + size);
    policy.queue().wait();

    _idx_DataType result_val = std::distance(array_1, res);
    q.memcpy(result, &result_val, sizeof(_idx_DataType)).wait(); // result[0] = std::distance(array_1, res);

    return event_ref;
}

template <typename _DataType, typename _idx_DataType>
void dpnp_argmax_c(void* array1_in, void* result1, size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_argmax_c<_DataType, _idx_DataType>(q_ref,
                                                                          array1_in,
                                                                          result1,
                                                                          size,
                                                                          dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType, typename _idx_DataType>
void (*dpnp_argmax_default_c)(void*, void*, size_t) = dpnp_argmax_c<_DataType, _idx_DataType>;

template <typename _DataType, typename _idx_DataType>
DPCTLSyclEventRef (*dpnp_argmax_ext_c)(DPCTLSyclQueueRef,
                                       void*,
                                       void*,
                                       size_t,
                                       const DPCTLEventVectorRef) = dpnp_argmax_c<_DataType, _idx_DataType>;

template <typename _DataType, typename _idx_DataType>
class dpnp_argmin_c_kernel;

template <typename _DataType, typename _idx_DataType>
DPCTLSyclEventRef dpnp_argmin_c(DPCTLSyclQueueRef q_ref,
                                void* array1_in,
                                void* result1,
                                size_t size,
                                const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;
    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));
    DPNPC_ptr_adapter<_DataType> input1_ptr(q_ref, array1_in, size);
    _DataType* array_1 = input1_ptr.get_ptr();
    _idx_DataType* result = reinterpret_cast<_idx_DataType*>(result1);

    auto policy =
        oneapi::dpl::execution::make_device_policy<class dpnp_argmin_c_kernel<_DataType, _idx_DataType>>(q);

    _DataType* res = std::min_element(policy, array_1, array_1 + size);
    policy.queue().wait();

    _idx_DataType result_val = std::distance(array_1, res);
    q.memcpy(result, &result_val, sizeof(_idx_DataType)).wait(); // result[0] = std::distance(array_1, res);

    return event_ref;
}

template <typename _DataType, typename _idx_DataType>
void dpnp_argmin_c(void* array1_in, void* result1, size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_argmin_c<_DataType, _idx_DataType>(q_ref,
                                                                          array1_in,
                                                                          result1,
                                                                          size,
                                                                          dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType, typename _idx_DataType>
void (*dpnp_argmin_default_c)(void*, void*, size_t) = dpnp_argmin_c<_DataType, _idx_DataType>;

template <typename _DataType, typename _idx_DataType>
DPCTLSyclEventRef (*dpnp_argmin_ext_c)(DPCTLSyclQueueRef,
                                       void*,
                                       void*,
                                       size_t,
                                       const DPCTLEventVectorRef) = dpnp_argmin_c<_DataType, _idx_DataType>;

void func_map_init_searching(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_argmax_default_c<int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_argmax_default_c<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_LNG][eft_INT] = {eft_INT, (void*)dpnp_argmax_default_c<int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_argmax_default_c<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_FLT][eft_INT] = {eft_INT, (void*)dpnp_argmax_default_c<float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_FLT][eft_LNG] = {eft_LNG, (void*)dpnp_argmax_default_c<float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_DBL][eft_INT] = {eft_INT, (void*)dpnp_argmax_default_c<double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_DBL][eft_LNG] = {eft_LNG, (void*)dpnp_argmax_default_c<double, int64_t>};

    fmap[DPNPFuncName::DPNP_FN_ARGMAX_EXT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_argmax_ext_c<int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX_EXT][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_argmax_ext_c<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX_EXT][eft_LNG][eft_INT] = {eft_INT, (void*)dpnp_argmax_ext_c<int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX_EXT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_argmax_ext_c<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX_EXT][eft_FLT][eft_INT] = {eft_INT, (void*)dpnp_argmax_ext_c<float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX_EXT][eft_FLT][eft_LNG] = {eft_LNG, (void*)dpnp_argmax_ext_c<float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX_EXT][eft_DBL][eft_INT] = {eft_INT, (void*)dpnp_argmax_ext_c<double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX_EXT][eft_DBL][eft_LNG] = {eft_LNG, (void*)dpnp_argmax_ext_c<double, int64_t>};

    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_argmin_default_c<int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_argmin_default_c<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_LNG][eft_INT] = {eft_INT, (void*)dpnp_argmin_default_c<int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_argmin_default_c<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_FLT][eft_INT] = {eft_INT, (void*)dpnp_argmin_default_c<float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_FLT][eft_LNG] = {eft_LNG, (void*)dpnp_argmin_default_c<float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_DBL][eft_INT] = {eft_INT, (void*)dpnp_argmin_default_c<double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_DBL][eft_LNG] = {eft_LNG, (void*)dpnp_argmin_default_c<double, int64_t>};

    fmap[DPNPFuncName::DPNP_FN_ARGMIN_EXT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_argmin_ext_c<int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN_EXT][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_argmin_ext_c<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN_EXT][eft_LNG][eft_INT] = {eft_INT, (void*)dpnp_argmin_ext_c<int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN_EXT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_argmin_ext_c<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN_EXT][eft_FLT][eft_INT] = {eft_INT, (void*)dpnp_argmin_ext_c<float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN_EXT][eft_FLT][eft_LNG] = {eft_LNG, (void*)dpnp_argmin_ext_c<float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN_EXT][eft_DBL][eft_INT] = {eft_INT, (void*)dpnp_argmin_ext_c<double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN_EXT][eft_DBL][eft_LNG] = {eft_LNG, (void*)dpnp_argmin_ext_c<double, int64_t>};

    return;
}
