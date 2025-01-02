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
