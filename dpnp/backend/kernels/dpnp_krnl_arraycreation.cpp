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
#include "dpnp_iface.hpp"
#include "dpnp_utils.hpp"
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"

template <typename _DataType>
class dpnp_ones_c_kernel;

template <typename _DataType>
DPCTLSyclEventRef dpnp_ones_c(DPCTLSyclQueueRef q_ref,
                              void *result,
                              size_t size,
                              const DPCTLEventVectorRef dep_event_vec_ref)
{
    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    _DataType *fill_value = reinterpret_cast<_DataType *>(
        sycl::malloc_shared(sizeof(_DataType), q));
    fill_value[0] = 1;

    DPCTLSyclEventRef event_ref = dpnp_initval_c<_DataType>(
        q_ref, result, fill_value, size, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);

    sycl::free(fill_value, q);

    return nullptr;
}

template <typename _DataType>
void dpnp_ones_c(void *result, size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref =
        dpnp_ones_c<_DataType>(q_ref, result, size, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_ones_default_c)(void *, size_t) = dpnp_ones_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_ones_like_c(DPCTLSyclQueueRef q_ref,
                                   void *result,
                                   size_t size,
                                   const DPCTLEventVectorRef dep_event_vec_ref)
{
    return dpnp_ones_c<_DataType>(q_ref, result, size, dep_event_vec_ref);
}

template <typename _DataType>
void dpnp_ones_like_c(void *result, size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref =
        dpnp_ones_like_c<_DataType>(q_ref, result, size, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_ones_like_default_c)(void *, size_t) = dpnp_ones_like_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_zeros_c(DPCTLSyclQueueRef q_ref,
                               void *result,
                               size_t size,
                               const DPCTLEventVectorRef dep_event_vec_ref)
{
    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    _DataType *fill_value = reinterpret_cast<_DataType *>(
        sycl::malloc_shared(sizeof(_DataType), q));
    fill_value[0] = 0;

    DPCTLSyclEventRef event_ref = dpnp_initval_c<_DataType>(
        q_ref, result, fill_value, size, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);

    sycl::free(fill_value, q);

    return nullptr;
}

template <typename _DataType>
void dpnp_zeros_c(void *result, size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref =
        dpnp_zeros_c<_DataType>(q_ref, result, size, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_zeros_default_c)(void *, size_t) = dpnp_zeros_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef dpnp_zeros_like_c(DPCTLSyclQueueRef q_ref,
                                    void *result,
                                    size_t size,
                                    const DPCTLEventVectorRef dep_event_vec_ref)
{
    return dpnp_zeros_c<_DataType>(q_ref, result, size, dep_event_vec_ref);
}

template <typename _DataType>
void dpnp_zeros_like_c(void *result, size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref =
        dpnp_zeros_like_c<_DataType>(q_ref, result, size, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_zeros_like_default_c)(void *,
                                  size_t) = dpnp_zeros_like_c<_DataType>;

void func_map_init_arraycreation(func_map_t &fmap)
{
    // Used in dpnp_rng_geometric_c
    fmap[DPNPFuncName::DPNP_FN_ONES][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_ones_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ONES][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_ones_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ONES][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_ones_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_ONES][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_ones_default_c<double>};
    fmap[DPNPFuncName::DPNP_FN_ONES][eft_BLN][eft_BLN] = {
        eft_BLN, (void *)dpnp_ones_default_c<bool>};
    fmap[DPNPFuncName::DPNP_FN_ONES][eft_C128][eft_C128] = {
        eft_C128, (void *)dpnp_ones_default_c<std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_ONES_LIKE][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_ones_like_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ONES_LIKE][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_ones_like_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ONES_LIKE][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_ones_like_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_ONES_LIKE][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_ones_like_default_c<double>};
    fmap[DPNPFuncName::DPNP_FN_ONES_LIKE][eft_BLN][eft_BLN] = {
        eft_BLN, (void *)dpnp_ones_like_default_c<bool>};
    fmap[DPNPFuncName::DPNP_FN_ONES_LIKE][eft_C128][eft_C128] = {
        eft_C128, (void *)dpnp_ones_like_default_c<std::complex<double>>};

    // Used in dpnp_rng_binomial_c, dpnp_rng_gamma_c, dpnp_rng_hypergeometric_c
    //         dpnp_rng_laplace_c, dpnp_rng_multinomial_c, dpnp_rng_weibull_c
    fmap[DPNPFuncName::DPNP_FN_ZEROS][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_zeros_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ZEROS][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_zeros_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ZEROS][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_zeros_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_ZEROS][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_zeros_default_c<double>};
    fmap[DPNPFuncName::DPNP_FN_ZEROS][eft_BLN][eft_BLN] = {
        eft_BLN, (void *)dpnp_zeros_default_c<bool>};
    fmap[DPNPFuncName::DPNP_FN_ZEROS][eft_C128][eft_C128] = {
        eft_C128, (void *)dpnp_zeros_default_c<std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_ZEROS_LIKE][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_zeros_like_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ZEROS_LIKE][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_zeros_like_default_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ZEROS_LIKE][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_zeros_like_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_ZEROS_LIKE][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_zeros_like_default_c<double>};
    fmap[DPNPFuncName::DPNP_FN_ZEROS_LIKE][eft_BLN][eft_BLN] = {
        eft_BLN, (void *)dpnp_zeros_like_default_c<bool>};
    fmap[DPNPFuncName::DPNP_FN_ZEROS_LIKE][eft_C128][eft_C128] = {
        eft_C128, (void *)dpnp_zeros_like_default_c<std::complex<double>>};

    return;
}
