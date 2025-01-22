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
#include <list>
#include <vector>

#include "dpnp_fptr.hpp"
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"
#include <dpnp_iface.hpp>

template <typename _DataType1, typename _DataType2>
class dpnp_choose_c_kernel;

template <typename _DataType1, typename _DataType2>
DPCTLSyclEventRef dpnp_choose_c(DPCTLSyclQueueRef q_ref,
                                void *result1,
                                void *array1_in,
                                void **choices1,
                                size_t size,
                                size_t choices_size,
                                size_t choice_size,
                                const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if ((array1_in == nullptr) || (result1 == nullptr) || (choices1 == nullptr))
    {
        return event_ref;
    }
    if (!size || !choices_size || !choice_size) {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    DPNPC_ptr_adapter<_DataType1> input1_ptr(q_ref, array1_in, size);
    _DataType1 *array_in = input1_ptr.get_ptr();

    // choices1 is a list of pointers to device memory,
    // which is allocating on the host, so memcpy to device memory is required
    DPNPC_ptr_adapter<_DataType2 *> choices_ptr(q_ref, choices1, choices_size,
                                                true);
    _DataType2 **choices = choices_ptr.get_ptr();

    for (size_t i = 0; i < choices_size; ++i) {
        DPNPC_ptr_adapter<_DataType2> choice_ptr(q_ref, choices[i],
                                                 choice_size);
        choices[i] = choice_ptr.get_ptr();
    }

    DPNPC_ptr_adapter<_DataType2> result1_ptr(q_ref, result1, size, false,
                                              true);
    _DataType2 *result = result1_ptr.get_ptr();

    sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
        const size_t idx = global_id[0];
        result[idx] = choices[array_in[idx]][idx];
    };

    auto kernel_func = [&](sycl::handler &cgh) {
        cgh.parallel_for<class dpnp_choose_c_kernel<_DataType1, _DataType2>>(
            gws, kernel_parallel_for_func);
    };

    sycl::event event = q.submit(kernel_func);
    choices_ptr.depends_on(event);

    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType1, typename _DataType2>
void dpnp_choose_c(void *result1,
                   void *array1_in,
                   void **choices1,
                   size_t size,
                   size_t choices_size,
                   size_t choice_size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_choose_c<_DataType1, _DataType2>(
        q_ref, result1, array1_in, choices1, size, choices_size, choice_size,
        dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType1, typename _DataType2>
void (*dpnp_choose_default_c)(void *, void *, void **, size_t, size_t, size_t) =
    dpnp_choose_c<_DataType1, _DataType2>;

template <typename _DataType1, typename _DataType2>
DPCTLSyclEventRef (*dpnp_choose_ext_c)(DPCTLSyclQueueRef,
                                       void *,
                                       void *,
                                       void **,
                                       size_t,
                                       size_t,
                                       size_t,
                                       const DPCTLEventVectorRef) =
    dpnp_choose_c<_DataType1, _DataType2>;

void func_map_init_indexing_func(func_map_t &fmap)
{
    fmap[DPNPFuncName::DPNP_FN_CHOOSE][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_choose_default_c<int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE][eft_INT][eft_LNG] = {
        eft_LNG, (void *)dpnp_choose_default_c<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE][eft_INT][eft_FLT] = {
        eft_FLT, (void *)dpnp_choose_default_c<int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_choose_default_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE][eft_LNG][eft_INT] = {
        eft_INT, (void *)dpnp_choose_default_c<int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_choose_default_c<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE][eft_LNG][eft_FLT] = {
        eft_FLT, (void *)dpnp_choose_default_c<int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_choose_default_c<int64_t, double>};

    fmap[DPNPFuncName::DPNP_FN_CHOOSE_EXT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_choose_ext_c<int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE_EXT][eft_INT][eft_LNG] = {
        eft_LNG, (void *)dpnp_choose_ext_c<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE_EXT][eft_INT][eft_FLT] = {
        eft_FLT, (void *)dpnp_choose_ext_c<int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE_EXT][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_choose_ext_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE_EXT][eft_LNG][eft_INT] = {
        eft_INT, (void *)dpnp_choose_ext_c<int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE_EXT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_choose_ext_c<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE_EXT][eft_LNG][eft_FLT] = {
        eft_FLT, (void *)dpnp_choose_ext_c<int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_CHOOSE_EXT][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_choose_ext_c<int64_t, double>};
    return;
}
