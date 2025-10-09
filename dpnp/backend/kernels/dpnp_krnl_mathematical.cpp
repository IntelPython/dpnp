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

#include <cmath>
#include <iostream>
#include <vector>

#include <dpnp_iface.hpp>

#include "dpnp_fptr.hpp"
#include "dpnp_utils.hpp"
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"

// dpctl tensor headers
#include "kernels/alignment.hpp"

using dpctl::tensor::kernels::alignment_utils::is_aligned;
using dpctl::tensor::kernels::alignment_utils::required_alignment;

static_assert(__SYCL_COMPILER_VERSION >= __SYCL_COMPILER_VECTOR_ABS_CHANGED,
              "SYCL DPC++ compiler does not meet minimum version requirement");

template <typename _KernelNameSpecialization1,
          typename _KernelNameSpecialization2>
class dpnp_modf_c_kernel;

template <typename _DataType_input, typename _DataType_output>
DPCTLSyclEventRef dpnp_modf_c(DPCTLSyclQueueRef q_ref,
                              void *array1_in,
                              void *result1_out,
                              void *result2_out,
                              size_t size,
                              const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));
    sycl::event event;

    DPNPC_ptr_adapter<_DataType_input> input1_ptr(q_ref, array1_in, size);
    _DataType_input *array1 = input1_ptr.get_ptr();
    _DataType_output *result1 =
        reinterpret_cast<_DataType_output *>(result1_out);
    _DataType_output *result2 =
        reinterpret_cast<_DataType_output *>(result2_out);

    if constexpr (std::is_same<_DataType_input, double>::value ||
                  std::is_same<_DataType_input, float>::value)
    {
        event = oneapi::mkl::vm::modf(q, size, array1, result2, result1);
    }
    else {
        sycl::range<1> gws(size);
        auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
            size_t i = global_id[0]; /*for (size_t i = 0; i < size; ++i)*/
            {
                double input_elem1 = static_cast<double>(array1[i]);
                auto res_multi_ptr = sycl::address_space_cast<
                    sycl::access::address_space::global_space,
                    sycl::access::decorated::yes>(&result1[i]);

                result2[i] = sycl::modf(input_elem1, res_multi_ptr);
            }
        };

        auto kernel_func = [&](sycl::handler &cgh) {
            cgh.parallel_for<
                class dpnp_modf_c_kernel<_DataType_input, _DataType_output>>(
                gws, kernel_parallel_for_func);
        };

        event = q.submit(kernel_func);
    }

    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType_input, typename _DataType_output>
void dpnp_modf_c(void *array1_in,
                 void *result1_out,
                 void *result2_out,
                 size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref =
        dpnp_modf_c<_DataType_input, _DataType_output>(q_ref, array1_in,
                                                       result1_out, result2_out,
                                                       size, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType_input, typename _DataType_output>
void (*dpnp_modf_default_c)(void *, void *, void *, size_t) =
    dpnp_modf_c<_DataType_input, _DataType_output>;

template <typename _DataType_input, typename _DataType_output>
DPCTLSyclEventRef (*dpnp_modf_ext_c)(DPCTLSyclQueueRef,
                                     void *,
                                     void *,
                                     void *,
                                     size_t,
                                     const DPCTLEventVectorRef) =
    dpnp_modf_c<_DataType_input, _DataType_output>;

void func_map_init_mathematical(func_map_t &fmap)
{

    fmap[DPNPFuncName::DPNP_FN_MODF][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_modf_default_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MODF][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_modf_default_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MODF][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_modf_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_MODF][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_modf_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_MODF_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_modf_ext_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MODF_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_modf_ext_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MODF_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_modf_ext_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_MODF_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_modf_ext_c<double, double>};

    return;
}
