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
#include "dpnp_utils.hpp"
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"
#include <dpnp_iface.hpp>

namespace mkl_blas = oneapi::mkl::blas::row_major;
namespace mkl_stats = oneapi::mkl::stats;

template <typename _KernelNameSpecialization1,
          typename _KernelNameSpecialization2,
          typename _KernelNameSpecialization3>
class dpnp_correlate_c_kernel;

template <typename _DataType_output,
          typename _DataType_input1,
          typename _DataType_input2>
DPCTLSyclEventRef dpnp_correlate_c(DPCTLSyclQueueRef q_ref,
                                   void *result_out,
                                   const void *input1_in,
                                   const size_t input1_size,
                                   const shape_elem_type *input1_shape,
                                   const size_t input1_shape_ndim,
                                   const void *input2_in,
                                   const size_t input2_size,
                                   const shape_elem_type *input2_shape,
                                   const size_t input2_shape_ndim,
                                   const size_t *where,
                                   const DPCTLEventVectorRef dep_event_vec_ref)
{
    (void)where;

    shape_elem_type dummy[] = {1};
    return dpnp_dot_c<_DataType_output, _DataType_input1, _DataType_input2>(
        q_ref, result_out,
        42,   // dummy result_size
        42,   // dummy result_ndim
        NULL, // dummy result_shape
        NULL, // dummy result_strides
        input1_in, input1_size, input1_shape_ndim, input1_shape,
        dummy, // dummy input1_strides
        input2_in, input2_size, input2_shape_ndim, input2_shape,
        dummy, // dummy input2_strides
        dep_event_vec_ref);
}

template <typename _DataType_output,
          typename _DataType_input1,
          typename _DataType_input2>
void dpnp_correlate_c(void *result_out,
                      const void *input1_in,
                      const size_t input1_size,
                      const shape_elem_type *input1_shape,
                      const size_t input1_shape_ndim,
                      const void *input2_in,
                      const size_t input2_size,
                      const shape_elem_type *input2_shape,
                      const size_t input2_shape_ndim,
                      const size_t *where)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref =
        dpnp_correlate_c<_DataType_output, _DataType_input1, _DataType_input2>(
            q_ref, result_out, input1_in, input1_size, input1_shape,
            input1_shape_ndim, input2_in, input2_size, input2_shape,
            input2_shape_ndim, where, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType_output,
          typename _DataType_input1,
          typename _DataType_input2>
void (*dpnp_correlate_default_c)(void *,
                                 const void *,
                                 const size_t,
                                 const shape_elem_type *,
                                 const size_t,
                                 const void *,
                                 const size_t,
                                 const shape_elem_type *,
                                 const size_t,
                                 const size_t *) =
    dpnp_correlate_c<_DataType_output, _DataType_input1, _DataType_input2>;

template <typename _DataType_output,
          typename _DataType_input1,
          typename _DataType_input2>
DPCTLSyclEventRef (*dpnp_correlate_ext_c)(DPCTLSyclQueueRef,
                                          void *,
                                          const void *,
                                          const size_t,
                                          const shape_elem_type *,
                                          const size_t,
                                          const void *,
                                          const size_t,
                                          const shape_elem_type *,
                                          const size_t,
                                          const size_t *,
                                          const DPCTLEventVectorRef) =
    dpnp_correlate_c<_DataType_output, _DataType_input1, _DataType_input2>;

void func_map_init_statistics(func_map_t &fmap)
{
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_correlate_default_c<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_INT][eft_LNG] = {
        eft_LNG, (void *)dpnp_correlate_default_c<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_INT][eft_FLT] = {
        eft_DBL, (void *)dpnp_correlate_default_c<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_correlate_default_c<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_LNG][eft_INT] = {
        eft_LNG, (void *)dpnp_correlate_default_c<int64_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_correlate_default_c<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_LNG][eft_FLT] = {
        eft_DBL, (void *)dpnp_correlate_default_c<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_correlate_default_c<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_FLT][eft_INT] = {
        eft_DBL, (void *)dpnp_correlate_default_c<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_FLT][eft_LNG] = {
        eft_DBL, (void *)dpnp_correlate_default_c<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_correlate_default_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_correlate_default_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_DBL][eft_INT] = {
        eft_DBL, (void *)dpnp_correlate_default_c<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_DBL][eft_LNG] = {
        eft_DBL, (void *)dpnp_correlate_default_c<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_DBL][eft_FLT] = {
        eft_DBL, (void *)dpnp_correlate_default_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_correlate_default_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_CORRELATE_EXT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_correlate_ext_c<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE_EXT][eft_INT][eft_LNG] = {
        eft_LNG, (void *)dpnp_correlate_ext_c<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE_EXT][eft_INT][eft_FLT] = {
        eft_DBL, (void *)dpnp_correlate_ext_c<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE_EXT][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_correlate_ext_c<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE_EXT][eft_LNG][eft_INT] = {
        eft_LNG, (void *)dpnp_correlate_ext_c<int64_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE_EXT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_correlate_ext_c<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE_EXT][eft_LNG][eft_FLT] = {
        eft_DBL, (void *)dpnp_correlate_ext_c<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE_EXT][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_correlate_ext_c<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE_EXT][eft_FLT][eft_INT] = {
        eft_DBL, (void *)dpnp_correlate_ext_c<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE_EXT][eft_FLT][eft_LNG] = {
        eft_DBL, (void *)dpnp_correlate_ext_c<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_correlate_ext_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE_EXT][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_correlate_ext_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE_EXT][eft_DBL][eft_INT] = {
        eft_DBL, (void *)dpnp_correlate_ext_c<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE_EXT][eft_DBL][eft_LNG] = {
        eft_DBL, (void *)dpnp_correlate_ext_c<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE_EXT][eft_DBL][eft_FLT] = {
        eft_DBL, (void *)dpnp_correlate_ext_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_correlate_ext_c<double, double, double>};

    return;
}
