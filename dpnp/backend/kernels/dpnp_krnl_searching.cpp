//*****************************************************************************
// Copyright (c) 2016-2020, Intel Corporation
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
void dpnp_argmax_c(void* array1_in, void* result1, size_t size)
{
    DPNPC_ptr_adapter<_DataType> input1_ptr(array1_in, size);
    _DataType* array_1 = input1_ptr.get_ptr();
    _idx_DataType* result = reinterpret_cast<_idx_DataType*>(result1);

    auto policy =
        oneapi::dpl::execution::make_device_policy<class dpnp_argmax_c_kernel<_DataType, _idx_DataType>>(DPNP_QUEUE);

    _DataType* res = std::max_element(policy, array_1, array_1 + size);
    policy.queue().wait();

    _idx_DataType result_val = std::distance(array_1, res);
    dpnp_memory_memcpy_c(result, &result_val, sizeof(_idx_DataType)); // result[0] = std::distance(array_1, res);

    return;
}

template <typename _DataType, typename _idx_DataType>
class dpnp_argmin_c_kernel;

template <typename _DataType, typename _idx_DataType>
void dpnp_argmin_c(void* array1_in, void* result1, size_t size)
{
    DPNPC_ptr_adapter<_DataType> input1_ptr(array1_in, size);
    _DataType* array_1 = input1_ptr.get_ptr();
    _idx_DataType* result = reinterpret_cast<_idx_DataType*>(result1);

    auto policy =
        oneapi::dpl::execution::make_device_policy<class dpnp_argmin_c_kernel<_DataType, _idx_DataType>>(DPNP_QUEUE);

    _DataType* res = std::min_element(policy, array_1, array_1 + size);
    policy.queue().wait();

    _idx_DataType result_val = std::distance(array_1, res);
    dpnp_memory_memcpy_c(result, &result_val, sizeof(_idx_DataType)); // result[0] = std::distance(array_1, res);

    return;
}

void func_map_init_searching(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_argmax_c<int, int>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_argmax_c<int, long>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_LNG][eft_INT] = {eft_INT, (void*)dpnp_argmax_c<long, int>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_argmax_c<long, long>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_FLT][eft_INT] = {eft_INT, (void*)dpnp_argmax_c<float, int>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_FLT][eft_LNG] = {eft_LNG, (void*)dpnp_argmax_c<float, long>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_DBL][eft_INT] = {eft_INT, (void*)dpnp_argmax_c<double, int>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_DBL][eft_LNG] = {eft_LNG, (void*)dpnp_argmax_c<double, long>};

    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_argmin_c<int, int>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_argmin_c<int, long>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_LNG][eft_INT] = {eft_INT, (void*)dpnp_argmin_c<long, int>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_argmin_c<long, long>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_FLT][eft_INT] = {eft_INT, (void*)dpnp_argmin_c<float, int>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_FLT][eft_LNG] = {eft_LNG, (void*)dpnp_argmin_c<float, long>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_DBL][eft_INT] = {eft_INT, (void*)dpnp_argmin_c<double, int>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_DBL][eft_LNG] = {eft_LNG, (void*)dpnp_argmin_c<double, long>};

    return;
}
