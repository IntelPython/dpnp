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

#include <cmath>
#include <iostream>

#include <backend_iface.hpp>
#include "backend_fptr.hpp"
#include "queue_sycl.hpp"

namespace mkl_stats = oneapi::mkl::stats;

template <typename _KernelNameSpecialization>
class custom_sum_c_kernel;

template <typename _DataType>
void custom_sum_c(void* array1_in, void* result1, size_t size)
{
    if (!size)
    {
        return;
    }

    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    if constexpr (std::is_same<_DataType, double>::value || std::is_same<_DataType, float>::value)
    {
        auto dataset = mkl_stats::make_dataset<mkl_stats::layout::row_major>(1, size, array_1);

        cl::sycl::event event = mkl_stats::raw_sum(DPNP_QUEUE, dataset, result);

        event.wait();
    }
    else
    {
        // cl::sycl::range<1> gws(size);
        auto policy = oneapi::dpl::execution::make_device_policy<custom_sum_c_kernel<_DataType>>(DPNP_QUEUE);

        // sycl::buffer<_DataType, 1> array_1_buf(array_1, gws);
        // auto it_begin = oneapi::dpl::begin(array_1_buf);
        // auto it_end = oneapi::dpl::end(array_1_buf);

        _DataType accumulator = 0;
        accumulator = std::reduce(policy, array_1, array_1 + size, _DataType(0), std::plus<_DataType>());

        policy.queue().wait();

        result[0] = accumulator;
    }

    return;
}

template <typename _KernelNameSpecialization>
class custom_prod_c_kernel;

template <typename _DataType>
void custom_prod_c(void* array1_in, void* result1, size_t size)
{
    if (!size)
    {
        return;
    }

    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    auto policy = oneapi::dpl::execution::make_device_policy<custom_prod_c_kernel<_DataType>>(DPNP_QUEUE);

    result[0] = std::reduce(policy, array_1, array_1 + size, _DataType(1), std::multiplies<_DataType>());

    policy.queue().wait();

    return;
}

void func_map_init_reduction(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_INT][eft_INT] = {eft_INT, (void*)custom_prod_c<int>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_LNG][eft_LNG] = {eft_LNG, (void*)custom_prod_c<long>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_prod_c<float>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_prod_c<double>};

    fmap[DPNPFuncName::DPNP_FN_SUM][eft_INT][eft_INT] = {eft_INT, (void*)custom_sum_c<int>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_LNG][eft_LNG] = {eft_LNG, (void*)custom_sum_c<long>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_FLT][eft_FLT] = {eft_FLT, (void*)custom_sum_c<float>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_DBL][eft_DBL] = {eft_DBL, (void*)custom_sum_c<double>};

    return;
}
