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

#include <dpnp_iface.hpp>
#include "dpnp_fptr.hpp"
#include "queue_sycl.hpp"

template <typename _DataType, typename _IndecesType>
class dpnp_take_c_kernel;

template <typename _DataType, typename _IndecesType>
void dpnp_take_c(void* array1_in, void* indices1, void* result1, size_t size)
{
    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);
    _IndecesType* indices = reinterpret_cast<_IndecesType*>(indices1);

    cl::sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
        const size_t idx = global_id[0];
        result[idx] = array_1[indices[idx]];
    };

    auto kernel_func = [&](cl::sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_take_c_kernel<_DataType, _IndecesType>>(gws, kernel_parallel_for_func);
    };

    cl::sycl::event event = DPNP_QUEUE.submit(kernel_func);

    event.wait();

    return;
}

void func_map_init_indexing_func(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_BOOL][eft_BOOL] = {eft_BOOL, (void*)dpnp_take_c<bool, long>};
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_take_c<int, long>};
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_take_c<long, long>};
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_take_c<float, long>};
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_take_c<double, long>};
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_C128][eft_C128] = {eft_C128, (void*)dpnp_take_c<std::complex<double>, long>};

    return;
}
