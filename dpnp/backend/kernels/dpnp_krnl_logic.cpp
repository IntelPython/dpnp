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

#include "dpnp_fptr.hpp"
#include "dpnp_iface.hpp"
#include "queue_sycl.hpp"

template <typename _DataType, typename _ResultType>
class dpnp_all_c_kernel;

template <typename _DataType, typename _ResultType>
void dpnp_all_c(const void* array1_in, void* result1, const size_t size)
{
    cl::sycl::event event;

    const _DataType* array_in = reinterpret_cast<const _DataType*>(array1_in);
    _ResultType* result = reinterpret_cast<_ResultType*>(result1);

    if (!array1_in || !result1)
    {
        return;
    }

    result[0] = true;

    if (!size)
    {
        return;
    }

    cl::sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
        size_t i = global_id[0];

        if (!array_in[i])
        {
            result[0] = false;
        }
    };

    auto kernel_func = [&](cl::sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_all_c_kernel<_DataType, _ResultType>>(gws, kernel_parallel_for_func);
    };

    event = DPNP_QUEUE.submit(kernel_func);

    event.wait();
}

template <typename _DataType1, typename _DataType2, typename _ResultType>
class dpnp_allclose_c_kernel;

template <typename _DataType1, typename _DataType2, typename _ResultType>
void dpnp_allclose_c(const void* array1_in, const void* array2_in, void* result1, const size_t size, double rtol, double atol)
{
    cl::sycl::event event;

    const _DataType1* array1 = reinterpret_cast<const _DataType1*>(array1_in);
    const _DataType2* array2 = reinterpret_cast<const _DataType2*>(array2_in);
    _ResultType* result = reinterpret_cast<_ResultType*>(result1);

    if (!array1_in || !result1)
    {
        return;
    }

    result[0] = true;

    if (!size)
    {
        return;
    }

    int* close = (int*)dpnp_memory_alloc_c(size * sizeof(int));
    int* sumclose = (int*)dpnp_memory_alloc_c(sizeof(int));

    cl::sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
        size_t i = global_id[0];

        if (std::abs(array1[i] - array2[i]) <= (atol + rtol * std::abs(array2[i]))){
            close[i] = 1;
        }
        else
        {
            close[i] = 0;
        }

    };

    auto kernel_func = [&](cl::sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_allclose_c_kernel<_DataType1, _DataType2, _ResultType>>(gws, kernel_parallel_for_func);
    };

    event = DPNP_QUEUE.submit(kernel_func);

    event.wait();

    dpnp_sum_c<int, int>(sumclose, close, &size, 1, NULL, 0, NULL, NULL);

    dpnp_memory_free_c(close);


    if (sumclose[0] == (int)size){

        result[0] = true;

    }
    else
    {
        result[0] = false;
    }

    dpnp_memory_free_c(sumclose);

}

template <typename _DataType, typename _ResultType>
class dpnp_any_c_kernel;

template <typename _DataType, typename _ResultType>
void dpnp_any_c(const void* array1_in, void* result1, const size_t size)
{
    cl::sycl::event event;

    const _DataType* array_in = reinterpret_cast<const _DataType*>(array1_in);
    _ResultType* result = reinterpret_cast<_ResultType*>(result1);

    if (!array1_in || !result1)
    {
        return;
    }

    result[0] = false;

    if (!size)
    {
        return;
    }

    cl::sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
        size_t i = global_id[0];

        if (array_in[i])
        {
            result[0] = true;
        }
    };

    auto kernel_func = [&](cl::sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_any_c_kernel<_DataType, _ResultType>>(gws, kernel_parallel_for_func);
    };

    event = DPNP_QUEUE.submit(kernel_func);

    event.wait();
}

void func_map_init_logic(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_ALL][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_all_c<bool, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALL][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_all_c<int, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALL][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_all_c<long, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALL][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_all_c<float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_all_c<double, bool>};

    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_INT][eft_INT] = {eft_BLN, (void*)dpnp_allclose_c<int, int, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_LNG][eft_INT] = {eft_BLN, (void*)dpnp_allclose_c<long, int, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_FLT][eft_INT] = {eft_BLN, (void*)dpnp_allclose_c<float, int, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_DBL][eft_INT] = {eft_BLN, (void*)dpnp_allclose_c<double, int, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_INT][eft_LNG] = {eft_BLN, (void*)dpnp_allclose_c<int, long, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_LNG][eft_LNG] = {eft_BLN, (void*)dpnp_allclose_c<long, long, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_FLT][eft_LNG] = {eft_BLN, (void*)dpnp_allclose_c<float, long, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_DBL][eft_LNG] = {eft_BLN, (void*)dpnp_allclose_c<double, long, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_INT][eft_FLT] = {eft_BLN, (void*)dpnp_allclose_c<int, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_LNG][eft_FLT] = {eft_BLN, (void*)dpnp_allclose_c<long, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_FLT][eft_FLT] = {eft_BLN, (void*)dpnp_allclose_c<float, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_DBL][eft_FLT] = {eft_BLN, (void*)dpnp_allclose_c<double, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_INT][eft_DBL] = {eft_BLN, (void*)dpnp_allclose_c<int, double, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_LNG][eft_DBL] = {eft_BLN, (void*)dpnp_allclose_c<long, double, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_FLT][eft_DBL] = {eft_BLN, (void*)dpnp_allclose_c<float, double, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_DBL][eft_DBL] = {eft_BLN, (void*)dpnp_allclose_c<double, double, bool>};

    fmap[DPNPFuncName::DPNP_FN_ANY][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_any_c<bool, bool>};
    fmap[DPNPFuncName::DPNP_FN_ANY][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_any_c<int, bool>};
    fmap[DPNPFuncName::DPNP_FN_ANY][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_any_c<long, bool>};
    fmap[DPNPFuncName::DPNP_FN_ANY][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_any_c<float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ANY][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_any_c<double, bool>};

    return;
}
