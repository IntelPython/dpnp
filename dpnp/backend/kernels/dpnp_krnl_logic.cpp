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
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"

template <typename _DataType, typename _ResultType>
class dpnp_all_c_kernel;

template <typename _DataType, typename _ResultType>
void dpnp_all_c(const void* array1_in, void* result1, const size_t size)
{
    if (!array1_in || !result1)
    {
        return;
    }

    sycl::event event;
    DPNPC_ptr_adapter<_DataType> input1_ptr(array1_in, size);
    DPNPC_ptr_adapter<_ResultType> result1_ptr(result1, 1, true, true);
    const _DataType* array_in = input1_ptr.get_ptr();
    _ResultType* result = result1_ptr.get_ptr();

    result[0] = true;

    if (!size)
    {
        return;
    }

    sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
        size_t i = global_id[0];

        if (!array_in[i])
        {
            result[0] = false;
        }
    };

    auto kernel_func = [&](sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_all_c_kernel<_DataType, _ResultType>>(gws, kernel_parallel_for_func);
    };

    event = DPNP_QUEUE.submit(kernel_func);

    event.wait();
}

template <typename _DataType1, typename _DataType2, typename _ResultType>
class dpnp_allclose_c_kernel;

template <typename _DataType1, typename _DataType2, typename _ResultType>
void dpnp_allclose_c(
    const void* array1_in, const void* array2_in, void* result1, const size_t size, double rtol_val, double atol_val)
{
    if (!array1_in || !result1)
    {
        return;
    }

    sycl::event event;
    DPNPC_ptr_adapter<_DataType1> input1_ptr(array1_in, size);
    DPNPC_ptr_adapter<_DataType2> input2_ptr(array2_in, size);
    DPNPC_ptr_adapter<_ResultType> result1_ptr(result1, 1, true, true);
    const _DataType1* array1 = input1_ptr.get_ptr();
    const _DataType2* array2 = input2_ptr.get_ptr();
    _ResultType* result = result1_ptr.get_ptr();

    result[0] = true;

    if (!size)
    {
        return;
    }

    sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
        size_t i = global_id[0];

        if (std::abs(array1[i] - array2[i]) > (atol_val + rtol_val * std::abs(array2[i])))
        {
            result[0] = false;
        }
    };

    auto kernel_func = [&](sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_allclose_c_kernel<_DataType1, _DataType2, _ResultType>>(gws,
                                                                                            kernel_parallel_for_func);
    };

    event = DPNP_QUEUE.submit(kernel_func);

    event.wait();
}

template <typename _DataType, typename _ResultType>
class dpnp_any_c_kernel;

template <typename _DataType, typename _ResultType>
void dpnp_any_c(const void* array1_in, void* result1, const size_t size)
{
    if (!array1_in || !result1)
    {
        return;
    }

    sycl::event event;
    DPNPC_ptr_adapter<_DataType> input1_ptr(array1_in, size);
    DPNPC_ptr_adapter<_ResultType> result1_ptr(result1, 1, true, true);
    const _DataType* array_in = input1_ptr.get_ptr();
    _ResultType* result = result1_ptr.get_ptr();

    result[0] = false;

    if (!size)
    {
        return;
    }

    sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
        size_t i = global_id[0];

        if (array_in[i])
        {
            result[0] = true;
        }
    };

    auto kernel_func = [&](sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_any_c_kernel<_DataType, _ResultType>>(gws, kernel_parallel_for_func);
    };

    event = DPNP_QUEUE.submit(kernel_func);

    event.wait();
}

void func_map_init_logic(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_ALL][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_all_c<bool, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALL][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_all_c<int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALL][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_all_c<int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALL][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_all_c<float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_all_c<double, bool>};

    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_INT][eft_INT] = {eft_BLN, (void*)dpnp_allclose_c<int32_t, int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_LNG][eft_INT] = {eft_BLN, (void*)dpnp_allclose_c<int64_t, int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_FLT][eft_INT] = {eft_BLN, (void*)dpnp_allclose_c<float, int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_DBL][eft_INT] = {eft_BLN, (void*)dpnp_allclose_c<double, int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_INT][eft_LNG] = {eft_BLN, (void*)dpnp_allclose_c<int32_t, int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_LNG][eft_LNG] = {eft_BLN, (void*)dpnp_allclose_c<int64_t, int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_FLT][eft_LNG] = {eft_BLN, (void*)dpnp_allclose_c<float, int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_DBL][eft_LNG] = {eft_BLN, (void*)dpnp_allclose_c<double, int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_INT][eft_FLT] = {eft_BLN, (void*)dpnp_allclose_c<int32_t, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_LNG][eft_FLT] = {eft_BLN, (void*)dpnp_allclose_c<int64_t, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_FLT][eft_FLT] = {eft_BLN, (void*)dpnp_allclose_c<float, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_DBL][eft_FLT] = {eft_BLN, (void*)dpnp_allclose_c<double, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_INT][eft_DBL] = {eft_BLN, (void*)dpnp_allclose_c<int32_t, double, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_LNG][eft_DBL] = {eft_BLN, (void*)dpnp_allclose_c<int64_t, double, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_FLT][eft_DBL] = {eft_BLN, (void*)dpnp_allclose_c<float, double, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_DBL][eft_DBL] = {eft_BLN, (void*)dpnp_allclose_c<double, double, bool>};

    fmap[DPNPFuncName::DPNP_FN_ANY][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_any_c<bool, bool>};
    fmap[DPNPFuncName::DPNP_FN_ANY][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_any_c<int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ANY][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_any_c<int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ANY][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_any_c<float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ANY][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_any_c<double, bool>};

    return;
}
