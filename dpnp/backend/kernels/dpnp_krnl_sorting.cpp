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
#include "queue_sycl.hpp"

template <typename _DataType, typename _idx_DataType>
struct _argsort_less
{
    _argsort_less(_DataType* data_ptr)
    {
        _data_ptr = data_ptr;
    }

    inline bool operator()(const _idx_DataType& idx1, const _idx_DataType& idx2)
    {
        return (_data_ptr[idx1] < _data_ptr[idx2]);
    }

private:
    _DataType* _data_ptr = nullptr;
};

template <typename _DataType, typename _idx_DataType>
class dpnp_argsort_c_kernel;

template <typename _DataType, typename _idx_DataType>
void dpnp_argsort_c(void* array1_in, void* result1, size_t size)
{
    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _idx_DataType* result = reinterpret_cast<_idx_DataType*>(result1);

    std::iota(result, result + size, 0);

    auto policy =
        oneapi::dpl::execution::make_device_policy<class dpnp_argsort_c_kernel<_DataType, _idx_DataType>>(DPNP_QUEUE);

    std::sort(policy, result, result + size, _argsort_less<_DataType, _idx_DataType>(array_1));

    policy.queue().wait();
}

// template void dpnp_argsort_c<double, long>(void* array1_in, void* result1, size_t size);
// template void dpnp_argsort_c<float, long>(void* array1_in, void* result1, size_t size);
// template void dpnp_argsort_c<long, long>(void* array1_in, void* result1, size_t size);
// template void dpnp_argsort_c<int, long>(void* array1_in, void* result1, size_t size);
// template void dpnp_argsort_c<double, int>(void* array1_in, void* result1, size_t size);
// template void dpnp_argsort_c<float, int>(void* array1_in, void* result1, size_t size);
// template void dpnp_argsort_c<long, int>(void* array1_in, void* result1, size_t size);
// template void dpnp_argsort_c<int, int>(void* array1_in, void* result1, size_t size);

template <typename _DataType>
struct _sort_less
{
    inline bool operator()(const _DataType& val1, const _DataType& val2)
    {
        return (val1 < val2);
    }
};

template <typename _DataType>
class dpnp_partition_c_kernel;

template <typename _DataType>
void dpnp_partition_c(const void* sort_array1_in, void* result1, const size_t kth, const size_t* shape, const size_t ndim)
{

    cl::sycl::event event;

    const _DataType* sort_arr = reinterpret_cast<const _DataType*>(sort_array1_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    size_t size_ = 1;
    for (size_t i = 0; i < ndim - 1; ++i)
    {
        size_ *= shape[i];
    }

    cl::sycl::range<2> gws(size_, kth+1);
    auto kernel_parallel_for_func = [=](cl::sycl::id<2> global_id) {
        size_t j = global_id[0];
        size_t k = global_id[1];

        _DataType val = sort_arr[j * shape[ndim - 1] + k];

        size_t ind = j * shape[ndim - 1] + k;
        for (size_t i = 0; i < shape[ndim - 1]; ++i)
        {
            if (result[j * shape[ndim - 1] + i] == val)
            {
                ind = j * shape[ndim - 1] + i;
                break;
            }
        }

        _DataType change_val = result[j * shape[ndim - 1] + k];
        result[j * shape[ndim - 1] + k] = val;
        result[ind] = change_val;
    };

    auto kernel_func = [&](cl::sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_partition_c_kernel<_DataType>>(gws, kernel_parallel_for_func);
    };

    event = DPNP_QUEUE.submit(kernel_func);

    event.wait();
}

template <typename _DataType>
class dpnp_sort_c_kernel;

template <typename _DataType>
void dpnp_sort_c(void* array1_in, void* result1, size_t size)
{
    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    std::copy(array_1, array_1 + size, result);

    auto policy = oneapi::dpl::execution::make_device_policy<class dpnp_sort_c_kernel<_DataType>>(DPNP_QUEUE);

    // fails without explicitly specifying of comparator or with std::less during kernels compilation
    // affects other kernels
    std::sort(policy, result, result + size, _sort_less<_DataType>());

    policy.queue().wait();
}

void func_map_init_sorting(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_ARGSORT][eft_INT][eft_INT] = {eft_LNG, (void*)dpnp_argsort_c<int, long>};
    fmap[DPNPFuncName::DPNP_FN_ARGSORT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_argsort_c<long, long>};
    fmap[DPNPFuncName::DPNP_FN_ARGSORT][eft_FLT][eft_FLT] = {eft_LNG, (void*)dpnp_argsort_c<float, long>};
    fmap[DPNPFuncName::DPNP_FN_ARGSORT][eft_DBL][eft_DBL] = {eft_LNG, (void*)dpnp_argsort_c<double, long>};

    fmap[DPNPFuncName::DPNP_FN_PARTITION][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_partition_c<int>};
    fmap[DPNPFuncName::DPNP_FN_PARTITION][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_partition_c<long>};
    fmap[DPNPFuncName::DPNP_FN_PARTITION][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_partition_c<float>};
    fmap[DPNPFuncName::DPNP_FN_PARTITION][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_partition_c<double>};

    fmap[DPNPFuncName::DPNP_FN_SORT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_sort_c<int>};
    fmap[DPNPFuncName::DPNP_FN_SORT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_sort_c<long>};
    fmap[DPNPFuncName::DPNP_FN_SORT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_sort_c<float>};
    fmap[DPNPFuncName::DPNP_FN_SORT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_sort_c<double>};

    return;
}
