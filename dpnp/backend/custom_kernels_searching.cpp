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

#include <backend_iface.hpp>
#include "backend_pstl.hpp"
#include "queue_sycl.hpp"

template <typename _DataType, typename _idx_DataType>
class custom_argmax_c_kernel;

template <typename _DataType, typename _idx_DataType>
void custom_argmax_c(void* array1_in, void* result1, size_t size)
{
    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _idx_DataType* result = reinterpret_cast<_idx_DataType*>(result1);

    auto policy =
        oneapi::dpl::execution::make_device_policy<class custom_argmax_c_kernel<_DataType, _idx_DataType>>(DPNP_QUEUE);

    _DataType* res = std::max_element(policy, array_1, array_1 + size);
    policy.queue().wait();

    result[0] = std::distance(array_1, res);

#if 0
    std::cout << "result " << result[0] << "\n";
#endif
}

template void custom_argmax_c<double, long>(void* array1_in, void* result1, size_t size);
template void custom_argmax_c<float, long>(void* array1_in, void* result1, size_t size);
template void custom_argmax_c<long, long>(void* array1_in, void* result1, size_t size);
template void custom_argmax_c<int, long>(void* array1_in, void* result1, size_t size);
template void custom_argmax_c<double, int>(void* array1_in, void* result1, size_t size);
template void custom_argmax_c<float, int>(void* array1_in, void* result1, size_t size);
template void custom_argmax_c<long, int>(void* array1_in, void* result1, size_t size);
template void custom_argmax_c<int, int>(void* array1_in, void* result1, size_t size);

template <typename _DataType, typename _idx_DataType>
class custom_argmin_c_kernel;

template <typename _DataType, typename _idx_DataType>
void custom_argmin_c(void* array1_in, void* result1, size_t size)
{
    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _idx_DataType* result = reinterpret_cast<_idx_DataType*>(result1);

    auto policy =
        oneapi::dpl::execution::make_device_policy<class custom_argmin_c_kernel<_DataType, _idx_DataType>>(DPNP_QUEUE);

    _DataType* res = std::min_element(policy, array_1, array_1 + size);
    policy.queue().wait();

    result[0] = std::distance(array_1, res);

#if 0
    std::cout << "result " << result[0] << "\n";
#endif
}

template void custom_argmin_c<double, long>(void* array1_in, void* result1, size_t size);
template void custom_argmin_c<float, long>(void* array1_in, void* result1, size_t size);
template void custom_argmin_c<long, long>(void* array1_in, void* result1, size_t size);
template void custom_argmin_c<int, long>(void* array1_in, void* result1, size_t size);
template void custom_argmin_c<double, int>(void* array1_in, void* result1, size_t size);
template void custom_argmin_c<float, int>(void* array1_in, void* result1, size_t size);
template void custom_argmin_c<long, int>(void* array1_in, void* result1, size_t size);
template void custom_argmin_c<int, int>(void* array1_in, void* result1, size_t size);
