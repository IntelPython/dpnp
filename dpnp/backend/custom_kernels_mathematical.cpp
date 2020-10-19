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
#include <vector>

#include <backend_iface.hpp>
#include "backend_utils.hpp"
#include "queue_sycl.hpp"

template <typename _KernelNameSpecialization>
class custom_elemwise_absolute_c_kernel;

template <typename _DataType>
void custom_elemwise_absolute_c(void* array1_in, void* result1, size_t size)
{
    if (!size)
    {
        return;
    }

    cl::sycl::event event;
    _DataType* array1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    if constexpr (std::is_same<_DataType, double>::value || std::is_same<_DataType, float>::value)
    {
        // https://docs.oneapi.com/versions/latest/onemkl/abs.html
        event = oneapi::mkl::vm::abs(DPNP_QUEUE, size, array1, result);
    }
    else
    {
        cl::sycl::range<1> gws(size);
        auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
            const size_t idx = global_id[0];

            if (array1[idx] >= 0)
            {
                result[idx] = array1[idx];
            }
            else
            {
                result[idx] = -1 * array1[idx];
            }
        };

        auto kernel_func = [&](cl::sycl::handler& cgh) {
            cgh.parallel_for<class custom_elemwise_absolute_c_kernel<_DataType>>(gws, kernel_parallel_for_func);
        };

        event = DPNP_QUEUE.submit(kernel_func);
    }

    event.wait();
}

template void custom_elemwise_absolute_c<double>(void* array1_in,
                                                 void* result1,
                                                 size_t size);
template void custom_elemwise_absolute_c<float>(void* array1_in,
                                                void* result1,
                                                size_t size);
template void custom_elemwise_absolute_c<long>(void* array1_in,
                                               void* result1,
                                               size_t size);
template void custom_elemwise_absolute_c<int>(void* array1_in,
                                              void* result1,
                                              size_t size);
