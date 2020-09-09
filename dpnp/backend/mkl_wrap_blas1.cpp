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

#include <algorithm>
#include <iostream>
#include <mkl_blas_sycl.hpp>

#include <backend/backend_iface.hpp>
#include "queue_sycl.hpp"

template <typename _DataType>
void mkl_blas_dot_c(void* array1_in, void* array2_in, void* result1, size_t size)
{
    cl::sycl::event status;

    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* array_2 = reinterpret_cast<_DataType*>(array2_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    try
    {
        status = oneapi::mkl::blas::dot(DPNP_QUEUE,
                                        size,
                                        array_1,
                                        1, // array_1 stride
                                        array_2,
                                        1, // array_2 stride
                                        result);
    }
    catch (cl::sycl::exception const& e)
    {
        std::cerr << "Caught synchronous SYCL exception during mkl_blas_dot_c():\n"
                  << e.what() << "\nOpenCL status: " << e.get_cl_code() << std::endl;
    }

    status.wait();
#if 0
    std::cout << "mkl_blas_dot_c res = " << result[0] << std::endl;
#endif
}

template void mkl_blas_dot_c<float>(void* array1_in, void* array2_in, void* result1, size_t size);
template void mkl_blas_dot_c<double>(void* array1_in, void* array2_in, void* result1, size_t size);
