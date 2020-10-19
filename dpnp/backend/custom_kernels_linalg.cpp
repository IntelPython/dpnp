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
#include <list>
#include <mkl_blas_sycl.hpp>

#include <backend_iface.hpp>
#include "backend_pstl.hpp"
#include "backend_utils.hpp"
#include "queue_sycl.hpp"

namespace mkl_blas = oneapi::mkl::blas::row_major;

template <typename _DataType>
class custom_matrix_rank_c_kernel;

template <typename _DataType>
void custom_matrix_rank_c(void* array1_in, void* result1, size_t* shape, size_t ndim)
{
    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    size_t elems = 1;
    long rank_val = 0;
    if (ndim > 1)
    {
        for (size_t i = 0; i < ndim; i++)
        {
            if (i == 0)
            {
                elems = shape[i];
            }
            else
            {
                if (shape[i] < elems)
                {
                    elems = shape[i];
                }
            }
        }
    }
    for (size_t i = 0; i < elems; i++)
    {
        size_t ind = 0;
        for (size_t j = 0; j < ndim; j++)
        {
            ind += shape[j] * i;
        }
        rank_val += array_1[ind];
    }
    result[0] = rank_val;

#if 0
    std::cout << "matrix_rank result " << result[0] << "\n";
#endif
}


template void
    custom_matrix_rank_c<double>(void* array1_in, void* result1, size_t* shape, size_t ndim);
template void
    custom_matrix_rank_c<float>(void* array1_in, void* result1, size_t* shape, size_t ndim);
template void
    custom_matrix_rank_c<long>(void* array1_in, void* result1, size_t* shape, size_t ndim);
template void custom_matrix_rank_c<int>(void* array1_in, void* result1, size_t* shape, size_t ndim);
