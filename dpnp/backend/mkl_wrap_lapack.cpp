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
#include <mkl_lapack_sycl.hpp>

#include <backend_iface.hpp>
#include "queue_sycl.hpp"

namespace mkl_lapack = oneapi::mkl::lapack;

template <typename _DataType>
void mkl_lapack_syevd_c(void* array_in, void* result1, size_t size)
{
    cl::sycl::event status;

    _DataType* array = reinterpret_cast<_DataType*>(array_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    _DataType* syevd_array = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(size * size * sizeof(_DataType)));
    dpnp_memory_memcpy_c(syevd_array, array, size * size * sizeof(_DataType));

    const std::int64_t lda = std::max<size_t>(1UL, size);

    const std::int64_t scratchpad_size = mkl_lapack::syevd_scratchpad_size<_DataType>(
        DPNP_QUEUE, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, size, lda);

    _DataType* scratchpad = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(scratchpad_size * sizeof(_DataType)));

    status = mkl_lapack::syevd(DPNP_QUEUE,               // queue
                               oneapi::mkl::job::vec,    // jobz
                               oneapi::mkl::uplo::upper, // uplo
                               size,                     // The order of the matrix A (0≤n)
                               syevd_array,              // will be overwritten with eigenvectors
                               lda,
                               result,
                               scratchpad,
                               scratchpad_size);
    status.wait();

    dpnp_memory_free_c(scratchpad);

    custom_elemwise_transpose_c<_DataType>(
        syevd_array, {(long)size, (long)size}, {(long)size, (long)size}, {1, 0}, array, size * size);

    dpnp_memory_free_c(syevd_array);
}

template void mkl_lapack_syevd_c<double>(void* array1_in, void* result1, size_t size);
template void mkl_lapack_syevd_c<float>(void* array1_in, void* result1, size_t size);
